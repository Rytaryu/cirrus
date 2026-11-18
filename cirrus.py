"""
cirrus: a two-provider forecasting bot for the Metaculus FutureEval tournaments.

This file is the entry point the GitHub Actions workflows run. It keeps the
Metaculus template (main.py) untouched and adds one idea on top of it:

    Each question is researched and forecast independently by two "lanes",
    one per AI provider (Claude and GPT). Every lane does its own web research
    with its provider's search tool and makes its own forecasts. The template's
    normal aggregation (median for binary questions) then combines the
    forecasts of both lanes.

If one lane fails (for example its API credits ran out), the other lane's
forecasts are still submitted and a notice is sent to Discord.

Everything you are likely to tune is in the CONFIGURATION section below.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Literal

import dotenv
import requests

from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (  # noqa: E402
    ForecastReport,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MonetaryCostManager,
    ReasonedPrediction,
    clean_indents,
)
from forecasting_tools.data_models.forecast_report import (  # noqa: E402
    ResearchWithPredictions,
)

from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (  # noqa: E402
    LitellmCostTracker,
)

from main import FallTemplateBot2026  # noqa: E402

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# forecasting-tools records the cost of a model call in two places: directly
# after the call returns, and again from a litellm callback. With the litellm
# version in uv.lock the callback runs after the direct path has already added
# the cost, so every call was counted twice (seen on 2026-11-13: reported costs
# were exactly double the token prices). Marking the tracker as initialized
# keeps the callback from being installed, which leaves the direct path only.
# tests/test_cirrus.py checks the reported cost against known token counts.
LitellmCostTracker._initialized = True

# litellm downloads its price list when it is imported and uses the copy bundled
# with the installed version only if that download fails. The bundled copy in
# litellm 1.105.1 (uv.lock) has no price for two of the bot's models, so after a
# failed download their calls would count as free: the per-question spending
# stop and the Discord cost summary would not see them. These entries fill that
# gap only; a price litellm already has always wins. Each entry names the
# provider, a related model whose features it copies, and the provider's list
# prices in USD per token.
FALLBACK_PRICES: dict[str, tuple[str, str, dict[str, float]]] = {
    # GPT-6.2 Astra, released 2026-11-17: $8 / $40, cached input $0.40.
    "gpt-6.2-astra": (
        "openai",
        "gpt-6-astra",
        {
            "input_cost_per_token": 8e-6,
            "output_cost_per_token": 4e-5,
            "cache_read_input_token_cost": 4e-7,
        },
    ),
    # Claude Haiku 5.5: $0.10 / $0.50, cache reads $0.01, cache writes $0.125.
    "claude-haiku-5-5": (
        "anthropic",
        "claude-sonnet-5-5",
        {
            "input_cost_per_token": 1e-7,
            "output_cost_per_token": 5e-7,
            "cache_read_input_token_cost": 1e-8,
            "cache_creation_input_token_cost": 1.25e-7,
        },
    ),
}


def add_missing_prices() -> list[str]:
    """Register FALLBACK_PRICES for models litellm has no price for. Returns their names."""
    import litellm

    added = []
    for model, (provider, template, prices) in FALLBACK_PRICES.items():
        if model in litellm.model_cost:
            continue
        # Keep the related model's features, drop all of its prices (including
        # tiered ones such as long-context or batch rates) and set ours.
        entry = {
            key: value
            for key, value in litellm.model_cost.get(template, {}).items()
            if "cost" not in key
        }
        entry.update(prices)
        entry["litellm_provider"] = provider
        entry.setdefault("mode", "chat")
        litellm.register_model({model: entry})
        added.append(model)
    if added:
        logger.warning(f"litellm had no price for {added}; using cirrus's fallback prices")
    return added


add_missing_prices()


class ResponsesLlm(GeneralLlm):
    """
    GeneralLlm for OpenAI's Responses API that keeps the whole reply.

    GeneralLlm reads only the first message item of a Responses API reply. A
    model that searches the web may write a short note before searching and the
    real answer afterwards, so this joins the text of every message item.
    """

    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(model=model, responses_api=True, **kwargs)

    def _normalize_response(self, raw_response, model_response):  # type: ignore[override]
        normalized = super()._normalize_response(raw_response, model_response)
        texts = [
            choice.message.content
            for choice in normalized.choices
            if isinstance(choice.message.content, str) and choice.message.content.strip()
        ]
        if len(texts) > 1:
            normalized.choices[0].message.content = "\n\n".join(texts)
        return normalized


# =============================================================================
# CONFIGURATION
# =============================================================================

# How a question is handled: every lane's researcher searches the web on its
# own, then every lane's forecaster reads all of the research reports together
# and forecasts. Each forecast therefore draws on two independent sources, and
# the forecasters differ only in the model that weighs the evidence.
#
# BUDGET. Each provider's API credits come with a subscription, about $100 a
# month. Production may use $80 of it; the rest is kept for test runs, later
# experiments and months with more questions than planned. Planning for 300
# questions a month (the seasonal tournament and MiniBench together) leaves
# $0.27 per provider per question: about $0.05 for its forecast and the rest
# for its research.

# Forecasts each lane's forecaster makes per question. With two lanes and 1,
# each question gets 2 forecasts, one per provider, so neither outweighs the
# other in the aggregate (the median, which for two values is their mean).
PREDICTIONS_PER_LANE = 1

# Safety stop, in USD, per provider per question. If a lane's research costs
# more than this, its forecaster sits the question out and the other lane's
# forecast is submitted alone. The research budget is about $0.22, so this
# only catches runaway cases.
LANE_COST_CAP_USD = 0.50

# How many research calls (one per lane per question) may run at the same time.
MAX_CONCURRENT_RESEARCH = 4

# Reasoning effort of the GPT forecaster: low, medium, high, xhigh or max.
GPT_FORECAST_EFFORT = "medium"

# Reasoning effort of the GPT researcher. More effort lets it plan its searches
# and look again for what is still missing. Reasoning tokens are billed as output.
GPT_RESEARCH_EFFORT = "high"

# How much page content each GPT web search returns: low, medium or high.
GPT_SEARCH_CONTEXT = "high"

# Web searches the Claude researcher may run per question. Search results are
# billed as input tokens and are re-read on every later step, so this is the
# main driver of the Claude lane's cost.
CLAUDE_MAX_SEARCHES = 5


@dataclass(frozen=True)
class Lane:
    """One provider's models. Each value builds a fresh GeneralLlm."""

    name: str
    api_key_env: str
    researcher: Callable[[], GeneralLlm]
    forecaster: Callable[[], GeneralLlm]
    helper: Callable[[], GeneralLlm]  # cheap model: parses and summarizes

    def is_available(self) -> bool:
        value = os.getenv(self.api_key_env, "").strip()
        return bool(value) and value != "REPLACE_ME"


# Model names are passed to litellm, hence the provider prefix. Prices are in
# USD per 1M tokens (input / output) as listed by each provider in Nov 2026.
LANES: dict[str, Lane] = {
    "claude": Lane(
        name="claude",
        api_key_env="ANTHROPIC_API_KEY",
        # Sonnet 5.5 ($2 / $10) with Anthropic's web search tool
        # ($10 per 1,000 searches; results are billed as input tokens).
        researcher=lambda: GeneralLlm(
            model="anthropic/claude-sonnet-5-5",
            tools=[
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": CLAUDE_MAX_SEARCHES,
                }
            ],
            max_tokens=6000,
            timeout=300,
            allowed_tries=2,
        ),
        # Opus 5.5 ($4 / $20).
        forecaster=lambda: GeneralLlm(
            model="anthropic/claude-opus-5-5",
            max_tokens=8000,
            timeout=240,
            allowed_tries=2,
        ),
        # Haiku 5.5 ($0.10 / $0.50).
        helper=lambda: GeneralLlm(
            model="anthropic/claude-haiku-5-5",
            max_tokens=4000,
            timeout=90,
            allowed_tries=2,
        ),
    ),
    "gpt": Lane(
        name="gpt",
        api_key_env="OPENAI_API_KEY",
        # GPT-6.1 Sol ($2 / $10) with OpenAI's web search tool ($10 per 1,000
        # calls; results are billed as input tokens). Web search needs the
        # Responses API, so every GPT call uses it.
        # Reasoning tokens count against max_output_tokens, so it leaves room
        # for them on top of the report.
        researcher=lambda: ResponsesLlm(
            model="openai/gpt-6.1-sol",
            tools=[{"type": "web_search", "search_context_size": GPT_SEARCH_CONTEXT}],
            reasoning={"effort": GPT_RESEARCH_EFFORT},
            max_output_tokens=12000,
            timeout=300,
            allowed_tries=2,
        ),
        # GPT-6.2 Astra ($8 / $40), OpenAI's most capable model. The first test
        # runs used GPT-6.1 Sol here and the lane cost about $0.14 per question,
        # of which the two forecasts were only $0.02; the judgment step is where
        # the lane's spare budget goes. Reasoning tokens are billed as output,
        # so the effort setting is the main cost control. Until 2026-11-18 this
        # was GPT-6 Astra ($10 / $50); GPT-6.2 Astra is 20% cheaper per token
        # and scores higher on OpenAI's and independent general benchmarks
        # (none of them measures forecasting). Each lane's record names the
        # model it used, so forecasts from before and after can be told apart.
        forecaster=lambda: ResponsesLlm(
            model="openai/gpt-6.2-astra",
            reasoning={"effort": GPT_FORECAST_EFFORT},
            max_output_tokens=12000,
            timeout=300,
            allowed_tries=2,
        ),
        # GPT-6 Luna ($0.10 / $0.50).
        helper=lambda: GeneralLlm(
            model="openai/gpt-6-luna",
            responses_api=True,
            max_output_tokens=4000,
            timeout=90,
            allowed_tries=2,
        ),
    ),
}

DEFAULT_LANE_ORDER = ("claude", "gpt")


# =============================================================================
# LANE SELECTION
# =============================================================================


def requested_lane_names() -> list[str]:
    """Lanes asked for through the CIRRUS_LANES env var (default: all)."""
    raw = os.getenv("CIRRUS_LANES", "").strip().lower()
    if not raw or raw == "both":
        return list(DEFAULT_LANE_ORDER)
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in names if name not in LANES]
    if unknown:
        raise ValueError(
            f"Unknown lane(s) in CIRRUS_LANES: {unknown}. Known lanes: {list(LANES)}"
        )
    return names


def active_lanes() -> list[Lane]:
    """Requested lanes whose API key is set."""
    return [LANES[name] for name in requested_lane_names() if LANES[name].is_available()]


def build_llms(lanes: list[Lane]) -> dict[str, str | GeneralLlm | None]:
    """
    The llm table for the bot. Lane models live under "<lane>/<purpose>". The
    plain purposes the template asks for ("default", "researcher", ...) point
    at the first lane and are only a fallback: inside a lane, get_llm() swaps
    in that lane's model.
    """
    if not lanes:
        raise RuntimeError(
            "No lane is available. Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY."
        )
    llms: dict[str, str | GeneralLlm | None] = {}
    first = lanes[0]
    llms["default"] = first.forecaster()
    llms["researcher"] = first.researcher()
    llms["parser"] = first.helper()
    llms["summarizer"] = first.helper()
    for lane in lanes:
        llms[f"{lane.name}/default"] = lane.forecaster()
        llms[f"{lane.name}/researcher"] = lane.researcher()
        llms[f"{lane.name}/parser"] = lane.helper()
        llms[f"{lane.name}/summarizer"] = lane.helper()
    return llms


# The lane the current asyncio task is working for. asyncio copies context
# variables into every task it creates, so everything a lane starts (research,
# forecasts, parsing) sees the same value without it being passed around.
_current_lane: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cirrus_current_lane", default=None
)
# The question the current asyncio task is working on (its page URL), so that
# costs can be added up per question and lane.
_current_question: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cirrus_current_question", default=None
)


class LaneCostCapReached(RuntimeError):
    """A lane spent more than LANE_COST_CAP_USD on one question's research."""


# =============================================================================
# NOTIFICATIONS
# =============================================================================

_SECRET_LOOKING = re.compile(r"\b(sk|key|tok)[-_][A-Za-z0-9_\-]{8,}\b", re.IGNORECASE)


def scrub(text: str, limit: int = 300) -> str:
    """Shorten an error message and blank anything that looks like a key."""
    cleaned = _SECRET_LOOKING.sub("[redacted]", text).replace("\n", " ")
    return cleaned if len(cleaned) <= limit else cleaned[:limit] + "..."


def notify_discord(message: str) -> None:
    """Post to the Discord webhook if one is configured. Never raises."""
    url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not url:
        return
    try:
        response = requests.post(url, json={"content": message[:1900]}, timeout=10)
        if response.status_code >= 300:
            logger.warning(f"Discord notification got HTTP {response.status_code}")
    except Exception as error:  # noqa: BLE001
        # Only the error type is logged: the full message can contain the URL.
        logger.warning(f"Discord notification failed: {type(error).__name__}")


# =============================================================================
# RECORDS
# =============================================================================
#
# Every comment the bot posts ends with a "RECORD" section: one line of JSON
# with the bot version, each lane's individual forecasts and costs, and the
# final forecast. Metaculus keeps the comments, so the full history of what
# each lane predicted can be rebuilt from Metaculus alone once questions
# resolve (for example to compare "Claude only", "GPT only" and the aggregate).
#
# Version 1 (until 2026-11-18): each lane forecast twice from its own research.
# Version 2 ("design": "pooled"): each lane's forecaster forecasts once from
# the research of all lanes, so a lane's predictions show its forecaster alone.

RECORD_VERSION = 2
RECORD_DESIGN = "pooled"
_FORECASTER_MARKER = re.compile(r"^\*Forecaster: (\w+)")
_MAX_RECORDED_PERCENTILES = 11


def compact_prediction(prediction: object) -> object:
    """A small JSON-friendly form of any prediction type the bot produces."""
    if isinstance(prediction, bool):
        return prediction
    if isinstance(prediction, (int, float)):
        return round(float(prediction), 4)
    options = getattr(prediction, "predicted_options", None)
    if options is not None:
        return {option.option_name: round(option.probability, 4) for option in options}
    percentiles = getattr(prediction, "declared_percentiles", None)
    if percentiles is not None:
        points = [[round(p.percentile, 4), p.value] for p in percentiles]
        if len(points) > _MAX_RECORDED_PERCENTILES:
            step = math.ceil(len(points) / (_MAX_RECORDED_PERCENTILES - 1))
            points = points[::step] + ([points[-1]] if (len(points) - 1) % step else [])
        return points
    parts = ("parent", "child", "prediction_yes", "prediction_no")
    if all(hasattr(prediction, part) for part in parts):
        return {part: compact_prediction(getattr(prediction, part)) for part in parts}
    if type(prediction).__name__ == "PredictionAffirmed":
        return "affirmed"
    return str(prediction)[:200]


def bot_version() -> str:
    """The commit the bot is running from (set by GitHub Actions)."""
    return (os.getenv("GITHUB_SHA") or "local")[:7]


def read_record(comment: str) -> dict | None:
    """The record stored in a comment written by this bot, if there is one."""
    for line in reversed(comment.splitlines()):
        if line.startswith('{"cirrus_record":'):
            return json.loads(line)
    return None


def github_run_url() -> str | None:
    """Link to the current GitHub Actions run, when running in one."""
    server = os.getenv("GITHUB_SERVER_URL")
    repository = os.getenv("GITHUB_REPOSITORY")
    run_id = os.getenv("GITHUB_RUN_ID")
    if server and repository and run_id:
        return f"{server}/{repository}/actions/runs/{run_id}"
    return None


# Metaculus answers a forecast on a closed question with HTTP 405 and
# {"error": "Question ... is already closed to forecasting !"}.
_CLOSED_TO_FORECASTING = "closed to forecasting"
_QUESTION_URL_IN_ERROR = re.compile(r"question url: '([^']+)'")


def closed_before_submission(error: BaseException) -> bool:
    """
    True when Metaculus refused the forecast because the question closed while
    the bot was still working on it. The question was open when the run fetched
    it, so this is a deadline the run missed, not a bug in the bot.

    The error itself or the errors it was raised from (__cause__) must say so.
    An exception group counts only if every error in it does, so that a real
    failure is never hidden behind a closed question.
    """
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _CLOSED_TO_FORECASTING in str(current).lower():
            return True
        if isinstance(current, BaseExceptionGroup) and current.exceptions:
            return all(closed_before_submission(inner) for inner in current.exceptions)
        current = current.__cause__
    return False


def split_closed(reports: list) -> tuple[list, list]:
    """
    Split a run's reports into (closed, rest): `closed` holds the errors for
    questions that closed before the forecast could be submitted, `rest`
    holds everything else in the original order.
    """
    closed = [r for r in reports if isinstance(r, BaseException) and closed_before_submission(r)]
    rest = [r for r in reports if not any(r is c for c in closed)]
    return closed, rest


def question_url_of(error: BaseException) -> str | None:
    """The question link the template puts in its per-question error message."""
    match = _QUESTION_URL_IN_ERROR.search(str(error))
    return match.group(1) if match else None


# =============================================================================
# THE BOT
# =============================================================================


@dataclass
class LaneStats:
    costs: list[float] = field(default_factory=list)  # one entry per question
    failures: list[str] = field(default_factory=list)  # one entry per failed step
    failed_questions: set[str] = field(default_factory=set)
    # Where the money went, summed over the run.
    research_cost: float = 0.0
    summary_cost: float = 0.0
    forecast_cost: float = 0.0  # forecasts and the parsing of their output


class Cirrus(FallTemplateBot2026):
    """
    The template bot with one research report built from every provider lane
    and one forecaster per lane.

    For each question, every lane's researcher searches the web at the same
    time; their reports are joined into one research text, and every lane's
    forecaster forecasts from all of it. Forecast prompts and output parsing
    are inherited from the template. This class adds the lanes, a research
    prompt written for web search, a per-lane spending stop and per-lane cost
    reporting.
    """

    def __init__(self, *, lanes: list[Lane] | None = None, **kwargs) -> None:
        self.lanes: list[Lane] = lanes if lanes is not None else active_lanes()
        if not self.lanes:
            raise RuntimeError(
                "No lane is available. Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY."
            )
        kwargs.setdefault("llms", build_llms(self.lanes))
        # One research text (made by all lanes together), forecast on by every
        # lane's forecaster.
        kwargs.setdefault("research_reports_per_question", 1)
        kwargs.setdefault(
            "predictions_per_research_report", len(self.lanes) * PREDICTIONS_PER_LANE
        )
        # One lane failing leaves half of the forecasts. Allow the question to
        # go ahead as long as at least one forecast survives.
        kwargs.setdefault("required_successful_predictions", 0.0)
        super().__init__(**kwargs)
        self.lane_stats: dict[str, LaneStats] = {
            lane.name: LaneStats() for lane in self.lanes
        }
        self._lane_turn: dict[int, int] = {}
        # question URL -> lane -> {"cost": ..., "error": ...}, for the record.
        self._lane_outcomes: dict[str, dict[str, dict]] = {}
        # question URL -> lane -> cost kind -> USD spent on that question.
        self._question_spend: dict[str, dict[str, dict[str, float]]] = {}
        self._research_limiter = asyncio.Semaphore(MAX_CONCURRENT_RESEARCH)

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm | None]:
        # Tells the base class which llm purposes exist, so it does not fall
        # back to its own (outdated) default models or warn about ours.
        lanes = active_lanes() or [LANES[name] for name in requested_lane_names()]
        return build_llms(lanes)

    def make_llm_dict(self) -> dict:
        # Only the lane models: the plain purposes are duplicates of lane 0.
        return {k: v for k, v in super().make_llm_dict().items() if "/" in k}

    # ------------------------------------------------------------------ lanes

    def get_llm(  # type: ignore[override]
        self,
        purpose: str = "default",
        guarantee_type: Literal["llm", "string_name"] | None = None,
    ) -> GeneralLlm | str:
        lane = _current_lane.get()
        if lane is not None and f"{lane}/{purpose}" in self._llms:
            purpose = f"{lane}/{purpose}"
        return super().get_llm(purpose, guarantee_type)  # type: ignore[arg-type]

    def _take_lane(self, question: MetaculusQuestion) -> Lane:
        """Hand out lanes in turn: first call for a question gets lane 0, ..."""
        key = id(question)
        turn = self._lane_turn.get(key, 0)
        self._lane_turn[key] = turn + 1
        return self.lanes[turn % len(self.lanes)]

    async def _research_and_make_predictions(
        self, question: MetaculusQuestion
    ) -> ResearchWithPredictions:
        """Research and forecasts for one question, with costs kept per lane."""
        url = question.page_url
        self._question_spend[url] = {
            lane.name: {"research_cost": 0.0, "forecast_cost": 0.0, "summary_cost": 0.0}
            for lane in self.lanes
        }
        outcomes = self._lane_outcomes.setdefault(url, {})
        for lane in self.lanes:
            outcomes.setdefault(lane.name, {})
        token = _current_question.set(url)
        try:
            return await super()._research_and_make_predictions(question)
        finally:
            _current_question.reset(token)
            for lane in self.lanes:
                spend = self._question_spend[url][lane.name]
                total = sum(spend.values())
                self.lane_stats[lane.name].costs.append(total)
                outcomes[lane.name]["cost"] = round(total, 4)
                outcomes[lane.name]["research_cost"] = round(spend["research_cost"], 4)
                logger.info(f"[{lane.name}] spent about ${total:.4f} on {url}")

    def _lane_failed(
        self, question: MetaculusQuestion, lane: Lane, stage: str, error: BaseException
    ) -> None:
        """Note a failed research or forecast step for the run summary and the record."""
        url = question.page_url
        stats = self.lane_stats[lane.name]
        stats.failures.append(f"{url} ({stage}): {type(error).__name__}: {scrub(str(error))}")
        stats.failed_questions.add(url)
        outcome = self._lane_outcomes.setdefault(url, {}).setdefault(lane.name, {})
        outcome.setdefault("error", f"{stage}: {type(error).__name__}")
        logger.error(f"[{lane.name}] {stage} failed on {url}: {error}")

    async def _make_prediction(
        self, question: MetaculusQuestion, research: str
    ) -> ReasonedPrediction:
        """One forecast by the next lane's forecaster, from the joint research."""
        lane = self._take_lane(question)
        token = _current_lane.set(lane.name)
        try:
            spent = (
                self._question_spend.get(question.page_url, {})
                .get(lane.name, {})
                .get("research_cost", 0.0)
            )
            if spent >= LANE_COST_CAP_USD:
                raise LaneCostCapReached(
                    f"Research cost about ${spent:.2f}, which is over the "
                    f"${LANE_COST_CAP_USD:.2f} cap per lane per question."
                )
            with MonetaryCostManager(0) as cost_manager:
                try:
                    prediction = await super()._make_prediction(question, research)
                finally:
                    self._add_cost("forecast_cost", cost_manager.current_usage)
            model = self.get_llm("default", "llm").model
            return prediction.model_copy(
                update={
                    "reasoning": f"*Forecaster: {lane.name} ({model})*\n\n{prediction.reasoning}"
                }
            )
        except Exception as error:
            self._lane_failed(question, lane, "forecast", error)
            raise
        finally:
            _current_lane.reset(token)

    async def summarize_research(self, question: MetaculusQuestion, research: str) -> str:
        # The summary only goes into the comment. It is written by the helper
        # of the first lane whose research worked, and counted as that lane's.
        outcomes = self._lane_outcomes.get(question.page_url, {})
        lane = next(
            (l for l in self.lanes if "error" not in outcomes.get(l.name, {})),
            self.lanes[0],
        )
        token = _current_lane.set(lane.name)
        try:
            with MonetaryCostManager(0) as cost_manager:
                try:
                    return await super().summarize_research(question, research)
                finally:
                    self._add_cost("summary_cost", cost_manager.current_usage)
        finally:
            _current_lane.reset(token)

    def _add_cost(self, kind: str, amount: float) -> None:
        lane = _current_lane.get()
        if lane not in self.lane_stats:
            return
        stats = self.lane_stats[lane]
        setattr(stats, kind, getattr(stats, kind) + amount)
        url = _current_question.get()
        if url in self._question_spend:
            self._question_spend[url][lane][kind] += amount

    # --------------------------------------------------------------- research

    async def run_research(self, question: MetaculusQuestion) -> str:
        """Every lane researches at the same time; the reports are joined."""
        results = await asyncio.gather(
            *(self._lane_research(lane, question) for lane in self.lanes),
            return_exceptions=True,
        )
        reports = []
        errors = []
        for lane, result in zip(self.lanes, results):
            if isinstance(result, BaseException):
                self._lane_failed(question, lane, "research", result)
                errors.append(result)
                continue
            model = self.get_llm(f"{lane.name}/researcher", "llm").model
            reports.append(f"# Research report from the {lane.name} lane ({model})\n\n{result}")
        if not reports:
            raise ExceptionGroup("Every lane's research failed", errors)  # type: ignore[arg-type]
        if len(reports) == 1:
            intro = "One research report follows; the other researcher failed on this question."
        else:
            intro = (
                f"{len(reports)} research assistants searched the web independently; "
                "their reports follow. They may repeat or contradict each other. Where "
                "they disagree, prefer the more recent and better sourced fact."
            )
        return intro + "\n\n" + "\n\n".join(reports)

    async def _lane_research(self, lane: Lane, question: MetaculusQuestion) -> str:
        token = _current_lane.set(lane.name)
        try:
            async with self._research_limiter:
                researcher = self.get_llm("researcher", "llm")
                with MonetaryCostManager(0) as cost_manager:
                    try:
                        research = await researcher.invoke(self._research_prompt(question))
                    finally:
                        self._add_cost("research_cost", cost_manager.current_usage)
            logger.info(f"[{lane.name}] Found research for URL {question.page_url}:\n{research}")
            return research
        finally:
            _current_lane.reset(token)

    @staticmethod
    def _research_prompt(question: MetaculusQuestion) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return clean_indents(
            f"""
            You are a research assistant to a professional forecaster. Today is {today} (UTC).

            Search the web for the most recent information that bears on the question below.
            Your own knowledge may be months out of date, so rely on what you find and say
            when each fact is from. Use a handful of well-chosen searches rather than many.

            {question.give_question_details_as_markdown()}

            Write a concise report for the forecaster with these parts:
            1. Current status: the latest facts that bear on how the question resolves, each with its date and source URL.
            2. Resolution source: if the criteria name a data source or a specific event, what that source shows right now.
            3. Base rates: how often similar events happened in the past, or how the relevant number has moved over a comparable period.
            4. Schedule: known dates before the question closes that could change the outcome.
            5. Other forecasts: prediction markets, expert estimates or polls on this exact question, if any.
            6. Gaps: what you looked for and could not find.

            Give the source URL next to every fact. Do not state a probability or a forecast of your own.
            """
        )

    # ---------------------------------------------------------------- records

    def build_record(
        self,
        question: MetaculusQuestion,
        research_prediction_collections: list[ResearchWithPredictions],
        aggregated_prediction: object,
        final_cost: float,
    ) -> dict:
        lanes: dict[str, dict] = {}
        outcomes = self._lane_outcomes.get(question.page_url, {})
        for lane in self.lanes:
            name = lane.name
            lanes[name] = {
                "research_model": self.get_llm(f"{name}/researcher", "llm").model,
                "forecast_model": self.get_llm(f"{name}/default", "llm").model,
                "predictions": [],
                **outcomes.get(name, {}),
            }
        for collection in research_prediction_collections:
            for prediction in collection.predictions:
                marker = _FORECASTER_MARKER.match(prediction.reasoning or "")
                if marker and marker.group(1) in lanes:
                    lanes[marker.group(1)]["predictions"].append(
                        compact_prediction(prediction.prediction_value)
                    )
        return {
            "cirrus_record": RECORD_VERSION,
            "design": RECORD_DESIGN,
            "commit": bot_version(),
            "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "question": {
                "post": question.id_of_post,
                "id": question.id_of_question,
                "type": type(question).__name__,
                "url": question.page_url,
            },
            "lanes": lanes,
            "final": compact_prediction(aggregated_prediction),
            "cost": round(final_cost, 4),
        }

    def _create_comment(
        self,
        question: MetaculusQuestion,
        research_prediction_collections: list[ResearchWithPredictions],
        aggregated_prediction: object,
        final_cost: float,
        time_spent_in_minutes: float,
    ) -> str:
        comment = super()._create_comment(
            question,
            research_prediction_collections,
            aggregated_prediction,  # type: ignore[arg-type]
            final_cost,
            time_spent_in_minutes,
        )
        try:
            record = self.build_record(
                question, research_prediction_collections, aggregated_prediction, final_cost
            )
            line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        except Exception as error:  # noqa: BLE001
            # The record is bookkeeping: never let it stop a forecast.
            logger.warning(f"Could not build the record for {question.page_url}: {error}")
            return comment
        return f"{comment.rstrip()}\n\n# RECORD\n```\n{line}\n```\n"

    # -------------------------------------------------------------- reporting

    def lane_report(self) -> str:
        """Per-lane cost and failure summary for the run log."""
        lines = ["Lane summary (costs are litellm estimates; the provider consoles are authoritative):"]
        for name, stats in self.lane_stats.items():
            attempts = len(stats.costs)
            if attempts == 0:
                lines.append(f"  {name}: no questions this run")
                continue
            total = sum(stats.costs)
            lines.append(
                f"  {name}: {attempts} question(s), about ${total:.4f} in total, "
                f"${total / attempts:.4f} per question, "
                f"failed on {len(stats.failed_questions)} question(s)"
            )
            lines.append(
                f"    per question: research ${stats.research_cost / attempts:.4f}, "
                f"forecasts ${stats.forecast_cost / attempts:.4f}, "
                f"summary ${stats.summary_cost / attempts:.4f}"
            )
        return "\n".join(lines)

    def run_message(
        self, run_mode: str, reports: list, published: bool, closed: list | None = None
    ) -> str | None:
        """
        The Discord message for one run: what was submitted, what it cost, what
        failed and which questions closed before their forecast could be
        submitted (`closed`, see split_closed). None when the run had nothing
        to forecast.
        """
        closed = closed or []
        done = [r for r in reports if isinstance(r, ForecastReport)]
        failed = [r for r in reports if isinstance(r, BaseException)]
        lane_failures = {n: s.failures for n, s in self.lane_stats.items() if s.failures}
        if not done and not failed and not closed and not lane_failures:
            return None

        verb = "submitted" if published else "produced without submitting"
        lines = [f"cirrus ({run_mode}): {verb} {len(done)} forecast(s), {len(failed)} question(s) failed."]
        if closed:
            urls = [question_url_of(error) or "(link unknown)" for error in closed]
            lines.append(
                f"Missed {len(closed)} question(s) that closed before the forecast "
                "could be submitted: " + ", ".join(f"<{url}>" for url in urls)
            )
        costs = [
            f"{name} ${sum(stats.costs):.2f}"
            for name, stats in self.lane_stats.items()
            if stats.costs
        ]
        if costs:
            lines.append("Cost (estimate): " + ", ".join(costs))
        for name, failures in lane_failures.items():
            lines.append(
                f"Lane {name} failed on {len(self.lane_stats[name].failed_questions)} "
                f"question(s). First error: {failures[0]}"
            )
        # <...> keeps Discord from adding a preview card for every link.
        shown = 8
        for report in done[:shown]:
            lines.append(f"- <{report.question.page_url}>")
        if len(done) > shown:
            lines.append(f"- and {len(done) - shown} more")
        run_url = github_run_url()
        if run_url:
            lines.append(f"Run: <{run_url}>")
        return "\n".join(lines)

    def notify_run(
        self, run_mode: str, reports: list, published: bool, closed: list | None = None
    ) -> None:
        message = self.run_message(run_mode, reports, published, closed)
        if message:
            notify_discord(message)


# =============================================================================
# ENTRY POINT
# =============================================================================

RunMode = Literal["tournament", "metaculus_cup", "test_questions"]

TOURNAMENT_URLS = {
    "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
    "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
    "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
}


def pick_sample(questions: list[MetaculusQuestion], limit: int) -> list[MetaculusQuestion]:
    """Up to `limit` questions, taking a different question type each time."""
    if limit <= 0 or len(questions) <= limit:
        return questions
    by_type: dict[str, list[MetaculusQuestion]] = {}
    for question in questions:
        by_type.setdefault(type(question).__name__, []).append(question)
    picked: list[MetaculusQuestion] = []
    while len(picked) < limit:
        for group in by_type.values():
            if group and len(picked) < limit:
                picked.append(group.pop(0))
    return picked


async def run(bot: Cirrus, run_mode: RunMode, max_questions: int = 0) -> list:
    client = MetaculusClient()
    if run_mode == "tournament":
        seasonal_reports = await bot.forecast_on_tournament(
            client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
        )
        minibench_reports = await bot.forecast_on_tournament(
            client.CURRENT_MINIBENCH_ID, return_exceptions=True
        )
        return list(seasonal_reports) + list(minibench_reports)
    if run_mode == "metaculus_cup":
        bot.skip_previously_forecasted_questions = False
        return list(
            await bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    # test_questions: the bot-testing-area tournament is unscored and has one
    # question of every type. https://www.metaculus.com/tournament/bot-testing-area/
    bot.skip_previously_forecasted_questions = False
    questions = bot.metaculus_client.get_all_open_questions_from_tournament(
        "bot-testing-area"
    )
    sample = pick_sample(questions, max_questions)
    print(f"Forecasting on {len(sample)} of {len(questions)} test questions.\n")
    return list(await bot.forecast_questions(sample, return_exceptions=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the cirrus forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="What to forecast on (default: tournament)",
    )
    parser.add_argument(
        "--lanes",
        type=str,
        default=None,
        help="Lanes to run: both, claude, gpt or a comma-separated list "
        "(default: the CIRRUS_LANES env var, else both)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="In test_questions mode, forecast on at most this many questions, "
        "one per question type first (default: 0 = all)",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Forecast without submitting anything to Metaculus",
    )
    args = parser.parse_args(argv)
    if args.mode == "tournament" and args.no_publish:
        # Tournament rules: a bot maker may not preview how the bot forecasts
        # on open tournament questions. A dry run there would be such a preview.
        parser.error(
            "--no-publish cannot be used with --mode tournament: the tournament "
            "rules forbid previewing forecasts on open tournament questions. "
            "Use --mode test_questions for dry runs."
        )
    return args


def log_level_for(run_mode: str) -> int:
    """
    Detailed logs (research, reasoning, forecasts) are off in tournament mode.
    The logs of a public repository can be read by anyone, and forecasts on
    open tournament questions should stay private until the questions close.
    Warnings, errors, costs and question links are still shown.
    """
    return logging.WARNING if run_mode == "tournament" else logging.INFO


def main() -> None:
    args = parse_args()
    run_mode: RunMode = args.mode
    logging.basicConfig(
        level=log_level_for(run_mode),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if args.lanes:
        os.environ["CIRRUS_LANES"] = args.lanes

    check_environment(strict=True)
    publish_to_metaculus = not args.no_publish
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)

    lanes = active_lanes()
    missing = [name for name in requested_lane_names() if not LANES[name].is_available()]
    if missing:
        logger.warning(
            f"Lane(s) {missing} skipped: "
            f"{[LANES[name].api_key_env for name in missing]} not set."
        )
    print(f"Lanes: {[lane.name for lane in lanes]}")
    if log_level_for(run_mode) > logging.INFO:
        print("Detailed logs are off in tournament mode; see the comments on Metaculus.")
    print()

    bot = Cirrus(
        lanes=lanes,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
    )

    # One event loop for the whole run: the bot's semaphore and the provider
    # clients must not be shared between loops.
    forecast_reports = asyncio.run(run(bot, run_mode, args.max_questions))
    # A question that closed while the bot was still forecasting it is a missed
    # deadline, not a failure: it is reported but does not fail the run.
    closed, forecast_reports = split_closed(forecast_reports)

    print(bot.lane_report())
    bot.notify_run(run_mode, forecast_reports, publish_to_metaculus, closed)
    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )
    for error in closed:
        print(f"⏰  Closed before the forecast was submitted: {question_url_of(error) or error}")
    # Raises (and so fails the workflow run) if any question failed outright.
    bot.log_report_summary(forecast_reports)


if __name__ == "__main__":
    main()
