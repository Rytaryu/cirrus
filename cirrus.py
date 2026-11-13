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
import logging
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

# Forecasts each lane makes per question. With two lanes this gives 2 x 2 = 4
# forecasts per question. Both lanes always make the same number, so neither
# provider outweighs the other in the median.
PREDICTIONS_PER_LANE = 2

# Safety stop, in USD, per lane per question. If a lane's research alone costs
# more than this, the lane makes no forecasts for that question and the other
# lane carries it. The target is about $0.20 per lane per question; this stop
# only catches runaway cases.
LANE_COST_CAP_USD = 0.50

# How many questions' research may run at the same time.
MAX_CONCURRENT_RESEARCH = 4

# Web searches the Claude researcher may run per question. Search results are
# billed as input tokens and are re-read on every later step, so this is the
# main driver of the Claude lane's cost.
CLAUDE_MAX_SEARCHES = 3


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
        researcher=lambda: ResponsesLlm(
            model="openai/gpt-6.1-sol",
            tools=[{"type": "web_search", "search_context_size": "medium"}],
            reasoning={"effort": "low"},
            max_output_tokens=6000,
            timeout=300,
            allowed_tries=2,
        ),
        forecaster=lambda: ResponsesLlm(
            model="openai/gpt-6.1-sol",
            max_output_tokens=8000,
            timeout=240,
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
# What the current lane has spent on the current question so far.
_lane_spend: contextvars.ContextVar[MonetaryCostManager | None] = contextvars.ContextVar(
    "cirrus_lane_spend", default=None
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


def github_run_url() -> str | None:
    """Link to the current GitHub Actions run, when running in one."""
    server = os.getenv("GITHUB_SERVER_URL")
    repository = os.getenv("GITHUB_REPOSITORY")
    run_id = os.getenv("GITHUB_RUN_ID")
    if server and repository and run_id:
        return f"{server}/{repository}/actions/runs/{run_id}"
    return None


# =============================================================================
# THE BOT
# =============================================================================


@dataclass
class LaneStats:
    costs: list[float] = field(default_factory=list)  # one entry per question
    failures: list[str] = field(default_factory=list)
    # Where the money went, summed over the run.
    research_cost: float = 0.0
    summary_cost: float = 0.0
    forecast_cost: float = 0.0  # forecasts and the parsing of their output


class Cirrus(FallTemplateBot2026):
    """
    The template bot, run once per provider lane for every question.

    Forecast prompts and output parsing are inherited from the template. This
    class adds the lanes, a research prompt written for web search, a per-lane
    spending stop and per-lane cost reporting.
    """

    def __init__(self, *, lanes: list[Lane] | None = None, **kwargs) -> None:
        self.lanes: list[Lane] = lanes if lanes is not None else active_lanes()
        if not self.lanes:
            raise RuntimeError(
                "No lane is available. Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY."
            )
        kwargs.setdefault("llms", build_llms(self.lanes))
        kwargs.setdefault("research_reports_per_question", len(self.lanes))
        kwargs.setdefault("predictions_per_research_report", PREDICTIONS_PER_LANE)
        # One lane failing leaves half of the forecasts. Allow the question to
        # go ahead as long as at least one forecast survives.
        kwargs.setdefault("required_successful_predictions", 0.0)
        super().__init__(**kwargs)
        self.lane_stats: dict[str, LaneStats] = {
            lane.name: LaneStats() for lane in self.lanes
        }
        self._lane_turn: dict[int, int] = {}
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
        lane = self._take_lane(question)
        token = _current_lane.set(lane.name)
        stats = self.lane_stats[lane.name]
        try:
            # A limit of 0 means "track only"; the cap is checked in _make_prediction.
            with MonetaryCostManager(0) as cost_manager:
                spend_token = _lane_spend.set(cost_manager)
                try:
                    result = await super()._research_and_make_predictions(question)
                finally:
                    _lane_spend.reset(spend_token)
                    stats.costs.append(cost_manager.current_usage)
                    logger.info(
                        f"[{lane.name}] spent about ${cost_manager.current_usage:.4f} "
                        f"on {question.page_url}"
                    )
        except Exception as error:
            stats.failures.append(
                f"{question.page_url}: {type(error).__name__}: {scrub(str(error))}"
            )
            logger.error(f"[{lane.name}] lane failed on {question.page_url}: {error}")
            raise
        finally:
            _current_lane.reset(token)

        models = (
            f"research: {self.get_llm(f'{lane.name}/researcher', 'llm').model}, "
            f"forecasts: {self.get_llm(f'{lane.name}/default', 'llm').model}"
        )
        result.research_report = (
            f"*Lane: {lane.name} ({models})*\n\n{result.research_report}"
        )
        return result

    async def _make_prediction(
        self, question: MetaculusQuestion, research: str
    ) -> ReasonedPrediction:
        spend = _lane_spend.get()
        if spend is not None and spend.current_usage >= LANE_COST_CAP_USD:
            raise LaneCostCapReached(
                f"Research cost about ${spend.current_usage:.2f}, which is over the "
                f"${LANE_COST_CAP_USD:.2f} cap per lane per question."
            )
        with MonetaryCostManager(0) as cost_manager:
            try:
                return await super()._make_prediction(question, research)
            finally:
                self._add_cost("forecast_cost", cost_manager.current_usage)

    async def summarize_research(self, question: MetaculusQuestion, research: str) -> str:
        with MonetaryCostManager(0) as cost_manager:
            try:
                return await super().summarize_research(question, research)
            finally:
                self._add_cost("summary_cost", cost_manager.current_usage)

    def _add_cost(self, kind: str, amount: float) -> None:
        lane = _current_lane.get()
        if lane in self.lane_stats:
            stats = self.lane_stats[lane]
            setattr(stats, kind, getattr(stats, kind) + amount)

    # --------------------------------------------------------------- research

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._research_limiter:
            researcher = self.get_llm("researcher", "llm")
            with MonetaryCostManager(0) as cost_manager:
                try:
                    research = await researcher.invoke(self._research_prompt(question))
                finally:
                    self._add_cost("research_cost", cost_manager.current_usage)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

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
                f"${total / attempts:.4f} per question, {len(stats.failures)} failure(s)"
            )
            lines.append(
                f"    per question: research ${stats.research_cost / attempts:.4f}, "
                f"forecasts ${stats.forecast_cost / attempts:.4f}, "
                f"summary ${stats.summary_cost / attempts:.4f}"
            )
        return "\n".join(lines)

    def run_message(self, run_mode: str, reports: list, published: bool) -> str | None:
        """
        The Discord message for one run: what was submitted, what it cost and
        what failed. None when the run had nothing to forecast.
        """
        done = [r for r in reports if isinstance(r, ForecastReport)]
        failed = [r for r in reports if isinstance(r, BaseException)]
        lane_failures = {n: s.failures for n, s in self.lane_stats.items() if s.failures}
        if not done and not failed and not lane_failures:
            return None

        verb = "submitted" if published else "produced without submitting"
        lines = [f"cirrus ({run_mode}): {verb} {len(done)} forecast(s), {len(failed)} question(s) failed."]
        costs = [
            f"{name} ${sum(stats.costs):.2f}"
            for name, stats in self.lane_stats.items()
            if stats.costs
        ]
        if costs:
            lines.append("Cost (estimate): " + ", ".join(costs))
        for name, failures in lane_failures.items():
            lines.append(
                f"Lane {name} failed on {len(failures)} question(s). First error: {failures[0]}"
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

    def notify_run(self, run_mode: str, reports: list, published: bool) -> None:
        message = self.run_message(run_mode, reports, published)
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    run_mode: RunMode = args.mode
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
    print(f"Lanes: {[lane.name for lane in lanes]}\n")

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

    print(bot.lane_report())
    bot.notify_run(run_mode, forecast_reports, publish_to_metaculus)
    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )
    # Raises (and so fails the workflow run) if any question failed outright.
    bot.log_report_summary(forecast_reports)


if __name__ == "__main__":
    main()
