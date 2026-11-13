"""
Offline tests for cirrus.py. No real API is called: a local HTTP server stands
in for the Anthropic Messages API and the OpenAI Responses API.

Run with:  uv run python -m unittest discover -s tests -v
"""

import asyncio
import json
import os
import re
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REQUESTS: list[tuple[str, dict]] = []
DOWN = {"anthropic": False, "openai": False}
CLAUDE_FORECAST = "(a) Two months are left ... Probability: 30%"
GPT_FORECAST = "(a) Two months are left ... Probability: 50%"


def _probability_in(text: str) -> float | None:
    found = re.findall(r"Probability:\s*(\d+)%", text)
    return int(found[-1]) / 100 if found else None


class FakeProviders(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # keep test output quiet
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQUESTS.append((self.path, body))
        flat = json.dumps(body)
        is_parser = "START TEXT" in flat
        is_summary = "summarize the following research" in flat
        if "/messages" in self.path:
            self._anthropic(body, flat, is_parser, is_summary)
        elif "/responses" in self.path:
            self._openai(body, flat, is_parser, is_summary)
        else:
            self._send(404, {"error": "unknown path"})

    def _anthropic(self, body: dict, flat: str, is_parser: bool, is_summary: bool) -> None:
        if DOWN["anthropic"]:
            self._send(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "Your credit balance is too low to access the Anthropic API."}})
            return
        if body.get("tools"):
            content = [
                {"type": "text", "text": "I'll search for this. "},
                {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "x"}},
                {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [{"type": "web_search_result", "url": "https://example.com/a", "title": "A", "encrypted_content": "abc", "page_age": "Nov 1, 2026"}]},
                {"type": "text", "text": "CLAUDE RESEARCH: nothing has changed (https://example.com/a)."},
            ]
            usage = {"input_tokens": 9000, "output_tokens": 400, "server_tool_use": {"web_search_requests": 1}}
        elif is_parser:
            content = [{"type": "text", "text": json.dumps({"prediction_in_decimal": _probability_in(flat)})}]
            usage = {"input_tokens": 900, "output_tokens": 20}
        elif is_summary:
            content = [{"type": "text", "text": "Claude summary."}]
            usage = {"input_tokens": 600, "output_tokens": 30}
        else:
            content = [{"type": "text", "text": CLAUDE_FORECAST}]
            usage = {"input_tokens": 3000, "output_tokens": 800}
        self._send(200, {"id": "msg_1", "type": "message", "role": "assistant", "model": body["model"], "content": content, "stop_reason": "end_turn", "stop_sequence": None, "usage": usage})

    def _openai(self, body: dict, flat: str, is_parser: bool, is_summary: bool) -> None:
        if DOWN["openai"]:
            self._send(429, {"error": {"message": "You exceeded your current quota", "type": "insufficient_quota", "code": "credit_balance_exhausted"}})
            return

        def message(index: int, text: str) -> dict:
            return {"id": f"msg_{index}", "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": text, "annotations": []}]}

        output: list[dict] = [{"id": "rs_1", "type": "reasoning", "summary": []}]
        if body.get("tools"):
            output += [
                message(0, "Let me look that up."),
                {"id": "ws_1", "type": "web_search_call", "status": "completed", "action": {"type": "search", "query": "x"}},
                message(1, "GPT RESEARCH: the latest figure is 12 ([source](https://example.org/b))."),
            ]
        elif is_parser:
            output.append(message(1, json.dumps({"prediction_in_decimal": _probability_in(flat)})))
        elif is_summary:
            output.append(message(1, "GPT summary."))
        else:
            output.append(message(1, GPT_FORECAST))
        usage = {"input_tokens": 3000, "output_tokens": 500, "total_tokens": 3500, "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 100}}
        self._send(200, {"id": "resp_1", "object": "response", "created_at": 1794550000, "status": "completed", "model": body["model"], "output": output, "usage": usage, "error": None, "incomplete_details": None, "parallel_tool_calls": True, "tool_choice": "auto", "tools": body.get("tools", []), "metadata": {}, "temperature": 1.0, "top_p": 1.0, "instructions": None, "max_output_tokens": body.get("max_output_tokens"), "previous_response_id": None, "reasoning": {"effort": "medium", "summary": None}, "text": {"format": {"type": "text"}}, "truncation": "disabled", "user": None})

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


_server = ThreadingHTTPServer(("127.0.0.1", 0), FakeProviders)
_port = _server.server_address[1]
threading.Thread(target=_server.serve_forever, daemon=True).start()

# Fake keys and local endpoints must be in place before cirrus is imported.
os.environ.update(
    METACULUS_TOKEN="test-token",
    ANTHROPIC_API_KEY="test-anthropic",
    OPENAI_API_KEY="test-openai",
    ANTHROPIC_API_BASE=f"http://127.0.0.1:{_port}",
    OPENAI_BASE_URL=f"http://127.0.0.1:{_port}/v1",
    OPENAI_API_BASE=f"http://127.0.0.1:{_port}/v1",
)
os.environ.pop("DISCORD_WEBHOOK_URL", None)
os.environ.pop("CIRRUS_LANES", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cirrus  # noqa: E402
from forecasting_tools import BinaryQuestion  # noqa: E402


def make_question(number: int) -> BinaryQuestion:
    return BinaryQuestion(
        question_text=f"Will event {number} happen?",
        id_of_post=number,
        id_of_question=100 + number,
        page_url=f"https://example.com/q/{number}",
        background_info="Background.",
        resolution_criteria="Resolves Yes if the event happens.",
        fine_print="",
    )


def forecast(questions: list[BinaryQuestion], **bot_kwargs):
    bot = cirrus.Cirrus(
        publish_reports_to_metaculus=False,
        skip_previously_forecasted_questions=False,
        extra_metadata_in_explanation=True,
        **bot_kwargs,
    )
    reports = asyncio.run(bot.forecast_questions(questions, return_exceptions=True))
    return bot, reports


class CirrusTests(unittest.TestCase):
    def setUp(self) -> None:
        REQUESTS.clear()
        DOWN.update(anthropic=False, openai=False)
        os.environ.pop("CIRRUS_LANES", None)

    def test_both_lanes_are_aggregated(self) -> None:
        bot, reports = forecast([make_question(1), make_question(2)])
        for report in reports:
            # Two Claude forecasts at 30% and two GPT forecasts at 50%.
            self.assertAlmostEqual(report.prediction, 0.40)
            self.assertEqual(report.errors, [])
            self.assertIn("*Lane: claude", report.explanation)
            self.assertIn("*Lane: gpt", report.explanation)
            # The note written before the search is kept, not read as the answer.
            self.assertIn("Let me look that up.", report.explanation)
            self.assertIn("GPT RESEARCH", report.explanation)
        self.assertEqual(len(bot.lane_stats["claude"].costs), 2)
        self.assertEqual(len(bot.lane_stats["gpt"].costs), 2)

    def test_requests_use_the_configured_models_and_search_tools(self) -> None:
        forecast([make_question(1)])
        seen = {(path, body["model"], json.dumps(body.get("tools"))) for path, body in REQUESTS}
        claude_search = json.dumps([{"type": "web_search_20250305", "name": "web_search", "max_uses": cirrus.CLAUDE_MAX_SEARCHES}])
        gpt_search = json.dumps([{"type": "web_search", "search_context_size": "medium"}])
        self.assertIn(("/v1/messages", "claude-sonnet-5-5", claude_search), seen)
        self.assertIn(("/v1/messages", "claude-opus-5-5", "null"), seen)
        self.assertIn(("/v1/messages", "claude-haiku-5-5", "null"), seen)
        self.assertIn(("/v1/responses", "gpt-6.1-sol", gpt_search), seen)
        self.assertIn(("/v1/responses", "gpt-6.1-sol", "null"), seen)
        self.assertIn(("/v1/responses", "gpt-6-luna", "null"), seen)

    def test_each_lane_only_sees_its_own_research(self) -> None:
        forecast([make_question(1)])
        forecaster_calls = [(path, json.dumps(body)) for path, body in REQUESTS if "interviewing for a job" in json.dumps(body)]
        self.assertEqual(len(forecaster_calls), 2 * cirrus.PREDICTIONS_PER_LANE)
        for path, flat in forecaster_calls:
            self.assertNotIn("*Lane:", flat)
            if "/messages" in path:
                self.assertIn("CLAUDE RESEARCH", flat)
                self.assertNotIn("GPT RESEARCH", flat)
            else:
                self.assertIn("GPT RESEARCH", flat)
                self.assertNotIn("CLAUDE RESEARCH", flat)

    def test_gpt_lane_down_leaves_claude_forecast(self) -> None:
        DOWN["openai"] = True
        bot, reports = forecast([make_question(1)])
        self.assertAlmostEqual(reports[0].prediction, 0.30)
        self.assertEqual(len(bot.lane_stats["gpt"].failures), 1)
        self.assertEqual(bot.lane_stats["claude"].failures, [])

    def test_claude_lane_down_leaves_gpt_forecast(self) -> None:
        DOWN["anthropic"] = True
        bot, reports = forecast([make_question(1)])
        self.assertAlmostEqual(reports[0].prediction, 0.50)
        self.assertEqual(len(bot.lane_stats["claude"].failures), 1)

    def test_both_lanes_down_fails_the_question(self) -> None:
        DOWN.update(anthropic=True, openai=True)
        _, reports = forecast([make_question(1)])
        self.assertIsInstance(reports[0], BaseException)

    def test_single_lane_selection(self) -> None:
        os.environ["CIRRUS_LANES"] = "claude"
        bot, reports = forecast([make_question(1)])
        self.assertEqual([lane.name for lane in bot.lanes], ["claude"])
        self.assertAlmostEqual(reports[0].prediction, 0.30)
        self.assertFalse(any("/responses" in path for path, _ in REQUESTS))

    def test_unknown_lane_is_rejected(self) -> None:
        os.environ["CIRRUS_LANES"] = "claude,gemini"
        with self.assertRaises(ValueError):
            cirrus.requested_lane_names()

    def test_lane_without_key_is_skipped(self) -> None:
        saved = os.environ.pop("OPENAI_API_KEY")
        try:
            self.assertEqual([lane.name for lane in cirrus.active_lanes()], ["claude"])
        finally:
            os.environ["OPENAI_API_KEY"] = saved

    def test_cost_cap_stops_forecasts_when_research_is_too_expensive(self) -> None:
        original = cirrus.LANE_COST_CAP_USD
        cirrus.LANE_COST_CAP_USD = 0.000001  # research alone exceeds this
        try:
            _, reports = forecast([make_question(1)])
        finally:
            cirrus.LANE_COST_CAP_USD = original
        self.assertIsInstance(reports[0], BaseException)
        self.assertIn("LaneCostCapReached", str(reports[0]))
        self.assertFalse(any("interviewing for a job" in json.dumps(body) for _, body in REQUESTS))

    def test_costs_are_split_by_role(self) -> None:
        bot, _ = forecast([make_question(1)])
        for name in ("claude", "gpt"):
            stats = bot.lane_stats[name]
            self.assertGreater(stats.research_cost, 0)
            self.assertGreater(stats.forecast_cost, 0)
            self.assertGreater(stats.summary_cost, 0)
            parts = stats.research_cost + stats.forecast_cost + stats.summary_cost
            self.assertAlmostEqual(parts, sum(stats.costs), places=6)
        self.assertIn("per question: research $", bot.lane_report())

    def test_reported_cost_matches_token_prices(self) -> None:
        # Prices in USD per token (input, output), from the providers' price lists.
        sonnet, opus, haiku = (2e-6, 1e-5), (4e-6, 2e-5), (1e-7, 5e-7)
        sol, luna = (2e-6, 1e-5), (1e-7, 5e-7)
        search = 0.01  # one web search
        forecasts = cirrus.PREDICTIONS_PER_LANE
        parses = forecasts * 2  # each forecast is parsed twice for validation

        def cost(price: tuple[float, float], tokens_in: int, tokens_out: int) -> float:
            return price[0] * tokens_in + price[1] * tokens_out

        # Token counts are the ones FakeProviders returns.
        claude = (
            cost(sonnet, 9000, 400) + search
            + cost(haiku, 600, 30)
            + forecasts * cost(opus, 3000, 800)
            + parses * cost(haiku, 900, 20)
        )
        gpt = (
            cost(sol, 3000, 500) + search
            + cost(luna, 3000, 500)
            + forecasts * cost(sol, 3000, 500)
            + parses * cost(luna, 3000, 500)
        )
        bot, reports = forecast([make_question(1)])
        self.assertAlmostEqual(sum(bot.lane_stats["claude"].costs), claude, places=6)
        self.assertAlmostEqual(sum(bot.lane_stats["gpt"].costs), gpt, places=6)
        self.assertAlmostEqual(reports[0].price_estimate, claude + gpt, places=6)

    def test_pick_sample_prefers_different_question_types(self) -> None:
        class Numeric(BinaryQuestion):
            pass

        binary = [make_question(n) for n in (1, 2, 3)]
        numeric = [Numeric(**make_question(n).model_dump()) for n in (4, 5)]
        picked = cirrus.pick_sample(binary + numeric, 3)
        self.assertEqual([q.id_of_post for q in picked], [1, 4, 2])
        self.assertEqual(len(cirrus.pick_sample(binary, 0)), 3)
        self.assertEqual(len(cirrus.pick_sample(binary, 10)), 3)

    def test_run_message_reports_submissions_costs_and_lane_failures(self) -> None:
        DOWN["openai"] = True
        bot, reports = forecast([make_question(1), make_question(2)])
        os.environ.update(GITHUB_SERVER_URL="https://github.com", GITHUB_REPOSITORY="o/r", GITHUB_RUN_ID="7")
        try:
            message = bot.run_message("tournament", reports, published=True)
        finally:
            for name in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"):
                os.environ.pop(name)
        self.assertIn("submitted 2 forecast(s), 0 question(s) failed", message)
        self.assertIn("claude $", message)
        self.assertIn("Lane gpt failed on 2 question(s)", message)
        self.assertIn("- <https://example.com/q/1>", message)
        self.assertIn("Run: <https://github.com/o/r/actions/runs/7>", message)
        self.assertLess(len(message), 1900)

    def test_run_message_is_empty_when_nothing_happened(self) -> None:
        bot, reports = forecast([])
        self.assertIsNone(bot.run_message("tournament", reports, published=True))

    def test_tournament_mode_refuses_dry_runs(self) -> None:
        with self.assertRaises(SystemExit):
            cirrus.parse_args(["--mode", "tournament", "--no-publish"])
        self.assertTrue(cirrus.parse_args(["--mode", "test_questions", "--no-publish"]).no_publish)
        self.assertEqual(cirrus.parse_args([]).mode, "tournament")

    def test_comment_ends_with_a_record_of_each_lane(self) -> None:
        os.environ["GITHUB_SHA"] = "abc1234def"
        try:
            _, reports = forecast([make_question(1)])
        finally:
            os.environ.pop("GITHUB_SHA")
        report = reports[0]
        record = cirrus.read_record(report.explanation)
        self.assertEqual(record["cirrus_record"], 1)
        self.assertEqual(record["commit"], "abc1234")
        self.assertEqual(record["question"]["type"], "BinaryQuestion")
        self.assertEqual(record["question"]["url"], "https://example.com/q/1")
        self.assertEqual(record["lanes"]["claude"]["predictions"], [0.3, 0.3])
        self.assertEqual(record["lanes"]["gpt"]["predictions"], [0.5, 0.5])
        self.assertEqual(record["lanes"]["claude"]["forecast_model"], "anthropic/claude-opus-5-5")
        self.assertGreater(record["lanes"]["gpt"]["cost"], 0)
        self.assertEqual(record["final"], 0.4)
        # The template's own sections are still where it expects them.
        self.assertIn("SUMMARY", report.summary)
        self.assertIn("Probability: 30%", report.first_rationale)

    def test_record_marks_a_failed_lane(self) -> None:
        DOWN["openai"] = True
        _, reports = forecast([make_question(1)])
        record = cirrus.read_record(reports[0].explanation)
        self.assertEqual(record["lanes"]["gpt"]["error"], "RateLimitError")
        self.assertEqual(record["lanes"]["gpt"]["predictions"], [])
        self.assertEqual(record["lanes"]["claude"]["predictions"], [0.3, 0.3])
        self.assertEqual(record["final"], 0.3)

    def test_compact_prediction_handles_every_prediction_type(self) -> None:
        from forecasting_tools import NumericDistribution, Percentile, PredictedOptionList
        from forecasting_tools.data_models.multiple_choice_report import PredictedOption

        options = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.7),
                PredictedOption(option_name="B", probability=0.3),
            ]
        )
        self.assertEqual(cirrus.compact_prediction(options), {"A": 0.7, "B": 0.3})
        numeric = NumericDistribution(
            declared_percentiles=[Percentile(percentile=p / 100, value=float(p)) for p in range(5, 100, 5)],
            open_upper_bound=True,
            open_lower_bound=True,
            upper_bound=100,
            lower_bound=0,
            zero_point=None,
        )
        points = cirrus.compact_prediction(numeric)
        self.assertLessEqual(len(points), 11)
        self.assertEqual(points[0], [0.05, 5.0])
        self.assertEqual(points[-1], [0.95, 95.0])
        self.assertEqual(cirrus.compact_prediction(0.123456), 0.1235)

    def test_tournament_mode_hides_detailed_logs(self) -> None:
        import logging

        self.assertEqual(cirrus.log_level_for("tournament"), logging.WARNING)
        self.assertEqual(cirrus.log_level_for("test_questions"), logging.INFO)

    def test_scrub_hides_key_like_strings(self) -> None:
        cleaned = cirrus.scrub("Incorrect API key provided: sk-proj-abcdEFGH12345678\nnext line")
        self.assertNotIn("sk-proj", cleaned)
        self.assertNotIn("\n", cleaned)


if __name__ == "__main__":
    unittest.main()
