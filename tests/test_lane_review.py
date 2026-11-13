"""Offline tests for lane_review: scoring, record selection and the report."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lane_review  # noqa: E402


def record(time: str, claude: list, gpt: list, final: object, **extra: object) -> dict:
    return {
        "cirrus_record": 1,
        "commit": "abc1234",
        "time": time,
        "question": {"post": 1, "id": 10, "type": "BinaryQuestion", "url": "u"},
        "lanes": {
            "claude": {"predictions": claude, "cost": 0.19},
            "gpt": {"predictions": gpt, "cost": 0.18, **extra},
        },
        "final": final,
        "cost": 0.37,
    }


def outcome(question_type: str, resolution: str, **extra: object) -> dict:
    return {
        "post_id": 1,
        "question_id": 10,
        "title": "Will it happen?",
        "url": "https://www.metaculus.com/questions/1/",
        "question_type": question_type,
        "resolution": resolution,
        "spot_peer_score": 12.5,
        **extra,
    }


class ScoringTests(unittest.TestCase):
    def test_binary_score_is_zero_at_even_odds_and_100_when_certain(self) -> None:
        self.assertAlmostEqual(lane_review.baseline_score(0.5, "yes", "binary"), 0.0)
        self.assertAlmostEqual(lane_review.baseline_score(0.5, "no", "binary"), 0.0)
        self.assertAlmostEqual(lane_review.baseline_score(1.0, "yes", "binary"), 99.86, places=2)
        self.assertLess(lane_review.baseline_score(0.9, "no", "binary"), -200)

    def test_multiple_choice_score_uses_the_number_of_options(self) -> None:
        uniform = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        self.assertAlmostEqual(lane_review.baseline_score(uniform, "c", "multiple_choice"), 0.0)
        confident = {"a": 0.7, "b": 0.1, "c": 0.1, "d": 0.1}
        self.assertAlmostEqual(
            lane_review.baseline_score(confident, "a", "multiple_choice"), 74.27, places=2
        )

    def test_unscorable_cases_return_none(self) -> None:
        self.assertIsNone(lane_review.baseline_score(0.4, "annulled", "binary"))
        self.assertIsNone(lane_review.baseline_score(None, "yes", "binary"))
        self.assertIsNone(lane_review.baseline_score({"a": 1.0}, "b", "multiple_choice"))
        self.assertIsNone(lane_review.baseline_score([[0.1, 1], [0.9, 2]], "1.5", "numeric"))

    def test_combine_matches_how_the_bot_aggregates(self) -> None:
        self.assertEqual(lane_review.combine([0.2, 0.4], "binary"), 0.30000000000000004)
        self.assertEqual(lane_review.combine([0.2, 0.4, 0.9], "binary"), 0.4)
        mixed = lane_review.combine([{"a": 0.6, "b": 0.4}, {"a": 0.2, "b": 0.8}], "multiple_choice")
        self.assertAlmostEqual(mixed["a"], 0.4)
        self.assertAlmostEqual(mixed["b"], 0.6)
        self.assertIsNone(lane_review.combine([], "binary"))

    def test_interval_hit_reads_numbers_dates_and_out_of_bounds(self) -> None:
        table = [[0.1, 10], [0.5, 20], [0.9, 30]]
        self.assertEqual(lane_review.interval_hit(table, "25"), (True, 0.8))
        self.assertEqual(lane_review.interval_hit(table, "31"), (False, 0.8))
        self.assertEqual(lane_review.interval_hit(table, "above_upper_bound"), (False, 0.8))
        dates = [[0.1, "2026-01-01T00:00:00Z"], [0.9, "2026-03-01T00:00:00Z"]]
        self.assertEqual(lane_review.interval_hit(dates, "2026-02-01T00:00:00Z"), (True, 0.8))
        self.assertIsNone(lane_review.interval_hit(table, "yes"))
        self.assertIsNone(lane_review.interval_hit("affirmed", "5"))


class RecordSelectionTests(unittest.TestCase):
    def test_the_record_standing_at_scoring_time_is_used(self) -> None:
        early = record("2026-11-16T10:00:00Z", [0.2], [0.3], 0.25)
        late = record("2026-11-16T15:00:00Z", [0.8], [0.9], 0.85)
        scored_at = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
        self.assertIs(lane_review.standing_record([late, early], scored_at), early)
        self.assertIs(lane_review.standing_record([late, early], None), late)
        before_any = datetime(2026, 11, 16, 9, 0, tzinfo=timezone.utc)
        self.assertIsNone(lane_review.standing_record([late, early], before_any))
        self.assertIsNone(lane_review.standing_record([], scored_at))


class ReportTests(unittest.TestCase):
    def rows(self) -> list[dict]:
        return [
            lane_review.score_question(
                outcome("binary", "yes"),
                record("2026-11-16T10:00:00Z", [0.8, 0.9], [0.4, 0.6], 0.7),
            ),
            lane_review.score_question(
                outcome("binary", "no", post_id=2, question_id=11),
                record("2026-11-16T10:00:00Z", [0.3, 0.3], [0.1, 0.1], 0.2),
            ),
            lane_review.score_question(
                outcome("numeric", "25", post_id=3, question_id=12),
                record(
                    "2026-11-16T10:00:00Z",
                    [[[0.1, 10], [0.9, 30]], [[0.1, 26], [0.9, 40]]],
                    [],
                    [[0.1, 12], [0.9, 28]],
                    error="RateLimitError",
                ),
            ),
            lane_review.score_question(outcome("binary", "yes", post_id=4, question_id=13), None),
        ]

    def test_each_lane_and_the_aggregate_are_scored(self) -> None:
        first = self.rows()[0]
        self.assertAlmostEqual(first["lanes"]["claude"]["score"], 76.55, places=2)
        self.assertAlmostEqual(first["lanes"]["gpt"]["score"], 0.0)
        self.assertAlmostEqual(first["final"]["score"], 48.54, places=2)
        self.assertEqual(first["commit"], "abc1234")
        self.assertEqual(first["lanes"]["claude"]["forecast"], 0.85)
        self.assertEqual(first["final"]["forecast"], 0.7)

    def test_summary_counts_and_compares_on_shared_questions(self) -> None:
        summary = lane_review.summarize(self.rows())
        self.assertEqual(summary["questions"], 4)
        self.assertEqual(summary["with_record"], 3)
        self.assertEqual(summary["by_type"], {"binary": 3, "numeric": 1})
        self.assertEqual(summary["lanes"]["claude"]["scored"], 2)
        self.assertEqual(summary["lanes"]["gpt"]["failures"], 1)
        self.assertEqual(summary["lanes"]["claude"]["cost"], 0.57)
        self.assertEqual(
            summary["lanes"]["claude"]["ranges"],
            {"forecasts": 2, "hit_rate": 0.5, "expected": 0.8},
        )
        self.assertIsNone(summary["lanes"]["gpt"]["ranges"])
        self.assertEqual(summary["final"]["ranges"]["hit_rate"], 1.0)
        self.assertEqual(summary["spot_peer"], {"questions": 4, "total": 50.0, "mean": 12.5})
        head_to_head = summary["comparisons"][0]
        self.assertEqual((head_to_head["first"], head_to_head["second"]), ("claude", "gpt"))
        self.assertEqual(head_to_head["questions"], 2)
        self.assertEqual(head_to_head["first_better"], 1)
        self.assertEqual(head_to_head["second_better"], 1)

    def test_report_renders_and_round_trips_through_json(self) -> None:
        rows = self.rows()
        report = lane_review.render(lane_review.summarize(rows), rows, "test")
        self.assertIn("| claude alone | 2 |", report)
        self.assertIn("| submitted aggregate | 2 |", report)
        self.assertIn("claude vs gpt", report)
        self.assertIn("Lowest scoring submitted forecasts", report)
        with tempfile.TemporaryDirectory() as folder:
            saved = Path(folder) / "rows.json"
            saved.write_text(json.dumps({"title": "test", "rows": rows}))
            with redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(lane_review.main(["--from-json", str(saved)]), 0)
        self.assertEqual(printed.getvalue().strip(), report.strip())

    def test_empty_input_still_renders(self) -> None:
        report = lane_review.render(lane_review.summarize([]), [], "nothing yet")
        self.assertIn("0 resolved questions", report)
        self.assertIn("Nothing to compare yet.", report)


class AccessTests(unittest.TestCase):
    def test_placeholder_token_means_the_network_supplies_it(self) -> None:
        for value in ("", "placeholder", "Placeholder", "proxy"):
            with mock.patch.dict(os.environ, {"METACULUS_TOKEN": value}):
                self.assertTrue(lane_review.uses_proxy_auth())
        with mock.patch.dict(os.environ, {"METACULUS_TOKEN": "a-real-token"}):
            self.assertFalse(lane_review.uses_proxy_auth())
        with mock.patch.dict(os.environ, clear=True):
            self.assertTrue(lane_review.uses_proxy_auth())

    def test_proxy_auth_sends_no_authorization_header(self) -> None:
        from forecasting_tools.helpers.metaculus_client import MetaculusClient

        original = MetaculusClient._get_auth_headers
        try:
            with mock.patch.dict(os.environ, {"METACULUS_TOKEN": "placeholder"}):
                lane_review.enable_proxy_auth()
                headers = MetaculusClient()._get_auth_headers()["headers"]
        finally:
            MetaculusClient._get_auth_headers = original
        self.assertNotIn("Authorization", headers)

    def test_stock_review_is_limited_to_resolved_questions(self) -> None:
        for blocked in (["review", "--tournament", "minibench"], ["review", "--post", "123"]):
            with redirect_stderr(io.StringIO()) as complaint:
                self.assertEqual(lane_review.run_bot_review(blocked), 2)
            self.assertIn("open questions", complaint.getvalue())


try:
    import metaculus_bot_review  # noqa: F401

    HAS_REVIEW_TOOL = True
except ImportError:
    HAS_REVIEW_TOOL = False


@unittest.skipUnless(HAS_REVIEW_TOOL, "needs the optional integrations group")
class FetchTests(unittest.TestCase):
    def test_rows_are_built_from_resolved_questions_and_their_records(self) -> None:
        from types import SimpleNamespace as Thing

        def question(post_id: int, question_id: int, **extra: object) -> Thing:
            fields = {
                "post_id": post_id,
                "question_id": question_id,
                "title": f"Q{question_id}",
                "url": f"https://www.metaculus.com/questions/{post_id}/",
                "question_type": "binary",
                "status": "resolved",
                "resolution": "yes",
                "spot_scoring_time": datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc),
                "scores": Thing(spot_peer_score=5.0),
                **extra,
            }
            return Thing(**fields)

        def comment(comment_id: int, post_id: int, body: dict, archived: bool = False) -> Thing:
            text = "report\n\n# RECORD\n```\n" + json.dumps(body) + "\n```"
            return Thing(
                id=comment_id,
                on_post=post_id,
                text="cut short" if archived else text,
                full_text=text,
                is_text_archived=archived,
            )

        first = record("2026-11-16T10:00:00Z", [0.8], [0.6], 0.7)
        second = record("2026-11-16T10:00:00Z", [0.2], [0.4], 0.3)
        second["question"]["id"] = 11
        elsewhere = record("2026-11-16T10:00:00Z", [0.5], [0.5], 0.5)
        elsewhere["question"]["id"] = 99
        comments = {
            True: [comment(1, 1, first), comment(2, 2, second, archived=True)],
            False: [comment(3, 9, elsewhere)],
        }
        client = mock.Mock()
        client.get_current_user_id.return_value = 7
        client.get_own_comments.side_effect = lambda is_private, **_: comments[is_private]
        client.get_comment.side_effect = lambda comment_id: Thing(
            text=next(c.full_text for c in comments[True] if c.id == comment_id)
        )
        outcomes = [
            question(1, 10),
            question(2, 11),
            question(3, 12, resolution="annulled"),
        ]
        with (
            mock.patch("metaculus_bot_review.client.ReviewClient", return_value=client),
            mock.patch(
                "metaculus_bot_review.outcomes.get_recently_resolved_outcomes",
                return_value=outcomes,
            ),
            redirect_stderr(io.StringIO()),
        ):
            rows = lane_review.fetch_rows(30, 100, 0)
        self.assertEqual([row["question_id"] for row in rows], [10, 11])
        self.assertAlmostEqual(rows[0]["lanes"]["claude"]["score"], 67.81, places=2)
        self.assertAlmostEqual(rows[1]["final"]["score"], -73.7, places=1)
        self.assertEqual(rows[0]["spot_peer_score"], 5.0)
        client.get_comment.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
