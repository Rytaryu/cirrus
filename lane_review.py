"""Compare cirrus's lanes on questions that have resolved.

Every comment cirrus posts ends with a record line holding each lane's own
forecasts next to the aggregate that was submitted. This script reads those
records back for resolved questions and scores each lane, so the question
"would Claude alone or GPT alone have done better than the aggregate?" has a
measured answer.

    uv run --group integrations python lane_review.py --resolved-since 30 \
        --output lane-review.json --summary lane-review.md

    # the stock review tool, with the same authentication and the same limits
    uv run --group integrations python lane_review.py bot-review review --resolved-since 30
    uv run --group integrations python lane_review.py bot-review show POST_ID --section research

Only resolved questions are ever fetched or printed. Tournament rules forbid
looking at the bot's forecasts on open questions and then changing the bot, so
there is no option that lists them.

Authentication: with a real METACULUS_TOKEN the token is sent as usual. With
the variable unset (or set to "placeholder"), no Authorization header is sent
and the network layer is expected to add one, which is how a Claude cloud
environment's network secret works.

Read-only: nothing here forecasts, comments or spends API credits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

PLACEHOLDER_TOKENS = {"", "placeholder", "proxy"}
UNSCORED_RESOLUTIONS = {None, "", "annulled", "ambiguous"}
CONTINUOUS_TYPES = ("numeric", "date", "discrete")
MIN_PROBABILITY = 0.001
DEFAULT_DAYS = 45
DEFAULT_MAX_COMMENTS = 3000


# =============================================================================
# AUTHENTICATION
# =============================================================================


def uses_proxy_auth() -> bool:
    """True when no real token is configured and the network must supply one."""
    return os.getenv("METACULUS_TOKEN", "").strip().lower() in PLACEHOLDER_TOKENS


def enable_proxy_auth() -> None:
    """Make every Metaculus client send requests without an Authorization header."""
    from forecasting_tools.helpers.metaculus_client import MetaculusClient

    os.environ["METACULUS_TOKEN"] = "proxy"  # the client refuses to start without one
    MetaculusClient._get_auth_headers = lambda self: {  # type: ignore[method-assign]
        "headers": {"Accept-Language": "en"}
    }


# =============================================================================
# SCORING (no network)
# =============================================================================


def _as_number(value: Any) -> float | None:
    """A number or a date as a float, or None if it is neither."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def combine(predictions: list, question_type: str | None) -> Any:
    """
    What one lane would have submitted on its own.

    Binary forecasts are combined by median and multiple choice by mean, the
    same way the bot combines all of its forecasts. Continuous forecasts are
    left as a list and judged one by one.
    """
    if not predictions:
        return None
    if question_type == "binary":
        values = [p for p in predictions if isinstance(p, (int, float))]
        return statistics.median(values) if values else None
    if question_type == "multiple_choice":
        tables = [p for p in predictions if isinstance(p, dict)]
        if not tables:
            return None
        options = list(dict.fromkeys(option for table in tables for option in table))
        return {
            option: sum(table.get(option, 0.0) for table in tables) / len(tables)
            for option in options
        }
    return predictions


def baseline_score(
    prediction: Any, resolution: str | None, question_type: str | None
) -> float | None:
    """
    Metaculus-style baseline score for a binary or multiple choice forecast.

    0 is what a uniform guess earns, 100 is a perfect forecast, and confident
    misses go far below zero. Continuous questions return None.
    """
    if prediction is None or resolution in UNSCORED_RESOLUTIONS:
        return None
    if question_type == "binary" and isinstance(prediction, (int, float)):
        if resolution not in ("yes", "no"):
            return None
        probability = prediction if resolution == "yes" else 1 - prediction
        option_count = 2
    elif question_type == "multiple_choice" and isinstance(prediction, dict):
        if resolution not in prediction or len(prediction) < 2:
            return None
        probability = prediction[resolution]
        option_count = len(prediction)
    else:
        return None
    probability = min(max(probability, MIN_PROBABILITY), 1 - MIN_PROBABILITY)
    return 100 * (math.log(probability) / math.log(option_count) + 1)


def interval_hit(prediction: Any, resolution: str | None) -> tuple[bool, float] | None:
    """
    Whether a continuous forecast's stated range contained the answer.

    :return: (hit, share of probability the range was supposed to hold), or
        None when the forecast or the resolution cannot be read as numbers
    """
    if not isinstance(prediction, list) or len(prediction) < 2:
        return None
    points = []
    for point in prediction:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        percentile, value = _as_number(point[0]), _as_number(point[1])
        if percentile is None or value is None:
            return None
        points.append((percentile, value))
    points.sort()
    (low_percentile, low), (high_percentile, high) = points[0], points[-1]
    nominal = high_percentile - low_percentile
    if resolution in ("above_upper_bound", "below_lower_bound"):
        return False, nominal
    answer = _as_number(resolution)
    if answer is None:
        return None
    return low <= answer <= high, nominal


def standing_record(records: list[dict], spot_scoring_time: datetime | None) -> dict | None:
    """The record of the run whose forecast was standing when the question was scored."""
    if not records:
        return None
    ordered = sorted(records, key=lambda record: record.get("time", ""))
    if spot_scoring_time is None:
        return ordered[-1]
    cutoff = spot_scoring_time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    standing = [record for record in ordered if record.get("time", "") <= cutoff]
    return standing[-1] if standing else None


def _judge(prediction: Any, resolution: str | None, question_type: str | None) -> dict:
    """Score one forecast (a lane's own, or the submitted aggregate)."""
    if question_type in CONTINUOUS_TYPES:
        forecasts = prediction if _is_list_of_forecasts(prediction) else [prediction]
        hits = [hit for hit in (interval_hit(f, resolution) for f in forecasts) if hit]
        return {
            "score": None,
            "forecast": None,
            "hits": [[hit, round(nominal, 4)] for hit, nominal in hits],
        }
    score = baseline_score(prediction, resolution, question_type)
    return {
        "score": None if score is None else round(score, 2),
        "forecast": _rounded(prediction),
        "hits": [],
    }


def _rounded(prediction: Any) -> Any:
    """A binary or multiple choice forecast, rounded for the saved rows."""
    if isinstance(prediction, (int, float)) and not isinstance(prediction, bool):
        return round(prediction, 4)
    if isinstance(prediction, dict):
        return {option: round(value, 4) for option, value in prediction.items()}
    return None


def _is_list_of_forecasts(prediction: Any) -> bool:
    """True for a list of percentile tables, False for a single percentile table."""
    return (
        isinstance(prediction, list)
        and bool(prediction)
        and isinstance(prediction[0], list)
        and bool(prediction[0])
        and isinstance(prediction[0][0], (list, tuple))
    )


def score_question(outcome: dict, record: dict | None) -> dict:
    """
    One row of the comparison: a resolved question, each lane's result, the aggregate's.

    :param outcome: post_id, question_id, title, url, question_type, resolution
        and spot_peer_score for a resolved question
    :param record: the cirrus record standing when the question was scored
    """
    question_type = outcome.get("question_type")
    resolution = outcome.get("resolution")
    row = {
        **{key: outcome.get(key) for key in ("post_id", "question_id", "title", "url")},
        "type": question_type,
        "resolution": resolution,
        "spot_peer_score": outcome.get("spot_peer_score"),
        "commit": record.get("commit") if record else None,
        "lanes": {},
        "final": None,
    }
    if record is None:
        return row
    for name, lane in (record.get("lanes") or {}).items():
        predictions = lane.get("predictions") or []
        row["lanes"][name] = {
            **_judge(combine(predictions, question_type), resolution, question_type),
            "forecasts": len(predictions),
            "cost": lane.get("cost"),
            "error": lane.get("error"),
        }
    row["final"] = _judge(record.get("final"), resolution, question_type)
    return row


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def _hit_rate(hits: list[list]) -> dict | None:
    if not hits:
        return None
    return {
        "forecasts": len(hits),
        "hit_rate": round(sum(1 for hit, _ in hits if hit) / len(hits), 3),
        "expected": round(sum(nominal for _, nominal in hits) / len(hits), 3),
    }


def _paired(rows: list[dict], first: str, second: str) -> dict | None:
    """Compare two forecasters on the questions where both have a score."""
    pairs = []
    for row in rows:
        a = row["final"] if first == "final" else row["lanes"].get(first)
        b = row["final"] if second == "final" else row["lanes"].get(second)
        if a and b and a["score"] is not None and b["score"] is not None:
            pairs.append((a["score"], b["score"]))
    if not pairs:
        return None
    differences = [a - b for a, b in pairs]
    spread = statistics.stdev(differences) if len(differences) > 1 else None
    return {
        "first": first,
        "second": second,
        "questions": len(pairs),
        "first_mean": _mean([a for a, _ in pairs]),
        "second_mean": _mean([b for _, b in pairs]),
        "mean_difference": _mean(differences),
        "standard_error": (
            None if spread is None else round(spread / math.sqrt(len(differences)), 2)
        ),
        "first_better": sum(1 for d in differences if d > 0),
        "second_better": sum(1 for d in differences if d < 0),
    }


def summarize(rows: list[dict]) -> dict:
    """Totals per lane and for the aggregate, plus head-to-head comparisons."""
    lane_names = list(dict.fromkeys(name for row in rows for name in row["lanes"]))
    lanes = {}
    for name in lane_names:
        entries = [row["lanes"][name] for row in rows if name in row["lanes"]]
        scores = [entry["score"] for entry in entries if entry["score"] is not None]
        lanes[name] = {
            "questions": len(entries),
            "scored": len(scores),
            "mean_score": _mean(scores),
            "ranges": _hit_rate([hit for entry in entries for hit in entry["hits"]]),
            "failures": sum(1 for entry in entries if entry["error"]),
            "cost": round(sum(entry["cost"] or 0 for entry in entries), 2),
        }
    finals = [row["final"] for row in rows if row["final"]]
    final_scores = [final["score"] for final in finals if final["score"] is not None]
    peer = [row["spot_peer_score"] for row in rows if row["spot_peer_score"] is not None]
    names = lane_names + ["final"]
    comparisons = [
        comparison
        for index, first in enumerate(names)
        for second in names[index + 1 :]
        if (comparison := _paired(rows, first, second))
    ]
    return {
        "questions": len(rows),
        "with_record": sum(1 for row in rows if row["final"] is not None),
        "by_type": dict(_count(row["type"] for row in rows)),
        "by_commit": dict(_count(row["commit"] for row in rows if row["commit"])),
        "lanes": lanes,
        "final": {
            "scored": len(final_scores),
            "mean_score": _mean(final_scores),
            "ranges": _hit_rate([hit for final in finals for hit in final["hits"]]),
        },
        "spot_peer": {
            "questions": len(peer),
            "total": round(sum(peer), 2) if peer else None,
            "mean": _mean(peer),
        },
        "comparisons": comparisons,
    }


def _count(values: Any) -> dict:
    counts: dict = defaultdict(int)
    for value in values:
        counts[value] += 1
    return counts


def _show(value: Any) -> str:
    return "-" if value is None else str(value)


def render(summary: dict, rows: list[dict], title: str) -> str:
    """The comparison as a short markdown report."""
    lines = [
        f"# Lane review: {title}",
        "",
        f"{summary['questions']} resolved questions, {summary['with_record']} with a "
        f"cirrus record. Types: {summary['by_type'] or '-'}. "
        f"Bot versions: {summary['by_commit'] or '-'}.",
        "",
        "Score is the baseline score on binary and multiple choice questions "
        "(0 = uniform guess, 100 = perfect). Ranges are continuous questions: how "
        "often the stated range held the answer, against how often it should have.",
        "",
        "| forecaster | scored | mean score | range hit rate (expected) | failures | cost |",
        "|---|---|---|---|---|---|",
    ]

    def ranges(entry: dict | None) -> str:
        if not entry:
            return "-"
        return f"{entry['hit_rate']} ({entry['expected']}), n={entry['forecasts']}"

    for name, lane in summary["lanes"].items():
        lines.append(
            f"| {name} alone | {lane['scored']} | {_show(lane['mean_score'])} | "
            f"{ranges(lane['ranges'])} | {lane['failures']} | ${lane['cost']} |"
        )
    final = summary["final"]
    lines.append(
        f"| submitted aggregate | {final['scored']} | {_show(final['mean_score'])} | "
        f"{ranges(final['ranges'])} | - | - |"
    )
    peer = summary["spot_peer"]
    lines += [
        "",
        f"Official spot peer score of the submitted forecasts: total {_show(peer['total'])}, "
        f"mean {_show(peer['mean'])} over {peer['questions']} questions.",
        "",
        "## Head to head (same questions only)",
        "",
    ]
    if not summary["comparisons"]:
        lines.append("Nothing to compare yet.")
    for c in summary["comparisons"]:
        lines.append(
            f"- {c['first']} vs {c['second']}: {c['first_mean']} vs {c['second_mean']} over "
            f"{c['questions']} questions, difference {c['mean_difference']} "
            f"(standard error {_show(c['standard_error'])}); "
            f"{c['first']} better on {c['first_better']}, {c['second']} on {c['second_better']}."
        )
    scored = [row for row in rows if row["final"] and row["final"]["score"] is not None]
    worst = sorted(scored, key=lambda row: row["final"]["score"])[:10]
    if worst:
        lines += ["", "## Lowest scoring submitted forecasts", ""]
        lines += ["| score | lanes | resolution | question |", "|---|---|---|---|"]
        for row in worst:
            lane_scores = ", ".join(
                f"{name} {_show(lane['score'])}" for name, lane in row["lanes"].items()
            )
            lines.append(
                f"| {row['final']['score']} | {lane_scores} | {row['resolution']} | "
                f"[{(row['title'] or '')[:80]}]({row['url']}) |"
            )
    return "\n".join(lines) + "\n"


# =============================================================================
# FETCHING
# =============================================================================


def fetch_rows(days: int, max_comments: int, seconds_between_requests: float) -> list[dict]:
    """Resolved questions from the last ``days`` days, scored lane by lane."""
    from metaculus_bot_review.client import ReviewClient
    from metaculus_bot_review.outcomes import get_recently_resolved_outcomes

    from cirrus import read_record

    client = ReviewClient(sleep_seconds_between_requests=seconds_between_requests)
    user_id = client.get_current_user_id()
    print(f"reviewing forecasts made by user {user_id}", file=sys.stderr)
    outcomes = [
        outcome
        for outcome in get_recently_resolved_outcomes(days, client)
        if outcome.status == "resolved" and outcome.resolution not in UNSCORED_RESOLUTIONS
    ]
    post_ids = {outcome.post_id for outcome in outcomes}
    records: dict[int, list[dict]] = defaultdict(list)
    for is_private in (True, False):
        comments = client.get_own_comments(
            is_private=is_private, user_id=user_id, max_comments=max_comments
        )
        for comment in comments:
            if comment.on_post not in post_ids:
                continue
            text = client.get_comment(comment.id).text if comment.is_text_archived else comment.text
            record = read_record(text)
            if record:
                records[record["question"]["id"]].append(record)
    rows = []
    for outcome in outcomes:
        scores = outcome.scores
        rows.append(
            score_question(
                {
                    "post_id": outcome.post_id,
                    "question_id": outcome.question_id,
                    "title": outcome.title,
                    "url": outcome.url,
                    "question_type": outcome.question_type,
                    "resolution": outcome.resolution,
                    "spot_peer_score": scores.spot_peer_score if scores else None,
                },
                standing_record(records[outcome.question_id], outcome.spot_scoring_time),
            )
        )
    return rows


def check_access() -> int:
    """Confirm Metaculus accepts our requests. Prints the bot's user id, nothing else."""
    from metaculus_bot_review.client import ReviewClient

    mode = "network-supplied token" if uses_proxy_auth() else "METACULUS_TOKEN"
    try:
        user_id = ReviewClient(sleep_seconds_between_requests=0).get_current_user_id()
    except Exception as error:  # noqa: BLE001 - any failure means no access
        print(f"Metaculus access FAILED ({mode}): {type(error).__name__}: {str(error)[:200]}")
        return 1
    print(f"Metaculus access OK ({mode}): signed in as user {user_id}")
    return 0


def run_bot_review(arguments: list[str]) -> int:
    """
    Run the stock ``bot-review`` command, limited to resolved questions.

    :param arguments: what would follow ``bot-review`` on the command line
    """
    if "--tournament" in arguments or "--post" in arguments:
        print(
            "Refusing: --tournament and --post can list forecasts on open questions, and "
            "tournament rules forbid previewing those. Use --resolved-since.",
            file=sys.stderr,
        )
        return 2

    from metaculus_bot_review import cli
    from metaculus_bot_review.client import ReviewClient
    from metaculus_bot_review.outcomes import get_outcomes_for_posts

    if arguments[:1] == ["show"]:
        post_ids = [argument for argument in arguments[1:] if argument.isdigit()][:1]
        if not post_ids:
            print("show needs a post id", file=sys.stderr)
            return 2
        outcomes = get_outcomes_for_posts(
            [int(post_ids[0])], ReviewClient(sleep_seconds_between_requests=0)
        )
        if not outcomes or any(outcome.status != "resolved" for outcome in outcomes):
            print(f"Refusing: post {post_ids[0]} has not resolved yet.", file=sys.stderr)
            return 2
    sys.argv = ["bot-review", *arguments]
    cli.main()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare cirrus's lanes on resolved questions")
    parser.add_argument("--check", action="store_true", help="only test Metaculus access")
    parser.add_argument("--resolved-since", type=int, default=DEFAULT_DAYS, metavar="DAYS",
                        help="look at questions that resolved in the last DAYS days")
    parser.add_argument("--from-json", metavar="FILE", help="re-render rows saved with --output")
    parser.add_argument("--output", help="write the scored rows and summary to this json file")
    parser.add_argument("--summary", help="write the markdown report to this file")
    parser.add_argument("--max-comments", type=int, default=DEFAULT_MAX_COMMENTS)
    parser.add_argument("--seconds-between-requests", type=float, default=0.7)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["bot-review"]:
        if uses_proxy_auth():
            enable_proxy_auth()
        return run_bot_review(argv[1:])
    args = parse_args(argv)
    if args.from_json:
        with open(args.from_json) as file:
            saved = json.load(file)
        rows, title = saved["rows"], saved["title"]
    else:
        if uses_proxy_auth():
            enable_proxy_auth()
        if args.check:
            return check_access()
        rows = fetch_rows(args.resolved_since, args.max_comments, args.seconds_between_requests)
        title = f"questions resolved in the last {args.resolved_since} days"
    summary = summarize(rows)
    report = render(summary, rows, title)
    print(report)
    if args.output:
        with open(args.output, "w") as file:
            json.dump({"title": title, "summary": summary, "rows": rows}, file, indent=2)
    if args.summary:
        with open(args.summary, "w") as file:
            file.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
