#!/usr/bin/env bash
# Forecast on new tournament questions every POLL_MINUTES for about
# LOOP_MINUTES. Used by run_bot_on_tournament.yaml; the comment at the top of
# that file explains why the bot loops instead of relying on the schedule.
#
# Environment: POLL_MINUTES, LOOP_MINUTES, ATTEMPT_TIMEOUT_MINUTES,
# WORKFLOW_FILE, RUN_URL, GH_TOKEN (to read the workflow's state), plus the
# keys cirrus.py needs. Exits 1 at the end if any attempt failed.
set -uo pipefail

here=$(dirname "$0")
poll_seconds=$(( POLL_MINUTES * 60 ))
deadline=$(( $(date +%s) + LOOP_MINUTES * 60 ))
attempts=0
failed=0
streak=0

workflow_enabled() {
  # Disabling the workflow in the Actions tab is how the bot is paused. If the
  # check itself fails (a GitHub hiccup), keep forecasting.
  local state
  state=$(gh api "repos/$GITHUB_REPOSITORY/actions/workflows/$WORKFLOW_FILE" --jq .state 2> /dev/null) || return 0
  [ "$state" = "active" ]
}

while :; do
  if ! workflow_enabled; then
    echo "The workflow has been disabled; stopping here."
    break
  fi

  started=$(date +%s)
  attempts=$(( attempts + 1 ))
  echo "::group::Attempt $attempts ($(date -u +%H:%M) UTC)"
  timeout --kill-after=60 "${ATTEMPT_TIMEOUT_MINUTES}m" uv run python cirrus.py --mode tournament
  status=$?
  echo "::endgroup::"

  if [ "$status" -ne 0 ]; then
    failed=$(( failed + 1 ))
    streak=$(( streak + 1 ))
    echo "::warning::Attempt $attempts failed with exit code $status."
    # Report the first failure of a streak only, so an outage that lasts hours
    # does not post every POLL_MINUTES.
    if [ "$streak" -eq 1 ]; then
      bash "$here/notify_discord.sh" "cirrus: a forecasting attempt failed (exit code $status). It tries again every $POLL_MINUTES minutes; further failures in a row are not reported. $RUN_URL"
    fi
  else
    if [ "$streak" -gt 0 ]; then
      bash "$here/notify_discord.sh" "cirrus: forecasting works again after $streak failed attempt(s). $RUN_URL"
    fi
    streak=0
  fi

  # Never start an attempt after the deadline, so that the last one finishes
  # well inside the job's time limit.
  next=$(( started + poll_seconds ))
  now=$(date +%s)
  if [ "$next" -ge "$deadline" ] || [ "$now" -ge "$deadline" ]; then
    break
  fi
  if [ "$next" -gt "$now" ]; then
    sleep $(( next - now ))
  fi
done

echo "$attempts attempt(s), $failed failed."
[ "$failed" -eq 0 ]
