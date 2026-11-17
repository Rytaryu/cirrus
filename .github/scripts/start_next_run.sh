#!/usr/bin/env bash
# Start the next run of the tournament workflow, so that forecasting goes on
# without relying on GitHub's schedule. Used by run_bot_on_tournament.yaml.
#
# Environment: WORKFLOW_FILE, GH_TOKEN (needs actions: write).
set -uo pipefail

state=$(gh api "repos/$GITHUB_REPOSITORY/actions/workflows/$WORKFLOW_FILE" --jq .state 2> /dev/null) || state=unknown
if [ "$state" != "active" ] && [ "$state" != "unknown" ]; then
  echo "The workflow is $state (paused); not starting another run."
  exit 0
fi

# The next run always uses the default branch, even if this one was started
# by hand from another branch.
branch=$(gh api "repos/$GITHUB_REPOSITORY" --jq .default_branch 2> /dev/null) || branch=main

for try in 1 2 3; do
  if gh workflow run "$WORKFLOW_FILE" --repo "$GITHUB_REPOSITORY" --ref "$branch"; then
    echo "Started the next run. It waits until this one has finished."
    exit 0
  fi
  sleep $(( try * 20 ))
done

echo "::error::Could not start the next run. The backup schedule will restart forecasting."
exit 1
