#!/usr/bin/env bash
# Post one message to the Discord webhook, if one is configured.
# Usage: notify_discord.sh "message". Never fails the step that calls it.
set -u

message=${1:?usage: notify_discord.sh MESSAGE}
if [ -z "${DISCORD_WEBHOOK_URL:-}" ]; then
  echo "DISCORD_WEBHOOK_URL is not set; skipping the notification."
  exit 0
fi
payload=$(jq -n --arg content "$message" '{content: $content}')
# curl's own error text never contains the URL, so the webhook stays secret.
if ! curl -fsS -H "Content-Type: application/json" -d "$payload" "$DISCORD_WEBHOOK_URL" > /dev/null; then
  echo "::warning::The Discord notification could not be sent."
fi
exit 0
