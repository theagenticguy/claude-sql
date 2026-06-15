#!/usr/bin/env bash
# Non-blocking docs-staleness hook — fires after codehub auto-reindex.
# When .codehub/docs/.docmeta.json exists and the graph_hash in the
# manifest disagrees with the live hash, emit a systemMessage suggesting
# /codehub-document --refresh. Never regenerates automatically —
# regeneration spends LLM credits and requires consent.

set -uo pipefail

# Only fire for git mutations we just auto-reindexed on.
if ! echo "${CLAUDE_TOOL_INPUT:-}" | grep -qE 'git (commit|merge|rebase|pull)'; then
  exit 0
fi

DOCMETA=".codehub/docs/.docmeta.json"
if [ ! -f "$DOCMETA" ]; then
  exit 0
fi

# Extract manifest hash. jq is a soft dependency; fall back to grep.
if command -v jq >/dev/null 2>&1; then
  MANIFEST_HASH=$(jq -r '.codehub_graph_hash // empty' "$DOCMETA" 2>/dev/null || true)
else
  MANIFEST_HASH=$(grep -o '"codehub_graph_hash":[[:space:]]*"[^"]*"' "$DOCMETA" | head -1 | sed 's/.*"codehub_graph_hash":[[:space:]]*"//;s/"$//')
fi

if [ -z "${MANIFEST_HASH:-}" ]; then
  exit 0
fi

# Live hash via the CLI. Keep timeout short so the hook never blocks the user.
LIVE_HASH=$(timeout 3 codehub status --format=hash 2>/dev/null | head -1 || true)

if [ -z "${LIVE_HASH:-}" ]; then
  # CLI not available or timed out; emit nothing.
  exit 0
fi

if [ "$MANIFEST_HASH" != "$LIVE_HASH" ]; then
  # systemMessage format: this text is surfaced to Claude, not the user shell.
  # Non-blocking — just a hint.
  printf '{"systemMessage":"Docs at .codehub/docs/ may be stale (graph_hash changed). Run /codehub-document --refresh when convenient."}\n'
fi

exit 0
