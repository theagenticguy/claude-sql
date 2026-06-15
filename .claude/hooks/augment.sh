#!/usr/bin/env bash
# augment.sh — Claude Code PreToolUse hook for Bash/Grep/Glob.
#
# Claude Code hands the tool invocation to this script on stdin as JSON:
#   { "tool_name": "Grep",
#     "tool_input": { "pattern": "..." | "command": "rg ..." } }
#
# We extract the first plausible search token, forward it to `codehub augment`,
# and stream that command's stderr back to Claude Code as additional context
# via the PreToolUse hookSpecificOutput envelope on stdout.
#
# Contract:
#   - Exit 0 regardless of what happens — hook failures must never block the
#     underlying tool call.
#   - Patterns shorter than 3 chars are skipped (too noisy to enrich).
#   - We rely on `python3` for JSON parsing so we don't take a dep on `jq`.

set -u

HOOK_INPUT=""
if [ -t 0 ]; then
  # No stdin — the legacy `CLAUDE_TOOL_INPUT` env var is our fallback.
  HOOK_INPUT="${CLAUDE_TOOL_INPUT:-}"
else
  # Non-interactive stdin: Claude Code always sends a JSON payload.
  HOOK_INPUT="$(cat)"
fi

if [ -z "$HOOK_INPUT" ]; then
  exit 0
fi

# Discover the codehub binary. Prefer an on-PATH install so repeated hook
# invocations avoid the `npx` spawn cost.
CODEHUB_BIN=""
if command -v codehub >/dev/null 2>&1; then
  CODEHUB_BIN="codehub"
elif command -v npx >/dev/null 2>&1; then
  CODEHUB_BIN="npx --yes @opencodehub/cli"
else
  exit 0
fi

# Parse tool_name + pattern using python3. Falls through to exit 0 if the
# interpreter is missing or the payload isn't valid JSON.
if ! command -v python3 >/dev/null 2>&1; then
  exit 0
fi

PATTERN="$(printf '%s' "$HOOK_INPUT" | python3 - <<'PY' 2>/dev/null
import json, sys, re

try:
    data = json.loads(sys.stdin.read() or "{}")
except Exception:
    sys.exit(0)

tool = (data.get("tool_name") or "").strip()
tool_input = data.get("tool_input") or {}
pattern = ""

if tool == "Grep":
    pattern = (tool_input.get("pattern") or "").strip()
elif tool == "Glob":
    raw = (tool_input.get("pattern") or "").strip()
    m = re.search(r"[*\\/]([a-zA-Z][a-zA-Z0-9_-]{2,})", raw)
    if m:
        pattern = m.group(1)
elif tool == "Bash":
    cmd = (tool_input.get("command") or "").strip()
    if re.search(r"\b(rg|grep|ag|ack)\b", cmd):
        tokens = cmd.split()
        skip_next = False
        flags_with_values = {
            "-e", "-f", "-m", "-A", "-B", "-C", "-g", "--glob",
            "-t", "--type", "--include", "--exclude",
        }
        found_cmd = False
        for tok in tokens:
            if skip_next:
                skip_next = False
                continue
            if not found_cmd:
                if re.search(r"(^|/)(rg|grep|ag|ack)$", tok):
                    found_cmd = True
                continue
            if tok.startswith("-"):
                if tok in flags_with_values:
                    skip_next = True
                continue
            cleaned = tok.strip("'\"")
            if len(cleaned) >= 3:
                pattern = cleaned
                break

pattern = re.sub(r"[^A-Za-z0-9_./-]", "", pattern)
if len(pattern) >= 3:
    print(pattern)
PY
)"

if [ -z "$PATTERN" ]; then
  exit 0
fi

# Run the augment command with a tight timeout. We capture stderr (the
# enriched text block) and discard stdout to keep the hook output channel
# dedicated to the PreToolUse envelope we emit below.
AUGMENT_OUTPUT="$( $CODEHUB_BIN augment -- "$PATTERN" 2>&1 >/dev/null || true )"

if [ -z "$AUGMENT_OUTPUT" ]; then
  exit 0
fi

# Emit the PreToolUse additionalContext envelope on stdout so Claude Code
# injects it into the model's context before the Grep/Glob/Bash call runs.
python3 - "$AUGMENT_OUTPUT" <<'PY' 2>/dev/null
import json, sys
text = sys.argv[1]
print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "additionalContext": text,
    }
}))
PY

exit 0
