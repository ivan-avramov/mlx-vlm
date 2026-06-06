#!/usr/bin/env bash
# Multi-turn curl test for per-chat PromptCacheState + DeltaNet snapshot rewind.
#
# Prereqs:
#   - Server running with --deltanet-rewind=auto --cache-session-max=8 (defaults)
#   - Model loaded (e.g., unsloth/Qwen3.6-27B-UD-MLX-6bit)
#   - Run with --log-level=DEBUG to observe rewind / cache reuse messages
#
# e.g.:
# uv run python -u -m mlx_vlm.server \
#    --model unsloth/Qwen3.6-27B-UD-MLX-6bit \
#    --host 127.0.0.1 --port 8001 \
#    --log-level DEBUG \
#    --cache-session-max 8 \
#    --deltanet-rewind auto \
#    --deltanet-ring-size 3
#
# Usage:
#   ./dev/test_session_cache_curl.sh [PORT] [MODEL]
#
# Watch the server log for:
#   - "chat_completions chat_id=test-XXX session_state=fresh|found"
#   - "DeltaNet snapshot rewind: restored from offset N, replaying M tokens"
#   - "Hybrid-Cache Rewind Guard: no snapshot available ..."

set -euo pipefail

PORT="${1:-8001}"
MODEL="${2:-unsloth/Qwen3.6-27B-UD-MLX-6bit}"
URL="http://127.0.0.1:${PORT}/v1/chat/completions"
CHAT_ID="test-rewind-$(date +%s)"

echo "Testing against ${URL}"
echo "Chat ID: ${CHAT_ID}"
echo

req() {
  local label="$1"; shift
  local body="$1"; shift
  echo "=== ${label} ==="
  curl -sS -N "${URL}" \
    -H "Content-Type: application/json" \
    -H "X-MLX-VLM-Chat-Id: ${CHAT_ID}" \
    -d "${body}" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
choices = data.get('choices', [])
if choices:
    msg = choices[0].get('message', {})
    print('Response:', (msg.get('content') or '').strip()[:200])
    print('finish_reason:', choices[0].get('finish_reason'))
usage = data.get('usage', {})
print('prompt_tokens:', usage.get('prompt_tokens'),
      'completion_tokens:', usage.get('completion_tokens'))
"
  echo
}

# Streaming variant: collects SSE deltas, prints concatenated content and
# the finish_reason from the final non-[DONE] chunk. Verifies the cached
# path's finish_reason plumbing reaches the wire.
req_stream() {
  local label="$1"; shift
  local body="$1"; shift
  echo "=== ${label} (streaming) ==="
  curl -sS -N "${URL}" \
    -H "Content-Type: application/json" \
    -H "Accept: text/event-stream" \
    -H "X-MLX-VLM-Chat-Id: ${CHAT_ID}" \
    -d "${body}" \
  | python3 -c "
import json, sys
text_parts = []
last_finish = None
chunk_count = 0
for raw in sys.stdin:
    raw = raw.strip()
    if not raw.startswith('data:'):
        continue
    payload = raw[len('data:'):].strip()
    if payload == '[DONE]':
        break
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        continue
    chunk_count += 1
    for ch in chunk.get('choices', []):
        delta = ch.get('delta', {})
        if 'content' in delta and delta['content']:
            text_parts.append(delta['content'])
        if ch.get('finish_reason'):
            last_finish = ch['finish_reason']
print('Chunks received:', chunk_count)
print('Response:', ''.join(text_parts).strip()[:200])
print('finish_reason (final SSE chunk):', last_finish)
"
  echo
}

# Turn 1: fresh prefill (no prior cache). Should emit session_state=fresh.
req "Turn 1 (fresh)" '{
  "model": "'"${MODEL}"'",
  "messages": [{"role":"user","content":"What is 2+2?"}],
  "stream": false,
  "max_tokens": 50,
  "temperature": 0.0
}'

# Turn 2: continue conversation. session_state=found, should hit a snapshot
# at end-of-turn-1. Watch for "DeltaNet snapshot rewind: restored from..."
req "Turn 2 (cache reuse expected)" '{
  "model": "'"${MODEL}"'",
  "messages": [
    {"role":"user","content":"What is 2+2?"},
    {"role":"assistant","content":"4"},
    {"role":"user","content":"Now what is 3+3?"}
  ],
  "stream": false,
  "max_tokens": 50,
  "temperature": 0.0
}'

# Turn 3a / 3b / 3c: Diagnostic — repeat Turn 2's prompt at temp=0.0 three
# times. By Turn 3 the snapshot ring holds two entries (post-turn-1 at
# offset ~27, post-turn-2 at offset ~40). find_nearest(prefix_len=38) picks
# the offset-27 snapshot and replays 11 tokens. With temp=0.0 sampling is
# deterministic — every run should produce the SAME answer, and that answer
# should match Turn 2's fresh-prefill ground truth ("6"). If we see "6"
# consistently → snapshot restore is correct. If we see a different but
# stable answer → state-restore bug. If answers vary → sampling is leaking
# in somehow (shouldn't with temp=0.0).
for i in a b c; do
  req "Turn 3${i} (snapshot rewind, temp=0.0 diagnostic)" '{
    "model": "'"${MODEL}"'",
    "messages": [
      {"role":"user","content":"What is 2+2?"},
      {"role":"assistant","content":"4"},
      {"role":"user","content":"Now what is 3+3?"}
    ],
    "stream": false,
    "max_tokens": 50,
    "temperature": 0.0
  }'
done

# Turn 4: edit the very first user message. prefix_len drops to ~0 →
# fall through to fresh re-prefill. Watch for "no snapshot available".
req "Turn 4 (edit Turn 1, fresh re-prefill expected)" '{
  "model": "'"${MODEL}"'",
  "messages": [{"role":"user","content":"What is 5+5?"}],
  "stream": false,
  "max_tokens": 50,
  "temperature": 0.0
}'

# Turn 5: streaming variant of a multi-turn cached request. Verifies the
# cached path's finish_reason plumbing reaches the SSE wire — the final
# data chunk before [DONE] should carry finish_reason="stop" or "length".
# Also exercises the streaming endpoint loop's per-token text accumulation.
req_stream "Turn 5 (streaming multi-turn)" '{
  "model": "'"${MODEL}"'",
  "messages": [
    {"role":"user","content":"What is 5+5?"},
    {"role":"assistant","content":"10"},
    {"role":"user","content":"What is 7+7?"}
  ],
  "stream": true,
  "max_tokens": 50,
  "temperature": 0.0
}'

# ===========================================================================
# Restore-path coverage (separate chat_id to avoid contaminating earlier turns)
# ===========================================================================
#
# Goal: explicitly exercise the snapshot-restore-and-replay path
# ("DeltaNet snapshot rewind: restored from offset N, replaying M tokens").
#
# The earlier turns can't hit this path post-fix because the test simulates an
# edited assistant message ("4" in place of "2 + 2 = 4"), which triggers the
# snapshot ring's stale-entry drop in PromptCacheState.update(). To force
# restore, we need the live cache's prior turn to extend verbatim — i.e.
# the prior assistant content sent in turn N+1 must be byte-equal to what
# turn N actually generated.
#
# Strategy:
#   R1 — fresh prefill of "What is 2+2?". Capture the verbatim assistant content.
#   R2 — multi-turn extension where prior assistant = R1's exact content.
#        find_prefix_length should reach end-of-R1 (no rewind, just extend).
#        Snapshot ring after R2: [snap@end_of_R1, snap@end_of_R2].
#   R3 — REGENERATE R2 (identical body). cached_len > prompt_len → rewind to
#        prompt_len. find_nearest(prompt_len) returns snap@end_of_R1. The
#        restore-and-replay path fires, replaying tokens in [snap.offset,
#        prompt_len). At temp=0.0, R3's content MUST equal R2's content.
#
# Caveat: the assistant template's auto-injected think-block prefix
# (e.g. "<think>\n\n</think>\n\n" on Qwen 3.6) only renders for the LAST
# assistant turn, not for prior assistants in the conversation history.
# That asymmetry by itself doesn't break the test — what matters is that
# the chat template renders R2's prior-assistant turn the same way both
# when R2 first runs (extending R1) and when R3 regenerates it. We send
# enable_thinking=false on every request to suppress think-block injection
# entirely, keeping the rendered prompt fully deterministic across runs.
# ===========================================================================

RESTORE_CHAT_ID="${CHAT_ID}-restore"
echo "=== Restore-path test (chat_id=${RESTORE_CHAT_ID}) ==="

# Like req() but prints ONLY the assistant content, suitable for $(...).
req_content() {
  local body="$1"; shift
  local cid="$1"; shift
  curl -sS -N "${URL}" \
    -H "Content-Type: application/json" \
    -H "X-MLX-VLM-Chat-Id: ${cid}" \
    -d "${body}" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
choices = data.get('choices', [])
if choices:
    msg = choices[0].get('message', {})
    sys.stdout.write((msg.get('content') or '').strip())
"
}

# R1: fresh prefill. Capture verbatim output.
ANS_R1="$(req_content '{
  "model": "'"${MODEL}"'",
  "messages": [{"role":"user","content":"What is 2+2?"}],
  "stream": false,
  "max_tokens": 50,
  "temperature": 0.0,
  "enable_thinking": false
}' "${RESTORE_CHAT_ID}")"
echo "R1 response: ${ANS_R1}"

# R2: extends R1 verbatim. Build body via Python so ANS_R1 is JSON-escaped
# safely (handles quotes, newlines, etc.).
R2_BODY=$(MODEL="${MODEL}" ANS="${ANS_R1}" python3 -c '
import json, os
print(json.dumps({
    "model": os.environ["MODEL"],
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": os.environ["ANS"]},
        {"role": "user", "content": "Now what is 3+3?"},
    ],
    "stream": False,
    "max_tokens": 50,
    "temperature": 0.0,
    "enable_thinking": False,
}))
')

ANS_R2="$(req_content "${R2_BODY}" "${RESTORE_CHAT_ID}")"
echo "R2 response: ${ANS_R2}"

# R3: regenerate R2 — identical body, identical chat_id. find_prefix_length
# reaches len(R2_prompt); cache holds len(R2_prompt) + len(R2_generated).
# Rewind needed; find_nearest returns snap@end_of_R1 → restore + replay.
ANS_R3="$(req_content "${R2_BODY}" "${RESTORE_CHAT_ID}")"
echo "R3 response: ${ANS_R3}"

if [[ -n "${ANS_R2}" && "${ANS_R2}" == "${ANS_R3}" ]]; then
  echo "RESTORE CHECK: PASS (R3 == R2 byte-for-byte)"
  echo "  Confirm in server log: 'DeltaNet snapshot rewind: restored from offset N, replaying M tokens' on R3."
  echo "  If you see 'no snapshot available for rewind to N (ring=0/1, ...)' instead, the restore"
  echo "  path didn't fire — likely the chat template's prior-assistant rendering diverged from"
  echo "  what R1 cached, and the post-fix divergence drop invalidated the snapshot during R2's update."
else
  echo "RESTORE CHECK: FAIL or path-not-fired"
  echo "  R2='${ANS_R2}'"
  echo "  R3='${ANS_R3}'"
  echo "  Investigation: check the server log for R3."
  echo "    - If you see 'restored from offset N' AND outputs differ → state-restore bug."
  echo "    - If you see 'no snapshot available' → divergence dropped the snapshot, see"
  echo "      'Snapshot ring: dropped N stale snapshot(s)' on R2's update."
fi
echo

# Same chat_id, fresh content — verify session is still keyed correctly
echo "=== Done ==="
echo "Chat IDs used:"
echo "  Main test:    ${CHAT_ID}"
echo "  Restore test: ${RESTORE_CHAT_ID}"
echo
echo "Look for these debug log lines on the server:"
echo "  chat_completions chat_id=${CHAT_ID} session_state=..."
echo "  Snapshot ring: dropped N stale snapshot(s) past divergence at offset M"
echo "    (Turn 2 in the main test should drop snap@end_of_Turn1 because the"
echo "    assistant message is edited '4' instead of generated '2 + 2 = 4'.)"
echo "  DeltaNet snapshot rewind: restored from offset N, replaying M tokens"
echo "    (Restore test R3 should fire this if templates align across turns.)"
echo "  Hybrid-Cache Rewind Guard: no snapshot available ..."
echo
echo "Diagnostic checks:"
echo "  - Turns 3a/3b/3c at temp=0.0 should produce IDENTICAL responses."
echo "    Post-fix expectation: all three say '6' (matches Turn 2's ground truth)."
echo "    Pre-fix they said '3' deterministically — that was the stale-snapshot bug."
echo "  - Turn 5 streaming should print 'finish_reason (final SSE chunk): stop'"
echo "    and a non-zero chunk count."
echo "  - Restore test (R2 vs R3) should print 'RESTORE CHECK: PASS' and the"
echo "    server log should show 'DeltaNet snapshot rewind: restored ...' on R3."
