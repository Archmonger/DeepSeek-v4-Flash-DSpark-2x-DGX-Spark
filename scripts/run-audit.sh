#!/usr/bin/env bash
# run-audit.sh — full DS4 DSpark serving audit (methodology documented in scripts/EVAL.md)
#
# Runs, against a LIVE endpoint:
#   Phase 1  Throughput matrix  — scripts/bench-miaai.py (MiaAI 08-14 methodology)
#   Phase 2  Spec-decode health — scripts/spec-acceptance.py (acceptance rate)
#   Phase 3  RULER-lite quality — scripts/ruler-lite.py (retrieval/tracing/aggregation at depth)
#   Phase 4  Tool calling       — scripts/tool-battery.py (incl. issue55 truncation at depth)
#   Phase 5  Garble sweep       — scripts/context-garble-sweep.py (cold prefill, tokenize-verified)
#
# Usage: bash scripts/run-audit.sh [--base-url http://127.0.0.1:8888/v1] [--model deepseek-v4-flash-0731]
#        [--lengths 8192,32768,131072,262144] [--tool-lengths 32768,131072] [--garble-lengths 2048,32768,131072]
# Exit 0 = all phases pass, 1 = any failure.
set -u
BASE_URL="${BASE_URL:-http://127.0.0.1:8888/v1}"
MODEL="${MODEL:-deepseek-v4-flash-0731}"
LENGTHS="${LENGTHS:-8192,32768,131072,262144}"
TOOL_LENGTHS="${TOOL_LENGTHS:-32768,131072}"
GARBLE_LENGTHS="${GARBLE_LENGTHS:-2048,32768,131072}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP=$(date +%Y%m%d-%H%M%S)
REPORT_DIR="results/audit-${STAMP}"
mkdir -p "$REPORT_DIR"
echo "=== DS4 audit $STAMP | $BASE_URL | $MODEL ==="

fail=0
run_phase() {
  local name="$1"; shift
  echo; echo "########## Phase: $name ##########"
  if "$@"; then echo "## $name: PASS"; else echo "## $name: FAIL"; fail=1; fi
}

run_phase "1 throughput" python3 "$SCRIPT_DIR/bench-miaai.py" --base-url "$BASE_URL" --model "$MODEL" \
  --prompt 256 --concurrency 1 --repeat 5 | tee "$REPORT_DIR/throughput.log"

run_phase "2 spec-acceptance" python3 "$SCRIPT_DIR/spec-acceptance.py" --base-url "$BASE_URL" \
  --model "$MODEL" --trials 5 --bench-script "$SCRIPT_DIR/bench-miaai.py" | tee "$REPORT_DIR/acceptance.log"

run_phase "3 ruler-lite quality" python3 "$SCRIPT_DIR/ruler-lite.py" --base-url "$BASE_URL" \
  --model "$MODEL" --lengths "$LENGTHS" --output "$REPORT_DIR/ruler-lite.json" | tee "$REPORT_DIR/ruler.log"

run_phase "4 tool battery" python3 "$SCRIPT_DIR/tool-battery.py" "$BASE_URL/chat/completions" "$MODEL" | tee "$REPORT_DIR/tool.log"

if command -v python3 >/dev/null; then
  run_phase "5 deep-context tool" python3 "$SCRIPT_DIR/deepctx-tool-battery.py" "$BASE_URL/chat/completions" "$MODEL" "$TOOL_LENGTHS" | tee "$REPORT_DIR/deeptool.log"
fi

run_phase "6 garble sweep" python3 "$SCRIPT_DIR/context-garble-sweep.py" --url "$BASE_URL" \
  --model "$MODEL" --lengths "$GARBLE_LENGTHS" --runs 1 --out "$REPORT_DIR/garble.md" | tee "$REPORT_DIR/garble.log"

echo; echo "=== AUDIT COMPLETE: $([ $fail -eq 0 ] && echo ALL-PASS || echo FAILURES) — reports in $REPORT_DIR ==="
exit $fail
