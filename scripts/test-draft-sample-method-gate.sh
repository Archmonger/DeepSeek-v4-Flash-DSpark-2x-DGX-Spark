#!/usr/bin/env bash
# CPU-only gate for the DRAFT_SAMPLE_METHOD -> --speculative-config boundary.
#
# The compose entrypoint interpolates the env value into JSON. Without a gate,
# a crafted value can use JSON escapes or duplicate keys to stay valid JSON and
# silently rewrite known fields (e.g. num_speculative_tokens), so vLLM never
# sees an invalid config. The entrypoint now resolves the value through a shell
# `case` that accepts exactly probabilistic|greedy before the JSON is built.
#
# This test extracts that gate + the SPECULATIVE_CONFIG assignment from
# docker-compose.dspark.yml itself (no copy of the logic, no docker needed) and
# runs it against the full input matrix: valid values must reproduce the exact
# JSON the compose file used to hardcode; everything else must exit nonzero
# without building a config and without executing embedded shell.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE="$ROOT/docker-compose.dspark.yml"
QUIET=0
[ "${1:-}" = "-q" ] && QUIET=1

pass=0
fail=0
say() { [ "$QUIET" = "1" ] || printf '  ok  %s\n' "$*"; }
ok() { pass=$((pass + 1)); say "$*"; }
bad() { fail=$((fail + 1)); printf '  FAIL %s\n' "$*" >&2; }

# Extract the shipped gate + assignment (entrypoint text uses $$ for literal $).
fragment="$(sed -n '/case "\$\${DRAFT_SAMPLE_METHOD/,/^ *SPECULATIVE_CONFIG=/p' "$COMPOSE" | sed 's/\$\$/$/g')"
if [ -z "$fragment" ] || ! printf '%s' "$fragment" | grep -q 'SPECULATIVE_CONFIG='; then
  echo "FAIL could not extract the DRAFT_SAMPLE_METHOD gate from $COMPOSE" >&2
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
printf '%s\n' "$fragment" >"$tmp/fragment.sh"
# Probe appended after the fragment: only reached when the gate accepts.
printf 'printf "%%s" "$SPECULATIVE_CONFIG"\n' >>"$tmp/fragment.sh"

OLD_DEFAULT='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
GREEDY_JSON='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy"}'

run_case() { # $1=mode(unset|set) $2=value-if-set -> stdout; rc in $RC
  RC=0
  if [ "$1" = "unset" ]; then
    OUT="$(env -u DRAFT_SAMPLE_METHOD -u MTP_NUM_TOKENS bash "$tmp/fragment.sh" 2>/dev/null)" || RC=$?
  else
    OUT="$(env MTP_NUM_TOKENS='' DRAFT_SAMPLE_METHOD="$2" bash "$tmp/fragment.sh" 2>/dev/null)" || RC=$?
  fi
}

expect_valid() { # $1=label $2=mode $3=value $4=want-json
  run_case "$2" "$3"
  if [ "$RC" -eq 0 ] && [ "$OUT" = "$4" ]; then
    ok "$1"
  else
    bad "$1: rc=$RC out=$OUT"
  fi
}

expect_reject() { # $1=label $2=value
  run_case set "$2"
  if [ "$RC" -ne 0 ] && [ -z "$OUT" ]; then
    ok "$1"
  else
    bad "$1: rc=$RC out=$OUT (must exit nonzero with no config)"
  fi
}

expect_valid "unset -> exact old hardcoded JSON" unset '' "$OLD_DEFAULT"
expect_valid "empty -> exact old hardcoded JSON" set '' "$OLD_DEFAULT"
expect_valid "explicit probabilistic -> old JSON" set 'probabilistic' "$OLD_DEFAULT"
expect_valid "greedy -> greedy JSON" set 'greedy' "$GREEDY_JSON"

expect_reject "normal invalid value rejected" 'random'
expect_reject "case variant rejected" 'Greedy'
expect_reject "JSON escape alias rejected" 'gree\u0064y'
expect_reject "duplicate-key payload rejected" 'probabilistic","num_speculative_tokens":9999,"draft_sample_method":"greedy'
expect_reject "embedded newline rejected" "$(printf 'probabilistic\ngreedy')"
expect_reject "shell metacharacters rejected" '$(id); `id`; ;&|'

# Injection probe: a command-substitution payload must not execute.
canary="$tmp/executed-canary"
run_case set "\$(touch $canary)"
if [ "$RC" -ne 0 ] && [ ! -e "$canary" ]; then
  ok "command substitution payload rejected and not executed"
else
  bad "command substitution payload: rc=$RC canary_exists=$([ -e "$canary" ] && echo yes || echo no)"
fi

# validate-dspark-config.sh must enforce the same contract before compose runs.
VAL="$ROOT/validate-dspark-config.sh"
if grep -q 'DRAFT_SAMPLE_METHOD must be one of: probabilistic, greedy' "$VAL" \
  && grep -q 'draft_sample_method=\${DRAFT_SAMPLE_METHOD}' "$VAL"; then
  ok "validate-dspark-config.sh enforces the same contract and reports the resolved value"
else
  bad "validate-dspark-config.sh missing the DRAFT_SAMPLE_METHOD contract or resolved-value report"
fi

printf 'RESULT: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
