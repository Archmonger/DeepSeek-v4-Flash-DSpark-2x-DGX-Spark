#!/usr/bin/env bash
# test-boot-shape-warmup.sh — CPU behavioral tests for scripts/boot-shape-warmup.sh.
#
# Uses the WARMUP_CURL seam to substitute a recording curl stub; every
# assertion is behavioral (exit codes, recorded request bodies/headers across
# /v1/models, /tokenize, /v1/completions and /v1/chat/completions, outcome
# tallies, stdout bucket-ladder lines). No GPU, no network.
# Run: bash scripts/test-boot-shape-warmup.sh [-q]
set -u

QUIET="${1:-}"
here="$(cd "$(dirname "$0")" && pwd)"
target="$here/boot-shape-warmup.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

pass=0 fail=0
ok()  { pass=$((pass + 1)); [ "$QUIET" = "-q" ] || printf '  ok  %s\n' "$*"; }
bad() { fail=$((fail + 1)); printf '  FAIL %s\n' "$*" >&2; }

# Recording curl stub.
# STUB_MODE: ok | chatfail | probefail | killparent | tokfail45 | badcount45.
cat > "$work/curl-stub" <<'STUB'
#!/usr/bin/env bash
rec="${STUB_REC:?}"
n=$(date +%s%N)-$$-$RANDOM
is_probe=0; is_chat=0; is_tok=0; is_comp=0; body=""; hdrs=""
prev=""
for a in "$@"; do
  case "$a" in
    */v1/models*) is_probe=1 ;;
    */v1/chat/completions*) is_chat=1 ;;
    */tokenize*) is_tok=1 ;;
    */v1/completions*) is_comp=1 ;;
  esac
  [ "$prev" = "-d" ] && body="$a"
  [ "$prev" = "-H" ] && hdrs="${hdrs}${a}"$'\n'
  prev="$a"
done
if [ "$is_probe" = 1 ]; then
  [ "${STUB_MODE:-ok}" = "probefail" ] && exit 22
  printf '%s\n' "$hdrs" > "$rec/probe-$n"
  exit 0
fi
if [ "$is_tok" = 1 ]; then
  printf '%s%s\n' "$hdrs" "$body" > "$rec/tok-$n"
  p=$(printf '%s' "$body" | sed -n 's/.*"prompt":"\([^"]*\)".*/\1/p')
  cnt=$(printf '%s' "$p" | wc -w | tr -d ' ')
  [ "${STUB_MODE:-ok}" = "tokfail45" ] && [ "$cnt" = 45 ] && exit 22
  [ "${STUB_MODE:-ok}" = "badcount45" ] && [ "$cnt" = 45 ] && cnt=$((cnt + 1))
  printf '{"tokens":[],"count":%s}\n' "$cnt"
  exit 0
fi
if [ "$is_comp" = 1 ]; then
  printf '%s%s\n' "$hdrs" "$body" > "$rec/comp-$n"
  exit 0
fi
# killparent mode: the stub's PPID is the fire() subshell running this curl;
# killing it simulates a request whose outcome is never written, leaving the
# pre-created result file empty for the tally.
case "$body" in *" c4-3]"*)
  [ "${STUB_MODE:-ok}" = "killparent" ] && { kill -9 "$PPID" 2>/dev/null; exit 137; } ;;
esac
printf '%s%s\n' "$hdrs" "$body" > "$rec/chat-$n"
[ "${STUB_MODE:-ok}" = "chatfail" ] && exit 22
exit 0
STUB
chmod +x "$work/curl-stub"

run_sweep() { # $1 = mode, $2 = record dir, rest: extra KEY=VALUE env; returns script exit code
  local mode=$1 dir=$2
  shift 2
  mkdir -p "$dir"
  env STUB_MODE="$mode" STUB_REC="$dir" WARMUP_CURL="$work/curl-stub" "$@" \
    bash "$target" http://stub:0 test-model > "$dir/stdout" 2> "$dir/stderr"
}

comp_prompt() { # $1 = recorded comp file -> echoed plain prompt text
  sed -n 's/.*"prompt":"\([^"]*\)".*/\1/p' "$1"
}

file_ts() { # $1 = recorded file path -> leading nanosecond timestamp of its name
  local b=${1##*/}
  b=${b#*-}          # strip probe-/chat-/tok-/comp- prefix
  printf '%s' "${b%%-*}"
}

# ---- 1. all-success run: exit 0; 10 chats + 6 completions + 6 tokenizes ------
rec="$work/r1"
if run_sweep ok "$rec"; then ok "all-success sweep exits 0"; else bad "all-success sweep exited nonzero"; fi

chat_n=$(ls "$rec"/chat-* 2>/dev/null | wc -l)
[ "$chat_n" -eq 10 ] && ok "10 chat requests fired" || bad "expected 10 chat requests, got $chat_n"
comp_n=$(ls "$rec"/comp-* 2>/dev/null | wc -l)
[ "$comp_n" -eq 6 ] && ok "6 ladder completions fired" || bad "expected 6 ladder completions, got $comp_n"
tok_n=$(ls "$rec"/tok-* 2>/dev/null | wc -l)
[ "$tok_n" -eq 6 ] && ok "6 tokenize verifications performed" || bad "expected 6 tokenize calls, got $tok_n"

for tag in c1-1 c2-1 c2-2 c4-1 c4-2 c4-3 c4-4 mid-1 longchunk-1; do
  grep -lq "warmup .* ${tag}\]" "$rec"/chat-* || bad "arm ${tag} missing from recorded bodies"
done
grep -lq 'warmup .* nothink-1\]' "$rec"/chat-* && ok "all 10 arm tags present" \
  || bad "nothink arm missing"

# ---- 2. thinking-off arm is exactly one --------------------------------------
tf=$(grep -l '"thinking":false' "$rec"/chat-* | wc -l)
[ "$tf" -eq 1 ] && ok "exactly one thinking:false arm" || bad "thinking:false arms: $tf (want 1)"
tt=$(grep -l '"thinking":true' "$rec"/chat-* | wc -l)
[ "$tt" -eq 9 ] && ok "9 thinking:true arms" || bad "thinking:true arms: $tt (want 9)"

# ---- 3. long arm actually crosses the 8192 chunk boundary --------------------
longlen=$(wc -c < "$(grep -l 'longchunk-1\]' "$rec"/chat-*)")
[ "$longlen" -gt 40000 ] && ok "longchunk body is ${longlen} bytes (> one 8192-token chunk)" \
  || bad "longchunk body only ${longlen} bytes"

# ---- 4. prefix-cache busting: per-run nonce differs, per-request tag differs -
rec2="$work/r2"
run_sweep ok "$rec2" >/dev/null 2>&1 || true
n1=$(grep -oh 'warmup [^ ]*' "$rec"/chat-*  | sort -u | head -1)
n2=$(grep -oh 'warmup [^ ]*' "$rec2"/chat-* | sort -u | head -1)
[ -n "$n1" ] && [ "$n1" != "$n2" ] && ok "nonce differs across runs ($n1 vs $n2)" \
  || bad "nonce identical across runs — prefix cache would skip the warmed prefill"
uniq_tags=$(grep -oh 'warmup [^]]*\]' "$rec"/chat-* | sort -u | wc -l)
[ "$uniq_tags" -eq 10 ] && ok "10 distinct chat tags within a run" || bad "distinct chat tags: $uniq_tags (want 10)"

# ---- 5. bucket ladder: exact declared token counts, all six live BLOCK keys --
counts=$(for f in "$rec"/comp-*; do comp_prompt "$f" | wc -w; done | sort -un | tr '\n' ' ')
[ "$counts" = "1 6 20 45 100 200 " ] \
  && ok "completions requested at exactly the six declared token counts" \
  || bad "completion prompt counts wrong: [$counts] want [1 6 20 45 100 200]"
for b in 8 16 32 64 128 256; do
  grep -q "BLOCK ${b} fired" "$rec/stdout" \
    && ok "BLOCK ${b} rung mapped and fired" \
    || bad "no stdout evidence that BLOCK ${b} was mapped/fired"
done
if grep -rqE 'arm c6| c6-[0-9]' "$rec" 2>/dev/null; then
  bad "C=6 arm still present somewhere in requests/logs"
else
  ok "no C=6 arm remains (deployed MAX_NUM_SEQS=4)"
fi

# ---- 6. tokenize gate: verified, authenticated, BEFORE its completion --------
for s in 1 6 20 45 100 200; do
  p="hello"; i=1
  while [ "$i" -lt "$s" ]; do p="$p hello"; i=$((i + 1)); done
  tfile=$(grep -lF "\"prompt\":\"$p\"" "$rec"/tok-* 2>/dev/null | head -1)
  cfile=$(grep -lF "\"prompt\":\"$p\"" "$rec"/comp-* 2>/dev/null | head -1)
  if [ -n "$tfile" ] && [ -n "$cfile" ] && [ "$(file_ts "$tfile")" -lt "$(file_ts "$cfile")" ]; then
    ok "s=${s}: /tokenize verified before its completion fired"
  else
    bad "s=${s}: tokenize-before-completion ordering violated or records missing"
  fi
done

# ---- 7. auth propagation: launcher bearer preferred over ambient VLLM_API_KEY -
rec5="$work/r5"
run_sweep ok "$rec5" DSPARK_WARMUP_BEARER=warmup-first-key VLLM_API_KEY=ambient-vllm-key >/dev/null 2>&1 \
  || bad "bearer-preference sweep exited nonzero"
missing_auth=0 req_n=0
for f in "$rec5"/chat-* "$rec5"/comp-* "$rec5"/tok-* "$rec5"/probe-*; do
  [ -f "$f" ] || continue
  req_n=$((req_n + 1))
  [ "$(head -n 1 "$f")" = "Authorization: Bearer warmup-first-key" ] || missing_auth=$((missing_auth + 1))
done
[ "$req_n" -eq 23 ] && ok "probe + 10 chats + 6 tokenizes + 6 completions recorded" \
  || bad "expected 23 recorded requests, got $req_n"
[ "$missing_auth" -eq 0 ] && ok "every request (incl. /tokenize gate) carries the launcher-provided bearer" \
  || bad "$missing_auth request(s) missing/mismatching Authorization header"
grep -rq "ambient-vllm-key" "$rec5" && bad "ambient VLLM_API_KEY value leaked into requests or logs" \
  || ok "ambient VLLM_API_KEY never sent nor logged"

# ---- 8. VLLM_API_KEY-only fallback preserved; open cluster sends no header ----
rec6="$work/r6"
run_sweep ok "$rec6" VLLM_API_KEY=sk-solo-vllm >/dev/null 2>&1 || true
solo_bad=0
for f in "$rec6"/chat-* "$rec6"/comp-* "$rec6"/tok-*; do
  [ -f "$f" ] || continue
  [ "$(head -n 1 "$f")" = "Authorization: Bearer sk-solo-vllm" ] || solo_bad=$((solo_bad + 1))
done
[ "$solo_bad" -eq 0 ] && ok "VLLM_API_KEY fallback authenticates every chat/ladder request" \
  || bad "$solo_bad requests unauthenticated under fallback"
rec7="$work/r7"
run_sweep ok "$rec7" DSPARK_WARMUP_BEARER= VLLM_API_KEY= >/dev/null 2>&1 || true
grep -rq "Authorization:" "$rec7" && bad "open cluster sent an Authorization header" \
  || ok "no keys configured -> no Authorization header anywhere"

# ---- 9. chat failures: exit 1, precise failure count on stderr ----------------
rec3="$work/r3"
if run_sweep chatfail "$rec3"; then bad "chatfail sweep exited 0"; else ok "chatfail sweep exits nonzero"; fi
grep -q "10 request(s) failed" "$rec3/stderr" && ok "failure count named on stderr" \
  || bad "expected '10 request(s) failed' on stderr"

# ---- 10. unreachable API: exit 1 fast, no requests fired ----------------------
rec4="$work/r4"
if run_sweep probefail "$rec4"; then bad "probefail sweep exited 0"; else ok "probefail sweep exits nonzero"; fi
post_probe=$(ls "$rec4"/chat-* "$rec4"/comp-* "$rec4"/tok-* 2>/dev/null | wc -l)
[ "$post_probe" -eq 0 ] && ok "nothing fired after failed probe" \
  || bad "$post_probe request(s) fired after failed probe"

# ---- 11. subshell dying before writing still tallies: 15/16, never n/n short ---
rec8="$work/r8"
if run_sweep killparent "$rec8"; then bad "killparent sweep exited 0"; else ok "lost outcome fails the sweep"; fi
grep -q "^boot-shape-warmup: 15/16 requests ok" "$rec8/stdout" \
  && ok "request whose subshell died before writing counted as failed" \
  || bad "summary not 15/16: $(grep 'requests ok' "$rec8/stdout" || true)"
grep -q "1 request(s) failed" "$rec8/stderr" || bad "failure count missing on stderr"

# ---- 12. tokenize endpoint error: rung skipped, counted, precise diagnostic ----
rec9="$work/r9"
if run_sweep tokfail45 "$rec9"; then bad "tokfail45 sweep exited 0"; else ok "tokenize-error rung fails the sweep"; fi
grep -q "rung s=45: POST /tokenize errored" "$rec9/stderr" \
  && ok "tokenize transport failure diagnosed precisely (rung s=45)" \
  || bad "missing precise diagnostic for tokenize transport failure"
tf_comp=$(ls "$rec9"/comp-* 2>/dev/null | wc -l)
[ "$tf_comp" -eq 5 ] && ok "failed-gate rung did NOT fire its completion" \
  || bad "expected 5 completions after skipped rung, got $tf_comp"
grep -q "^boot-shape-warmup: 15/16 requests ok" "$rec9/stdout" \
  && ok "skipped rung still counted (15/16)" \
  || bad "summary not 15/16 after tokenize transport failure"

# ---- 13. tokenize count mismatch: rung skipped, counted, secret-free diag ------
rec10="$work/r10"
if run_sweep badcount45 "$rec10"; then bad "badcount45 sweep exited 0"; else ok "count-mismatch rung fails the sweep"; fi
grep -q "/tokenize reported 46 tokens, need exactly 45" "$rec10/stderr" \
  && ok "count mismatch diagnosed precisely (46 vs 45)" \
  || bad "missing precise mismatch diagnostic"
bc_comp=$(ls "$rec10"/comp-* 2>/dev/null | wc -l)
[ "$bc_comp" -eq 5 ] && ok "mismatched rung did NOT fire its completion" \
  || bad "expected 5 completions after mismatched rung, got $bc_comp"
grep -q "^boot-shape-warmup: 15/16 requests ok" "$rec10/stdout" \
  && ok "mismatched rung still counted (15/16)" \
  || bad "summary not 15/16 after count mismatch"

printf 'test-boot-shape-warmup: %d ok, %d failed\n' "$pass" "$fail"
exit "$((fail > 0 ? 1 : 0))"
