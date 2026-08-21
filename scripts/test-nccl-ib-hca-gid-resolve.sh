#!/usr/bin/env bash
# CPU-only behavioral gates for the NCCL_IB_HCA -> RoCEv2 GID index resolver
# used by start-deepseek-v4-flash-dspark.sh's NCCL_IB_GID_AUTO=1 path.
#
# The resolver mirrors NCCL's selector semantics (parseStringList in
# src/misc/utils.cc) on the node that owns the sysfs tree: optional leading "^"
# (exclude), then optional "=" (exact match instead of prefix match), then
# comma-separated name[:port[:rail[:plane]]] tokens. An absent *or empty* port
# field means any port; a non-empty one is atoi() (base 10, stops at the first
# non-digit). Every selected member must validate against the preferred match IP
# or an IPv4 on its own netdev; a member with no usable RoCEv2 GID fails closed
# (exit 1). Members are reconciled by intersecting their sets of usable indexes
# — a member can have several, so an arbitrary per-member pick would report a
# false disagreement — and only an empty intersection fails closed (exit 3),
# because NCCL_IB_GID_INDEX is one global value per rank.
#
# The suite extracts the launcher's own functions and runs the launcher's own
# generated lookup script (transport stubbed, sysfs root redirected, `ip`
# stubbed from a fixture) against fake sysfs trees — so it exercises the
# shipped code, and running it against the pre-fix launcher fails these
# checks *behaviorally* (wrong result / wrong exit code), not merely because
# a helper cannot be extracted.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
START="$ROOT/start-deepseek-v4-flash-dspark.sh"
QUIET=0
[ "${1:-}" = "-q" ] && QUIET=1

pass=0
fail=0
say() { [ "$QUIET" = "1" ] || printf '  ok  %s\n' "$*"; }
ok() { pass=$((pass + 1)); say "$*"; }
bad() { fail=$((fail + 1)); printf '  FAIL %s\n' "$*" >&2; }

# --- extract the launcher's own code (works on pre-fix launchers too, so the
# regression shows up as behavioral FAILs below) ---
eval "$(awk '/^ipv4_to_gid_suffix\(\) \{$/,/^\}$/' "$START")"
resolver_body_src="$(awk '/^NCCL_HCA_RESOLVER_BODY=/,/^\)"$/' "$START")"
if [ -n "$resolver_body_src" ]; then
  eval "$resolver_body_src"
else
  NCCL_HCA_RESOLVER_BODY=""
fi
eval "$(
  awk '/^resolve_rocev2_gid_index\(\) \{$/,/^\}$/' "$START" \
    | sed -e 's|bash -c "\$remote"|printf %s "$remote"|' \
          -e 's|ssh "\$ssh_target" "bash -s" <<<"\$remote"|printf %s "$remote"|'
)"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# `ip` stub: answers `ip -4 -o addr show dev NAME` from $IP_FIXTURE
# ("<netdev> <ipv4>" lines), mimicking the real -o output shape.
stub="$tmp/stub-bin"
mkdir -p "$stub"
cat >"$stub/ip" <<'STUB'
#!/usr/bin/env bash
dev="${!#}"
[ -f "${IP_FIXTURE:-}" ] || exit 0
while read -r d a; do
  [ "$d" = "$dev" ] && printf '2: %s    inet %s/24 brd 0.0.0.0 scope global %s\n' "$d" "$a" "$d"
done <"$IP_FIXTURE"
exit 0
STUB
chmod +x "$stub/ip"

mk_gid() { # $1=root $2=dev $3=port $4=index $5=gid $6=type [$7=ndev]
  local d="$1/sys/class/infiniband/$2/ports/$3"
  mkdir -p "$d/gids" "$d/gid_attrs/types" "$d/gid_attrs/ndevs"
  printf '%s\n' "$5" >"$d/gids/$4"
  printf '%s\n' "$6" >"$d/gid_attrs/types/$4"
  if [ -n "${7:-}" ]; then printf '%s\n' "$7" >"$d/gid_attrs/ndevs/$4"; fi
}

resolve() { # $1=root $2=fixture $3=spec $4=match-ip [$5=ssh-target]
  resolve_rocev2_gid_index "${5:-}" "$3" "$4" \
    | sed "s|/sys/class/infiniband|$1/sys/class/infiniband|g" \
    | IP_FIXTURE="$2" PATH="$stub:$PATH" bash
}

expect_idx() { # $1=label $2=want $3=root $4=fixture $5=spec $6=ip [$7=ssh]
  local rc=0 got=""
  got="$(resolve "$3" "$4" "$5" "$6" "${7:-}" 2>/dev/null)" || rc=$?
  if [ "$rc" -eq 0 ] && [ "$got" = "$2" ]; then
    ok "$1"
  else
    bad "$1: rc=$rc got='$got' want='$2'"
  fi
}

expect_rc() { # $1=label $2=want-rc $3=root $4=fixture $5=spec $6=ip
  local rc=0
  resolve "$3" "$4" "$5" "$6" >/dev/null 2>&1 || rc=$?
  if [ "$rc" -eq "$2" ]; then
    ok "$1"
  else
    bad "$1: rc=$rc want=$2"
  fi
}

# --- head layout: three single-port HCAs on one shared link address, plus a
# device on a different address with no ndev metadata ---
# 10.0.22.1 -> ...ffff:0a00:1601 ; 10.0.22.2 -> ...ffff:0a00:1602
h="$tmp/head"
hfx="$tmp/head.fixture"
: >"$hfx"
mk_gid "$h" devA 1 0 '::ffff:0a00:1601' 'RoCE v1'
mk_gid "$h" devA 1 3 '::ffff:0a00:1601' 'RoCE v2'
mk_gid "$h" devB 1 3 '::ffff:0a00:1601' 'RoCE v2'
mk_gid "$h" devC 1 5 '::ffff:0a00:1602' 'RoCE v2'
mk_gid "$h" devE 1 6 '::ffff:0a00:1601' 'RoCE v2'

expect_idx "bare device resolves (prefix match)" 3 "$h" "$hfx" 'devA' '10.0.22.1'
expect_idx "exact-match single device" 3 "$h" "$hfx" '=devA' '10.0.22.1'
expect_idx "exact multi-HCA list (original regression)" 3 "$h" "$hfx" '=devA,devB' '10.0.22.1'
expect_idx "comma list without =" 3 "$h" "$hfx" 'devA,devB' '10.0.22.1'
expect_idx "exact token skips missing device, keeps real one" 3 "$h" "$hfx" '=devA,devGone' '10.0.22.1'
expect_idx "exclusion selects the rest" 3 "$h" "$hfx" '^devC,devE' '10.0.22.1'
expect_idx "exact exclusion selects the rest" 3 "$h" "$hfx" '^=devC,devE' '10.0.22.1'
expect_idx "single member on its own address" 5 "$h" "$hfx" 'devC' '10.0.22.2'
expect_rc "member with no usable GID fails closed" 1 "$h" "$hfx" 'devC' '10.0.22.1'
expect_rc "selector matching nothing fails closed" 1 "$h" "$hfx" '=devGone' '10.0.22.1'
expect_rc "shared-IP selection with an unvalidatable member fails closed (empty selector = all ports)" 1 "$h" "$hfx" '' '10.0.22.1'
expect_rc "intra-node index disagreement fails closed (exit 3)" 3 "$h" "$hfx" '=devA,devE' '10.0.22.1'

# disagreement diagnostic names the members and their usable sets
rc=0
err="$(resolve "$h" "$hfx" '=devA,devE' '10.0.22.1' 2>&1 >/dev/null)" || rc=$?
if [ "$rc" -eq 3 ] && printf '%s' "$err" | grep -q 'share no common RoCEv2 GID index' \
  && printf '%s' "$err" | grep -q 'devA:1=3' && printf '%s' "$err" | grep -q 'devE:1=6'; then
  ok "disagreement diagnostic lists each member's usable set"
else
  bad "disagreement diagnostic wrong (rc=$rc): $err"
fi

# --- worker layout: dual-port HCAs on distinct per-port link addresses,
# resolved via each member's own netdev (no shared match IP involved) ---
# 10.0.25.x -> ...ffff:0a00:19xx ; 10.0.26.x -> ...ffff:0a00:1axx
w="$tmp/worker"
wfx="$tmp/worker.fixture"
mk_gid "$w" devM 1 2 '::ffff:0a00:1901' 'RoCE v2' enm1
mk_gid "$w" devM 2 2 '::ffff:0a00:1902' 'RoCE v2' enm2
mk_gid "$w" devN 1 2 '::ffff:0a00:1a01' 'RoCE v2' enn1
mk_gid "$w" devN 2 9 '::ffff:0a00:1a02' 'RoCE v2' enn2
printf '%s\n' 'enm1 10.0.25.1' 'enm2 10.0.25.2' 'enn1 10.0.26.1' 'enn2 10.0.26.2' >"$wfx"

expect_idx "omitted port = both ports, validated per member (own-addr)" 2 "$w" "$wfx" 'devM' '10.0.99.99'
expect_idx "explicit port 1" 2 "$w" "$wfx" 'devN:1' '10.0.99.99'
expect_idx "explicit port 2" 9 "$w" "$wfx" 'devN:2' '10.0.99.99'
expect_rc "multiport disagreement fails closed (exit 3)" 3 "$w" "$wfx" 'devN' '10.0.99.99'
expect_idx "distinct per-HCA addresses both validate via own netdev" 2 "$w" "$wfx" '=devM:1,devN:1' '10.0.99.99'
expect_idx "match-ip member + own-addr member agree" 2 "$w" "$wfx" '=devM:1,devN:1' '10.0.25.1'
expect_idx "port exclusion (^dev:port honors the port)" 2 "$w" "$wfx" '^devN:2' '10.0.99.99'
expect_rc "non-numeric port token matches no port" 1 "$w" "$wfx" 'devM:abc' '10.0.99.99'

# --- NCCL port-field grammar ---
# parseStringList splits name[:port[:rail[:plane]]]; an absent or empty port
# field is -1 (any port), a non-empty one is atoi() (base 10, stops at the first
# non-digit). Ports 1/8/10 exist here so leading-zero forms are distinguishable:
# "08" must be 8 (not a bad octal literal) and "010" must be 10 (not octal 8).
# 10.0.32.1 -> ...0a00:2001 ; 10.0.32.8 -> ...0a00:2008 ; 10.0.32.16 -> ...0a00:2010
pg="$tmp/portgrammar"
pgfx="$tmp/pg.fixture"
mk_gid "$pg" devP 1 4 '::ffff:0a00:2001' 'RoCE v2' enp1
mk_gid "$pg" devP 8 7 '::ffff:0a00:2008' 'RoCE v2' enp8
mk_gid "$pg" devP 10 8 '::ffff:0a00:2010' 'RoCE v2' enp10
printf '%s\n' 'enp1 10.0.32.1' 'enp8 10.0.32.8' 'enp10 10.0.32.16' >"$pgfx"

expect_idx "explicit port 1 (baseline for the grammar cases)" 4 "$pg" "$pgfx" 'devP:1' '10.0.99.99'
expect_idx "leading-zero port :08 is decimal 8, not a bad octal literal" 7 "$pg" "$pgfx" 'devP:08' '10.0.99.99'
expect_idx "leading-zero port :010 is decimal 10, not octal 8" 8 "$pg" "$pgfx" 'devP:010' '10.0.99.99'
expect_idx "signed port :+8 parses as 8" 7 "$pg" "$pgfx" 'devP:+8' '10.0.99.99'
expect_idx "atoi stops at the first non-digit (:8abc -> 8)" 7 "$pg" "$pgfx" 'devP:8abc' '10.0.99.99'
expect_idx "atoi skips leading blanks (: 8 -> 8)" 7 "$pg" "$pgfx" 'devP: 8' '10.0.99.99'
expect_idx "rail/plane fields are parsed off, port still wins (:8:2:0)" 7 "$pg" "$pgfx" 'devP:8:2:0' '10.0.99.99'
expect_rc "non-numeric port is atoi 0 and matches no real port" 1 "$pg" "$pgfx" 'devP:abc' '10.0.99.99'
# Empty port field == absent == any port. All three ports here need different
# indexes, so wildcard selection is observable as exit 3 (a port-0 reading would
# instead match nothing and exit 1).
expect_rc "explicit empty port field is a wildcard, not port 0" 3 "$pg" "$pgfx" 'devP:' '10.0.99.99'
expect_rc "empty port field before a rail field is also a wildcard" 3 "$pg" "$pgfx" 'devP::2' '10.0.99.99'
expect_idx "omitted port on a single-port match still resolves" 4 "$pg" "$pgfx" 'devP:1:0' '10.0.99.99'

# --- per-member index sets are intersected, not compared pairwise ---
# Each member has two usable RoCEv2 indexes and they overlap on 5. Taking one
# arbitrary winner per member picks 1 for devX and 3 for devY (lowest scanned
# first) and reports a false disagreement; intersecting the sets finds 5.
# 10.0.40.1 -> ...0a00:2801 ; 10.0.40.9 -> ...0a00:2809
# 10.0.41.1 -> ...0a00:2901 ; 10.0.41.9 -> ...0a00:2909 ; 10.0.42.1 -> ...0a00:2a01
x="$tmp/multi"
xfx="$tmp/multi.fixture"
mk_gid "$x" devX 1 1 '::ffff:0a00:2801' 'RoCE v2' enx1
mk_gid "$x" devX 1 5 '::ffff:0a00:2809' 'RoCE v2' enx1
mk_gid "$x" devY 1 3 '::ffff:0a00:2901' 'RoCE v2' eny1
mk_gid "$x" devY 1 5 '::ffff:0a00:2909' 'RoCE v2' eny1
mk_gid "$x" devZ 1 2 '::ffff:0a00:2a01' 'RoCE v2' enz1
printf '%s\n' 'enx1 10.0.40.1' 'enx1 10.0.40.9' 'eny1 10.0.41.1' 'eny1 10.0.41.9' 'enz1 10.0.42.1' >"$xfx"

expect_idx "members with overlapping index sets resolve to the common index" 5 "$x" "$xfx" '=devX:1,devY:1' '10.0.99.99'
expect_idx "single member still takes its lowest usable index" 1 "$x" "$xfx" '=devX:1' '10.0.99.99'
expect_rc "genuinely disjoint index sets still fail closed (exit 3)" 3 "$x" "$xfx" '=devX:1,devZ:1' '10.0.99.99'

# The empty-intersection diagnostic must show each member's whole usable set,
# not one arbitrary pick, so the operator can see why no pin can work.
rc=0
err="$(resolve "$x" "$xfx" '=devX:1,devZ:1' '10.0.99.99' 2>&1 >/dev/null)" || rc=$?
if [ "$rc" -eq 3 ] && printf '%s' "$err" | grep -q 'devX:1=1,5' && printf '%s' "$err" | grep -q 'devZ:1=2'; then
  ok "empty-intersection diagnostic lists each member's full usable set"
else
  bad "usable-set diagnostic wrong (rc=$rc): $err"
fi

# A match-IP hit inside the intersection is preferred over a lower own-addr one.
expect_idx "match-ip index inside the intersection wins over a lower own-addr index" 5 "$x" "$xfx" '=devX:1,devY:1' '10.0.40.9'

# --- independent head/worker layouts in one run (ssh transport path) ---
expect_idx "head layout resolves independently" 3 "$h" "$hfx" '=devA,devB' '10.0.22.1'
expect_idx "worker layout resolves independently over ssh path" 9 "$w" "$wfx" 'devN:2' '10.0.99.99' 'user@worker'

# --- prefix collisions and literal token transport ---
r3="$tmp/roce"
rfx="$tmp/roce.fixture"
: >"$rfx"
mk_gid "$r3" rocep1s0f1 1 3 '::ffff:0a00:1601' 'RoCE v2'
mk_gid "$r3" roceP2p1s0f1 1 3 '::ffff:0a00:1601' 'RoCE v2'

expect_idx "prefix collision: one token matches both HCAs" 3 "$r3" "$rfx" 'roce' '10.0.22.1'
expect_idx "empty selector matches all (agreeing) ports" 3 "$r3" "$rfx" '' '10.0.22.1'
expect_rc "exact form does not prefix-match" 1 "$r3" "$rfx" '=roce' '10.0.22.1'
expect_idx "prefix token shorter than device name" 3 "$r3" "$rfx" 'rocep1' '10.0.22.1'

trapdir="$tmp/glob-trap"
mkdir -p "$trapdir"
touch "$trapdir/devA-trap" "$trapdir/rocep1s0f1"
rc=0
( cd "$trapdir" && resolve "$r3" "$rfx" '=dev*' '10.0.22.1' >/dev/null 2>&1 ) || rc=$?
if [ "$rc" -eq 1 ]; then
  ok "glob metacharacters stay literal (no pathname expansion, fail closed)"
else
  bad "glob token misbehaved: rc=$rc (want 1)"
fi
expect_rc "whitespace inside a token transports literally, fails closed" 1 "$r3" "$rfx" '=de vA' '10.0.22.1'
expect_rc "missing sysfs tree fails closed" 1 "$tmp/nonexistent" "$rfx" 'devA' '10.0.22.1'

printf 'RESULT: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
