#!/usr/bin/env bash
# CPU-only gates for the NCCL_IB_HCA -> sysfs device expansion used by
# start-deepseek-v4-flash-dspark.sh's NCCL_IB_GID_AUTO=1 path.
#
# Before this, the resolver used the raw NCCL_IB_HCA value as a sysfs directory
# name, so the multi-HCA exact-match form ("=devA,devB" — what
# NCCL_IB_MERGE_NICS requires) produced /sys/class/infiniband/=devA,devB/... and
# the launcher FATALed before starting either rank.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
START="$ROOT/start-deepseek-v4-flash-dspark.sh"
QUIET=0
[ "${1:-}" = "-q" ] && QUIET=1

pass=0
fail=0
say() { [ "$QUIET" = "1" ] || printf '  ok  %s\n' "$*"; }
check() {
  local what="$1" got="$2" want="$3"
  if [ "$got" = "$want" ]; then
    pass=$((pass + 1))
    say "$what"
  else
    fail=$((fail + 1))
    printf '  FAIL %s: got %q want %q\n' "$what" "$got" "$want" >&2
  fi
}

# Pull the helper out of the launcher instead of duplicating it, so the test
# fails if the launcher's copy drifts.
helper="$(awk '/^nccl_ib_hca_devices\(\) \{$/,/^\}$/' "$START")"
if [ -z "$helper" ]; then
  echo "FAIL could not extract nccl_ib_hca_devices() from $START" >&2
  exit 1
fi
eval "$helper"

check "bare device passes through"        "$(nccl_ib_hca_devices 'rocep1s0f1')"                  'rocep1s0f1'
check "exact-match single device"         "$(nccl_ib_hca_devices '=rocep1s0f1')"                 'rocep1s0f1'
check "exact-match multi-HCA (merge NIC)" "$(nccl_ib_hca_devices '=rocep1s0f1,roceP2p1s0f1')"    'rocep1s0f1 roceP2p1s0f1'
check "comma list without ="              "$(nccl_ib_hca_devices 'mlx5_0,mlx5_1')"               'mlx5_0 mlx5_1'
check ":port suffix stripped"             "$(nccl_ib_hca_devices '=mlx5_0:1,mlx5_1:1')"          'mlx5_0 mlx5_1'
check "exclusion spec -> scan all"        "$(nccl_ib_hca_devices '^mlx5_1')"                     ''
check "empty spec -> scan all"            "$(nccl_ib_hca_devices '')"                            ''
check "no sysfs metacharacters survive"   "$(nccl_ib_hca_devices '=a,b' | tr -d 'ab ')"          ''

# End-to-end against a fake sysfs tree. Run the launcher's *own* generated
# lookup script (transport stubbed out, sysfs root redirected) so this covers
# the shipped code and not a copy of it.
eval "$(awk '/^ipv4_to_gid_suffix\(\) \{$/,/^\}$/' "$START")"
eval "$(
  awk '/^resolve_rocev2_gid_index\(\) \{$/,/^\}$/' "$START" \
    | sed -e 's|bash -c "\$remote"|printf %s "$remote"|' \
          -e 's|ssh "\$ssh_target" "bash -s" <<<"\$remote"|printf %s "$remote"|'
)"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mk_hca() { # $1=name $2=gid index $3=gid value $4=type
  local d="$tmp/sys/class/infiniband/$1/ports/1"
  mkdir -p "$d/gids" "$d/gid_attrs/types"
  printf '%s\n' "$3" >"$d/gids/$2"
  printf '%s\n' "$4" >"$d/gid_attrs/types/$2"
}
# 10.0.22.1 -> ...ffff:0a00:1601
mk_hca devA 0 '::ffff:0a00:1601' 'RoCE v1'
mk_hca devA 3 '::ffff:0a00:1601' 'RoCE v2'
mk_hca devB 3 '::ffff:0a00:1601' 'RoCE v2'
mk_hca devC 5 '::ffff:0a00:1602' 'RoCE v2'

resolve() { # $1=NCCL_IB_HCA spec  $2=IPv4 to match
  resolve_rocev2_gid_index "" "$1" "$2" \
    | sed "s|/sys/class/infiniband|$tmp/sys/class/infiniband|g" \
    | bash
}

check "single HCA resolves RoCE v2 index" "$(resolve 'devA' '10.0.22.1')" '3'
check "multi-HCA resolves (was FATAL)"    "$(resolve '=devA,devB' '10.0.22.1')" '3'
check "exclusion spec scans every HCA"    "$(resolve '^devC' '10.0.22.2')" '5'
if resolve 'devC' '10.0.22.1' >/dev/null 2>&1; then
  fail=$((fail + 1))
  printf '  FAIL non-matching IP must not resolve\n' >&2
else
  pass=$((pass + 1))
  say "non-matching IP fails cleanly"
fi

printf 'RESULT: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
