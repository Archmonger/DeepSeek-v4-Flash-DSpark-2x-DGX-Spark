#!/usr/bin/env bash
# hotfix-dsv4-autotune-prod-dense.sh — merge tuned configs for production dense budgets
#
# BACKGROUND:
#   The DeepSeek-V4-Flash-DSpark autotuner (flashinfer mla/_sparse_mla_sm120.py)
#   only buckets the *token* axis of sparse_mla_sm120_decode_dsv4. The dense /
#   extra_topk axis is matched EXACTLY against the autotune cache, and the
#   boot-time warmup never physically runs dense budgets >= 1024 (it only
#   produces dense 0/128/512). So every production decode shape — dense
#   256/1024/2048/8192 at T=6..58 — is a cache MISS and falls back to the C++
#   closed-form heuristic (tactic=-1).
#
#   Measured on spark1 (idle GPU, 2026-08-18): heuristic vs tuned
#   chunks_per_block costs +20% to +82% per kernel call on exactly those
#   shapes, ~+29-33% traffic-weighted across the observed production mix.
#   The dense=8192 shapes are CUDA-graph-capture shapes that replay on EVERY
#   decode step.
#
# WHAT THIS DOES:
#   Ships a pre-generated autotune supplement
#   (dsv4-autotune-prod-dense-supp.json, produced by running the REAL tuning
#   pass against dense 256/1024/2048/8192; 24 configs covering all T buckets)
#   and patches vllm's write_flashinfer_autotune_cache() so that on every
#   boot, after the standard warmup tuning writes its (dense<=512-only) cache,
#   the production-dense configs are merged back in. Idempotent: the merge is
#   run on every boot regardless of prior state, and the hotfix must be
#   re-applied only if the container is rebuilt from image.
#
# WHY MERGE (not seed): vllm's write_flashinfer_autotune_cache() atomically
#   os.replace()s the cache on every startup, so any pre-seeded file would be
#   clobbered at the next boot. Patching the writer makes production-dense
#   coverage permanent across restarts.
#
# USAGE/DEPLOYMENT:
#   Place both files in the miaai patches/ directory:
#     - hotfix-dsv4-autotune-prod-dense.sh
#     - dsv4-autotune-prod-dense-supp.json
#   They are mounted into the container at /opt/dspark-patches/:ro by
#   docker-compose.dspark.yml (DSPARK_PATCHES_DIR). The compose entrypoint can
#   run this like the other hotfix scripts:
#       if [ "$${DSPARK_SKIP_DSV4_AUTOTUNE_DENSE_HOTFIX:-0}" != "1" ] && [ -f /opt/dspark-patches/hotfix-dsv4-autotune-prod-dense.sh ]; then
#         bash /opt/dspark-patches/hotfix-dsv4-autotune-prod-dense.sh || true
#       fi
#   Skip entirely with DSPARK_SKIP_DSV4_AUTOTUNE_DENSE_HOTFIX=1.
#
# NOTE: This does NOT modify the fork's tuning config, so a future boot tuning
#   pass still only profiles dense<=512 — but the merged supplement covers the
#   production dense axis, so the runtime reads tuned configs regardless.
#   (A proper upstream fix would bucket the dense axis in the fork's tuning
#   config; this is the low-risk operational fix.)
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
CACHE_WRITER="$VLLM_ROOT/model_executor/warmup/flashinfer_autotune_cache.py"
SUPP_SRC="/opt/dspark-patches/dsv4-autotune-prod-dense-supp.json"
SUPP_EMBED="${SUPP_EMBED:-/usr/local/lib/python3.12/dist-packages/vllm/dsv4-autotune-prod-dense-supp.json}"

# Marker inserted into the patched writer so the patch is idempotent.
PATCH_MARKER="dv4-prod-dense-merge"

if [ ! -f "$CACHE_WRITER" ]; then
  echo "ERROR: vllm flashinfer_autotune_cache.py not found at $CACHE_WRITER" >&2
  exit 1
fi
if [ ! -f "$SUPP_SRC" ] && [ ! -f "$SUPP_EMBED" ]; then
  echo "ERROR: production-dense autotune supplement not found (tried $SUPP_SRC, $SUPP_EMBED)" >&2
  exit 1
fi

echo "=== Hotfix: merge production-dense DSv4 autotune configs (perf cliff) ==="

# 1. Stage the supplement into the container image layer so it survives even
#    if the read-only /opt mount disappears, and the writer can read it.
if [ -f "$SUPP_SRC" ]; then
  cp -f "$SUPP_SRC" "$SUPP_EMBED"
  echo " [OK] supplement staged: $SUPP_EMBED"
fi

# 2. Patch the cache writer to merge the supplement in.
if grep -qF "$PATCH_MARKER" "$CACHE_WRITER"; then
  echo " [skip] cache writer already patched"
else
  python3 - "$CACHE_WRITER" "$SUPP_EMBED" "$PATCH_MARKER" <<'PYEOF'
import sys
from pathlib import Path
p = Path(sys.argv[1])
embed = sys.argv[2]
marker = sys.argv[3]
text = p.read_text()
old = '''def write_flashinfer_autotune_cache(cache_path: Path, contents: bytes) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)'''
new = '''def write_flashinfer_autotune_cache(cache_path: Path, contents: bytes) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _dv4_prod_dense_supp = Path(%r)
    if _dv4_prod_dense_supp.is_file():
        try:
            import json as _json
            _base = _json.loads(contents)
            _supp = _json.loads(_dv4_prod_dense_supp.read_text())
            _merged = {}
            _merged.update({k: v for k, v in _supp.items() if k != "_metadata"})
            _merged.update({k: v for k, v in _base.items() if k != "_metadata"})
            if "_metadata" in _base:
                _merged["_metadata"] = _base["_metadata"]
            elif "_metadata" in _supp:
                _merged["_metadata"] = _supp["_metadata"]
            _c = _json.dumps(_merged).encode()  # %s: production-dense merge
            if _c != contents:
                contents = _c
        except Exception:
            pass''' % (embed, marker)
if old not in text:
    print(" [WARN] write_flashinfer_autotune_cache body not found; skipping patch.", file=sys.stderr)
    sys.exit(0)
p.write_text(text.replace(old, new, 1))
print(" [OK] cache writer patched to merge production-dense configs")
PYEOF
fi

echo "=== Verification ==="
if ! grep -qF "$PATCH_MARKER" "$CACHE_WRITER"; then
  echo "[FAIL] cache writer patch marker missing" >&2
  exit 1
fi
python3 - "$CACHE_WRITER" <<'PYEOF'
import ast, sys
from pathlib import Path
# static check: the writer must reference the embedded supplement
t = Path(sys.argv[1]).read_text()
if "dsv4-autotune-prod-dense-supp.json" in t and "def write_flashinfer_autotune_cache" in t:
    print("[OK] writer references supplement and is patched")
    sys.exit(0)
print("[FAIL] writer not fully patched", file=sys.stderr)
sys.exit(1)
PYEOF
echo "[OK] hotfix-dsv4-autotune-prod-dense: applied (restart vLLM to load merged cache)"
