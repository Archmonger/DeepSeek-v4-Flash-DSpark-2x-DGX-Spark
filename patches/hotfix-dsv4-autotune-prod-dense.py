#!/usr/bin/env python3
"""Cover production dense budgets for the DSv4 sparse-MLA decode autotune cache.

BACKGROUND
    FlashInfer's DSv4 SM120 sparse-MLA decode tuning config buckets only the
    *token* axis. The dense / ``extra_topk`` axis is matched EXACTLY against the
    autotune cache, and the boot warmup only physically runs the dense budgets
    it can reach from ``_DEEPSEEK_V4_DSPARK_DECODE_AUTOTUNE_SEQ_LENS``. Every
    production decode shape on a dense budget the boot pass never ran is a cache
    MISS and falls back to the C++ closed-form heuristic (``tactic=-1``).

    This ships a pre-generated supplement (``dsv4-autotune-prod-dense-supp.json``,
    24 configs: token axis {1,4,8,16,32,64} x dense {256,1024,2048,8192},
    produced by the real autotuner) and merges it into the boot-tuned cache.

WHERE THE MERGE HOOKS, AND WHY
    ``vllm/model_executor/warmup/kernel_warmup.py`` owns the DSv4 cache
    end-to-end: FlashInfer's ``sparse_mla_sm120_decode_dsv4_autotune`` context
    manager writes the file, the tuning leader reads the bytes back, they are
    broadcast to every rank, rank>0 writes them to its own copy, and then EVERY
    rank calls ``AutoTuner.get().load_configs(cache_path)``. vLLM's
    ``write_flashinfer_autotune_cache()`` is NOT part of that path, so patching
    it has no effect on DSv4 decode.

    The merge is therefore injected on the leader, after it reads the freshly
    tuned cache and BEFORE ``world.broadcast_object``:

      * the broadcast then carries merged bytes, so rank>0 writes a merged file;
      * the leader also rewrites its own on-disk cache, because upstream never
        does and every rank loads configs from disk;
      * both ranks consequently load an identical config set. Merging after the
        broadcast, or on one rank only, would leave the ranks dispatching
        different kernel tactics.

SAFETY
    * The feature is experimental and default-off. It runs only when
      ``DSPARK_ENABLE_DSV4_AUTOTUNE_DENSE_HOTFIX=1``.
    * The exact supplement digest and its ``_metadata`` fingerprint (FlashInfer /
      CUDA / cuBLAS / cuDNN versions and GPU name) must match the patcher and the
      running build.
    * Enabled mode is fail-closed: missing, changed, unreadable, incompatible, or
      unwritable inputs stop boot rather than silently serving with the
      heuristic fallback.
    * Anchors are pinned against the vendored copy of the same file at
      ``recipe/overlay/vllm/model_executor/warmup/kernel_warmup.py`` and gated by
      ``scripts/ci-validate.sh``, so anchor drift fails before serving.

REGENERATING THE SUPPLEMENT (needs the target GPU; not a CPU step)
    A new image or FlashInfer build needs a fresh supplement tuned on that build:

    1. Leave the feature disabled (the default) and widen the boot sweep until it
       physically runs the production dense budgets — extend
       ``_DEEPSEEK_V4_DSPARK_DECODE_AUTOTUNE_SEQ_LENS`` in
       ``vllm/model_executor/warmup/kernel_warmup.py`` so the DSv4 indexer
       reaches dense 256/1024/2048/8192.
    2. Extract the production-dense entries from the cache that boot produced
       (``_resolve_flashinfer_autotune_file()``, i.e.
       ``$VLLM_CACHE_ROOT/flashinfer_autotune_cache/.../<hash>/autotune_configs.json``)::

           python3 -c 'import ast,json,sys;\
           d=json.load(open(sys.argv[1]));\
           want={256,1024,2048,8192};\
           out={k:v for k,v in d.items() if k!="_metadata"\
                and ast.literal_eval(k)[0]=="sparse_mla_sm120_decode_dsv4"\
                and ast.literal_eval(k)[3][2] in want};\
           out["_metadata"]=d["_metadata"];\
           json.dump(out,open(sys.argv[2],"w"),indent=2,sort_keys=True)' \
             autotune_configs.json patches/dsv4-autotune-prod-dense-supp.json

    3. Revert step 1, update ``SUPP_SHA256`` and
       ``_DV4_PROD_DENSE_SHA256`` in this file to the new supplement's SHA-256,
       then run ``scripts/ci-validate.sh``.

USAGE
    python3 hotfix-dsv4-autotune-prod-dense.py [--vllm-root DIR] [--check] [-q]

    Applied from the compose entrypoint before ``exec vllm`` only when
    ``DSPARK_ENABLE_DSV4_AUTOTUNE_DENSE_HOTFIX=1``. Idempotent: a marker comment
    makes re-runs a no-op while the staged supplement is refreshed.

    A boot tuning pass still only profiles the dense budgets it can reach; the
    durable upstream fix is to bucket the dense axis in FlashInfer's
    ``_decode_dsv4_tuning_config()``. This is the operational fix.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

MARKER = "dv4-prod-dense-merge"
SUPP_BASENAME = "dsv4-autotune-prod-dense-supp.json"
SUPP_SHA256 = "9ff86acaef00a2ab96e220351a1aea6e0b51b70ef7b25c9ade98f48783fcec0b"
DEFAULT_VLLM_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MOUNTED_PATCHES_DIR = Path("/opt/dspark-patches")
TARGET_RELPATH = Path("model_executor/warmup/kernel_warmup.py")

# Prerequisites the injected helper closes over in kernel_warmup.py's scope.
PREREQS = ("logger = init_logger(", "from pathlib import Path")

# Pinned vendored anchor. The narrower block
#     tune_results: bytes | None = None ... world.broadcast_object(...)
# appears TWICE in kernel_warmup.py (once in the DSv4 function, once in the
# generic flashinfer_autotune()), so the anchor is scoped by the DSv4-only
# `sparse_mla_sm120_decode_dsv4_autotune` line to stay unambiguous.
ANCHOR_OLD = '''            with sparse_mla_sm120_decode_dsv4_autotune(cache_path=str(cache_path)):
                run_autotune_shapes()
        else:
            run_autotune_shapes()

    tune_results: bytes | None = None
    if is_leader and cache_path.exists():
        with open(cache_path, "rb") as f:
            tune_results = f.read()

    tune_results = world.broadcast_object(tune_results, src=0)
'''

ANCHOR_NEW = '''            with sparse_mla_sm120_decode_dsv4_autotune(cache_path=str(cache_path)):
                run_autotune_shapes()
        else:
            run_autotune_shapes()

    tune_results: bytes | None = None
    if is_leader:
        if not cache_path.exists():
            raise RuntimeError(
                "DSv4 production-dense autotune is enabled, but boot tuning "
                f"did not create {cache_path}"
            )
        with open(cache_path, "rb") as f:
            tune_results = f.read()
        # dv4-prod-dense-merge: widen dense coverage before the broadcast so
        # every rank receives and loads the same config set.
        tune_results = _dv4_prod_dense_merge(tune_results, cache_path)

    tune_results = world.broadcast_object(tune_results, src=0)
'''

HELPER_ANCHOR = "def _deepseek_v4_sparse_mla_decode_autotune(\n"

HELPER_BLOCK = '''# --- dv4-prod-dense-merge (MiaAI patches/hotfix-dsv4-autotune-prod-dense.py) ---
_DV4_PROD_DENSE_BASENAME = "dsv4-autotune-prod-dense-supp.json"
_DV4_PROD_DENSE_SHA256 = "9ff86acaef00a2ab96e220351a1aea6e0b51b70ef7b25c9ade98f48783fcec0b"


def _dv4_prod_dense_supp_candidates() -> "list[Path]":
    """Staged copy inside the vllm package first, then the read-only mount."""
    return [
        Path(__file__).resolve().parents[2] / _DV4_PROD_DENSE_BASENAME,
        Path("/opt/dspark-patches") / _DV4_PROD_DENSE_BASENAME,
    ]


def _dv4_prod_dense_merge(tune_results: bytes, cache_path: "Path") -> bytes:
    """Merge the pinned production-dense configs into the leader's DSv4 cache.

    This helper exists only when the operator explicitly enables the hotfix.
    Any missing, stale, incompatible, or unwritable input is therefore fatal:
    silently serving with the heuristic fallback would invalidate the run.
    """
    import hashlib as _hashlib
    import json as _json
    import os as _os

    candidates = _dv4_prod_dense_supp_candidates()
    supp_path = next((p for p in candidates if p.is_file()), None)
    if supp_path is None:
        raise RuntimeError(
            "DSv4 production-dense autotune supplement not found; looked in "
            + ", ".join(str(p) for p in candidates)
        )

    try:
        supp_bytes = supp_path.read_bytes()
        base = _json.loads(tune_results)
        supp = _json.loads(supp_bytes)
    except (ValueError, OSError) as exc:
        raise RuntimeError(
            "DSv4 production-dense autotune cache or supplement is unreadable"
        ) from exc
    digest = _hashlib.sha256(supp_bytes).hexdigest()
    if digest != _DV4_PROD_DENSE_SHA256:
        raise RuntimeError(
            "DSv4 production-dense autotune supplement digest mismatch: "
            f"expected {_DV4_PROD_DENSE_SHA256}, got {digest}"
        )
    if not isinstance(base, dict) or not isinstance(supp, dict):
        raise RuntimeError(
            "DSv4 production-dense autotune cache or supplement is not a JSON object"
        )

    base_meta = base.get("_metadata")
    supp_meta = supp.get("_metadata")
    if not isinstance(base_meta, dict) or not isinstance(supp_meta, dict):
        raise RuntimeError(
            "DSv4 production-dense autotune cache or supplement has no "
            "_metadata fingerprint"
        )
    if base_meta != supp_meta:
        differing = sorted(
            key
            for key in set(base_meta) | set(supp_meta)
            if base_meta.get(key) != supp_meta.get(key)
        )
        raise RuntimeError(
            "DSv4 production-dense autotune supplement targets a different build; "
            "mismatched _metadata keys: "
            + ", ".join(differing)
        )

    added = [key for key in supp if key != "_metadata" and key not in base]
    if not added:
        logger.info(
            "DSv4 production-dense autotune coverage already present in the "
            "boot-tuned cache; no merge needed."
        )
        return tune_results

    # Boot-tuned entries win on collision: they were measured on this machine.
    merged = {key: value for key, value in supp.items() if key != "_metadata"}
    merged.update({key: value for key, value in base.items() if key != "_metadata"})
    merged["_metadata"] = base_meta
    payload = _json.dumps(merged).encode()

    # Upstream never rewrites the leader's own cache file, but every rank loads
    # configs from disk, so the merged payload has to land here as well.
    tmp_path = cache_path.with_name(cache_path.name + ".dv4tmp")
    try:
        with open(tmp_path, "wb") as handle:
            handle.write(payload)
            handle.flush()
            _os.fsync(handle.fileno())
        _os.replace(tmp_path, cache_path)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            "DSv4 production-dense autotune merged cache could not be persisted"
        ) from exc

    logger.info(
        "Merged %d production-dense DSv4 sparse-MLA decode configs into the "
        "autotune cache (%d boot-tuned entries kept, %d total); every rank will "
        "load this set from %s.",
        len(added),
        len(base) - 1,
        len(merged) - 1,
        cache_path,
    )
    return payload


# --- end dv4-prod-dense-merge ---


'''


def validate_supplement(text: str) -> tuple[dict | None, str]:
    """Parse and validate the exact supplement paired with this patcher."""
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != SUPP_SHA256:
        return None, f"digest mismatch (expected {SUPP_SHA256}, got {digest})"
    try:
        data = json.loads(text)
    except ValueError as exc:
        return None, f"not valid JSON ({exc})"
    if not isinstance(data, dict):
        return None, "top-level value is not a JSON object"
    meta = data.get("_metadata")
    if not isinstance(meta, dict) or not meta:
        return None, "missing or empty _metadata fingerprint"
    configs = [key for key in data if key != "_metadata"]
    if not configs:
        return None, "contains no tuned configs"
    return data, f"{len(configs)} configs, fingerprint {sorted(meta)}"


def apply_text(src: str) -> tuple[str, str]:
    """Return (new_source, status) without touching the filesystem.

    Statuses: ``applied``, ``already``, ``prereq-missing``, ``anchor-missing``,
    ``anchor-ambiguous``, ``helper-anchor-missing``, ``helper-anchor-ambiguous``,
    ``broken-output``.
    """
    if MARKER in src:
        return src, "already"

    for prereq in PREREQS:
        if prereq not in src:
            return src, "prereq-missing"

    call_hits = src.count(ANCHOR_OLD)
    if call_hits == 0:
        return src, "anchor-missing"
    if call_hits > 1:
        return src, "anchor-ambiguous"

    helper_hits = src.count(HELPER_ANCHOR)
    if helper_hits == 0:
        return src, "helper-anchor-missing"
    if helper_hits > 1:
        return src, "helper-anchor-ambiguous"

    out = src.replace(ANCHOR_OLD, ANCHOR_NEW, 1)
    out = out.replace(HELPER_ANCHOR, HELPER_BLOCK + HELPER_ANCHOR, 1)

    try:
        ast.parse(out)
    except SyntaxError:
        return src, "broken-output"
    return out, "applied"


def _log(quiet: bool, message: str) -> None:
    if not quiet:
        print(message)


def atomic_write(path: Path, payload: bytes) -> None:
    """Replace one file without exposing partial bytes to a restart."""
    tmp = path.with_name(path.name + ".dv4tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--vllm-root",
        type=Path,
        default=DEFAULT_VLLM_ROOT,
        help=f"installed vllm package root (default: {DEFAULT_VLLM_ROOT})",
    )
    parser.add_argument(
        "--supplement",
        type=Path,
        default=None,
        help="supplement JSON to stage (default: the read-only patches mount)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what would happen; write nothing",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="only report problems")
    args = parser.parse_args(argv)

    target = args.vllm_root / TARGET_RELPATH
    if not target.is_file():
        print(f"[FAIL] kernel_warmup.py not found at {target}", file=sys.stderr)
        return 1

    supp_src = args.supplement or (MOUNTED_PATCHES_DIR / SUPP_BASENAME)
    supp_staged = args.vllm_root / SUPP_BASENAME

    # Validate whichever copy we are going to rely on before touching anything.
    supp_for_check = supp_src if supp_src.is_file() else supp_staged
    if not supp_for_check.is_file():
        print(
            f"[FAIL] production-dense supplement not found (tried {supp_src}, {supp_staged})",
            file=sys.stderr,
        )
        return 1
    parsed, reason = validate_supplement(supp_for_check.read_text())
    if parsed is None:
        print(f"[FAIL] supplement {supp_for_check}: {reason}", file=sys.stderr)
        return 1
    _log(args.quiet, f"  supplement ok: {supp_for_check} ({reason})")

    src = target.read_text()
    new_src, status = apply_text(src)

    if args.check:
        _log(args.quiet, f"  [check] {target}: would report status={status}")
        return 0 if status in ("applied", "already") else 1

    if supp_src.is_file() and supp_src.resolve() != supp_staged.resolve():
        # Refresh this on every invocation, including container restarts where
        # kernel_warmup.py is already patched but the mounted supplement changed.
        atomic_write(supp_staged, supp_src.read_bytes())
        _log(args.quiet, f"  staged supplement -> {supp_staged}")

    if status == "already":
        _log(args.quiet, f"  [skip] {target} already patched ({MARKER})")
    elif status == "applied":
        atomic_write(target, new_src.encode())
        _log(args.quiet, f"  [ok] patched {target}")
    else:
        print(
            f"[FAIL] {target}: {status} — DSv4 production-dense autotune merge NOT "
            "installed. The vendored anchor in "
            "recipe/overlay/vllm/model_executor/warmup/kernel_warmup.py no longer "
            "matches this image; re-pin the anchor before relying on dense coverage.",
            file=sys.stderr,
        )
        return 1

    # Re-read from disk and prove the installed file is importable Python that
    # actually contains the merge hook. A substring grep is not enough.
    final = target.read_text()
    try:
        tree = ast.parse(final)
    except SyntaxError as exc:
        print(f"[FAIL] {target} is not valid Python after patching: {exc}", file=sys.stderr)
        return 1
    helpers = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    if "_dv4_prod_dense_merge" not in helpers:
        print(f"[FAIL] {target}: merge helper missing after patch", file=sys.stderr)
        return 1
    if "_dv4_prod_dense_merge(tune_results, cache_path)" not in final:
        print(f"[FAIL] {target}: merge is never called on the DSv4 path", file=sys.stderr)
        return 1
    if not supp_staged.is_file() and not supp_src.is_file():
        print("[FAIL] supplement is not readable from inside the container", file=sys.stderr)
        return 1

    _log(
        args.quiet,
        "[OK] hotfix-dsv4-autotune-prod-dense: merge hook installed on the DSv4 "
        "sparse-MLA decode path (leader merges pre-broadcast; all ranks load the "
        "same configs).",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
