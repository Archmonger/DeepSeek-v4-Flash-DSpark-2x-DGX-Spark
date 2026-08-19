#!/usr/bin/env python3
"""Unit tests for the DSv4 production-dense autotune coverage hotfix (no GPU).

Covers three things the review of PR #83 found unproven:

1. the patcher hooks the DSv4 sparse-MLA decode path that actually owns the
   cache, and its anchor is pinned to the vendored kernel_warmup.py;
2. the injected merge helper behaves correctly, including refusing to load a
   supplement tuned for a different build;
3. both ranks end up loading an identical config set.

The helper closes over `logger` and `Path` from kernel_warmup's module scope, so
it is exercised by extracting the injected region and exec'ing it against stubs.
That keeps the test free of torch / vllm / flashinfer imports.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-dsv4-autotune-prod-dense.py"
SUPPLEMENT = ROOT / "patches" / "dsv4-autotune-prod-dense-supp.json"
VENDORED = ROOT / "recipe/overlay/vllm/model_executor/warmup/kernel_warmup.py"


def _load_hotfix():
    spec = importlib.util.spec_from_file_location("dv4_prod_dense", HOTFIX)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Logger:
    """Minimal stand-in for vllm's module logger."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def _add(self, level: str, msg: str, *args) -> None:
        self.records.append((level, msg % args if args else msg))

    def warning(self, msg, *args) -> None:
        self._add("WARNING", msg, *args)

    def info(self, msg, *args) -> None:
        self._add("INFO", msg, *args)

    def text(self) -> str:
        return "\n".join(f"{lvl}: {msg}" for lvl, msg in self.records)

    def levels(self) -> set[str]:
        return {lvl for lvl, _ in self.records}


MOD = _load_hotfix()


def _extract_helper(patched_src: str, fake_module_file: Path):
    """exec the injected helper region with stub globals; return (ns, logger)."""
    start = patched_src.index("# --- dv4-prod-dense-merge (MiaAI")
    end = patched_src.index("# --- end dv4-prod-dense-merge ---")
    logger = _Logger()
    ns = {"Path": Path, "logger": logger, "__file__": str(fake_module_file)}
    exec(compile(patched_src[start:end], "<injected-helper>", "exec"), ns)
    return ns, logger


class AnchorPinningTest(unittest.TestCase):
    """The pinned anchor must match the vendored kernel_warmup.py exactly."""

    def setUp(self):
        self.src = VENDORED.read_text()

    def test_call_anchor_present_and_unique(self):
        self.assertEqual(
            self.src.count(MOD.ANCHOR_OLD),
            1,
            "pinned DSv4 call anchor must appear exactly once in the vendored file",
        )

    def test_helper_anchor_present_and_unique(self):
        self.assertEqual(self.src.count(MOD.HELPER_ANCHOR), 1)

    def test_anchor_is_scoped_to_the_dsv4_function(self):
        # The narrow read+broadcast block is NOT unique: the generic
        # flashinfer_autotune() carries a byte-identical copy. Patching that one
        # would tune the wrong cache, so the anchor must stay DSv4-scoped.
        narrow = (
            "    tune_results: bytes | None = None\n"
            "    if is_leader and cache_path.exists():\n"
            '        with open(cache_path, "rb") as f:\n'
            "            tune_results = f.read()\n"
            "\n"
            "    tune_results = world.broadcast_object(tune_results, src=0)\n"
        )
        self.assertGreater(self.src.count(narrow), 1, "narrow block should be ambiguous")
        idx = self.src.index(MOD.ANCHOR_OLD)
        dsv4 = self.src.rfind("def _deepseek_v4_sparse_mla_decode_autotune", 0, idx)
        generic = self.src.rfind("def flashinfer_autotune", 0, idx)
        self.assertGreater(dsv4, generic, "anchor must sit inside the DSv4 function")

    def test_prereqs_present_in_vendored_file(self):
        for prereq in MOD.PREREQS:
            self.assertIn(prereq, self.src)

    def test_merge_lands_before_the_broadcast(self):
        patched, status = MOD.apply_text(self.src)
        self.assertEqual(status, "applied")
        call = patched.index("_dv4_prod_dense_merge(tune_results, cache_path)")
        bcast = patched.index("tune_results = world.broadcast_object(tune_results, src=0)")
        self.assertLess(
            call,
            bcast,
            "merge must happen before broadcast_object or rank>0 gets unmerged bytes",
        )

    def test_generic_autotune_path_untouched(self):
        patched, _ = MOD.apply_text(self.src)
        self.assertEqual(patched.count("_dv4_prod_dense_merge(tune_results, cache_path)"), 1)
        # The generic function must still be byte-identical.
        def generic(text: str) -> str:
            return text[text.index("def flashinfer_autotune(runner"):]
        self.assertEqual(generic(patched), generic(self.src))


class ApplyTextTest(unittest.TestCase):
    def setUp(self):
        self.src = VENDORED.read_text()

    def test_applies_to_vendored_source(self):
        out, status = MOD.apply_text(self.src)
        self.assertEqual(status, "applied")
        self.assertIn(MOD.MARKER, out)
        ast.parse(out)

    def test_output_is_valid_python(self):
        out, _ = MOD.apply_text(self.src)
        tree = ast.parse(out)
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        self.assertIn("_dv4_prod_dense_merge", names)
        self.assertIn("_deepseek_v4_sparse_mla_decode_autotune", names)

    def test_idempotent(self):
        once, _ = MOD.apply_text(self.src)
        twice, status = MOD.apply_text(once)
        self.assertEqual(status, "already")
        self.assertEqual(once, twice)
        self.assertEqual(once.count("_dv4_prod_dense_merge(tune_results, cache_path)"), 1)

    def test_anchor_missing_is_reported_not_guessed(self):
        broken = self.src.replace(
            "            tune_results = f.read()\n",
            '            tune_results = f.read()  # upstream drift\n',
        )
        out, status = MOD.apply_text(broken)
        self.assertEqual(status, "anchor-missing")
        self.assertEqual(out, broken, "source must be left untouched on anchor drift")

    def test_anchor_ambiguity_is_refused(self):
        doubled = self.src + "\n\n" + self.src
        _, status = MOD.apply_text(doubled)
        self.assertIn(status, ("anchor-ambiguous", "helper-anchor-ambiguous"))

    def test_prereq_missing_is_refused(self):
        stripped = self.src.replace("logger = init_logger(", "logger = _make_logger(")
        _, status = MOD.apply_text(stripped)
        self.assertEqual(status, "prereq-missing")


class SupplementValidationTest(unittest.TestCase):
    def test_shipped_supplement_is_valid(self):
        parsed, reason = MOD.validate_supplement(SUPPLEMENT.read_text())
        self.assertIsNotNone(parsed, reason)

    def test_shipped_supplement_axes(self):
        data = json.loads(SUPPLEMENT.read_text())
        configs = [k for k in data if k != "_metadata"]
        self.assertEqual(len(configs), 24)
        tokens, dense = set(), set()
        for key in configs:
            op, _runner, shapes, extras = ast.literal_eval(key)
            self.assertEqual(op, "sparse_mla_sm120_decode_dsv4")
            tokens.add(shapes[0][0])
            dense.add(extras[2])
        self.assertEqual(tokens, {1, 4, 8, 16, 32, 64})
        self.assertEqual(dense, {256, 1024, 2048, 8192})

    def test_rejects_garbage(self):
        for bad in ("{", "[]", "null", "{}", '{"_metadata": {}}', '{"a": 1}'):
            parsed, reason = MOD.validate_supplement(bad)
            self.assertIsNone(parsed, f"{bad!r} should be rejected, got {reason}")


class MergeBehaviourTest(unittest.TestCase):
    """Exercise the injected helper the way vLLM would call it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.pkg = root / "vllm"
        (self.pkg / "model_executor" / "warmup").mkdir(parents=True)
        self.module_file = self.pkg / "model_executor" / "warmup" / "kernel_warmup.py"
        self.patched, status = MOD.apply_text(VENDORED.read_text())
        assert status == "applied"
        self.supp = json.loads(SUPPLEMENT.read_text())
        (self.pkg / MOD.SUPP_BASENAME).write_text(json.dumps(self.supp))
        self.cache = root / "cache" / "autotune_configs.json"
        self.cache.parent.mkdir(parents=True)
        self.addCleanup(self.tmp.cleanup)

    def _helper(self):
        ns, logger = _extract_helper(self.patched, self.module_file)
        return ns["_dv4_prod_dense_merge"], logger

    def _base(self, meta=None, extra=None):
        base = {
            "('sparse_mla_sm120_decode_dsv4', 'boot', ((1, 32, 512),), (True, True, 512, True))": [
                "SparseMlaDecodeV3Runner",
                7,
            ],
            "_metadata": copy.deepcopy(meta if meta is not None else self.supp["_metadata"]),
        }
        if extra:
            base.update(extra)
        return base

    def test_happy_path_merges_and_persists(self):
        merge, logger = self._helper()
        base = self._base()
        raw = json.dumps(base).encode()
        self.cache.write_bytes(raw)

        out = merge(raw, self.cache)
        merged = json.loads(out)

        self.assertEqual(len([k for k in merged if k != "_metadata"]), 25)
        for key in (k for k in self.supp if k != "_metadata"):
            self.assertIn(key, merged)
        self.assertEqual(merged["_metadata"], base["_metadata"])
        # Leader's own on-disk cache must also be merged: every rank loads
        # configs from disk, and upstream never rewrites the leader's file.
        self.assertEqual(json.loads(self.cache.read_bytes()), merged)
        self.assertNotIn("WARNING", logger.levels(), logger.text())

    def test_both_ranks_load_identical_configs(self):
        """Leader merge -> broadcast bytes -> rank>0 write -> same config set."""
        merge, _ = self._helper()
        raw = json.dumps(self._base()).encode()
        self.cache.write_bytes(raw)

        broadcast_payload = merge(raw, self.cache)          # leader, pre-broadcast
        leader_on_disk = self.cache.read_bytes()            # what the leader loads

        rank1_cache = Path(self.tmp.name) / "rank1" / "autotune_configs.json"
        rank1_cache.parent.mkdir(parents=True)
        rank1_cache.write_bytes(broadcast_payload)          # upstream rank>0 write

        self.assertEqual(json.loads(leader_on_disk), json.loads(rank1_cache.read_bytes()))

    def test_boot_tuned_entry_wins_collision(self):
        merge, _ = self._helper()
        clashing = sorted(k for k in self.supp if k != "_metadata")[0]
        base = self._base(extra={clashing: ["SparseMlaDecodeV3Runner", 1234]})
        raw = json.dumps(base).encode()
        self.cache.write_bytes(raw)

        merged = json.loads(merge(raw, self.cache))
        self.assertEqual(merged[clashing], ["SparseMlaDecodeV3Runner", 1234])
        self.assertNotEqual(merged[clashing], self.supp[clashing])

    def test_metadata_mismatch_skips_loudly(self):
        merge, logger = self._helper()
        meta = copy.deepcopy(self.supp["_metadata"])
        meta["flashinfer_version"] = "0.7.0"
        raw = json.dumps(self._base(meta=meta)).encode()
        self.cache.write_bytes(raw)

        out = merge(raw, self.cache)
        self.assertEqual(out, raw, "mismatched build must not be merged")
        self.assertEqual(self.cache.read_bytes(), raw, "cache must be left untouched")
        self.assertIn("WARNING", logger.levels())
        self.assertIn("flashinfer_version", logger.text())
        self.assertIn("different build", logger.text())

    def test_gpu_mismatch_skips(self):
        merge, logger = self._helper()
        meta = copy.deepcopy(self.supp["_metadata"])
        meta["gpu"] = "NVIDIA H100"
        raw = json.dumps(self._base(meta=meta)).encode()
        out = merge(raw, self.cache)
        self.assertEqual(out, raw)
        self.assertIn("gpu", logger.text())

    def test_missing_metadata_skips(self):
        merge, logger = self._helper()
        base = self._base()
        del base["_metadata"]
        raw = json.dumps(base).encode()
        out = merge(raw, self.cache)
        self.assertEqual(out, raw)
        self.assertIn("_metadata", logger.text())

    def test_already_covered_is_a_noop(self):
        merge, logger = self._helper()
        base = self._base()
        base.update({k: v for k, v in self.supp.items() if k != "_metadata"})
        raw = json.dumps(base).encode()
        self.cache.write_bytes(raw)

        out = merge(raw, self.cache)
        self.assertEqual(out, raw)
        self.assertIn("already present", logger.text())
        self.assertNotIn("WARNING", logger.levels())

    def test_missing_supplement_skips(self):
        (self.pkg / MOD.SUPP_BASENAME).unlink()
        merge, logger = self._helper()
        raw = json.dumps(self._base()).encode()
        out = merge(raw, self.cache)
        self.assertEqual(out, raw)
        self.assertIn("not found", logger.text())

    def test_corrupt_supplement_skips(self):
        (self.pkg / MOD.SUPP_BASENAME).write_text("{ not json")
        merge, logger = self._helper()
        raw = json.dumps(self._base()).encode()
        out = merge(raw, self.cache)
        self.assertEqual(out, raw)
        self.assertIn("WARNING", logger.levels())

    def test_corrupt_boot_cache_skips(self):
        merge, logger = self._helper()
        out = merge(b"{ truncated", self.cache)
        self.assertEqual(out, b"{ truncated")
        self.assertIn("WARNING", logger.levels())

    def test_unwritable_cache_falls_back_to_unmerged(self):
        merge, logger = self._helper()
        raw = json.dumps(self._base()).encode()
        missing_dir = Path(self.tmp.name) / "nope" / "autotune_configs.json"
        out = merge(raw, missing_dir)
        self.assertEqual(out, raw, "must not claim a merge it could not persist")
        self.assertIn("WARNING", logger.levels())


class PatcherCliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.pkg = root / "vllm"
        (self.pkg / "model_executor" / "warmup").mkdir(parents=True)
        (self.pkg / "model_executor" / "warmup" / "kernel_warmup.py").write_text(
            VENDORED.read_text()
        )
        self.addCleanup(self.tmp.cleanup)

    def test_apply_then_reapply(self):
        self.assertEqual(
            MOD.main(["--vllm-root", str(self.pkg), "--supplement", str(SUPPLEMENT), "-q"]), 0
        )
        # supplement staged inside the package so it survives the mount going away
        self.assertTrue((self.pkg / MOD.SUPP_BASENAME).is_file())
        self.assertEqual(
            MOD.main(["--vllm-root", str(self.pkg), "--supplement", str(SUPPLEMENT), "-q"]), 0
        )

    def test_check_mode_writes_nothing(self):
        target = self.pkg / "model_executor" / "warmup" / "kernel_warmup.py"
        before = target.read_text()
        rc = MOD.main(
            ["--vllm-root", str(self.pkg), "--supplement", str(SUPPLEMENT), "--check", "-q"]
        )
        self.assertEqual(rc, 0)
        self.assertEqual(target.read_text(), before)

    def test_missing_target_fails(self):
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        self.assertEqual(
            MOD.main(["--vllm-root", str(empty), "--supplement", str(SUPPLEMENT), "-q"]), 1
        )

    def test_anchor_drift_fails_loudly(self):
        target = self.pkg / "model_executor" / "warmup" / "kernel_warmup.py"
        target.write_text(
            target.read_text().replace(
                "            tune_results = f.read()\n",
                "            tune_results = f.read()  # drift\n",
            )
        )
        self.assertEqual(
            MOD.main(["--vllm-root", str(self.pkg), "--supplement", str(SUPPLEMENT), "-q"]),
            1,
            "anchor drift must fail loudly instead of silently not patching",
        )

    def test_corrupt_supplement_fails_closed(self):
        bad = Path(self.tmp.name) / "bad.json"
        bad.write_text('{"_metadata": {}}')
        self.assertEqual(
            MOD.main(["--vllm-root", str(self.pkg), "--supplement", str(bad), "-q"]), 1
        )
        target = self.pkg / "model_executor" / "warmup" / "kernel_warmup.py"
        self.assertNotIn(MOD.MARKER, target.read_text())


if __name__ == "__main__":
    unittest.main()
