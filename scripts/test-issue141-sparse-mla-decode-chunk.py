#!/usr/bin/env python3
"""CPU-only source-lock and row-coupling tests for issue #141 workaround."""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PATCHER_PATH = ROOT / "patches/hotfix-dsv4-issue141-sparse-mla-decode-chunk.py"
COMPOSE = ROOT / "docker-compose.dspark.yml"
START = ROOT / "start-deepseek-v4-flash-dspark.sh"

spec = importlib.util.spec_from_file_location("issue141_hotfix", PATCHER_PATH)
assert spec is not None and spec.loader is not None
hotfix = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hotfix
spec.loader.exec_module(hotfix)

# Independent exact source fixture from Anemll/dspark-vllm-gx10 revision
# 47503f8e38dadd4dededca798150db2619594fce:
# overlay/vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py
PINNED_OLD_METHOD = '''    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        extra_sparse_indices = None
        extra_sparse_lengths = None
        if not swa_only:
            if attn_metadata is None:
                raise RuntimeError(
                    "Sparse MLA metadata is required for compressed layers."
                )
            if swa_metadata.is_valid_token is None:
                raise RuntimeError(
                    "SWA validity metadata is required for compressed layers."
                )
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                if self.topk_indices_buffer is None:
                    raise RuntimeError(
                        "C4A decode requires top-k indices from the indexer."
                    )
                block_size = attn_metadata.block_size // self.compress_ratio
                global_indices, extra_sparse_lengths = (
                    compute_global_topk_indices_and_lens(
                        self.topk_indices_buffer[:num_decode_tokens],
                        swa_metadata.token_to_req_indices,
                        attn_metadata.block_table[:num_decodes],
                        block_size,
                        is_valid,
                    )
                )
                extra_sparse_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                extra_sparse_indices = attn_metadata.c128a_global_decode_topk_indices
                extra_sparse_lengths = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        swa_indices = self._pad_decode_sparse_indices(swa_indices)
        q = self._prepare_query(q, output)
        swa_cache = self._as_sparse_cache(self.swa_cache_layer.kv_cache)
        extra_cache = self._as_sparse_cache(kv_cache) if kv_cache is not None else None
        if extra_cache is not None and extra_sparse_indices is None:
            raise RuntimeError(
                "Compressed sparse MLA decode requires compressed sparse indices."
            )
        flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=swa_cache,
            workspace_buffer=self._get_workspace(q.device),
            sparse_indices=swa_indices,
            compressed_kv_cache=extra_cache,
            out=output,
            bmm1_scale=self.scale,
            sinks=self.attn_sink,
            kv_layout="NHD",
            swa_topk_lens=swa_lens,
            extra_sparse_indices=extra_sparse_indices,
            extra_sparse_topk_lens=extra_sparse_lengths,
        )
'''

# Independent expected injected block. The behavior tests execute the exact
# production block only after this literal comparison and the frozen full-new
# digest prevent production edits from reshaping their own fixture.
EXPECTED_HOT_BLOCK = '''        workspace = self._get_workspace(q.device)
        if num_decode_tokens <= 64:
            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=q,
                swa_kv_cache=swa_cache,
                workspace_buffer=workspace,
                sparse_indices=swa_indices,
                compressed_kv_cache=extra_cache,
                out=output,
                bmm1_scale=self.scale,
                sinks=self.attn_sink,
                kv_layout="NHD",
                swa_topk_lens=swa_lens,
                extra_sparse_indices=extra_sparse_indices,
                extra_sparse_topk_lens=extra_sparse_lengths,
            )
            return

        # [issue141-hotfix] SM120 DSv4 decode/prefill cutoff is 64 rows.
        for row_start in range(0, num_decode_tokens, 64):
            rows = slice(row_start, min(row_start + 64, num_decode_tokens))
            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=q[rows],
                swa_kv_cache=swa_cache,
                workspace_buffer=workspace,
                sparse_indices=swa_indices[rows],
                compressed_kv_cache=extra_cache,
                out=output[rows],
                bmm1_scale=self.scale,
                sinks=self.attn_sink,
                kv_layout="NHD",
                swa_topk_lens=swa_lens[rows],
                extra_sparse_indices=(
                    extra_sparse_indices[rows]
                    if extra_sparse_indices is not None
                    else None
                ),
                extra_sparse_topk_lens=(
                    extra_sparse_lengths[rows]
                    if extra_sparse_lengths is not None
                    else None
                ),
            )
'''

# Literal digests are filled from the independently reviewed bytes above and
# FlashInfer 0472b9b3f2fba11b463f8526f390297d52a8aad7 guard fragments. They are
# intentionally not derived from the production constants at assertion time.
FROZEN_DIGESTS = {
    "old-method": "48b021444736516453a63a9a3131b56ef758d7df13d6fb2ee628d837bc374948",
    "new-method": "fecc2d019725c43fa87b3596c9e686b9f86692ced6cf9aed95e57cb031b38e84",
    "core:decode-workspace-64-cutoff": "c28707d48b98a009b06691e54de3beab4c70f2e829f741e39a270a722c3f55fc",
    "core:sm120-adapter-call-shape": "8e6bedc655afe475e2503d29c6ca995e710ff6167336f2119db55f18f978c662",
    "core:sm120-segment-mapping": "cea7675c63488a8b3f77dd3909217b3f0b25f319a212a7804e607a19fdb00ed9",
    "core:sm120-runner-mapping": "d775fb9098ee21e955cb6da68b1b9fae17c0094508bd862e04a853c30ba9a2a8",
    "core:public-sm120-keyword-mapping": "145fac7d7f4e3c2c11ae2ec377b1e72b0cdc51b68263fe060b6d3d18c13f7afa",
    "sparse:decode-cutoff-definition": "8d4f88919650685c2e60cbb51b104fdd9cc92d16d8ce77aa6b2bff8c0beb8c75",
    "sparse:dsv4-dispatch-predicate": "2a10cbdb397d2939f39a8df919f4ec734af961e3700a72cb681c9b4da15a9095",
    "sparse:custom-op-mutation-contract": "3c2b24af3d528fb62e4806a040a6591e9ff63413f310e577ab8037e99249accc",
    "sparse:dsv4-decode-branch": "a33ff181ad6eb685fca0691d7f8db6948861e536d312cc9eafb9f9f2947cff09",
    "sparse:paged-orchestrator-fallback": "bed60113cee17c755297c8a3979163ddb79318885c010f7d09e4a8381b2e4cfb",
}

EXPECTED_KEYWORDS = (
    "query",
    "swa_kv_cache",
    "workspace_buffer",
    "sparse_indices",
    "compressed_kv_cache",
    "out",
    "bmm1_scale",
    "sinks",
    "kv_layout",
    "swa_topk_lens",
    "extra_sparse_indices",
    "extra_sparse_topk_lens",
)
ROW_COUPLED = (
    "query",
    "sparse_indices",
    "out",
    "swa_topk_lens",
    "extra_sparse_indices",
    "extra_sparse_topk_lens",
)
SHARED = (
    "swa_kv_cache",
    "workspace_buffer",
    "compressed_kv_cache",
    "bmm1_scale",
    "sinks",
)
ROW_MATRIX = {
    1: [(0, 1)],
    63: [(0, 63)],
    64: [(0, 64)],
    65: [(0, 64), (64, 65)],
    96: [(0, 64), (64, 96)],
    128: [(0, 64), (64, 128)],
    129: [(0, 64), (64, 128), (128, 129)],
    192: [(0, 64), (64, 128), (128, 192)],
    224: [(0, 64), (64, 128), (128, 192), (192, 224)],
    288: [(0, 64), (64, 128), (128, 192), (192, 256), (256, 288)],
    576: [(n, n + 64) for n in range(0, 576, 64)],
}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def adapter_source(method: str) -> str:
    return "class DeepseekV4FlashInferSM120Attention:\n" + method + "\nTAIL = 141\n"


def guard_source(guards: tuple[tuple[str, str], ...]) -> str:
    return "\n# fixture boundary\n".join(fragment for _, fragment in guards)


class FixtureMixin:
    def fixture(
        self,
        method: str = PINNED_OLD_METHOD,
        *,
        core_source: str | None = None,
        sparse_source: str | None = None,
    ) -> tuple[Path, Path, Path, Path]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        target = root / "flashinfer_sparse.py"
        core = root / "_core.py"
        sparse = root / "_sparse_mla_sm120.py"
        target.write_text(adapter_source(method), encoding="utf-8")
        target.chmod(0o640)
        core.write_text(
            core_source if core_source is not None else guard_source(hotfix.CORE_GUARDS),
            encoding="utf-8",
        )
        sparse.write_text(
            sparse_source
            if sparse_source is not None
            else guard_source(hotfix.SPARSE_GUARDS),
            encoding="utf-8",
        )
        return root, target, core, sparse

    def assert_rejected_without_write(
        self,
        target: Path,
        core: Path,
        sparse: Path,
    ) -> None:
        before = target.read_bytes()
        before_mode = stat.S_IMODE(target.stat().st_mode)
        with self.assertRaises(hotfix.PatchError):
            hotfix.patch_paths(target, core, sparse)
        self.assertEqual(target.read_bytes(), before)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), before_mode)


class SourceLockTest(FixtureMixin, unittest.TestCase):
    def test_independent_method_and_guard_digests(self):
        self.assertEqual(hotfix.OLD_METHOD, PINNED_OLD_METHOD)
        start = hotfix.NEW_METHOD.index("        workspace = self._get_workspace(q.device)\n")
        self.assertEqual(hotfix.NEW_METHOD[start:], EXPECTED_HOT_BLOCK)
        actual = {
            "old-method": sha256(PINNED_OLD_METHOD),
            "new-method": sha256(hotfix.NEW_METHOD),
        }
        actual.update(
            {f"core:{name}": sha256(fragment) for name, fragment in hotfix.CORE_GUARDS}
        )
        actual.update(
            {
                f"sparse:{name}": sha256(fragment)
                for name, fragment in hotfix.SPARSE_GUARDS
            }
        )
        self.assertEqual(actual, FROZEN_DIGESTS)

    def test_exact_old_applies_only_whole_method_and_preserves_mode(self):
        _, target, core, sparse = self.fixture()
        before = target.read_text(encoding="utf-8")
        result = hotfix.patch_paths(target, core, sparse)
        after = target.read_text(encoding="utf-8")
        self.assertEqual(result, "applied and verified")
        self.assertEqual(after, before.replace(PINNED_OLD_METHOD, hotfix.NEW_METHOD, 1))
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o640)
        self.assertEqual(hotfix.method_state(after), "new")

    def test_exact_new_is_byte_mode_and_inode_idempotent(self):
        _, target, core, sparse = self.fixture(hotfix.NEW_METHOD)
        before = target.read_bytes()
        before_stat = target.stat()
        result = hotfix.patch_paths(target, core, sparse)
        after_stat = target.stat()
        self.assertEqual(result, "already applied and verified")
        self.assertEqual(target.read_bytes(), before)
        self.assertEqual(after_stat.st_ino, before_stat.st_ino)
        self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)
        self.assertEqual(stat.S_IMODE(after_stat.st_mode), 0o640)

    def test_explicit_three_path_fixture_override(self):
        _, target, core, sparse = self.fixture()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = hotfix.main(["patcher", str(target), str(core), str(sparse)])
        self.assertEqual(rc, 0, stderr.getvalue())
        self.assertIn("applied and verified", stdout.getvalue())
        self.assertEqual(hotfix.method_state(target.read_text()), "new")

    def test_missing_target_or_guard_source_rejects(self):
        for missing in ("target", "core", "sparse"):
            with self.subTest(missing=missing):
                _, target, core, sparse = self.fixture()
                paths = {"target": target, "core": core, "sparse": sparse}
                paths[missing].unlink()
                survivors = {
                    name: path.read_bytes()
                    for name, path in paths.items()
                    if path.exists()
                }
                with self.assertRaises(hotfix.PatchError):
                    hotfix.patch_paths(target, core, sparse)
                for name, content in survivors.items():
                    self.assertEqual(paths[name].read_bytes(), content)

    def test_missing_duplicate_mixed_partial_and_drift_states_reject(self):
        drifted_keyword = PINNED_OLD_METHOD.replace(
            "            sinks=self.attn_sink,\n",
            "            seq_lens=None,\n            sinks=self.attn_sink,\n",
            1,
        )
        vanilla_without_anemll_pad = PINNED_OLD_METHOD.replace(
            "        swa_indices = self._pad_decode_sparse_indices(swa_indices)\n",
            "",
            1,
        )
        cases = {
            "missing": "    pass\n",
            "duplicate-old": PINNED_OLD_METHOD + PINNED_OLD_METHOD,
            "duplicate-new": hotfix.NEW_METHOD + hotfix.NEW_METHOD,
            "mixed": PINNED_OLD_METHOD + hotfix.NEW_METHOD,
            "partial-marker": PINNED_OLD_METHOD + "    " + hotfix.MARK + "\n",
            "changed-keyword-sm100-like": drifted_keyword,
            "changed-new-keyword-sm100-like": hotfix.NEW_METHOD.replace(
                "                sinks=self.attn_sink,\n",
                "                seq_lens=None,\n                sinks=self.attn_sink,\n",
                1,
            ),
            "vanilla-without-anemll-pad": vanilla_without_anemll_pad,
        }
        for name, method in cases.items():
            with self.subTest(name=name):
                _, target, core, sparse = self.fixture(method)
                self.assert_rejected_without_write(target, core, sparse)

    def test_flashinfer_drift_rejects_old_and_already_patched_targets(self):
        core_ok = guard_source(hotfix.CORE_GUARDS)
        sparse_ok = guard_source(hotfix.SPARSE_GUARDS)
        drifts = {
            "core-cutoff-65": (
                core_ok.replace("if num_tokens > 64:", "if num_tokens > 65:", 1),
                sparse_ok,
            ),
            "core-sm120-mapping": (
                core_ok.replace("out=out_for_sm120,", "out=out_for_sm120.clone(),", 1),
                sparse_ok,
            ),
            "core-public-keyword": (
                core_ok.replace(
                    "swa_topk_lens=swa_topk_lens,",
                    "sparse_topk_lens=swa_topk_lens,",
                    1,
                ),
                sparse_ok,
            ),
            "sparse-cutoff-65": (
                core_ok,
                sparse_ok.replace("_DECODE_MAX_TOKENS = 64", "_DECODE_MAX_TOKENS = 65", 1),
            ),
            "sparse-duplicate-cutoff": (
                core_ok,
                sparse_ok + "\n_DECODE_MAX_TOKENS = 64\n",
            ),
            "sparse-condition-removed": (
                core_ok,
                sparse_ok.replace(
                    "num_tokens <= _DECODE_MAX_TOKENS",
                    "num_tokens < _DECODE_MAX_TOKENS",
                    1,
                ),
            ),
            "cache-declared-mutable": (
                core_ok,
                sparse_ok.replace(
                    'mutates_args=("output", "out_lse", "mid_out", "mid_lse"),',
                    'mutates_args=("output", "out_lse", "mid_out", "mid_lse", "kv_cache"),',
                    1,
                ),
            ),
        }
        for drift, (core_source, sparse_source) in drifts.items():
            for state, method in (("old", PINNED_OLD_METHOD), ("new", hotfix.NEW_METHOD)):
                with self.subTest(drift=drift, state=state):
                    _, target, core, sparse = self.fixture(
                        method,
                        core_source=core_source,
                        sparse_source=sparse_source,
                    )
                    self.assert_rejected_without_write(target, core, sparse)

    def test_syntax_invalid_replacement_is_rejected_before_publication(self):
        _, target, core, sparse = self.fixture()
        bad_new = hotfix.NEW_METHOD.replace(
            "        workspace = self._get_workspace(q.device)\n",
            "        if (\n        workspace = self._get_workspace(q.device)\n",
            1,
        )
        with mock.patch.object(hotfix, "NEW_METHOD", bad_new):
            self.assert_rejected_without_write(target, core, sparse)

    def test_atomic_replace_failure_leaves_original(self):
        root, target, core, sparse = self.fixture()
        before = target.read_bytes()
        with mock.patch.object(hotfix.os, "replace", side_effect=OSError("injected")):
            with self.assertRaises(hotfix.PatchError):
                hotfix.patch_paths(target, core, sparse)
        self.assertEqual(target.read_bytes(), before)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o640)
        self.assertEqual([p for p in root.iterdir() if p.name.startswith(".flashinfer_sparse.py")], [])

    def test_post_write_failure_atomically_restores_original(self):
        root, target, core, sparse = self.fixture()
        before = target.read_bytes()
        with mock.patch.object(
            hotfix,
            "_verify_published",
            side_effect=hotfix.PatchError("injected post-write failure"),
        ):
            with self.assertRaises(hotfix.PatchError):
                hotfix.patch_paths(target, core, sparse)
        self.assertEqual(target.read_bytes(), before)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o640)
        self.assertEqual([p for p in root.iterdir() if p.name.startswith(".flashinfer_sparse.py")], [])


class Backing:
    def __init__(self, rows: int, width: int, *, output: bool = False):
        self.rows = tuple(range(rows))
        self.width = width
        self.views = 0
        self.writes = [0] * rows if output else None
        self.values = [None] * rows if output else None


class FakeTensor:
    """Row views share backing; no tensor package or data copy is involved."""

    def __init__(self, backing: Backing, rows: tuple[int, ...] | None = None):
        self.backing = backing
        self.rows = backing.rows if rows is None else rows
        self.shape = (len(self.rows), backing.width)
        self.device = "fake-sm121a"

    def __getitem__(self, key: slice) -> "FakeTensor":
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise AssertionError(f"only contiguous row slices are allowed: {key!r}")
        self.backing.views += 1
        return FakeTensor(self.backing, self.rows[key])

    def write_rows(self) -> None:
        if self.backing.writes is None or self.backing.values is None:
            raise AssertionError("write attempted through a non-output tensor")
        for row in self.rows:
            self.backing.writes[row] += 1
            self.backing.values[row] = row

    def clone(self):  # pragma: no cover - a call is an immediate test failure
        raise AssertionError("clone allocation is forbidden")

    def contiguous(self):  # pragma: no cover - a call is an immediate test failure
        raise AssertionError("contiguous copy is forbidden")


class SelfStub:
    def __init__(self, workspace: object, scale: object, sinks: object):
        self.workspace = workspace
        self.scale = scale
        self.attn_sink = sinks
        self.workspace_calls = 0

    def _get_workspace(self, device: object) -> object:
        if device != "fake-sm121a":
            raise AssertionError(f"unexpected device {device!r}")
        self.workspace_calls += 1
        return self.workspace


class InjectedInnerFailure(RuntimeError):
    pass


class CallRecorder:
    def __init__(self, originals: dict[str, object], fail_call: int | None = None):
        self.originals = originals
        self.fail_call = fail_call
        self.calls: list[dict[str, object]] = []
        self.return_objects: list[object] = []

    def __call__(self, **kwargs):
        if tuple(kwargs) != EXPECTED_KEYWORDS:
            raise AssertionError(f"keyword drift: {tuple(kwargs)!r}")
        rows = kwargs["query"].rows
        for name in ROW_COUPLED:
            value = kwargs[name]
            if value is None:
                if name not in ("extra_sparse_indices", "extra_sparse_topk_lens"):
                    raise AssertionError(f"required row tensor {name} is None")
            elif value.rows != rows:
                raise AssertionError(f"row mismatch for {name}: {value.rows} != {rows}")
        for name in SHARED:
            if kwargs[name] is not self.originals[name]:
                raise AssertionError(f"shared argument {name} was replaced or sliced")
        if kwargs["kv_layout"] != "NHD":
            raise AssertionError("kv_layout changed")
        self.calls.append(kwargs)
        if self.fail_call == len(self.calls):
            raise InjectedInnerFailure(f"injected call {self.fail_call}")
        kwargs["out"].write_rows()
        returned = object()
        self.return_objects.append(returned)
        return returned


def compile_exact_hot_block():
    start = hotfix.NEW_METHOD.index("        workspace = self._get_workspace(q.device)\n")
    block = hotfix.NEW_METHOD[start:]
    source = (
        "def run(num_decode_tokens, q, swa_cache, swa_indices, extra_cache, "
        "output, swa_lens, extra_sparse_indices, extra_sparse_lengths, self):\n"
        + textwrap.indent(textwrap.dedent(block), "    ")
    )
    namespace: dict[str, object] = {}
    exec(compile(source, "<issue141-injected-block>", "exec"), namespace)
    return namespace["run"]


class ChunkSemanticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = staticmethod(compile_exact_hot_block())

    def execute_case(
        self,
        rows: int,
        primary_width: int,
        extra_width: int | None,
        *,
        fail_call: int | None = None,
    ):
        q = FakeTensor(Backing(rows, 512))
        sparse = FakeTensor(Backing(rows, primary_width))
        output = FakeTensor(Backing(rows, 512, output=True))
        swa_lens = FakeTensor(Backing(rows, 1))
        if extra_width is None:
            extra_indices = None
            extra_lens = None
            extra_cache = None
        else:
            extra_indices = FakeTensor(Backing(rows, extra_width))
            extra_lens = FakeTensor(Backing(rows, 1))
            extra_cache = object()
        swa_cache = object()
        workspace = object()
        scale = object()
        sinks = object()
        self_obj = SelfStub(workspace, scale, sinks)
        originals = {
            "query": q,
            "sparse_indices": sparse,
            "out": output,
            "swa_topk_lens": swa_lens,
            "extra_sparse_indices": extra_indices,
            "extra_sparse_topk_lens": extra_lens,
            "swa_kv_cache": swa_cache,
            "workspace_buffer": workspace,
            "compressed_kv_cache": extra_cache,
            "bmm1_scale": scale,
            "sinks": sinks,
        }
        recorder = CallRecorder(originals, fail_call)
        self.runner.__globals__["flashinfer_trtllm_batch_decode_sparse_mla_dsv4"] = recorder
        args = (
            rows,
            q,
            swa_cache,
            sparse,
            extra_cache,
            output,
            swa_lens,
            extra_indices,
            extra_lens,
            self_obj,
        )
        return args, originals, recorder, self_obj, output

    def test_call_ast_has_exact_sm120_keywords_and_explicit_slice_inventory(self):
        tree = ast.parse("class Target:\n" + hotfix.NEW_METHOD)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "flashinfer_trtllm_batch_decode_sparse_mla_dsv4"
        ]
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertEqual(tuple(kw.arg for kw in call.keywords), EXPECTED_KEYWORDS)
            self.assertEqual(call.args, [])
        fast, chunked = sorted(calls, key=lambda call: call.lineno)
        fast_values = {kw.arg: ast.unparse(kw.value) for kw in fast.keywords}
        chunk_values = {kw.arg: ast.unparse(kw.value) for kw in chunked.keywords}
        self.assertEqual(
            {name: fast_values[name] for name in ROW_COUPLED},
            {
                "query": "q",
                "sparse_indices": "swa_indices",
                "out": "output",
                "swa_topk_lens": "swa_lens",
                "extra_sparse_indices": "extra_sparse_indices",
                "extra_sparse_topk_lens": "extra_sparse_lengths",
            },
        )
        self.assertEqual(chunk_values["query"], "q[rows]")
        self.assertEqual(chunk_values["sparse_indices"], "swa_indices[rows]")
        self.assertEqual(chunk_values["out"], "output[rows]")
        self.assertEqual(chunk_values["swa_topk_lens"], "swa_lens[rows]")
        self.assertIn("extra_sparse_indices[rows]", chunk_values["extra_sparse_indices"])
        self.assertIn("else None", chunk_values["extra_sparse_indices"])
        self.assertIn("extra_sparse_lengths[rows]", chunk_values["extra_sparse_topk_lens"])
        self.assertIn("else None", chunk_values["extra_sparse_topk_lens"])
        for name in SHARED:
            self.assertNotIn("[rows]", chunk_values[name])
        for forbidden in (".clone(", ".contiguous(", "torch.cat(", ".cat("):
            self.assertNotIn(forbidden, EXPECTED_HOT_BLOCK)

    def test_required_row_matrix_and_identity_contracts(self):
        # Extra width 0 is the SWA-only/None shape; 512 and 8192 cover the
        # compressed C4A/C128A plumbing while both primary widths are exercised.
        configs = ((128, None), (512, None), (128, 512), (512, 8192))
        for rows, expected_spans in ROW_MATRIX.items():
            for primary_width, extra_width in configs:
                with self.subTest(
                    rows=rows,
                    primary_width=primary_width,
                    extra_width=extra_width or 0,
                ):
                    args, originals, recorder, self_obj, output = self.execute_case(
                        rows, primary_width, extra_width
                    )
                    result = self.runner(*args)
                    spans = [
                        (call["query"].rows[0], call["query"].rows[-1] + 1)
                        for call in recorder.calls
                    ]
                    self.assertEqual(spans, expected_spans)
                    self.assertTrue(all(end - start <= 64 for start, end in spans))
                    self.assertEqual(
                        [row for start, end in spans for row in range(start, end)],
                        list(range(rows)),
                    )
                    self.assertEqual(self_obj.workspace_calls, 1)
                    self.assertIsNone(result)
                    self.assertEqual(output.backing.writes, [1] * rows)
                    self.assertEqual(output.backing.values, list(range(rows)))
                    if rows <= 64:
                        self.assertEqual(len(recorder.calls), 1)
                        for name in ROW_COUPLED:
                            self.assertIs(recorder.calls[0][name], originals[name])
                    else:
                        for call in recorder.calls:
                            for name in ROW_COUPLED:
                                if originals[name] is None:
                                    self.assertIsNone(call[name])
                                else:
                                    self.assertIs(call[name].backing, originals[name].backing)
                                    self.assertIsNot(call[name], originals[name])
                    # The fake inner returns replacement objects deliberately;
                    # the adapter ignores every return and fills the original out.
                    self.assertTrue(recorder.return_objects)
                    self.assertTrue(all(obj is not output for obj in recorder.return_objects))

    def test_second_chunk_failure_propagates_without_retry_or_reorder(self):
        args, _, recorder, self_obj, output = self.execute_case(
            192, 512, 8192, fail_call=2
        )
        with self.assertRaisesRegex(InjectedInnerFailure, "injected call 2"):
            self.runner(*args)
        self.assertEqual(
            [(call["query"].rows[0], call["query"].rows[-1] + 1) for call in recorder.calls],
            [(0, 64), (64, 128)],
        )
        self.assertEqual(output.backing.writes[:64], [1] * 64)
        self.assertEqual(output.backing.writes[64:], [0] * 128)
        self.assertEqual(self_obj.workspace_calls, 1)


class WiringContractTest(unittest.TestCase):
    def test_default_zero_exact_one_and_read_only_worker_wiring(self):
        compose = COMPOSE.read_text(encoding="utf-8")
        start = START.read_text(encoding="utf-8")
        self.assertIn(
            'DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK: "${DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK:-0}"',
            compose,
        )
        self.assertIn(
            '${DSPARK_ISSUE141_HOTFIX:-./patches/hotfix-dsv4-issue141-sparse-mla-decode-chunk.py}:/opt/hotfix-dsv4-issue141-sparse-mla-decode-chunk.py:ro',
            compose,
        )
        self.assertIn(
            'if [ "$${DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK:-0}" = "1" ]; then python3 /opt/hotfix-dsv4-issue141-sparse-mla-decode-chunk.py || exit 1; fi;',
            compose,
        )
        self.assertIn(
            'scp "$DSPARK_ISSUE141_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-dsv4-issue141-sparse-mla-decode-chunk.py"',
            start,
        )
        self.assertIn(
            "DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK='$DSPARK_ISSUE141_EFFECTIVE'",
            start,
        )
        self.assertIn(
            "DSPARK_ISSUE141_HOTFIX='./patches/hotfix-dsv4-issue141-sparse-mla-decode-chunk.py'",
            start,
        )


if __name__ == "__main__":
    unittest.main()
