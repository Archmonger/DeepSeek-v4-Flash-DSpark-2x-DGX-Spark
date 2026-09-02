"""Issue #175 follow-up: the MoE gates must not host-sync once per layer.

``token_routing_kind`` ends in ``.sum().item()``. It used to run in every MoE
gate (43 per forward, and prefill is never CUDA-graphed, so text prompts paid
it too). The batch is now classified once in ``embed_input_ids`` and carried on
vLLM's per-step ``ForwardContext``.

Run inside the container:
  python3 tests/test_issue175_routing_kind_once.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patches"))

from vision_exp import apply as ap  # noqa: E402
from vision_exp import image_processor as ip  # noqa: E402
from vllm.forward_context import override_forward_context  # noqa: E402

ID = ip.IMAGE_TOKEN_ID
N_MOE_LAYERS = 43

_real_item = torch.Tensor.item
_counter = {"n": 0}


def _counting_item(self, *a, **kw):
    _counter["n"] += 1
    return _real_item(self, *a, **kw)


torch.Tensor.item = _counting_item


def syncs():
    return _counter["n"]


def reset():
    _counter["n"] = 0
    ip.set_pending_routing_kind(None)


FAILURES = []


def check(name, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    if not ok:
        FAILURES.append(name)


def fake_ctx():
    return types.SimpleNamespace(tag="forward-context")


def embed_fn():
    """The real patched embed_input_ids over a stub base embedding."""

    def orig(self, input_ids):
        return torch.zeros((int(input_ids.numel()), 4), dtype=torch.float32)

    return ap.make_embed_input_ids(orig)


def gate_reads(ids, n=N_MOE_LAYERS):
    """Simulate the N MoE gates of one forward reading the routing kind."""
    return {ip.current_routing_kind(ids) for _ in range(n)}


# --------------------------------------------------------------------------
print("1. text batch: embed + 43 gate reads")
reset()
embed = embed_fn()
ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
with override_forward_context(None):
    out = embed(None, ids, multimodal_embeddings=None, is_multimodal=None)
ctx = fake_ctx()
with override_forward_context(ctx):
    kinds = gate_reads(ids)
check("text kind", kinds, {"text"})
check("text .item() count", syncs(), 0)
check("text embed shape", tuple(out.shape), (6, 4))

# --------------------------------------------------------------------------
print("2. all-image batch: embed + 43 gate reads")
reset()
embed = embed_fn()
ids = torch.full((8,), ID, dtype=torch.long)
mm = [torch.ones((8, 4), dtype=torch.float32)]
is_mm = torch.ones(8, dtype=torch.bool)
with override_forward_context(None):
    embed(None, ids, multimodal_embeddings=mm, is_multimodal=is_mm)
ctx = fake_ctx()
with override_forward_context(ctx):
    kinds = gate_reads(ids)
check("image kind", kinds, {"image"})
check("image .item() count <= 1", syncs() <= 1, True)
check("image .item() count", syncs(), 1)

# --------------------------------------------------------------------------
print("3. mixed batch: embed + 43 gate reads")
reset()
embed = embed_fn()
ids = torch.tensor([7, ID, ID, ID, 9], dtype=torch.long)
mm = [torch.ones((3, 4), dtype=torch.float32)]
is_mm = torch.tensor([False, True, True, True, False])
with override_forward_context(None):
    embed(None, ids, multimodal_embeddings=mm, is_multimodal=is_mm)
ctx = fake_ctx()
with override_forward_context(ctx):
    kinds = gate_reads(ids)
check("mixed kind", kinds, {"mixed"})
check("mixed .item() count", syncs(), 1)

# --------------------------------------------------------------------------
print("4. empty mm list is text (encoder ran but produced nothing)")
reset()
embed = embed_fn()
ids = torch.tensor([1, 2, 3], dtype=torch.long)
with override_forward_context(None):
    embed(None, ids, multimodal_embeddings=[], is_multimodal=None)
ctx = fake_ctx()
with override_forward_context(ctx):
    kinds = gate_reads(ids)
check("empty-mm kind", kinds, {"text"})
check("empty-mm .item() count", syncs(), 0)

# --------------------------------------------------------------------------
print("5. fallback: no forward context -> identical to token_routing_kind")
for name, t in [
    ("text", torch.tensor([1, 2, 3])),
    ("image", torch.full((4,), ID)),
    ("mixed", torch.tensor([1, ID, 2])),
    ("empty", torch.tensor([], dtype=torch.long)),
    ("none", None),
    ("list", [1, ID, 3]),
]:
    reset()
    with override_forward_context(None):
        got = ip.current_routing_kind(t)
    want = ip.token_routing_kind(t)
    check(f"fallback {name}", got, want)

# --------------------------------------------------------------------------
print("6. fallback inside a context with no pending kind (dummy/warmup run)")
reset()
ids = torch.tensor([1, ID, 2], dtype=torch.long)
ctx = fake_ctx()
with override_forward_context(ctx):
    kinds = gate_reads(ids)
check("no-pending kind", kinds, {"mixed"})
check("no-pending syncs (memoized after 1st gate)", syncs(), 1)

# --------------------------------------------------------------------------
print("7. per-step isolation: a new ForwardContext never reuses the old kind")
reset()
embed = embed_fn()
img_ids = torch.full((4,), ID, dtype=torch.long)
with override_forward_context(None):
    embed(
        None,
        img_ids,
        multimodal_embeddings=[torch.ones((4, 4))],
        is_multimodal=torch.ones(4, dtype=torch.bool),
    )
ctx_a = fake_ctx()
with override_forward_context(ctx_a):
    a = ip.current_routing_kind(img_ids)
txt_ids = torch.tensor([1, 2, 3], dtype=torch.long)
with override_forward_context(None):
    embed(None, txt_ids, multimodal_embeddings=None, is_multimodal=None)
ctx_b = fake_ctx()
with override_forward_context(ctx_b):
    b = ip.current_routing_kind(txt_ids)
check("step A kind", a, "image")
check("step B kind (fresh ctx)", b, "text")
check("ctx A attr", getattr(ctx_a, ip._ROUTING_KIND_CTX_ATTR, None), "image")
check("ctx B attr", getattr(ctx_b, ip._ROUTING_KIND_CTX_ATTR, None), "text")

# --------------------------------------------------------------------------
print("8. stale pending is one-shot: a second forward cannot reuse it")
reset()
ip.set_pending_routing_kind("image")
with override_forward_context(fake_ctx()):
    first = ip.current_routing_kind(torch.tensor([1, 2, 3]))
with override_forward_context(fake_ctx()):
    second = ip.current_routing_kind(torch.tensor([1, 2, 3]))
check("1st consumes pending", first, "image")
check("2nd falls back (not stale)", second, "text")

# --------------------------------------------------------------------------
print("9. CUDA-graph capture short-circuits before any kind resolution")
reset()
calls = {"orig": 0}


class _Gate:
    e_score_correction_bias_vl = torch.zeros(4)


class _Router:
    scoring_func = "sigmoid"
    top_k = 2
    renormalize = True
    routed_scaling_factor = 1.0
    num_fused_shared_experts = 0

    def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
        calls["orig"] += 1
        return ("orig", input_ids)


router = _Router()
ap._wrap_router_compute_routing(router, _Gate())
real_capturing = torch.cuda.is_current_stream_capturing
torch.cuda.is_current_stream_capturing = lambda: True
try:
    ip.set_pending_routing_kind("image")  # would route image if consulted
    img_ids = torch.full((4,), ID, dtype=torch.long)
    with override_forward_context(fake_ctx()):
        r = router._compute_routing(torch.zeros((4, 4)), torch.zeros((4, 4)), None, input_ids=img_ids)
    check("capturing -> orig路径", r[0], "orig")
    check("capturing .item() count", syncs(), 0)
    check("capturing did not consume pending", ip._PENDING_ROUTING_KIND, "image")
finally:
    torch.cuda.is_current_stream_capturing = real_capturing

# --------------------------------------------------------------------------
print("10. non-capturing text step: gate wrapper takes orig path, zero syncs")
reset()
calls["orig"] = 0
torch.cuda.is_current_stream_capturing = lambda: False
try:
    embed = embed_fn()
    txt = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    with override_forward_context(None):
        embed(None, txt, multimodal_embeddings=None, is_multimodal=None)
    with override_forward_context(fake_ctx()):
        for _ in range(N_MOE_LAYERS):
            router._compute_routing(torch.zeros((4, 4)), torch.zeros((4, 4)), None, input_ids=txt)
    check("text step orig calls", calls["orig"], N_MOE_LAYERS)
    check("text step .item() count", syncs(), 0)
finally:
    torch.cuda.is_current_stream_capturing = real_capturing

# --------------------------------------------------------------------------
print("11. fused_topk_bias_split_vl honours a precomputed kind (no re-scan)")
reset()
import vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router as ftb_mod

seen = {}


def _fake_ftb(**kw):
    seen.update(kw)
    return ("w", "ids")


_real_ftb = ftb_mod.fused_topk_bias
_real_curr = ap.current_routing_kind


def _boom(*a, **k):
    raise AssertionError("current_routing_kind must not run when kind is given")


ftb_mod.fused_topk_bias = _fake_ftb
ap.current_routing_kind = _boom
try:
    vl_bias = torch.zeros(4)
    r = ap.fused_topk_bias_split_vl(
        hidden_states=torch.zeros((2, 2)),
        gating_output=torch.zeros((2, 2)),
        scoring_func="sigmoid",
        e_score_correction_bias=torch.ones(4),
        e_score_correction_bias_vl=vl_bias,
        topk=1,
        renormalize=False,
        indices_type=None,
        input_tokens=torch.tensor([ID, ID]),
        hash_indices_table="HASH",
        routed_scaling_factor=1.0,
        kind="image",
    )
    check(
        "kind=image -> bias_vl used",
        seen["e_score_correction_bias"].data_ptr() == vl_bias.data_ptr(),
        True,
    )
    check("kind=image -> hash table skipped", seen["hash_indices_table"], None)
    check("kind=image -> no re-scan syncs", syncs(), 0)

    seen.clear()
    r = ap.fused_topk_bias_split_vl(
        hidden_states=torch.zeros((2, 2)),
        gating_output=torch.zeros((2, 2)),
        scoring_func="sigmoid",
        e_score_correction_bias=torch.ones(4),
        e_score_correction_bias_vl=vl_bias,
        topk=1,
        renormalize=False,
        indices_type=None,
        input_tokens=torch.tensor([1, 2]),
        hash_indices_table="HASH",
        routed_scaling_factor=1.0,
        kind="text",
    )
    check("kind=text -> text bias + hash table", seen["hash_indices_table"], "HASH")
    check("kind=text -> no re-scan syncs", syncs(), 0)
finally:
    ftb_mod.fused_topk_bias = _real_ftb
    ap.current_routing_kind = _real_curr

print("12. kind=None still resolves (backwards compatible call)")
reset()
with override_forward_context(None):
    k = None
    ftb_mod.fused_topk_bias = _fake_ftb
    try:
        seen.clear()
        ap.fused_topk_bias_split_vl(
            hidden_states=torch.zeros((2, 2)),
            gating_output=torch.zeros((2, 2)),
            scoring_func="sigmoid",
            e_score_correction_bias=torch.ones(4),
            e_score_correction_bias_vl=torch.zeros(4),
            topk=1,
            renormalize=False,
            indices_type=None,
            input_tokens=torch.tensor([1, 2, 3]),
            hash_indices_table="HASH",
            routed_scaling_factor=1.0,
        )
        check("kind=None text -> hash table kept", seen["hash_indices_table"], "HASH")
        check("kind=None text -> 1 fallback sync", syncs(), 1)
    finally:
        ftb_mod.fused_topk_bias = _real_ftb

print()
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILED -> {FAILURES}")
    sys.exit(1)
print("RESULT: ALL PASS")
