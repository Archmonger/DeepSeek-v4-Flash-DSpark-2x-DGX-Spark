"""Issue #175 follow-up: the MoE gates must not host-sync once per layer.

``token_routing_kind`` ends in ``.sum().item()``. It used to run in every MoE
gate (43 per forward, and prefill is never CUDA-graphed, so text prompts paid
it too). The batch is now classified once in ``embed_input_ids`` and carried on
vLLM's per-step ``ForwardContext``, bound to the model that published it.

Run inside the container:
  python3 tests/test_issue175_routing_kind_once.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
from torch import nn

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
    ip._WARNED.clear()


FAILURES = []


def check(name, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    if not ok:
        FAILURES.append(name)


def raises(name, exc_type, fn):
    try:
        fn()
    except exc_type as e:
        print(f"  [PASS] {name}: raised {type(e).__name__}")
        return
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {name}: raised {type(e).__name__}, want {exc_type.__name__}")
        FAILURES.append(name)
        return
    print(f"  [FAIL] {name}: no exception, want {exc_type.__name__}")
    FAILURES.append(name)


def fake_ctx():
    return types.SimpleNamespace(tag="forward-context")


class FakeLM:
    """Stands in for DeepseekV4ForCausalLM as the embed_input_ids receiver."""


def embed_fn():
    def orig(self, input_ids):
        return torch.zeros((int(input_ids.numel()), 4), dtype=torch.float32)

    return ap.make_embed_input_ids(orig)


def gate_reads(ids, owner_id, n=N_MOE_LAYERS):
    """Simulate the N MoE gates of one forward reading the routing kind."""
    return {ip.current_routing_kind(ids, owner_id=owner_id) for _ in range(n)}


# --------------------------------------------------------------------------
print("1. text batch: embed + 43 gate reads")
reset()
embed, lm = embed_fn(), FakeLM()
ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
with override_forward_context(None):
    out = embed(lm, ids, multimodal_embeddings=None, is_multimodal=None)
with override_forward_context(fake_ctx()):
    kinds = gate_reads(ids, id(lm))
check("text kind", kinds, {"text"})
check("text .item() count", syncs(), 0)
check("text embed shape", tuple(out.shape), (6, 4))

# --------------------------------------------------------------------------
print("2. all-image batch: embed + 43 gate reads")
reset()
embed, lm = embed_fn(), FakeLM()
ids = torch.full((8,), ID, dtype=torch.long)
with override_forward_context(None):
    embed(
        lm,
        ids,
        multimodal_embeddings=[torch.ones((8, 4))],
        is_multimodal=torch.ones(8, dtype=torch.bool),
    )
with override_forward_context(fake_ctx()):
    kinds = gate_reads(ids, id(lm))
check("image kind", kinds, {"image"})
check("image .item() count", syncs(), 1)

# --------------------------------------------------------------------------
print("3. mixed batch: embed + 43 gate reads")
reset()
embed, lm = embed_fn(), FakeLM()
ids = torch.tensor([7, ID, ID, ID, 9], dtype=torch.long)
with override_forward_context(None):
    embed(
        lm,
        ids,
        multimodal_embeddings=[torch.ones((3, 4))],
        is_multimodal=torch.tensor([False, True, True, True, False]),
    )
with override_forward_context(fake_ctx()):
    kinds = gate_reads(ids, id(lm))
check("mixed kind", kinds, {"mixed"})
check("mixed .item() count", syncs(), 1)

# --------------------------------------------------------------------------
print("4. empty mm list is text (encoder ran but produced nothing)")
reset()
embed, lm = embed_fn(), FakeLM()
ids = torch.tensor([1, 2, 3], dtype=torch.long)
with override_forward_context(None):
    embed(lm, ids, multimodal_embeddings=[], is_multimodal=None)
with override_forward_context(fake_ctx()):
    kinds = gate_reads(ids, id(lm))
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
    check(f"fallback {name}", got, ip.token_routing_kind(t))

# --------------------------------------------------------------------------
print("6. fallback inside a context with no pending kind (dummy/warmup run)")
reset()
ids = torch.tensor([1, ID, 2], dtype=torch.long)
with override_forward_context(fake_ctx()):
    kinds = gate_reads(ids, None)
check("no-pending kind", kinds, {"mixed"})
check("no-pending syncs (memoized after 1st gate)", syncs(), 1)

# --------------------------------------------------------------------------
print("7. per-step isolation: a new ForwardContext never reuses the old kind")
reset()
embed, lm = embed_fn(), FakeLM()
img_ids = torch.full((4,), ID, dtype=torch.long)
with override_forward_context(None):
    embed(
        lm,
        img_ids,
        multimodal_embeddings=[torch.ones((4, 4))],
        is_multimodal=torch.ones(4, dtype=torch.bool),
    )
ctx_a = fake_ctx()
with override_forward_context(ctx_a):
    a = ip.current_routing_kind(img_ids, owner_id=id(lm))
txt_ids = torch.tensor([1, 2, 3], dtype=torch.long)
with override_forward_context(None):
    embed(lm, txt_ids, multimodal_embeddings=None, is_multimodal=None)
ctx_b = fake_ctx()
with override_forward_context(ctx_b):
    b = ip.current_routing_kind(txt_ids, owner_id=id(lm))
check("step A kind", a, "image")
check("step B kind (fresh ctx)", b, "text")
check("ctx A attr", getattr(ctx_a, ip._ROUTING_KIND_CTX_ATTR, None), "image")
check("ctx B attr", getattr(ctx_b, ip._ROUTING_KIND_CTX_ATTR, None), "text")

# --------------------------------------------------------------------------
print("8. stale pending is one-shot: a second forward cannot reuse it")
reset()
ip.set_pending_routing_kind("image", 777)
with override_forward_context(fake_ctx()):
    first = ip.current_routing_kind(torch.tensor([1, 2, 3]), owner_id=777)
with override_forward_context(fake_ctx()):
    second = ip.current_routing_kind(torch.tensor([1, 2, 3]), owner_id=777)
check("1st consumes pending", first, "image")
check("2nd falls back (not stale)", second, "text")

# --------------------------------------------------------------------------
print("9. DSpark drafter cannot eat the target's pending kind (owner stamp)")
reset()
target, drafter = FakeLM(), FakeLM()
embed = embed_fn()
img_ids = torch.full((4,), ID, dtype=torch.long)
with override_forward_context(None):
    embed(
        target,
        img_ids,
        multimodal_embeddings=[torch.ones((4, 4))],
        is_multimodal=torch.ones(4, dtype=torch.bool),
    )
reset_syncs = _counter["n"]
# The drafter opens its own ForwardContext and runs its own wrapped MoE gates.
with override_forward_context(fake_ctx()):
    stolen = ip.current_routing_kind(torch.tensor([1, 2, 3]), owner_id=id(drafter))
check("drafter gets its own answer, not 'image'", stolen, "text")
check("pending survives for its owner", ip._PENDING_ROUTING_KIND[0], "image")
check("owner mismatch warned (not silent)", len(ip._WARNED), 1)
with override_forward_context(fake_ctx()):
    mine = ip.current_routing_kind(img_ids, owner_id=id(target))
check("target still gets 'image'", mine, "image")

# --------------------------------------------------------------------------
print("10. CUDA-graph capture short-circuits before any kind resolution")
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


lm = FakeLM()
router = _Router()
ap._wrap_router_compute_routing(router, _Gate())
router._dspark_routing_owner = id(lm)
real_capturing = torch.cuda.is_current_stream_capturing
torch.cuda.is_current_stream_capturing = lambda: True
try:
    ip.set_pending_routing_kind("image", id(lm))  # would route image if consulted
    with override_forward_context(fake_ctx()):
        r = router._compute_routing(
            torch.zeros((4, 4)), torch.zeros((4, 4)), None,
            input_ids=torch.full((4,), ID, dtype=torch.long),
        )
    check("capturing -> orig path", r[0], "orig")
    check("capturing .item() count", syncs(), 0)
    check("capturing did not consume pending", ip._PENDING_ROUTING_KIND[0], "image")
finally:
    torch.cuda.is_current_stream_capturing = real_capturing

# --------------------------------------------------------------------------
print("11. non-capturing text step: gate wrapper takes orig path, zero syncs")
reset()
calls["orig"] = 0
torch.cuda.is_current_stream_capturing = lambda: False
try:
    embed = embed_fn()
    txt = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    with override_forward_context(None):
        embed(lm, txt, multimodal_embeddings=None, is_multimodal=None)
    with override_forward_context(fake_ctx()):
        for _ in range(N_MOE_LAYERS):
            router._compute_routing(torch.zeros((4, 4)), torch.zeros((4, 4)), None, input_ids=txt)
    check("text step orig calls", calls["orig"], N_MOE_LAYERS)
    check("text step .item() count", syncs(), 0)
finally:
    torch.cuda.is_current_stream_capturing = real_capturing

# --------------------------------------------------------------------------
print("12. fused_topk_bias_split_vl honours a precomputed kind (no re-scan)")
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
    common = dict(
        hidden_states=torch.zeros((2, 2)),
        gating_output=torch.zeros((2, 2)),
        scoring_func="sigmoid",
        e_score_correction_bias=torch.ones(4),
        e_score_correction_bias_vl=vl_bias,
        topk=1,
        renormalize=False,
        indices_type=None,
        hash_indices_table="HASH",
        routed_scaling_factor=1.0,
    )
    ap.fused_topk_bias_split_vl(input_tokens=torch.tensor([ID, ID]), kind="image", **common)
    check(
        "kind=image -> bias_vl used",
        seen["e_score_correction_bias"].data_ptr() == vl_bias.data_ptr(),
        True,
    )
    check("kind=image -> hash table skipped", seen["hash_indices_table"], None)
    check("kind=image -> no re-scan syncs", syncs(), 0)
    seen.clear()
    ap.fused_topk_bias_split_vl(input_tokens=torch.tensor([1, 2]), kind="text", **common)
    check("kind=text -> text bias + hash table", seen["hash_indices_table"], "HASH")
    check("kind=text -> no re-scan syncs", syncs(), 0)
finally:
    ftb_mod.fused_topk_bias = _real_ftb
    ap.current_routing_kind = _real_curr

print("13. kind=None still resolves (backwards compatible call)")
reset()
ftb_mod.fused_topk_bias = _fake_ftb
try:
    seen.clear()
    with override_forward_context(None):
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

# --------------------------------------------------------------------------
print("14. hardening: no silent degradation")
reset()


class _Slotted:
    __slots__ = ()


ip.set_pending_routing_kind("text", 1)
with override_forward_context(_Slotted()):
    try:
        ip.current_routing_kind(torch.tensor([1, 2]), owner_id=1)
        print("  [FAIL] setattr on a locked ForwardContext must raise")
        FAILURES.append("locked ctx raises")
    except AttributeError as e:
        print(f"  [PASS] setattr on a locked ForwardContext raises: {type(e).__name__}")

raises(
    "_num_tokens on an unsized object raises (no silent 0 -> 'image')",
    TypeError,
    lambda: ap._num_tokens(object()),
)
raises(
    "routing_kind_from_mm(n_tokens=0) raises",
    ValueError,
    lambda: ip.routing_kind_from_mm(3, 0),
)

_saved = ip._FORWARD_CONTEXT_HOOKS
ip._FORWARD_CONTEXT_HOOKS = None
import vllm.forward_context as _fcmod

_saved_fn = _fcmod.is_forward_context_available
del _fcmod.is_forward_context_available
try:
    raises(
        "missing vllm forward_context symbol is a hard failure",
        AttributeError,
        lambda: ip._forward_context_or_none(),
    )
finally:
    _fcmod.is_forward_context_available = _saved_fn
    ip._FORWARD_CONTEXT_HOOKS = _saved

# --------------------------------------------------------------------------
print("15. tag_routing_owner binds gates by object graph, not by name")


class _Experts(nn.Module):
    def __init__(self, router):
        super().__init__()
        self.router = router


class _MoE(nn.Module):
    def __init__(self, with_vl):
        super().__init__()
        self.gate = nn.Module()
        self.gate.e_score_correction_bias_vl = torch.zeros(4) if with_vl else None
        self.experts = _Experts(_Router())


class _Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_MoE(True), _MoE(True), _MoE(False)])


stack = _Stack()
n = ap.tag_routing_owner(stack, 4242)
check("tagged only bias_vl gates", n, 2)
check("moe tagged", stack.layers[0]._dspark_routing_owner, 4242)
check("router tagged", stack.layers[0].experts.router._dspark_routing_owner, 4242)
check("non-vl gate untagged", getattr(stack.layers[2], "_dspark_routing_owner", None), None)

print()
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILED -> {FAILURES}")
    sys.exit(1)
print("RESULT: ALL PASS")
