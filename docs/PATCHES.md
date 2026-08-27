# Patch 1 & Patch 2 — detailed reference

Both patches live in the DSpark vLLM overlay and together make `--max-num-seqs > 1`
**correct** under vLLM-v1 continuous batching. Single-stream and uniform-static
batches keep the original code path (byte-identical).

Files changed:

| file | + | − | role |
|---|---:|---:|---|
| `vllm/v1/spec_decode/dspark_proposer.py` | 158 | 10 | draft loop, slot map, ragged context (Patch 1+2+2b) |
| `vllm/models/deepseek_v4/nvidia/dspark.py` | 110 | 12 | persistent KV store (`store_main_kv`), `prefill_main` |
| `vllm/v1/worker/gpu_model_runner.py` | 10 | 0 | thread `req_ids` into `propose()` |

---

## Patch 1 — request-stable KV slot

### Symptom
At `max_num_seqs>1`, draft acceptance collapsed toward 0 (garbage drafts), even
though nothing crashed — the engine silently degraded to single-stream quality.

### Root cause
DSpark's draft keeps one persistent cross-step tensor per attention module —
`DeepSeekV4DSparkAttention.main_kv_cache`, shape `[max_num_seqs, window, head_dim]`
— a per-row **ring buffer** holding each sequence's sliding-window KV history. It
was read/written by **batch-row position** (`main_kv_cache[:batch_size]`). The
draft proposer carried **no request identity**.

Under vLLM-v1 continuous batching the running set is *condensed* whenever a request
finishes (a later request is moved into the freed row). The model's persistent
`main_kv_cache` row is **not** moved with it, so after a condense a request reads a
ring buffer that belongs to a **different** request → corrupted draft context →
acceptance collapse. (Single-stream never condenses row 0, which is why it worked.)

### Fix
Key the persistent cache by a **stable per-request slot** instead of batch row:

- `dspark_proposer.py`: add `self._req_id_to_slot: dict[str,int]` and
  `self._free_slots`. `_row_to_slot(req_ids)` reclaims slots of finished requests,
  assigns a free slot (lowest-first) to new ones, and returns the slot per row in
  `req_ids` order. A persistent, cudagraph-captured `_draft_slot_index_buffer`
  carries the slots into the graphed draft read path.
- `dspark.py`: `store_main_kv` and `forward_dspark` index the cache by
  `slot_index` (gather `index_select` on read, scatter `index_copy_` on write)
  instead of `[:batch_size]`.
- `gpu_model_runner.py`: pass `req_ids=self.input_batch.req_ids` into `propose()`
  (only for the DSpark proposer).

### Why it's safe
The math is unchanged — it only re-routes which physical row a request uses. When
the computed permutation is identity (a genuine single-request-at-a-time server
always gets slot 0), the code takes the **original in-place write path,
byte-for-byte**. Gating is on the *permutation identity*, not on `batch==1`, so the
"batch condenses to one surviving request holding a non-zero slot" case stays
correct.

---

## Patch 2 — ragged context path

### Symptom
Under real (independent / staggered) arrivals at `max_num_seqs>1`, the server
returned HTTP 500:

```
ValueError: DSpark currently requires uniform flattened per-request inputs;
got 41 rows for batch_size=2.   (dspark_proposer.py: _view_by_request)
```

### Root cause
`prepare_context` reshaped the flat target hidden states into a **rectangular**
`[batch, seq, H]` via `_view_by_request` / `_positions_by_request`, asserting every
request contributed the **same** number of rows. With chunked prefill (required —
disabling it needs `max_num_batched_tokens >= max_model_len`, infeasible at long
context) a single step **mixes prefill and decode** rows, so per-request row counts
differ (e.g. "41 rows for batch_size=2" = one request prefilling alongside one
decoding). Rectangular reshape is impossible → crash. The static benchmark passed
only because all prompts were identical length (uniform).

### Fix
Make the context path **ragged** using `query_start_loc` (per-request segment
offsets) — the same mechanism `_trim_rejected_target_context` already used:

- `dspark_proposer.py` `prepare_context`: detect non-uniform segment lengths
  (`ragged = len(set(seg_lengths)) != 1`). In the ragged branch, skip the
  rectangular view; compute each request's draft anchor with a flat index
  `anchor_idx = starts + clamp(len - rejected - 1, 0, len-1)` and
  `index_select` the per-request last hidden/positions. Pass the flat hidden +
  `query_start_loc` + `slot_index` to `prefill_main`.
- `dspark.py`: `store_main_kv(..., query_start_loc=...)` dispatches to a new
  `_store_main_kv_ragged` that loops requests via `query_start_loc`, truncates each
  segment to the last `window_size` rows, computes `slots = positions % window`,
  applies the rejected-suffix mask, and `index_copy_`s into that request's slot.
  `prefill_main` threads `query_start_loc` through and skips the rectangular view in
  ragged mode.

### Why it's safe
Storage is **position-addressed** (`positions % window`), so it never needed
uniform lengths — only the intermediate rectangular view did. When lengths are
uniform (`query_start_loc is None` / static / single-stream) the original
rectangular fast-path runs unchanged. Ragged/mixed steps run **eager** (mixed steps
are never cudagraph-captured), so dynamic Python loops / variable shapes are safe;
the uniform decode-only graphed path is untouched.

### Scope
Only the `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1` path was made ragged (the path
used in serving). The legacy `_trim_rejected_target_context` path still assumes
uniform. **Run with `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`.**

---

## Patch 2b — ragged detection independent of rejection

### Symptom (found by the GSM8K quality eval)
After Patch 2, a prefill-heavy step with **no rejection** still 500'd:
`ValueError: ... got 166 rows for batch_size=3` at `_view_by_request`. Earlier
staggered tests (uniform-ish prompts) missed it; GSM8K's varied prompt lengths hit
it.

### Root cause
Patch 2 computed `ragged` **only inside** `if gpu_mask and num_rejected_tokens_gpu
is not None`. On steps with no rejection (`num_rejected=None`, e.g. fresh requests
prefilling), detection was skipped and the code fell through to the rectangular
`_view_by_request` → crash. Raggedness depends on `query_start_loc` segment lengths,
**not** on rejection.

### Fix
- Enter the detection/ragged branch whenever `_gpu_rejected_context_mask` is on,
  **regardless of `num_rejected_tokens_gpu`** (which may be `None`).
- In the ragged anchor, default `rejected` to zeros when `num_rejected_tokens_gpu is
  None`. `_store_main_kv_ragged` already handled `None` (no masking).

### Validation
GSM8K N=8 (200 Q) — the load that crashed pre-fix — now completes with **0 errors**,
93.5% accuracy vs 95.0% sequential, **97.5% per-question agreement** (quality-neutral
within batch FP-nondeterminism).

---

## Issue #21 — `encode_arguments_to_dsml` corrupts dict tool arguments

### Symptom
Multi-turn tool calling fails after the first successful tool turn: prior assistant
tool calls are re-encoded into the prompt with a single wrapped `arguments`
parameter instead of the real keys. The model then imitates that corrupt history.

### Root cause
HF checkpoint `encoding/encoding_dsv4.py` always does `json.loads(tool_call["arguments"])`.
When `arguments` is already a `dict` (common in OpenAI-compatible replay),
`json.loads` raises and the `except` path wraps it as `{"arguments": <dict>}`.

Upstream: `deepseek-ai/DeepSeek-V4-Flash-0731` `encoding/encoding_dsv4.py` (same
bug in the Keys abliterated snapshot). Not a vLLM recipe weights bug.

### Fix
Dispatch on type before parsing; keep the wrap only for non-JSON strings.
Applied at container boot after encoder install via
`patches/hotfix-encoding-dsv4-issue21.py` (mounted at
`/opt/hotfix-encoding-dsv4-issue21.py`).

### Test
```bash
python3 scripts/test-encoding-dsv4-issue21.py
```

---

## Issue #22 — `nvfp4_ds_mla` long-context decode regression

### Symptom
With `--kv-cache-dtype nvfp4_ds_mla` (the recipe default), decode throughput
drops to ~1 tok/s at 600K+ context, while `fp8_ds_mla` maintains ~17 tok/s at
the same context length. Short-context throughput (~66 tok/s) is unaffected.

### Root cause
`flashmla_sparse.py` line 880 dispatches `nvfp4_ds_mla` to the slow
`_forward_bf16_kv` kernel path instead of the fast `_forward_fp8_kv` path.
The584-byte KV layout is identical for both dtypes on DSV4; only the kernel
dispatch differs.

```python
# Line 880 in flashmla_sparse.py
use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"
# nvfp4_ds_mla → False → slow _forward_bf16_kv (~1 tok/s at 600K)
# fp8_ds_mla   → True  → fast _forward_fp8_kv (~17 tok/s at 600K)
```

### Fix
```python
use_fp8_cache = self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")
```

### Hotfix for running containers
```bash
docker exec <container> bash /path/to/hotfix-nvfp4-ds-mla-issue22.sh
# Then restart the vLLM process inside the container.
```

### File changed
| file | change |
|---|---|
| `v1/attention/backends/mla/flashmla_sparse.py` | `use_fp8_cache` check: include `nvfp4_ds_mla` |

---

## Issue #52 — trailing assistant turn closes with EOS (no-op loop)

### Symptom
An agent harness gets stuck emitting empty turns: 1-2 generated tokens with
`finish_reason: "stop"`, no tool call, fragments of hallucinated markup
(`<result observation="no-op"></content>`, stray `</parameter>`). During a live
incident 6 of 37 requests generated ≤10 tokens, all `stop`, zero `length`.

### Root cause
`render_message()` (HF checkpoint `encoding/encoding_dsv4.py`, installed as
`vllm/tokenizers/deepseek_v4_encoding.py`) appends the generation header only
when the trailing message is `user` or `developer`. A request whose `messages`
ends with an **assistant** message is closed with EOS and gets no header, so
the prompt ends on a bare EOS and the model generates from a dead state.
Self-sustaining: the harness records the empty turn, so the next request is
also assistant-final.

### Fix
Widen the separate generation-header transition condition to also match only
the final assistant message
(`patches/hotfix-dsv4-assistant-final-continuation.py`). The checkpoint encoder
has no `add_generation_prompt` input; the patch preserves the closed assistant
turn, then appends a fresh generation header. Reopening the turn with `wo_eos`
was measured worse (1-token empty generation on a complete turn) and is not
used. Runs after the entrypoint copies the encoder into place.

### Extension — trailing `latest_reminder` annotation (Issue #120)
A harness retry can append a trailing `latest_reminder` message after the
re-sent partial assistant turn. The reminder defeats the fix above: stock
closes the assistant turn with EOS, renders the bare reminder after it, and
the prompt still ends with no generation header — a dead state the model
escapes with immediate EOS or hallucinated markup. Verified on the real
checkpoint encoder (`encoding_dsv4.py`, snapshot `9e165c30`): a
reminder tail directly after `user`/`developer` already ends inside the
pending generation slot — the checkpoint emits
`ASSISTANT_SP_TOKEN` + thinking token *before* such a reminder — so those
tails are correct as-is.

The transition condition is therefore widened by exactly one more clause: a
**final** `latest_reminder` whose immediate predecessor is an **assistant**
message gains one fresh generation header appended after the reminder content
(thinking mode `<｜Assistant｜><think>`, chat mode `<｜Assistant｜></think>`).
Reminder tails after user/developer, reminders mid-transcript, task-precedence
rendering, and every assistant-final shape are byte-identical to the
pre-extension hotfix behavior; the post-write self-check additionally fails
closed if a patched encoder double-headers a user→reminder tail.

### Flag (default OFF = stock)
| value | behavior |
|---|---|
| `0` / unset / anything ≠ `1` | stock renderer; patcher mounted/synced to the worker but **never invoked** |
| `1` | patcher runs at container boot, chained with `\|\| exit 1` |

Fail-closed when ON: missing encoder file, missing anchor, or a failed
post-write self-check (patched module must import and render a
trailing-assistant transcript with a generation header and an
assistant-plus-trailing-`latest_reminder` transcript with one fresh header,
without appending a second header to a user→reminder tail) → nonzero exit,
boot aborts; a failed self-check **restores the original file bytes** first. An
already-patched encoder is re-validated (idempotent), never double-patched.

### Evidence status
Render/no-regression evidence is from prior head `f08cd6c`. The measured
positive-path evidence is a causal one-prompt A/B via `/v1/completions`:
trailing turn left open → 183 tokens, coherent continuation; closed with EOS →
400 tokens of raw `<|DSML|tool_calls>` markup emitted as text. **No rescue
claim**: the live no-op-loop defect did not reproduce in that session, so no
measured stuck-harness recovery exists.

A first gated-ON boot on `d4b31daf` failed closed before serving because a
review-requested guard named the nonexistent checkpoint variable
`add_generation_prompt`; the original encoder bytes were restored. Corrected
code commit `0864014` then passed serialized live proof on both ranks. OFF:
effective flag `0`, no patch marker, and the assistant-final render ended on
EOS. ON: effective flag `1`, both ranks logged `patched and verified`, and the
same 98 stock token IDs were preserved with exactly
`<|Assistant|><think>` appended. A live continuation completed with
`alpha beta`. Re-running the patcher on both ranks reported
`already applied and verified`. A deliberate anchor-drift boot exited `1` on
both ranks, entered restart/failure state, and never served the API.

Extension evidence is CPU-only so far: the patched patcher was applied to a
copy of the real checkpoint encoder (snapshot `9e165c30`) and a 16-case
render matrix confirmed the fixed shape gains exactly one fresh header while
every other shape stays byte-identical to the pre-extension hotfix. **No live
serving validation of the reminder-tail rescue exists yet** — run the same
gated-ON/OFF boot proof on both ranks before relying on it in production.

### Test
```bash
python3 scripts/test-assistant-final-continuation.py
```

---

## Issue #141 — sparse-MLA verify-decode chunking workaround (default OFF)

### Evidence and scope

Issue #141 is a **stochastic** TP=2 engine death sampled inside FlashInfer's
SM120 sparse-MLA paged-attention fallback; the pinned revision (`0472b9b3`)
routes calls of at most 64 rows to its standalone DSv4 decode kernel instead.
On one reporting pair, splitting at 64 survived 3,145,728 generated tokens
while a 65-row split failed in the first comparison round; the second failing
pair has **not** repeated that A/B, and the live TP/stream mechanism remains
unknown. This is an opt-in path-avoidance **workaround**, not a root-cause
fix: the fixed 64 is the pinned kernel dispatch boundary, not a deployment
concurrency threshold.

### Target and semantics

`patches/hotfix-dsv4-issue141-sparse-mla-decode-chunk.py` runs before vLLM is
imported and edits only the installed Anemll adapter method:

```
/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py
DeepseekV4FlashInferSM120Attention._forward_decode
```

Calls with at most 64 rows retain one call with the original unsliced objects.
Larger calls run sequential, monotonically increasing `slice` views of at most
64 rows. Exactly six row-coupled arguments are sliced together: `query`,
`sparse_indices`, `out`, `swa_topk_lens`, and the optional
`extra_sparse_indices` / `extra_sparse_topk_lens`. Both KV-cache objects,
workspace, sinks, scale, and layout remain shared by identity. Each inner call
writes a disjoint view of the existing output; no `clone`, `cat`, contiguous
copy, output replacement, retry, stream change, or generic FlashInfer wrapper
is introduced.

The patcher locks the **entire** pinned Anemll `_forward_decode` method,
including its overlay-only `_pad_decode_sparse_indices` call. It also requires
the load-bearing fragments of the pinned FlashInfer `_core.py` and
`_sparse_mla_sm120.py`: the 64-row workspace cutoff, the sliced SM120 call
signature, `_DECODE_MAX_TOKENS = 64`, the DSv4 decode dispatch predicate, and
the custom-op mutation contract that excludes both caches. An exact old method
applies once; an exact new method is recompiled and reverified without a
write. Missing, duplicate, mixed, partial-marker, SM100-like, or drifted
source is rejected.

Before publication the complete updated adapter source is compiled. Publication
uses a mode-preserving same-directory temporary file and `os.replace`; committed
bytes, mode, marker state, and syntax are checked afterward. Any failure after
the rename atomically restores and verifies the original bytes before returning
nonzero. The enabled Compose gate is chained with `|| exit 1`, so incompatible
source never reaches `exec vllm`.

### Enablement and rollback

| value | behavior |
|---|---|
| unset, `0`, or anything other than exact `1` | patcher is not invoked; installed adapter bytes remain stock |
| exact `1` | validate pinned sources, atomically apply or reverify, and fail boot closed on any error |

Enable in the authoritative `.env.dspark`:

```bash
DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK=1
./stop-deepseek-v4-flash-dspark.sh
./start-deepseek-v4-flash-dspark.sh
```

The launcher reports the resolved 0/1 state and syncs the selected patch source
to the worker; both rank entrypoints receive the same normalized flag and run
the same source preflight. Check both rank logs for `applied and verified` (or
`already applied and verified`) and then make a real generation request with a
terminal `finish_reason`. `/health` alone is insufficient: it remained 200 in
reported stalled/silently truncated incidents.

Rollback by setting `0` (or removing the variable), then run the same paired
stop/start flow. A process restart or `docker compose restart` is **not** a
rollback: the modified site-packages file remains in that container's writable
layer. Recreating both containers restores immutable image bytes.

### Validation status and remaining gates

The committed CPU suite freezes the pinned method and guard digests, exercises
exact apply/idempotence/drift/atomic rollback, and executes the injected block
against shared-backing fake tensors across the 1–576 boundary row matrix in
SWA-only and compressed-cache shapes. It also checks exact-1 fail-closed
Compose ordering and worker wiring.

The initial live TP=2 campaign covered pinned-image apply/boot on both ranks
and a short concurrency-16 generation with post-run smoke and restore. Before
relying on the workaround, close the outstanding gates: disposable pinned-image
extraction plus SM121a numerical/CUDA-graph tests, a two-rank OFF/ON/drift boot
proof, and repeated stochastic generation soaks verifying terminal
`finish_reason`, rank stability, throughput, and peak scratch memory. The
second failing pair still needs the 64/65 comparison; one clean burst cannot
close a stochastic issue.

### Test

```bash
python3 scripts/test-issue141-sparse-mla-decode-chunk.py
```
