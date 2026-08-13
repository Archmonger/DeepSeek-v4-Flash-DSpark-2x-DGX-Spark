#!/usr/bin/env python3
"""Hotfix: enable thinking_token_budget on the V2 runner (issue #31).

Upstream vLLM 0.25.2.dev0 rejects thinking_token_budget on the V2 model
runner (HTTP 400). DSpark only exists on V2, so this recipe had no way to
close a reasoning block before max_tokens.

This patch:
  1. Drops the V2 400 in the input processor.
  2. Installs a V2 sampler hook that counts tokens after the last
     reasoning-start marker and, once the budget is exhausted, forces the
     next sampled token(s) to the reasoning-end sequence (</think>).
     The count is incremental (prime a short tail, then only new tokens).
     A full-prefix Python rescan every decode step is not acceptable: #34
     enables a default budget on every omitted-field request.
  3. Threads ReasoningConfig into the V2 Sampler so start/end ids are known.

When the request omits thinking_token_budget, DEFAULT_THINKING_TOKEN_BUDGET
(default 32768) is used. Empty/0 restores unbounded think. When the client
omits max_tokens, DEFAULT_MAX_TOKENS (default 131072) is used (issue #34).

Idempotent. Patches files under
/usr/local/lib/python3.12/dist-packages/vllm/
"""
from pathlib import Path

VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARK = "# [issue31-hotfix] v2 thinking_token_budget"

THINKING_BUDGET_PY = r'''# SPDX-License-Identifier: Apache-2.0
# [issue31-hotfix] V2 thinking_token_budget (not upstream)
"""Force reasoning-end tokens when thinking_token_budget is exhausted.

#34 assigns a default budget to every omitted-field request, so this hook
runs on ordinary decode. Do **not** copy or linearly rescan the full prefix
on every sample step — that is O(context) Python + a GPU sync per token and
collapses long-context decode to a few tok/s.

Instead: prime once from a short tail (DSV4 puts <think> at the end of the
formatted prompt), then scan only newly appended tokens.
"""

from __future__ import annotations

import os


def _env_optional_int(name: str, default: int | None) -> int | None:
    # Missing key → recipe default. Empty or 0 → unbounded / omit.
    if name not in os.environ:
        return default
    raw = os.environ.get(name, "").strip()
    if raw == "" or raw == "0":
        return None
    return int(raw)


def _as_int_list(x) -> list[int]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    # numpy / torch: prefer a host copy when the tensor is still on device.
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        try:
            x = x.cpu()
        except Exception:
            pass
    if hasattr(x, "tolist"):
        data = x.tolist()
        if isinstance(data, list):
            return [int(v) for v in data]
        return [int(data)]
    return [int(x)]


def _last_subseq(seq: list[int], pat: list[int]) -> int:
    if not pat or len(seq) < len(pat):
        return -1
    last = -1
    plen = len(pat)
    for i in range(0, len(seq) - plen + 1):
        if seq[i : i + plen] == pat:
            last = i
    return last


class ThinkingBudgetState:
    # DSV4 chat template emits <think> at the end of the formatted prompt.
    # A short tail is enough to locate the active block; do not slurp 1M tokens.
    _PRIME_TAIL = 256

    def __init__(self, req_states, reasoning_config, max_num_reqs: int):
        self.req_states = req_states
        self.max_num_reqs = int(max_num_reqs)
        self.budget = [-1] * self.max_num_reqs
        self.last_start = [-1] * self.max_num_reqs
        self.last_end = [-1] * self.max_num_reqs
        self.scanned_n = [0] * self.max_num_reqs
        self.primed = [False] * self.max_num_reqs
        start = None
        end = None
        if reasoning_config is not None:
            start = reasoning_config.reasoning_start_token_ids
            end = reasoning_config.reasoning_end_token_ids
        self.start_ids = list(start or [])
        self.end_ids = list(end or [])
        self.enabled = bool(self.end_ids)
        self.overlap = max(len(self.start_ids), len(self.end_ids), 1) - 1
        # Tests / diagnostics: tokens actually read from the sequence buffer.
        self.tokens_read = 0

    def add_request(self, req_idx: int, sampling_params) -> None:
        b = getattr(sampling_params, "thinking_token_budget", None)
        if b is None:
            b = _env_optional_int("DEFAULT_THINKING_TOKEN_BUDGET", 32768)
        self.budget[req_idx] = -1 if b is None else int(b)
        self.last_start[req_idx] = -1
        self.last_end[req_idx] = -1
        self.scanned_n[req_idx] = 0
        self.primed[req_idx] = False

    def _read_tokens(self, req_idx: int, start: int, end: int) -> list[int]:
        if end <= start:
            return []
        token_buf = self.req_states.all_token_ids
        if hasattr(token_buf, "_uva_buf"):
            row = token_buf._uva_buf.cpu[req_idx]
        else:
            row = token_buf.gpu[req_idx]
            if hasattr(row, "detach"):
                row = row.detach()
            if hasattr(row, "cpu"):
                row = row.cpu()
        sl = row[start:end]
        if hasattr(sl, "tolist"):
            out = [int(x) for x in sl.tolist()]
        else:
            out = [int(x) for x in sl]
        self.tokens_read += len(out)
        return out

    def _last_abs(self, tokens: list[int], pat: list[int], origin: int) -> int:
        rel = _last_subseq(tokens, pat)
        return -1 if rel < 0 else origin + rel

    def _refresh(self, req_idx: int, n: int) -> None:
        if n < self.scanned_n[req_idx]:
            self.primed[req_idx] = False
            self.last_start[req_idx] = -1
            self.last_end[req_idx] = -1
            self.scanned_n[req_idx] = 0
        if not self.primed[req_idx]:
            tail = min(n, max(self._PRIME_TAIL, self.overlap + 1))
            lo = n - tail
            toks = self._read_tokens(req_idx, lo, n)
            self.last_start[req_idx] = self._last_abs(toks, self.start_ids, lo)
            self.last_end[req_idx] = self._last_abs(toks, self.end_ids, lo)
            self.scanned_n[req_idx] = n
            self.primed[req_idx] = True
            return
        if n <= self.scanned_n[req_idx]:
            return
        lo = max(0, self.scanned_n[req_idx] - self.overlap)
        toks = self._read_tokens(req_idx, lo, n)
        ls = self._last_abs(toks, self.start_ids, lo)
        le = self._last_abs(toks, self.end_ids, lo)
        if ls >= 0:
            self.last_start[req_idx] = ls
        if le >= 0:
            self.last_end[req_idx] = le
        self.scanned_n[req_idx] = n

    def decide(
        self,
        expanded_idx_mapping,
        expanded_local_pos,
        idx_mapping,
    ) -> tuple[list[int], list[int]]:
        """Return (logit rows, end-token ids) that must be forced this step."""
        if not self.enabled:
            return [], []
        idx = _as_int_list(idx_mapping)
        if not any(0 <= i < self.max_num_reqs and self.budget[i] >= 0 for i in idx):
            return [], []

        req_rows = _as_int_list(expanded_idx_mapping)
        local_pos = _as_int_list(expanded_local_pos)
        if len(local_pos) < len(req_rows):
            local_pos = local_pos + [0] * (len(req_rows) - len(local_pos))

        seen: set[int] = set()
        for req_idx in req_rows:
            if req_idx < 0 or req_idx in seen:
                continue
            if req_idx >= self.max_num_reqs or self.budget[req_idx] < 0:
                continue
            seen.add(req_idx)
            prefill_len = int(self.req_states.prefill_len.np[req_idx])
            n = int(self.req_states.num_computed_tokens_np[req_idx])
            if n < prefill_len:
                continue
            self._refresh(req_idx, n)

        force_rows: list[int] = []
        force_toks: list[int] = []
        for row, (req_idx, lpos) in enumerate(zip(req_rows, local_pos)):
            if req_idx < 0 or req_idx >= self.max_num_reqs:
                continue
            budget = self.budget[req_idx]
            if budget < 0:
                continue
            last_start = self.last_start[req_idx]
            last_end = self.last_end[req_idx]
            if last_start < 0 or last_end > last_start:
                continue
            n = int(self.req_states.num_computed_tokens_np[req_idx])
            think_count = n - (last_start + len(self.start_ids))
            # This logit is the next generated token at offset lpos in the
            # draft+bonus window (0 = first new token this step).
            if think_count + int(lpos) < budget:
                continue
            overflow = think_count + int(lpos) - budget
            end_i = min(overflow, len(self.end_ids) - 1)
            force_rows.append(row)
            force_toks.append(int(self.end_ids[end_i]))
        return force_rows, force_toks

    def apply(
        self,
        logits,
        expanded_idx_mapping,
        expanded_local_pos,
        idx_mapping_np,
    ) -> None:
        force_rows, force_toks = self.decide(
            expanded_idx_mapping, expanded_local_pos, idx_mapping_np
        )
        if not force_rows or logits is None:
            return
        import torch

        # Wipe the forced rows then pin the end token so later top-k cannot
        # drop it.
        rows = torch.tensor(force_rows, device=logits.device, dtype=torch.long)
        toks = torch.tensor(force_toks, device=logits.device, dtype=torch.long)
        logits[rows] = float("-inf")
        logits[rows, toks] = 1.0e9
'''


def _patch_file(path: Path, old: str, new: str, what: str) -> None:
    src = path.read_text()
    # Must test the full replacement, not just MARK: several hunks share one
    # file, and MARK is written by the first hunk.
    if new in src:
        print(f"[issue31-hotfix] already applied: {what} ({path})")
        return
    if old not in src:
        raise SystemExit(f"[issue31-hotfix] anchor not found for {what} in {path}")
    path.write_text(src.replace(old, new, 1))
    print(f"[issue31-hotfix] patched {what} in {path}")


def main() -> None:
    dest = VLLM / "v1/worker/gpu/sample/thinking_budget.py"
    dest.write_text(THINKING_BUDGET_PY)
    print(f"[issue31-hotfix] wrote {dest}")

    _patch_file(
        VLLM / "v1/engine/input_processor.py",
        """                if self.use_v2_model_runner:
                    raise ValueError(
                        "thinking_token_budget is not yet supported by the V2 "
                        "model runner. Run vLLM with VLLM_USE_V2_MODEL_RUNNER=0 "
                        "to use thinking_token_budget."
                    )""",
        f"""                if self.use_v2_model_runner:
                    # {MARK}: implemented in V2 Sampler (DSpark).
                    pass""",
        "input_processor V2 gate",
    )

    sampler = VLLM / "v1/worker/gpu/sample/sampler.py"
    _patch_file(
        sampler,
        """from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates
from vllm.v1.worker.gpu.states import RequestState""",
        f"""from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState  # {MARK}""",
        "sampler import",
    )
    _patch_file(
        sampler,
        """        num_speculative_tokens: int = 1,
        use_fp64_gumbel: bool = False,
    ):""",
        f"""        num_speculative_tokens: int = 1,
        use_fp64_gumbel: bool = False,
        reasoning_config=None,  # {MARK}
    ):""",
        "sampler init signature",
    )
    _patch_file(
        sampler,
        """        self.num_speculative_tokens = num_speculative_tokens
        self.use_flashinfer = flashinfer_sampler_supported()""",
        f"""        self.num_speculative_tokens = num_speculative_tokens
        self.use_flashinfer = flashinfer_sampler_supported()
        self.thinking_budget_state = ThinkingBudgetState(  # {MARK}
            req_states, reasoning_config, max_num_reqs
        )""",
        "sampler thinking state",
    )
    _patch_file(
        sampler,
        """        self.sampling_states.add_request(req_idx, sampling_params)
        self.penalties_state.add_request(req_idx, sampling_params)""",
        f"""        self.sampling_states.add_request(req_idx, sampling_params)
        self.thinking_budget_state.add_request(req_idx, sampling_params)  # {MARK}
        self.penalties_state.add_request(req_idx, sampling_params)""",
        "sampler add_request",
    )
    _patch_file(
        sampler,
        """        # Apply min_p in place.
        self.sampling_states.apply_min_p(logits, expanded_idx_mapping, idx_mapping_np)

        if skip_top_k_top_p:
            return logits""",
        f"""        # Apply min_p in place.
        self.sampling_states.apply_min_p(logits, expanded_idx_mapping, idx_mapping_np)

        # {MARK}: force </think> before top-k so the token cannot be dropped.
        self.thinking_budget_state.apply(
            logits, expanded_idx_mapping, expanded_local_pos, idx_mapping_np
        )

        if skip_top_k_top_p:
            return logits""",
        "sampler apply",
    )

    cfg = VLLM / "config/vllm.py"
    _patch_file(
        cfg,
        """        if self.reasoning_config is not None:
            logger.warning_once(
                "Model Runner V2 does not yet support the thinking_token_budget "
                "request parameter. Set VLLM_USE_V2_MODEL_RUNNER=0 if this is required."
            )""",
        f"""        if self.reasoning_config is not None:
            # {MARK}: budget is implemented in the V2 Sampler.
            pass""",
        "vllm.py stale V2 warning",
    )

    _patch_file(
        VLLM / "entrypoints/serve/utils/api_utils.py",
        """    fallback_max_tokens = (
        max_tokens
        if max_tokens is not None
        else default_sampling_params.get("max_tokens")
    )""",
        """    fallback_max_tokens = (
        max_tokens
        if max_tokens is not None
        else default_sampling_params.get("max_tokens")
    )
    if fallback_max_tokens is None:
        # [issue34-hotfix] recipe default when the client omits max_tokens.
        import os
        raw = os.environ.get("DEFAULT_MAX_TOKENS", "131072").strip()
        if raw and raw != "0":
            fallback_max_tokens = int(raw)""",
        "get_max_tokens recipe default",
    )

    runner = VLLM / "v1/worker/gpu/model_runner.py"
    _patch_file(
        runner,
        """            self.sampler = Sampler(
                max_num_reqs=self.max_num_reqs,
                vocab_size=self.vocab_size,
                device=self.device,
                req_states=self.req_states,
                logprobs_mode=self.model_config.logprobs_mode,
                num_speculative_tokens=self.decode_query_len,
                use_fp64_gumbel=self.model_config.use_fp64_gumbel,
            )""",
        f"""            self.sampler = Sampler(
                max_num_reqs=self.max_num_reqs,
                vocab_size=self.vocab_size,
                device=self.device,
                req_states=self.req_states,
                logprobs_mode=self.model_config.logprobs_mode,
                num_speculative_tokens=self.decode_query_len,
                use_fp64_gumbel=self.model_config.use_fp64_gumbel,
                reasoning_config=self.vllm_config.reasoning_config,  # {MARK}
            )""",
        "model_runner Sampler kwarg",
    )
    print("[issue31-hotfix] done")


if __name__ == "__main__":
    main()
