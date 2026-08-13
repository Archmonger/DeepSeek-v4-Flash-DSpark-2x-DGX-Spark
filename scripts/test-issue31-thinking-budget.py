#!/usr/bin/env python3
"""Unit tests for the V2 thinking-budget hotfix (no live serve required)."""
from __future__ import annotations

import importlib.util
import os
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-dsv4-issue31-v2-thinking-budget.py"


def _load_state_class():
    spec = importlib.util.spec_from_file_location("hotfix_issue31", HOTFIX)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ns: dict = {}
    exec(mod.THINKING_BUDGET_PY, ns)
    return ns["ThinkingBudgetState"], ns["_last_subseq"], ns["_env_optional_int"]


class _Arr:
    def __init__(self, data):
        self.np = data


class _Uva:
    def __init__(self, rows):
        self.cpu = rows


class _Tokens:
    def __init__(self, rows):
        self._uva_buf = _Uva(rows)


class _ReqStates:
    def __init__(self, rows, prefill, computed):
        self.all_token_ids = _Tokens(rows)
        self.prefill_len = _Arr(prefill)
        self.num_computed_tokens_np = computed


class _Reasoning:
    def __init__(self, start, end):
        self.reasoning_start_token_ids = start
        self.reasoning_end_token_ids = end


class _Params:
    def __init__(self, budget=None):
        self.thinking_token_budget = budget


class Issue31ThinkingBudgetTest(unittest.TestCase):
    def setUp(self):
        self.Cls, self.last_subseq, self.env_int = _load_state_class()

    def _state(self, rows, prefill, computed, start=(1,), end=(2,), max_reqs=4):
        req = _ReqStates(rows, prefill, computed)
        st = self.Cls(req, _Reasoning(list(start), list(end)), max_reqs)
        return st

    def test_last_subseq(self):
        self.assertEqual(self.last_subseq([9, 1, 8, 1, 7], [1]), 3)
        self.assertEqual(self.last_subseq([1, 2, 3, 1, 2], [1, 2]), 3)
        self.assertEqual(self.last_subseq([1, 2], [9]), -1)
        self.assertEqual(self.last_subseq([], [1]), -1)

    def test_env_default_and_unbounded(self):
        saved = os.environ.pop("DEFAULT_THINKING_TOKEN_BUDGET", None)
        try:
            self.assertEqual(self.env_int("DEFAULT_THINKING_TOKEN_BUDGET", 32768), 32768)
            os.environ["DEFAULT_THINKING_TOKEN_BUDGET"] = "0"
            self.assertIsNone(self.env_int("DEFAULT_THINKING_TOKEN_BUDGET", 32768))
            os.environ["DEFAULT_THINKING_TOKEN_BUDGET"] = ""
            self.assertIsNone(self.env_int("DEFAULT_THINKING_TOKEN_BUDGET", 32768))
            os.environ["DEFAULT_THINKING_TOKEN_BUDGET"] = "64"
            self.assertEqual(self.env_int("DEFAULT_THINKING_TOKEN_BUDGET", 32768), 64)
        finally:
            if saved is None:
                os.environ.pop("DEFAULT_THINKING_TOKEN_BUDGET", None)
            else:
                os.environ["DEFAULT_THINKING_TOKEN_BUDGET"] = saved

    def test_omit_field_gets_recipe_default(self):
        saved = os.environ.pop("DEFAULT_THINKING_TOKEN_BUDGET", None)
        try:
            rows = [[0] * 8]
            st = self._state(rows, [4], [4])
            st.add_request(0, _Params(None))
            self.assertEqual(st.budget[0], 32768)
        finally:
            if saved is None:
                os.environ.pop("DEFAULT_THINKING_TOKEN_BUDGET", None)
            else:
                os.environ["DEFAULT_THINKING_TOKEN_BUDGET"] = saved

    def test_explicit_budget_wins(self):
        rows = [[7, 7, 1, 9, 9, 9, 0, 0]]
        st = self._state(rows, [3], [6])
        st.add_request(0, _Params(64))
        self.assertEqual(st.budget[0], 64)

    def test_unbounded_skips_decide(self):
        rows = [[7, 1] + [9] * 20]
        st = self._state(rows, [2], [22])
        st.add_request(0, types.SimpleNamespace(thinking_token_budget=None))
        saved = os.environ.get("DEFAULT_THINKING_TOKEN_BUDGET")
        os.environ["DEFAULT_THINKING_TOKEN_BUDGET"] = "0"
        try:
            st.add_request(0, _Params(None))
            rows_out, toks = st.decide([0], [0], [0])
            self.assertEqual(rows_out, [])
            self.assertEqual(st.tokens_read, 0)
        finally:
            if saved is None:
                os.environ.pop("DEFAULT_THINKING_TOKEN_BUDGET", None)
            else:
                os.environ["DEFAULT_THINKING_TOKEN_BUDGET"] = saved

    def test_force_when_budget_exhausted(self):
        # prompt ... <think>=1, then 4 think tokens; budget 4 → next token forced
        seq = [7, 7, 1, 9, 9, 9, 9]
        st = self._state([seq + [0] * 8], [3], [7], start=(1,), end=(2,))
        st.add_request(0, _Params(4))
        rows, toks = st.decide([0], [0], [0])
        self.assertEqual(rows, [0])
        self.assertEqual(toks, [2])

    def test_no_force_below_budget(self):
        seq = [7, 7, 1, 9, 9]
        st = self._state([seq + [0] * 8], [3], [5], start=(1,), end=(2,))
        st.add_request(0, _Params(8))
        rows, toks = st.decide([0], [0], [0])
        self.assertEqual(rows, [])
        self.assertEqual(toks, [])

    def test_no_force_after_natural_end(self):
        seq = [7, 1, 9, 9, 2, 8, 8]
        st = self._state([seq], [2], [7], start=(1,), end=(2,))
        st.add_request(0, _Params(2))
        rows, _ = st.decide([0], [0], [0])
        self.assertEqual(rows, [])

    def test_spec_window_forces_overflow_token(self):
        # n=5, last_start=2 (token 1), think_count=2; budget=3
        # lpos=0 still under; lpos=1 hits budget
        seq = [7, 7, 1, 9, 9]
        st = self._state([seq + [0] * 4], [3], [5], start=(1,), end=(2,))
        st.add_request(0, _Params(3))
        rows, toks = st.decide([0, 0], [0, 1], [0])
        self.assertEqual(rows, [1])
        self.assertEqual(toks, [2])

    def test_long_prefix_is_not_fully_rescanned(self):
        n = 20_000
        row = [9] * n
        row[-3] = 1  # <think> near the end of the prompt
        computed = [n]
        st = self._state([row], [n], computed, start=(1,), end=(2,))
        st.add_request(0, _Params(32_768))
        st.decide([0], [0], [0])
        first_read = st.tokens_read
        self.assertLessEqual(first_read, st._PRIME_TAIL)
        self.assertGreater(first_read, 0)

        # Grow by 4 decode tokens; only the new slice (+overlap) may be read.
        row.extend([9, 9, 9, 9])
        computed[0] = n + 4
        st.tokens_read = 0
        st.decide([0], [0], [0])
        self.assertLessEqual(st.tokens_read, 8)
        self.assertGreater(st.tokens_read, 0)

        # Same step, second row (spec window): no extra sequence read.
        st.tokens_read = 0
        st.decide([0, 0], [0, 1], [0])
        self.assertEqual(st.tokens_read, 0)

    def test_incremental_sees_new_end_token(self):
        seq = [7, 1, 9, 9, 9]
        st = self._state([seq + [0] * 8], [2], [5], start=(1,), end=(2,))
        st.add_request(0, _Params(2))
        rows, _ = st.decide([0], [0], [0])
        self.assertEqual(rows, [0])  # already over budget

        # Model emitted </think> itself; next step must stop forcing.
        st.req_states.all_token_ids._uva_buf.cpu[0][5] = 2
        st.req_states.num_computed_tokens_np[0] = 6
        rows, _ = st.decide([0], [0], [0])
        self.assertEqual(rows, [])

    def test_multitoken_end_straddles_step_boundary(self):
        start, end = [1], [2, 3]
        seq = [7, 1, 9, 9]
        st = self._state([seq + [0] * 8], [2], [4], start=start, end=end)
        st.add_request(0, _Params(100))
        st.decide([0], [0], [0])
        st.req_states.all_token_ids._uva_buf.cpu[0][4] = 2
        st.req_states.num_computed_tokens_np[0] = 5
        st.decide([0], [0], [0])
        st.req_states.all_token_ids._uva_buf.cpu[0][5] = 3
        st.req_states.num_computed_tokens_np[0] = 6
        rows, _ = st.decide([0], [0], [0])
        self.assertEqual(st.last_end[0], 4)
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
