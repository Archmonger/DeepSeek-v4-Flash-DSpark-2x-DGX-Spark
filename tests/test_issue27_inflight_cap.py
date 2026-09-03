#!/usr/bin/env python3
"""Behavioral tests for the #27 partial-prefill admission-cap hotfix."""
from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-dsv4-issue27-partial-prefill-concurrency.py"
ENV_NAME = "DSPARK_MAX_INFLIGHT_PREFILLS"
PATH_MARKER = 'Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py")'
R2_MARK = "# [issue27-r2]"
# Pre-r2 revisions of the patcher: any application carrying MARK without the
# r2 marker must be refused by the current patcher (fail-closed).
PRE_R2_REVS = (
    "89caf6877537144ce2a1161786e05f435853f1d1",
    "c444d7032957f5a5437261d5366fd06b27a01760",
)

FIXTURE = """\
class Request:
    def __init__(self, request_id, num_computed_tokens, num_tokens,
                 num_output_placeholders=0):
        self.request_id = request_id
        self.num_computed_tokens = num_computed_tokens
        self.num_tokens = num_tokens
        self.num_prompt_tokens = num_tokens  # R-era gate compatibility
        self.num_output_placeholders = num_output_placeholders


class _Logger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    @staticmethod
    def _fmt(message, args):
        return message % args if args else message

    def warning(self, message, *args):
        self.warnings.append(self._fmt(message, args))

    def info(self, message, *args):
        self.infos.append(self._fmt(message, args))


logger = _Logger()


class Scheduler:
    def __init__(self, config_cap=1):
        self.scheduler_config = type(
            "SchedulerConfig",
            (),
            {"max_num_partial_prefills": config_cap},
        )()
        self.max_num_running_reqs = 8
        self.num_waiting_for_streaming_input = 0
        self.running = []
        self.waiting = [Request("w0", 0, 512)]
        self.current_step = 0
        self.step_scheduled = {}

        # In-flight requests still prefilling (prefill chunks + in-progress
        # async KV loads). Their remaining-block reservation gates async loads.
        self._inflight_prefills: set[Request] = set()

    def schedule(self):
        token_budget = 1
        admitted = 0
        can_schedule_waiting = True
        num_scheduled_tokens = dict(self.step_scheduled)
        if can_schedule_waiting:
            while self.waiting and token_budget > 0:
                num_running = len(self.running) + self.num_waiting_for_streaming_input
                if num_running >= self.max_num_running_reqs:
                    break

                request = self.waiting.pop(0)
                num_new_tokens = min(
                    1024, request.num_tokens - request.num_computed_tokens
                )
                num_scheduled_tokens[request.request_id] = num_new_tokens
                self.running.append(request)
                if (
                    request.num_computed_tokens + num_new_tokens
                    < request.num_tokens
                ):
                    self._inflight_prefills.add(request)
                admitted += 1
        return admitted
"""


def _apply_to(path: Path, patcher_text: str | None = None) -> None:
    txt = HOTFIX.read_text() if patcher_text is None else patcher_text
    txt = txt.replace(PATH_MARKER, f"Path({str(path)!r})")
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            exec(compile(txt, "hotfix", "exec"), {})
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise


def _status_of(path: Path) -> str:
    txt = HOTFIX.read_text().replace(PATH_MARKER, f"Path({str(path)!r})")
    buf = io.StringIO()
    with mock.patch("sys.argv", ["hotfix", "--status"]), \
            contextlib.redirect_stdout(buf):
        try:
            exec(compile(txt, "status", "exec"), {})
        except SystemExit as exc:
            assert exc.code in (None, 0)
    return buf.getvalue()


def _load_scheduler(raw: str | None, config_cap: int = 1):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "scheduler.py"
        path.write_text(FIXTURE)
        _apply_to(path)
        patched = path.read_text()

    namespace: dict = {}
    exec(compile(patched, str(path), "exec"), namespace)
    with mock.patch.dict(os.environ, {}, clear=False):
        if raw is None:
            os.environ.pop(ENV_NAME, None)
        else:
            os.environ[ENV_NAME] = raw
        scheduler = namespace["Scheduler"](config_cap)
    return scheduler, namespace["logger"], patched


class _RunningReq:
    """Fake request exposing the pinned Request prefill counters."""

    def __init__(self, request_id, num_computed_tokens, num_tokens,
                 num_output_placeholders=0):
        self.request_id = request_id
        self.num_computed_tokens = num_computed_tokens
        self.num_tokens = num_tokens
        self.num_prompt_tokens = num_tokens  # R-era gate compatibility
        self.num_output_placeholders = num_output_placeholders


def _mid_prefill():
    return _RunningReq("m", 1024, 22829)


class Issue27InflightCapTest(unittest.TestCase):
    def test_valid_and_fallback_values_are_cached(self):
        cases = (
            (None, 1),
            ("", 1),
            ("   ", 1),
            ("0", 1),
            ("-1", 1),
            ("1", 1),
            ("2", 2),
            ("3", 3),
            ("4", 3),
            (" 2 ", 2),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                scheduler, logger, _ = _load_scheduler(raw)
                self.assertEqual(
                    scheduler._dspark_max_inflight_prefills,
                    expected,
                )
                self.assertEqual(logger.warnings, [])

    def test_malformed_values_warn_and_use_config_fallback(self):
        for raw in ("two", "2.0", "1x"):
            with self.subTest(raw=raw):
                scheduler, logger, _ = _load_scheduler(raw, config_cap=2)
                self.assertEqual(scheduler._dspark_max_inflight_prefills, 2)
                self.assertEqual(len(logger.warnings), 1)
                self.assertIn(ENV_NAME, logger.warnings[0])

    def test_config_fallback_is_also_clamped(self):
        scheduler, _, _ = _load_scheduler(None, config_cap=4)
        self.assertEqual(scheduler._dspark_max_inflight_prefills, 3)

    def test_schedule_uses_cached_cap_without_rereading_environment(self):
        scheduler, _, _ = _load_scheduler("2")
        inflight = object()
        scheduler._inflight_prefills.add(inflight)
        with mock.patch.dict(os.environ, {ENV_NAME: "two"}):
            self.assertEqual(scheduler.schedule(), 1)
        self.assertIn(inflight, scheduler._inflight_prefills)

    def test_admission_stops_at_cached_cap(self):
        scheduler, _, _ = _load_scheduler("1")
        scheduler.running.append(_mid_prefill())
        self.assertEqual(scheduler.schedule(), 0)

    def test_t1_lost_track_blocks_admission_and_warns(self):
        scheduler, logger, _ = _load_scheduler("1")
        scheduler.running.append(_mid_prefill())
        self.assertEqual(scheduler.schedule(), 0)
        self.assertEqual(len(logger.warnings), 1)
        self.assertIn("undercount: tracked=0 running=1", logger.warnings[0])

    def test_t2_completed_prefill_does_not_block_or_warn(self):
        scheduler, logger, _ = _load_scheduler("1")
        scheduler.running.append(_RunningReq("c", 1024, 1024))
        self.assertEqual(scheduler.schedule(), 1)
        self.assertEqual(logger.warnings, [])

    def test_t3_cap_two_admits_one_blocks_two(self):
        scheduler, logger, _ = _load_scheduler("2")
        one = _mid_prefill()
        scheduler.running.append(one)
        scheduler._inflight_prefills.add(one)
        self.assertEqual(scheduler.schedule(), 1)
        self.assertEqual(logger.warnings, [])

        scheduler, _, _ = _load_scheduler("2")
        # bookkeeping lost: prefills in running only -> gate must still block
        scheduler.running.extend([_mid_prefill(), _mid_prefill()])
        self.assertEqual(scheduler.schedule(), 0)

    def test_t4_async_kv_only_tracked_entry_is_tolerated(self):
        scheduler, logger, _ = _load_scheduler("1")
        scheduler._inflight_prefills.add(object())  # WAITING_FOR_REMOTE_KVS: not in running
        self.assertEqual(scheduler.schedule(), 1)
        self.assertEqual(logger.warnings, [])

    def test_init_logs_resolved_cap_and_env_once(self):
        scheduler, logger, _ = _load_scheduler("1")
        expected = "[issue27-hotfix] in-flight prefill cap=1 env='1' sched=%x" % id(
            scheduler
        )
        self.assertEqual(
            [line for line in logger.infos if "in-flight prefill cap=" in line],
            [expected],
        )

    def test_undercount_tripwire_is_bounded_at_sixteen(self):
        scheduler, logger, _ = _load_scheduler("1")
        scheduler.running.append(_mid_prefill())
        for _ in range(17):
            scheduler.waiting.append(_RunningReq("w", 0, 512))
            scheduler.schedule()
        self.assertEqual(len(logger.warnings), 16)
        self.assertIn("undercount: tracked=0 running=1", logger.warnings[0])

    def test_apply_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scheduler.py"
            path.write_text(FIXTURE)
            _apply_to(path)
            patched = path.read_text()
            _apply_to(path)
            self.assertEqual(path.read_text(), patched)

    def test_t6_single_chunk_same_step_not_counted(self):
        scheduler, logger, _ = _load_scheduler("1")
        scheduler.running.append(_RunningReq("a", 0, 100))
        scheduler.step_scheduled = {"a": 100}
        self.assertEqual(scheduler.schedule(), 1)
        self.assertEqual(logger.warnings, [])

    def test_t7_last_chunk_this_step_releases_slot(self):
        scheduler, logger, _ = _load_scheduler("1")
        last = _RunningReq("a", 21805, 22829)
        scheduler.running.append(last)
        scheduler.step_scheduled = {"a": 1024}
        scheduler._inflight_prefills.add(last)
        self.assertEqual(scheduler.schedule(), 1)
        self.assertEqual(logger.warnings, [])

    def test_t8_decoder_never_counted(self):
        scheduler, logger, _ = _load_scheduler("1")
        scheduler.running.append(
            _RunningReq("d", 1007, 1000, num_output_placeholders=7)
        )
        scheduler.step_scheduled = {"d": 7}
        self.assertEqual(scheduler.schedule(), 1)
        self.assertEqual(logger.warnings, [])

    def test_t9_mid_prefill_skipped_this_step_blocks_and_warns(self):
        scheduler, logger, _ = _load_scheduler("1")
        scheduler.running.append(_RunningReq("a", 1024, 22829))
        scheduler.step_scheduled = {}
        self.assertEqual(scheduler.schedule(), 0)
        self.assertEqual(len(logger.warnings), 1)
        self.assertIn("undercount: tracked=0 running=1", logger.warnings[0])

    def test_t10_mixed_burst_cap_two(self):
        scheduler, logger, _ = _load_scheduler("2")
        scheduler.running.append(_RunningReq("a", 0, 100))
        scheduler.running.append(_RunningReq("b", 0, 22829))
        scheduler.step_scheduled = {"a": 100, "b": 1024}
        scheduler._inflight_prefills.add(scheduler.running[1])
        scheduler.waiting = [_RunningReq("x", 0, 5000), _RunningReq("y", 0, 5000)]
        self.assertEqual(scheduler.schedule(), 1)
        self.assertEqual(logger.warnings, [])

    def test_t11_tripwire_true_positive_only(self):
        scheduler, logger, _ = _load_scheduler("1")
        scheduler.running.append(_RunningReq("b", 1024, 22829))
        scheduler.step_scheduled = {}
        scheduler.schedule()
        self.assertEqual(len(logger.warnings), 1)

        scheduler, logger, _ = _load_scheduler("1")
        scheduler.running.append(_RunningReq("a", 0, 100))
        scheduler.step_scheduled = {"a": 100}
        scheduler.schedule()
        self.assertEqual(logger.warnings, [])

    def test_t12_refuses_pre_r2_gate_and_status_labels(self):
        stale_text = None
        for ref in PRE_R2_REVS:
            try:
                got = subprocess.run(
                    ["git", "-C", str(ROOT), "show",
                     f"{ref}:patches/hotfix-dsv4-issue27-partial-prefill-concurrency.py"],
                    capture_output=True, text=True, check=True,
                ).stdout
            except subprocess.CalledProcessError:
                continue
            if R2_MARK not in got and "issue27-hotfix" in got:
                stale_text = got
                break

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scheduler.py"
            path.write_text(FIXTURE)
            self.assertIn("NOT APPLIED", _status_of(path))

            if stale_text is not None:
                _apply_to(path, stale_text)
            else:
                # No pre-r2 patcher reachable in history: synthesize the
                # stale state (gate marker present, no r2 marker).
                path.write_text(
                    FIXTURE + "\n# [issue27-hotfix] enforce "
                    "max_num_partial_prefills on admission\n"
                )
            self.assertIn("APPLIED (pre-r2, stale)", _status_of(path))
            stale_bytes = path.read_bytes()

            with self.assertRaises(SystemExit) as ctx:
                _apply_to(path)
            self.assertEqual(ctx.exception.code, 1)
            self.assertEqual(path.read_bytes(), stale_bytes)

            path.write_text(FIXTURE)
            _apply_to(path)
            first = path.read_bytes()
            self.assertIn("APPLIED (r2)", _status_of(path))
            _apply_to(path)
            self.assertEqual(path.read_bytes(), first)


if __name__ == "__main__":
    sys.exit(0 if unittest.main(verbosity=2) else 1)
