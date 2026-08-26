#!/usr/bin/env python3
"""Regression tests for issue #117's opt-in TP dispatch trace."""

from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HOTFIX_PATH = REPO_ROOT / "patches/hotfix-vllm-issue117-dispatch-trace.py"


def _load_hotfix():
    spec = importlib.util.spec_from_file_location("issue117_dispatch_hotfix", HOTFIX_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {HOTFIX_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HOTFIX = _load_hotfix()


def _pristine_fixture() -> str:
    header = '''from __future__ import annotations

import os
import pickle
import time
import traceback
from enum import Enum
from functools import partial
from typing import Any


class _Logger:
    def __init__(self):
        self.records = []

    def info(self, message, *args):
        self.records.append(message % args)

    def exception(self, message, *args):
        self.records.append(message % args)


_LOGGER = _Logger()


def init_logger(_name):
    return _LOGGER

'''
    support = '''
class _Cloudpickle:
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL

    @staticmethod
    def dumps(value, protocol=None):
        return pickle.dumps(value, protocol=protocol)

    @staticmethod
    def loads(value):
        return pickle.loads(value)


cloudpickle = _Cloudpickle()


class AsyncModelRunnerOutput:
    def __init__(self, output):
        self.output = output

    def get_output(self):
        return self.output


class _Queue:
    def __init__(self, items=None, shared=None):
        self.items = shared if shared is not None else list(items or ())
        self.current_idx = 0

    def enqueue(self, value, timeout=None):
        self.items.append(value)
        self.current_idx = (self.current_idx + 1) % 10

    def dequeue(self, timeout=None, indefinite=False):
        if not self.items:
            raise StopIteration
        value = self.items.pop(0)
        self.current_idx = (self.current_idx + 1) % 10
        return value


class _Worker:
    def execute_model(self):
        return "worker-result"


class MultiprocExecutor:
    def collective_rpc(self, method, args=(), kwargs=None, output_rank=None):
        deadline = None
        kwargs = kwargs or {}
'''
    response_setup = '''        response_mqs = self.response_mqs
        if output_rank is not None:
            response_mqs = (response_mqs[output_rank],)
'''
    driver_return = '''        return get_response()


class WorkerProc:
    class ResponseStatus(Enum):
        SUCCESS = 1
        FAILURE = 2

'''
    async_wrapper = '''
    def async_output_busy_loop(self):
'''
    return (
        header
        + HOTFIX.OLD_MODULE_GATE
        + support
        + HOTFIX.OLD_DRIVER_DISPATCH
        + response_setup
        + HOTFIX.OLD_DRIVER_RESPONSE
        + driver_return
        + HOTFIX.OLD_WORKER_ENQUEUE_OUTPUT
        + "\n"
        + HOTFIX.OLD_WORKER_HANDLE_OUTPUT
        + async_wrapper
        + HOTFIX.OLD_WORKER_ASYNC_LOOP
        + "\n"
        + HOTFIX.OLD_WORKER_BUSY_LOOP
    )


def _execute_fixture(source: str, trace_enabled: bool):
    previous = os.environ.get("DSPARK_ISSUE117_DISPATCH_TRACE")
    try:
        if trace_enabled:
            os.environ["DSPARK_ISSUE117_DISPATCH_TRACE"] = "1"
        else:
            os.environ.pop("DSPARK_ISSUE117_DISPATCH_TRACE", None)
        namespace: dict[str, object] = {}
        exec(compile(source, "multiproc_executor.py", "exec"), namespace)
    finally:
        if previous is None:
            os.environ.pop("DSPARK_ISSUE117_DISPATCH_TRACE", None)
        else:
            os.environ["DSPARK_ISSUE117_DISPATCH_TRACE"] = previous

    queue_type = namespace["_Queue"]
    response_status = namespace["WorkerProc"].ResponseStatus

    shared_messages: list[object] = []
    driver = namespace["MultiprocExecutor"]()
    driver.rpc_broadcast_mq = queue_type(shared=shared_messages)
    driver.response_mqs = [queue_type([(response_status.SUCCESS, "driver-result")])]
    result = driver.collective_rpc("execute_model", output_rank=0)
    wire_message = tuple(shared_messages[0])

    worker = namespace["WorkerProc"]()
    worker.rpc_broadcast_mq = queue_type(shared=shared_messages)
    worker.worker_response_mq = queue_type()
    worker.worker = namespace["_Worker"]()
    worker.rank = 0
    worker.use_async_scheduling = False
    try:
        worker.worker_busy_loop()
    except StopIteration:
        pass
    else:
        raise AssertionError("worker loop fixture did not stop")

    return namespace["_LOGGER"].records, driver, worker, result, wire_message


class Issue117DispatchTraceTest(unittest.TestCase):
    def test_trace_correlates_driver_and_worker_without_wire_changes(self) -> None:
        patched, status = HOTFIX.patch_text(_pristine_fixture())
        self.assertEqual(status, "applied")

        records, driver, worker, result, wire_message = _execute_fixture(patched, True)
        self.assertEqual(result, "driver-result")
        self.assertEqual(
            records,
            [
                "[issue117-dispatch-trace] disp seq=1 method=execute_model out_rank=0 idx=0 pid="
                + str(os.getpid()),
                "[issue117-dispatch-trace] resp_ok seq=1 method=execute_model pid="
                + str(os.getpid()),
                "[issue117-dispatch-trace] recv seq=1 method=execute_model out_rank=0 idx=0 rank=0 pid="
                + str(os.getpid()),
                "[issue117-dispatch-trace] done seq=1 method=execute_model rank=0 pid="
                + str(os.getpid()),
                "[issue117-dispatch-trace] resp seq=1 status=SUCCESS idx=0 rank=0 pid="
                + str(os.getpid()),
            ],
        )
        self.assertEqual(wire_message, ("execute_model", (), {}, 0))
        self.assertEqual(worker.worker_response_mq.items[0][0].name, "SUCCESS")

    def test_disabled_trace_emits_nothing_and_allocates_no_counters(self) -> None:
        patched, status = HOTFIX.patch_text(_pristine_fixture())
        self.assertEqual(status, "applied")

        records, driver, worker, result, _ = _execute_fixture(patched, False)
        self.assertEqual(result, "driver-result")
        self.assertEqual(records, [])
        self.assertFalse(hasattr(driver, "_issue117_disp_seq"))
        self.assertFalse(hasattr(worker, "_issue117_recv_seq"))

    def test_cli_is_atomic_idempotent_and_fail_closed(self) -> None:
        pristine = _pristine_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "multiproc_executor.py"
            target.write_text(pristine)
            target.chmod(0o640)

            applied = subprocess.run(
                [sys.executable, str(HOTFIX_PATH), str(target)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(applied.returncode, 0, applied.stderr)
            self.assertIn("applied", applied.stdout)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o640)

            first_bytes = target.read_bytes()
            again = subprocess.run(
                [sys.executable, str(HOTFIX_PATH), str(target)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertIn("already applied", again.stdout)
            self.assertEqual(target.read_bytes(), first_bytes)

            status_result = subprocess.run(
                [sys.executable, str(HOTFIX_PATH), "--status", str(target)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(status_result.returncode, 0, status_result.stderr)
            self.assertIn("APPLIED", status_result.stdout)

            drifted = pristine.replace("self.rpc_broadcast_mq.enqueue", "self.rpc_broadcast_mq.put", 1)
            target.write_text(drifted)
            before = target.read_bytes()
            refused = subprocess.run(
                [sys.executable, str(HOTFIX_PATH), str(target)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(refused.returncode, 1)
            self.assertIn("source drift", refused.stderr)
            self.assertEqual(target.read_bytes(), before)




if __name__ == "__main__":
    unittest.main()
