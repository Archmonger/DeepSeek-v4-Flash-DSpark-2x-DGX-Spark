#!/usr/bin/env python3
"""Opt-in dispatch/completion trace for the issue #117 TP=2 divergence experiment.

The pinned Anemll image (vLLM 0.25.2.dev0) drives TP workers from one
EngineCore/driver process through a single-writer shm-ring broadcast queue
(``MultiprocExecutor.collective_rpc`` -> ``WorkerProc.worker_busy_loop``) and
collects per-rank responses through per-worker response queues. When a step
hangs there is no way to tell whether the driver never delivered the step to a
worker, a worker received it and stalled on device execution, or a finished
response never reached the driver. This startup patch instruments
``vllm/v1/executor/multiproc_executor.py`` -- and only that file -- with
compact, structured, single-line records gated behind
``DSPARK_ISSUE117_DISPATCH_TRACE=1`` (off by default: one module-constant
branch per site, no records, no counters).

Records (all prefixed ``[issue117-dispatch-trace]``):

  driver  disp          seq method out_rank idx pid   dispatch enqueue starts
  driver  resp_ok       seq method pid                every awaited rank replied
  driver  resp_timeout  seq method rank idx pid       dequeue deadline expired
  driver  resp_fail     seq method rank pid           worker sent FAILURE
  worker  recv          seq method out_rank idx rank pid   dispatch dequeued
  worker  done          seq method rank pid           RPC callable returned
  worker  fail          seq method rank pid           RPC callable raised
  worker  resp          seq status idx rank pid       response enqueue starts

``seq`` is the per-endpoint FIFO position of the message in its queue; the
broadcast queue has exactly one writer and in-order readers, so driver ``disp
seq=N`` and worker ``recv seq=N`` name the same message on every rank -- no
field is added to the wire protocol. ``idx`` is the shm ring slot of the
message (``MessageQueue.current_idx``; -1 when the endpoint has no local shm
ring, e.g. cross-node zmq readers). Engine generation is derived, not
duplicated: ``EngineCore.step`` dispatches exactly one ``execute_model`` per
generation, so the running count of ``disp method=execute_model`` records is
the generation number (this image has no step-id field in SchedulerOutput and
no step counter in the non-DP EngineCore, so ``vllm/v1/engine/core.py`` is
deliberately left untouched).

Reading a hang: ``disp`` without ``recv`` on some rank = lost/undelivered
dispatch (driver blocked inside enqueue also logs the stock ring-buffer
warning); ``recv`` without ``done``/``fail`` = the callable is stuck on the
device; ``done`` without ``resp`` under async scheduling = the async output
wait (``AsyncModelRunnerOutput.get_output``) is stuck; ``resp`` without
driver ``resp_ok``/``resp_timeout`` progress = response transport loss.

Known limits: ``disp``/``resp`` precede their (blocking) enqueue writes by
design; a worker killed mid-call emits nothing further (the executor's death
monitor logs that); cross-node zmq PUB/SUB readers have no backpressure, so
seq correlation on remote ranks assumes no high-water-mark drop -- the issue
#117 topology (single node, TP=2) uses only the lossless local shm ring.

Idempotent and fail-closed: all anchors are byte-exact against the pinned
image source and preflighted before any write; the single target file is
published atomically (same-directory temp file + fsync + os.replace, mode
preserved, directory fsync best-effort) and verified after publish, restoring
the original bytes if verification fails. Usage::

    hotfix-vllm-issue117-dispatch-trace.py [--status] [MULTIPROC_EXECUTOR.py]

The optional positional target overrides the default installed path for
fixture tests. Not wired into any entrypoint by default; integrators invoke
it explicitly.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py"
)
LABEL = "issue117 dispatch trace            :"

OLD_MODULE_GATE = '''logger = init_logger(__name__)
'''
NEW_MODULE_GATE = OLD_MODULE_GATE + '''
# [issue117-dispatch-trace] opt-in TP dispatch diagnostics (issue #117).
# DSPARK_ISSUE117_DISPATCH_TRACE=1 emits one compact logger.info line per
# phase to discriminate a lost/undelivered dispatch from a device execution
# stall:
#   driver: disp / resp_ok / resp_timeout / resp_fail
#   worker: recv / done / fail / resp
# seq is the per-endpoint FIFO position of the message in its queue. The
# broadcast queue has exactly one writer and in-order readers, so driver
# disp seq=N and worker recv seq=N are the same message. idx is the shm
# ring slot of the message (MessageQueue.current_idx; -1 = endpoint reads
# or writes without a local shm ring). No field is added to the wire
# protocol and no request/tensor content is ever logged.
_ISSUE117_TRACE = os.environ.get("DSPARK_ISSUE117_DISPATCH_TRACE") == "1"


def _issue117_next_seq(obj: Any, attr: str) -> int:
    # [issue117-dispatch-trace] per-endpoint FIFO message counter.
    seq = getattr(obj, attr, 0) + 1
    setattr(obj, attr, seq)
    return seq


def _issue117_method_name(method: Any) -> str:
    # [issue117-dispatch-trace] never log pickled callables, only names.
    return method if isinstance(method, str) else "<serialized-callable>"
'''
OLD_DRIVER_DISPATCH = '''        if isinstance(method, str):
            send_method = method
        else:
            send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)
        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))
'''
NEW_DRIVER_DISPATCH = '''        if isinstance(method, str):
            send_method = method
        else:
            send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)
        # [issue117-dispatch-trace] driver dispatch record, before the
        # (potentially blocking) broadcast enqueue: a driver stuck waiting
        # for a free ring slot still shows disp with no matching recv.
        i117_seq = 0
        if _ISSUE117_TRACE:
            i117_seq = _issue117_next_seq(self, "_issue117_disp_seq")
            logger.info(
                "[issue117-dispatch-trace] disp seq=%d method=%s out_rank=%s"
                " idx=%d pid=%d",
                i117_seq,
                _issue117_method_name(method),
                output_rank,
                self.rpc_broadcast_mq.current_idx,
                os.getpid(),
            )
        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))
'''
OLD_DRIVER_RESPONSE = '''        def get_response():
            responses = []
            for mq in response_mqs:
                dequeue_timeout = (
                    None if deadline is None else (deadline - time.monotonic())
                )
                try:
                    status, result = mq.dequeue(timeout=dequeue_timeout)
                except TimeoutError as e:
                    raise TimeoutError(f"RPC call to {method} timed out.") from e
                if status != WorkerProc.ResponseStatus.SUCCESS:
                    raise RuntimeError(
                        f"Worker failed with error '{result}', please check the"
                        " stack trace above for the root cause"
                    )
                responses.append(result)
            return responses[0] if output_rank is not None else responses
'''
NEW_DRIVER_RESPONSE = '''        def get_response():
            responses = []
            for mq_index, mq in enumerate(response_mqs):
                # [issue117-dispatch-trace] rank owning the response queue
                # we block on next (response_mqs is rank-ordered when
                # waiting on every rank).
                i117_rank = output_rank if output_rank is not None else mq_index
                dequeue_timeout = (
                    None if deadline is None else (deadline - time.monotonic())
                )
                try:
                    status, result = mq.dequeue(timeout=dequeue_timeout)
                except TimeoutError as e:
                    if _ISSUE117_TRACE:
                        logger.info(
                            "[issue117-dispatch-trace] resp_timeout seq=%d"
                            " method=%s rank=%s idx=%d pid=%d",
                            i117_seq,
                            _issue117_method_name(method),
                            i117_rank,
                            mq.current_idx,
                            os.getpid(),
                        )
                    raise TimeoutError(f"RPC call to {method} timed out.") from e
                if status != WorkerProc.ResponseStatus.SUCCESS:
                    if _ISSUE117_TRACE:
                        logger.info(
                            "[issue117-dispatch-trace] resp_fail seq=%d"
                            " method=%s rank=%s pid=%d",
                            i117_seq,
                            _issue117_method_name(method),
                            i117_rank,
                            os.getpid(),
                        )
                    raise RuntimeError(
                        f"Worker failed with error '{result}', please check the"
                        " stack trace above for the root cause"
                    )
                responses.append(result)
            if _ISSUE117_TRACE:
                logger.info(
                    "[issue117-dispatch-trace] resp_ok seq=%d method=%s pid=%d",
                    i117_seq,
                    _issue117_method_name(method),
                    os.getpid(),
                )
            return responses[0] if output_rank is not None else responses
'''
OLD_WORKER_BUSY_LOOP = '''    def worker_busy_loop(self):
        """Main busy loop for Multiprocessing Workers"""
        assert self.rpc_broadcast_mq is not None
        while True:
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(
                indefinite=True
            )
            try:
                if isinstance(method, str):
                    func = getattr(self.worker, method)
                elif isinstance(method, bytes):
                    func = partial(cloudpickle.loads(method), self.worker)

                output = func(*args, **kwargs)

                if output_rank is None or self.rank == output_rank:
                    self.handle_output(output)
            except Exception as e:
                # Notes have been introduced in python 3.11
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception("WorkerProc hit an exception.")
                # exception might not be serializable, so we convert it to
                # string, only for logging purpose.
                if output_rank is None or self.rank == output_rank:
                    self.handle_output(e)
'''
NEW_WORKER_BUSY_LOOP = '''    def worker_busy_loop(self):
        """Main busy loop for Multiprocessing Workers"""
        assert self.rpc_broadcast_mq is not None
        while True:
            # [issue117-dispatch-trace] shm ring slot the next message is
            # read from; equals the driver-side idx of the same message.
            i117_idx = self.rpc_broadcast_mq.current_idx
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(
                indefinite=True
            )
            i117_seq = 0
            if _ISSUE117_TRACE:
                i117_seq = _issue117_next_seq(self, "_issue117_recv_seq")
                logger.info(
                    "[issue117-dispatch-trace] recv seq=%d method=%s"
                    " out_rank=%s idx=%d rank=%d pid=%d",
                    i117_seq,
                    _issue117_method_name(method),
                    output_rank,
                    i117_idx,
                    self.rank,
                    os.getpid(),
                )
            try:
                if isinstance(method, str):
                    func = getattr(self.worker, method)
                elif isinstance(method, bytes):
                    func = partial(cloudpickle.loads(method), self.worker)

                output = func(*args, **kwargs)

                if _ISSUE117_TRACE:
                    logger.info(
                        "[issue117-dispatch-trace] done seq=%d method=%s"
                        " rank=%d pid=%d",
                        i117_seq,
                        _issue117_method_name(method),
                        self.rank,
                        os.getpid(),
                    )
                if output_rank is None or self.rank == output_rank:
                    self.handle_output(output, i117_seq)
            except Exception as e:
                # Notes have been introduced in python 3.11
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception("WorkerProc hit an exception.")
                if _ISSUE117_TRACE:
                    logger.info(
                        "[issue117-dispatch-trace] fail seq=%d method=%s"
                        " rank=%d pid=%d",
                        i117_seq,
                        _issue117_method_name(method),
                        self.rank,
                        os.getpid(),
                    )
                # exception might not be serializable, so we convert it to
                # string, only for logging purpose.
                if output_rank is None or self.rank == output_rank:
                    self.handle_output(e, i117_seq)
'''
OLD_WORKER_ENQUEUE_OUTPUT = '''    def enqueue_output(self, output: Any):
        """Prepares output from the worker and enqueues it to the
        worker_response_mq. If the output is an Exception, it is
        converted to a FAILURE response.
        """
        if isinstance(output, AsyncModelRunnerOutput):
            try:
                output = output.get_output()
            except Exception as e:
                logger.exception("Error getting async model runner output")
                output = e

        if isinstance(output, Exception):
            result = (WorkerProc.ResponseStatus.FAILURE, str(output))
        else:
            result = (WorkerProc.ResponseStatus.SUCCESS, output)
        if (response_mq := self.worker_response_mq) is not None:
            response_mq.enqueue(result)
'''
NEW_WORKER_ENQUEUE_OUTPUT = '''    def enqueue_output(self, output: Any, i117_seq: int = 0):
        """Prepares output from the worker and enqueues it to the
        worker_response_mq. If the output is an Exception, it is
        converted to a FAILURE response.
        """
        if isinstance(output, AsyncModelRunnerOutput):
            try:
                output = output.get_output()
            except Exception as e:
                logger.exception("Error getting async model runner output")
                output = e

        if isinstance(output, Exception):
            result = (WorkerProc.ResponseStatus.FAILURE, str(output))
        else:
            result = (WorkerProc.ResponseStatus.SUCCESS, output)
        if (response_mq := self.worker_response_mq) is not None:
            # [issue117-dispatch-trace] response record, after any async
            # output wait (a device stall parks between done and resp) and
            # before the response enqueue itself.
            if _ISSUE117_TRACE:
                logger.info(
                    "[issue117-dispatch-trace] resp seq=%d status=%s idx=%d"
                    " rank=%d pid=%d",
                    i117_seq,
                    result[0].name,
                    response_mq.current_idx,
                    self.rank,
                    os.getpid(),
                )
            response_mq.enqueue(result)
'''
OLD_WORKER_HANDLE_OUTPUT = '''    def handle_output(self, output: Any):
        """Handles output from the worker. If async scheduling is enabled,
        it is passed to the async_output_busy_loop thread. Otherwise, it is
        enqueued directly to the worker_response_mq.
        """
        if self.use_async_scheduling:
            self.async_output_queue.put(output)
        else:
            self.enqueue_output(output)
'''
NEW_WORKER_HANDLE_OUTPUT = '''    def handle_output(self, output: Any, i117_seq: int = 0):
        """Handles output from the worker. If async scheduling is enabled,
        it is passed to the async_output_busy_loop thread. Otherwise, it is
        enqueued directly to the worker_response_mq.
        """
        if self.use_async_scheduling:
            self.async_output_queue.put((output, i117_seq))
        else:
            self.enqueue_output(output, i117_seq)
'''
OLD_WORKER_ASYNC_LOOP = '''        while True:
            output = self.async_output_queue.get()
            self.enqueue_output(output)
'''
NEW_WORKER_ASYNC_LOOP = '''        while True:
            output, i117_seq = self.async_output_queue.get()
            self.enqueue_output(output, i117_seq)
'''

HUNKS: tuple[tuple[str, str, str], ...] = (
    ("module-gate", OLD_MODULE_GATE, NEW_MODULE_GATE),
    ("driver-dispatch", OLD_DRIVER_DISPATCH, NEW_DRIVER_DISPATCH),
    ("driver-response", OLD_DRIVER_RESPONSE, NEW_DRIVER_RESPONSE),
    ("worker-busy-loop", OLD_WORKER_BUSY_LOOP, NEW_WORKER_BUSY_LOOP),
    ("worker-enqueue-output", OLD_WORKER_ENQUEUE_OUTPUT, NEW_WORKER_ENQUEUE_OUTPUT),
    ("worker-handle-output", OLD_WORKER_HANDLE_OUTPUT, NEW_WORKER_HANDLE_OUTPUT),
    ("worker-async-loop", OLD_WORKER_ASYNC_LOOP, NEW_WORKER_ASYNC_LOOP),
)
# OLD_MODULE_GATE is the first line of NEW_MODULE_GATE, so it still counts
# exactly once after a successful apply; every other OLD block is rewritten.
_OLD_COUNTS_AFTER_APPLY = (1, 0, 0, 0, 0, 0, 0)


def patch_text(source: str) -> tuple[str, str]:
    """Return updated source and applied/skipped/drift status."""
    old_counts = tuple(source.count(old) for _, old, _ in HUNKS)
    new_counts = tuple(source.count(new) for _, _, new in HUNKS)
    if new_counts == (1,) * len(HUNKS) and old_counts == _OLD_COUNTS_AFTER_APPLY:
        return source, "skipped"
    if old_counts == (1,) * len(HUNKS) and new_counts == (0,) * len(HUNKS):
        updated = source
        for _, old, new in HUNKS:
            updated = updated.replace(old, new, 1)
        compile(updated, "multiproc_executor.py", "exec")
        return updated, "applied"
    detail = ",".join(
        f"{name}:old={old_count},new={new_count}"
        for (name, _, _), old_count, new_count in zip(HUNKS, old_counts, new_counts)
        if not (old_count == 1 and new_count == 0)
    )
    return source, f"drift:{detail}"


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _atomic_write(target: Path, data: bytes, mode: int) -> None:
    """Same-directory temp file + fsync + os.replace; no partial target state."""
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix="." + target.name + ".", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, target)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    _fsync_dir(target.parent)


def main(argv: list[str]) -> int:
    status_only = len(argv) > 1 and argv[1] == "--status"
    remaining = argv[2:] if status_only else argv[1:]
    if len(remaining) > 1:
        print(
            f"usage: {argv[0]} [--status] [MULTIPROC_EXECUTOR.py]",
            file=sys.stderr,
        )
        return 2
    target = Path(remaining[0]) if remaining else DEFAULT_TARGET
    if not target.is_file():
        print(f"[issue117-dispatch-trace] missing {target}", file=sys.stderr)
        return 1

    original = target.read_bytes()
    updated, status = patch_text(original.decode("utf-8"))

    if status_only:
        if status == "skipped":
            print(LABEL, "APPLIED")
            return 0
        detail = "patchable" if status == "applied" else status
        print(LABEL, f"NOT APPLIED ({detail})")
        return 1

    if status == "skipped":
        print(f"[issue117-dispatch-trace] already applied to {target}")
        return 0
    if status != "applied":
        print(
            f"[issue117-dispatch-trace] source drift; refusing to patch ({status})",
            file=sys.stderr,
        )
        return 1

    mode = target.stat().st_mode & 0o7777
    data = updated.encode("utf-8")
    _atomic_write(target, data, mode)
    if target.read_bytes() != data:
        _atomic_write(target, original, mode)
        print(
            "[issue117-dispatch-trace] post-write verification failed; "
            f"restored original {target}",
            file=sys.stderr,
        )
        return 1
    print(f"[issue117-dispatch-trace] applied to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
