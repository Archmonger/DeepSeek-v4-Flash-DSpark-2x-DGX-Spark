# SPDX-License-Identifier: Apache-2.0
"""Apply the host-KV allocator patch in EVERY python process, engine and workers.

Python imports `sitecustomize` automatically at interpreter startup, which is the
only seam that reliably reaches vLLM's spawned worker processes. (The disk tier gets
there via `spec_module_path`, but that only exists when a --kv-transfer-config is
configured; this has to work without one so the host-KV change can be A/B tested
against the standing profile with exactly one variable moved.)

The patch cannot be applied at startup -- vLLM is not imported yet -- so this hooks
`SourceFileLoader.exec_module` and fires the moment
`vllm.v1.worker.gpu_model_runner` finishes loading, which is the earliest point at
which `GPUModelRunner` exists and still well before any KV cache is allocated.

Gated on DSV4_HOST_KV=1; a no-op otherwise.
"""
import os
import sys

if os.environ.get("DSV4_HOST_KV") == "1":
    import importlib.machinery

    _TARGET = "vllm.v1.worker.gpu_model_runner"
    _orig_exec_module = importlib.machinery.SourceFileLoader.exec_module

    def _exec_module(self, module):
        _orig_exec_module(self, module)
        if getattr(module, "__name__", None) != _TARGET:
            return
        try:
            import dsv4_vllm_patches

            dsv4_vllm_patches.apply_host_kv_alloc()
        except Exception as e:  # fail loud: a silent miss means EFAULT later
            print(
                f"[dsv4-sitecustomize] host-KV patch FAILED in pid {os.getpid()}: {e!r}",
                file=sys.stderr,
                flush=True,
            )
            raise

    importlib.machinery.SourceFileLoader.exec_module = _exec_module
    print(
        f"[dsv4-sitecustomize] host-KV hook armed in pid {os.getpid()}",
        file=sys.stderr,
        flush=True,
    )
