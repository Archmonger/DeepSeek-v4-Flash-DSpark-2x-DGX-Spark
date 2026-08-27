#!/usr/bin/env python3
"""Generate docker-compose.lmcache.yml from docker-compose.dspark.yml.

Every LMCache-specific change is applied inside one entrypoint branch that is
taken only when DSPARK_ENABLE_LMCACHE is exactly "1" (repo convention, same
shape as DSPARK_ENABLE_ISSUE31_GPU_HOTFIX / DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX).
Inside that branch, immediately before `exec vllm serve`:
  - the LMCacheMPConnector --kv-transfer-config (hardcoded escaped JSON in the
    compose command: .env values are bash-sourced by the launcher, which
    strips quotes, so JSON cannot ride an env var in this stack)
  - export PYTHONHASHSEED=0 (chunk keys use Python's randomized hash();
    unpinned, every restart invalidates the whole cache)
  - unset PYTORCH_CUDA_ALLOC_CONF (vLLM rejects KV connectors alongside
    expandable_segments:True)

With the flag unset or 0 the branch is not taken, so the engine process gets
byte-identical env and argv to stock: the stock PYTORCH_CUDA_ALLOC_CONF
service env entry is left untouched and no PYTHONHASHSEED entry is added.
The only additions to the rendered config are inert: the DSPARK_ENABLE_LMCACHE
pass-through (default "0") and an empty $${KVT_ARGS} expansion.

Usage:
  patch-compose-lmcache.py <src> <dst> <server-urls>
  server-urls: comma-separated, e.g.
    tcp://192.168.104.10:6667,tcp://192.168.104.11:6667
"""
import sys

if len(sys.argv) != 4:
    sys.exit(__doc__)
src, dst, urls = sys.argv[1], sys.argv[2], sys.argv[3]
if any(c in urls for c in ' "\\'):
    sys.exit("server-urls must not contain spaces, quotes, or backslashes")
s = open(src).read()

a1 = 'if [ -n "$${DSPARK_REVISION:-}" ]; then REVISION_ARGS="--revision $${DSPARK_REVISION}"; fi;'
assert a1 in s, "anchor (REVISION_ARGS) not found — compose layout changed?"
kvt = (
    '{\\"kv_connector\\":\\"LMCacheMPConnector\\",\\"kv_role\\":\\"kv_both\\",'
    '\\"kv_connector_extra_config\\":{\\"lmcache.mp.server_urls\\":\\"' + urls + '\\"}}'
)
s = s.replace(
    a1,
    a1
    + '\n        KVT_ARGS="";'
    + '\n        if [ "$${DSPARK_ENABLE_LMCACHE:-0}" = "1" ]; then'
    + ' KVT_ARGS="--kv-transfer-config ' + kvt + '";'
    + ' export PYTHONHASHSEED=0;'
    + ' unset PYTORCH_CUDA_ALLOC_CONF; fi;',
    1,
)

a2 = "        $${VLLM_QUANTIZATION_ARGS}\n"
assert a2 in s, "anchor (serve args) not found"
s = s.replace(a2, a2 + "        $${KVT_ARGS}\n", 1)

a3 = 'PYTORCH_CUDA_ALLOC_CONF: "${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"'
assert a3 in s, "anchor (alloc conf) not found — compose layout changed?"

a4 = "      HF_HOME: /cache/huggingface\n"
assert a4 in s, "anchor (env block) not found"
s = s.replace(
    a4,
    a4 + '      DSPARK_ENABLE_LMCACHE: "${DSPARK_ENABLE_LMCACHE:-0}"\n',
    1,
)

# Opt-in boundary, enforced here so a future edit cannot quietly reintroduce an
# unconditional env change: with the flag off the engine must see stock env.
assert a3 in s, "generated compose must leave the stock PYTORCH_CUDA_ALLOC_CONF entry intact"
assert "\n      PYTHONHASHSEED:" not in s, "PYTHONHASHSEED must be exported inside the gate, not set in the service env"
assert s.count("DSPARK_ENABLE_LMCACHE") == 3, "expected exactly one gate plus one env pass-through"

open(dst, "w").write(s)
print("wrote", dst)
