#!/usr/bin/env python3
"""Generate docker-compose.lmcache.yml from docker-compose.dspark.yml.

Adds, gated on DSPARK_ENABLE_LMCACHE=1:
  - the LMCacheMPConnector --kv-transfer-config (hardcoded escaped JSON in the
    compose command: .env values are bash-sourced by the launcher, which
    strips quotes, so JSON cannot ride an env var in this stack)
  - PYTHONHASHSEED=0 in the container env (chunk keys use Python's randomized
    hash(); unpinned, every restart invalidates the whole cache)
  - clears PYTORCH_CUDA_ALLOC_CONF (vLLM rejects KV connectors alongside
    expandable_segments:True)

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
    + ' KVT_ARGS="--kv-transfer-config ' + kvt + '"; fi;',
    1,
)

a2 = "        $${VLLM_QUANTIZATION_ARGS}\n"
assert a2 in s, "anchor (serve args) not found"
s = s.replace(a2, a2 + "        $${KVT_ARGS}\n", 1)

a3 = 'PYTORCH_CUDA_ALLOC_CONF: "${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"'
assert a3 in s, "anchor (alloc conf) not found"
s = s.replace(a3, 'PYTORCH_CUDA_ALLOC_CONF: "${PYTORCH_CUDA_ALLOC_CONF_LMC:-}"', 1)

a4 = "      HF_HOME: /cache/huggingface\n"
assert a4 in s, "anchor (env block) not found"
s = s.replace(
    a4,
    a4
    + '      DSPARK_ENABLE_LMCACHE: "${DSPARK_ENABLE_LMCACHE:-0}"\n'
    + '      PYTHONHASHSEED: "0"\n',
    1,
)

open(dst, "w").write(s)
print("wrote", dst)
