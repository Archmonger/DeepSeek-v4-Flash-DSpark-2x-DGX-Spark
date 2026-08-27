# EXPERIMENTAL: LMCache KV offload for the 2× Spark pair

Persistent KV cache across engine restarts: a ~107K-token context that costs
~65 s to re-prefill reloads in **~1.8–1.9 s** (measured n=2 on this recipe,
GB10 pair, `nvfp4_ds_mla`). KV is held by per-node `lmcache server` processes
(L1 CPU + filesystem L2 on the local NVMe) that survive engine restarts.

**Status: experimental.** Serving-path verified (boot, store, warm hits,
reload) across repeated trials; the *failure paths* upstream are still being
hardened — see Known issues. Do not enable on a pair you cannot restart.

## How it fits this recipe

- One `lmcache server` container per node (see `run-lmcache-server.sh`),
  ZMQ on the fabric IPs.
- `patch-compose-lmcache.py` generates `docker-compose.lmcache.yml` from
  `docker-compose.dspark.yml`: injects `--kv-transfer-config` for
  `LMCacheMPConnector` (gated on `DSPARK_ENABLE_LMCACHE=1`), passes
  `PYTHONHASHSEED=0`, and clears `PYTORCH_CUDA_ALLOC_CONF`
  (vLLM rejects KV connectors alongside `expandable_segments:True`).
- Launch with `COMPOSE_FILE=$PWD/docker-compose.lmcache.yml ./start-…sh`.

## Requirements baked into a derived image (the pinned Anemll image lacks them)

```
pip install --no-deps lmcache==0.5.4
pip install sortedcontainers aiofile aiofiles cupy-cuda13x
```
`cupy` matters: without it the server *silently* fails GPU-context creation
and every engine registration kills the vLLM head (LMCache #4759 covers the
fail-fast ask). The lmcache wheel's bundled `cuda_ops` is ABI-mismatched
against this image's torch — it soft-falls back to torch ops (works; slower
stores). Building it from source against the image's torch works
(`TORCH_CUDA_ARCH_LIST=12.1a`; the image's CUDA toolkit is header-trimmed —
fill cusparse/cusolver/cufft headers from the `nvidia-*-cu13` pip wheels).

## Non-negotiable configuration

- **`PYTHONHASHSEED=0` on every process** (servers AND engine): chunk keys use
  Python's randomized `hash()`; without pinning, every restart invalidates the
  entire cache (LMCache #1788).
- **L1 sized to hold your largest context** (`--l1-size-gb 12` for ~150K-token
  docs) — mid-store L1 eviction has stalled stores in our testing.
- **Treat the pair + servers as one lifecycle unit.** Restarting a server
  loses its GPU contexts; the current upstream code then parks every
  lookup-hit request forever (scheduler heartbeat bug — fix upstream as
  LMCache PR #4764 — plus non-propagating lookup errors). Until those land:
  run servers with `--restart no`, alert on server exit, and on ANY
  `No GPU context found` in a server log, restart the whole pair.

## Known issues (upstream)

- LMCache #4759 — the full GB10 field report (hang modes, evidence, stacks)
- LMCache PR #4764 — scheduler heartbeat never starts (dead servers stay
  "healthy" forever); merged = parks become graceful degradation
- LMCache PR #4754 — timeline-semaphore event IPC selector (defensive on
  driver 580/CUDA 13 platforms)
- Servers can exit under engine-boot memory pressure on 128 GB unified-memory
  nodes (engine weight-load spike + resident L1); monitor them.

## Usage

On each node, from the recipe checkout (derived image already built):

```
# 1. one server per node, bound to THAT node's fabric IP
./lmcache/run-lmcache-server.sh 192.168.104.10        # head
./lmcache/run-lmcache-server.sh 192.168.104.11        # worker

# 2. generate the compose overlay (once, same file both nodes)
python3 lmcache/patch-compose-lmcache.py \
  docker-compose.dspark.yml docker-compose.lmcache.yml \
  tcp://192.168.104.10:6667,tcp://192.168.104.11:6667

# 3. enable + launch through the normal recipe entry point
export DSPARK_ENABLE_LMCACHE=1
COMPOSE_FILE=$PWD/docker-compose.lmcache.yml ./start-deepseek-v4-flash.sh
```

With `DSPARK_ENABLE_LMCACHE` unset (or `0`) the generated compose boots the
stock recipe unchanged — the connector args are gated at runtime, so one
compose file serves both modes.

To verify it's working: the head engine log shows `LMCacheMPConnector`
at startup, and a repeated long-context request logs a lookup hit with TTFT
dropping from full-prefill cost to ~2 s.
