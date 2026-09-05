# `kv-disk-tier` — disk backed KV cache

Backs the prefix cache with NVMe so a large context survives GPU KV eviction and
restores from disk instead of paying a cold prefill. Multi-node TP safe (per-node
sharded tier) and needs no vLLM source change — modules are bind-mounted.

## Two variants (pick for your vLLM tree)

| Variant | files | vLLM |
|---|---|---|
| `port` (0.25.x) | `dsv4_kv_disk_tier.py` / `dsv4_shard_tier.py` | Anemll `0.1.1` (vLLM 0.25.2) |
| `v021` | `dsv4_*_v021.py` | vLLM 0.21.1 images |

The Anemll image this repo ships is 0.25.x, so it uses the `port` files. The
`v021` pair is kept for older images and is not interchangeable.

## Build (once per node)

```bash
KV_SRC=/opt/dsv4-kv ./build.sh   # builds libdsv4_batch_copy.so (sm_121a)
```

The `.so` is the scatter-gather copy kernel that replaces `cuMemcpyBatchAsync`,
which segfaults in the driver above ~23k descriptors.

## Enable

Set in `.env.dspark` (start syncs it to the worker):

```bash
DSPARK_ENABLE_DISK_TIER=1      # the on/off switch
KV_SRC=/opt/dsv4-kv            # host dir with the modules + built .so (both nodes)
KVDISK_DIR=${HOME}/kvdisk      # per-node NVMe cache dir
KV_CPU_BYTES=4294967296        # pinned CPU staging tier; caps largest restorable prompt
KV_DISK_BYTES=150000000000     # per-node NVMe quota
```

`--kv-transfer-config` and the tier JSON are assembled from those knobs, and
`PYTORCH_CUDA_ALLOC_CONF` is unset automatically (the tier rejects
`expandable_segments`). `PYTHONHASHSEED=0` is required (block hashes are salted
from it). `gpu_clear.sh` clears a stale container and leftover `/dev/shm` staging
before a relaunch.

## Disable

Remove `DSPARK_ENABLE_DISK_TIER` (or set it to `0`). Everything else is inert.

## Recent fixes (vLLM 0.25.x `port` variant)

- **Store/load race (crash).** `_sliding_window_lookup_patched` now uses the
  vLLM 0.25.x `LookupResult` enum (`HIT` / `HIT_PENDING` / `RETRY` / `MISS`).
  The previous code used the old bool/`None` contract, so *every* lookup was
  counted as a hit — including mid-store `HIT_PENDING` blocks — and
  `update_state_after_alloc` handed them to `prepare_load()`, whose stock
  `assert block.is_ready` killed the EngineCore on any two requests sharing a
  prefix. `HIT_PENDING`/`RETRY` now defer the lookup instead.

- **Large-copy chunking.** The scatter-gather path now slices batches larger
  than `DSV4_MAX_COPIES_PER_BATCH` (default `8192`) with a per-slice stream
  sync, so a large restore (≈244k descriptors for 1M tokens) can't submit one
  unbounded batch to the driver.

## Known limitation — NVRM driver OOM on very large contexts

Large single-request contexts trigger a GPU-driver memory-descriptor exhaustion
(`NVRM: nvCheckOkFailedNoLog … _memdescAllocInternal … NV_ERR_NO_MEMORY`), which
is *not* host RAM and *not* fixed by the chunking above. Observed on this fleet:

- ~750K-token fill → container crash.
- ~293K-token fill → engine hang, and a subsequent `docker restart` can hard-hang
  the node (kernel alive, sshd dead) until the driver resets.

The disk tier's *restore* path is the same copy path the chunking bounds, but
the prefill/workspace allocations that exhaust the driver pool are not under
this module's control. **Keep single-request contexts ≤ ~60K tokens with the
disk tier enabled**; the tier is verified stable there. Larger contexts need the
driver-level memdesc issue addressed (or `DSPARK_ENABLE_DISK_TIER=0`).
