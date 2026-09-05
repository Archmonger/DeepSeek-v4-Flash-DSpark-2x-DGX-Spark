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
