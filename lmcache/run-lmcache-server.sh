#!/usr/bin/env bash
# Launch the per-node LMCache MP server. Run once per node with that node's
# fabric IP. Requires the derived image (see README.md).
set -euo pipefail
FABRIC_IP="${1:?usage: run-lmcache-server.sh <this-node-fabric-ip> [image]}"
IMAGE="${2:-dspark-vllm-gx10:lmcache054}"
DISK="${LMCACHE_DISK_DIR:-$HOME/lmcache-disk}"
mkdir -p "$DISK"
docker rm -f lmcache-server >/dev/null 2>&1 || true
# --restart no is deliberate: a dead server must be VISIBLE (see README —
# an auto-restarted empty server currently wedges the engine silently).
docker run -d --name lmcache-server --network host --ipc host --gpus all \
  --restart no \
  -e PYTHONHASHSEED=0 \
  -v "$DISK:/lmcache-disk" \
  --entrypoint lmcache "$IMAGE" server \
  --host "$FABRIC_IP" --port 6667 --chunk-size 256 \
  --l1-size-gb "${LMCACHE_L1_GB:-12}" --l1-use-lazy --eviction-policy LRU \
  --l2-adapter '{"type":"fs","base_path":"/lmcache-disk"}' \
  --disable-observability
echo "lmcache server up on ${FABRIC_IP}:6667 (L1 ${LMCACHE_L1_GB:-12}G, fs L2 at $DISK)"
