#!/usr/bin/env bash
# Build the scatter-gather copy kernel. Must run on each node (or build once and
# copy the .so, which is what up-dsv4-diskcache.sh does).
#
# This kernel is what makes large restores possible at all: the NVIDIA driver
# segfaults inside libcuda.so.1 above ~23,000 cuMemcpyBatchAsync descriptors,
# and a 1M-token restore needs ~246,000. See ../docs/KV-DISK-TIER.md.
set -e
KV_SRC="${KV_SRC:-/opt/dsv4-kv}"
IMG="${IMG:-vllm-dspark-runtime:dspark-nvfp4-stage-c}"
ARCH="${ARCH:-sm_121a}"     # GB10. Use sm_120a on consumer Blackwell.

docker run --rm --gpus all -v "$KV_SRC":/src --entrypoint bash "$IMG" -lc "
  export PATH=/opt/env/bin:\$PATH
  export LD_LIBRARY_PATH=/opt/env/lib:/opt/env/targets/sbsa-linux/lib:\$LD_LIBRARY_PATH
  nvcc -O3 -arch=$ARCH -shared -Xcompiler -fPIC \
       -o /src/libdsv4_batch_copy.so /src/dsv4_batch_copy.cu
  ls -la /src/libdsv4_batch_copy.so
"
echo "built $KV_SRC/libdsv4_batch_copy.so"
