#!/usr/bin/env bash
# Serve the uncensored Qwen3-30B-A3B (abliterated) GGUF over the LAN as an
# OpenAI-compatible endpoint using llama.cpp's llama-server.
#
# Hardware: RTX 3060 12GB (CUDA0) + GTX 1080 8GB (CUDA1), Ryzen 5 5600X, 128GB RAM.
# The 30B MoE (18.5GB Q4) is split across BOTH cards via --tensor-split so the
# expert layers live in the 1080's VRAM instead of CPU RAM. Only a few expert
# layers stay on CPU (--n-cpu-moe) to leave room for the KV cache.
#
# Built against CUDA 12.9 (build-multigpu) because CUDA 13 dropped Pascal/sm_61
# support that the 1080 needs. Tuned 2026-06-11: prefill ~1000 tok/s, gen ~45,
# tool-calls ~3s. Raise N_CPU_MOE / shift TENSOR_SPLIT toward the 3060 if OOM.
set -euo pipefail

LLAMA_BIN="${LLAMA_BIN:-/home/akclark/llama.cpp/build-multigpu/bin/llama-server}"
MODEL="${MODEL:-/home/akclark/models/qwen3-abliterated-30b-Q4_K_M.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8081}"
CTX="${CTX:-40960}"             # model's native max window (YaRN-64K clamps here anyway, w/o gain)
PARALLEL="${PARALLEL:-1}"       # single agent sequence gets the full context
NGL="${NGL:-99}"               # all layers on GPU; experts spread by tensor-split
N_CPU_MOE="${N_CPU_MOE:-12}"    # expert layers kept on CPU (headroom for KV)
TENSOR_SPLIT="${TENSOR_SPLIT:-0.72,0.28}"  # CUDA0=3060 : CUDA1=1080
KV_TYPE="${KV_TYPE:-q8_0}"      # quantized KV cache to fit
THREADS="${THREADS:-6}"        # 5600X physical cores
ALIAS="${ALIAS:-qwen3-abliterated-30b}"

# Expose BOTH GPUs, pinned by UUID and ordered 3060(CUDA0) then 1080(CUDA1) so the
# tensor-split fractions map correctly regardless of PCI/index reordering.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-2400d806-cea5-c683-f4ad-459c4a156986,GPU-72be24e0-0677-7518-5c1a-effc1b25f2ea}"
# The multigpu binary links CUDA 12.9 runtime libs.
export LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}"

exec "$LLAMA_BIN" \
  --model "$MODEL" \
  --alias "$ALIAS" \
  --host "$HOST" --port "$PORT" \
  --ctx-size "$CTX" \
  --parallel "$PARALLEL" \
  --n-gpu-layers "$NGL" \
  --n-cpu-moe "$N_CPU_MOE" \
  --tensor-split "$TENSOR_SPLIT" \
  --cache-type-k "$KV_TYPE" \
  --cache-type-v "$KV_TYPE" \
  --threads "$THREADS" \
  --flash-attn on \
  --jinja \
  --metrics
