#!/usr/bin/env bash
set -euo pipefail

# 模型路径
MODEL_PATH=/data0/home/qwen/thgu/RAG/llm/Qwen3-14B

python -m vllm.entrypoints.api_server \
  --model "$MODEL_PATH" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --api-key your_secure_api_key \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --dtype float16 \
  --max-num-seqs 64 \
  --disable-cuda-graphs