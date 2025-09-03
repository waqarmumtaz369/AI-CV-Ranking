#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vllm serve Qwen/Qwen3-0.6B \
  --gpu-memory-utilization 0.7 \
  --max-model-len 2048 \
  --max-num-seqs 4 \
  --host 0.0.0.0 \
  --port 8000