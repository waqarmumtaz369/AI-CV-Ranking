#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate vLLM virtual environment
source vllm-env/bin/activate

vllm serve Qwen/Qwen2.5-0.5B \
  --gpu-memory-utilization 0.7 \
  --max-model-len 3072 \
  --max-num-seqs 6 \
  --host 0.0.0.0 \
  --port 8000
