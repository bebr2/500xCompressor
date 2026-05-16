#!/bin/bash
# Find max batch size - each test is a fresh process

cd /mnt/hdfs/wangchangyue/500xCompressor/codes

MODEL_PATH="/mnt/hdfs/wangchangyue/LLM/Qwen3-8B"
START_BSZ=1
MAX_BSZ=16

echo "Testing batch_size=1 to see actual error..."
echo ""

# Run once with full output visible
deepspeed --num_gpus=8 --master_port=11470 test_bsz.py \
    --model_path "$MODEL_PATH" \
    --batch_size 1 \
    --num_mem 256 \
    --max_length 2048

echo ""
echo "Exit code: $?"