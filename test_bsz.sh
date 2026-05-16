#!/bin/bash
# Binary search for maximum batch size
# Each test runs in a fresh deepspeed process to avoid NCCL state corruption

cd /mnt/hdfs/wangchangyue/500xCompressor/codes

MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/wangchangyue/LLM/Qwen3-8B}"
START_BSZ="${START_BSZ:-1}"
MAX_BSZ="${MAX_BSZ:-16}"
NUM_MEM="${NUM_MEM:-256}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MASTER_PORT="${MASTER_PORT:-11470}"
NUM_GPUS="${NUM_GPUS:-8}"

echo "=============================================="
echo "Finding max batch size with DeepSpeed ZeRO-3"
echo "Model: $MODEL_PATH"
echo "GPUs: $NUM_GPUS"
echo "Search range: $START_BSZ - $MAX_BSZ"
echo "=============================================="

LOW=$START_BSZ
HIGH=$MAX_BSZ
MAX_FOUND=0

while [ $LOW -le $HIGH ]; do
    MID=$(( (LOW + HIGH) / 2 ))

    echo ""
    echo "[Range $LOW-$HIGH] Testing batch_size=$MID..."

    # Run test in fresh process
    # Exit code 0 = success, 1 = failed
    deepspeed --num_gpus=$NUM_GPUS --master_port=$MASTER_PORT test_bsz.py \
        --model_path "$MODEL_PATH" \
        --batch_size $MID \
        --num_mem $NUM_MEM \
        --max_length $MAX_LENGTH \
        > /tmp/bsz_test_$MID.log 2>&1

    RESULT=$?

    if [ $RESULT -eq 0 ]; then
        echo "  SUCCESS! batch_size=$MID works"
        MAX_FOUND=$MID
        LOW=$((MID + 1))
    else
        echo "  FAILED (exit code $RESULT)"
        HIGH=$((MID - 1))
    fi

    # Change port to avoid collision on next run
    MASTER_PORT=$((MASTER_PORT + 1))
done

echo ""
echo "=============================================="
echo "RESULTS:"
echo "  per_device_train_batch_size = $MAX_FOUND"
echo "  Total batch size ($NUM_GPUS GPUs) = $MAX_FOUND * $NUM_GPUS"
echo "=============================================="

if [ $MAX_FOUND -eq 0 ]; then
    echo "WARNING: Even batch_size=$START_BSZ failed!"
fi