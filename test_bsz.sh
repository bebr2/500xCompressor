#!/bin/bash
# Binary search for maximum batch size
# Each test runs in a fresh deepspeed process

cd /mnt/hdfs/wangchangyue/500xCompressor/codes

MODEL_PATH="/mnt/hdfs/wangchangyue/LLM/phi-4"
START_BSZ=1
MAX_BSZ=2
NUM_GPUS=8
MASTER_PORT=11470

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

    # Run test in fresh process, capture all output
    LOG_FILE="/tmp/bsz_test_$MID.log"
    deepspeed --num_gpus=$NUM_GPUS --master_port=$MASTER_PORT test_bsz.py \
        --model_path "$MODEL_PATH" \
        --batch_size $MID \
        --num_mem 256 \
        --max_length 2048 \
        > "$LOG_FILE" 2>&1

    RESULT=$?

    if [ $RESULT -eq 0 ]; then
        echo "  SUCCESS! batch_size=$MID works"
        MAX_FOUND=$MID
        LOW=$((MID + 1))
    else
        echo "  FAILED (exit code $RESULT)"
        echo ""
        echo "========== FULL ERROR LOG =========="
        cat "$LOG_FILE"
        echo "========== END ERROR LOG =========="
        HIGH=$((MID - 1))
    fi

    # Change port to avoid collision
    MASTER_PORT=$((MASTER_PORT + 1))
done

echo ""
echo "=============================================="
echo "RESULTS:"
echo "  per_device_train_batch_size = $MAX_FOUND"
echo "  Total batch size ($NUM_GPUS GPUs) = $((MAX_FOUND * NUM_GPUS))"
echo "=============================================="

if [ $MAX_FOUND -eq 0 ]; then
    echo "WARNING: Even batch_size=$START_BSZ failed!"
fi