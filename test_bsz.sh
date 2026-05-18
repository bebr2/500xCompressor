#!/bin/bash
# Binary search for maximum batch size
# Each test runs in a fresh deepspeed process

cd /mnt/hdfs/wangchangyue/500xCompressor/codes

MODEL_PATH="/mnt/hdfs/wangchangyue/LLM/Qwen3-8B"
LORA_PATH="/mnt/hdfs/wangchangyue/500xCompressor/output/500xCompressor_pretrain-Qwen3-8B/checkpoint_best/pytorch_model.bin"
STAGE="finetune"       # pretrain or finetune
COMPRESSOR="500x"      # 500x or icae
MAX_LENGTH=512
MAX_QA_LEN=46
GRAD_ACCUM=8
MAX_STEPS=2
INCLUDE_EVAL=0
EVAL_BSZ=48
START_BSZ=1
MAX_BSZ=4
NUM_GPUS=8
MASTER_PORT=11470

echo "=============================================="
echo "Finding max batch size with DeepSpeed ZeRO-3"
echo "Model: $MODEL_PATH"
echo "Stage: $STAGE"
echo "Compressor: $COMPRESSOR"
echo "Max context length: $MAX_LENGTH"
echo "Gradient accumulation steps: $GRAD_ACCUM"
echo "Max optimizer steps: $MAX_STEPS"
echo "Include eval: $INCLUDE_EVAL"
if [ "$STAGE" = "finetune" ]; then
    echo "Max QA length: $MAX_QA_LEN"
    echo "LoRA path: $LORA_PATH"
fi
if [ "$INCLUDE_EVAL" = "1" ]; then
    echo "Eval batch size: $EVAL_BSZ"
fi
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
    CMD=(deepspeed --num_gpus=$NUM_GPUS --master_port=$MASTER_PORT test_bsz.py \
        --stage "$STAGE" \
        --compressor "$COMPRESSOR" \
        --model_path "$MODEL_PATH" \
        --batch_size $MID \
        --per_device_eval_batch_size "$EVAL_BSZ" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --max_steps "$MAX_STEPS" \
        --num_mem 256 \
        --max_length "$MAX_LENGTH" \
        --max_qa_len "$MAX_QA_LEN")

    if [ "$STAGE" = "finetune" ]; then
        CMD+=(--lora_path "$LORA_PATH")
    fi

    if [ "$INCLUDE_EVAL" = "1" ]; then
        CMD+=(--include_eval)
    fi

    "${CMD[@]}" > "$LOG_FILE" 2>&1

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
