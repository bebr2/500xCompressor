deepspeed test_bsz.py \
    --model_path /mnt/hdfs/wangchangyue/LLM/Qwen3-8B \
    --num_mem 256 \
    --max_length 2048 \
    --test_mode pretrain \
    --start_bsz 2 \
    --max_bsz_limit 16