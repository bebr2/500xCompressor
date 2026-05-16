cd /mnt/hdfs/wangchangyue/500xCompressor/codes/finetuning

deepspeed finetune_500xCompressor.py \
    --model_path /mnt/hdfs/wangchangyue/LLM/Qwen3-8B \
    --num_mem 256 \
    --max_length 2048 \
    --max_qa_len 46 \
    --train_text_path /mnt/hdfs/wangchangyue/500xCompressor/data/train_qa.jsonl \
    --test_text_path /mnt/hdfs/wangchangyue/500xCompressor/data/test_qa.jsonl \
    --lora_path /mnt/hdfs/wangchangyue/500xCompressor/output/500xCompressor_pretrain/checkpoint_best/pytorch_model.bin \
    --output_dir /mnt/hdfs/wangchangyue/500xCompressor/output/500xCompressor_finetune \
    --logging_dir /mnt/hdfs/wangchangyue/500xCompressor/logs/500xCompressor_finetune \
    --project_name 500xCompressor-finetuning \
    --deepspeed_config /mnt/hdfs/wangchangyue/500xCompressor/codes/deepspeed_configurations.json \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5 \
    --save_steps 100 \
    --eval_steps 500 \
    --warmup_steps 300
