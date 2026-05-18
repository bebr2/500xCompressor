cd /mnt/hdfs/wangchangyue/500xCompressor/codes

python convert_checkpoint.py /mnt/hdfs/wangchangyue/500xCompressor/output/ICAE_pretrain-Qwen3-8B

cd /mnt/hdfs/wangchangyue/500xCompressor/codes/finetuning

deepspeed  --master_port=11470 finetune_ICAE.py \
    --model_path /mnt/hdfs/wangchangyue/LLM/Qwen3-8B \
    --num_mem 256 \
    --max_length 512 \
    --max_qa_len 46 \
    --train_text_path /mnt/hdfs/wangchangyue/500xCompressor/data/train_qa.jsonl \
    --test_text_path /mnt/hdfs/wangchangyue/500xCompressor/data/test_qa.jsonl \
    --lora_path /mnt/hdfs/wangchangyue/500xCompressor/output/ICAE_pretrain-Qwen3-8B/checkpoint_best/pytorch_model.bin \
    --output_dir /mnt/hdfs/wangchangyue/500xCompressor/output/ICAE_finetune-Qwen3-8B \
    --logging_dir /mnt/hdfs/wangchangyue/500xCompressor/logs/ICAE_finetune-Qwen3-8B \
    --project_name ICAE-finetuning \
    --deepspeed_config /mnt/hdfs/wangchangyue/500xCompressor/codes/deepspeed_configurations.json \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5 \
    --save_steps 300 \
    --eval_steps 300 \
    --warmup_steps 300
