cd /mnt/hdfs/wangchangyue/500xCompressor/codes

python convert_checkpoint.py /mnt/hdfs/wangchangyue/500xCompressor/output/500xCompressor_pretrain

cd /mnt/hdfs/wangchangyue/500xCompressor/codes/finetuning

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 deepspeed  --master_port=11470 finetune_500xCompressor.py \
    --model_path /mnt/hdfs/wangchangyue/LLM/phi-4 \
    --num_mem 256 \
    --max_length 512 \
    --max_qa_len 46 \
    --train_text_path /mnt/hdfs/wangchangyue/500xCompressor/data/train_qa.jsonl \
    --test_text_path /mnt/hdfs/wangchangyue/500xCompressor/data/test_qa.jsonl \
    --lora_path /mnt/hdfs/wangchangyue/500xCompressor/output/500xCompressor_pretrain/checkpoint_best/pytorch_model.bin \
    --output_dir /mnt/hdfs/wangchangyue/500xCompressor/output/500xCompressor_finetune2 \
    --logging_dir /mnt/hdfs/wangchangyue/500xCompressor/logs/500xCompressor_finetune2 \
    --project_name "500xCompressor-finetuning" \
    --deepspeed_config /mnt/hdfs/wangchangyue/500xCompressor/codes/deepspeed_configurations.json \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-5 \
    --save_steps 300 \
    --eval_steps 300 \
    --warmup_steps 300
