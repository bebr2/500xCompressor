cd /mnt/hdfs/wangchangyue/500xCompressor/codes/pretraining

deepspeed  --master_port=11470 train_500xCompressor.py \
    --model_path /mnt/hdfs/wangchangyue/LLM/Qwen3-8B \
    --num_mem 256 \
    --max_length 2048 \
    --train_text_path /mnt/hdfs/wangchangyue/500xCompressor/data/train.txt \
    --test_text_path /mnt/hdfs/wangchangyue/500xCompressor/data/test.txt \
    --output_dir /mnt/hdfs/wangchangyue/500xCompressor/output/500xCompressor_pretrain \
    --logging_dir /mnt/hdfs/wangchangyue/500xCompressor/logs/500xCompressor_pretrain \
    --project_name 500xCompressor-pretraining \
    --deepspeed_config /mnt/hdfs/wangchangyue/500xCompressor/codes/deepspeed_configurations.json \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --save_steps 300 \
    --eval_steps 100 \
    --warmup_steps 300
