cd /mnt/hdfs/wangchangyue/500xCompressor/codes/pretraining

deepspeed  --master_port=11470 train_ICAE.py \
    --model_path /mnt/hdfs/wangchangyue/LLM/phi-4 \
    --num_mem 256 \
    --max_length 512 \
    --train_text_path /mnt/hdfs/wangchangyue/500xCompressor/data/train.txt \
    --test_text_path /mnt/hdfs/wangchangyue/500xCompressor/data/test.txt \
    --output_dir /mnt/hdfs/wangchangyue/500xCompressor/output/ICAE_pretrain \
    --logging_dir /mnt/hdfs/wangchangyue/500xCompressor/logs/ICAE_pretrain \
    --project_name ICAE-pretraining \
    --deepspeed_config /mnt/hdfs/wangchangyue/500xCompressor/codes/deepspeed_configurations.json \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --save_steps 200 \
    --eval_steps 100 \
    --warmup_steps 300
