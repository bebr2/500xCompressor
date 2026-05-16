"""
Test a single batch size. Exit with code 0 if success, 1 if failed.
The binary search is handled by an external shell script.

Usage:
  deepspeed test_bsz.py --model_path /path/to/model --batch_size 4
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import gc
import tempfile
import json
import shutil
from peft import LoraConfig
from transformers import AutoTokenizer, TrainingArguments, Trainer

from codes.pretraining.L3LoraL3 import L3LoraL3


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, max_length, eos_token_id):
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.length = 50

    def __len__(self):
        return self.length

    def __getitem__(self, _idx):
        input_ids = torch.randint(0, 1000, (self.max_length,), dtype=torch.long)
        target_tokens = torch.full((self.max_length + 1,), -100, dtype=torch.long)
        text_eos_tokens = input_ids.tolist()
        text_eos_tokens.append(self.eos_token_id)
        target_tokens[0:len(text_eos_tokens)] = torch.tensor(text_eos_tokens, dtype=torch.long)
        return {"input_ids": input_ids, "labels": target_tokens}


def create_ds_config(path, hidden_size):
    config = {
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "stage3_gather_16bit_weights_on_model_save": False
        },
        "gradient_accumulation_steps": 1,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "bf16": {"enabled": True}
    }
    with open(path, 'w') as f:
        json.dump(config, f)


def main():
    args = parse_args()

    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    temp_dir = tempfile.mkdtemp()

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

        lora_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=args.target_modules
        )

        model = L3LoraL3(
            llama_path=args.model_path, max_length=args.max_length,
            lora_config=lora_config, num_mem=args.num_mem, device=device
        )
        model.config = model.llama.config

        create_ds_config(os.path.join(temp_dir, "ds.json"), model.hidden_size)

        dataset = DummyDataset(args.max_length, tokenizer.eos_token_id)

        training_args = TrainingArguments(
            output_dir=temp_dir,
            max_steps=2,
            per_device_train_batch_size=args.batch_size,
            deepspeed=os.path.join(temp_dir, "ds.json"),
            save_strategy="no",
            report_to="none",
            bf16=True,
            dataloader_num_workers=0,
        )

        trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
        trainer.train()

        if local_rank == 0:
            print(f"SUCCESS: batch_size={args.batch_size}")

        # Exit 0 = success
        sys.exit(0)

    except Exception as e:
        if local_rank == 0:
            if "out of memory" in str(e).lower():
                print(f"OOM: batch_size={args.batch_size}")
            else:
                print(f"ERROR: {e}")

        # Exit 1 = failed
        sys.exit(1)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_mem", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    main()