"""
Test a single batch size. Exit with code 0 if success, 1 if failed.
The binary search is handled by an external shell script.

Usage:
  deepspeed test_bsz.py --stage pretrain --model_path /path/to/model --batch_size 4
  deepspeed test_bsz.py --stage finetune --model_path /path/to/model --lora_path /path/to/pytorch_model.bin --batch_size 4
"""
import argparse
import gc
import json
import os
import shutil
import sys
import tempfile

import torch
from peft import LoraConfig
from transformers import AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codes.finetuning.ICAEL3QA import ICAEL3QA
from codes.finetuning.L3LoraL3QA import L3LoraL3QA
from codes.pretraining.ICAEL3 import ICAEL3
from codes.pretraining.L3LoraL3 import L3LoraL3


class DummyPretrainDataset(torch.utils.data.Dataset):
    def __init__(self, max_length, num_mem, eos_token_id, compressor):
        self.max_length = max_length
        self.num_mem = num_mem
        self.eos_token_id = eos_token_id
        self.compressor = compressor
        self.length = 50

    def __len__(self):
        return self.length

    def __getitem__(self, _idx):
        input_ids = torch.randint(0, 1000, (self.max_length,), dtype=torch.long)
        text_eos_tokens = input_ids.tolist()
        text_eos_tokens.append(self.eos_token_id)

        if self.compressor == "icae":
            prompt_len = 1
            labels = torch.full((self.num_mem + prompt_len + self.max_length,), -100, dtype=torch.long)
            start = self.num_mem + prompt_len - 1
        else:
            labels = torch.full((self.max_length + 1,), -100, dtype=torch.long)
            start = 0

        labels[start:start + len(text_eos_tokens)] = torch.tensor(text_eos_tokens, dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


class DummyFinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, max_length, max_qa_len, num_mem, eos_token_id, compressor):
        self.max_length = max_length
        self.max_qa_len = max_qa_len
        self.num_mem = num_mem
        self.eos_token_id = eos_token_id
        self.compressor = compressor
        self.length = 50

    def __len__(self):
        return self.length

    def __getitem__(self, _idx):
        context_ids = torch.randint(0, 1000, (self.max_length,), dtype=torch.long)
        qa_ids = torch.randint(0, 1000, (self.max_qa_len,), dtype=torch.long)
        input_ids = torch.cat((context_ids, qa_ids), dim=0)

        if self.compressor == "icae":
            labels = torch.full((self.num_mem + self.max_qa_len,), -100, dtype=torch.long)
            label_start = self.num_mem
        else:
            labels = torch.full((self.max_qa_len,), -100, dtype=torch.long)
            label_start = 0

        labels[label_start:] = qa_ids
        labels[-1] = self.eos_token_id
        return {"input_ids": input_ids, "labels": labels}


def create_ds_config(path):
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
    with open(path, "w") as f:
        json.dump(config, f)


def create_model(args, lora_config, device):
    if args.stage == "pretrain":
        if args.compressor == "icae":
            return ICAEL3(
                llama_path=args.model_path,
                max_length=args.max_length,
                lora_config=lora_config,
                num_mem=args.num_mem,
                device=device
            )

        return L3LoraL3(
            llama_path=args.model_path,
            max_length=args.max_length,
            lora_config=lora_config,
            num_mem=args.num_mem,
            device=device
        )

    if not args.lora_path:
        raise ValueError("--lora_path is required when --stage finetune")

    if args.compressor == "icae":
        return ICAEL3QA(
            llama_path=args.model_path,
            max_context_length=args.max_length,
            lora_path=args.lora_path,
            lora_config=lora_config,
            num_mem=args.num_mem,
            device=device
        )

    return L3LoraL3QA(
        llama_path=args.model_path,
        max_context_length=args.max_length,
        lora_path=args.lora_path,
        lora_config=lora_config,
        num_mem=args.num_mem,
        device=device
    )


def create_dataset(args, eos_token_id):
    if args.stage == "finetune":
        return DummyFinetuneDataset(
            max_length=args.max_length,
            max_qa_len=args.max_qa_len,
            num_mem=args.num_mem,
            eos_token_id=eos_token_id,
            compressor=args.compressor
        )

    return DummyPretrainDataset(
        max_length=args.max_length,
        num_mem=args.num_mem,
        eos_token_id=eos_token_id,
        compressor=args.compressor
    )


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    temp_dir = tempfile.mkdtemp()

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.target_modules
        )

        model = create_model(args, lora_config, device)
        model.config = model.llama.config

        ds_config = os.path.join(temp_dir, "ds.json")
        create_ds_config(ds_config)

        dataset = create_dataset(args, eos_token_id)

        training_args = TrainingArguments(
            output_dir=temp_dir,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            deepspeed=ds_config,
            save_strategy="no",
            eval_strategy="steps" if args.include_eval else "no",
            eval_steps=1,
            report_to="none",
            bf16=True,
            dataloader_num_workers=0,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset if args.include_eval else None
        )
        trainer.train()

        if local_rank == 0:
            print(
                f"SUCCESS: stage={args.stage}, compressor={args.compressor}, "
                f"batch_size={args.batch_size}, max_length={args.max_length}, "
                f"max_qa_len={args.max_qa_len}, grad_accum={args.gradient_accumulation_steps}, "
                f"include_eval={args.include_eval}, eval_batch_size={args.per_device_eval_batch_size}"
            )

        sys.exit(0)

    except Exception as e:
        if local_rank == 0:
            if "out of memory" in str(e).lower():
                print(f"OOM: batch_size={args.batch_size}")
            else:
                print(f"ERROR: {e}")

        sys.exit(1)

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        shutil.rmtree(temp_dir, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pretrain", "finetune"], default="pretrain")
    parser.add_argument("--compressor", choices=["500x", "icae"], default="500x")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=48)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=2)
    parser.add_argument("--include_eval", action="store_true")
    parser.add_argument("--num_mem", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_qa_len", type=int, default=46)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
