"""
Test maximum batch size with DeepSpeed ZeRO-3.
Uses file-based coordination to avoid NCCL synchronization issues.

Usage with deepspeed launcher:
  deepspeed --master_port=11470 test_bsz.py --model_path /path/to/model --start_bsz 1 --max_bsz_limit 16
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
import argparse
import gc
import tempfile
import json
import time
from pathlib import Path
from peft import LoraConfig
from transformers import AutoTokenizer, TrainingArguments, Trainer

from codes.pretraining.L3LoraL3 import L3LoraL3


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def create_deepspeed_config(output_path, hidden_size):
    config = {
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": hidden_size * hidden_size,
            "stage3_prefetch_bucket_size": hidden_size * hidden_size,
            "stage3_param_persistence_threshold": hidden_size,
            "sub_group_size": 1e9,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": False
        },
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "steps_per_print": 100,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
        "bf16": {"enabled": True}
    }
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, max_length, eos_token_id):
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.length = 100

    def __len__(self):
        return self.length

    def __getitem__(self, _idx):
        input_ids = torch.randint(0, 1000, (self.max_length,), dtype=torch.long)
        target_tokens = torch.full((self.max_length + 1,), -100, dtype=torch.long)
        text_eos_tokens = input_ids.tolist()
        text_eos_tokens.append(self.eos_token_id)
        target_tokens[0:len(text_eos_tokens)] = torch.tensor(text_eos_tokens, dtype=torch.long)
        return {"input_ids": input_ids, "labels": target_tokens}


def test_single_batch_size(args, batch_size, rank, local_rank):
    """Test one batch size. Returns True if success, False if failed."""
    device = torch.device(f"cuda:{local_rank}")

    temp_dir = tempfile.mkdtemp()
    ds_config_path = os.path.join(temp_dir, "ds_config.json")
    output_dir = os.path.join(temp_dir, "output")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        eos_token_id = tokenizer.eos_token_id

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.target_modules
        )

        if rank == 0:
            print(f"  Loading model for batch_size={batch_size}...")

        model = L3LoraL3(
            llama_path=args.model_path,
            max_length=args.max_length,
            lora_config=lora_config,
            num_mem=args.num_mem,
            device=device
        )
        model.config = model.llama.config

        create_deepspeed_config(ds_config_path, model.hidden_size)

        dataset = DummyDataset(args.max_length, eos_token_id)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            max_steps=2,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            deepspeed=ds_config_path,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            bf16=True,
            dataloader_num_workers=0,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        if rank == 0:
            mem_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  SUCCESS! batch_size={batch_size}, memory={mem_gb:.2f}GB")

        del trainer, model
        clear_memory()
        return True

    except Exception as e:
        error_str = str(e).lower()
        if rank == 0:
            mem_gb = torch.cuda.max_memory_allocated() / 1024**3
            if "out of memory" in error_str:
                print(f"  OOM! batch_size={batch_size}, memory={mem_gb:.2f}GB")
            else:
                print(f"  Error: {type(e).__name__}")

        clear_memory()
        return False

    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def file_based_barrier(barrier_dir, rank, world_size, barrier_name):
    """Simple barrier using files. All processes must reach it before continuing."""
    barrier_file = barrier_dir / f"{barrier_name}_complete"

    # Signal arrival
    arrival_file = barrier_dir / f"{barrier_name}_rank{rank}"
    arrival_file.touch()

    # Wait for all ranks
    while True:
        arrived = sum(1 for r in range(world_size) if (barrier_dir / f"{barrier_name}_rank{r}").exists())
        if arrived == world_size:
            break
        time.sleep(0.1)

    # Clean up (only rank 0)
    if rank == 0:
        for r in range(world_size):
            (barrier_dir / f"{barrier_name}_rank{r}").unlink(missing_ok=True)
        barrier_file.touch()

    # Wait for completion signal
    while not barrier_file.exists():
        time.sleep(0.1)

    if rank == 0:
        barrier_file.unlink(missing_ok=True)


def main():
    args = parse_args()

    # deepspeed launcher sets these
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    rank = int(os.environ.get('RANK', local_rank))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    torch.cuda.set_device(local_rank)

    # Create a shared directory for coordination
    coord_dir = Path(tempfile.gettempdir()) / f"bsz_test_{os.getpid()}"
    coord_dir.mkdir(exist_ok=True)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Finding max batch size with DeepSpeed ZeRO-3")
        print(f"GPUs: {world_size}")
        print(f"Search range: {args.start_bsz} - {args.max_bsz_limit}")
        print(f"{'='*60}")

        # Clean up any old files
        for f in coord_dir.glob("*"):
            f.unlink()

    # Binary search for max batch size
    max_bsz = 0
    low = args.start_bsz
    high = args.max_bsz_limit
    test_round = 0

    while low <= high:
        mid = (low + high) // 2
        test_round += 1

        if rank == 0:
            print(f"\n[Round {test_round}] Testing batch_size={mid} (range: {low}-{high})")

        # All ranks must participate in the test
        success = test_single_batch_size(args, mid, rank, local_rank)

        # Write result to file (rank 0 only)
        result_file = coord_dir / f"result_{test_round}"
        if rank == 0:
            result_file.write_text("1" if success else "0")

        # Barrier: wait for all ranks to complete test
        file_based_barrier(coord_dir, rank, world_size, f"test_{test_round}")

        # Read result (all ranks read the same value from rank 0's file)
        result = int(result_file.read_text())
        success = (result == 1)

        # Barrier: wait for all ranks to read result
        file_based_barrier(coord_dir, rank, world_size, f"read_{test_round}")

        # Update search bounds (all ranks do the same)
        if success:
            max_bsz = mid
            low = mid + 1
        else:
            high = mid - 1

    # Final report (rank 0 only)
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"  max per_device_train_batch_size = {max_bsz}")
        print(f"  total_batch_size ({world_size} GPUs) = {max_bsz * world_size}")
        print(f"{'='*60}")

        if max_bsz == 0:
            print("\nWARNING: Even start_bsz failed!")
            print("Try reducing max_length or num_mem")

        # Clean up coordination directory
        for f in coord_dir.glob("*"):
            f.unlink()
        coord_dir.rmdir()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_mem", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--start_bsz", type=int, default=1)
    parser.add_argument("--max_bsz_limit", type=int, default=64)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    main()