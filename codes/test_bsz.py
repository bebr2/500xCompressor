"""
Test maximum batch size for training on multi-GPU setup with DeepSpeed.
This script finds the largest batch size that fits in GPU memory.
Uses the actual training components from train_500xCompressor.py.
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
import argparse
import gc
import tempfile
from datetime import timedelta
from peft import LoraConfig
from transformers import AutoTokenizer, TrainingArguments, Trainer

# Import actual training components
from codes.pretraining.L3LoraL3 import L3LoraL3


def get_memory_info():
    """Get current GPU memory usage."""
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    return {
        "allocated_GB": allocated,
        "reserved_GB": reserved,
        "max_allocated_GB": max_allocated
    }


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def create_deepspeed_config(output_path, hidden_size):
    """Create DeepSpeed config with ZeRO Stage 3."""
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
        "bf16": {
            "enabled": True
        }
    }

    with open(output_path, 'w') as f:
        import json
        json.dump(config, f, indent=2)
    return output_path


class DummyTextDataset(torch.utils.data.Dataset):
    """Dummy dataset for batch size testing, mimics TextDataset structure."""
    def __init__(self, max_length, eos_token_id):
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.length = 100

    def __len__(self):
        return self.length

    def __getitem__(self, _idx):
        # Generate random input_ids
        input_ids = torch.randint(0, 1000, (self.max_length,), dtype=torch.long)
        # Labels: input tokens + EOS token (same as TextDataset)
        target_tokens = torch.full((self.max_length + 1,), -100, dtype=torch.long)
        text_eos_tokens = input_ids.tolist()
        text_eos_tokens.append(self.eos_token_id)
        text_eos_tokens_len = len(text_eos_tokens)
        target_tokens[0:0+text_eos_tokens_len] = torch.tensor(text_eos_tokens, dtype=torch.long)
        return {"input_ids": input_ids, "labels": target_tokens}


def test_batch_size(args, batch_size, rank, local_rank, world_size):
    """Test if a specific batch size works."""
    # Initialize process group for this test (fresh start)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=3))
    torch.cuda.set_device(local_rank)

    # Create temp directory for this test
    temp_dir = tempfile.mkdtemp()
    ds_config_path = os.path.join(temp_dir, "ds_config.json")
    output_dir = os.path.join(temp_dir, "output")

    try:
        # Get EOS token ID
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        eos_token_id = tokenizer.eos_token_id

        # Create model (same as train_500xCompressor.py)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.target_modules
        )

        device = torch.device(f"cuda:{local_rank}")

        if rank == 0:
            print(f"  Testing batch_size={batch_size}...")

        model = L3LoraL3(
            llama_path=args.model_path,
            max_length=args.max_length,
            lora_config=lora_config,
            num_mem=args.num_mem,
            device=device
        )
        # Set config for DeepSpeed (required)
        model.config = model.llama.config

        hidden_size = model.hidden_size
        create_deepspeed_config(ds_config_path, hidden_size)

        # Create dummy dataset (same structure as TextDataset)
        dataset = DummyTextDataset(args.max_length, eos_token_id)

        # Training arguments (minimal steps for testing)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            max_steps=2,  # Just 2 steps to test
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

        # Success
        mem_info = get_memory_info()
        if rank == 0:
            print(f"  SUCCESS! batch_size={batch_size}, GPU memory: {mem_info['max_allocated_GB']:.2f}GB")

        # Cleanup
        del trainer
        del model
        clear_memory()

        # Destroy process group for fresh start next test
        if dist.is_initialized():
            dist.destroy_process_group()

        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        return True

    except Exception as e:
        error_str = str(e).lower()
        is_oom = "out of memory" in error_str or "oom" in error_str

        if is_oom:
            if rank == 0:
                mem_info = get_memory_info()
                print(f"  OOM! batch_size={batch_size}, GPU memory: {mem_info['max_allocated_GB']:.2f}GB")
        else:
            if rank == 0:
                print(f"  Error ({type(e).__name__}): {e}")

        # Cleanup
        clear_memory()

        # Destroy process group
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except:
                pass

        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        return False


def find_max_batch_size(args, rank, local_rank, world_size):
    """Binary search for maximum batch size."""
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Finding max batch size with DeepSpeed ZeRO-3")
        print(f"  GPUs: {world_size}")
        print(f"  Max length: {args.max_length}")
        print(f"  Num mem tokens: {args.num_mem}")
        print(f"  Search range: {args.start_bsz} - {args.max_bsz_limit}")
        print(f"{'='*60}")

    max_bsz = 0
    low = args.start_bsz
    high = args.max_bsz_limit

    while low <= high:
        mid = (low + high) // 2

        if rank == 0:
            print(f"\n[{low}-{high}] Testing batch_size={mid}...")

        # Each test runs with fresh process group
        success = test_batch_size(args, mid, rank, local_rank, world_size)

        # Sync result across GPUs using a fresh process group
        if world_size > 1:
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl', timeout=timedelta(minutes=1))

            # Broadcast result from rank 0
            result_tensor = torch.tensor([1.0 if success else 0.0],
                                         dtype=torch.float32,
                                         device=torch.device(f"cuda:{local_rank}"))
            dist.all_reduce(result_tensor, op=dist.ReduceOp.MIN)
            success = (result_tensor.item() == 1.0)

            # Update search bounds on all ranks
            if success:
                max_bsz = mid
                low = mid + 1
            else:
                high = mid - 1

            # Broadcast updated bounds
            bounds_tensor = torch.tensor([max_bsz, low, high],
                                         dtype=torch.int32,
                                         device=torch.device(f"cuda:{local_rank}"))
            dist.broadcast(bounds_tensor, src=0)
            max_bsz, low, high = bounds_tensor.tolist()

            # Destroy process group
            dist.destroy_process_group()
        else:
            # Single GPU
            if success:
                max_bsz = mid
                low = mid + 1
            else:
                high = mid - 1

    return max_bsz


def parse_args():
    parser = argparse.ArgumentParser(description="Test max batch size with DeepSpeed")

    # Model arguments (same as train_500xCompressor.py)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_mem", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=2048)

    # LoRA arguments (same as train_500xCompressor.py)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    # Search arguments
    parser.add_argument("--start_bsz", type=int, default=1)
    parser.add_argument("--max_bsz_limit", type=int, default=64)

    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Get distributed info from environment
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if rank == 0:
        print(f"\nGPU Info:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, Memory: {props.total_memory/1024**3:.1f}GB")

    max_bsz = find_max_batch_size(args, rank, local_rank, world_size)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"  per_device_train_batch_size = {max_bsz}")
        print(f"  Total batch size ({world_size} GPUs) = {max_bsz * world_size}")
        print(f"{'='*60}")

        if max_bsz == 0:
            print("\nWARNING: Even batch_size=1 failed!")
            print("Try reducing max_length or num_mem, or use gradient_accumulation_steps")