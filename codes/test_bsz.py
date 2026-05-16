"""
Test maximum batch size for training on multi-GPU setup with DeepSpeed.
This script finds the largest batch size that fits in GPU memory.
Uses DeepSpeed ZeRO Stage 3 for model parameter sharding.
"""
import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
import os
import gc
import json
import tempfile
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

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

def setup_distributed():
    """Setup distributed training environment."""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def create_deepspeed_config(output_path, batch_size, gradient_accumulation_steps=1):
    """Create DeepSpeed config with proper settings."""
    config = {
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "sub_group_size": 1e9,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": False
        },
        "gradient_accumulation_steps": gradient_accumulation_steps,
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
        json.dump(config, f, indent=2)
    return output_path

class DummyDataset(torch.utils.data.Dataset):
    """Dummy dataset for batch size testing."""
    def __init__(self, max_length, num_mem):
        self.max_length = max_length
        self.num_mem = num_mem
        self.length = 100

    def __len__(self):
        return self.length

    def __getitem__(self, _idx):
        input_ids = torch.randint(0, 1000, (self.max_length,))
        # Labels: input tokens + EOS token
        labels = torch.full((self.max_length + 1,), -100, dtype=torch.long)
        labels[:self.max_length] = input_ids
        return {"input_ids": input_ids, "labels": labels}

class TestModel(nn.Module):
    """Wrapper model that includes memory embeddings."""
    def __init__(self, base_model, num_mem, hidden_size):
        super().__init__()
        self.llama = base_model
        self.num_mem = num_mem
        self.hidden_size = hidden_size

        # Memory embeddings - will be managed by DeepSpeed
        self.memory_embeddings = nn.Parameter(
            torch.randn(1, num_mem, hidden_size, dtype=torch.bfloat16)
        )

    def forward(self, input_ids, labels=None, **kwargs):
        # Get embeddings
        inputs_embeds = self.llama.get_input_embeddings()(input_ids)

        # Add memory embeddings
        batch_size = inputs_embeds.shape[0]
        mem_emb = self.memory_embeddings.repeat(batch_size, 1, 1)
        inputs_embeds = torch.cat((inputs_embeds, mem_emb), dim=1)

        # Forward through base model
        outputs = self.llama(inputs_embeds=inputs_embeds, **kwargs)

        return outputs

def test_batch_size(args, batch_size, rank, local_rank, world_size):
    """Test if a specific batch size works."""
    from peft import LoraConfig, get_peft_model

    # Create temp directory for this test
    temp_dir = tempfile.mkdtemp()
    ds_config_path = os.path.join(temp_dir, "ds_config.json")
    output_dir = os.path.join(temp_dir, "output")

    create_deepspeed_config(ds_config_path, batch_size)

    if rank == 0:
        print(f"  Testing batch_size={batch_size}, creating config at {ds_config_path}")

    # Define LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules
    )

    # Load base model - DeepSpeed will handle sharding
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    hidden_size = base_model.config.hidden_size

    # Apply LoRA
    base_model = get_peft_model(base_model, lora_config)

    # Ensure LoRA params are bfloat16
    for name, param in base_model.named_parameters():
        if 'lora' in name and param.dtype != torch.bfloat16:
            param.data = param.data.to(torch.bfloat16)

    # Wrap with memory embeddings
    model = TestModel(base_model, args.num_mem, hidden_size)

    # Create dataset
    dataset = DummyDataset(args.max_length, args.num_mem)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=3,  # Just 3 steps to test
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        deepspeed=ds_config_path,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        bf16=True,
        dataloader_num_workers=0,
    )

    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        # This will trigger DeepSpeed initialization with proper sharding
        trainer.train()

        # Get memory info
        mem_info = get_memory_info()

        if rank == 0:
            print(f"  SUCCESS! batch_size={batch_size}, "
                  f"GPU memory: {mem_info['max_allocated_GB']:.2f}GB")

        # Cleanup
        del trainer
        del model
        del base_model
        clear_memory()

        # Remove temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        return True

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if rank == 0:
                mem_info = get_memory_info()
                print(f"  OOM! batch_size={batch_size}, "
                      f"GPU memory: {mem_info['max_allocated_GB']:.2f}GB")

            # Cleanup
            clear_memory()

            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

            return False
        else:
            if rank == 0:
                print(f"  Error: {e}")
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise e
    except Exception as e:
        if rank == 0:
            print(f"  Error: {e}")
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e

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

        success = test_batch_size(args, mid, rank, local_rank, world_size)

        # Sync across GPUs
        if world_size > 1:
            success_tensor = torch.tensor([1 if success else 0],
                                          dtype=torch.int32,
                                          device=torch.device(f"cuda:{local_rank}"))
            dist.all_reduce(success_tensor, op=dist.ReduceOp.MIN)
            success = (success_tensor.item() == 1)

        if success:
            max_bsz = mid
            low = mid + 1
        else:
            high = mid - 1

    return max_bsz

def parse_args():
    parser = argparse.ArgumentParser(description="Test max batch size with DeepSpeed")

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
    args = parse_args()

    rank, local_rank, world_size = setup_distributed()

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
        print(f"  Total batch size (8 GPUs) = {max_bsz * world_size}")
        print(f"{'='*60}")

        if max_bsz == 0:
            print("\nWARNING: Even batch_size=1 failed!")
            print("Try reducing max_length or num_mem, or use gradient_accumulation_steps")