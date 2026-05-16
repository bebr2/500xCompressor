"""
Test maximum batch size for training on multi-GPU setup with DeepSpeed.
This script finds the largest batch size that fits in GPU memory.
"""
import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
import os
import gc
import json
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'OMPI_COMM_WORLD_RANK' in os.environ:
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    else:
        print("No distributed environment detected, running single GPU test")
        return 0, 0, 1

    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size

def create_deepspeed_config(output_path):
    """Create a minimal DeepSpeed config for testing."""
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
        "gradient_accumulation_steps": 1,
        "gradient_clipping": "auto",
        "steps_per_print": 2000,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False
    }

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    return output_path

class DummyDataset(torch.utils.data.Dataset):
    """Dummy dataset for batch size testing."""
    def __init__(self, max_length, num_mem, mode="pretrain"):
        self.max_length = max_length
        self.num_mem = num_mem
        self.mode = mode
        self.length = 100

    def __len__(self):
        return self.length

    def __getitem__(self, _idx):
        if self.mode == "pretrain":
            input_ids = torch.randint(0, 1000, (self.max_length,))
            labels = torch.full((self.max_length + 1,), -100, dtype=torch.long)
            labels[:self.max_length] = input_ids
            labels[self.max_length] = 151645  # Qwen EOS token
            return {"input_ids": input_ids, "labels": labels}
        else:
            # QA mode
            total_len = self.max_length + 256  # max_qa_len
            input_ids = torch.randint(0, 1000, (total_len,))
            labels = torch.full((256,), -100, dtype=torch.long)
            return {"input_ids": input_ids, "labels": labels}

def test_with_deepspeed_trainer(args, batch_size, mode, rank, local_rank, _world_size):
    """Test batch size using HuggingFace Trainer with DeepSpeed."""
    from transformers import TrainingArguments, Trainer

    # Create temp deepspeed config
    ds_config_path = "/tmp/test_deepspeed_config.json"
    create_deepspeed_config(ds_config_path)

    # Load model
    if rank == 0:
        print(f"Loading model from {args.model_path}...")

    llama = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    hidden_size = llama.config.hidden_size

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    from peft import get_peft_model
    llama = get_peft_model(llama, lora_config)

    # Enable gradients for LoRA params only
    for name, param in llama.named_parameters():
        param.requires_grad = False
        if 'lora' in name:
            param.requires_grad = True

    # Add memory embeddings
    memory_embeddings = nn.Parameter(
        torch.randn(1, args.num_mem, hidden_size, dtype=torch.bfloat16)
    )
    memory_embeddings.requires_grad = True

    # Monkey patch to include memory embeddings
    original_forward = llama.forward

    def patched_forward(input_ids=None, inputs_embeds=None, **kwargs):
        if input_ids is not None and inputs_embeds is None:
            inputs_embeds = llama.get_input_embeddings()(input_ids)

        # Add memory embeddings
        if inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
            mem_emb = memory_embeddings.repeat(batch_size, 1, 1).to(inputs_embeds.device)
            inputs_embeds = torch.cat((inputs_embeds, mem_emb), dim=1)

        return original_forward(inputs_embeds=inputs_embeds, **kwargs)

    llama.forward = patched_forward
    llama.memory_embeddings = memory_embeddings

    # Create dummy dataset
    dataset = DummyDataset(args.max_length, args.num_mem, mode=mode)

    # Training arguments with DeepSpeed
    training_args = TrainingArguments(
        output_dir="/tmp/test_output",
        num_train_epochs=1,
        max_steps=3,  # Only run a few steps to test
        per_device_train_batch_size=batch_size,
        deepspeed=ds_config_path,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )

    try:
        trainer = Trainer(
            model=llama,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        if rank == 0:
            mem_info = get_memory_info()
            print(f"Batch size {batch_size} SUCCESS - "
                  f"Memory: allocated={mem_info['allocated_GB']:.2f}GB, "
                  f"max={mem_info['max_allocated_GB']:.2f}GB")

        # Clean up
        del trainer
        del llama
        clear_memory()

        return True

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if rank == 0:
                mem_info = get_memory_info()
                print(f"Batch size {batch_size} OOM - "
                      f"Memory: max={mem_info['max_allocated_GB']:.2f}GB")
            clear_memory()
            return False
        else:
            raise e

def find_max_batch_size(args, mode, rank, local_rank, world_size):
    """Binary search for maximum batch size."""
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Testing max batch size for: {mode}")
        print(f"World size: {world_size} GPUs")
        print(f"Max length: {args.max_length}")
        print(f"Num mem: {args.num_mem}")
        print(f"{'='*60}")

    max_bsz = 0
    low = args.start_bsz
    high = args.max_bsz_limit

    # Binary search
    while low <= high:
        mid = (low + high) // 2

        if rank == 0:
            print(f"\nTesting batch size: {mid}")

        success = test_with_deepspeed_trainer(args, mid, mode, rank, local_rank, world_size)

        # Sync success status across all ranks
        success_tensor = torch.tensor([1 if success else 0], device=torch.device(f"cuda:{local_rank}"))
        if dist.is_initialized():
            dist.all_reduce(success_tensor, op=dist.ReduceOp.MIN)
        success = success_tensor.item() == 1

        if success:
            max_bsz = mid
            low = mid + 1
        else:
            high = mid - 1

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Maximum batch size for {mode}: {max_bsz}")
        print(f"Recommended per_device_train_batch_size: {max_bsz}")
        print(f"Effective total batch size (with {world_size} GPUs): {max_bsz * world_size}")
        print(f"{'='*60}")

    return max_bsz

def parse_args():
    parser = argparse.ArgumentParser(description="Test maximum batch size with DeepSpeed")

    # Model arguments
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_mem", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=2048)

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Search arguments
    parser.add_argument("--start_bsz", type=int, default=1)
    parser.add_argument("--max_bsz_limit", type=int, default=32)

    # Test mode
    parser.add_argument("--test_mode", type=str, default="pretrain",
                        choices=["pretrain", "qa"])

    # DeepSpeed arguments (passed by deepspeed launcher)
    parser.add_argument("--local_rank", type=int, default=-1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        print(f"GPU Count: {world_size}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, Total: {props.total_memory / 1024**3:.1f}GB")

    # Run test
    max_bsz = find_max_batch_size(args, args.test_mode, rank, local_rank, world_size)

    if rank == 0:
        print(f"\nFINAL RESULT:")
        print(f"  Mode: {args.test_mode}")
        print(f"  per_device_train_batch_size: {max_bsz}")
        print(f"  Total effective batch size: {max_bsz * world_size}")