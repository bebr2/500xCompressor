#!/usr/bin/env python3
"""
Convert DeepSpeed checkpoint to pytorch_model.bin for fine-tuning.

Usage:
    python convert_checkpoint.py /path/to/output_dir [--output_name checkpoint_best]

Example:
    python convert_checkpoint.py /mnt/hdfs/wangchangyue/500xCompressor/output/ICAE_pretrain
"""

import os
import re
import json
import shutil
import subprocess
import argparse
import torch
from pathlib import Path


def find_checkpoints(output_dir: str) -> list[tuple[str, int]]:
    """Find all checkpoint directories and return (path, step) pairs."""
    checkpoints = []
    for item in os.listdir(output_dir):
        match = re.match(r'checkpoint-(\d+)', item)
        if match:
            step = int(match.group(1))
            checkpoint_path = os.path.join(output_dir, item)
            if os.path.isdir(checkpoint_path):
                checkpoints.append((checkpoint_path, step))
    return sorted(checkpoints, key=lambda x: x[1], reverse=True)


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the checkpoint with the highest step number."""
    checkpoints = find_checkpoints(output_dir)
    if checkpoints:
        return checkpoints[0][0]
    return None


def load_sharded_weights(checkpoint_dir: str) -> dict:
    """Load weights from sharded pytorch_model files."""
    index_file = os.path.join(checkpoint_dir, "pytorch_model.bin.index.json")

    if os.path.exists(index_file):
        # Sharded format
        with open(index_file, "r") as f:
            index = json.load(f)

        weight_map = index.get("weight_map", {})
        all_weights = {}

        # Get unique shard files in a stable order for repeatable conversion logs.
        shard_files = sorted(set(weight_map.values()))

        for shard_file in shard_files:
            shard_path = os.path.join(checkpoint_dir, shard_file)
            if os.path.exists(shard_path):
                shard_weights = torch.load(shard_path, map_location="cpu")
                all_weights.update(shard_weights)

        return all_weights
    else:
        # Single file format
        single_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(single_file):
            return torch.load(single_file, map_location="cpu")
        return {}


def merge_sharded_weights(checkpoint_dir: str, output_file: str) -> bool:
    """Merge sharded weights into single pytorch_model.bin."""
    weights = load_sharded_weights(checkpoint_dir)

    if not weights:
        print(f"ERROR: No weights found in {checkpoint_dir}")
        return False

    # Save merged weights
    torch.save(weights, output_file)
    print(f"Merged {len(weights)} tensors into {output_file}")
    return True


def remove_path(path: str) -> None:
    """Remove a file or directory at path."""
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def try_remove_path(path: str) -> None:
    """Best-effort cleanup that should not make a successful conversion fail."""
    try:
        remove_path(path)
    except OSError as e:
        print(f"WARNING: Failed to remove temporary path {path}: {e}")


def repair_sharded_output_dir(model_dir: str) -> bool:
    """Replace an existing sharded pytorch_model.bin directory with one bin file."""
    index_file = os.path.join(model_dir, "pytorch_model.bin.index.json")
    if not os.path.isdir(model_dir) or not os.path.exists(index_file):
        return False

    parent_dir = os.path.dirname(model_dir)
    temp_output = os.path.join(parent_dir, "pytorch_model.bin.tmp")

    if os.path.exists(temp_output):
        remove_path(temp_output)

    print(f"Detected existing sharded directory: {model_dir}")
    print("Merging it into a single pytorch_model.bin file...")
    if not merge_sharded_weights(model_dir, temp_output):
        if os.path.exists(temp_output):
            remove_path(temp_output)
        return False

    shutil.rmtree(model_dir)
    shutil.move(temp_output, model_dir)
    print(f"SUCCESS: Repaired {model_dir}")
    return True


def convert_checkpoint(checkpoint_dir: str, output_file: str) -> bool:
    """
    Run zero_to_fp32.py to convert DeepSpeed checkpoint to pytorch_model.bin.
    Handles both single file and sharded outputs.

    Returns True if successful, False otherwise.
    """
    zero_script = os.path.join(checkpoint_dir, "zero_to_fp32.py")

    if not os.path.exists(zero_script):
        print(f"ERROR: zero_to_fp32.py not found in {checkpoint_dir}")
        return False

    # Check if global_step directory exists
    global_step_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("global_step")]
    if not global_step_dirs:
        print(f"ERROR: No global_step directory found in {checkpoint_dir}")
        return False

    print(f"Converting checkpoint: {checkpoint_dir}")
    print(f"Output file: {output_file}")

    try:
        result = subprocess.run(
            ["python", zero_script, ".", "pytorch_model.bin"],
            cwd=checkpoint_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"ERROR: Conversion failed")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False

        generated_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

        # DeepSpeed can create a directory named pytorch_model.bin that contains
        # pytorch_model-xxxxx-of-xxxxx.bin shards plus pytorch_model.bin.index.json.
        if os.path.isdir(generated_path):
            index_file = os.path.join(generated_path, "pytorch_model.bin.index.json")
            if os.path.exists(index_file):
                print("Detected sharded directory format, merging into single file...")
                if merge_sharded_weights(generated_path, output_file):
                    try_remove_path(generated_path)
                    print(f"SUCCESS: Created merged {output_file}")
                    return True
                return False

            print(f"ERROR: {generated_path} is a directory, but no pytorch_model.bin.index.json was found")
            return False

        # Check for sharded format (index.json exists)
        index_file = os.path.join(checkpoint_dir, "pytorch_model.bin.index.json")

        if os.path.exists(index_file):
            print("Detected sharded format, merging into single file...")
            # Merge shards into single file
            if merge_sharded_weights(checkpoint_dir, output_file):
                # Clean up sharded files
                for f in os.listdir(checkpoint_dir):
                    if f.startswith("pytorch_model") and f.endswith(".bin") or f == "pytorch_model.bin.index.json":
                        if f != "pytorch_model.bin":
                            try_remove_path(os.path.join(checkpoint_dir, f))
                print(f"SUCCESS: Created merged {output_file}")
                return True
            return False
        else:
            # Single file format - move to output location
            if os.path.isfile(generated_path):
                shutil.move(generated_path, output_file)
                print(f"SUCCESS: Created {output_file}")
                return True
            else:
                print(f"ERROR: pytorch_model.bin not generated")
                return False

    except Exception as e:
        print(f"ERROR: Exception during conversion: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert DeepSpeed checkpoint for fine-tuning")
    parser.add_argument("output_dir", help="Path to pretrain output directory (e.g., /path/to/ICAE_pretrain)")
    parser.add_argument("--output_name", default="checkpoint_best",
                        help="Name of output directory (default: checkpoint_best)")
    parser.add_argument("--force", action="store_true",
                        help="Force conversion even if checkpoint_best exists")
    args = parser.parse_args()

    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        print(f"ERROR: Directory not found: {output_dir}")
        return 1

    # Check if checkpoint_best already exists
    best_dir = os.path.join(output_dir, args.output_name)
    best_model = os.path.join(best_dir, "pytorch_model.bin")

    if os.path.exists(best_model):
        if os.path.isfile(best_model) and not args.force:
            print(f"checkpoint_best already exists: {best_model}")
            print("Use --force to overwrite")
            return 0

        if os.path.isdir(best_model) and not args.force:
            if repair_sharded_output_dir(best_model):
                print(f"\nReady for fine-tuning!")
                print(f"Use --lora_path {best_model}")
                return 0

            print(f"ERROR: {best_model} exists but is a directory, not a single model file")
            print("Use --force to remove it and regenerate a single pytorch_model.bin file")
            return 1

    # Find latest checkpoint
    latest_checkpoint = find_latest_checkpoint(output_dir)

    if not latest_checkpoint:
        print(f"ERROR: No checkpoints found in {output_dir}")
        return 1

    print(f"Found latest checkpoint: {latest_checkpoint}")

    # Create output directory
    os.makedirs(best_dir, exist_ok=True)

    if os.path.exists(best_model) and args.force:
        remove_path(best_model)

    # Convert
    if convert_checkpoint(latest_checkpoint, best_model):
        print(f"\nReady for fine-tuning!")
        print(f"Use --lora_path {best_model}")
        return 0
    else:
        # Clean up empty directory
        if os.path.exists(best_dir) and not os.listdir(best_dir):
            os.rmdir(best_dir)
        return 1


if __name__ == "__main__":
    exit(main())
