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
import shutil
import subprocess
import argparse
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


def convert_checkpoint(checkpoint_dir: str, output_file: str) -> bool:
    """
    Run zero_to_fp32.py to convert DeepSpeed checkpoint to pytorch_model.bin.

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

        # Check if pytorch_model.bin was created
        generated_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(generated_file):
            # Move to output location
            shutil.move(generated_file, output_file)
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

    if os.path.exists(best_model) and not args.force:
        print(f"checkpoint_best already exists: {best_model}")
        print("Use --force to overwrite")
        return 0

    # Find latest checkpoint
    latest_checkpoint = find_latest_checkpoint(output_dir)

    if not latest_checkpoint:
        print(f"ERROR: No checkpoints found in {output_dir}")
        return 1

    print(f"Found latest checkpoint: {latest_checkpoint}")

    # Create output directory
    os.makedirs(best_dir, exist_ok=True)

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