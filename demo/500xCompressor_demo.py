import json
import time
import torch
import numpy as np
import torch.nn as nn
import argparse
from peft import LoraConfig
from L3LoraL3 import L3LoraL3
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments, Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Demo for 500xCompressor")

    # Model arguments
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Path to the base LLM model")
    parser.add_argument("--lora_path_regen", type=str, required=True,
                        help="Path to LoRA parameters for regeneration")
    parser.add_argument("--lora_path_qa", type=str, required=True,
                        help="Path to LoRA parameters for question-answering")
    parser.add_argument("--num_mem", type=int, default=256,
                        help="Number of compressed tokens")
    parser.add_argument("--mode", type=str, default="qa", choices=["regeneration", "qa"],
                        help="Mode: regeneration or qa")

    # Token arguments
    parser.add_argument("--context_len", type=int, default=2048,
                        help="Max number of input tokens")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Max number of tokens to be compressed")
    parser.add_argument("--max_new_tokens", type=int, default=96,
                        help="Max number of new tokens to generate")

    # Demo inputs
    parser.add_argument("--context", type=str, default=None,
                        help="Context text to compress")
    parser.add_argument("--question", type=str, default=None,
                        help="Question for QA mode")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda")

    # Default context and question if not provided
    if args.context is None:
        args.context = """We show that every reciprocity sheaf gives rise to a cycle (pre)module in the sense of Rost over a perfect field. Over a perfect field of positive characteristic, we show that the first cohomology group of a logarithmic de Rham-Witt sheaf has a partial cycle module structure. As a consequence, we show that Kato complexes of logarithmic de Rham-Witt sheaves satisfy functoriality properties similar to Rost's cycle complexes."""
    if args.question is None:
        args.question = "Over what type of field do we show that Kato complexes satisfy functoriality properties?"

    # identify the lora path according to the mode
    if args.mode == "regeneration":
        lora_path = args.lora_path_regen
    elif args.mode == "qa":
        lora_path = args.lora_path_qa
    else:
        print("""Please specify the mode: "regeneration" or "qa"."""")
        exit(1)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # load model
    model = L3LoraL3(
        llama_path=args.model_path,
        max_length=args.max_length,
        lora_path=lora_path,
        lora_config=lora_config,
        num_mem=args.num_mem,
        device=device
    )
    model = model.to(device)

    back_tokens = torch.full((args.context_len,), model.tokenizer.eos_token_id, dtype=torch.long)

    text_tokens = model.tokenizer(args.context,
                                truncation=True,
                                max_length=args.max_length,
                                return_tensors="pt",
                                add_special_tokens=False).input_ids[0]

    back_tokens[0:0+text_tokens.shape[0]] = text_tokens

    # compress the text
    past_key_values = model.compress(text=args.context, text_tokens=back_tokens.unsqueeze(0), output_path=None)
    for i, layer in enumerate(past_key_values):
        key_shape, value_shape = layer[0].shape, layer[1].shape
        break

    # identify the prompt according to the mode
    if args.mode == "regeneration":
        prompt = "bos"
    elif args.mode == "qa":
        prompt = f"Question: {args.question} Answer: "

    # regenerate the compressed text or do QA based on the compressed tokens
    predicted_text = model.predict(
        past_key_values=past_key_values,
        max_new_tokens=args.max_new_tokens,
        prompt=prompt
    )

    print("Predicted text: " + predicted_text)