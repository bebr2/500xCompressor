import argparse


DEFAULT_CONTEXT = """We show that every reciprocity sheaf gives rise to a cycle (pre)module in the sense of Rost over a perfect field. Over a perfect field of positive characteristic, we show that the first cohomology group of a logarithmic de Rham-Witt sheaf has a partial cycle module structure. As a consequence, we show that Kato complexes of logarithmic de Rham-Witt sheaves satisfy functoriality properties similar to Rost's cycle complexes."""
DEFAULT_QUESTION = "Over what type of field do we show that Kato complexes satisfy functoriality properties?"
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Demo for ICAE QA/regeneration inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/hdfs/wangchangyue/LLM/Qwen3-8B",
        help="Path to the base LLM model",
    )
    parser.add_argument(
        "--lora_path_regen",
        type=str,
        default="",
        help="Path to ICAE LoRA parameters for regeneration",
    )
    parser.add_argument(
        "--lora_path_qa",
        type=str,
        default="/mnt/hdfs/wangchangyue/500xCompressor/output/ICAE_finetune-Qwen3-8B/checkpoint_best/pytorch_model.bin",
        help="Path to ICAE LoRA parameters for QA",
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="Hugging Face cache directory")
    parser.add_argument("--use_auth_token", type=str, default=None, help="Hugging Face auth token")
    parser.add_argument("--num_mem", type=int, default=256, help="Number of compressed tokens")
    parser.add_argument("--mode", choices=["regeneration", "qa"], default="qa")
    parser.add_argument("--max_length", type=int, default=512, help="Max context tokens to compress")
    parser.add_argument(
        "--context_len",
        type=int,
        default=None,
        help="Deprecated. ICAE demo now pads directly to --max_length to match baseline inference.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=96, help="Max generated tokens")
    parser.add_argument("--context", type=str, default=DEFAULT_CONTEXT, help="Context text to compress")
    parser.add_argument("--question", type=str, default=DEFAULT_QUESTION, help="Question for QA mode")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, nargs="+", default=list(DEFAULT_TARGET_MODULES))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    import torch
    from peft import LoraConfig
    from L3ICAE import L3ICAE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.context_len is not None and args.context_len != args.max_length:
        print("WARNING: --context_len is ignored; using --max_length for ICAE compression padding.")

    if args.mode == "regeneration":
        lora_path = args.lora_path_regen
        if not lora_path:
            raise ValueError("--lora_path_regen is required when --mode regeneration")
    else:
        lora_path = args.lora_path_qa
        if not lora_path:
            raise ValueError("--lora_path_qa is required when --mode qa")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )

    model = L3ICAE(
        llama_path=args.model_path,
        cache_dir=args.cache_dir,
        use_auth_token=args.use_auth_token,
        max_length=args.max_length,
        lora_path=lora_path,
        lora_config=lora_config,
        num_mem=args.num_mem,
        device=device,
    )
    model = model.to(device)
    model.eval()

    if args.mode == "regeneration":
        prompt = "ae"
    else:
        prompt = f"Question: {args.question}\nAnswer: "

    with torch.no_grad():
        mem_vec = model.compress(text=args.context, output_path=None)
        predicted_text = model.predict(
            mem_vec=mem_vec,
            max_new_tokens=args.max_new_tokens,
            prompt=prompt,
        )

    print("Predicted text: " + predicted_text)
