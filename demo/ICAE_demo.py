import json
import time
import torch
import numpy as np
import torch.nn as nn
from peft import LoraConfig
from L3ICAE import L3ICAE
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments, Trainer

if __name__ == "__main__":
    device = torch.device("cuda")

    # base LLM name in huggingface
    llama_path = "/mnt/hdfs/wangchangyue/LLM/Qwen3-8B"
    # cache path to save the LLM
    cache_dir = None
    # huggingface token to use llama model
    use_auth_token="<to be filled>"
    # llama lora parameters for regeneration
    lora_path_regen = "<to be filled>"
    # llama lora parameters for question-answering
    lora_path_qa = "/mnt/hdfs/wangchangyue/500xCompressor/output/ICAE_finetune-Qwen3-8B/checkpoint_best/pytorch_model.bin"
    # number of tokens used for compression
    num_mem = 256
    # "regeneration" or "qa"
    mode = "qa"

    # max number of input tokens
    context_len = 500
    # max number of input tokens to be compressed
    max_length = 96
    # max number of new tokens to be generated
    max_new_tokens = 96

    context = """# Coffee Price Data — Single-Day Lookup Guide\n\n## Data Source\n\nThe coffee database (`LoadDB[coffee]`) contains daily coffee futures price data. Columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, `Currency`. Dates are stored in YYYY-MM-DD format. All values are stored as strings internally.\n\n## Standard Lookup Workflow\n\nFor any question about a specific trading day, use this three-step workflow:\n\n1. `LoadDB[coffee]`\n2. `FilterDB[Date=YYYY-MM-DD]` — filter to the target date (no spaces around `=`, no quotes around the date value)\n3. `GetValue[ColumnName]` — retrieve the value from the relevant column\n\n**Important**: Write the filter without spaces around the operator and without quotes around the date: `FilterDB[Date=2018-03-01]`. The column names are case-sensitive; use `Open`, `High`, `Low`, `Close`, `Volume` exactly as shown.\n\n## Column Semantics\n\n- `Open` — price at market open\n- `High` — highest price reached during the day\n- `Low` — lowest price reached during the day\n- `Close` — price at market close\n- `Volume` — number of contracts traded\n\n## Derived Computations\n\nSome questions require computing a derived value from two retrieved fields. Use `Calculate[formula]` after retrieving the needed values:\n\n**Percentage change** (based on opening and closing prices):\n- Formula: `(Close - Open) / Open * 100`\n- Round the result to two decimal places before appending `%`\n- Example: if Open = 150.0 and Close = 153.0, then `Calculate[(153.0 - 150.0) / 150.0 * 100]` gives 2.0, which rounds to 2.0 and is reported as `2.0%`. For a result like 1.8234, round to two decimal places: `1.82%`.\n\n**Price direction** (bullish vs bearish):\n- Compare `Close` to `Open` directly\n- If `Close > Open`: the day was **bullish**\n- If `Close <= Open` (including equal): the day was **bearish**\n\n**Daily price spread** (difference between high and low):\n- Formula: `High - Low`\n- Round the result to two decimal places; report as a plain number (no currency unit)\n\n## Notes\n\n- If `FilterDB` returns 0 rows for a date, that trading day may not exist in the data (weekends, holidays). Verify the date and retry.\n- Do not use FilterDB with `>=` or `<=` for date ranges — use a single exact-match filter for single-day lookups."""

    # identify the lora path according to the mode
    if mode == "regeneration":
        lora_path = lora_path_regen
    elif mode == "qa":
        lora_path = lora_path_qa
    else:
        print("""Please specify the mode: "regeneration" or "qa".""")

    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # load llama-3-lora + llama-3
    model = L3ICAE(llama_path=llama_path, 
                    cache_dir=cache_dir,
                    use_auth_token=use_auth_token,
                    max_length=max_length,
                    lora_path=lora_path,
                    lora_config=lora_config,
                    num_mem=num_mem,
                    device=device)
    model = model.to(device)
        
    back_tokens = torch.full((context_len,), model.tokenizer.eos_token_id, dtype=torch.long)

    text_tokens = model.tokenizer(context, 
                                truncation=True, 
                                max_length=max_length, 
                                return_tensors="pt",
                                add_special_tokens=False).input_ids[0]

    back_tokens[0:0+text_tokens.shape[0]] = text_tokens

    # compress the text
    mem_vec = model.compress(text=context, text_tokens=back_tokens.unsqueeze(0), output_path = None)

    
    while True:

        question = input("Please input your question: ")
        if question == "exit":
            break

        # identify the prompt according to the mode
        if mode == "regeneration":
            prompt = "ae"
        elif mode == "qa":
            prompt = f"Question: {question} Answer: "

        # regenerate the compressed text or do QA based on the compressed tokens
        predicted_text = model.predict(mem_vec=mem_vec, 
                                        max_new_tokens=max_new_tokens, 
                                        prompt=prompt)

        print("Predicted text: " + predicted_text)


