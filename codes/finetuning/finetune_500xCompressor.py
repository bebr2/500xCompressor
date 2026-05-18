import json
import torch
import wandb
import numpy as np
import torch.nn as nn
from rouge import Rouge
from peft import LoraConfig
import argparse
import torch.optim as optim
from L3LoraL3QA import L3LoraL3QA
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments, Trainer
from nltk.translate.bleu_score import sentence_bleu

def read_jsonl_file(file_path):
    """
    Load lines of texts.

    Args:
        file_path (str): Path for lines of texts.

    Returns:
        (List[str]): List of texts.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line.strip())
            data.append(json_data)
    return data


def fit_qa_tokens(q_tokens, a_tokens, max_qa_len):
    q_tokens = q_tokens.reshape(-1)
    a_tokens = a_tokens.reshape(-1)

    if len(q_tokens) >= max_qa_len:
        return q_tokens[:max_qa_len], a_tokens[:0]

    max_answer_len = max_qa_len - len(q_tokens)
    return q_tokens, a_tokens[:max_answer_len]


class TextDataset(Dataset):
    def __init__(
        self,
        text_file,
        llama_path,
        max_context_length,
        max_qa_len,
        num_mem,
        eos_token_id
    ):
        """
        Create the training or evaluation dataset.

        Args:
            text_file (str): Path for lines of texts.
            llama_path (str): Path for the base LLM.
            max_context_length (int): Max number of tokens to be compressed.
            max_qa_len (int): Max number of tokens in each QA pair.
            num_mem (int): Number of compressed tokens.
            eos_token_id (int): EOS token ID for the tokenizer.
        """
        self.text = read_jsonl_file(text_file)
        self.tokenizer = AutoTokenizer.from_pretrained(llama_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_context_length = max_context_length
        self.max_qa_len = max_qa_len
        self.num_mem = num_mem
        self.eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        # context tokens (padding)
        c_tokens = self.tokenizer(self.text[idx]["context"],
                                    truncation=True,
                                    padding="max_length",
                                    max_length=self.max_context_length,
                                    return_tensors="pt",
                                    add_special_tokens=False).input_ids.squeeze()

        # question tokens
        question = self.text[idx]["question"]
        q_tokens = self.tokenizer(f"Question: {question} Answer: ",
                                    return_tensors="pt",
                                    add_special_tokens=False).input_ids.squeeze()

        # answer tokens
        a_tokens = self.tokenizer(self.text[idx]["answer"],
                                    return_tensors="pt",
                                    add_special_tokens=False).input_ids.squeeze()
        if a_tokens.shape == torch.Size([]):
            a_tokens = self.tokenizer(self.text[idx]["answer"],
                                        return_tensors="pt",
                                        add_special_tokens=False).input_ids.reshape(1)
        q_tokens, a_tokens = fit_qa_tokens(q_tokens, a_tokens, self.max_qa_len)
        qa_tokens = torch.cat((q_tokens, a_tokens), dim=0)

        # input tokens: context (padding) + question + answer
        input_ids = torch.full((self.max_context_length+self.max_qa_len,), self.eos_token_id, dtype=torch.long)
        input_ids[:self.max_context_length] = c_tokens
        input_ids[self.max_context_length:self.max_context_length+len(qa_tokens)] = qa_tokens

        # target tokens: answer for the question
        target_tokens = torch.full((self.max_qa_len,), -100, dtype=torch.long)
        target_tokens[len(q_tokens)-1:len(q_tokens)-1+len(a_tokens)+1] = torch.cat((a_tokens, torch.tensor([self.eos_token_id])), dim=0)

        return {"input_ids": input_ids, "labels": target_tokens}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune 500xCompressor model")

    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the base LLM model")
    parser.add_argument("--num_mem", type=int, default=256,
                        help="Number of compressed tokens")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Max number of context tokens to be compressed")
    parser.add_argument("--max_qa_len", type=int, default=46,
                        help="Max number of QA tokens")

    # Data arguments
    parser.add_argument("--train_text_path", type=str, required=True,
                        help="Path to training jsonl file")
    parser.add_argument("--test_text_path", type=str, required=True,
                        help="Path to test/eval jsonl file")

    # LoRA path
    parser.add_argument("--lora_path", type=str, required=True,
                        help="Pretrained LoRA parameters path for initialization")

    # Output arguments
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save model checkpoints")
    parser.add_argument("--logging_dir", type=str, required=True,
                        help="Directory for logging")
    parser.add_argument("--project_name", type=str, default="500xCompressor-finetuning",
                        help="Wandb project name")

    # DeepSpeed arguments
    parser.add_argument("--deepspeed_config", type=str, required=True,
                        help="Path to DeepSpeed configuration JSON file")

    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per GPU for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=48,
                        help="Batch size per GPU for evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every X steps")
    parser.add_argument("--warmup_steps", type=int, default=300,
                        help="Number of warmup steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--eval_accumulation_steps", type=int, default=4,
                        help="Evaluation accumulation steps")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        help="Target modules for LoRA")

    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # DeepSpeed local_rank (passed automatically by deepspeed launcher)
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank for distributed training (passed by DeepSpeed)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f"cuda")

    # Get EOS token ID from tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id
    print(f"EOS token ID: {eos_token_id}")

    # create the train and text datasets
    train_dataset = TextDataset(args.train_text_path, args.model_path, args.max_length, args.max_qa_len, args.num_mem, eos_token_id)
    test_dataset = TextDataset(args.test_text_path, args.model_path, args.max_length, args.max_qa_len, args.num_mem, eos_token_id)
    print("Dataset created.")

    # LoRA configurations
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules
    )

    wandb.init(project=args.project_name, dir="/tmp/wandb")

    # ====================
    # Compression model
    # ====================
    print("Loading llama + lora + llama ...")
    model = L3LoraL3QA(llama_path=args.model_path,
                max_context_length=args.max_length,
                lora_path=args.lora_path,
                lora_config=lora_config,
                num_mem=args.num_mem,
                device=device)
    print("Number of trainable parameters in the model: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Model is on CUDA device:", torch.cuda.current_device())
    model.config = model.llama.config
    print("model.llama.config: ", model.llama.config)
    print("llama + lora + llama loaded successfully.")

    # ====================
    # Training
    # ====================
    # Keep anomaly detection off for normal training; it adds autograd overhead.
    torch.autograd.set_detect_anomaly(False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=8,
        bf16=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        logging_dir=args.logging_dir,
        logging_steps=1,
        report_to=["wandb"],
        deepspeed=args.deepspeed_config,
        learning_rate=args.learning_rate,
        save_total_limit=args.save_total_limit,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # If the checkpoint path is not None
    # resume finetuning
    if args.resume_from_checkpoint is None:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    evaluation_results = trainer.evaluate()
    print("evaluation_results: ", evaluation_results)
