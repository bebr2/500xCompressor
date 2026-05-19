from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from pathlib import Path
from queue import Empty
from typing import Any, Callable

import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
_SKILL_BLOCK_RE = re.compile(r"^\s*Relevant Skill:\s*.*?\n\s*\n", re.DOTALL | re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ICAE/500xCompressor baselines on SRA-Bench-style inputs."
    )
    parser.add_argument("--baseline", choices=["icae", "500x"], required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Base LLM path")
    parser.add_argument("--lora_path", type=str, required=True, help="Finetuned baseline checkpoint")

    parser.add_argument("--dataset", "--name", type=str, default="", help="Dataset name, default all")
    parser.add_argument("--method", type=str, default="naive", help="naive, golden_skill, or retrieval label")
    parser.add_argument("--retrieval_results", type=str, default="", help="Retrieval JSON for retrieval methods")
    parser.add_argument("--top_k", type=int, default=1)

    parser.add_argument("--skillrag_root", type=str, default="", help="SkillRAG-main root")
    parser.add_argument("--rerank_root", type=str, default="", help="Rerank repo root containing code/skillrag_vendor")
    parser.add_argument("--instances_dir", type=str, default="", help="Directory containing <dataset>.json")
    parser.add_argument("--corpus_path", type=str, default="", help="Skill corpus path or directory")
    parser.add_argument("--toolqa_data_dir", type=str, default="", help="ToolQA external corpus directory")

    parser.add_argument("--num_mem", type=int, default=256, help="Compressed token count")
    parser.add_argument("--max_length", type=int, default=4096, help="Max selected-skill context tokens to compress")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Direct-generation max new tokens")
    parser.add_argument("--toolqa_max_steps", type=int, default=20)
    parser.add_argument("--toolqa_step_tokens", type=int, default=512)

    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bias", type=str, default="none")
    parser.add_argument("--lora_task_type", type=str, default="CAUSAL_LM")
    parser.add_argument("--target_modules", type=str, nargs="+", default=list(_DEFAULT_TARGET_MODULES))

    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--fp16", type=str, default="false", help="SRA-compatible precision flag")
    parser.add_argument("--torch_dtype", choices=["bf16", "fp16", "float32"], default="bf16")

    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--result_path", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable_parallel", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Limit pending instances for smoke tests")
    return parser.parse_args()


def _parse_target_modules(raw: Any) -> list[str]:
    if raw is None:
        return list(_DEFAULT_TARGET_MODULES)
    if isinstance(raw, str):
        values = [raw]
    else:
        values = list(raw)

    out: list[str] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                out.append(item)
    return out or list(_DEFAULT_TARGET_MODULES)


def _torch_dtype(name: str) -> torch.dtype:
    norm = str(name).strip().lower()
    if norm == "fp16":
        return torch.float16
    if norm == "float32":
        return torch.float32
    return torch.bfloat16


def _candidate_rerank_roots(rerank_root: str) -> list[Path]:
    roots: list[Path] = []
    if rerank_root.strip():
        roots.append(Path(rerank_root))
    env_root = os.environ.get("RERANK_ROOT", "").strip()
    if env_root:
        roots.append(Path(env_root))
    roots.append(_REPO_ROOT.parent / "Rerank")
    roots.append(Path("/mnt/hdfs/wangchangyue/Rerank"))

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def _resolve_rerank_root(rerank_root: str) -> Path:
    for root in _candidate_rerank_roots(rerank_root):
        if (root / "code" / "skillrag_vendor").exists():
            return root
    tried = [str(x) for x in _candidate_rerank_roots(rerank_root)]
    raise FileNotFoundError(
        "Cannot locate Rerank repo with code/skillrag_vendor. "
        f"Set --rerank_root explicitly. Tried: {tried}"
    )


def _load_sra_modules(rerank_root: str) -> dict[str, Any]:
    root = _resolve_rerank_root(rerank_root)
    code_root = root / "code"
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))

    prompts = importlib.import_module("skillrag_vendor.prompts")
    toolqa_pkg = importlib.import_module("skillrag_vendor.toolqa")
    fewshots = importlib.import_module("skillrag_vendor.toolqa.fewshots")
    react = importlib.import_module("skillrag_vendor.toolqa.react")
    return {
        "rerank_root": root,
        "ALL_DATASETS": getattr(prompts, "ALL_DATASETS"),
        "build_prompt": getattr(prompts, "build_prompt"),
        "ToolEnvironment": getattr(toolqa_pkg, "ToolEnvironment"),
        "TOOLQA_EXAMPLES": getattr(fewshots, "TOOLQA_EXAMPLES", ""),
        "ReActAgent": getattr(react, "ReActAgent"),
    }


def _load_toolqa_examples(skillrag_root: str, fallback: str) -> str:
    if skillrag_root.strip():
        src = Path(skillrag_root) / "src"
        if src.exists() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
        try:
            module = importlib.import_module("skillrag.toolqa.fewshots")
            examples = getattr(module, "TOOLQA_EXAMPLES", "")
            if isinstance(examples, str) and examples.strip():
                return examples
        except Exception:
            pass
    return fallback


def _resolve_instances_dir(args: argparse.Namespace, rerank_root: Path) -> Path:
    if args.instances_dir.strip():
        return Path(args.instances_dir)
    if args.skillrag_root.strip():
        return Path(args.skillrag_root) / "data" / "bench" / "instances"
    return rerank_root / "prepare" / "output" / "trainset"


def _resolve_default_corpus_path(args: argparse.Namespace, rerank_root: Path) -> str:
    if args.corpus_path.strip():
        return args.corpus_path.strip()
    if args.skillrag_root.strip():
        return str(Path(args.skillrag_root) / "data" / "bench" / "corpus" / "corpus.json")
    return str(rerank_root / "prepare" / "output" / "corpus.json")


def _resolve_toolqa_data_dir(args: argparse.Namespace) -> Path:
    if args.toolqa_data_dir.strip():
        path = Path(args.toolqa_data_dir)
        if path.exists():
            return path
        raise FileNotFoundError(f"toolqa_data_dir does not exist: {path}")
    if args.skillrag_root.strip():
        for relative in ("data/external/toolqa", "data/external_corpus"):
            path = Path(args.skillrag_root) / relative
            if path.exists():
                return path
    raise FileNotFoundError("ToolQA data directory is required for dataset=toolqa. Set --toolqa_data_dir.")


def _resolve_corpus_file(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_file():
        return path
    if path.is_dir():
        for candidate in (path / "corpus.json", path / "corpus.json" / "corpus.json"):
            if candidate.is_file():
                return candidate
    if path.suffix == "":
        with_suffix = Path(str(path) + ".json")
        if with_suffix.is_file():
            return with_suffix
    raise FileNotFoundError(f"Cannot resolve corpus file from path: {path_like}")


def _collect_skill_items(obj: Any, out: list[dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        if "skill_id" in obj:
            out.append(obj)
        for value in obj.values():
            if isinstance(value, (dict, list)):
                _collect_skill_items(value, out)
        return
    if isinstance(obj, list):
        for item in obj:
            _collect_skill_items(item, out)


def load_skill_corpus(path_like: str) -> dict[str, dict[str, Any]]:
    corpus_file = _resolve_corpus_file(path_like)
    mapping: dict[str, dict[str, Any]] = {}

    if corpus_file.suffix.lower() == ".jsonl":
        with open(corpus_file, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                if isinstance(row, dict):
                    skill_id = str(row.get("skill_id", "")).strip()
                    if skill_id:
                        mapping[skill_id] = row
    else:
        with open(corpus_file, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        items: list[dict[str, Any]] = []
        _collect_skill_items(payload, items)
        for item in items:
            skill_id = str(item.get("skill_id", "")).strip()
            if skill_id:
                mapping[skill_id] = item

    if not mapping:
        raise ValueError(f"No skill rows with skill_id found in corpus: {corpus_file}")
    return mapping


def load_instances(instances_dir: Path, dataset: str) -> list[dict[str, Any]]:
    candidates = [
        instances_dir / f"{dataset}.json",
        instances_dir / dataset / "instances.json",
        instances_dir / dataset / "Qwen3-32B" / "golden_skill_messages.json",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Dataset instances file not found. Tried: {[str(x) for x in candidates]}")

    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    if isinstance(payload, dict):
        for key in ("instances", "data", "items", "records", "samples"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload in instances file: {path}")

    out: list[dict[str, Any]] = []
    for row in payload:
        if isinstance(row, dict):
            row.setdefault("dataset", dataset)
            out.append(row)
    return out


def load_retrieval_map(path: str, top_k: int) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        raise ValueError("retrieval_results must contain a top-level 'results' list")

    out: dict[str, list[str]] = {}
    for row in results:
        if not isinstance(row, dict):
            continue
        instance_id = str(row.get("instance_id", "")).strip()
        retrieved = row.get("retrieved")
        if not instance_id or not isinstance(retrieved, list):
            continue

        skill_ids: list[str] = []
        for item in retrieved[: max(1, int(top_k))]:
            if isinstance(item, dict):
                skill_id = str(item.get("skill_id", "")).strip()
                if skill_id:
                    skill_ids.append(skill_id)
            elif isinstance(item, str) and item.strip():
                skill_ids.append(item.strip())
        out[instance_id] = skill_ids
    return out


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _select_skill_ids(
    instance: dict[str, Any],
    method: str,
    retrieval_map: dict[str, list[str]] | None,
) -> list[str]:
    if method == "naive":
        return []
    if method == "golden_skill":
        return _as_string_list(instance.get("skill_annotations")) or _as_string_list(instance.get("skill_ids_used"))
    if retrieval_map is None:
        return []
    instance_id = str(instance.get("instance_id", "")).strip()
    return list(retrieval_map.get(instance_id, []))


def _valid_skill_ids(skill_ids: list[str], corpus: dict[str, dict[str, Any]]) -> list[str]:
    return [skill_id for skill_id in skill_ids if skill_id in corpus]


def _skill_context(skill_ids: list[str], corpus: dict[str, dict[str, Any]]) -> str:
    texts: list[str] = []
    for skill_id in skill_ids:
        row = corpus.get(skill_id, {})
        content = row.get("content", "") if isinstance(row, dict) else ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if content.strip():
            texts.append(content.strip())
    return "\n---\n".join(texts)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _looks_like_messages(obj: Any) -> bool:
    return isinstance(obj, list) and all(isinstance(x, dict) and "role" in x for x in obj)


def _extract_messages(obj: Any) -> list[dict[str, Any]] | None:
    if _looks_like_messages(obj):
        return obj
    if isinstance(obj, list):
        if len(obj) == 1 and _looks_like_messages(obj[0]):
            return obj[0]
        for item in obj:
            found = _extract_messages(item)
            if found is not None:
                return found
    return None


def _prompt_from_messages(instance: dict[str, Any]) -> tuple[str, str] | None:
    messages = _extract_messages(instance.get("messages"))
    if messages is None:
        return None

    system_parts: list[str] = []
    user_text = ""
    seen_user = False
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = _content_to_text(message.get("content", ""))
        if role == "system" and not seen_user:
            system_parts.append(content)
        elif role == "user" and not user_text:
            user_text = content
            seen_user = True
        elif role == "assistant" and seen_user:
            break

    if not user_text.strip():
        return None
    user_text = _SKILL_BLOCK_RE.sub("", user_text).strip()
    return "\n\n".join(x for x in system_parts if x.strip()), user_text


def _build_case_prompt(instance: dict[str, Any], build_prompt: Callable[[dict[str, Any], str], tuple[str, str]]) -> tuple[str, str]:
    try:
        return build_prompt(instance, method="naive")
    except Exception:
        fallback = _prompt_from_messages(instance)
        if fallback is not None:
            return fallback
        raise


def _format_question(system_text: str, user_text: str) -> str:
    parts = [text.strip() for text in (system_text, user_text) if text and text.strip()]
    return "\n\n".join(parts)


def _strip_prompt_echo(generated: str, step_n: int | None = None) -> str:
    text = generated.strip()
    if step_n is None:
        return text
    marker = f"Thought {step_n}:"
    if marker in text:
        return text.rsplit(marker, 1)[-1].strip()
    return text


def _resolve_checkpoint_file(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_file():
        return path
    if path.is_dir():
        candidates = [
            path / "pytorch_model.bin",
            path / "adapter_model.bin",
            path / "model.safetensors",
            path / "adapter_model.safetensors",
            path / "checkpoint_best" / "pytorch_model.bin",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Cannot resolve checkpoint file from: {path_like}")


def _load_checkpoint_state(path_like: str) -> dict[str, torch.Tensor]:
    path = _resolve_checkpoint_file(path_like)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("safetensors is required to load this checkpoint") from exc
        state = load_file(str(path))
    else:
        state = torch.load(path, map_location="cpu")

    if isinstance(state, dict):
        for key in ("state_dict", "model", "module"):
            nested = state.get(key)
            if isinstance(nested, dict) and nested and all(torch.is_tensor(v) for v in nested.values()):
                state = nested
                break

    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint payload: {path}")
    return {str(k): v for k, v in state.items() if torch.is_tensor(v)}


class BaselineCompressorQA(nn.Module):
    def __init__(self, args: argparse.Namespace, device: torch.device):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.baseline = args.baseline
        self.device = device
        self.max_length = int(args.max_length)
        self.num_mem = int(args.num_mem)
        self.dtype = _torch_dtype(args.torch_dtype)

        llama = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        hidden_size = int(llama.config.hidden_size)
        lora_config = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            bias=str(args.lora_bias),
            task_type=str(args.lora_task_type),
            target_modules=_parse_target_modules(args.target_modules),
        )
        self.llama = get_peft_model(llama, lora_config)
        self.llama.eval()
        for name, param in self.llama.named_parameters():
            param.requires_grad = False
            if "lora" in name and param.dtype != self.dtype:
                param.data = param.data.to(self.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.memory_embeddings = nn.Parameter(
            torch.randn(1, self.num_mem, hidden_size, dtype=self.dtype, device=device),
            requires_grad=False,
        )
        self.hidden_size = hidden_size

        self.llama.to(device)
        self._load_trainable_checkpoint(args.lora_path)
        self.stop_token_ids = self._build_stop_token_ids()

    def _build_stop_token_ids(self) -> set[int]:
        stop_ids: set[int] = set()
        for value in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id):
            if value is not None:
                stop_ids.add(int(value))
        for token in ("<|eot_id|>", "<|endoftext|>", "<|im_end|>"):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0 and token_id != self.tokenizer.unk_token_id:
                stop_ids.add(int(token_id))
        return stop_ids

    def _load_trainable_checkpoint(self, lora_path: str) -> None:
        state = _load_checkpoint_state(lora_path)
        loaded = 0
        missing: list[str] = []
        mismatched: list[str] = []

        with torch.no_grad():
            for name, param in self.named_parameters():
                if "lora" not in name and "memory_embeddings" not in name:
                    continue

                candidates = [name, f"module.{name}"]
                if name.startswith("llama."):
                    stripped = name[len("llama.") :]
                    candidates.extend([stripped, f"module.{stripped}"])

                tensor = next((state[key] for key in candidates if key in state), None)
                if tensor is None:
                    missing.append(name)
                    continue
                if tuple(tensor.shape) != tuple(param.shape):
                    mismatched.append(f"{name}: ckpt={tuple(tensor.shape)} model={tuple(param.shape)}")
                    continue
                param.copy_(tensor.to(device=param.device, dtype=param.dtype))
                loaded += 1

        print(f"Loaded {loaded} trainable tensors from {lora_path}")
        if missing:
            print(f"WARNING: missing {len(missing)} trainable tensors, first few: {missing[:8]}")
        if mismatched:
            print(f"WARNING: shape-mismatched tensors, first few: {mismatched[:8]}")

    def _pad_context_tokens(self, text: str) -> torch.Tensor:
        pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        tokens = torch.full((self.max_length,), int(pad_id), dtype=torch.long)
        if text.strip():
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0]
            tokens[: encoded.shape[0]] = encoded
        return tokens.unsqueeze(0).to(self.device)

    def _compress_icae(self, context_text: str) -> torch.Tensor:
        text_tokens = self._pad_context_tokens(context_text)
        text_embeds = self.llama.get_input_embeddings()(text_tokens)
        memory_embeds = self.memory_embeddings.repeat(text_embeds.shape[0], 1, 1)
        encoder_input = torch.cat((text_embeds, memory_embeds), dim=1)
        output = self.llama(inputs_embeds=encoder_input, output_hidden_states=True)
        return output.hidden_states[-1][:, -self.num_mem :, :]

    def _compress_500x(self, context_text: str):
        text_tokens = self._pad_context_tokens(context_text)
        text_embeds = self.llama.get_input_embeddings()(text_tokens)
        memory_embeds = self.memory_embeddings.repeat(text_embeds.shape[0], 1, 1)
        encoder_input = torch.cat((text_embeds, memory_embeds), dim=1)
        output = self.llama(inputs_embeds=encoder_input, use_cache=True)
        cache = output.past_key_values
        if hasattr(cache, "to_legacy_cache"):
            legacy = cache.to_legacy_cache()
            trimmed = tuple((k[:, :, -self.num_mem :, :], v[:, :, -self.num_mem :, :]) for k, v in legacy)
            return type(cache).from_legacy_cache(trimmed)
        return tuple((k[:, :, -self.num_mem :, :], v[:, :, -self.num_mem :, :]) for k, v in cache)

    def _next_token(self, logits: torch.Tensor, do_sample: bool, temperature: float, top_p: float) -> torch.Tensor:
        if not do_sample or temperature <= 0:
            return torch.argmax(logits, dim=-1)

        logits = logits / max(float(temperature), 1e-6)
        if 0 < top_p < 1:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove_mask = cumulative > float(top_p)
            remove_mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
            probs = torch.softmax(sorted_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            return sorted_indices.gather(-1, sampled).squeeze(-1)

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _decode_ids(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def _generate_icae(
        self,
        mem_vec: torch.Tensor,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> tuple[str, int]:
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        prompt_embeds = self.llama.get_input_embeddings()(prompt_ids)
        input_embeds = torch.cat((mem_vec, prompt_embeds), dim=1)
        generated: list[int] = []
        past_key_values = None

        for step in range(int(max_new_tokens)):
            with self.llama.disable_adapter():
                if step == 0:
                    output = self.llama(inputs_embeds=input_embeds, use_cache=True)
                else:
                    output = self.llama(inputs_embeds=input_embeds, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            next_token = self._next_token(output.logits[:, -1, :], do_sample, temperature, top_p)
            token_id = int(next_token.item())
            if token_id in self.stop_token_ids:
                break
            generated.append(token_id)
            input_embeds = self.llama.get_input_embeddings()(next_token.unsqueeze(0))

        return self._decode_ids(generated), len(generated)

    def _generate_500x(
        self,
        past_key_values,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> tuple[str, int]:
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        generated: list[int] = []

        for _ in range(int(max_new_tokens)):
            input_embeds = self.llama.get_input_embeddings()(input_tokens)
            with self.llama.disable_adapter():
                output = self.llama(inputs_embeds=input_embeds, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            next_token = self._next_token(output.logits[:, -1, :], do_sample, temperature, top_p)
            token_id = int(next_token.item())
            if token_id in self.stop_token_ids:
                break
            generated.append(token_id)
            input_tokens = next_token.unsqueeze(0)

        return self._decode_ids(generated), len(generated)

    @torch.no_grad()
    def answer(
        self,
        context_text: str,
        question_text: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> dict[str, Any]:
        prompt = f"Question: {question_text}\nAnswer: "
        compress_start = time.time()
        compressed = self._compress_icae(context_text) if self.baseline == "icae" else self._compress_500x(context_text)
        compress_time = time.time() - compress_start

        predict_start = time.time()
        if self.baseline == "icae":
            generated, generated_tokens = self._generate_icae(
                compressed, prompt, max_new_tokens, do_sample, temperature, top_p
            )
        else:
            generated, generated_tokens = self._generate_500x(
                compressed, prompt, max_new_tokens, do_sample, temperature, top_p
            )
        predict_time = time.time() - predict_start

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "text": generated.strip(),
            "compress_time": compress_time,
            "predict_time": predict_time,
            "generated_tokens": generated_tokens,
        }


def _run_generation(
    model: BaselineCompressorQA,
    context_text: str,
    system_text: str,
    user_text: str,
    args: argparse.Namespace,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    question = _format_question(system_text, user_text)
    return model.answer(
        context_text=context_text,
        question_text=question,
        max_new_tokens=int(max_new_tokens if max_new_tokens is not None else args.max_new_tokens),
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
    )


def _run_qa_case(
    model: BaselineCompressorQA,
    instance: dict[str, Any],
    method: str,
    corpus: dict[str, dict[str, Any]],
    retrieval_map: dict[str, list[str]] | None,
    args: argparse.Namespace,
    model_short_name: str,
    build_prompt: Callable[[dict[str, Any], str], tuple[str, str]],
) -> dict[str, Any]:
    instance_id = str(instance.get("instance_id", ""))
    dataset = str(instance.get("dataset", "")) or "unknown"
    skill_ids = _valid_skill_ids(_select_skill_ids(instance, method, retrieval_map), corpus)
    context_text = _skill_context(skill_ids, corpus)
    system_text, user_text = _build_case_prompt(instance, build_prompt)

    result = _run_generation(model, context_text, system_text, user_text, args)
    out: dict[str, Any] = {
        "instance_id": instance_id,
        "dataset": dataset,
        "method": method,
        "model": model_short_name,
        "raw_output": result["text"],
        "baseline": args.baseline,
        "compress_time": round(float(result["compress_time"]), 4),
        "predict_time": round(float(result["predict_time"]), 4),
        "generated_tokens": int(result["generated_tokens"]),
    }
    if skill_ids:
        out["skill_ids_used"] = skill_ids
    return out


def _run_toolqa_case(
    model: BaselineCompressorQA,
    instance: dict[str, Any],
    method: str,
    corpus: dict[str, dict[str, Any]],
    retrieval_map: dict[str, list[str]] | None,
    tool_env,
    examples: str,
    args: argparse.Namespace,
    model_short_name: str,
    ReActAgent,
) -> dict[str, Any]:
    case_start = time.time()
    instance_id = str(instance.get("instance_id", ""))
    question = str(instance.get("question", "")).strip()
    if not question:
        fallback = _prompt_from_messages(instance)
        question = fallback[1] if fallback is not None else ""

    skill_ids = _valid_skill_ids(_select_skill_ids(instance, method, retrieval_map), corpus)
    context_text = _skill_context(skill_ids, corpus)
    tool_env.reset()

    def _model_generate(
        system_text: str,
        user_text: str,
        max_tokens: int,
        step_n: int,
        stop_token: str | None,
    ) -> str:
        result = _run_generation(
            model=model,
            context_text=context_text,
            system_text=system_text,
            user_text=user_text,
            args=args,
            max_new_tokens=max_tokens,
        )
        text = _strip_prompt_echo(result["text"], step_n=step_n)
        if stop_token and stop_token in text:
            text = text.split(stop_token, 1)[0]
        return text

    agent = ReActAgent(
        question=question,
        tools=tool_env,
        model_generate=_model_generate,
        examples=examples,
        max_steps=int(args.toolqa_max_steps),
        max_tokens=int(args.toolqa_step_tokens),
        method="naive",
        skills=[],
        skill_ids=[],
        skill_mode="naive",
    )
    agent.run()

    elapsed = time.time() - case_start
    out: dict[str, Any] = {
        "instance_id": instance_id,
        "dataset": "toolqa",
        "method": method,
        "model": model_short_name,
        "raw_output": agent.scratchpad,
        "baseline": args.baseline,
        "n_steps": max(0, int(agent.step_n) - 1),
        "finished": bool(agent.finished),
        "halted": bool(agent.is_halted()),
        "time_seconds": round(elapsed, 2),
    }
    if skill_ids:
        out["skill_ids_used"] = skill_ids
    return out


def _read_done_instance_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            try:
                row = json.loads(line)
            except Exception:
                continue
            instance_id = row.get("instance_id") if isinstance(row, dict) else None
            if isinstance(instance_id, str) and instance_id.strip():
                done.add(instance_id.strip())
    return done


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_evenly(items: list[tuple[int, dict[str, Any]]], num_parts: int) -> list[list[tuple[int, dict[str, Any]]]]:
    shards: list[list[tuple[int, dict[str, Any]]]] = [[] for _ in range(max(1, num_parts))]
    for idx, item in enumerate(items):
        shards[idx % len(shards)].append(item)
    return shards


def _model_short_name(args: argparse.Namespace) -> str:
    return f"{Path(args.model_path).name}-{args.baseline}"


def _dataset_output_path(dataset: str, args: argparse.Namespace, model_short_name: str, is_single_dataset: bool) -> Path:
    if args.result_path.strip():
        if not is_single_dataset:
            raise ValueError("--result_path can only be used with a single dataset")
        return Path(args.result_path)
    output_root = Path(args.output_root) if args.output_root.strip() else (_REPO_ROOT / "results_sra_baseline")
    return output_root / dataset / model_short_name / f"{args.method}.jsonl"


def _validate_args(args: argparse.Namespace) -> None:
    if args.method not in {"naive", "golden_skill"} and not args.retrieval_results.strip():
        raise ValueError("retrieval_results is required when method is not naive/golden_skill")
    if int(args.top_k) <= 0:
        raise ValueError("top_k must be > 0")
    if int(args.max_length) <= 0:
        raise ValueError("max_length must be > 0")
    if int(args.num_mem) <= 0:
        raise ValueError("num_mem must be > 0")


def _worker_infer_shard(
    worker_id: int,
    gpu_id: int,
    shard: list[tuple[int, dict[str, Any]]],
    args: argparse.Namespace,
    dataset: str,
    corpus_path: str,
    retrieval_map: dict[str, list[str]] | None,
    toolqa_data_dir: str,
    toolqa_examples: str,
    model_short_name: str,
    progress_queue,
    result_queue,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        sra = _load_sra_modules(args.rerank_root)
        build_prompt = sra["build_prompt"]
        ReActAgent = sra["ReActAgent"]
        ToolEnvironment = sra["ToolEnvironment"]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = BaselineCompressorQA(args, device=device)
        model.eval()

        needs_corpus = args.method != "naive"
        corpus = load_skill_corpus(corpus_path) if needs_corpus else {}
        tool_env = None
        if dataset == "toolqa":
            tool_env = ToolEnvironment(Path(toolqa_data_dir))
            tool_env._ensure_retrievers_ready()

        rows: list[dict[str, Any]] = []
        for order_idx, instance in shard:
            if dataset == "toolqa":
                row = _run_toolqa_case(
                    model,
                    instance,
                    args.method,
                    corpus,
                    retrieval_map,
                    tool_env,
                    toolqa_examples,
                    args,
                    model_short_name,
                    ReActAgent,
                )
            else:
                row = _run_qa_case(
                    model,
                    instance,
                    args.method,
                    corpus,
                    retrieval_map,
                    args,
                    model_short_name,
                    build_prompt,
                )
            row["__order__"] = order_idx
            rows.append(row)
            progress_queue.put(1)
        result_queue.put({"worker_id": worker_id, "results": rows})
    except Exception as exc:
        result_queue.put(
            {
                "worker_id": worker_id,
                "fatal_error": True,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _run_parallel_dataset(
    indexed_instances: list[tuple[int, dict[str, Any]]],
    args: argparse.Namespace,
    dataset: str,
    corpus_path: str,
    retrieval_map: dict[str, list[str]] | None,
    toolqa_data_dir: str,
    toolqa_examples: str,
    model_short_name: str,
    visible_gpu_count: int,
) -> list[dict[str, Any]]:
    shards = _split_evenly(indexed_instances, visible_gpu_count)
    worker_specs = [(gpu_id, shard) for gpu_id, shard in enumerate(shards) if shard]
    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    result_queue = ctx.Queue()
    processes = []

    for worker_id, (gpu_id, shard) in enumerate(worker_specs):
        process = ctx.Process(
            target=_worker_infer_shard,
            args=(
                worker_id,
                gpu_id,
                shard,
                args,
                dataset,
                corpus_path,
                retrieval_map,
                toolqa_data_dir,
                toolqa_examples,
                model_short_name,
                progress_queue,
                result_queue,
            ),
        )
        process.start()
        processes.append(process)

    merged: list[dict[str, Any]] = []
    finished = 0
    completed_workers: set[int] = set()
    bar = _tqdm(total=len(indexed_instances), desc=f"Infer-{dataset}", dynamic_ncols=True) if _tqdm else None
    while finished < len(processes):
        try:
            progress_queue.get(timeout=0.2)
            if bar is not None:
                bar.update(1)
        except Empty:
            pass

        try:
            payload = result_queue.get_nowait()
        except Empty:
            for worker_idx, process in enumerate(processes):
                if worker_idx in completed_workers:
                    continue
                if process.exitcode not in (None, 0):
                    if bar is not None:
                        bar.close()
                    for other in processes:
                        if other.is_alive():
                            other.terminate()
                    raise RuntimeError(
                        f"Worker {worker_idx} exited before returning results: exitcode={process.exitcode}"
                    )
            continue

        if payload.get("fatal_error"):
            if bar is not None:
                bar.close()
            for process in processes:
                process.terminate()
            for process in processes:
                process.join(timeout=2)
            raise RuntimeError(
                f"Worker {payload.get('worker_id')} failed: {payload.get('error')}\n{payload.get('traceback', '')}"
            )
        merged.extend(payload.get("results", []))
        completed_workers.add(int(payload.get("worker_id", -1)))
        finished += 1

        for worker_idx, process in enumerate(processes):
            if worker_idx in completed_workers:
                continue
            if process.exitcode not in (None, 0):
                if bar is not None:
                    bar.close()
                for other in processes:
                    if other.is_alive():
                        other.terminate()
                raise RuntimeError(f"Worker {worker_idx} exited before returning results: exitcode={process.exitcode}")

    if bar is not None:
        bar.close()
    for process in processes:
        process.join()
    bad_exits = [f"pid={p.pid}, exitcode={p.exitcode}" for p in processes if p.exitcode not in (0, None)]
    if bad_exits:
        raise RuntimeError("Parallel inference failed:\n" + "\n".join(bad_exits))
    return merged


def _run_single_dataset(
    dataset: str,
    args: argparse.Namespace,
    instances_dir: Path,
    corpus_path: str,
    retrieval_map: dict[str, list[str]] | None,
    toolqa_data_dir: str,
    toolqa_examples: str,
    model_short_name: str,
    is_single_dataset: bool,
    sra: dict[str, Any],
) -> None:
    instances = load_instances(instances_dir, dataset)
    out_path = _dataset_output_path(dataset, args, model_short_name, is_single_dataset)

    if args.overwrite and out_path.exists():
        out_path.unlink()

    done_ids = _read_done_instance_ids(out_path) if args.resume else set()
    pending = [row for row in instances if str(row.get("instance_id", "")).strip() not in done_ids]
    if args.limit > 0:
        pending = pending[: int(args.limit)]

    if not pending:
        print(f"[{dataset}] No pending instances. Output: {out_path}")
        return

    print(
        f"[{dataset}] total={len(instances)}, done={len(done_ids)}, pending={len(pending)}, "
        f"baseline={args.baseline}, method={args.method}"
    )

    indexed = list(enumerate(pending))
    visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_parallel = not args.disable_parallel and visible_gpu_count > 1 and len(indexed) > 1
    if use_parallel:
        print(f"[{dataset}] Parallel mode: one worker per visible GPU ({visible_gpu_count})")
        rows = _run_parallel_dataset(
            indexed,
            args,
            dataset,
            corpus_path,
            retrieval_map,
            toolqa_data_dir,
            toolqa_examples,
            model_short_name,
            visible_gpu_count,
        )
    else:
        if args.disable_parallel:
            print(f"[{dataset}] Parallel disabled by flag")
        elif visible_gpu_count <= 1:
            print(f"[{dataset}] Parallel disabled due to single visible GPU/CPU")
        else:
            print(f"[{dataset}] Parallel disabled due to small pending size")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BaselineCompressorQA(args, device=device)
        model.eval()

        needs_corpus = args.method != "naive"
        corpus = load_skill_corpus(corpus_path) if needs_corpus else {}
        tool_env = None
        if dataset == "toolqa":
            tool_env = sra["ToolEnvironment"](Path(toolqa_data_dir))
            tool_env._ensure_retrievers_ready()

        rows = []
        bar = _tqdm(total=len(indexed), desc=f"Infer-{dataset}", dynamic_ncols=True) if _tqdm else None
        for order_idx, instance in indexed:
            if dataset == "toolqa":
                row = _run_toolqa_case(
                    model,
                    instance,
                    args.method,
                    corpus,
                    retrieval_map,
                    tool_env,
                    toolqa_examples,
                    args,
                    model_short_name,
                    sra["ReActAgent"],
                )
            else:
                row = _run_qa_case(
                    model,
                    instance,
                    args.method,
                    corpus,
                    retrieval_map,
                    args,
                    model_short_name,
                    sra["build_prompt"],
                )
            row["__order__"] = order_idx
            rows.append(row)
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()

    rows.sort(key=lambda item: int(item.get("__order__", 0)))
    for row in rows:
        row.pop("__order__", None)
    _append_jsonl(out_path, rows)
    print(f"[{dataset}] Wrote {len(rows)} rows to {out_path}")


def main() -> None:
    mp.freeze_support()
    args = parse_args()
    _validate_args(args)

    sra = _load_sra_modules(args.rerank_root)
    datasets = [args.dataset.strip()] if args.dataset.strip() else list(sra["ALL_DATASETS"])
    instances_dir = _resolve_instances_dir(args, sra["rerank_root"])
    corpus_path = _resolve_default_corpus_path(args, sra["rerank_root"])
    model_short_name = _model_short_name(args)

    retrieval_map = None
    if args.method not in {"naive", "golden_skill"}:
        retrieval_map = load_retrieval_map(args.retrieval_results, args.top_k)
        print(f"Loaded retrieval map: {len(retrieval_map)} instances")

    toolqa_data_dir = ""
    toolqa_examples = ""
    if "toolqa" in datasets:
        toolqa_data_dir = str(_resolve_toolqa_data_dir(args))
        toolqa_examples = _load_toolqa_examples(args.skillrag_root, sra["TOOLQA_EXAMPLES"])

    print(f"Datasets: {datasets}")
    print(f"Instances dir: {instances_dir}")
    print(f"Corpus path: {corpus_path}")
    print(f"Model: {model_short_name}")

    for dataset in datasets:
        _run_single_dataset(
            dataset=dataset,
            args=args,
            instances_dir=instances_dir,
            corpus_path=corpus_path,
            retrieval_map=retrieval_map,
            toolqa_data_dir=toolqa_data_dir,
            toolqa_examples=toolqa_examples,
            model_short_name=model_short_name,
            is_single_dataset=(len(datasets) == 1),
            sra=sra,
        )


if __name__ == "__main__":
    main()
