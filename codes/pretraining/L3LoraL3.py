import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model


class L3LoraL3(nn.Module):
    def __init__(self, llama_path, max_length, lora_config, num_mem, device):
        """
        Create the compression model: LLM-LoRA + LLM.

        Args:
            llama_path (str): Path for the base LLM.
            max_length (int): Max number of tokens to be compressed.
            lora_config (LoraConfig): LoRA configurations.
            num_mem (int): Number of compressed tokens.
            device (torch.device): CPU or GPU.
        """
        super(L3LoraL3, self).__init__()

        llama = AutoModelForCausalLM.from_pretrained(
            llama_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        hidden_size = llama.config.hidden_size
        print(f"Model hidden size: {hidden_size}")

        self.llama = get_peft_model(llama, lora_config)
        for name, param in self.llama.named_parameters():
            param.requires_grad = False
            if 'lora' in name:
                param.requires_grad = True
                if param.dtype != torch.bfloat16:
                    param.data = param.data.to(torch.bfloat16)

        print(f"Total parameters of llama: {sum(p.numel() for p in self.llama.parameters())}")

        self.tokenizer = AutoTokenizer.from_pretrained(llama_path, trust_remote_code=True)
        print("tokenizer loaded.")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.num_mem = num_mem
        self.memory_embeddings = nn.Parameter(
            torch.randn(1, num_mem, hidden_size, dtype=torch.bfloat16).to(device)
        )
        self.memory_embeddings.requires_grad = True
        self.device = device
        self.hidden_size = hidden_size

        if self.tokenizer.bos_token_id is not None:
            self.bos_token_id = self.tokenizer.bos_token_id
            print(f"BOS token ID: {self.bos_token_id}")
        else:
            self.bos_token_id = self.tokenizer.eos_token_id
            print(f"Model has no BOS token, using EOS token ID as fallback: {self.bos_token_id}")

    def forward(self, input_ids, labels=None, **kwargs):
        ####################
        # Encoder - llama+lora
        ####################
        text_tokens = input_ids
        target_tokens = labels
        text_tok_embeddings = self.llama.get_input_embeddings()(text_tokens).to(self.device)
        memory_tok_embeddings = self.memory_embeddings.repeat(text_tok_embeddings.shape[0], 1, 1).to(self.device)
        encoder_input_embeddings = torch.cat((text_tok_embeddings, memory_tok_embeddings), dim=1)
        encoder_output = self.llama(inputs_embeds=encoder_input_embeddings)
        past_key_values = encoder_output.past_key_values

        # 打印 DynamicCache 的属性来调试
        print(f"past_key_values type: {type(past_key_values).__name__}")
        print(f"past_key_values attributes: {[a for a in dir(past_key_values) if not a.startswith('_')]}")

        # DynamicCache: 直接操作内部缓存
        # 新版 transformers 可能用不同属性名
        if hasattr(past_key_values, 'key_cache'):
            for i in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[i] = past_key_values.key_cache[i][:, :, -self.num_mem:, :]
                past_key_values.value_cache[i] = past_key_values.value_cache[i][:, :, -self.num_mem:, :]
        elif hasattr(past_key_values, 'self_attention_cache'):
            # 可能是这个属性名
            for i in range(len(past_key_values.self_attention_cache)):
                past_key_values.self_attention_cache[i] = past_key_values.self_attention_cache[i][:, :, -self.num_mem:, :]
        else:
            # fallback: 用 to_legacy_cache 转 tuple 再处理
            print("Using to_legacy_cache fallback")
            legacy = past_key_values.to_legacy_cache()
            trimmed = tuple(
                (k[:, :, -self.num_mem:, :], v[:, :, -self.num_mem:, :])
                for k, v in legacy
            )
            # 转回 DynamicCache - 但这个方法可能不行
            # 直接传 tuple 让 decoder 内部处理
            trimmed_cache = trimmed

        trimmed_cache = past_key_values

        ####################
        # Decoder - llama
        ####################
        prompt_tokens = torch.tensor([self.bos_token_id], device=self.device)
        prompt_tok_embeddings = self.llama.get_input_embeddings()(prompt_tokens)
        prompt_tok_embeddings = prompt_tok_embeddings.repeat(text_tok_embeddings.shape[0], 1, 1)

        decoder_input_embeddings = torch.cat((prompt_tok_embeddings, text_tok_embeddings), dim=1)
        with self.llama.disable_adapter():
            decoder_output = self.llama(
                inputs_embeds=decoder_input_embeddings,
                past_key_values=trimmed_cache
            )
        all_logits = decoder_output.logits

        loss = self.criterion(all_logits.view(-1, all_logits.size(-1)), target_tokens.view(-1))

        return {'loss': loss, 'logits': all_logits}