import torch
import torch.nn as nn
from peft import get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

def load_lora_parameters(model, lora_params_path):
    """
    Initialize the LoRA parameters.

    model (AutoModelForCausalLM): LLM with LoRA parameters.
    lora_params_path (str): LoRA parameters path for initialization.
    """
    # load the LoRA parameters
    lora_params = torch.load(lora_params_path)

    # initialize the LoRA parameters and the compressed token in the LLM
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in lora_params:
                if 'lora' in name or 'memory_embeddings' in name:
                    param.copy_(lora_params[name])
                else:
                    print(f"No saved parameter for {name}")
            elif "lora" in name:
                print(f"No saved parameter for {name}")

class L3LoraL3(nn.Module):
    def __init__(
        self,
        llama_path,
        max_length,
        lora_path,
        lora_config,
        num_mem,
        device
    ):
        """
        Create the compression model: LLM-LoRA + LLM.

        Args:
            llama_path (str): Path for the base LLM.
            max_length (int): Max number of context tokens to be compressed (truncation or padding).
            lora_path (str): Pretrained LoRA parameters for initialization.
            lora_config (LoraConfig): LoRA configurations.
            num_mem (int): Number of compressed tokens.
            device (torch.device): CPU or GPU.
        """
        super(L3LoraL3, self).__init__()
        # load the original base LLM model
        llama = AutoModelForCausalLM.from_pretrained(
            llama_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        # Get hidden size from model config (compatible with different models)
        hidden_size = llama.config.hidden_size
        print(f"Model hidden size: {hidden_size}")

        # add LoRA parameters to the LLM
        self.llama = get_peft_model(llama, lora_config)
        # all the parameters are not trainable, ensure dtype consistency
        self.llama.eval()
        for name, param in self.llama.named_parameters():
            param.requires_grad = False
            # Convert any LoRA params to bfloat16 if needed
            if 'lora' in name and param.dtype != torch.bfloat16:
                param.data = param.data.to(torch.bfloat16)
        print(f"Total parameters of llama: {sum(p.numel() for p in self.llama.parameters())}")
        # load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llama_path, trust_remote_code=True)
        print("tokenizer loaded.")
        # set the padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # max number of context tokens to be compressed (truncation or padding)
        self.max_length = max_length
        # number of compressed tokens
        self.num_mem = num_mem
        # create the compressed token - use dynamic hidden_size
        self.memory_embeddings = nn.Parameter(torch.randn(1, num_mem, hidden_size, dtype=torch.bfloat16).to(device))
        # initialize the LoRA parameters and the compressed token
        load_lora_parameters(self, lora_path)
        self.device = device
        self.hidden_size = hidden_size
        # Handle BOS token - some models (like Qwen) don't have BOS token
        if self.tokenizer.bos_token_id is not None:
            self.bos_token_id = self.tokenizer.bos_token_id
            print(f"BOS token ID: {self.bos_token_id}")
        else:
            # Use EOS token as fallback for models without BOS
            self.bos_token_id = self.tokenizer.eos_token_id
            print(f"Model has no BOS token, using EOS token ID as fallback: {self.bos_token_id}")

    def compress(self, text, text_tokens=None, output_path=None):
        """
        Compress the context into compressed tokens.

        Args:
            text (List[str]): context to be compressed.
            text_token (torch.tensor): context tokens.
            output_path (str): Path to save the K V values for the compressed tokens.

        Returns:
            trimmed_past_key_values: K V values for the compressed tokens.
        """
        # If the input is not context token
        # use the tokenizer to tokenize the context
        if text_tokens is None:
            text_tokens = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.to(self.device)
        else:
            text_tokens = text_tokens.to(self.device)

        text_tok_embeddings = self.llama.get_input_embeddings()(text_tokens)
        # initialize compressed tokens
        memory_tok_embeddings = self.memory_embeddings.repeat(text_tok_embeddings.shape[0], 1, 1).to(self.device)

        # encoder input: original context + compressed tokens
        encoder_input_embeddings = torch.cat((text_tok_embeddings, memory_tok_embeddings), dim=1)
        encoder_output = self.llama(inputs_embeds=encoder_input_embeddings)

        # K V values for the encoder output
        past_key_values = encoder_output.past_key_values

        # DynamicCache 格式：直接切片内部 key_cache/value_cache
        for i in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[i] = past_key_values.key_cache[i][:, :, -self.num_mem:, :]
            past_key_values.value_cache[i] = past_key_values.value_cache[i][:, :, -self.num_mem:, :]
        trimmed_past_key_values = past_key_values

        # save the K V values for the compressed tokens
        if output_path is not None:
            # 保存为 tuple 格式以便兼容
            legacy_cache = trimmed_past_key_values.to_legacy_cache() if hasattr(trimmed_past_key_values, 'to_legacy_cache') else trimmed_past_key_values
            torch.save(legacy_cache, output_path)
            print(f"Saved compressed past_key_values to {output_path}")

        return trimmed_past_key_values

    def predict(self, past_key_values, max_new_tokens, prompt):
        """
        Regenerate the compressed text or do QA based on the compressed tokens.

        Args:
            past_key_values: K V values for the compressed tokens (DynamicCache or tuple).
            max_new_tokens (int): Maximum number of new tokens to generate.
            prompt (str): Prompt text.

        Returns:
            generated_text (str): Generated text.
        """
        # whether the model ends automatically
        end = False

        # 如果传入的是 tuple 格式，需要转换为 DynamicCache
        if isinstance(past_key_values, tuple):
            cache = DynamicCache()
            for layer_idx, (key, value) in enumerate(past_key_values):
                cache.update(layer_idx, key, value)
            past_key_values = cache

        # 获取 batch size
        batch_size = past_key_values.key_cache[0].size(0)

        # input prompt tokens:
        # BOS token for regenerating the compressed text
        # question tokens for QA
        if prompt == "bos":
            prompt_tokens = [[self.bos_token_id]]
        else:
            prompt_tokens = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                add_special_tokens=False
            )
        prompt_tokens = prompt_tokens * batch_size
        input_tokens = torch.tensor(prompt_tokens, device=self.device)

        generated_text = []
        for _ in range(max_new_tokens):
            input_embeddings = self.llama.get_input_embeddings()(input_tokens)

            # predict the next new token by the original LLM
            with self.llama.disable_adapter():
                output = self.llama(
                    inputs_embeds=input_embeddings,
                    past_key_values=past_key_values
                )
            logits = output.logits
            past_key_values = output.past_key_values

            # choose the token id with the highest probability
            next_token = torch.argmax(logits[:, -1, :], dim=-1)

            # stop generating new tokens when meet end tokens
            # Use tokenizer's eos_token_id instead of hardcoded value
            if next_token == self.tokenizer.eos_token_id:
                end = True
                break

            # add the new token to generated tokens
            generated_text.append(next_token.item())

            # update the input token
            input_tokens = next_token.unsqueeze(0)

        # length of the generated tokens
        generated_token_length = len(generated_text)
        generated_text = self.tokenizer.decode(
            generated_text,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return generated_text, end, generated_token_length