import torch
from transformers import Trainer
from loguru import logger

class CustomTrainer(Trainer):
    def __init__(self, *args, tokenizer, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.code_sep_id = tokenizer.additional_special_tokens_ids[0]

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        prompts = inputs.pop("input_ids")  # with instruction
        with torch.cuda.amp.autocast(enabled=False):
            attention_mask_prompts = prompts.ne(self.tokenizer.pad_token_id).bool()
            all_attention_mask_contexts = []
            for j, prompt in enumerate(prompts):
                idx = [i for i, token_id in enumerate(prompt) if token_id == self.code_sep_id]
                if len(idx) < 1:
                    logger.warning(f"There should be at least one code snippet in the input {j}, use full input")
                    all_attention_mask_contexts.append(attention_mask_prompts[j])
                else:
                    context_idx = idx[0] + 1
                    attention_mask_context = attention_mask_prompts[j]
                    attention_mask_context[:context_idx] = 0
                    all_attention_mask_contexts.append(attention_mask_context)
            attention_mask_contexts = torch.stack(all_attention_mask_contexts)
            del all_attention_mask_contexts
            # torch.cuda.empty_cache()
            full_outputs = model(input_ids=prompts, attention_mask=attention_mask_prompts, labels=labels)
            full_loss = full_outputs.loss
            reduced_outputs = model(input_ids=prompts, attention_mask=attention_mask_contexts, labels=labels)
            reduced_loss = reduced_outputs.loss
            # ifd_loss = full_loss / (reduced_loss + 1e-8)  # avoid division by 0
            ifd_loss = 0.0
            loss_diff = full_loss - reduced_loss
            weight = 1.0 * (1 + max(loss_diff, 0))
            ifd_loss += weight * full_loss  # avoid bad instruction
        return (ifd_loss, full_outputs) if return_outputs else ifd_loss