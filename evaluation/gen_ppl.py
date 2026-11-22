import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class GenPPLResult:
    nll_sum: torch.Tensor
    token_count: torch.Tensor
    ppl: torch.Tensor
    mean_nll: float
    mean_ppl: float


@torch.no_grad()
def generation_perplexity(
    ref_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    generated_texts: List[str],
    *,
    device: Optional[str] = None,
    score_only_generated: bool = True,
    batch_size: int = 8,
    max_length: Optional[int] = None,
) -> GenPPLResult:

    assert len(prompts) == len(generated_texts), "prompts and generated_texts must align"
    ref_model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ref_model.to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    
    use_offsets = getattr(tokenizer, "is_fast", False)

    n = len(prompts)
    nll_sums = []
    tok_counts = []

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_prompts = prompts[batch_start:batch_end]
        batch_gens = generated_texts[batch_start:batch_end]

        full_texts = [p + g for p, g in zip(batch_prompts, batch_gens)]

        _full_max = max_length if max_length is not None else 1024
        enc_full = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=_full_max,
            return_offsets_mapping=use_offsets,
        ).to(device)

        input_ids = enc_full["input_ids"]
        attn_mask = enc_full["attention_mask"]
        offset_mapping = enc_full.get("offset_mapping", None)
        
        if offset_mapping is not None:
            offset_mapping = offset_mapping.cpu()
            attn_mask_cpu  = attn_mask.cpu()

        logits = ref_model(input_ids=input_ids, attention_mask=attn_mask).logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask   = attn_mask[:, 1:]

        include = shift_mask.bool().clone()

        if score_only_generated:
            B = input_ids.size(0)
            T = input_ids.size(1)

            if offset_mapping is not None:
                real_token = (
                    (attn_mask_cpu == 1) &
                    ~((offset_mapping[:, :, 0] == 0) & (offset_mapping[:, :, 1] == 0))
                )

                prompt_char_lens = torch.tensor(
                    [len(p) for p in batch_prompts]
                ).unsqueeze(1)
                is_gen = real_token & (offset_mapping[:, :, 0] >= prompt_char_lens)

                # argmax returns 0 for all-False rows, so fix those up explicitly.
                no_gen    = ~is_gen.any(dim=1)
                first_gen = is_gen.long().argmax(dim=1)
                first_gen[no_gen] = T

                # In shift_labels (= input_ids[:, 1:]) the first generated token sits at first_gen - 1.
                start_pos_vec = (first_gen - 1).clamp(min=0).to(device)
                positions = torch.arange(T - 1, device=device).unsqueeze(0)
                include = include & (positions >= start_pos_vec.unsqueeze(1))
            else:
                enc_prompt = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                prompt_lens = enc_prompt["attention_mask"].sum(dim=1).to(device)
                # argmax on attn_mask gives the index of the first real token (N for left-padding with N pads).
                padding_offset = attn_mask.long().argmax(dim=1)
                for i in range(B):
                    start_pos = max(int((padding_offset[i] + prompt_lens[i]).item()) - 1, 0)
                    include[i, :start_pos] = False

        log_probs = F.log_softmax(shift_logits, dim=-1)
        nll_tok   = -log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        nll_sum   = (nll_tok * include.float()).sum(dim=1)
        tok_count = include.sum(dim=1).to(torch.long)

        nll_sums.append(nll_sum.detach().cpu())
        tok_counts.append(tok_count.detach().cpu())

    nll_sum   = torch.cat(nll_sums,   dim=0)
    tok_count = torch.cat(tok_counts, dim=0)

    tok_count_safe  = torch.clamp(tok_count, min=1)
    mean_nll_per_ex = nll_sum / tok_count_safe.float()
    ppl_per_ex      = torch.exp(mean_nll_per_ex)

    total_nll = float(nll_sum.sum().item())
    total_tok = int(tok_count.sum().item())
    mean_nll  = total_nll / max(total_tok, 1)
    mean_ppl  = math.exp(mean_nll)

    return GenPPLResult(
        nll_sum=nll_sum,
        token_count=tok_count,
        ppl=ppl_per_ex,
        mean_nll=mean_nll,
        mean_ppl=mean_ppl,
    )
