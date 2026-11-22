import math
import os

import torch


def default_device():
    """NCM_DEVICE if set, else CUDA, else Apple-silicon MPS, else CPU."""
    if os.environ.get("NCM_DEVICE"):
        return os.environ["NCM_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def wide_float(device):
    """float64 where supported; MPS has no float64, so fall back to float32 there."""
    return torch.float32 if str(device).startswith("mps") else torch.float64


def count_params(model, exclude_params=[]):
    n_params = sum(p.numel() for p in model.parameters())
    
    non_embd_params = n_params
    
    if exclude_params:
        for name, param in model.named_parameters():
            if any([exclude_param in name for exclude_param in exclude_params]):
                non_embd_params -= param.numel()
                
    return n_params, non_embd_params


def round_levels(levels, start, end):
    rounded_levels = [start]
    for i in range(1, len(levels)-1):
        if math.floor(levels[i]) > rounded_levels[i-1] and math.ceil(levels[i]) < end:
            rounded_levels.append(math.floor(levels[i]))
        elif math.floor(levels[i]) <= rounded_levels[i-1] and math.ceil(levels[i]) > rounded_levels[i-1] and math.ceil(levels[i]) < end:
            rounded_levels.append(math.ceil(levels[i]))
        else:
            rounded_levels.append(rounded_levels[i-1] + 1)
            
    rounded_levels.append(end)
        
    return rounded_levels

