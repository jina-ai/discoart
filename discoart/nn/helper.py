import numpy as np
import torch
import random


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def detach_gpu(val):
    if isinstance(val, (int, float)):
        return val
    else:
        return val.detach().cpu().item()
