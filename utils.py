import torch
import torch.nn.init as init
import numpy as np
import torch.nn as nn
import random

def fix_random_seeds(seed: int) -> None:
    """
    Use in the beginning of the program only.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    np.random.seed(seed)
    random.seed(seed)

def normc_initializer(m):
    with torch.no_grad():
        if type(m) == nn.Linear:
            init.normal_(m.weight)
            m.weight *= 1.0 / torch.sqrt(torch.sum(m.weight**2))

def count_parameters(policy) -> int:
    return sum(p.numel() for p in policy.parameters())

def unroll_parameters(parameters) -> torch.tensor:
    parameters = [parameter.data.flatten() for parameter in parameters]
    parameters = torch.cat(parameters, dim=0)

    return parameters

def fill_policy_parameters(policy, parameters) -> None:
    cur_ind = 0
    for param in policy.parameters():
        size = len(param.data)
        param.data = parameters[cur_ind:cur_ind+size].view(param.data.size())
        cur_ind += size

def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y
