import torch
import torch.nn.functional as F

def l1(out1, out2, aggregate=True):
    """
    Compute L1 Distance on last dimension of inputs.
    """
    out = torch.abs(out1-out2)
    if not aggregate:
        return out
    out = out.sum(dim=-1, keepdim=True)
    return out

def dot(out1, out2, aggregate=True):
    """
    Compute Dot product on last dimension of inputs.
    """
    out = (out1*out2)
    if not aggregate:
        return out
    out = out.sum(dim=-1, keepdim=True)
    return out

def cosine(out1, out2, aggregate=True):
    """
    Compute Cosine similarity on last dimension of inputs.
    """
    out1 = F.normalize(out1, dim=-1)
    out2 = F.normalize(out2, dim=-1)
    return dot(out1, out2, aggregate=aggregate)