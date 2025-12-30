import torch, math
from einops import einsum

def Softmax(x: torch.Tensor, di: int) -> torch.Tensor:
    max_value, _ = torch.max(x, dim=di, keepdim=True)
    exp_value = torch.exp(x-max_value)
    sum_value = torch.sum(exp_value, dim=di, keepdim=True)
    return exp_value/sum_value

def ScaledDotAttention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor|None=None)->torch.Tensor:
    dk = Q.size()[-1]
    prod_qk = einsum(Q, K, "... seq_len1 dk, ... seq_len2 dk -> ... seq_len1 seq_len2")/math.sqrt(dk)
    mask_value = torch.zeros_like(mask, dtype=torch.float32)
    if mask != None:
        mask_value.masked_fill_(~mask, float("-inf"))
    atten_value = Softmax(prod_qk+mask_value, di=-1) # (... seq_len seq_len)
    return einsum(atten_value, V, "... seq_len1 seq_len2, ... seq_len2 dv -> ... seq_len1 dv")
