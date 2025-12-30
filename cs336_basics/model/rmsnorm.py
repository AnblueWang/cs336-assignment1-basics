from torch.nn import Module, Parameter
import torch
from einops import reduce,einsum


class RMSNorm(Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device|None=None, dtype: torch.dtype|None=None):
        super().__init__()
        self.gain = Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        sqrt_mean = 1/torch.sqrt(reduce(torch.square(x), "... dim -> ...", "mean")+self.eps)
        result = einsum(einsum(x,self.gain, "... dim, dim -> ... dim"), sqrt_mean, "... dim, ... -> ... dim")
        return result.to(in_type)