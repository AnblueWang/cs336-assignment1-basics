from torch.nn import Module
import torch
from einops import einsum
from cs336_basics.model import linear

class SwiGluFFN(Module):
    def __init__(self, d_model:int, d_ff:int, device:torch.device|None=None, dtype:torch.dtype|None=None):
        super().__init__()
        self.weight1 = linear.Linear(d_model, d_ff, device=device, dtype=dtype)
        self.weight2 = linear.Linear(d_ff, d_model, device=device, dtype=dtype)
        self.weight3 = linear.Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        p1 = self.weight1.forward(x)
        activation = einsum(p1, torch.sigmoid(p1), "... d_ff, ... d_ff -> ... d_ff")
        p2 = self.weight3.forward(x)
        activation_glu = einsum(activation, p2, "... d_ff, ... d_ff -> ... d_ff")
        result = self.weight2.forward(activation_glu)
        return result