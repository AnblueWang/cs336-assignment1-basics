from torch.nn import Module, Parameter
import torch, math
from einops import einsum

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        assert in_features > 0 and out_features > 0
        super().__init__()
        std = math.sqrt(2.0/(in_features+out_features))
        self.weights = Parameter(torch.nn.init.trunc_normal_(torch.randn(out_features, in_features, device=device, dtype=dtype), 0, std=std, a=-3*std, b=3*std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... in_features, out_features in_features -> ... out_features")