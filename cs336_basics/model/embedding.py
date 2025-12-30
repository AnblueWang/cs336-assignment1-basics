from torch.nn import Module, Parameter
import torch
from einops import einsum

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device|None=None, dtype:torch.dtype|None=None):
        super().__init__()
        self.embeddings = Parameter(torch.nn.init.trunc_normal_(torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype), 0, 1, -3, 3))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]