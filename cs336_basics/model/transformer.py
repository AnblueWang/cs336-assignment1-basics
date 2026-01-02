import torch
from torch.nn import Module, ModuleList
from cs336_basics.model import rmsnorm, multiheadselfattention, swigluffn, embedding, linear, functions

class TransformerBlock(Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, device:torch.device|None=None, dtype:torch.dtype|None=None):
        super().__init__()
        self.attention_rms = rmsnorm.RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.mha = multiheadselfattention.MultiHeadSelfAttention(num_heads, d_model, device, dtype)
        self.ffn_rms = rmsnorm.RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = swigluffn.SwiGluFFN(d_model, d_ff, device, dtype)

    def forward(self, in_features:torch.Tensor, theta: float) -> torch.Tensor:
        batch,seq_len,_ = in_features.size()
        positions = torch.tile(torch.arange(seq_len), (batch,1))
        y1 = in_features + self.mha.forward(self.attention_rms.forward(in_features), theta, positions)
        output = y1 + self.ffn.forward(self.ffn_rms(y1))
        return output

class TransformerLM(Module):
    def __init__(self, vocab_size:int, context_length:int, num_layers:int, d_model:int, num_heads:int, d_ff:int, device:torch.device|None=None, dtype:torch.dtype|None=None):
        super().__init__()
        self.embedding = embedding.Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = ModuleList([TransformerBlock(d_model, num_heads, d_ff, device, dtype) for _ in range(num_layers)])
        self.output_norm = rmsnorm.RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = linear.Linear(d_model, vocab_size, device, dtype)

    def forward(self, in_indices:torch.Tensor, theta:float|None=None) ->torch.Tensor:
        features = self.embedding.forward(in_indices)
        for layer in self.layers:
            features = layer.forward(features, theta)
        return self.lm_head.forward(self.output_norm(features))
