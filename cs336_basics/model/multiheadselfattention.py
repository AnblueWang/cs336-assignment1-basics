import torch
from torch.nn import Module
from einops import einsum, rearrange
from cs336_basics.model import linear, functions, rope

class MultiHeadSelfAttention(Module):
    def __init__(self, head_num:int, d_model: int, device:torch.device|None=None, dtype:torch.dtype|None=None):
        super().__init__()
        self.weight_K = linear.Linear(d_model, d_model, device=device, dtype=dtype)
        self.weight_Q = linear.Linear(d_model, d_model, device=device, dtype=dtype)
        self.weight_V = linear.Linear(d_model, d_model, device=device, dtype=dtype)
        self.weight_O = linear.Linear(d_model, d_model, device=device, dtype=dtype)
        self.head_num = head_num

    def forward(self, x: torch.Tensor, theta:float|None=None, token_positions: torch.Tensor|None=None) -> torch.Tensor:
        seq_len = x.size()[-2]
        d_model = x.size()[-1]
        K = rearrange(self.weight_K.forward(x), "batch seq_len (h d_k) -> h batch seq_len d_k", h=self.head_num)
        Q = rearrange(self.weight_Q.forward(x), "batch seq_len (h d_k) -> h batch seq_len d_k", h=self.head_num)
        V = rearrange(self.weight_V.forward(x), "batch seq_len (h d_k) -> h batch seq_len d_k", h=self.head_num)

        mask = torch.tril(torch.ones((seq_len,seq_len)), diagonal=0).to(torch.bool)
        if theta != None and token_positions != None:
            rope_layer = rope.RoPE(theta=theta, d_k=d_model/self.head_num, max_seq_len=seq_len)
            K = rope_layer.forward(K, token_positions=token_positions)
            Q = rope_layer.forward(Q, token_positions=token_positions)
        mha = rearrange(functions.scaled_dot_attention(Q, K, V, mask), "h batch seq_len d_k -> batch seq_len (h d_k)")
        return self.weight_O.forward(mha)



    