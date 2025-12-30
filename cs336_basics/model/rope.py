import torch
from torch.nn import Module

class RoPE(Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        pi = torch.arange(max_seq_len, device=device) #(max_seq_len)
        theta_k = 1/(theta**(torch.arange(0, d_k,2,device=device).float()/d_k)) # (d_k/2)
        theta_ik = torch.outer(pi,theta_k) #(max_seq_len, d_k/2)
        cos = torch.repeat_interleave(torch.cos(theta_ik), 2, dim=-1) #(max_seq_len, d_k)
        sin = torch.repeat_interleave(torch.sin(theta_ik), 2, dim=-1) #(max_seq_len, d_k)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor)-> torch.Tensor:
        first_half = x[..., 1::2]
        second_half = x[..., ::2]
        new_x = torch.stack((-first_half,second_half),dim=-1).flatten(-2) # [-x2, x1, -x3, x4]
        return x*(self.cos[token_positions])+new_x*(self.sin[token_positions])
    
