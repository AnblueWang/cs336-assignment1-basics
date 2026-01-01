import torch, math, numpy as np, random, os
from einops import einsum
import typing

def softmax(x: torch.Tensor, di: int) -> torch.Tensor:
    max_value, _ = torch.max(x, dim=di, keepdim=True)
    exp_value = torch.exp(x-max_value)
    sum_value = torch.sum(exp_value, dim=di, keepdim=True)
    return exp_value/sum_value

def scaled_dot_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor|None=None)->torch.Tensor:
    dk = Q.size()[-1]
    prod_qk = einsum(Q, K, "... seq_len1 dk, ... seq_len2 dk -> ... seq_len1 seq_len2")/math.sqrt(dk)
    mask_value = torch.zeros_like(mask, dtype=torch.float32)
    if mask != None:
        mask_value.masked_fill_(~mask, float("-inf"))
    atten_value = softmax(prod_qk+mask_value, di=-1) # (... seq_len seq_len)
    return einsum(atten_value, V, "... seq_len1 seq_len2, ... seq_len2 dv -> ... seq_len1 dv")

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_value, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits-max_value
    sum_value = torch.sum(torch.exp(logits), dim=-1) #(batch, seq_len, vocab_size) -> (batch, seq_len)
    first_half = -torch.gather(logits, dim=-1, index=targets.long().unsqueeze(-1)).squeeze(-1) #(batch, seq_len, vocab_size) -> (batch, seq_len)
    second_half = torch.log(sum_value) #(batch, seq_len)
    return torch.mean(first_half+second_half)
    
def learning_rate_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int):
    if it < warmup_iters:
        return it*max_learning_rate/warmup_iters
    elif it > cosine_cycle_iters:
        return min_learning_rate
    else:
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos((it-warmup_iters)*math.pi/(cosine_cycle_iters-warmup_iters)))
    
def gradient_clipping(parameters: typing.Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float=1e-6) -> None:
    parameters = [p for p in parameters if p.grad is not None]
    # Compute global norm
    total_norm = torch.norm(torch.stack([torch.norm(p.grad, 2) for p in parameters]), 2)
    # Scale all parameters by the same factor
    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for p in parameters:
            p.grad.mul_(clip_coef)

def get_batch_input(input: np.array, batch_size: int, context_length: int, device: torch.device|str="cpu"):
    input_len = len(input)
    
    result = torch.zeros((batch_size, context_length), device=device, dtype=torch.int)
    labels = torch.zeros((batch_size, context_length), device=device, dtype=torch.int)
    for i in range(batch_size):
        index = random.randint(0, input_len-context_length-1)
        result[i] = torch.tensor(input[index: index+context_length], device=device)
        labels[i] = torch.tensor(input[index+1: index+context_length+1], device=device)
    return (result, labels)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration:int,  out: str| os.PathLike| typing.BinaryIO| typing.IO[bytes]):
    state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(state_dict, out)

def load_checkpoint(src: str| os.PathLike| typing.BinaryIO| typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    return state_dict["iteration"]