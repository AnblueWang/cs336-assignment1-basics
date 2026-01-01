from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
class AdamW(torch.optim.Optimizer):
    def __init__(self, params: torch.nn.Parameter, lr:float=1e-3, betas:tuple[float,float]=(0.9,0.99), eps=1e-8, weight_decay:float=0.1):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps":eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lamd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("step", 0) # Get iteration number from the state, or initial value.
                moment1 = state.get("moment1", 0)
                moment2 = state.get("moment2", 0)
                grad = p.grad.data # Get the gradient of loss with respect to p.
                moment1 = beta1*moment1 + (1-beta1)*grad
                moment2 = beta2*moment2 + (1-beta2)*torch.square(grad)
                lr_t = lr * math.sqrt(1-beta2**(t+1)) / (1-beta1**(t+1))
                p.data -= lr_t*moment1/(torch.sqrt(moment2) + eps)
                p.data -= lr*lamd*p.data
                state["step"] = t + 1 # Increment iteration number.
                state["moment1"] = moment1
                state["moment2"] = moment2
        return loss