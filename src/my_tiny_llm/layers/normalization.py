import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        """Exercise 9: initialize the learnable RMSNorm parameters."""
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Exercise 10: normalize in float32 and restore the input dtype."""
        orig_dtype = x.dtype
        
        x = x.to(torch.float32)
        norm = (x.norm(dim=-1, keepdim=True)) * (self.dim**-0.5)
        x = x / (norm + self.eps)
        return (x * self.weight.to(x.dtype)).to(orig_dtype)
