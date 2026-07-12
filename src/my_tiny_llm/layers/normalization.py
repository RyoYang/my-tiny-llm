import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        """Exercise 9: initialize the learnable RMSNorm parameters."""
        raise NotImplementedError("TODO: initialize RMSNorm")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Exercise 10: normalize in float32 and restore the input dtype."""
        raise NotImplementedError("TODO: implement RMSNorm")
