import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Exercise 1: implement numerically stable softmax with PyTorch tensors."""
    raise NotImplementedError("TODO: implement softmax")


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exercise 2: implement a linear projection with an optional bias."""
    raise NotImplementedError("TODO: implement linear")


def silu(x: torch.Tensor) -> torch.Tensor:
    """Exercise 3: implement the SiLU activation."""
    raise NotImplementedError("TODO: implement silu")
