import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Exercise 1: implement numerically stable softmax with PyTorch tensors."""
    '''
    1. Subtract the maximum value along the specified dimension to improve numerical stability.
    2. Exponentiate the shifted values.
    3. Normalize by dividing by the sum along the specified dimension.
    '''

    x_ex = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    return x_ex / x_ex.sum(dim=dim, keepdim=True)


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exercise 2: implement a linear projection with an optional bias."""
    result = torch.matmul(x, weight.T)
    if bias is not None:
        result = result + bias
    return result


def silu(x: torch.Tensor) -> torch.Tensor:
    """Exercise 3: implement the SiLU activation."""
    # SiLU(x) = x * sigmoid(x)
    sigmoid = 1 / (1 + torch.exp(-x))
    return x * sigmoid
