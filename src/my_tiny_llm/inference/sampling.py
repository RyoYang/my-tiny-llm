from collections.abc import Callable

import torch


def make_sampler(
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Exercise 23: build greedy, temperature, top-k, and top-p sampling."""
    raise NotImplementedError("TODO: implement token sampling")
