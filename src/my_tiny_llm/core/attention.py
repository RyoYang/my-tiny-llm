import torch


def scaled_dot_product_attention_simple(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exercise 4: implement scaled dot-product attention."""
    raise NotImplementedError("TODO: implement simple attention")


def causal_mask(
    query_length: int,
    key_length: int,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Exercise 5: build an additive causal attention mask."""
    raise NotImplementedError("TODO: implement a causal mask")


def scaled_dot_product_attention_grouped(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | str | None = None,
) -> torch.Tensor:
    """Exercise 6: extend attention to grouped query and key/value heads."""
    raise NotImplementedError("TODO: implement grouped-query attention")


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        query_weight: torch.Tensor,
        key_weight: torch.Tensor,
        value_weight: torch.Tensor,
        output_weight: torch.Tensor,
    ) -> None:
        """Exercise 7: validate and store multi-head projection weights."""
        raise NotImplementedError("TODO: initialize multi-head attention")

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Exercise 8: project, attend, merge heads, and project the output."""
        raise NotImplementedError("TODO: run multi-head attention")
