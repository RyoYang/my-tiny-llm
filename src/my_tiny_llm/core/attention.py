import torch
from .ops import linear

def scaled_dot_product_attention_simple(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exercise 4: implement scaled dot-product attention."""
    dim_k = query.size(-1)
    scaled_factor = scale if scale is not None else 1.0 / (dim_k ** 0.5)
    scaled_scores = torch.matmul(query, key.transpose(-2, -1)) * scaled_factor

    if mask is not None:
        scaled_scores = scaled_scores + mask
    attention_weights = torch.softmax(scaled_scores, dim=-1)
    return torch.matmul(attention_weights, value)


def causal_mask(
    query_length: int,
    key_length: int,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Exercise 5: build an additive causal attention mask."""
    mask = torch.triu(
        torch.ones(query_length, key_length, dtype=dtype, device=device),
        diagonal=key_length - query_length + 1,
    )
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


def scaled_dot_product_attention_grouped(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | str | None = None,
) -> torch.Tensor:
    """Exercise 6: extend attention to grouped query and key/value heads."""
    query_heads = query.size(-3)
    kv_heads = key.size(-3)
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if query_heads % kv_heads != 0:
        raise ValueError("query heads must be divisible by key/value heads")

    repeats = query_heads // kv_heads
    key = key.repeat_interleave(repeats, dim=-3)
    value = value.repeat_interleave(repeats, dim=-3)

    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError(f"unsupported attention mask: {mask}")
        mask = causal_mask(
            query.size(-2), key.size(-2), query.dtype, query.device
        )
    return scaled_dot_product_attention_simple(
        query, key, value, scale=scale, mask=mask
    )


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
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.query_weight = query_weight
        self.key_weight = key_weight
        self.value_weight = value_weight
        self.output_weight = output_weight
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Exercise 8: project, attend, merge heads, and project the output."""
        batch_size_q, sequence_length_q, _ = query.shape
        batch_size_k, sequence_length_k, _ = key.shape
        batch_size_v, sequence_length_v, _ = value.shape
        if batch_size_q != batch_size_k or batch_size_q != batch_size_v:
            raise ValueError("Batch sizes of query, key, and value must match")
        if sequence_length_q != sequence_length_k or sequence_length_q != sequence_length_v:
            raise ValueError("Sequence lengths of query, key, and value must match")
        query = linear(query, self.query_weight).reshape(
            batch_size_q, sequence_length_q, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = linear(key, self.key_weight).reshape(
            batch_size_k, sequence_length_k, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = linear(value, self.value_weight).reshape(
            batch_size_v, sequence_length_v, self.num_heads, self.head_dim
        ).transpose(1, 2)
        attn_output = scaled_dot_product_attention_simple(
            query, key, value, scale=self.scale, mask=mask
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size_q, sequence_length_q, self.hidden_size
        )
        return linear(attn_output, self.output_weight)


