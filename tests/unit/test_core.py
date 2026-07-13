import torch
import torch.nn.functional as functional

from my_tiny_llm import (
    SimpleMultiHeadAttention,
    causal_mask,
    linear,
    scaled_dot_product_attention_grouped,
    scaled_dot_product_attention_simple,
    silu,
    softmax,
)


def test_tensor_primitives_match_torch() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5)
    weight = torch.randn(7, 5)
    bias = torch.randn(7)
    torch.testing.assert_close(softmax(x, dim=-1), torch.softmax(x, dim=-1))
    torch.testing.assert_close(linear(x, weight, bias), functional.linear(x, weight, bias))
    torch.testing.assert_close(silu(x), functional.silu(x))


def test_simple_attention_matches_torch() -> None:
    torch.manual_seed(1)
    query = torch.randn(2, 3, 5, 8)
    key = torch.randn(2, 3, 7, 8)
    value = torch.randn(2, 3, 7, 8)
    mask = torch.randn(2, 3, 5, 7)
    actual = scaled_dot_product_attention_simple(query, key, value, mask=mask)
    expected = functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_causal_attention_with_equal_heads_matches_torch() -> None:
    torch.manual_seed(2)
    query = torch.randn(2, 6, 4, 8)
    key = torch.randn(2, 6, 4, 8)
    value = torch.randn(2, 6, 4, 8)
    actual = scaled_dot_product_attention_grouped(
        query, key, value, mask="causal"
    )
    expected = functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask(4, 4, query.dtype),
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_simple_multi_head_attention_matches_torch() -> None:
    torch.manual_seed(3)
    batch_size, sequence_length, hidden_size, num_heads = 2, 5, 12, 3
    query = torch.randn(batch_size, sequence_length, hidden_size)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    weights = [torch.randn(hidden_size, hidden_size) for _ in range(4)]
    mask = torch.randn(sequence_length, sequence_length)
    attention = SimpleMultiHeadAttention(
        hidden_size, num_heads, *weights
    )
    actual = attention(query, key, value, mask)

    head_dim = hidden_size // num_heads
    projected = [
        functional.linear(tensor, weight)
        .reshape(batch_size, sequence_length, num_heads, head_dim)
        .transpose(1, 2)
        for tensor, weight in zip((query, key, value), weights[:3])
    ]
    expected = functional.scaled_dot_product_attention(
        *projected, attn_mask=mask, scale=head_dim**-0.5
    )
    expected = functional.linear(
        expected.transpose(1, 2).reshape(batch_size, sequence_length, hidden_size),
        weights[3],
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)