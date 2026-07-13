import pytest
import torch
import torch.nn.functional as functional

from my_tiny_llm.core import (
    SimpleMultiHeadAttention,
    causal_mask,
    scaled_dot_product_attention_grouped,
    scaled_dot_product_attention_simple,
)


def test_simple_attention_matches_torch_with_additive_mask() -> None:
    torch.manual_seed(2)
    query = torch.randn(2, 3, 5, 8)
    key = torch.randn(2, 3, 7, 8)
    value = torch.randn(2, 3, 7, 8)
    mask = torch.randn(2, 3, 5, 7)

    actual = scaled_dot_product_attention_simple(query, key, value, mask=mask)
    expected = functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask
    )

    assert actual.shape == (2, 3, 5, 8)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_simple_attention_honors_custom_scale() -> None:
    torch.manual_seed(3)
    query = torch.randn(2, 4, 6)
    key = torch.randn(2, 5, 6)
    value = torch.randn(2, 5, 7)

    actual = scaled_dot_product_attention_simple(
        query, key, value, scale=0.25
    )
    expected = functional.scaled_dot_product_attention(
        query, key, value, scale=0.25
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_causal_mask_has_expected_square_values() -> None:
    actual = causal_mask(4, 4, torch.float32)
    expected = torch.tensor(
        [
            [0.0, -torch.inf, -torch.inf, -torch.inf],
            [0.0, 0.0, -torch.inf, -torch.inf],
            [0.0, 0.0, 0.0, -torch.inf],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_causal_mask_aligns_rectangular_queries_to_latest_keys() -> None:
    actual = causal_mask(2, 5, torch.float64)
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, -torch.inf],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    assert actual.shape == (2, 5)
    torch.testing.assert_close(actual, expected)


def test_causal_attention_with_equal_heads_matches_torch() -> None:
    torch.manual_seed(4)
    query = torch.randn(2, 6, 4, 8)
    key = torch.randn(2, 6, 4, 8)
    value = torch.randn(2, 6, 4, 8)

    actual = scaled_dot_product_attention_grouped(
        query, key, value, mask="causal"
    )
    reference_mask = torch.triu(
        torch.full((4, 4), -torch.inf), diagonal=1
    )
    expected = functional.scaled_dot_product_attention(
        query, key, value, attn_mask=reference_mask
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_grouped_attention_matches_explicitly_repeated_kv_heads() -> None:
    torch.manual_seed(5)
    query = torch.randn(2, 6, 4, 8)
    key = torch.randn(2, 2, 4, 8)
    value = torch.randn(2, 2, 4, 8)

    actual = scaled_dot_product_attention_grouped(query, key, value)
    expected = functional.scaled_dot_product_attention(
        query,
        key.repeat_interleave(3, dim=1),
        value.repeat_interleave(3, dim=1),
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_grouped_attention_rejects_incompatible_heads() -> None:
    query = torch.randn(1, 3, 2, 4)
    key = torch.randn(1, 2, 2, 4)
    value = torch.randn_like(key)

    with pytest.raises(ValueError, match="divisible"):
        scaled_dot_product_attention_grouped(query, key, value)


def test_grouped_attention_rejects_unknown_string_mask() -> None:
    query = torch.randn(1, 2, 2, 4)

    with pytest.raises(ValueError, match="unsupported"):
        scaled_dot_product_attention_grouped(
            query, query, query, mask="not-a-mask"
        )


def test_multi_head_attention_matches_torch() -> None:
    torch.manual_seed(6)
    batch_size, sequence_length, hidden_size, num_heads = 2, 5, 12, 3
    query = torch.randn(batch_size, sequence_length, hidden_size)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    weights = [torch.randn(hidden_size, hidden_size) for _ in range(4)]
    mask = torch.randn(sequence_length, sequence_length)
    attention = SimpleMultiHeadAttention(hidden_size, num_heads, *weights)

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
        expected.transpose(1, 2).reshape(
            batch_size, sequence_length, hidden_size
        ),
        weights[3],
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_multi_head_attention_rejects_non_divisible_hidden_size() -> None:
    weights = [torch.randn(10, 10) for _ in range(4)]

    with pytest.raises(ValueError, match="divisible"):
        SimpleMultiHeadAttention(10, 3, *weights)
