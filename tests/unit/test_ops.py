import pytest
import torch
import torch.nn.functional as functional

from my_tiny_llm.core import linear, silu, softmax


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_softmax_matches_torch_along_requested_dimension(dim: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5)

    actual = softmax(x, dim=dim)

    torch.testing.assert_close(actual, torch.softmax(x, dim=dim))
    torch.testing.assert_close(
        actual.sum(dim=dim), torch.ones_like(actual.sum(dim=dim))
    )


def test_softmax_is_stable_for_large_values() -> None:
    x = torch.tensor([[10_000.0, 10_001.0, 9_999.0]])

    actual = softmax(x, dim=-1)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.softmax(x, dim=-1))


@pytest.mark.parametrize("use_bias", [False, True])
def test_linear_matches_torch_for_batched_input(use_bias: bool) -> None:
    torch.manual_seed(1)
    x = torch.randn(2, 3, 5)
    weight = torch.randn(7, 5)
    bias = torch.randn(7) if use_bias else None

    actual = linear(x, weight, bias)

    assert actual.shape == (2, 3, 7)
    torch.testing.assert_close(actual, functional.linear(x, weight, bias))


def test_silu_matches_torch_for_extreme_values() -> None:
    x = torch.tensor([-1_000.0, -3.0, 0.0, 3.0, 1_000.0])

    actual = silu(x)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, functional.silu(x))
