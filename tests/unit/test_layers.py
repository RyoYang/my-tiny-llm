import pytest
import torch
from torch import nn

from my_tiny_llm.layers import RMSNorm, apply_rope


def test_rmsnorm_initializes_registered_unit_weight() -> None:
    norm = RMSNorm(dim=8, eps=1e-6)

    assert isinstance(norm.weight, nn.Parameter)
    assert norm.weight.shape == (8,)
    assert norm.eps == 1e-6
    torch.testing.assert_close(norm.weight, torch.ones(8))
    assert "weight" in norm.state_dict()


def test_rmsnorm_matches_torch() -> None:
    torch.manual_seed(10)
    x = torch.randn(2, 5, 8)
    weight = torch.randn(8)
    actual_norm = RMSNorm(dim=8, eps=1e-6)
    expected_norm = nn.RMSNorm(8, eps=1e-6)
    with torch.no_grad():
        actual_norm.weight.copy_(weight)
        expected_norm.weight.copy_(weight)

    actual = actual_norm(x)
    expected = expected_norm(x)

    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_rmsnorm_preserves_low_precision_dtype() -> None:
    torch.manual_seed(11)
    x = torch.randn(2, 3, 8).to(torch.bfloat16)
    norm = RMSNorm(dim=8, eps=1e-6)

    actual = norm(x)

    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()


def _reference_rope(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    theta: float,
) -> torch.Tensor:
    head_dim = x.shape[-1]
    inv_freq = theta ** (
        -torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32)
        / head_dim
    )
    frequencies = position_ids.float().unsqueeze(-1) * inv_freq
    cos = frequencies.cos().unsqueeze(2)
    sin = frequencies.sin().unsqueeze(2)
    first_half, second_half = x.float().chunk(2, dim=-1)
    rotated = torch.cat(
        (
            first_half * cos - second_half * sin,
            second_half * cos + first_half * sin,
        ),
        dim=-1,
    )
    return rotated.to(x.dtype)


def test_rope_leaves_position_zero_unchanged() -> None:
    torch.manual_seed(12)
    x = torch.randn(2, 1, 3, 8)
    position_ids = torch.zeros((2, 1), dtype=torch.long)

    actual = apply_rope(x, position_ids, theta=10_000.0)

    torch.testing.assert_close(actual, x)


def test_rope_matches_reference_for_batched_positions() -> None:
    torch.manual_seed(13)
    x = torch.randn(2, 4, 3, 8)
    position_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])

    actual = apply_rope(x, position_ids, theta=1_000_000.0)
    expected = _reference_rope(x, position_ids, theta=1_000_000.0)

    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_rope_rejects_odd_head_dimension() -> None:
    x = torch.randn(1, 2, 1, 7)
    position_ids = torch.tensor([[0, 1]])

    with pytest.raises(ValueError, match="even"):
        apply_rope(x, position_ids, theta=10_000.0)
