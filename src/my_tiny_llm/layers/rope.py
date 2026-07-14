import torch


def apply_rope(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    theta: float,
) -> torch.Tensor:
    """Exercise 11: apply rotary position embeddings to Q or K."""
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("head dimension must be even for RoPE")
    first_half, second_half = x.float().chunk(2, dim=-1)
    dimension_indices = torch.arange(
        0,
        head_dim,
        2,
        device=x.device,
        dtype=torch.float32,
    )
    inv_freq = theta ** (-dimension_indices / head_dim)
    frequencies = position_ids.float().unsqueeze(-1) * inv_freq
    cos = frequencies.cos().unsqueeze(2)
    sin = frequencies.sin().unsqueeze(2)
    rotated = torch.cat(
        (
            first_half * cos - second_half * sin,
            second_half * cos + first_half * sin,
        ),
        dim=-1,
    )
    return rotated.to(x.dtype)

