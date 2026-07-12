import torch


def apply_rope(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    theta: float,
) -> torch.Tensor:
    """Exercise 11: apply rotary position embeddings to Q or K."""
    raise NotImplementedError("TODO: implement RoPE")
