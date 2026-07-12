import torch

from ..models import Qwen3ForCausalLM


@torch.inference_mode()
def generate(
    model: Qwen3ForCausalLM,
    tokenizer: object,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
) -> str:
    """Exercise 24: prefill once, decode with a cache, and stop at EOS."""
    raise NotImplementedError("TODO: implement autoregressive generation")
