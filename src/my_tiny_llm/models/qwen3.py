import torch
from torch import nn

from .config import Qwen3Config

LayerCache = tuple[torch.Tensor, torch.Tensor]
KVCache = list[LayerCache]


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        """Exercise 12: create Qwen3 Q/K/V/O projections and Q/K norms."""
        raise NotImplementedError("TODO: initialize Qwen3 attention")

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        mask: torch.Tensor | str | None = "causal",
        past_key_value: LayerCache | None = None,
    ) -> tuple[torch.Tensor, LayerCache]:
        """Exercise 13: project, normalize, rotate, cache, and attend."""
        raise NotImplementedError("TODO: implement Qwen3 attention")


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        """Exercise 14: initialize the gated Qwen3 MLP projections."""
        raise NotImplementedError("TODO: initialize Qwen3 MLP")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Exercise 15: implement the gated SiLU MLP."""
        raise NotImplementedError("TODO: implement Qwen3 MLP")


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        """Exercise 16: compose attention, MLP, and normalization modules."""
        raise NotImplementedError("TODO: initialize a decoder layer")

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        mask: torch.Tensor | str | None,
        past_key_value: LayerCache | None = None,
    ) -> tuple[torch.Tensor, LayerCache]:
        """Exercise 17: implement pre-norm residual decoder flow."""
        raise NotImplementedError("TODO: run a decoder layer")


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        """Exercise 18: initialize embeddings, decoder layers, and final norm."""
        raise NotImplementedError("TODO: initialize the Qwen3 model")

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        mask: torch.Tensor | str | None,
        past_key_values: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Exercise 19: run all decoder layers and collect their caches."""
        raise NotImplementedError("TODO: run the Qwen3 model")


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        """Exercise 20: initialize the base model and language-model head."""
        raise NotImplementedError("TODO: initialize Qwen3ForCausalLM")

    def tie_weights(self) -> None:
        """Exercise 21: share the embedding and language-model weights."""
        raise NotImplementedError("TODO: tie embedding weights")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        """Exercise 22: build positions/masks, run the model, and return logits."""
        raise NotImplementedError("TODO: implement causal language-model forward")
