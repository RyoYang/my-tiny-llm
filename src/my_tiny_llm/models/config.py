from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = False
    attention_bias: bool = False

    @classmethod
    def from_hf_config(cls, config: Any) -> "Qwen3Config":
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        return cls(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=getattr(
                config,
                "head_dim",
                config.hidden_size // config.num_attention_heads,
            ),
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(
                config, "rope_theta", rope_parameters.get("rope_theta", 1_000_000.0)
            ),
            tie_word_embeddings=config.tie_word_embeddings,
            attention_bias=getattr(config, "attention_bias", False),
        )
