from types import SimpleNamespace

import pytest

from my_tiny_llm.models import Qwen3Config


def _hf_config(**overrides):
    values = {
        "vocab_size": 101,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "rope_parameters": {"rope_theta": 500_000.0},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_config_converts_current_transformers_schema() -> None:
    config = Qwen3Config.from_hf_config(_hf_config())

    assert config.vocab_size == 101
    assert config.head_dim == 8
    assert config.rope_theta == 500_000.0
    assert config.attention_bias is False


def test_config_prefers_legacy_rope_theta_when_present() -> None:
    config = Qwen3Config.from_hf_config(
        _hf_config(rope_theta=250_000.0, head_dim=16, attention_bias=True)
    )

    assert config.head_dim == 16
    assert config.rope_theta == 250_000.0
    assert config.attention_bias is True


def test_config_is_immutable() -> None:
    config = Qwen3Config.from_hf_config(_hf_config())

    with pytest.raises(AttributeError):
        config.hidden_size = 64