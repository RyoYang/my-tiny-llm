import torch

from my_tiny_llm.layers import RMSNorm
from my_tiny_llm.models import Qwen3Config, Qwen3ForCausalLM
from my_tiny_llm.models.qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3Model,
)


def _config(**overrides) -> Qwen3Config:
    values = {
        "vocab_size": 41,
        "hidden_size": 24,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 3,
        "num_key_value_heads": 3,
        "head_dim": 8,
        "max_position_embeddings": 32,
    }
    values.update(overrides)
    return Qwen3Config(**values)


def test_qwen3_attention_initializes_expected_projections() -> None:
    config = _config()
    attention = Qwen3Attention(config)

    assert attention.q_proj.weight.shape == (24, 24)
    assert attention.k_proj.weight.shape == (24, 24)
    assert attention.v_proj.weight.shape == (24, 24)
    assert attention.o_proj.weight.shape == (24, 24)
    assert isinstance(attention.q_norm, RMSNorm)
    assert isinstance(attention.k_norm, RMSNorm)


def test_qwen3_attention_returns_output_and_growing_cache() -> None:
    torch.manual_seed(30)
    attention = Qwen3Attention(_config()).eval()
    x = torch.randn(2, 4, 24)
    positions = torch.arange(4).expand(2, -1)

    output, cache = attention(x, positions)
    next_output, next_cache = attention(
        torch.randn(2, 1, 24),
        torch.full((2, 1), 4),
        past_key_value=cache,
    )

    assert output.shape == x.shape
    assert cache[0].shape == (2, 3, 4, 8)
    assert cache[1].shape == (2, 3, 4, 8)
    assert next_output.shape == (2, 1, 24)
    assert next_cache[0].shape[-2] == 5
    assert next_cache[1].shape[-2] == 5


def test_qwen3_mlp_matches_gated_silu_definition() -> None:
    torch.manual_seed(31)
    mlp = Qwen3MLP(_config())
    x = torch.randn(2, 3, 24)

    actual = mlp(x)
    expected = mlp.down_proj(
        torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x)
    )

    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected)


def test_decoder_layer_returns_residual_output_and_cache() -> None:
    torch.manual_seed(32)
    layer = Qwen3DecoderLayer(_config()).eval()
    x = torch.randn(2, 4, 24)
    positions = torch.arange(4).expand(2, -1)

    output, cache = layer(x, positions, mask="causal")

    assert output.shape == x.shape
    assert len(cache) == 2
    assert cache[0].shape[-2] == 4


def test_qwen3_model_builds_requested_layers_and_returns_each_cache() -> None:
    torch.manual_seed(33)
    config = _config(num_hidden_layers=3)
    model = Qwen3Model(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    positions = torch.arange(4).unsqueeze(0)

    output, cache = model(input_ids, positions, mask="causal")

    assert model.embed_tokens.weight.shape == (41, 24)
    assert len(model.layers) == 3
    assert output.shape == (1, 4, 24)
    assert len(cache) == 3


def test_causal_lm_ties_embedding_weights_when_configured() -> None:
    model = Qwen3ForCausalLM(_config(tie_word_embeddings=True))

    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_causal_lm_returns_vocab_logits() -> None:
    torch.manual_seed(34)
    model = Qwen3ForCausalLM(_config()).eval()

    logits = model(torch.tensor([[1, 2, 3]]))

    assert logits.shape == (1, 3, 41)