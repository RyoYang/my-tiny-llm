import torch
from transformers import Qwen3Config as TransformersQwen3Config
from transformers import Qwen3ForCausalLM as TransformersQwen3ForCausalLM

from my_tiny_llm.models import Qwen3Config, Qwen3ForCausalLM


def test_qwen3_logits_match_transformers() -> None:
    torch.manual_seed(4)
    hf_config = TransformersQwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        tie_word_embeddings=False,
        attention_bias=False,
    )
    reference = TransformersQwen3ForCausalLM(hf_config).eval()
    model = Qwen3ForCausalLM(Qwen3Config.from_hf_config(hf_config)).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    input_ids = torch.tensor([[1, 7, 3, 22, 5], [4, 2, 9, 6, 11]])

    with torch.inference_mode():
        expected = reference(input_ids).logits
        actual = model(input_ids)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_kv_cache_matches_full_sequence() -> None:
    torch.manual_seed(5)
    config = Qwen3Config(
        vocab_size=41,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
    )
    model = Qwen3ForCausalLM(config).eval()
    input_ids = torch.tensor([[3, 8, 2, 7, 4, 1]])

    with torch.inference_mode():
        full_logits = model(input_ids)
        cached_logits = []
        cache = None
        for position in range(input_ids.shape[1]):
            logits, cache = model(
                input_ids[:, position : position + 1],
                past_key_values=cache,
                use_cache=True,
            )
            cached_logits.append(logits)

    torch.testing.assert_close(
        torch.cat(cached_logits, dim=1), full_logits, rtol=2e-5, atol=2e-5
    )