from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen3Config as TransformersQwen3Config

import my_tiny_llm.checkpoints.loader as loading
from my_tiny_llm.models import Qwen3Config, Qwen3ForCausalLM


def test_load_model_from_local_safetensors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    hf_config = TransformersQwen3Config(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    hf_config.save_pretrained(tmp_path)
    source = Qwen3ForCausalLM(Qwen3Config.from_hf_config(hf_config))
    save_file(source.state_dict(), tmp_path / "model.safetensors")
    tokenizer = object()
    monkeypatch.setattr(loading, "snapshot_download", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(
        loading.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )

    restored, restored_tokenizer = loading.load_model(
        "local-test-model",
        device="cpu",
        dtype=torch.float32,
        local_files_only=True,
    )

    assert restored_tokenizer is tokenizer
    assert restored.training is False
    for name, expected in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected)