import json
from pathlib import Path

import pytest

from my_tiny_llm.checkpoints.loader import _checkpoint_files, resolve_model_name


@pytest.mark.parametrize(
    ("shortcut", "expected"),
    [
        ("qwen3-0.6b", "Qwen/Qwen3-0.6B"),
        ("QWEN3-1.7B", "Qwen/Qwen3-1.7B"),
        ("qwen3-4b", "Qwen/Qwen3-4B"),
        ("qwen3-8b", "Qwen/Qwen3-8B"),
        ("local/model", "local/model"),
    ],
)
def test_resolve_model_name(shortcut: str, expected: str) -> None:
    assert resolve_model_name(shortcut) == expected


def test_checkpoint_files_finds_single_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()

    assert _checkpoint_files(tmp_path) == [checkpoint]


def test_checkpoint_files_reads_unique_sorted_shards(tmp_path: Path) -> None:
    index = {
        "weight_map": {
            "model.layers.1.weight": "model-00002-of-00002.safetensors",
            "model.layers.0.weight": "model-00001-of-00002.safetensors",
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    assert _checkpoint_files(tmp_path) == [
        tmp_path / "model-00001-of-00002.safetensors",
        tmp_path / "model-00002-of-00002.safetensors",
    ]


def test_checkpoint_files_rejects_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="SafeTensors"):
        _checkpoint_files(tmp_path)
