from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

from ..models import Qwen3Config, Qwen3ForCausalLM


MODEL_SHORTCUTS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
}


def resolve_model_name(model_name: str) -> str:
    return MODEL_SHORTCUTS.get(model_name.lower(), model_name)


def _checkpoint_files(model_path: Path) -> list[Path]:
    """Exercise 25: find single-file or sharded SafeTensors checkpoints."""
    raise NotImplementedError("TODO: discover checkpoint files")


def load_model(
    model_name: str,
    device: torch.device | str,
    dtype: torch.dtype,
    local_files_only: bool = False,
) -> tuple[Qwen3ForCausalLM, object]:
    """Exercise 26: download, construct, populate, and place a Qwen3 model."""
    _ = (snapshot_download, load_file, AutoConfig, AutoTokenizer, Qwen3Config)
    raise NotImplementedError("TODO: load a SafeTensors Qwen3 checkpoint")
