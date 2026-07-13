from argparse import Namespace

import pytest
import torch

import my_tiny_llm.cli as cli


def _args(**overrides) -> Namespace:
    values = {
        "model": "qwen3-0.6b",
        "prompt": "hello",
        "device": "auto",
        "max_new_tokens": 12,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 20,
        "enable_thinking": True,
        "local_files_only": True,
    }
    values.update(overrides)
    return Namespace(**values)


class FakeTokenizer:
    def __init__(self) -> None:
        self.template_call = None

    def apply_chat_template(self, messages, **kwargs) -> str:
        self.template_call = (messages, kwargs)
        return "formatted prompt"


def test_main_selects_cpu_and_forwards_generation_options(
    monkeypatch,
    capsys,
) -> None:
    tokenizer = FakeTokenizer()
    model = object()
    load_call = {}
    generate_call = {}
    monkeypatch.setattr(cli, "_parse_args", lambda: _args())
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)

    def fake_load_model(model_name, **kwargs):
        load_call.update(model_name=model_name, **kwargs)
        return model, tokenizer

    def fake_generate(received_model, received_tokenizer, prompt, **kwargs):
        generate_call.update(
            model=received_model,
            tokenizer=received_tokenizer,
            prompt=prompt,
            **kwargs,
        )
        return "generated text"

    monkeypatch.setattr(cli, "load_model", fake_load_model)
    monkeypatch.setattr(cli, "generate", fake_generate)

    cli.main()

    assert load_call == {
        "model_name": "qwen3-0.6b",
        "device": "cpu",
        "dtype": torch.float32,
        "local_files_only": True,
    }
    assert tokenizer.template_call == (
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"},
        ],
        {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": True,
        },
    )
    assert generate_call == {
        "model": model,
        "tokenizer": tokenizer,
        "prompt": "formatted prompt",
        "max_new_tokens": 12,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 20,
    }
    assert capsys.readouterr().out == "generated text\n"


def test_main_rejects_requested_cuda_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_parse_args", lambda: _args(device="cuda"))
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        cli.main()