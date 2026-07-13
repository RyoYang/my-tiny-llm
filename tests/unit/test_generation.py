from typing import Any

import pytest
import torch
from torch import nn

from my_tiny_llm.inference import generate


class ScriptedModel(nn.Module):
    def __init__(self, next_tokens: list[int], vocab_size: int = 8) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.next_tokens = next_tokens
        self.vocab_size = vocab_size
        self.calls: list[dict[str, Any]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = False,
    ):
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        token = self.next_tokens[len(self.calls) - 1]
        logits = torch.full(
            (*input_ids.shape, self.vocab_size),
            -100.0,
            device=input_ids.device,
        )
        logits[:, -1, token] = 100.0
        return logits, {"step": len(self.calls)}


class FakeTokenizer:
    eos_token_id = 3

    def __init__(self) -> None:
        self.decoded_tokens: list[int] | None = None

    def __call__(
        self,
        prompt: str,
        return_tensors: str,
        add_special_tokens: bool,
    ) -> dict[str, torch.Tensor]:
        assert prompt == "hello"
        assert return_tensors == "pt"
        assert add_special_tokens is False
        return {"input_ids": torch.tensor([[1, 2]])}

    def decode(self, tokens: list[int], skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        self.decoded_tokens = tokens
        return "decoded"


def test_generate_prefills_once_then_decodes_with_cache_until_eos() -> None:
    model = ScriptedModel(next_tokens=[4, 3])
    tokenizer = FakeTokenizer()

    text = generate(model, tokenizer, "hello", max_new_tokens=5)

    assert text == "decoded"
    assert tokenizer.decoded_tokens == [4]
    assert len(model.calls) == 2
    torch.testing.assert_close(model.calls[0]["input_ids"], torch.tensor([[1, 2]]))
    torch.testing.assert_close(model.calls[1]["input_ids"], torch.tensor([[4]]))
    assert model.calls[0]["past_key_values"] is None
    assert model.calls[1]["past_key_values"] == {"step": 1}
    assert all(call["use_cache"] for call in model.calls)
    assert not any(call["grad_enabled"] for call in model.calls)


def test_generate_stops_at_max_new_tokens_without_eos() -> None:
    model = ScriptedModel(next_tokens=[4, 5, 6])
    tokenizer = FakeTokenizer()

    generate(model, tokenizer, "hello", max_new_tokens=3)

    assert tokenizer.decoded_tokens == [4, 5, 6]
    assert len(model.calls) == 3


def test_generate_rejects_non_positive_token_limit() -> None:
    with pytest.raises(ValueError, match="positive"):
        generate(ScriptedModel([3]), FakeTokenizer(), "hello", max_new_tokens=0)
