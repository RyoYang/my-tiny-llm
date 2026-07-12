import torch

from my_tiny_llm.inference import make_sampler


def test_greedy_sampler_selects_largest_logit() -> None:
    logits = torch.tensor([[1.0, -2.0, 4.0, 3.0]])
    assert make_sampler()(logits).item() == 2


def test_top_k_one_is_deterministic() -> None:
    logits = torch.tensor([[1.0, -2.0, 4.0, 3.0]])
    sampler = make_sampler(temperature=0.7, top_k=1)
    assert sampler(logits).item() == 2


def test_negative_temperature_is_rejected() -> None:
    try:
        make_sampler(temperature=-0.1)
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative temperature should fail")