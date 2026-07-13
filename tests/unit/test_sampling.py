import pytest
import torch

from my_tiny_llm.inference import make_sampler


def test_greedy_sampler_selects_largest_logit() -> None:
    logits = torch.tensor([[1.0, -2.0, 4.0, 3.0], [5.0, 2.0, 1.0, 0.0]])

    actual = make_sampler()(logits)

    torch.testing.assert_close(actual, torch.tensor([2, 0]))


def test_top_k_one_is_deterministic() -> None:
    logits = torch.tensor([[1.0, -2.0, 4.0, 3.0]])
    sampler = make_sampler(temperature=0.7, top_k=1)
    assert sampler(logits).item() == 2


def test_top_k_never_samples_a_filtered_token() -> None:
    torch.manual_seed(20)
    logits = torch.tensor([[10.0, 9.0, 8.0, -100.0]]).expand(100, -1)

    sampled = make_sampler(temperature=0.7, top_k=2)(logits)

    assert set(sampled.tolist()) <= {0, 1}


def test_top_p_keeps_the_first_token_over_the_threshold() -> None:
    torch.manual_seed(21)
    logits = torch.tensor([[10.0, 9.0, 8.0]]).expand(20, -1)

    sampled = make_sampler(temperature=1.0, top_p=0.6)(logits)

    torch.testing.assert_close(sampled, torch.zeros(20, dtype=torch.long))


def test_negative_temperature_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_sampler(temperature=-0.1)