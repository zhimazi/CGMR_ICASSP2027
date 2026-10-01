import torch

from cgmr.objective import (
    mixed_repair_loss,
    restricted_score_gradient,
    restricted_target_cross_entropy,
)


def test_score_gradient_matches_autograd():
    scores = torch.tensor([0.2, -0.1, -0.7], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.1, 0.7, 0.2], dtype=torch.float64)
    loss = restricted_target_cross_entropy(scores, q, normalizer=2.5)
    loss.backward()
    expected = restricted_score_gradient(scores.detach(), q, normalizer=2.5)
    assert torch.allclose(scores.grad, expected)


def test_mixed_repair_loss():
    old = torch.tensor(2.0)
    current = torch.tensor(4.0)
    assert torch.isclose(mixed_repair_loss(old, current), torch.tensor(3.0))
