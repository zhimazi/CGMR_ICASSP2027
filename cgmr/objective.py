"""Amortization utilities for fitting projected CGMR targets.

For an old-language row, the paper minimizes KL(q* || pi_theta) on the fixed
candidate support. Since q* is stop-gradient, the q* entropy is constant and
this is equivalent to target cross-entropy against the student's restricted
posterior. The score-gradient helper exposes the exact softmax(s) - q* identity
used by the memory-efficient reference runner.
"""

from __future__ import annotations

import torch


def _validate_target(scores: torch.Tensor, target_q: torch.Tensor) -> None:
    if scores.ndim != 1 or target_q.ndim != 1 or scores.shape != target_q.shape:
        raise ValueError("scores and target_q must be same-shaped 1D tensors")
    if torch.any(target_q < 0):
        raise ValueError("target_q must be non-negative")
    if not torch.isclose(target_q.sum(), target_q.new_tensor(1.0), atol=1e-6):
        raise ValueError("target_q must sum to one")


def restricted_target_cross_entropy(
    sequence_log_scores: torch.Tensor,
    target_q: torch.Tensor,
    *,
    normalizer: float = 1.0,
) -> torch.Tensor:
    """Old-row CGMR fitting loss on a fixed candidate support (Eq. 8)."""

    _validate_target(sequence_log_scores, target_q)
    if normalizer <= 0:
        raise ValueError("normalizer must be positive")
    log_pi = torch.log_softmax(sequence_log_scores, dim=0)
    return -(target_q.detach() * log_pi).sum() / float(normalizer)


def restricted_score_gradient(
    sequence_log_scores: torch.Tensor,
    target_q: torch.Tensor,
    *,
    normalizer: float = 1.0,
) -> torch.Tensor:
    """Exact derivative of target cross-entropy with respect to sequence scores."""

    _validate_target(sequence_log_scores, target_q)
    if normalizer <= 0:
        raise ValueError("normalizer must be positive")
    return (torch.softmax(sequence_log_scores, dim=0) - target_q.detach()) / float(
        normalizer
    )


def mixed_repair_loss(
    old_loss: torch.Tensor,
    current_loss: torch.Tensor,
    *,
    old_weight: float = 0.5,
) -> torch.Tensor:
    """Combine projected old targets with current-language teacher-forced CE."""

    if not 0.0 <= old_weight <= 1.0:
        raise ValueError("old_weight must lie in [0, 1]")
    return old_weight * old_loss + (1.0 - old_weight) * current_loss
