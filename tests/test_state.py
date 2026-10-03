import pytest

from cgmr.states import classify_state, first_passage, state_path


def test_state_n_a_d():
    assert classify_state(2, 2, 1) == "N"
    assert classify_state(2, 4, 2) == "A"
    assert classify_state(2, 4, 3) == "D"


def test_first_passage_ranking_first():
    observations = [
        (0, 2, 1),
        (16, 4, 2),
        (32, 5, 3),
    ]
    result = first_passage(2, observations)
    assert result.tau_rec == 16
    assert result.tau_loss == 32
    assert result.onset == "ranking_first"
    assert [state for _, state in state_path(2, observations)] == ["N", "A", "D"]


def test_first_passage_candidate_loss_at_onset():
    result = first_passage(2, [(0, 1, 1), (16, 4, 3)])
    assert result.tau_rec == result.tau_loss == 16
    assert result.onset == "candidate_loss_at_onset"


def test_first_passage_right_censored():
    result = first_passage(2, [(0, 1, 1), (16, 2, 1)])
    assert result.tau_rec is None
    assert result.tau_loss is None
    assert result.onset == "right_censored"


def test_rejects_inconsistent_oracle_error():
    with pytest.raises(ValueError, match="cannot exceed"):
        classify_state(2, 2, 3)
