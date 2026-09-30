from cgmr.projection import classify_state


def test_state_n():
    assert classify_state(baseline_error=2, adapted_1best_error=2, oracle_beam_error=1) == "N"
    assert classify_state(baseline_error=2, adapted_1best_error=1, oracle_beam_error=1) == "N"


def test_state_a():
    assert classify_state(baseline_error=2, adapted_1best_error=4, oracle_beam_error=2) == "A"
    assert classify_state(baseline_error=2, adapted_1best_error=4, oracle_beam_error=1) == "A"


def test_state_d():
    assert classify_state(baseline_error=2, adapted_1best_error=4, oracle_beam_error=3) == "D"
