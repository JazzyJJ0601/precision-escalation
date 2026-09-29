import numpy as np
from pe.gate import entropy_gate, calibrate_threshold


def test_uniform_logits():
    # High entropy (close to log(4) ~ 1.38)
    logits = np.array([1.0, 1.0, 1.0, 1.0])
    assert entropy_gate(logits, 1.0) == True


def test_peaked_logits():
    # Low entropy (close to 0)
    logits = np.array([10.0, 0.0, 0.0, 0.0])
    assert entropy_gate(logits, 1.0) == False


def test_calibration():
    np.random.seed(42)
    entropies = np.random.uniform(0, 10, 1000)
    frac = 0.8
    thresh = calibrate_threshold(entropies, frac)
    # Threshold at 80th percentile -> roughly 20% should exceed it
    actual_frac = np.mean(entropies > thresh)
    expected_frac = 1.0 - frac
    assert abs(actual_frac - expected_frac) < 0.05, f"{actual_frac} vs {expected_frac}"
