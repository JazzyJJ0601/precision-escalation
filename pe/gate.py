import numpy as np


def entropy_gate(logits, threshold):
    """Return True if softmax entropy of 1D logits exceeds threshold."""
    logits = np.asarray(logits, dtype=np.float64)
    # Numerical stability
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)
    entropy = -np.sum(probs * np.log(probs))
    return entropy > threshold


def calibrate_threshold(entropies, escalate_fraction):
    """Return the quantile threshold at the given escalation fraction."""
    entropies = np.asarray(entropies)
    return np.quantile(entropies, escalate_fraction)
