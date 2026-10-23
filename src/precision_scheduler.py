"""Fixed, configurable initial heuristics; no calibration or learned policy."""
import math

def precision_for_score(score, thresholds=(0.30, 0.65)):
    low, high = thresholds
    if not (0 <= low < high <= 1) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Expected finite [0,1] score and ordered [0,1] thresholds')
    return 4 if score < low else 8 if score < high else 16
