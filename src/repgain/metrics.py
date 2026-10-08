"""Pure functions for episode-repeatability diagnostics.

No I/O, no randomness here (bootstrap resampling lives in `repgain.bootstrap`),
so every function is trivial to property-test. These implement the estimators
from Sections 2-3 of the paper: a fixed personalization adapter scored on two
separate episodes of the same (subject, group) cell gives two draws of the
same underlying gain; the spread between those draws is the measurement's
episode repeatability.
"""

from __future__ import annotations

import numpy as np


def episode_repeatability_sigma(paired_diff: np.ndarray) -> float:
    """RMS scale of a single episode's deviation, from paired differences.

    If D1, D2 are two independent draws of the same (subject, group) gain with
    per-episode noise variance sigma^2, then D1 - D2 has variance 2*sigma^2.
    This estimator inverts that: sigma = sqrt(mean((D1 - D2)^2) / 2).
    """
    diff = np.asarray(paired_diff, dtype=float)
    if diff.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(diff**2) / 2))


def sign_flip_rate(gain_a: np.ndarray, gain_b: np.ndarray) -> tuple[float, int, int]:
    """Fraction of cells whose positive/negative status differs between episodes.

    Cells where both episodes measure exactly zero gain are excluded from the
    denominator (there is no sign to disagree on). Returns
    (flip_rate, n_flipped, n_eligible).
    """
    a = np.asarray(gain_a, dtype=float)
    b = np.asarray(gain_b, dtype=float)
    eligible = (a != 0) | (b != 0)
    flipped = ((a > 0) != (b > 0)) & eligible
    n_eligible = int(eligible.sum())
    if n_eligible == 0:
        return float("nan"), 0, 0
    return float(flipped.sum() / n_eligible), int(flipped.sum()), n_eligible


def reference_half_widths(sigma: float, n_independent_blocks: int = 2) -> tuple[float, float]:
    """95% reference half-widths for a per-subject gain built from several episodes.

    `outer` assumes no benefit at all from averaging multiple episode-level
    components (the conservative "no further averaging" bound); `inner`
    assumes `n_independent_blocks` uncorrelated repetitions of this same
    component. The true precision of an averaged multi-group gain lies
    between the two unless the unreplicated groups differ systematically in
    scale (see `agent`/paper Section 4).
    """
    outer = 1.96 * sigma
    inner = outer / np.sqrt(n_independent_blocks)
    return inner, outer


def median_abs(values: np.ndarray) -> float:
    """Median absolute value, used for the "typical gain magnitude" comparator."""
    return float(np.median(np.abs(np.asarray(values, dtype=float))))
