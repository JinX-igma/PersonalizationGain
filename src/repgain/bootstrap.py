"""Participant-level (cluster) bootstrap, with an explicit, required seed.

Every estimator in this package is computed across users, so independence
across *rows* of a dataframe cannot be assumed when a user contributes more
than one row. The bootstrap here always resamples whole subjects, never rows,
and never defaults a seed silently -- callers must pass one so a reported CI
is exactly reproducible.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    lo: float
    hi: float
    alpha: float
    draws: np.ndarray

    def __repr__(self) -> str:  # concise enough to print in a notebook
        pct = 100 * (1 - self.alpha)
        return f"BootstrapResult({self.estimate:.4f}, {pct:.0f}% CI=[{self.lo:.4f}, {self.hi:.4f}])"


def participant_bootstrap(
    subjects: Sequence[Any],
    statistic: Callable[[np.ndarray], float],
    *,
    seed: int,
    n_boot: int = 5000,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Resample subjects with replacement and recompute `statistic` each time.

    `statistic` receives an array of subject ids (with repeats) and must look
    up that subject's own data and return one scalar. This indirection is
    what lets the same helper back a bootstrap over per-row diffs
    (`covariance_check`-style) or over a pre-aggregated per-subject value
    (`repeatability_stratified`-style): the aggregation lives in the closure
    the caller passes in, not here.
    """
    rng = np.random.default_rng(seed)
    subjects_arr = np.asarray(subjects)
    n = len(subjects_arr)
    if n == 0:
        raise ValueError("participant_bootstrap requires at least one subject")

    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        picked = rng.choice(subjects_arr, size=n, replace=True)
        draws[i] = statistic(picked)
    draws.sort()

    estimate = float(statistic(subjects_arr))
    lo_idx = int(alpha / 2 * n_boot)
    hi_idx = min(int((1 - alpha / 2) * n_boot), n_boot - 1)
    return BootstrapResult(
        estimate=estimate, lo=float(draws[lo_idx]), hi=float(draws[hi_idx]), alpha=alpha, draws=draws
    )
