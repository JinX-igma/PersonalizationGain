"""Public entry point: `RepeatabilityCheck`.

Turns the paper's diagnostic into something any personalization study can run
on its own results, without re-deriving the estimator:

    from repgain import RepeatabilityCheck

    check = RepeatabilityCheck(n_boot=2000, seed=0)
    report = check.run(df)   # columns: subject, episode, gain[, group]
    print(report.summary())
    report.plot()

`df` needs at least two episodes per (subject, group) cell of a
personalization gain that was measured with one *fixed* adapter -- i.e. the
adapter must not be refit between the two episodes, or the two measurements
are not repeats of the same thing. If you only have raw per-episode scores
under two conditions, pass `condition` + `score` columns instead of `gain`;
`RepeatabilityCheck` will pair them into a gain for you.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from repgain.bootstrap import BootstrapResult, participant_bootstrap
from repgain.metrics import (
    episode_repeatability_sigma,
    median_abs,
    reference_half_widths,
    sign_flip_rate,
)

_SINGLE_GROUP = "_all"


@dataclass
class RepeatabilityReport:
    sigma: float
    sigma_ci: BootstrapResult
    flip_rate: float
    n_flipped: int
    n_eligible_pairs: int
    per_subject_gain: pd.Series
    median_abs_gain: float
    inner_half_width: float
    outer_half_width: float
    n_pairs: int
    n_subjects: int
    n_independent_blocks: int
    _gain_a: np.ndarray = field(repr=False)
    _gain_b: np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        g = self.per_subject_gain.abs()
        return {
            "sigma": self.sigma,
            "sigma_ci_lo": self.sigma_ci.lo,
            "sigma_ci_hi": self.sigma_ci.hi,
            "flip_rate": self.flip_rate,
            "n_flipped": self.n_flipped,
            "n_eligible_pairs": self.n_eligible_pairs,
            "median_abs_gain": self.median_abs_gain,
            "inner_half_width": self.inner_half_width,
            "outer_half_width": self.outer_half_width,
            "n_pairs": self.n_pairs,
            "n_subjects": self.n_subjects,
            "n_subjects_within_inner": int((g <= self.inner_half_width).sum()),
            "n_subjects_within_outer": int((g <= self.outer_half_width).sum()),
        }

    def summary(self) -> str:
        d = self.to_dict()
        pct = 100 * (1 - self.sigma_ci.alpha)
        lines = [
            f"episode repeatability sigma = {d['sigma']:.4f} "
            f"({pct:.0f}% CI [{d['sigma_ci_lo']:.4f}, {d['sigma_ci_hi']:.4f}], "
            f"{d['n_pairs']} episode pairs, {d['n_subjects']} subjects)",
            f"median |per-subject gain| = {d['median_abs_gain']:.4f}",
            f"sign flips between repeated episodes: {d['flip_rate']:.0%} "
            f"({d['n_flipped']}/{d['n_eligible_pairs']})",
            f"reference half-widths: inner=±{d['inner_half_width']:.4f} "
            f"(assumes {self.n_independent_blocks} uncorrelated blocks), "
            f"outer=±{d['outer_half_width']:.4f} (no further-averaging benefit)",
            f"{d['n_subjects_within_outer']}/{d['n_subjects']} subjects have |gain| inside the outer "
            f"half-width; {d['n_subjects_within_inner']}/{d['n_subjects']} inside the inner half-width",
        ]
        return "\n".join(lines)

    def plot(self, ax=None):
        """Per-subject gain scattered against the two reference half-widths.

        Mirrors Figure 1(b) of the paper. Returns the Figure (creates one
        with Agg-friendly defaults if `ax` is not given).
        """
        import matplotlib.pyplot as plt

        owns_fig = ax is None
        if owns_fig:
            fig, ax = plt.subplots(figsize=(4.2, 2.6))
        else:
            fig = ax.figure

        g = self.per_subject_gain.to_numpy()
        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.08, 0.08, size=len(g))
        for x in (-self.outer_half_width, self.outer_half_width):
            ax.axvline(x, color="0.3", lw=1.0, ls=(0, (1.4, 1.6)))
        for x in (-self.inner_half_width, self.inner_half_width):
            ax.axvline(x, color="0.3", lw=1.0, ls=(0, (5, 2)))
        ax.axvline(0, color="0.15", lw=1.0)
        ax.scatter(g, jitter, s=24, facecolor="#3d7fb5", edgecolor="0.2", linewidth=0.4, alpha=0.9)
        ax.set_yticks([])
        ax.set_xlabel("per-subject personalization gain")
        ax.set_title("gain vs. episode-repeatability reference scales", fontsize=9)
        if owns_fig:
            fig.tight_layout()
        return fig


class RepeatabilityCheck:
    """Estimate how much a personalization gain moves between two episodes of
    the same (subject, group) cell, holding the adapter fixed.
    """

    def __init__(
        self,
        n_boot: int = 2000,
        seed: int = 0,
        alpha: float = 0.05,
        n_independent_blocks: int = 2,
        baseline_label: str = "baseline",
        personalized_label: str = "personalized",
    ):
        self.n_boot = n_boot
        self.seed = seed
        self.alpha = alpha
        self.n_independent_blocks = n_independent_blocks
        self.baseline_label = baseline_label
        self.personalized_label = personalized_label

    def _gains(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "group" not in df.columns:
            df["group"] = _SINGLE_GROUP
        if "gain" in df.columns:
            return df[["subject", "group", "episode", "gain"]]
        required = {"condition", "score"}
        if not required.issubset(df.columns):
            raise ValueError(
                "RepeatabilityCheck.run needs either a 'gain' column, or both "
                "'condition' and 'score' columns to derive one"
            )
        wide = df.pivot_table(
            index=["subject", "group", "episode"], columns="condition", values="score"
        )
        missing = {self.baseline_label, self.personalized_label} - set(wide.columns)
        if missing:
            raise ValueError(f"condition column is missing expected label(s): {missing}")
        wide["gain"] = wide[self.personalized_label] - wide[self.baseline_label]
        return wide.reset_index()[["subject", "group", "episode", "gain"]]

    def run(self, df: pd.DataFrame) -> RepeatabilityReport:
        gains = self._gains(df)

        subjects_order: list = []
        pair_diffs: list[float] = []
        pair_subjects: list = []
        gain_a: list[float] = []
        gain_b: list[float] = []
        for (subject, _group), cell in gains.groupby(["subject", "group"]):
            values = cell.sort_values("episode")["gain"].to_numpy(dtype=float)
            if len(values) < 2:
                continue
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    pair_diffs.append(values[i] - values[j])
                    pair_subjects.append(subject)
                    gain_a.append(values[i])
                    gain_b.append(values[j])
            if subject not in subjects_order:
                subjects_order.append(subject)

        if not pair_diffs:
            raise ValueError(
                "no (subject, group) cell has two or more episodes; RepeatabilityCheck "
                "needs repeated episodes under one fixed adapter to measure repeatability"
            )

        pair_diffs_arr = np.asarray(pair_diffs, dtype=float)
        gain_a_arr, gain_b_arr = np.asarray(gain_a, dtype=float), np.asarray(gain_b, dtype=float)
        pair_subjects_arr = np.asarray(pair_subjects)

        by_subject = {
            s: pair_diffs_arr[pair_subjects_arr == s] for s in np.unique(pair_subjects_arr)
        }

        def statistic(picked_subjects: np.ndarray) -> float:
            values = np.concatenate([by_subject[s] for s in picked_subjects])
            return episode_repeatability_sigma(values)

        sigma_ci = participant_bootstrap(
            list(by_subject), statistic, seed=self.seed, n_boot=self.n_boot, alpha=self.alpha
        )
        flip_rate, n_flipped, n_eligible = sign_flip_rate(gain_a_arr, gain_b_arr)
        per_subject_gain = gains.groupby("subject")["gain"].mean()
        inner, outer = reference_half_widths(sigma_ci.estimate, self.n_independent_blocks)

        return RepeatabilityReport(
            sigma=sigma_ci.estimate,
            sigma_ci=sigma_ci,
            flip_rate=flip_rate,
            n_flipped=n_flipped,
            n_eligible_pairs=n_eligible,
            per_subject_gain=per_subject_gain,
            median_abs_gain=median_abs(per_subject_gain.to_numpy()),
            inner_half_width=inner,
            outer_half_width=outer,
            n_pairs=len(pair_diffs_arr),
            n_subjects=len(by_subject),
            n_independent_blocks=self.n_independent_blocks,
            _gain_a=gain_a_arr,
            _gain_b=gain_b_arr,
        )
