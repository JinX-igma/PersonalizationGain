"""Simultaneous participant-and-design robustness.

The enumeration in `results/episode_split_enumeration.csv` shows the
cohort-average gain is positive under every one of the 216 admissible
outcome designs -- but that is a statement about this specific cohort.
Resampling participants (within fold, preserving each participant's gains
across all 216 designs) and recomputing min-over-designs mean gain in every
replicate turns it into a statement that also accounts for participant
sampling, conditional on the two fitted shared models.

Stage 2: reads only the committed `results/episode_split_peruser.csv`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SEED, N_BOOT = 0, 5000


def main() -> dict:
    pu = pd.read_csv(RES / "episode_split_peruser.csv")
    rng = np.random.default_rng(SEED)
    out: dict = {}

    for fold in sorted(pu["fold"].unique()):
        f = pu[pu.fold == fold]
        matrix = f.pivot_table(index="subject", columns="design", values="gain").to_numpy()
        n_subjects = matrix.shape[0]
        subject_idx = np.arange(n_subjects)
        obs_min = float(matrix.mean(axis=0).min())
        obs_max = float(matrix.mean(axis=0).max())

        mins = np.empty(N_BOOT)
        for b in range(N_BOOT):
            picked = rng.choice(subject_idx, n_subjects, replace=True)
            mins[b] = matrix[picked].mean(axis=0).min()
        mins.sort()
        lo = float(mins[int(0.025 * N_BOOT)])
        frac_positive = float((mins > 0).mean())

        out[fold] = dict(
            n_users=n_subjects,
            n_designs=matrix.shape[1],
            observed_min_over_designs=round(obs_min, 4),
            observed_max_over_designs=round(obs_max, 4),
            boot_min_2p5=round(lo, 4),
            frac_replicates_min_positive=round(frac_positive, 4),
        )

    out_path = RES / "design_participant_bootstrap.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    return out


if __name__ == "__main__":
    main()
