"""Stratified episode-to-episode repeatability, with a participant-cluster
bootstrap and a transparent sensitivity band for the three activities that
were never repeated (jog, sit, std have only one recording each).

Replaces a single pooled variance + single propagated interval, which would
assume a common exchangeable episode component across all six activities --
an assumption the data cannot support, since only three activities were
repeated. Terminology: "episode-to-episode repeatability component under this
benchmark protocol", not "episode noise variance".

Stage 2: reads only the committed `results/episode_transport.csv` and
`results/motionsense_decision_audit_per_user.csv`. No raw data needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from repgain.bootstrap import participant_bootstrap
from repgain.metrics import episode_repeatability_sigma

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ACTS_REPEATED = ["dws", "ups", "wlk"]
BOOTSTRAP_SEED, N_BOOT = 0, 5000


def main() -> dict:
    d = pd.read_csv(RES / "episode_transport.csv")
    d["diff"] = d.gain_a - d.gain_b

    per_act_sigma = {
        a: episode_repeatability_sigma(d.loc[d.activity == a, "diff"].to_numpy())
        for a in ACTS_REPEATED
    }

    by_subject_act = {
        (s, a): g["diff"].to_numpy()
        for (s, a), g in d.groupby(["subject", "activity"])
    }
    subjects = sorted(d["subject"].unique())

    def make_statistic(target_act: str | None):
        def statistic(picked_subjects: np.ndarray) -> float:
            acts = ACTS_REPEATED if target_act is None else [target_act]
            values = np.concatenate(
                [
                    by_subject_act[(s, a)]
                    for s in picked_subjects
                    for a in acts
                    if (s, a) in by_subject_act
                ]
            )
            return episode_repeatability_sigma(values)

        return statistic

    ci = {}
    for key, act in [("pooled", None), *[(a, a) for a in ACTS_REPEATED]]:
        boot = participant_bootstrap(subjects, make_statistic(act), seed=BOOTSTRAP_SEED, n_boot=N_BOOT)
        ci[key] = [round(boot.lo, 4), round(boot.hi, 4)]

    meas = np.array([per_act_sigma[a] for a in ACTS_REPEATED])
    pu = pd.read_csv(RES / "motionsense_decision_audit_per_user.csv")
    g = pu["future_gain"].abs()
    scenarios = {
        "optimistic (unreplicated = smallest measured)": float(meas.min()),
        "central   (unreplicated = median measured)": float(np.median(meas)),
        "pessimistic (unreplicated = largest measured)": float(meas.max()),
    }
    band = {}
    for label, sigma_unreplicated in scenarios.items():
        var = (np.sum(meas**2) + 3 * sigma_unreplicated**2) / 36
        half_width = 1.96 * float(np.sqrt(var))
        band[label] = dict(
            assumed_sigma=round(sigma_unreplicated, 4),
            half_width=round(half_width, 4),
            users_within_noise=int((g <= half_width).sum()),
        )

    result = dict(
        per_activity_sigma={k: round(v, 4) for k, v in per_act_sigma.items()},
        bootstrap_ci=ci,
        sensitivity_band=band,
        n_bootstrap=N_BOOT,
    )
    out_path = RES / "repeatability_stratified.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    return result


if __name__ == "__main__":
    main()
