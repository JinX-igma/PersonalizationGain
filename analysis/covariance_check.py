"""Does the independence-across-activities assumption behind the propagated
interval hold? Two checks that require no new assumptions:

 1. cross-activity correlation of the paired differences D1-D2 across users;
 2. the DIRECT repeatability of the three-activity aggregate, obtained by
    forming each user's 3-activity mean gain separately from episode set 1
    and episode set 2. This needs no independence assumption at all, and is
    compared against what independence would have implied.

Stage 2: reads only `results/episode_transport.csv` (committed, Stage 1
output). No raw data access needed.
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
ACTS = ["dws", "ups", "wlk"]
BOOTSTRAP_SEED, N_BOOT = 0, 5000


def main() -> dict:
    d = pd.read_csv(RES / "episode_transport.csv")
    d["diff"] = d.gain_a - d.gain_b

    wide = d.pivot_table(index="subject", columns="activity", values="diff")[ACTS].dropna()
    corr = wide.corr()
    offdiag = [corr.iloc[i, j] for i in range(3) for j in range(i + 1, 3)]
    mean_offdiag_corr = float(np.mean(offdiag))

    paired = d.pivot_table(index="subject", columns="activity", values=["gain_a", "gain_b"])
    mean_a = paired["gain_a"][ACTS].mean(axis=1)
    mean_b = paired["gain_b"][ACTS].mean(axis=1)
    direct_var = float(np.mean((mean_a - mean_b) ** 2) / 2)
    sigma_per_act = {
        a: episode_repeatability_sigma(d.loc[d.activity == a, "diff"].to_numpy()) for a in ACTS
    }
    indep_var = sum(v**2 for v in sigma_per_act.values()) / 9
    sd3 = float(np.sqrt(direct_var))

    by_subject = {s: (float(mean_a.loc[s]), float(mean_b.loc[s])) for s in mean_a.index}

    def sd3_statistic(picked_subjects: np.ndarray) -> float:
        a = np.array([by_subject[s][0] for s in picked_subjects])
        b = np.array([by_subject[s][1] for s in picked_subjects])
        return episode_repeatability_sigma(a - b)

    ci = participant_bootstrap(
        list(by_subject), sd3_statistic, seed=BOOTSTRAP_SEED, n_boot=N_BOOT
    )

    nz = (d.gain_a != 0) | (d.gain_b != 0)
    result = dict(
        mean_offdiag_corr=round(mean_offdiag_corr, 4),
        direct_var_3act=round(direct_var, 6),
        indep_var_3act=round(indep_var, 6),
        ratio=round(direct_var / indep_var, 3),
        direct_sd_3act=round(sd3, 4),
        direct_sd_ci=[round(ci.lo, 4), round(ci.hi, 4)],
        zero_cells=int((~nz).sum()),
    )
    out_path = RES / "covariance_check.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    return result


if __name__ == "__main__":
    main()
