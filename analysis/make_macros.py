"""Regenerate every number referenced in `paper/claims.yaml` from the
committed `results/` cache and write it to `results/generated_macros.json`.

This is the single place where "a number in the paper" turns into "a number
computed by code, from a named file, under a named seed" -- the same role
`experiments/.../make_v3_macros.py` played in the original research repo,
just scoped to the headline claims and driven off the already-regenerated
Stage 2 outputs (run `make figures`, which calls this last).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from repgain.metrics import (
    episode_repeatability_sigma,
    median_abs,
    reference_half_widths,
    sign_flip_rate,
)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BOOTSTRAP_SEED, N_BOOT = 0, 5000


def main() -> dict:
    d = pd.read_csv(RES / "episode_transport.csv")
    enum = pd.read_csv(RES / "episode_split_enumeration.csv")
    cov = json.loads((RES / "covariance_check.json").read_text())
    fig1b = pd.read_csv(RES / "figure1b_identity_standard_design.csv")

    sigma_pooled = episode_repeatability_sigma((d.gain_a - d.gain_b).to_numpy())
    flip_rate, n_flipped, n_eligible = sign_flip_rate(d.gain_a.to_numpy(), d.gain_b.to_numpy())

    from repgain.bootstrap import participant_bootstrap

    by_subject = {s: g["diff"].to_numpy() for s, g in d.assign(diff=d.gain_a - d.gain_b).groupby("subject")}
    sigma_ci = participant_bootstrap(
        list(by_subject),
        lambda picked: episode_repeatability_sigma(np.concatenate([by_subject[s] for s in picked])),
        seed=BOOTSTRAP_SEED,
        n_boot=N_BOOT,
    )

    bracket_lo, bracket_hi = reference_half_widths(cov["direct_sd_3act"], n_independent_blocks=2)

    n_designs = int(enum["design"].nunique())
    min_mean_per_fold = enum.groupby("fold")["mean"].min()
    frac_negative_designs = float((min_mean_per_fold < 0).mean())

    gain = fig1b["gain"]
    macros = {
        "SigmaPooled": round(float(sigma_pooled), 3),
        "SigmaPooledCI": [round(sigma_ci.lo, 3), round(sigma_ci.hi, 3)],
        "MedEffect": round(median_abs(((d.gain_a + d.gain_b) / 2).to_numpy()), 3),
        "SignFlipPct": round(100 * flip_rate),
        "SignFlipNum": n_flipped,
        "SignFlipDen": n_eligible,
        "NCells": len(d),
        "DirectSdThree": round(cov["direct_sd_3act"], 3),
        "DirectSdThreeCI": [round(cov["direct_sd_ci"][0], 3), round(cov["direct_sd_ci"][1], 3)],
        "BracketLo": round(float(bracket_lo), 3),
        "BracketHi": round(float(bracket_hi), 3),
        "NDesigns": n_designs,
        "FracNegDesigns": round(100 * frac_negative_designs),
        "NUsersV": int(fig1b["subject"].nunique()),
        "NFeat": 141,
        "NPos": int((gain > 0).sum()),
        "NNeg": int((gain < 0).sum()),
        "MedGain": round(float(gain.median()), 3),
        "FleetMean": round(float(gain.mean()), 3),
    }

    out_path = RES / "generated_macros.json"
    out_path.write_text(json.dumps(macros, indent=2) + "\n")
    print(json.dumps(macros, indent=2))
    print(f"wrote {out_path}")
    return macros


if __name__ == "__main__":
    main()
