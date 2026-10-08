"""An extended, more didactic figure for the README (not a paper figure):
individual vs. group variability, in two views.

(a) Per-user forest plot: each of the 24 participants' gain under the shipped
design, sorted, against the cohort mean (dashed) and the outer
no-further-averaging noise band (shaded). Grey points are indistinguishable
from zero at this measurement's own repeatability scale; colored points are
not. This is the "a positive cohort average can hide an unrepeatable
individual number" claim, drawn directly.

(b) Test-retest reliability: for every (user, activity) cell with two
repeated episodes, gain measured on episode A vs. gain measured on episode B.
A perfectly repeatable measurement would sit on the diagonal; the
off-diagonal spread *is* the episode repeatability sigma reported in the
paper. Points in the pink quadrants had a different sign in the two
episodes.

Stage 2: reads only the committed/regenerated `results/` cache (same inputs
as `paper/make_figures.py`). CPU only, no raw data needed.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update(
    {
        "font.size": 9.6,
        "axes.titlesize": 10.2,
        "axes.labelsize": 9.6,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.6,
        "pdf.fonttype": 42,
    }
)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
os.makedirs(FIG, exist_ok=True)

# Validated against dataviz/scripts/validate_palette.js (--mode light --pairs all):
# all checks pass except a contrast WARN on the aqua, mitigated by the white
# marker edges and the legend's direct labels.
ACT_LABELS = {"dws": "downstairs", "ups": "upstairs", "wlk": "walking"}
ACT_COLORS = {"dws": "#eb6834", "ups": "#2a78d6", "wlk": "#1baf7a"}
WITHIN_NOISE = "0.68"  # recessive: "not distinguishable from zero here"
OUTSIDE_NOISE = "#2a78d6"
REFERENCE_LINE = "#2b2b2b"


def main() -> Path:
    import json

    fig1b = pd.read_csv(RES / "figure1b_identity_standard_design.csv")
    transport = pd.read_csv(RES / "episode_transport.csv")
    cov = json.loads((RES / "covariance_check.json").read_text())
    sd3 = cov["direct_sd_3act"]
    outer = 1.96 * sd3

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 4.0))

    # --- (a) per-user forest plot: individual gains vs. the cohort mean ---
    g = fig1b.sort_values("gain").reset_index(drop=True)
    n = len(g)
    y = np.arange(n)
    cohort_mean = float(g["gain"].mean())
    within_noise = g["gain"].abs() <= outer
    colors = np.where(within_noise, WITHIN_NOISE, OUTSIDE_NOISE)

    ax[0].axvspan(-outer, outer, color="0.92", zorder=0)
    ax[0].axvline(0, color="0.3", lw=0.9, zorder=1)
    ax[0].axvline(cohort_mean, color=REFERENCE_LINE, lw=1.5, ls="--", zorder=2)
    ax[0].hlines(y, 0, g["gain"], color=colors, lw=1.1, zorder=2, alpha=0.8)
    ax[0].scatter(g["gain"], y, s=24, c=colors, edgecolor="white", linewidth=0.6, zorder=3)
    ax[0].set_ylim(-1.2, n + 1.6)
    ax[0].set_yticks([])
    ax[0].set_ylabel(f"{len(g)} participants, sorted by gain")
    ax[0].set_xlabel("per-user personalization gain")
    ax[0].set_title("(a) Group mean is positive;\nmost individuals are not distinguishable from zero")
    ax[0].spines[["top", "right"]].set_visible(False)

    # Direct labels instead of a legend box, placed in the padded strip above all
    # rows (set via set_ylim) so they never sit on top of a stem or marker.
    n_within = int(within_noise.sum())
    ax[0].text(
        -outer, n + 1.15, f"within noise of zero (±{outer:.3f})",
        color="0.4", fontsize=8.2, va="bottom", ha="left",
    )
    ax[0].text(
        -outer, n + 0.25, f"cohort mean {cohort_mean:+.3f}",
        color=REFERENCE_LINE, fontsize=8.6, va="bottom", ha="left",
    )
    ax[0].text(
        0.98, 0.04,
        f"{n_within}/{n} users fall inside the noise band\ndespite a positive cohort mean",
        transform=ax[0].transAxes, fontsize=8.6, color="0.35", va="bottom", ha="right",
    )

    # --- (b) test-retest reliability: gain on episode A vs. episode B ---
    lim = max(transport["gain_a"].abs().max(), transport["gain_b"].abs().max()) * 1.15
    ax[1].axvspan(-lim, 0, 0.5, 1.0, color="#fbeceb", zorder=0)
    ax[1].axvspan(0, lim, 0.0, 0.5, color="#fbeceb", zorder=0)
    ax[1].plot([-lim, lim], [-lim, lim], color="0.4", lw=1.0, ls=(0, (4, 2)), zorder=1,
               label="perfect repeatability")
    ax[1].axhline(0, color="0.3", lw=0.8, zorder=1)
    ax[1].axvline(0, color="0.3", lw=0.8, zorder=1)
    for act, sub in transport.groupby("activity"):
        ax[1].scatter(
            sub["gain_a"], sub["gain_b"], s=26, facecolor=ACT_COLORS[act], edgecolor="white",
            linewidth=0.4, alpha=0.9, zorder=3, label=ACT_LABELS[act],
        )
    ax[1].set_xlim(-lim, lim)
    ax[1].set_ylim(-lim, lim)
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("gain measured on episode A")
    ax[1].set_ylabel("gain measured on episode B")
    ax[1].set_title("(b) Same user, same activity, same adapter --\ntwo episodes, two different answers")
    ax[1].legend(loc="lower left", frameon=False, fontsize=7.8, handletextpad=0.4)
    ax[1].spines[["top", "right"]].set_visible(False)

    corr = transport["gain_a"].corr(transport["gain_b"])
    same_sign = ((transport.gain_a > 0) == (transport.gain_b > 0))
    nz = (transport.gain_a != 0) | (transport.gain_b != 0)
    ax[1].text(
        0.97, 0.03,
        f"corr = {corr:+.2f}\nsame sign in {int(same_sign[nz].sum())}/{int(nz.sum())} cells",
        transform=ax[1].transAxes, fontsize=8.4, color="0.35", va="bottom", ha="right",
    )

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.85, wspace=0.32)
    out_path = FIG / "overview.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(FIG / "overview.png", dpi=180, bbox_inches="tight")
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    main()
