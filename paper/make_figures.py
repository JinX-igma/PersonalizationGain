"""Regenerate the paper's figure from the Stage 2 cache in `results/`.

Stage 2 only: reads `episode_transport.csv`, `covariance_check.json`,
`motionsense_decision_audit_per_user.csv`, and
`figure1b_identity_standard_design.csv` -- all either committed or produced
by the other `analysis/*.py` scripts, which `make figures` runs first. No
raw data access, no refitting, CPU only.
"""

from __future__ import annotations

import json
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
        "axes.titlesize": 9.6,
        "axes.labelsize": 9.6,
        "xtick.labelsize": 9.6,
        "ytick.labelsize": 9.6,
        "legend.fontsize": 9.6,
        "pdf.fonttype": 42,
    }
)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
os.makedirs(FIG, exist_ok=True)


def stacked_offsets(values: np.ndarray, bin_width: float, step: float) -> np.ndarray:
    """Stack nearby x values symmetrically so every participant remains visible."""
    bins = np.round(values / bin_width).astype(int)
    offsets = np.zeros(len(values), dtype=float)
    for bin_id in np.unique(bins):
        members = np.flatnonzero(bins == bin_id)
        offsets[members] = (np.arange(len(members)) - (len(members) - 1) / 2) * step
    return offsets


def main() -> Path:
    d = pd.read_csv(RES / "episode_transport.csv")
    d["m"] = (d.gain_a + d.gain_b) / 2
    d["df"] = d.gain_a - d.gain_b
    cv = json.loads((RES / "covariance_check.json").read_text())
    sd3 = cv["direct_sd_3act"]
    inner, outer = 1.96 * sd3 / np.sqrt(2), 1.96 * sd3

    pu = pd.read_csv(RES / "motionsense_decision_audit_per_user.csv")
    corrected = pd.read_csv(RES / "figure1b_identity_standard_design.csv")
    # Preserve original user order; only replace the adapter-dependent x values
    # with the ones selected under the shipped design (see analysis/figure1b.py).
    pu["future_gain"] = pu.subject_id.map(corrected.set_index("subject").gain)
    assert len(pu) == 24 and pu.future_gain.notna().all()

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.55))
    ACCENT = "#c0703c"  # panel (a): warm hue, deliberately outside panel (b)'s blue ramp
    BLUE = "#3d7fb5"  # panel (b): one hue, carries no variable within the panel

    # Panel (a): two non-negative magnitude distributions, identical encoding in both
    # rows, both taken over the same 72 user-activity cells of episode_transport.csv.
    rows_a = [
        (d.df.abs().to_numpy(), 1.0, "|episode gain\ndifference|"),
        (d.m.abs().to_numpy(), 0.0, "|activity-level\ngain|"),
    ]
    for values, y_position, label in rows_a:
        offsets = stacked_offsets(values, bin_width=0.030, step=0.042)
        ax[0].scatter(
            values, y_position + offsets, s=24, facecolor=ACCENT, edgecolor="white",
            linewidth=0.35, alpha=0.9, zorder=3,
        )
        med = float(np.median(values))
        ax[0].plot([med, med], [y_position - 0.32, y_position + 0.32], color="0.2", lw=1.1, zorder=4)
        ax[0].text(
            med, y_position + 0.44, f"median {med:.3f}", ha="center", va="bottom",
            fontsize=9.6, color="0.2",
        )
    ax[0].set_xlim(-0.02, 0.82)
    ax[0].set_ylim(-0.66, 1.86)
    ax[0].set_yticks([1.0, 0.0], [rows_a[0][2], rows_a[1][2]])
    ax[0].set_xlabel("Magnitude (accuracy units)")
    ax[0].set_title("(a) Episode variation vs.\ngain magnitude", fontsize=9.6, loc="left")

    # Panel (b): one signed distribution read against symmetric repeatability thresholds.
    g = pu.future_gain.to_numpy()
    offsets = stacked_offsets(g, bin_width=0.015, step=0.058)
    point_y = 0.55 + offsets
    for x_value in (-inner, inner):
        ax[1].axvline(
            x_value, ymax=((1.06 if x_value > 0 else 0.76) + 0.52) / 1.98, color="0.30",
            lw=1.0, ls=(0, (5, 2)), zorder=1,
        )
    for x_value in (-outer, outer):
        ax[1].axvline(
            x_value, ymax=((1.30 if x_value > 0 else 0.76) + 0.52) / 1.98, color="0.30",
            lw=1.0, ls=(0, (1.4, 1.6)), zorder=1,
        )
    ax[1].axvline(0, ymax=(0.76 + 0.52) / 1.98, color="0.15", lw=1.0, zorder=2)
    ax[1].scatter(g, point_y, s=27, facecolor=BLUE, edgecolor="0.2", linewidth=0.45, alpha=0.9, zorder=3)
    for x_value, y_label, text in (
        (inner, 1.06, f"uncorrelated triplets  ±{inner:.3f}".replace("0.", ".")),
        (outer, 1.30, f"no further averaging  ±{outer:.3f}".replace("0.", ".")),
    ):
        ax[1].plot([x_value - 0.012, x_value], [y_label, y_label], color="0.45", lw=0.7, zorder=2)
        ax[1].text(x_value - 0.018, y_label, text, ha="right", va="center", fontsize=9.6, color="0.35")
    ax[1].text(0, 0.78, "no change", ha="center", va="bottom", fontsize=9.6, color="0.15")
    ax[1].text(
        -0.215, -0.30, "$\\leftarrow$ lower\naccuracy", ha="left", va="center", fontsize=9.6, color="0.35"
    )
    ax[1].text(
        0.335, -0.30, "higher $\\rightarrow$\naccuracy", ha="right", va="center", fontsize=9.6, color="0.35"
    )
    ax[1].set_xlim(-0.22, 0.34)
    ax[1].set_ylim(-0.52, 1.46)
    ax[1].set_yticks([])
    ax[1].spines["left"].set_visible(False)
    ax[1].set_xlabel("Per-user personalization gain")
    ax[1].set_title("(b) Per-user gains vs.\nreference scales", fontsize=9.6, loc="left")

    for axis in ax:
        axis.spines[["top", "right"]].set_visible(False)
        for spine in axis.spines.values():
            spine.set_linewidth(0.6)
        axis.tick_params(labelsize=9.6, width=0.6, length=2.5)

    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.24, top=0.77, wspace=0.30)
    out_path = FIG / "repeatability.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(FIG / "repeatability.png", dpi=180, bbox_inches="tight")

    md, sdd = d.df.mean(), d.df.std(ddof=1)
    loa_lo, loa_hi = md - 1.96 * sdd, md + 1.96 * sdd
    print(f"mean diff {md:+.4f}  SD(diff) {sdd:.4f}  95% LoA [{loa_lo:+.4f}, {loa_hi:+.4f}]")
    print(f"reference half-widths: inner +/-{inner:.4f}  outer +/-{outer:.4f}")
    abs_g = np.abs(g)
    print(
        f"magnitude comparison by |gain|: <= {inner:.3f}: {(abs_g <= inner).sum()}/24 | "
        f"between: {((abs_g > inner) & (abs_g <= outer)).sum()}/24 | "
        f"> {outer:.3f}: {(abs_g > outer).sum()}/24"
    )
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    main()
