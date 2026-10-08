"""Repeated-episode transport diagnostic.

The shipped evaluation design has one outcome trial per (user, activity)
cell, so trial-to-trial variability of the personalization gain is not
identified by that design alone. Three MotionSense activities (dws, ups, wlk)
have TWO long trials each. Holding out both, fitting ONE adapter on the
remaining 13 trials, and scoring that same fixed adapter separately on each
held-out trial yields two independently recorded realizations of the same
(user, activity) gain. This measures directly what the shipped design cannot
-- and is the source table for `analysis/covariance_check.py`,
`analysis/repeatability_stratified.py`, and `paper/make_figures.py`.

Stage 1: needs `results/motionsense_decision_windows_w50_s50_t100.csv` and
`results/motionsense_decision_audit_recording_split.csv`, both produced by
`experiments/extract_and_fit.py`. Output: `results/episode_transport.csv`
(committed as a frozen Stage 2 input; rerun this only to verify or refresh
it).
"""

import json
import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from repgain.adapters import BACKBONE_C, MAX_ITER, SEED, fit_anchored
from repgain.adapters import activity_trial_weights as atw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PAIRED = {"dws": [1, 2], "ups": [3, 4], "wlk": [7, 8]}   # activities with two long trials
C_FIXED, ANCHOR = 10.0, "identity"                       # development-selected policy

win = pd.read_csv(os.path.join(RES, "motionsense_decision_windows_w50_s50_t100.csv"))
spl = pd.read_csv(os.path.join(RES, "motionsense_decision_audit_recording_split.csv"))
feats = [c for c in win.columns if c not in
         ("window_id","session_key","subject_id","activity","label_id","trial",
          "window_index","raw_start","raw_end_exclusive","source_rows","source_file")]

rows = []
for fold in sorted(spl.split_id.unique()):
    s = spl[spl.split_id == fold]
    tr = win[win.session_key.isin(set(s.loc[s.used_for_final_backbone, "session_key"]))]
    sc = StandardScaler(); w = atw(tr)
    sc.fit(tr[feats].to_numpy(float), sample_weight=w)
    bb = LogisticRegression(C=BACKBONE_C, solver="lbfgs", max_iter=MAX_ITER,
                            tol=1e-6, random_state=SEED)
    bb.fit(sc.transform(tr[feats].to_numpy(float)), tr["label_id"].to_numpy(int), sample_weight=w)

    for subj in sorted(s.loc[s.subject_role == "target", "subject_id"].unique()):
        sf = win[win.subject_id == subj]
        for act, held in PAIRED.items():
            F = sf[~((sf.activity == act) & (sf.trial.isin(held)))]   # 13 trials
            zf = bb.decision_function(sc.transform(F[feats].to_numpy(float)))
            W, b, ok = fit_anchored(zf, F["label_id"].to_numpy(int), atw(F), C_FIXED, ANCHOR)
            g = {}
            for t in held:
                U = sf[(sf.activity == act) & (sf.trial == t)]
                zu = bb.decision_function(sc.transform(U[feats].to_numpy(float)))
                y = U["label_id"].to_numpy(int)
                base = float((bb.classes_[zu.argmax(1)] != y).mean())
                corr = float((bb.classes_[(zu @ W.T + b).argmax(1)] != y).mean())
                g[t] = dict(gain=base - corr, base=base, n=len(U))
            rows.append(dict(fold=fold, subject=int(subj), activity=act, converged=ok,
                             trial_a=held[0], trial_b=held[1],
                             gain_a=g[held[0]]["gain"], gain_b=g[held[1]]["gain"],
                             base_a=g[held[0]]["base"], base_b=g[held[1]]["base"],
                             n_a=g[held[0]]["n"], n_b=g[held[1]]["n"]))

df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "episode_transport.csv"), index=False)
df["disc"] = (df.gain_a - df.gain_b).abs()
df["mean_gain"] = (df.gain_a + df.gain_b) / 2
same = ((df.gain_a > 0) == (df.gain_b > 0)) | ((df.gain_a == 0) & (df.gain_b == 0))
nz = (df.gain_a != 0) | (df.gain_b != 0)

print(f"=== REPEATED-EPISODE TRANSPORT ({len(df)} user-activity cells, "
      f"{df.subject.nunique()} users x {len(PAIRED)} activities) ===")
print(f"  adapter: {ANCHOR}-anchored, C={C_FIXED}, fitted once on 13 trials per cell")
print(f"  all converged: {bool(df.converged.all())}\n")
print(f"  median |gain_a - gain_b|          : {df.disc.median():.4f}")
print(f"  median |mean gain|                : {df.mean_gain.abs().median():.4f}")
print(f"  ratio (discrepancy / effect)      : {df.disc.median()/max(df.mean_gain.abs().median(),1e-9):.2f}")
print(f"  90th pct |gain_a - gain_b|        : {df.disc.quantile(.9):.4f}")
print(f"  sign agreement across the 2 trials: {int(same[nz].sum())}/{int(nz.sum())} "
      f"= {100*same[nz].mean():.1f}%")
print(f"  corr(gain_a, gain_b)              : {df.gain_a.corr(df.gain_b):.3f}")
print(f"  cells where one trial says + and the other says -: "
      f"{int((~same & nz).sum())}/{int(nz.sum())}")
print()
for act in PAIRED:
    d = df[df.activity == act]
    sa = ((d.gain_a > 0) == (d.gain_b > 0))
    print(f"  {act}: median|disc|={d.disc.median():.4f}  corr={d.gain_a.corr(d.gain_b):+.3f}  "
          f"sign agree={100*sa.mean():.0f}%")
json.dump(dict(n_cells=len(df), median_discrepancy=round(float(df.disc.median()), 4),
               median_abs_effect=round(float(df.mean_gain.abs().median()), 4),
               sign_agreement=round(float(same[nz].mean()), 4),
               corr=round(float(df.gain_a.corr(df.gain_b)), 4)),
          open(os.path.join(RES, "episode_transport_summary.json"), "w"), indent=2)
