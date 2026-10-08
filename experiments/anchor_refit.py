"""Identity-anchored vs zero-anchored corrector refit -- a regularization-path
sensitivity check.

Motivation: a corrector fit with an off-the-shelf L2 penalty shrinks W toward
0, but the no-adaptation baseline is the identity map on frozen logits. So
that regularization path never approaches "no adaptation" -- as C decreases,
the adapter degenerates toward a prior-dominated classifier instead. A
reported "harm is flat across a C range" therefore does not test
conservatism by itself.

This script refits the same 6x6+bias map on raw frozen logits anchored at
A in {0, I} (see `repgain.adapters.affine_logit.fit_anchored`), so that under
A = I, C -> 0 recovers the baseline exactly. Both anchors share one
optimizer, so the only difference is the shrinkage target.

Estimand: risk = error averaged over trials within activity, then over the
six activities with equal weight. Per-user gains here are point estimates on
the designated outcome recordings; they are NOT accompanied by intervals,
because the design has one outcome trial per (user, activity) and per-user
future-trial variance is unidentified -- that is exactly what
`episode_transport.py` and `enumerate_future_episode_splits.py` measure
instead.

Stage 1: needs `results/motionsense_decision_windows_w50_s50_t100.csv` and
`results/motionsense_decision_audit_recording_split.csv`, both produced by
`experiments/extract_and_fit.py`.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from repgain.adapters import BACKBONE_C, MAX_ITER, SEED, equal_activity_risk, fit_anchored
from repgain.adapters import activity_trial_weights as atw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CGRID = [0.01, 0.1, 1.0, 10.0]

win = pd.read_csv(os.path.join(RES, "motionsense_decision_windows_w50_s50_t100.csv"))
spl = pd.read_csv(os.path.join(RES, "motionsense_decision_audit_recording_split.csv"))
feats = [c for c in win.columns if c not in
         ("window_id", "session_key", "subject_id", "activity", "label_id", "trial",
          "window_index", "raw_start", "raw_end_exclusive", "source_rows", "source_file")]
print(f"{len(feats)} features, {len(win)} windows")

rows = []
for fold in sorted(spl["split_id"].unique()):
    s = spl[spl["split_id"] == fold]
    bb_keys = set(s.loc[s["used_for_final_backbone"], "session_key"])
    tr = win[win["session_key"].isin(bb_keys)]
    sc = StandardScaler(); wgt = atw(tr)
    sc.fit(tr[feats].to_numpy(float), sample_weight=wgt)
    bb = LogisticRegression(C=BACKBONE_C, solver="lbfgs", max_iter=MAX_ITER,
                            tol=1e-6, random_state=SEED)
    bb.fit(sc.transform(tr[feats].to_numpy(float)), tr["label_id"].to_numpy(int),
           sample_weight=wgt)
    print(f"{fold}: backbone on {tr['subject_id'].nunique()} subj / {len(tr)} windows")

    for subj in sorted(s.loc[s["subject_role"] == "target", "subject_id"].unique()):
        ss = s[s["subject_id"] == subj]
        fit_k = set(ss.loc[ss["target_corrector_fit"], "session_key"])
        fut_k = set(ss.loc[ss["target_future_outcome"], "session_key"])
        F = win[win["session_key"].isin(fit_k)]
        U = win[win["session_key"].isin(fut_k)]
        zf = bb.decision_function(sc.transform(F[feats].to_numpy(float)))
        zu = bb.decision_function(sc.transform(U[feats].to_numpy(float)))
        base_risk = equal_activity_risk(U, bb.classes_[zu.argmax(1)])
        wf = atw(F); yf = F["label_id"].to_numpy(int)
        for anchor in ("zero", "identity"):
            for C in CGRID:
                W, b, ok = fit_anchored(zf, yf, wf, C, anchor)
                pred = bb.classes_[(zu @ W.T + b).argmax(1)]
                rows.append(dict(fold=fold, subject=int(subj), anchor=anchor, C=C,
                                 converged=bool(ok), base_risk=base_risk,
                                 corr_risk=equal_activity_risk(U, pred),
                                 gain=base_risk - equal_activity_risk(U, pred),
                                 dist_to_identity=float(np.linalg.norm(W - np.eye(6))),
                                 norm_W=float(np.linalg.norm(W))))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "anchor_refit_per_user.csv"), index=False)

print("\n=== HARM RATE AND MEAN GAIN BY ANCHOR AND REGULARIZATION (n=24 users) ===")
print(f"{'anchor':>9} {'C':>7} {'harm_rate':>10} {'mean_gain':>10} {'median':>9} "
      f"{'worst':>8} {'||W-I||':>9} {'||W||':>8}")
summary = {}
for anchor in ("zero", "identity"):
    for C in CGRID:
        g = df[(df.anchor == anchor) & (df.C == C)]
        hr = float((g.gain < 0).mean())
        summary[f"{anchor}/C={C}"] = dict(
            harm_rate=round(hr, 4), mean_gain=round(float(g.gain.mean()), 4),
            median_gain=round(float(g.gain.median()), 4), worst=round(float(g.gain.min()), 4),
            dist_to_identity=round(float(g.dist_to_identity.mean()), 3))
        print(f"{anchor:>9} {C:>7} {hr:>10.3f} {g.gain.mean():>10.4f} "
              f"{g.gain.median():>9.4f} {g.gain.min():>8.4f} "
              f"{g.dist_to_identity.mean():>9.3f} {g.norm_W.mean():>8.3f}")
print(f"\nall converged: {bool(df.converged.all())}")
json.dump(summary, open(os.path.join(RES, "anchor_refit_summary.json"), "w"), indent=2)
print(f"wrote {os.path.join(RES,'anchor_refit_per_user.csv')} and anchor_refit_summary.json")
