"""Exhaustive enumeration of admissible future-episode designs.

MotionSense gives dws/ups/wlk three trials each and jog/sit/std two, so
choosing exactly one outcome trial per activity admits 3^3 * 2^3 = 216
evaluation designs. The shipped protocol is one of them (all six short
trials). This enumerates the complete sensitivity surface. The 216 designs
are a FINITE SET OF DESIGN CHOICES, not random draws: report range and
counts, never a p-value over designs.

Pre-declared materiality criteria (fixed before running, from the option
review):
 1. a fold shows both positive and negative fleet mean in >=10% of designs
 2. central 80% range of fold mean gain >= 0.04
 3. the all-short protocol lies outside the 10th-90th percentile of designs
 4. majority-positive sign conclusion holds in <80% of designs in either fold

Stage 1: needs `results/motionsense_decision_windows_w50_s50_t100.csv` and
`results/motionsense_decision_audit_recording_split.csv`, both produced by
`experiments/extract_and_fit.py`. Outputs `results/episode_split_enumeration.csv`
and `results/episode_split_peruser.csv`, both committed as frozen Stage 2
inputs -- `analysis/figure1b.py`, `analysis/design_participant_bootstrap.py`,
and `analysis/make_macros.py` read them, not this script's live output.
"""

import itertools
import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from repgain.adapters import BACKBONE_C, MAX_ITER, SEED, fit_anchored
from repgain.adapters import activity_trial_weights as atw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
ACTS = ["dws", "jog", "sit", "std", "ups", "wlk"]
SHORT = {"dws": 11, "jog": 16, "sit": 13, "std": 14, "ups": 12, "wlk": 15}
C_FIXED, ANCHOR = 10.0, "identity"

win = pd.read_csv(os.path.join(RES, "motionsense_decision_windows_w50_s50_t100.csv"))
spl = pd.read_csv(os.path.join(RES, "motionsense_decision_audit_recording_split.csv"))
feats = [c for c in win.columns if c not in
         ("window_id","session_key","subject_id","activity","label_id","trial",
          "window_index","raw_start","raw_end_exclusive","source_rows","source_file")]
trials = {a: sorted(win.loc[win.activity == a, "trial"].unique()) for a in ACTS}
designs = list(itertools.product(*[trials[a] for a in ACTS]))
print(f"{len(designs)} admissible designs; shipped = {tuple(SHORT[a] for a in ACTS)}")


def binom_ge(k, n):
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


rows = []; peruser = []
for fold in sorted(spl.split_id.unique()):
    s = spl[spl.split_id == fold]
    tr = win[win.session_key.isin(set(s.loc[s.used_for_final_backbone, "session_key"]))]
    sc = StandardScaler(); w = atw(tr)
    sc.fit(tr[feats].to_numpy(float), sample_weight=w)
    bb = LogisticRegression(C=BACKBONE_C, solver="lbfgs", max_iter=MAX_ITER,
                            tol=1e-6, random_state=SEED)
    bb.fit(sc.transform(tr[feats].to_numpy(float)), tr["label_id"].to_numpy(int), sample_weight=w)
    tgt = sorted(s.loc[s.subject_role == "target", "subject_id"].unique())
    cache = {}
    for u in tgt:                                   # precompute numpy views once per user
        sf = win[win.subject_id == u]
        z = bb.decision_function(sc.transform(sf[feats].to_numpy(float)))
        acode = sf.activity.map({a: i for i, a in enumerate(ACTS)}).to_numpy()
        cache[u] = dict(z=z, y=sf["label_id"].to_numpy(int), act=acode,
                        trial=sf.trial.to_numpy(), frame=sf)
    for di, des in enumerate(designs):
        sel = np.array(des)
        gains = []
        for u in tgt:
            c = cache[u]
            m = c["trial"] == sel[c["act"]]
            F = c["frame"][~m]
            W, b, _ = fit_anchored(c["z"][~m], c["y"][~m], atw(F), C_FIXED, ANCHOR)
            zu, yu, au = c["z"][m], c["y"][m], c["act"][m]
            wrong_b = bb.classes_[zu.argmax(1)] != yu
            wrong_c = bb.classes_[(zu @ W.T + b).argmax(1)] != yu
            per = [wrong_b[au == i].mean() - wrong_c[au == i].mean() for i in range(6)]
            gains.append(float(np.mean(per)))
        g = np.array(gains); pos = int((g > 0).sum()); neg = int((g < 0).sum())
        for uu, gg in zip(tgt, gains):
            peruser.append(dict(fold=fold, design=di, subject=int(uu), gain=gg))
        rows.append(dict(fold=fold, design=di,
                         is_shipped=all(des[i] == SHORT[a] for i, a in enumerate(ACTS)),
                         n_short=sum(des[i] == SHORT[a] for i, a in enumerate(ACTS)),
                         mean=g.mean(), median=float(np.median(g)), pos=pos, neg=neg,
                         sign_p=binom_ge(pos, pos + neg) if pos + neg else np.nan,
                         worst=g.min()))
    print(f"  {fold} done")
df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "episode_split_enumeration.csv"), index=False)
pd.DataFrame(peruser).to_csv(os.path.join(RES, "episode_split_peruser.csv"), index=False)

print("\n=== SENSITIVITY SURFACE OVER 216 ADMISSIBLE DESIGNS ===")
crit = {}
for fold in sorted(df.fold.unique()):
    d = df[df.fold == fold]; sh = d[d.is_shipped].iloc[0]
    lo, hi = d["mean"].quantile(.1), d["mean"].quantile(.9)
    both = ((d["mean"] > 0).any() and (d["mean"] < 0).any())
    frac_neg = float((d["mean"] < 0).mean())
    sig = float((d.sign_p < .05).mean())
    pctile = float((d["mean"] < sh["mean"]).mean() * 100)
    print(f"\n  {fold}: shipped mean={sh['mean']:+.4f} (percentile {pctile:.1f} of designs)")
    print(f"    mean gain range   [{d['mean'].min():+.4f}, {d['mean'].max():+.4f}]")
    print(f"    central 80% range [{lo:+.4f}, {hi:+.4f}]  width={hi-lo:.4f}")
    print(f"    designs with negative fleet mean: {frac_neg*100:.1f}%")
    print(f"    designs where majority-positive holds at p<.05: {sig*100:.1f}%")
    crit[fold] = dict(c1=bool(both and frac_neg >= .10), c2=bool((hi-lo) >= .04),
                      c3=bool(not (lo <= sh["mean"] <= hi)), c4=bool(sig < .80),
                      shipped_percentile=round(pctile,1), width80=round(float(hi-lo),4),
                      frac_negative=round(frac_neg,4), frac_sig=round(sig,4))
print("\n=== PRE-DECLARED MATERIALITY CRITERIA ===")
for f, c in crit.items():
    print(f"  {f}: C1(neg>=10%)={c['c1']}  C2(width>=.04)={c['c2']}  "
          f"C3(shipped outside 10-90)={c['c3']}  C4(sign holds <80%)={c['c4']}")
met = any(v for c in crit.values() for k, v in c.items() if k.startswith("c"))
print(f"\n  ANY criterion met -> episode designation is materially consequential: {met}")
json.dump(crit, open(os.path.join(RES, "episode_split_enumeration_summary.json"), "w"), indent=2)
