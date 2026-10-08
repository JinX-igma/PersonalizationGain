"""The fixed per-user adapter used throughout the paper's experiments.

The shared backbone is a multinomial `LogisticRegression` fit once per fold.
A user's "adapter" is an affine map (W, b) applied to that backbone's frozen
logits, fit by the same optimizer regardless of anchor:

    minimize_{W,b}  weighted multinomial cross-entropy(z @ W.T + b, y)
                     + (1 / 2C) * (||W - A||_F^2 + ||b||^2)

with A = I (identity anchor, used everywhere in the shipped results) or
A = 0 (zero anchor, used only in the regularization-path sensitivity check in
`experiments/anchor_refit.py`). Anchoring at A = I matters because it is the
only choice under which C -> 0 recovers the no-adaptation baseline exactly;
anchoring at A = 0 instead drags the adapter toward a prior-dominated
classifier as C shrinks, so it cannot represent "no adaptation."

This module intentionally has no dependency on MotionSense column names
beyond ("activity", "trial", "subject_id") for the trial-balancing weights,
and ("activity", "trial") for the risk aggregation -- both are generic
groupby reductions, not dataset-specific logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

BACKBONE_C = 1.0
MAX_ITER = 5000
SEED = 20240521


def activity_trial_weights(frame: pd.DataFrame) -> np.ndarray:
    """Equalize subjects, activities, trials within activity, then windows.

    Without this, activities and trials with more windows (longer
    recordings) would dominate the fit and the risk estimate purely because
    they contributed more rows, not because they matter more.
    """
    k = frame[["subject_id", "activity", "trial"]]
    sc = k.groupby(["subject_id", "activity", "trial"]).size()
    tc = k.drop_duplicates().groupby(["subject_id", "activity"]).size()
    ac = k.drop_duplicates(["subject_id", "activity"]).groupby("subject_id").size()
    ns = k["subject_id"].nunique()
    w = np.array(
        [
            1.0
            / (
                ns
                * ac.loc[r.subject_id]
                * tc.loc[(r.subject_id, r.activity)]
                * sc.loc[(r.subject_id, r.activity, r.trial)]
            )
            for r in k.itertuples(index=False)
        ]
    )
    return w * len(w) / w.sum()


def equal_activity_risk(frame: pd.DataFrame, pred: np.ndarray) -> float:
    """Error averaged over trials within an activity, then over activities equally."""
    d = frame[["activity", "trial"]].copy()
    d["wrong"] = (pred != frame["label_id"].to_numpy()).astype(float)
    per_trial = d.groupby(["activity", "trial"])["wrong"].mean()
    return float(per_trial.groupby("activity").mean().mean())


def fit_anchored(
    z: np.ndarray, y: np.ndarray, w: np.ndarray, c: float, anchor: str
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Fit the weighted-multinomial affine corrector anchored at 0 or the identity.

    Returns (W, b, converged). `z` is the backbone's frozen `decision_function`
    output (n_samples, n_classes); `y` are integer class indices; `w` are
    per-row weights (typically from `activity_trial_weights`).
    """
    k = z.shape[1]
    anchor_matrix = np.eye(k) if anchor == "identity" else np.zeros((k, k))
    y_onehot = np.zeros((len(y), k))
    y_onehot[np.arange(len(y)), y] = 1.0
    w_norm = w / w.sum()

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        w_mat = theta[: k * k].reshape(k, k)
        b = theta[k * k :]
        s = z @ w_mat.T + b
        s = s - s.max(axis=1, keepdims=True)
        log_softmax = s - np.log(np.exp(s).sum(axis=1, keepdims=True))
        nll = -np.sum(w_norm * (log_softmax * y_onehot).sum(axis=1))
        reg = (np.sum((w_mat - anchor_matrix) ** 2) + np.sum(b**2)) / (2 * c)
        p = np.exp(log_softmax)
        grad_logits = w_norm[:, None] * (p - y_onehot)
        grad = np.concatenate(
            [(grad_logits.T @ z + (w_mat - anchor_matrix) / c).ravel(), grad_logits.sum(0) + b / c]
        )
        return nll + reg, grad

    theta0 = np.concatenate([anchor_matrix.ravel(), np.zeros(k)])
    result = minimize(
        objective,
        theta0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": MAX_ITER, "ftol": 1e-14, "gtol": 1e-10},
    )
    w_mat = result.x[: k * k].reshape(k, k)
    return w_mat, result.x[k * k :], bool(result.success)
