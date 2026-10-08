import numpy as np
import pandas as pd
import pytest

from repgain.adapters.affine_logit import activity_trial_weights, equal_activity_risk, fit_anchored


def _synthetic_logits(seed=0, n=120, k=3):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, size=n)
    z = rng.normal(0, 1, size=(n, k))
    z[np.arange(n), y] += 3.0  # make the backbone's own prediction informative
    w = np.ones(n)
    return z, y, w


def test_identity_anchor_with_tiny_c_recovers_the_identity_map():
    z, y, w = _synthetic_logits()
    w_mat, b, converged = fit_anchored(z, y, w, c=1e-6, anchor="identity")
    assert converged
    assert np.allclose(w_mat, np.eye(3), atol=1e-3)
    assert np.allclose(b, 0.0, atol=1e-3)


def test_zero_anchor_with_tiny_c_shrinks_to_zero():
    z, y, w = _synthetic_logits()
    w_mat, b, converged = fit_anchored(z, y, w, c=1e-6, anchor="zero")
    assert converged
    assert np.allclose(w_mat, 0.0, atol=1e-3)
    assert np.allclose(b, 0.0, atol=1e-3)


def test_larger_c_moves_the_fit_further_from_its_anchor():
    # C sets the inverse penalty strength: a larger C weakens the pull toward
    # the anchor, so the fitted W should drift further from it monotonically.
    z, y, w = _synthetic_logits(seed=1, n=600)
    dists = []
    for c in (0.1, 1.0, 10.0):
        w_mat, _, ok = fit_anchored(z, y, w, c=c, anchor="identity")
        assert ok
        dists.append(np.linalg.norm(w_mat - np.eye(3)))
    assert dists[0] < dists[1] < dists[2]


def test_fit_anchored_is_deterministic():
    z, y, w = _synthetic_logits(seed=2)
    w1, b1, _ = fit_anchored(z, y, w, c=1.0, anchor="identity")
    w2, b2, _ = fit_anchored(z, y, w, c=1.0, anchor="identity")
    assert np.array_equal(w1, w2)
    assert np.array_equal(b1, b2)


def _tiny_frame():
    rows = []
    for subj in (1, 2):
        for act, n_trials in (("a", 2), ("b", 1)):
            for trial in range(n_trials):
                for _ in range(3):  # 3 windows per trial
                    rows.append(dict(subject_id=subj, activity=act, trial=trial, label_id=0))
    return pd.DataFrame(rows)


def test_activity_trial_weights_sum_to_number_of_rows():
    frame = _tiny_frame()
    w = activity_trial_weights(frame)
    assert w.sum() == pytest.approx(len(frame))


def test_activity_trial_weights_balance_across_trials_within_activity():
    # activity "a" has 2 trials (6 rows total), activity "b" has 1 trial (3 rows);
    # after balancing, each activity should carry equal total weight per subject.
    frame = _tiny_frame()
    frame["w"] = activity_trial_weights(frame)
    per_activity = frame[frame.subject_id == 1].groupby("activity")["w"].sum()
    assert per_activity["a"] == pytest.approx(per_activity["b"])


def test_equal_activity_risk_is_zero_for_perfect_predictions():
    frame = _tiny_frame()
    pred = frame["label_id"].to_numpy()
    assert equal_activity_risk(frame, pred) == 0.0


def test_equal_activity_risk_is_one_when_everything_is_wrong():
    frame = _tiny_frame()
    pred = frame["label_id"].to_numpy() + 1
    assert equal_activity_risk(frame, pred) == 1.0
