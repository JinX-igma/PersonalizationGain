import numpy as np
import pandas as pd
import pytest

from repgain import RepeatabilityCheck


def _two_episode_df(gains_by_subject, group="act"):
    rows = []
    for subject, (g1, g2) in gains_by_subject.items():
        rows.append(dict(subject=subject, group=group, episode=1, gain=g1))
        rows.append(dict(subject=subject, group=group, episode=2, gain=g2))
    return pd.DataFrame(rows)


def test_identical_repeated_episodes_give_zero_repeatability_sigma():
    df = _two_episode_df({s: (0.05, 0.05) for s in range(10)})
    report = RepeatabilityCheck(n_boot=200, seed=0).run(df)
    assert report.sigma == 0.0
    assert report.flip_rate == 0.0


def test_sigma_matches_hand_computed_value():
    gains = {1: (0.10, -0.10), 2: (0.20, 0.00), 3: (0.05, 0.05)}
    df = _two_episode_df(gains)
    report = RepeatabilityCheck(n_boot=200, seed=0).run(df)
    diffs = np.array([0.10 - (-0.10), 0.20 - 0.00, 0.05 - 0.05])
    expected = np.sqrt(np.mean(diffs**2) / 2)
    assert report.sigma == pytest.approx(expected)
    assert report.n_pairs == 3


def test_run_is_deterministic_given_the_same_seed():
    rng = np.random.default_rng(0)
    gains = {s: tuple(rng.normal(0, 0.1, size=2)) for s in range(15)}
    df = _two_episode_df(gains)
    r1 = RepeatabilityCheck(n_boot=500, seed=7).run(df)
    r2 = RepeatabilityCheck(n_boot=500, seed=7).run(df)
    assert r1.sigma_ci.lo == r2.sigma_ci.lo
    assert r1.sigma_ci.hi == r2.sigma_ci.hi


def test_condition_score_input_matches_precomputed_gain_input():
    gains = {1: (0.10, -0.10), 2: (0.20, 0.00)}
    gain_df = _two_episode_df(gains)

    rows = []
    for subject, (g1, g2) in gains.items():
        for episode, gain in ((1, g1), (2, g2)):
            rows.append(
                dict(subject=subject, group="act", episode=episode, condition="baseline", score=0.5)
            )
            rows.append(
                dict(
                    subject=subject,
                    group="act",
                    episode=episode,
                    condition="personalized",
                    score=0.5 + gain,
                )
            )
    score_df = pd.DataFrame(rows)

    check = RepeatabilityCheck(n_boot=50, seed=0)
    r_gain = check.run(gain_df)
    r_score = check.run(score_df)
    assert r_gain.sigma == pytest.approx(r_score.sigma)


def test_raises_when_no_cell_has_two_episodes():
    df = pd.DataFrame(
        [dict(subject=1, group="act", episode=1, gain=0.1), dict(subject=2, group="act", episode=1, gain=0.2)]
    )
    with pytest.raises(ValueError):
        RepeatabilityCheck().run(df)


def test_summary_and_plot_do_not_raise():
    gains = {s: (0.05, -0.05) for s in range(6)}
    df = _two_episode_df(gains)
    report = RepeatabilityCheck(n_boot=100, seed=0).run(df)
    text = report.summary()
    assert "episode repeatability sigma" in text
    fig = report.plot()
    assert fig is not None
