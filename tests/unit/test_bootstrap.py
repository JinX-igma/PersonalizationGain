import numpy as np
import pytest

from repgain.bootstrap import participant_bootstrap


def _mean_statistic(data):
    def statistic(picked_subjects):
        return float(np.mean([data[s] for s in picked_subjects]))

    return statistic


def test_same_seed_is_bitwise_reproducible():
    data = {s: float(s) for s in range(20)}
    r1 = participant_bootstrap(list(data), _mean_statistic(data), seed=0, n_boot=500)
    r2 = participant_bootstrap(list(data), _mean_statistic(data), seed=0, n_boot=500)
    assert np.array_equal(r1.draws, r2.draws)
    assert r1.lo == r2.lo and r1.hi == r2.hi


def test_different_seed_gives_different_draws():
    data = {s: float(s) for s in range(20)}
    r1 = participant_bootstrap(list(data), _mean_statistic(data), seed=0, n_boot=500)
    r2 = participant_bootstrap(list(data), _mean_statistic(data), seed=1, n_boot=500)
    assert not np.array_equal(r1.draws, r2.draws)


def test_ci_brackets_estimate_for_symmetric_data():
    rng = np.random.default_rng(42)
    data = {s: float(v) for s, v in enumerate(rng.normal(0, 1, size=200))}
    result = participant_bootstrap(list(data), _mean_statistic(data), seed=0, n_boot=2000)
    assert result.lo <= result.estimate <= result.hi


def test_requires_at_least_one_subject():
    with pytest.raises(ValueError):
        participant_bootstrap([], lambda picked: 0.0, seed=0, n_boot=10)


def test_constant_statistic_gives_zero_width_ci():
    data = {s: 1.0 for s in range(10)}
    result = participant_bootstrap(list(data), _mean_statistic(data), seed=0, n_boot=200)
    assert result.lo == pytest.approx(1.0)
    assert result.hi == pytest.approx(1.0)
