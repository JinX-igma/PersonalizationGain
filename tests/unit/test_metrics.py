import numpy as np
import pytest

from repgain.metrics import (
    episode_repeatability_sigma,
    median_abs,
    reference_half_widths,
    sign_flip_rate,
)


def test_sigma_is_zero_when_episodes_agree_exactly():
    diffs = np.zeros(10)
    assert episode_repeatability_sigma(diffs) == 0.0


def test_sigma_matches_closed_form():
    diffs = np.array([0.1, -0.1, 0.2, -0.2])
    expected = np.sqrt(np.mean(diffs**2) / 2)
    assert episode_repeatability_sigma(diffs) == pytest.approx(expected)


def test_sigma_empty_input_is_nan():
    assert np.isnan(episode_repeatability_sigma(np.array([])))


def test_sign_flip_rate_counts_disagreements_only():
    a = np.array([0.1, 0.1, -0.1, 0.0])
    b = np.array([0.2, -0.2, -0.2, 0.0])
    # cell 0: both positive -> agree; cell 1: + vs - -> flip; cell 2: both negative -> agree
    # cell 3: both exactly zero -> excluded from denominator
    rate, n_flipped, n_eligible = sign_flip_rate(a, b)
    assert n_eligible == 3
    assert n_flipped == 1
    assert rate == pytest.approx(1 / 3)


def test_sign_flip_rate_all_excluded_is_nan():
    a = np.zeros(3)
    b = np.zeros(3)
    rate, n_flipped, n_eligible = sign_flip_rate(a, b)
    assert np.isnan(rate)
    assert n_flipped == 0 and n_eligible == 0


def test_reference_half_widths_ordering_and_ratio():
    inner, outer = reference_half_widths(0.1, n_independent_blocks=2)
    assert inner < outer
    assert outer == pytest.approx(1.96 * 0.1)
    assert inner == pytest.approx(outer / np.sqrt(2))


def test_reference_half_widths_more_blocks_tightens_inner():
    inner_2, outer = reference_half_widths(0.1, n_independent_blocks=2)
    inner_4, _ = reference_half_widths(0.1, n_independent_blocks=4)
    assert inner_4 < inner_2


def test_median_abs():
    assert median_abs(np.array([-3.0, 1.0, 2.0])) == pytest.approx(2.0)
