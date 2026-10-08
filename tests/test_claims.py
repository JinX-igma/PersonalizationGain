"""Regenerate every macro in `paper/claims.yaml` from the committed
`results/` cache and check it against the value actually published in the
paper. This is the executable version of the claim-evidence map: if someone
edits an analysis script and a reported number silently changes, this test
fails in CI.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "paper" / "claims.yaml"
sys.path.insert(0, str(ROOT / "analysis"))


@pytest.fixture(scope="module")
def generated_macros() -> dict:
    # Run the Stage 2 chain in dependency order, exactly as `make figures` does.
    covariance_check = importlib.import_module("covariance_check")
    figure1b = importlib.import_module("figure1b")
    make_macros = importlib.import_module("make_macros")

    covariance_check.main()
    figure1b.main()
    return make_macros.main()


def _claims() -> list[dict]:
    return yaml.safe_load(CLAIMS_PATH.read_text())["claims"]


def _check_one(macro_name: str, expected, tolerance, macros: dict):
    assert macro_name in macros, f"macro {macro_name!r} was not produced by analysis/make_macros.py"
    actual = macros[macro_name]
    assert actual == pytest.approx(expected, abs=tolerance), (
        f"{macro_name}: expected {expected} (+/- {tolerance}), got {actual}"
    )


@pytest.mark.parametrize("claim", _claims(), ids=lambda c: c["id"])
def test_claim_matches_generated_macro(claim: dict, generated_macros: dict):
    macro = claim["macro"]
    expected = claim["expected"]
    tolerance = claim["tolerance"]

    if isinstance(macro, list):
        tolerances = tolerance if isinstance(tolerance, list) else [tolerance] * len(macro)
        for name, exp, tol in zip(macro, expected, tolerances):
            _check_one(name, exp, tol, generated_macros)
    else:
        _check_one(macro, expected, tolerance, generated_macros)


def test_every_claim_has_a_unique_id():
    ids = [c["id"] for c in _claims()]
    assert len(ids) == len(set(ids))
