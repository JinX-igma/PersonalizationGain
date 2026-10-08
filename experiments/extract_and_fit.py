#!/usr/bin/env python3
"""Decision-aware MotionSense audit with disjoint enrollment and future data.

This is an independent result line.  It never writes any pre-existing
``motionsense_loto_*`` artifact.  For each target user, nine designated long
recordings form the labeled enrollment set.  The frozen population classifier
is evaluated on those recordings to obtain activity-balanced enrollment error,
and the fixed 6x6+bias logit corrector is fit only on those recordings.  Six
designated short recordings are then used exactly once to estimate future
baseline risk, corrected risk, and their signed difference.

Population-model fitting is subject-disjoint.  Corrector regularization is
selected only from development subjects under the same enrollment-to-future
protocol and locked before target outcomes are evaluated.  Each 12-user fleet
is audited with exact enumeration of all size-3 and size-6 subsets.  An optional
gender-stratified repeated-split analysis performs a complete refit and fresh
development-only selection for every partition; it is a split sensitivity,
not a population confidence interval.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import softmax
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DEFAULT_REPO_ROOT = PROJECT_ROOT / "data" / "motion-sense"
DEFAULT_RUN_NAME = "motionsense_decision_audit"
FORBIDDEN_RUN_NAMES = {
    "motionsense_primary",
    "motionsense_pca_sensitivity",
    "motionsense_loto_primary",
    "motionsense_loto_long_to_short",
}
PRIMARY_SEED = 20260825
PRIMARY_WINDOW_SAMPLES = 50
PRIMARY_STEP_SAMPLES = 50
PRIMARY_TRIM_SAMPLES = 100
PRIMARY_EXPECTED_WINDOWS = 26640
BACKBONE_C = 1.0
DEFAULT_C_VALUES = (0.01, 0.1, 1.0, 10.0)
MAX_ITER = 5000
SENSITIVITY_SEED = 20260826

ACTIVITIES = ("dws", "ups", "sit", "std", "wlk", "jog")
ACTIVITY_TO_ID = {activity: index for index, activity in enumerate(ACTIVITIES)}
ACTIVITY_TRIALS = {
    "dws": (1, 2, 11),
    "ups": (3, 4, 12),
    "sit": (5, 13),
    "std": (6, 14),
    "wlk": (7, 8, 15),
    "jog": (9, 16),
}
LONG_TRIALS = {
    "dws": (1, 2),
    "ups": (3, 4),
    "sit": (5,),
    "std": (6,),
    "wlk": (7, 8),
    "jog": (9,),
}
SHORT_TRIALS = {
    "dws": (11,),
    "ups": (12,),
    "sit": (13,),
    "std": (14,),
    "wlk": (15,),
    "jog": (16,),
}

# Generated once with StratifiedKFold(n_splits=2, shuffle=True,
# random_state=20260825), stratifying data_subjects_info.csv gender. The IDs
# are hard-coded so later sklearn changes cannot silently alter the protocol.
OUTER_FOLDS = {
    "fold_1": {
        "development": (2, 5, 7, 9, 10, 11, 12, 13, 14, 16, 18, 21),
        "target": (1, 3, 4, 6, 8, 15, 17, 19, 20, 22, 23, 24),
    },
    "fold_2": {
        "development": (1, 3, 4, 6, 8, 15, 17, 19, 20, 22, 23, 24),
        "target": (2, 5, 7, 9, 10, 11, 12, 13, 14, 16, 18, 21),
    },
}

# Each development subject is inner validation exactly once. These IDs were
# fixed before any target outcome was observed.
INNER_VALIDATION = {
    "fold_1": (
        (7, 11, 13, 14),
        (9, 10, 18, 21),
        (2, 5, 12, 16),
    ),
    "fold_2": (
        (1, 3, 6, 22),
        (4, 15, 19, 24),
        (8, 17, 20, 23),
    ),
}

SENSOR_COLUMNS = (
    "attitude.roll",
    "attitude.pitch",
    "attitude.yaw",
    "gravity.x",
    "gravity.y",
    "gravity.z",
    "rotationRate.x",
    "rotationRate.y",
    "rotationRate.z",
    "userAcceleration.x",
    "userAcceleration.y",
    "userAcceleration.z",
)
VECTOR_GROUPS = {
    "gravity_mag": ("gravity.x", "gravity.y", "gravity.z"),
    "rotation_mag": ("rotationRate.x", "rotationRate.y", "rotationRate.z"),
    "acceleration_mag": (
        "userAcceleration.x",
        "userAcceleration.y",
        "userAcceleration.z",
    ),
}
TRIAL_PATTERN = re.compile(r"^(?P<activity>[a-z]+)_(?P<trial>\d+)$")
SUBJECT_PATTERN = re.compile(r"^sub_(?P<subject>\d+)\.csv$")


@dataclass(frozen=True)
class CandidateSpec:
    kind: str
    c_value: float | None

    @property
    def name(self) -> str:
        if self.kind == "identity":
            return "identity"
        return f"linear_logits_C={self.c_value:g}"


@dataclass
class BackboneBundle:
    scaler: StandardScaler
    model: LogisticRegression
    n_iter: int
    training_subjects: tuple[int, ...]
    training_windows: int
    state_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--window-samples", type=int, default=PRIMARY_WINDOW_SAMPLES)
    parser.add_argument("--step-samples", type=int, default=PRIMARY_STEP_SAMPLES)
    parser.add_argument("--trim-samples", type=int, default=PRIMARY_TRIM_SAMPLES)
    parser.add_argument("--c-values", type=float, nargs="+", default=list(DEFAULT_C_VALUES))
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument(
        "--split-sensitivity",
        type=int,
        default=100,
        help="Number of deterministic gender-stratified 12/12 refits (0 skips).",
    )
    parser.add_argument("--sensitivity-seed", type=int, default=SENSITIVITY_SEED)
    parser.add_argument(
        "--sensitivity-c-values",
        type=float,
        nargs="+",
        default=None,
        help="Optional independent candidate grid for split sensitivity.",
    )
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def model_state_digest(scaler: StandardScaler, model: LogisticRegression) -> str:
    return array_digest(
        np.asarray(scaler.mean_),
        np.asarray(scaler.scale_),
        np.asarray(model.coef_),
        np.asarray(model.intercept_),
        np.asarray(model.classes_),
    )


def serializable_environment() -> dict[str, str]:
    return {
        "python": sys.version,
        "operating_system": platform.system(),
        "operating_system_release": platform.release(),
        "machine_architecture": platform.machine(),
        "processor": platform.processor() or "not_reported",
        "logical_cpu_count": str(os.cpu_count()),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": plt.matplotlib.__version__,
    }


def json_safe(value: object) -> object:
    """Convert numpy/pandas scalars and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def repository_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def feature_names() -> list[str]:
    scalar_statistics = (
        "mean",
        "std",
        "min",
        "max",
        "median",
        "q25",
        "q75",
        "rms",
        "madiff",
        "stddiff",
    )
    magnitude_statistics = ("mean", "std", "min", "max", "median", "rms", "madiff")
    names = [
        f"{column}__{statistic}"
        for column in SENSOR_COLUMNS
        for statistic in scalar_statistics
    ]
    names.extend(
        f"{group_name}__{statistic}"
        for group_name in VECTOR_GROUPS
        for statistic in magnitude_statistics
    )
    return names


def summarize_window(values: np.ndarray, column_index: dict[str, int]) -> np.ndarray:
    differences = np.diff(values, axis=0)
    scalar = np.column_stack(
        [
            values.mean(axis=0),
            values.std(axis=0),
            values.min(axis=0),
            values.max(axis=0),
            np.median(values, axis=0),
            np.quantile(values, 0.25, axis=0),
            np.quantile(values, 0.75, axis=0),
            np.sqrt(np.mean(np.square(values), axis=0)),
            np.mean(np.abs(differences), axis=0),
            differences.std(axis=0),
        ]
    ).reshape(-1)
    magnitude_features: list[float] = []
    for columns in VECTOR_GROUPS.values():
        indices = [column_index[column] for column in columns]
        magnitude = np.linalg.norm(values[:, indices], axis=1)
        magnitude_difference = np.diff(magnitude)
        magnitude_features.extend(
            [
                float(magnitude.mean()),
                float(magnitude.std()),
                float(magnitude.min()),
                float(magnitude.max()),
                float(np.median(magnitude)),
                float(np.sqrt(np.mean(np.square(magnitude)))),
                float(np.mean(np.abs(magnitude_difference))),
            ]
        )
    return np.concatenate([scalar, np.asarray(magnitude_features, dtype=float)])


def source_csvs(data_root: Path) -> list[Path]:
    paths = sorted(data_root.glob("*/*.csv"))
    if len(paths) != 360:
        raise ValueError(f"Expected 360 MotionSense recordings, found {len(paths)}")
    return paths


def source_tree_sha256(paths: Sequence[Path], relative_root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(relative_root)).encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


def extract_window_features(
    data_root: Path,
    window_samples: int,
    step_samples: int,
    trim_samples: int,
) -> pd.DataFrame:
    names = feature_names()
    column_index = {column: index for index, column in enumerate(SENSOR_COLUMNS)}
    rows: list[dict[str, object]] = []
    observed_sessions: set[tuple[int, str, int]] = set()

    for path in source_csvs(data_root):
        trial_match = TRIAL_PATTERN.match(path.parent.name)
        subject_match = SUBJECT_PATTERN.match(path.name)
        if trial_match is None or subject_match is None:
            raise ValueError(f"Unexpected MotionSense source path: {path}")
        activity = trial_match.group("activity")
        trial = int(trial_match.group("trial"))
        subject_id = int(subject_match.group("subject"))
        if activity not in ACTIVITY_TO_ID or trial not in ACTIVITY_TRIALS[activity]:
            raise ValueError(f"Unexpected activity/trial in {path}")
        session_tuple = (subject_id, activity, trial)
        if session_tuple in observed_sessions:
            raise AssertionError(f"Duplicate source recording: {session_tuple}")
        observed_sessions.add(session_tuple)

        raw = pd.read_csv(path)
        missing = [column for column in SENSOR_COLUMNS if column not in raw]
        if missing:
            raise ValueError(f"Missing sensor columns in {path}: {missing}")
        source_index = raw.iloc[:, 0].to_numpy(dtype=int)
        if not np.array_equal(source_index, np.arange(len(raw), dtype=int)):
            raise ValueError(f"Non-contiguous source row index in {path}")
        values = raw[list(SENSOR_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite sensor value in {path}")
        if len(values) - 2 * trim_samples < window_samples:
            raise ValueError(f"Recording too short after trim: {path}")

        session_key = f"sub_{subject_id:02d}/{activity}_{trial}"
        source_file = str(path.relative_to(data_root.parents[1]))
        usable_end = len(values) - trim_samples
        for window_index, raw_start in enumerate(
            range(trim_samples, usable_end - window_samples + 1, step_samples)
        ):
            raw_end = raw_start + window_samples
            summary = summarize_window(values[raw_start:raw_end], column_index)
            window_id = f"{session_key}/{raw_start}:{raw_end}"
            row: dict[str, object] = {
                "window_id": window_id,
                "session_key": session_key,
                "subject_id": subject_id,
                "activity": activity,
                "label_id": ACTIVITY_TO_ID[activity],
                "trial": trial,
                "window_index": window_index,
                "raw_start": raw_start,
                "raw_end_exclusive": raw_end,
                "source_rows": len(values),
                "source_file": source_file,
            }
            row.update(dict(zip(names, summary, strict=True)))
            rows.append(row)

    frame = pd.DataFrame(rows)
    validate_feature_frame(frame, names, window_samples, step_samples, trim_samples)
    return frame


def validate_feature_frame(
    frame: pd.DataFrame,
    names: Sequence[str],
    window_samples: int,
    step_samples: int,
    trim_samples: int,
) -> None:
    expected_subjects = set(range(1, 25))
    if set(frame["subject_id"].astype(int)) != expected_subjects:
        raise AssertionError("Feature frame does not contain subjects 1 through 24")
    if frame["window_id"].duplicated().any():
        raise AssertionError("Duplicate window_id values")
    sessions = frame[["subject_id", "activity", "trial", "session_key"]].drop_duplicates()
    if len(sessions) != 360:
        raise AssertionError(f"Expected 360 sessions, observed {len(sessions)}")
    if not (sessions.groupby("subject_id").size() == 15).all():
        raise AssertionError("Every subject must have exactly 15 sessions")
    for activity, trials in ACTIVITY_TRIALS.items():
        observed = set(sessions.loc[sessions.activity == activity, "trial"].astype(int))
        if observed != set(trials):
            raise AssertionError(f"Trial mismatch for {activity}: {sorted(observed)}")
    values = frame[list(names)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("Feature frame contains non-finite values")
    if not (frame["raw_end_exclusive"] - frame["raw_start"] == window_samples).all():
        raise AssertionError("Window length mismatch")
    if not (frame["raw_start"] >= trim_samples).all():
        raise AssertionError("A window begins inside the leading trim")
    if not (frame["raw_end_exclusive"] <= frame["source_rows"] - trim_samples).all():
        raise AssertionError("A window ends inside the trailing trim")
    for _, group in frame.groupby("session_key", sort=False):
        starts = np.sort(group["raw_start"].to_numpy(dtype=int))
        if len(starts) > 1 and not np.all(np.diff(starts) == step_samples):
            raise AssertionError(f"Non-uniform stride in {group.session_key.iloc[0]}")
    if (
        window_samples == PRIMARY_WINDOW_SAMPLES
        and step_samples == PRIMARY_STEP_SAMPLES
        and trim_samples == PRIMARY_TRIM_SAMPLES
        and len(frame) != PRIMARY_EXPECTED_WINDOWS
    ):
        raise AssertionError(
            f"Primary source drift: expected {PRIMARY_EXPECTED_WINDOWS} windows, found {len(frame)}"
        )


def load_or_build_features(
    args: argparse.Namespace,
    source_tree_digest: str,
) -> tuple[pd.DataFrame, list[str], Path, Path]:
    data_root = args.repo_root / "data" / "A_DeviceMotion_data"
    cache = PROJECT_ROOT / "results" / (
        f"motionsense_decision_windows_w{args.window_samples}_s{args.step_samples}_t{args.trim_samples}.csv"
    )
    provenance_path = cache.with_suffix(".provenance.json")
    names = feature_names()
    expected_provenance = {
        "extractor_version": "motionsense_decision_statistical_features_v1",
        "source_tree_sha256": source_tree_digest,
        "window_samples": args.window_samples,
        "step_samples": args.step_samples,
        "trim_samples_each_edge": args.trim_samples,
        "feature_names_sha256": hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest(),
    }
    if cache.exists() and not args.rebuild_features:
        if not provenance_path.exists():
            raise ValueError(
                f"Feature cache lacks bound provenance: {provenance_path}. Re-run with --rebuild-features."
            )
        observed_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if observed_provenance != expected_provenance:
            raise ValueError(
                "Feature cache provenance does not match the source tree/protocol; "
                "re-run with --rebuild-features"
            )
        frame = pd.read_csv(cache)
        validate_feature_frame(
            frame,
            names,
            args.window_samples,
            args.step_samples,
            args.trim_samples,
        )
    else:
        frame = extract_window_features(
            data_root,
            args.window_samples,
            args.step_samples,
            args.trim_samples,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache, index=False)
        provenance_path.write_text(
            json.dumps(expected_provenance, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        # Bind first-run numerics to the same serialized representation used by
        # every reproduction.  Otherwise CSV round-trip rounding can alter a
        # near-tied enrollment-error ranking between the first and later runs.
        frame = pd.read_csv(cache)
        validate_feature_frame(
            frame,
            names,
            args.window_samples,
            args.step_samples,
            args.trim_samples,
        )
    return frame, names, cache, provenance_path


def validate_fixed_splits(subject_info: pd.DataFrame) -> None:
    all_subjects = set(range(1, 25))
    first_dev = set(OUTER_FOLDS["fold_1"]["development"])
    first_target = set(OUTER_FOLDS["fold_1"]["target"])
    second_dev = set(OUTER_FOLDS["fold_2"]["development"])
    second_target = set(OUTER_FOLDS["fold_2"]["target"])
    if first_dev | first_target != all_subjects or first_dev & first_target:
        raise AssertionError("fold_1 outer split is not a partition")
    if second_dev | second_target != all_subjects or second_dev & second_target:
        raise AssertionError("fold_2 outer split is not a partition")
    if first_dev != second_target or first_target != second_dev:
        raise AssertionError("Outer 2-fold partitions are not complementary")
    if set(subject_info["code"].astype(int)) != all_subjects:
        raise AssertionError("Subject metadata does not contain IDs 1 through 24")
    gender = subject_info.set_index("code")["gender"].astype(int)
    for fold_name, split in OUTER_FOLDS.items():
        for role in ("development", "target"):
            counts = gender.loc[list(split[role])].value_counts().to_dict()
            if counts != {1: 7, 0: 5}:
                raise AssertionError(f"{fold_name} {role} is not gender balanced 7/5: {counts}")
        validation_groups = [set(group) for group in INNER_VALIDATION[fold_name]]
        if set.union(*validation_groups) != set(split["development"]):
            raise AssertionError(f"{fold_name} inner validation does not cover development subjects")
        if sum(len(group) for group in validation_groups) != len(set.union(*validation_groups)):
            raise AssertionError(f"{fold_name} inner validation groups overlap")


def activity_trial_weights(frame: pd.DataFrame) -> np.ndarray:
    """Equalize subjects, activities, trials within activity, then windows."""
    keys = frame[["subject_id", "activity", "trial"]].copy()
    session_counts = keys.groupby(["subject_id", "activity", "trial"]).size()
    trial_counts = (
        keys.drop_duplicates()
        .groupby(["subject_id", "activity"])
        .size()
    )
    activity_counts = keys.drop_duplicates(["subject_id", "activity"]).groupby("subject_id").size()
    subject_count = int(keys["subject_id"].nunique())
    weights = np.empty(len(frame), dtype=float)
    for position, row in enumerate(keys.itertuples(index=False)):
        denominator = (
            subject_count
            * int(activity_counts.loc[row.subject_id])
            * int(trial_counts.loc[(row.subject_id, row.activity)])
            * int(session_counts.loc[(row.subject_id, row.activity, row.trial)])
        )
        weights[position] = 1.0 / denominator
    weights *= len(weights) / weights.sum()
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise AssertionError("Invalid activity/trial inverse weights")
    return weights


def decision_logits(model: LogisticRegression, inputs: np.ndarray) -> np.ndarray:
    logits = np.asarray(model.decision_function(inputs), dtype=float)
    if logits.ndim == 1:
        logits = np.column_stack([-logits, logits])
    return logits


def labels_from_logits(logits: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return np.asarray(classes)[np.argmax(logits, axis=1)]


def fit_backbone(
    training_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    training_subjects: Sequence[int],
    seed: int,
) -> BackboneBundle:
    actual_subjects = tuple(sorted(training_frame["subject_id"].unique().astype(int)))
    expected_subjects = tuple(sorted(map(int, training_subjects)))
    if actual_subjects != expected_subjects:
        raise AssertionError(
            f"Backbone subject mismatch: actual={actual_subjects}, expected={expected_subjects}"
        )
    x = training_frame[list(feature_columns)].to_numpy(dtype=float)
    y = training_frame["label_id"].to_numpy(dtype=int)
    weights = activity_trial_weights(training_frame)
    scaler = StandardScaler()
    scaler.fit(x, sample_weight=weights)
    scaled = scaler.transform(x)
    model = LogisticRegression(
        C=BACKBONE_C,
        solver="lbfgs",
        l1_ratio=0.0,
        max_iter=MAX_ITER,
        tol=1e-6,
        random_state=seed,
    )
    model.fit(scaled, y, sample_weight=weights)
    n_iter = int(np.max(model.n_iter_))
    if n_iter >= model.max_iter:
        raise RuntimeError(f"Backbone failed to converge: n_iter={model.n_iter_.tolist()}")
    if set(model.classes_.astype(int)) != set(ACTIVITY_TO_ID.values()):
        raise AssertionError("Backbone did not observe all six activities")
    return BackboneBundle(
        scaler=scaler,
        model=model,
        n_iter=n_iter,
        training_subjects=expected_subjects,
        training_windows=len(training_frame),
        state_sha256=model_state_digest(scaler, model),
    )


def fit_linear_corrector(
    logits: np.ndarray,
    labels: np.ndarray,
    fit_frame: pd.DataFrame,
    c_value: float,
    seed: int,
) -> tuple[LogisticRegression, StandardScaler, int, np.ndarray, np.ndarray]:
    weights = activity_trial_weights(fit_frame)
    logit_scaler = StandardScaler()
    logit_scaler.fit(logits, sample_weight=weights)
    scaled_logits = logit_scaler.transform(logits)
    model = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        l1_ratio=0.0,
        max_iter=MAX_ITER,
        tol=1e-6,
        random_state=seed,
    )
    model.fit(scaled_logits, labels, sample_weight=weights)
    n_iter = int(np.max(model.n_iter_))
    if n_iter >= model.max_iter:
        raise RuntimeError(
            f"Corrector failed to converge: C={c_value}, n_iter={model.n_iter_.tolist()}"
        )
    if set(model.classes_.astype(int)) != set(ACTIVITY_TO_ID.values()):
        raise AssertionError("Corrector fit did not observe all six activities")
    # Absorb fit-only standardization into the learned affine map. Deployment
    # remains exactly one raw-logit 6x6+bias map (42 parameters).
    raw_coef = np.asarray(model.coef_) / np.asarray(logit_scaler.scale_)[None, :]
    raw_intercept = np.asarray(model.intercept_) - raw_coef @ np.asarray(logit_scaler.mean_)
    return model, logit_scaler, n_iter, raw_coef, raw_intercept


def corrector_specs(c_values: Iterable[float]) -> list[CandidateSpec]:
    values = sorted(set(float(value) for value in c_values))
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("Every corrector C must be finite and positive")
    return [CandidateSpec("identity", None)] + [
        CandidateSpec("linear_logits", value) for value in values
    ]


def recording_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "outer_fold",
        "subject_id",
        "activity",
        "trial",
        "session_key",
        "label_id",
        "baseline_prediction",
        "corrected_prediction",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"Prediction table missing columns: {sorted(missing)}")
    work = predictions.copy()
    work["baseline_incorrect"] = work["baseline_prediction"] != work["label_id"]
    work["corrected_incorrect"] = work["corrected_prediction"] != work["label_id"]
    grouped = work.groupby(
        ["outer_fold", "subject_id", "activity", "trial", "session_key"],
        as_index=False,
        sort=True,
    ).agg(
        n_windows=("window_id", "size"),
        candidate_name=("candidate_name", "first"),
        candidate_c=("candidate_c", "first"),
        baseline_error=("baseline_incorrect", "mean"),
        corrected_error=("corrected_incorrect", "mean"),
    )
    grouped["signed_gain"] = grouped["baseline_error"] - grouped["corrected_error"]
    return grouped


def user_metrics_from_recordings(
    recordings: pd.DataFrame,
    expected_recordings: int | None = None,
) -> pd.DataFrame:
    # Trials are equal within an activity; activities are equal within a user.
    activity = recordings.groupby(
        ["outer_fold", "subject_id", "activity"], as_index=False, sort=True
    ).agg(
        n_trials=("trial", "nunique"),
        n_windows=("n_windows", "sum"),
        candidate_name=("candidate_name", "first"),
        candidate_c=("candidate_c", "first"),
        baseline_error=("baseline_error", "mean"),
        corrected_error=("corrected_error", "mean"),
    )
    users = activity.groupby(["outer_fold", "subject_id"], as_index=False, sort=True).agg(
        n_activities=("activity", "nunique"),
        n_recordings=("n_trials", "sum"),
        n_windows=("n_windows", "sum"),
        candidate_name=("candidate_name", "first"),
        candidate_c=("candidate_c", "first"),
        baseline_error=("baseline_error", "mean"),
        corrected_error=("corrected_error", "mean"),
    )
    if not (users["n_activities"] == 6).all():
        raise AssertionError("User metric does not contain all six activities")
    if expected_recordings is not None and not (users["n_recordings"] == expected_recordings).all():
        raise AssertionError(
            f"User metric does not contain expected {expected_recordings} recordings"
        )
    users["baseline_accuracy"] = 1.0 - users["baseline_error"]
    users["corrected_accuracy"] = 1.0 - users["corrected_error"]
    users["signed_gain"] = users["baseline_error"] - users["corrected_error"]
    users["positive_gain"] = users["signed_gain"].clip(lower=0.0)
    users["rho_signed"] = np.where(
        users["baseline_error"] > 0,
        users["signed_gain"] / users["baseline_error"],
        np.nan,
    )
    return users


def evaluate_subject_loto(
    outer_fold: str,
    subject_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    backbone: BackboneBundle,
    candidate: CandidateSpec,
    seed: int,
    collect_predictions: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subject_ids = subject_frame["subject_id"].unique().astype(int)
    if len(subject_ids) != 1:
        raise AssertionError("LOTO evaluator received more than one subject")
    subject_id = int(subject_ids[0])
    if subject_id in backbone.training_subjects:
        raise AssertionError(f"Subject {subject_id} leaked into the backbone training set")
    session_keys = sorted(subject_frame["session_key"].unique().tolist())
    if len(session_keys) != 15:
        raise AssertionError(f"Subject {subject_id} has {len(session_keys)} sessions, expected 15")

    x = subject_frame[list(feature_columns)].to_numpy(dtype=float)
    scaled = backbone.scaler.transform(x)
    base_logits = decision_logits(backbone.model, scaled)
    labels = subject_frame["label_id"].to_numpy(dtype=int)
    baseline_predictions = labels_from_logits(base_logits, backbone.model.classes_)
    all_prediction_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []

    for session_index, heldout_session in enumerate(session_keys):
        test_mask = subject_frame["session_key"].to_numpy() == heldout_session
        fit_mask = ~test_mask
        fit_frame = subject_frame.loc[fit_mask].copy()
        test_frame = subject_frame.loc[test_mask].copy()
        fit_sessions = sorted(fit_frame["session_key"].unique().tolist())
        test_sessions = set(test_frame["session_key"].unique())
        session_overlap = set(fit_sessions).intersection(test_sessions)
        window_overlap = set(fit_frame["window_id"]).intersection(test_frame["window_id"])
        if heldout_session in fit_sessions or session_overlap or window_overlap:
            raise AssertionError(f"LOTO leakage for {subject_id} {heldout_session}")
        if len(fit_sessions) != 14 or test_sessions != {heldout_session}:
            raise AssertionError("LOTO did not produce a 14-session fit and 1-session test")
        if set(fit_frame["activity"]) != set(ACTIVITIES):
            raise AssertionError("LOTO fit set does not contain all activities")

        n_iter = 0
        converged = True
        corrector_state = hashlib.sha256(b"identity").hexdigest()
        residual_l2 = np.zeros(int(test_mask.sum()), dtype=float)
        if candidate.kind == "identity":
            corrected_predictions = baseline_predictions[test_mask]
            corrected_logits = base_logits[test_mask]
            used_fit_windows = 0
            used_fit_sessions = 0
        else:
            assert candidate.c_value is not None
            corrector, logit_scaler, n_iter, raw_coef, raw_intercept = fit_linear_corrector(
                base_logits[fit_mask],
                labels[fit_mask],
                fit_frame,
                candidate.c_value,
                seed + subject_id * 1000 + session_index,
            )
            corrected_logits = decision_logits(
                corrector, logit_scaler.transform(base_logits[test_mask])
            )
            absorbed_logits = base_logits[test_mask] @ raw_coef.T + raw_intercept
            if not np.allclose(corrected_logits, absorbed_logits, rtol=1e-10, atol=1e-10):
                raise AssertionError("Scaled corrector was not exactly absorbed into raw-logit map")
            corrected_predictions = labels_from_logits(corrected_logits, corrector.classes_)
            residual_l2 = np.linalg.norm(corrected_logits - base_logits[test_mask], axis=1)
            used_fit_windows = int(fit_mask.sum())
            used_fit_sessions = len(fit_sessions)
            corrector_state = array_digest(
                raw_coef,
                raw_intercept,
                np.asarray(corrector.classes_),
            )

        part = test_frame[
            [
                "window_id",
                "session_key",
                "subject_id",
                "activity",
                "label_id",
                "trial",
                "window_index",
                "raw_start",
                "raw_end_exclusive",
                "source_file",
            ]
        ].copy()
        part.insert(0, "outer_fold", outer_fold)
        part["candidate_name"] = candidate.name
        part["candidate_c"] = candidate.c_value
        part["baseline_prediction"] = baseline_predictions[test_mask]
        part["corrected_prediction"] = corrected_predictions
        part["baseline_confidence"] = softmax(base_logits[test_mask], axis=1).max(axis=1)
        part["corrected_confidence"] = softmax(corrected_logits, axis=1).max(axis=1)
        part["residual_l2"] = residual_l2
        if collect_predictions:
            all_prediction_parts.append(part)
        else:
            # Metrics still require predictions, but tuning need not retain all
            # window-level metadata after this subject is scored.
            all_prediction_parts.append(part)

        fit_session_digest = hashlib.sha256("\n".join(fit_sessions).encode("utf-8")).hexdigest()
        audit_rows.append(
            {
                "outer_fold": outer_fold,
                "subject_id": subject_id,
                "heldout_session": heldout_session,
                "heldout_activity": str(test_frame["activity"].iloc[0]),
                "heldout_trial": int(test_frame["trial"].iloc[0]),
                "heldout_windows": int(test_mask.sum()),
                "eligible_fit_sessions": len(fit_sessions),
                "eligible_fit_windows": int(fit_mask.sum()),
                "used_fit_sessions": used_fit_sessions,
                "used_fit_windows": used_fit_windows,
                "fit_activity_count": int(fit_frame["activity"].nunique()),
                "session_overlap_count": len(session_overlap),
                "window_overlap_count": len(window_overlap),
                "heldout_absent_from_fit": heldout_session not in fit_sessions,
                "candidate_name": candidate.name,
                "candidate_c": candidate.c_value,
                "corrector_parameter_capacity": 42,
                "corrector_converged": converged,
                "corrector_n_iter": n_iter,
                "fit_sessions_sha256": fit_session_digest,
                "corrector_state_sha256": corrector_state,
            }
        )

    predictions = pd.concat(all_prediction_parts, ignore_index=True)
    if len(predictions) != len(subject_frame):
        raise AssertionError("LOTO predictions do not cover every subject window exactly once")
    if predictions["window_id"].duplicated().any():
        raise AssertionError("A LOTO window was predicted more than once")
    recordings = recording_metrics(predictions)
    users = user_metrics_from_recordings(recordings, expected_recordings=15)
    if len(users) != 1:
        raise AssertionError("LOTO evaluator did not produce exactly one user metric")
    return predictions, recordings, pd.DataFrame(audit_rows)


def protocol_trial_mask(frame: pd.DataFrame, trial_map: dict[str, tuple[int, ...]]) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for activity, trials in trial_map.items():
        mask |= (frame["activity"] == activity) & frame["trial"].isin(trials)
    return mask


def evaluate_subject_long_to_short(
    outer_fold: str,
    subject_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    backbone: BackboneBundle,
    candidate: CandidateSpec,
    seed: int,
    collect_predictions: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit one user head on nine long recordings and test six short recordings."""
    subject_ids = subject_frame["subject_id"].unique().astype(int)
    if len(subject_ids) != 1:
        raise AssertionError("Long-to-short evaluator received more than one subject")
    subject_id = int(subject_ids[0])
    if subject_id in backbone.training_subjects:
        raise AssertionError(f"Subject {subject_id} leaked into backbone training")
    if subject_frame.session_key.nunique() != 15:
        raise AssertionError("Long-to-short subject must contain all 15 recordings")

    fit_mask_series = protocol_trial_mask(subject_frame, LONG_TRIALS)
    test_mask_series = protocol_trial_mask(subject_frame, SHORT_TRIALS)
    if (fit_mask_series & test_mask_series).any() or not (fit_mask_series | test_mask_series).all():
        raise AssertionError("Long/short trial maps do not partition the subject recordings")
    fit_mask = fit_mask_series.to_numpy(dtype=bool)
    test_mask = test_mask_series.to_numpy(dtype=bool)
    fit_frame = subject_frame.loc[fit_mask].copy()
    test_frame = subject_frame.loc[test_mask].copy()
    fit_sessions = sorted(fit_frame.session_key.unique().tolist())
    test_sessions = sorted(test_frame.session_key.unique().tolist())
    if len(fit_sessions) != 9 or len(test_sessions) != 6:
        raise AssertionError("Long-to-short must have nine fit and six test recordings")
    if set(fit_frame.activity) != set(ACTIVITIES) or set(test_frame.activity) != set(ACTIVITIES):
        raise AssertionError("Long-to-short fit/test must each contain all six activities")
    session_overlap = set(fit_sessions) & set(test_sessions)
    window_overlap = set(fit_frame.window_id) & set(test_frame.window_id)
    if session_overlap or window_overlap:
        raise AssertionError("Long-to-short recording/window leakage")

    x = subject_frame[list(feature_columns)].to_numpy(dtype=float)
    scaled = backbone.scaler.transform(x)
    base_logits = decision_logits(backbone.model, scaled)
    labels = subject_frame.label_id.to_numpy(dtype=int)
    baseline_predictions = labels_from_logits(base_logits[test_mask], backbone.model.classes_)
    n_iter = 0
    corrector_state = hashlib.sha256(b"identity").hexdigest()
    residual_l2 = np.zeros(int(test_mask.sum()), dtype=float)
    if candidate.kind == "identity":
        corrected_logits = base_logits[test_mask]
        corrected_predictions = baseline_predictions
        used_fit_sessions = 0
        used_fit_windows = 0
    else:
        assert candidate.c_value is not None
        corrector, logit_scaler, n_iter, raw_coef, raw_intercept = fit_linear_corrector(
            base_logits[fit_mask],
            labels[fit_mask],
            fit_frame,
            candidate.c_value,
            seed + subject_id * 1000,
        )
        corrected_logits = decision_logits(
            corrector, logit_scaler.transform(base_logits[test_mask])
        )
        absorbed_logits = base_logits[test_mask] @ raw_coef.T + raw_intercept
        if not np.allclose(corrected_logits, absorbed_logits, rtol=1e-10, atol=1e-10):
            raise AssertionError("Long-to-short corrector absorption failed")
        corrected_predictions = labels_from_logits(corrected_logits, corrector.classes_)
        residual_l2 = np.linalg.norm(corrected_logits - base_logits[test_mask], axis=1)
        used_fit_sessions = len(fit_sessions)
        used_fit_windows = int(fit_mask.sum())
        corrector_state = array_digest(raw_coef, raw_intercept, np.asarray(corrector.classes_))

    part = test_frame[
        [
            "window_id",
            "session_key",
            "subject_id",
            "activity",
            "label_id",
            "trial",
            "window_index",
            "raw_start",
            "raw_end_exclusive",
            "source_file",
        ]
    ].copy()
    part.insert(0, "outer_fold", outer_fold)
    part["candidate_name"] = candidate.name
    part["candidate_c"] = candidate.c_value
    part["baseline_prediction"] = baseline_predictions
    part["corrected_prediction"] = corrected_predictions
    part["baseline_confidence"] = softmax(base_logits[test_mask], axis=1).max(axis=1)
    part["corrected_confidence"] = softmax(corrected_logits, axis=1).max(axis=1)
    part["residual_l2"] = residual_l2
    if len(part) != int(test_mask.sum()) or part.window_id.duplicated().any():
        raise AssertionError("Long-to-short predictions do not cover test windows once")

    fit_session_digest = hashlib.sha256("\n".join(fit_sessions).encode("utf-8")).hexdigest()
    audit_rows = []
    for heldout_session in test_sessions:
        heldout = test_frame[test_frame.session_key == heldout_session]
        audit_rows.append(
            {
                "outer_fold": outer_fold,
                "subject_id": subject_id,
                "heldout_session": heldout_session,
                "heldout_activity": str(heldout.activity.iloc[0]),
                "heldout_trial": int(heldout.trial.iloc[0]),
                "heldout_windows": len(heldout),
                "eligible_fit_sessions": len(fit_sessions),
                "eligible_fit_windows": int(fit_mask.sum()),
                "used_fit_sessions": used_fit_sessions,
                "used_fit_windows": used_fit_windows,
                "fit_activity_count": int(fit_frame.activity.nunique()),
                "session_overlap_count": 0,
                "window_overlap_count": 0,
                "heldout_absent_from_fit": heldout_session not in fit_sessions,
                "candidate_name": candidate.name,
                "candidate_c": candidate.c_value,
                "corrector_parameter_capacity": 42,
                "corrector_converged": True,
                "corrector_n_iter": n_iter,
                "fit_sessions_sha256": fit_session_digest,
                "corrector_state_sha256": corrector_state,
            }
        )
    recordings = recording_metrics(part)
    users = user_metrics_from_recordings(recordings, expected_recordings=6)
    if len(users) != 1 or int(users.n_recordings.iloc[0]) != 6:
        raise AssertionError("Long-to-short did not produce one six-recording user metric")
    return part, recordings, pd.DataFrame(audit_rows)


def evaluate_subject_protocol(
    evaluation_protocol: str,
    outer_fold: str,
    subject_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    backbone: BackboneBundle,
    candidate: CandidateSpec,
    seed: int,
    collect_predictions: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if evaluation_protocol == "loto":
        return evaluate_subject_loto(
            outer_fold,
            subject_frame,
            feature_columns,
            backbone,
            candidate,
            seed,
            collect_predictions,
        )
    if evaluation_protocol == "long_to_short":
        return evaluate_subject_long_to_short(
            outer_fold,
            subject_frame,
            feature_columns,
            backbone,
            candidate,
            seed,
            collect_predictions,
        )
    raise KeyError(evaluation_protocol)


def tune_corrector_on_development(
    outer_fold: str,
    development_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    candidate_specs: Sequence[CandidateSpec],
    seed: int,
    evaluation_protocol: str,
) -> tuple[CandidateSpec, pd.DataFrame, pd.DataFrame]:
    """Select one global candidate without accepting any target data argument."""
    development_ids = set(development_frame["subject_id"].unique().astype(int))
    expected_ids = set(OUTER_FOLDS[outer_fold]["development"])
    if development_ids != expected_ids:
        raise AssertionError("Development tuning frame has unexpected subjects")

    candidate_user_rows: list[dict[str, object]] = []
    backbone_rows: list[dict[str, object]] = []
    for inner_index, validation_ids_tuple in enumerate(INNER_VALIDATION[outer_fold], start=1):
        validation_ids = set(validation_ids_tuple)
        training_ids = sorted(development_ids - validation_ids)
        if validation_ids & set(training_ids) or len(training_ids) != 8:
            raise AssertionError("Inner subject split is not disjoint 8/4")
        inner_train = development_frame[development_frame["subject_id"].isin(training_ids)]
        bundle = fit_backbone(
            inner_train,
            feature_columns,
            training_ids,
            seed + inner_index,
        )
        backbone_rows.append(
            {
                "outer_fold": outer_fold,
                "stage": f"inner_{inner_index}",
                "training_subjects": "|".join(map(str, training_ids)),
                "validation_subjects": "|".join(map(str, sorted(validation_ids))),
                "training_windows": len(inner_train),
                "backbone_n_iter": bundle.n_iter,
                "backbone_converged": bundle.n_iter < MAX_ITER,
                "backbone_state_sha256": bundle.state_sha256,
            }
        )
        for subject_id in sorted(validation_ids):
            subject_frame = development_frame[development_frame["subject_id"] == subject_id]
            if subject_id in bundle.training_subjects:
                raise AssertionError("Inner validation subject leaked into backbone training")
            for candidate in candidate_specs:
                predictions, recordings, loto_audit = evaluate_subject_protocol(
                    evaluation_protocol=evaluation_protocol,
                    outer_fold=f"{outer_fold}_inner_{inner_index}",
                    subject_frame=subject_frame,
                    feature_columns=feature_columns,
                    backbone=bundle,
                    candidate=candidate,
                    seed=seed + inner_index * 100_000,
                    collect_predictions=False,
                )
                expected_recordings = 15 if evaluation_protocol == "loto" else 6
                users = user_metrics_from_recordings(
                    recordings, expected_recordings=expected_recordings
                )
                user = users.iloc[0]
                candidate_user_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "inner_fold": inner_index,
                        "validation_subject_id": subject_id,
                        "candidate_kind": candidate.kind,
                        "candidate_name": candidate.name,
                        "candidate_c": candidate.c_value,
                        "baseline_error": float(user.baseline_error),
                        "corrected_error": float(user.corrected_error),
                        "signed_gain": float(user.signed_gain),
                        "rho_signed": float(user.rho_signed) if pd.notna(user.rho_signed) else None,
                        "evaluation_recordings": len(loto_audit),
                        "corrector_fits": (
                            0
                            if candidate.kind == "identity"
                            else (15 if evaluation_protocol == "loto" else 1)
                        ),
                        "all_session_disjoint": bool(
                            (loto_audit.session_overlap_count == 0).all()
                            and (loto_audit.window_overlap_count == 0).all()
                            and loto_audit.heldout_absent_from_fit.all()
                        ),
                        "all_correctors_converged": bool(loto_audit.corrector_converged.all()),
                        "maximum_corrector_n_iter": int(loto_audit.corrector_n_iter.max()),
                    }
                )
                del predictions

    audit = pd.DataFrame(candidate_user_rows)
    aggregate = audit.groupby(
        ["outer_fold", "candidate_kind", "candidate_name", "candidate_c"],
        dropna=False,
        as_index=False,
        sort=True,
    ).agg(
        validation_users=("validation_subject_id", "nunique"),
        mean_baseline_error=("baseline_error", "mean"),
        mean_corrected_error=("corrected_error", "mean"),
        mean_signed_gain=("signed_gain", "mean"),
        median_signed_gain=("signed_gain", "median"),
        positive_users=("signed_gain", lambda values: int((values > 0).sum())),
        negative_users=("signed_gain", lambda values: int((values < 0).sum())),
        all_session_disjoint=("all_session_disjoint", "all"),
        all_correctors_converged=("all_correctors_converged", "all"),
        maximum_corrector_n_iter=("maximum_corrector_n_iter", "max"),
    )
    if not (aggregate.validation_users == 12).all():
        raise AssertionError("Each candidate must be validated on all 12 development subjects")
    if not aggregate.all_session_disjoint.all() or not aggregate.all_correctors_converged.all():
        raise AssertionError("A development candidate failed a hard audit")

    # Corrected error is already equal-user, equal-activity, equal-trial. On an
    # exact tie prefer identity, then the more regularized (smaller-C) head.
    aggregate["complexity_order"] = np.where(aggregate.candidate_kind == "identity", 0, 1)
    aggregate["c_tiebreak"] = aggregate.candidate_c.fillna(0.0)
    ranked = aggregate.sort_values(
        ["mean_corrected_error", "complexity_order", "c_tiebreak"],
        ascending=[True, True, True],
        kind="stable",
    )
    selected_row = ranked.iloc[0]
    selected = CandidateSpec(
        kind=str(selected_row.candidate_kind),
        c_value=(
            None
            if str(selected_row.candidate_kind) == "identity"
            else float(selected_row.candidate_c)
        ),
    )
    aggregate["selected"] = aggregate.candidate_name == selected.name
    audit = audit.merge(
        aggregate[
            ["outer_fold", "candidate_name", "selected", "mean_corrected_error", "mean_signed_gain"]
        ],
        on=["outer_fold", "candidate_name"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_aggregate"),
    )
    candidate_audit = pd.concat(
        [
            audit.assign(audit_level="subject"),
            aggregate.assign(audit_level="aggregate"),
        ],
        ignore_index=True,
        sort=False,
    )
    return selected, candidate_audit, pd.DataFrame(backbone_rows)


def allocation_curve(per_user: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for outer_fold, fold_frame in per_user.groupby("outer_fold", sort=True):
        by_error = fold_frame.sort_values(
            ["baseline_error", "subject_id"], ascending=[False, True], kind="stable"
        )
        by_positive_gain = fold_frame[fold_frame.positive_gain > 0].sort_values(
            ["positive_gain", "subject_id"], ascending=[False, True], kind="stable"
        )
        by_signed_gain = fold_frame.sort_values(
            ["signed_gain", "subject_id"], ascending=[False, True], kind="stable"
        )
        total_positive = float(fold_frame["positive_gain"].sum())
        for budget in range(1, len(fold_frame) + 1):
            error_selected = by_error.head(budget)
            # At-most-B gain-aware oracle never deploys a harmful corrector.
            oracle_selected = by_positive_gain.head(budget)
            # Exactly-B signed oracle is retained separately for audits that
            # require every capacity slot to be filled.
            exact_oracle_selected = by_signed_gain.head(budget)
            error_positive = float(error_selected["positive_gain"].sum())
            oracle_positive = float(oracle_selected["positive_gain"].sum())
            error_signed = float(error_selected["signed_gain"].sum())
            oracle_signed = oracle_positive
            exact_oracle_signed = float(exact_oracle_selected["signed_gain"].sum())
            error_ids = set(error_selected["subject_id"].astype(int))
            oracle_ids = set(oracle_selected["subject_id"].astype(int))
            exact_oracle_ids = set(exact_oracle_selected["subject_id"].astype(int))
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "budget": budget,
                    "budget_fraction": budget / len(fold_frame),
                    "error_positive_utility": error_positive,
                    "oracle_positive_utility": oracle_positive,
                    "error_signed_utility": error_signed,
                    "oracle_signed_utility": oracle_signed,
                    "exact_b_oracle_signed_utility": exact_oracle_signed,
                    "signed_regret": oracle_signed - error_signed,
                    "exact_b_signed_regret": exact_oracle_signed - error_signed,
                    "error_capture_ratio": (
                        min(1.0, max(0.0, error_positive / oracle_positive))
                        if oracle_positive > 0
                        else np.nan
                    ),
                    "random_expected_capture_ratio": (
                        (budget / len(fold_frame) * total_positive) / oracle_positive
                        if oracle_positive > 0
                        else np.nan
                    ),
                    "top_b_overlap_count": len(error_ids & oracle_ids),
                    "top_b_overlap_fraction": len(error_ids & oracle_ids) / budget,
                    "oracle_selected_count": len(oracle_ids),
                    "error_selected_subjects": "|".join(map(str, sorted(error_ids))),
                    "oracle_selected_subjects": "|".join(map(str, sorted(oracle_ids))),
                    "exact_b_oracle_selected_subjects": "|".join(
                        map(str, sorted(exact_oracle_ids))
                    ),
                }
            )
    result = pd.DataFrame(rows)
    tolerance = 1e-12
    if (result.signed_regret < -tolerance).any() or (result.exact_b_signed_regret < -tolerance).any():
        raise AssertionError("Allocation oracle produced negative regret")
    return result


def rank_association(frame: pd.DataFrame, outcome: str) -> dict[str, float | int | None]:
    valid = frame[["baseline_error", outcome]].dropna()
    if len(valid) < 3 or valid.nunique().min() < 2:
        return {"n": int(len(valid)), "spearman": None, "kendall": None}
    return {
        "n": int(len(valid)),
        "spearman": float(spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic),
        "kendall": float(kendalltau(valid.iloc[:, 0], valid.iloc[:, 1]).statistic),
    }


def stratified_bootstrap_spearman(
    frame: pd.DataFrame,
    outcome: str,
    iterations: int,
    seed: int,
) -> dict[str, float | int | None]:
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    fold_frames = [part.reset_index(drop=True) for _, part in frame.groupby("outer_fold", sort=True)]
    for _ in range(iterations):
        parts = [part.iloc[rng.integers(0, len(part), len(part))] for part in fold_frames]
        sample = pd.concat(parts, ignore_index=True)
        valid = sample[["baseline_error", outcome]].dropna()
        if len(valid) < 3 or valid.nunique().min() < 2:
            continue
        statistic = float(spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic)
        if math.isfinite(statistic):
            estimates.append(statistic)
    if not estimates:
        return {
            "iterations_requested": iterations,
            "iterations_valid": 0,
            "low": None,
            "high": None,
        }
    return {
        "iterations_requested": iterations,
        "iterations_valid": len(estimates),
        "low": float(np.quantile(estimates, 0.025)),
        "high": float(np.quantile(estimates, 0.975)),
    }


def fleet_subject_bootstrap(
    fold_frame: pd.DataFrame,
    iterations: int,
    seed: int,
    budgets: Sequence[int] = (3, 6),
) -> dict[str, object]:
    """Bootstrap one 12-user fleet by sampling subject rows with replacement.

    Repeated subjects are treated as repeated fleet slots drawn from the
    empirical user population. A slot index supplies a deterministic tie break.
    This is a superpopulation-style uncertainty diagnostic, not an uncertainty
    interval for the ranking of the fixed 12 unique deployed users.
    """
    frame = fold_frame.sort_values("subject_id", kind="stable").reset_index(drop=True)
    if len(frame) != 12 or frame.subject_id.nunique() != 12:
        raise AssertionError("Fleet bootstrap requires 12 unique subject rows")
    rng = np.random.default_rng(seed)
    spearman_estimates: list[float] = []
    capture_estimates: dict[int, list[float]] = {int(budget): [] for budget in budgets}
    for _ in range(iterations):
        sample = frame.iloc[rng.integers(0, len(frame), size=len(frame))].copy()
        sample["bootstrap_slot"] = np.arange(len(sample), dtype=int)
        if sample[["baseline_error", "signed_gain"]].nunique().min() >= 2:
            estimate = float(spearmanr(sample.baseline_error, sample.signed_gain).statistic)
            if math.isfinite(estimate):
                spearman_estimates.append(estimate)
        by_error = sample.sort_values(
            ["baseline_error", "bootstrap_slot"],
            ascending=[False, True],
            kind="stable",
        )
        by_gain = sample.sort_values(
            ["positive_gain", "bootstrap_slot"],
            ascending=[False, True],
            kind="stable",
        )
        for budget in budgets:
            oracle_positive = float(by_gain.head(budget).positive_gain.sum())
            if oracle_positive <= 0:
                continue
            error_positive = float(by_error.head(budget).positive_gain.sum())
            capture_estimates[int(budget)].append(
                min(1.0, max(0.0, error_positive / oracle_positive))
            )

    def interval(values: Sequence[float]) -> dict[str, float | int | None]:
        if not values:
            return {
                "iterations_valid": 0,
                "percentile_2_5": None,
                "percentile_97_5": None,
            }
        return {
            "iterations_valid": len(values),
            "percentile_2_5": float(np.quantile(values, 0.025)),
            "percentile_97_5": float(np.quantile(values, 0.975)),
        }

    return {
        "iterations_requested": iterations,
        "resampling_unit": "subject row within outer fleet",
        "duplicate_handling": (
            "Repeated draws are distinct fleet slots; bootstrap_slot breaks exact ranking ties."
        ),
        "model_refitting": False,
        "interpretation": (
            "Percentile interval under empirical-user resampling; not a confidence interval "
            "for the fixed ranking of 12 unique deployed users."
        ),
        "error_gain_spearman_95": interval(spearman_estimates),
        "worst_error_gain_capture_95": {
            f"budget_{budget}": {
                "budget": int(budget),
                "budget_fraction": float(budget / len(frame)),
                **interval(capture_estimates[int(budget)]),
            }
            for budget in budgets
        },
    }


def make_split_audit(frame: pd.DataFrame, evaluation_protocol: str) -> pd.DataFrame:
    sessions = frame.groupby(
        ["subject_id", "activity", "trial", "session_key", "source_file"],
        as_index=False,
        sort=True,
    ).agg(
        n_windows=("window_id", "size"),
        first_raw_start=("raw_start", "min"),
        last_raw_end_exclusive=("raw_end_exclusive", "max"),
        source_rows=("source_rows", "first"),
    )
    rows: list[pd.DataFrame] = []
    for outer_fold, split in OUTER_FOLDS.items():
        part = sessions.copy()
        development = set(split["development"])
        target = set(split["target"])
        part["outer_fold"] = outer_fold
        part["outer_role"] = np.where(
            part.subject_id.isin(development), "development", "target"
        )
        part["used_for_final_backbone"] = part.subject_id.isin(development)
        if evaluation_protocol == "loto":
            part["target_used_for_corrector_fit"] = part.subject_id.isin(target)
            part["target_evaluated"] = part.subject_id.isin(target)
        else:
            long_mask = protocol_trial_mask(part, LONG_TRIALS)
            short_mask = protocol_trial_mask(part, SHORT_TRIALS)
            part["target_used_for_corrector_fit"] = part.subject_id.isin(target) & long_mask
            part["target_evaluated"] = part.subject_id.isin(target) & short_mask
        inner_map: dict[int, int] = {}
        for inner_index, ids in enumerate(INNER_VALIDATION[outer_fold], start=1):
            inner_map.update({subject_id: inner_index for subject_id in ids})
        part["inner_validation_fold"] = part.subject_id.map(inner_map)
        rows.append(part)
    audit = pd.concat(rows, ignore_index=True)
    if len(audit) != 720:
        raise AssertionError("Split audit must have 360 sessions for each outer fold")
    return audit.sort_values(
        ["outer_fold", "outer_role", "subject_id", "activity", "trial"], kind="stable"
    )


def make_figure(
    per_user: pd.DataFrame,
    allocation: pd.DataFrame,
    selected_candidates: dict[str, CandidateSpec],
    evaluation_protocol: str,
    output_root: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    styles = {
        "fold_1": {"color": "#0072B2", "marker": "o", "label": "Outer fold 1"},
        "fold_2": {"color": "#D55E00", "marker": "s", "label": "Outer fold 2"},
    }
    for outer_fold, fold_frame in per_user.groupby("outer_fold", sort=True):
        style = styles[outer_fold]
        axes[0].scatter(
            fold_frame["baseline_error"],
            fold_frame["signed_gain"],
            s=36,
            color=style["color"],
            marker=style["marker"],
            edgecolor="black",
            linewidth=0.4,
            label=style["label"],
        )
        for row in fold_frame.itertuples(index=False):
            if pd.notna(row.signed_gain):
                axes[0].annotate(
                    str(int(row.subject_id)),
                    (row.baseline_error, row.signed_gain),
                    xytext=(2, 2),
                    textcoords="offset points",
                    fontsize=6,
                )
    axes[0].axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel(r"Baseline trial-balanced error $E_u$")
    axes[0].set_ylabel(r"Signed personalized gain $G_u$")
    axes[0].set_title("(a) Error versus held-out gain")
    axes[0].legend(frameon=False, fontsize=7)

    plotted_curve = False
    for outer_fold, fold_curve in allocation.groupby("outer_fold", sort=True):
        valid = fold_curve.dropna(subset=["error_capture_ratio"])
        if valid.empty:
            continue
        plotted_curve = True
        style = styles[outer_fold]
        axes[1].plot(
            valid["budget_fraction"],
            valid["error_capture_ratio"],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.4,
            markersize=3.5,
            label=f"Worst-error: {style['label']}",
        )
    axes[1].axhline(1.0, color="#333333", linewidth=0.9, label="Gain-aware oracle")
    axes[1].set_xlabel("Fraction of users personalized")
    axes[1].set_ylabel("Fraction of oracle positive gain captured")
    axes[1].set_title("(b) Fleet allocation consequence")
    if plotted_curve:
        axes[1].legend(frameon=False, fontsize=6.5, loc="best")
    else:
        axes[1].text(
            0.5,
            0.5,
            "No positive held-out gain\nunder selected dev-only correctors",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
            fontsize=8,
        )
    protocol_label = "strict recording LOTO" if evaluation_protocol == "loto" else "long-to-short"
    figure.suptitle(
        f"MotionSense {protocol_label}: subject-disjoint personalized allocation",
        fontsize=8.5,
    )
    figure.savefig(output_root.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_root.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _legacy_main_unused() -> None:
    runtime_start = time.perf_counter()
    args = parse_args()
    if args.run_name in FORBIDDEN_RUN_NAMES or not args.run_name.startswith("motionsense_loto_"):
        raise ValueError(
            "Run name must start with motionsense_loto_ and must not reuse long-to-short output names"
        )
    if args.evaluation_protocol == "long_to_short" and args.run_name == DEFAULT_RUN_NAME:
        raise ValueError(
            "Long-to-short sensitivity requires a distinct run name, for example "
            "motionsense_loto_long_to_short"
        )
    results_dir = PROJECT_ROOT / "results"
    figures_dir = PROJECT_ROOT / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    subject_info_path = args.repo_root / "data" / "data_subjects_info.csv"
    subject_info = pd.read_csv(subject_info_path)
    validate_fixed_splits(subject_info)
    data_root = args.repo_root / "data" / "A_DeviceMotion_data"
    source_paths = source_csvs(data_root)
    source_tree_digest = source_tree_sha256(source_paths, args.repo_root)
    frame, feature_columns, feature_cache, feature_provenance = load_or_build_features(
        args, source_tree_digest
    )
    candidate_specs = corrector_specs(args.c_values)
    split_audit = make_split_audit(frame, args.evaluation_protocol)

    all_predictions: list[pd.DataFrame] = []
    all_recordings: list[pd.DataFrame] = []
    all_loto_audits: list[pd.DataFrame] = []
    all_candidate_audits: list[pd.DataFrame] = []
    all_backbone_audits: list[pd.DataFrame] = []
    selected_candidates: dict[str, CandidateSpec] = {}
    fold_summaries: list[dict[str, object]] = []

    for fold_index, (outer_fold, split) in enumerate(OUTER_FOLDS.items(), start=1):
        development_ids = list(split["development"])
        target_ids = list(split["target"])
        if set(development_ids) & set(target_ids):
            raise AssertionError("Outer development and target subjects overlap")
        development_frame = frame[frame["subject_id"].isin(development_ids)].copy()
        target_frame = frame[frame["subject_id"].isin(target_ids)].copy()

        selected, candidate_audit, inner_backbone_audit = tune_corrector_on_development(
            outer_fold,
            development_frame,
            feature_columns,
            candidate_specs,
            args.seed + fold_index * 1_000_000,
            args.evaluation_protocol,
        )
        selected_candidates[outer_fold] = selected
        selected_development = candidate_audit[
            (candidate_audit.audit_level == "aggregate") & (candidate_audit.selected == True)  # noqa: E712
        ]
        if len(selected_development) != 1:
            raise AssertionError("Development selection audit must identify exactly one candidate")
        selected_development_row = selected_development.iloc[0]
        all_candidate_audits.append(candidate_audit)
        all_backbone_audits.append(inner_backbone_audit)

        final_bundle = fit_backbone(
            development_frame,
            feature_columns,
            development_ids,
            args.seed + fold_index * 10_000_000,
        )
        all_backbone_audits.append(
            pd.DataFrame(
                [
                    {
                        "outer_fold": outer_fold,
                        "stage": "final_development_refit",
                        "training_subjects": "|".join(map(str, sorted(development_ids))),
                        "validation_subjects": "",
                        "training_windows": len(development_frame),
                        "backbone_n_iter": final_bundle.n_iter,
                        "backbone_converged": final_bundle.n_iter < MAX_ITER,
                        "backbone_state_sha256": final_bundle.state_sha256,
                    }
                ]
            )
        )
        frozen_state_before = final_bundle.state_sha256

        fold_prediction_parts: list[pd.DataFrame] = []
        fold_recording_parts: list[pd.DataFrame] = []
        fold_audit_parts: list[pd.DataFrame] = []
        for subject_id in target_ids:
            subject_frame = target_frame[target_frame["subject_id"] == subject_id].copy()
            predictions, recordings, loto_audit = evaluate_subject_protocol(
                evaluation_protocol=args.evaluation_protocol,
                outer_fold=outer_fold,
                subject_frame=subject_frame,
                feature_columns=feature_columns,
                backbone=final_bundle,
                candidate=selected,
                seed=args.seed + fold_index * 100_000_000,
                collect_predictions=True,
            )
            fold_prediction_parts.append(predictions)
            fold_recording_parts.append(recordings)
            fold_audit_parts.append(loto_audit)

        frozen_state_after = model_state_digest(final_bundle.scaler, final_bundle.model)
        if frozen_state_before != frozen_state_after:
            raise AssertionError("Frozen backbone or development normalizer changed during target LOTO")
        fold_predictions = pd.concat(fold_prediction_parts, ignore_index=True)
        fold_recordings = pd.concat(fold_recording_parts, ignore_index=True)
        fold_loto_audit = pd.concat(fold_audit_parts, ignore_index=True)
        expected_target_frame = (
            target_frame
            if args.evaluation_protocol == "loto"
            else target_frame[protocol_trial_mask(target_frame, SHORT_TRIALS)]
        )
        expected_target_windows = len(expected_target_frame)
        if len(fold_predictions) != expected_target_windows:
            raise AssertionError("Outer target predictions are incomplete")
        if fold_predictions["window_id"].duplicated().any():
            raise AssertionError("A target window was evaluated more than once in an outer fold")
        if not (
            (fold_loto_audit.session_overlap_count == 0).all()
            and (fold_loto_audit.window_overlap_count == 0).all()
            and fold_loto_audit.heldout_absent_from_fit.all()
            and fold_loto_audit.corrector_converged.all()
        ):
            raise AssertionError("Outer target LOTO failed a hard disjointness/convergence check")

        expected_recordings_per_user = 15 if args.evaluation_protocol == "loto" else 6
        fold_users = user_metrics_from_recordings(
            fold_recordings, expected_recordings=expected_recordings_per_user
        )
        fold_summaries.append(
            {
                "outer_fold": outer_fold,
                "development_subjects": development_ids,
                "target_subjects": target_ids,
                "development_windows": len(development_frame),
                "target_source_windows": len(target_frame),
                "target_evaluation_windows": len(expected_target_frame),
                "selected_candidate": asdict(selected),
                "selected_candidate_name": selected.name,
                "selected_development_mean_corrected_error": float(
                    selected_development_row.mean_corrected_error
                ),
                "selected_development_mean_signed_gain": float(
                    selected_development_row.mean_signed_gain
                ),
                "final_backbone_n_iter": final_bundle.n_iter,
                "final_backbone_state_sha256": final_bundle.state_sha256,
                "mean_baseline_error": float(fold_users.baseline_error.mean()),
                "mean_corrected_error": float(fold_users.corrected_error.mean()),
                "mean_signed_gain": float(fold_users.signed_gain.mean()),
                "positive_zero_negative_users": {
                    "positive": int((fold_users.signed_gain > 0).sum()),
                    "zero": int((fold_users.signed_gain == 0).sum()),
                    "negative": int((fold_users.signed_gain < 0).sum()),
                },
            }
        )
        all_predictions.append(fold_predictions)
        all_recordings.append(fold_recordings)
        all_loto_audits.append(fold_loto_audit)

    predictions = pd.concat(all_predictions, ignore_index=True).sort_values(
        ["outer_fold", "subject_id", "activity", "trial", "window_index"], kind="stable"
    )
    recordings = pd.concat(all_recordings, ignore_index=True).sort_values(
        ["outer_fold", "subject_id", "activity", "trial"], kind="stable"
    )
    loto_audit = pd.concat(all_loto_audits, ignore_index=True).sort_values(
        ["outer_fold", "subject_id", "heldout_activity", "heldout_trial"], kind="stable"
    )
    candidate_audit = pd.concat(all_candidate_audits, ignore_index=True)
    backbone_audit = pd.concat(all_backbone_audits, ignore_index=True)
    expected_recordings_per_user = 15 if args.evaluation_protocol == "loto" else 6
    per_user = user_metrics_from_recordings(
        recordings, expected_recordings=expected_recordings_per_user
    )
    if len(per_user) != 24 or per_user.subject_id.nunique() != 24:
        raise AssertionError("Every MotionSense subject must be an outer target exactly once")
    expected_evaluation_frame = (
        frame
        if args.evaluation_protocol == "loto"
        else frame[protocol_trial_mask(frame, SHORT_TRIALS)]
    )
    if (
        len(predictions) != len(expected_evaluation_frame)
        or set(predictions.window_id) != set(expected_evaluation_frame.window_id)
    ):
        raise AssertionError("Outer predictions do not cover every protocol test window once")
    expected_recording_rows = 360 if args.evaluation_protocol == "loto" else 144
    if len(recordings) != expected_recording_rows or len(loto_audit) != expected_recording_rows:
        raise AssertionError(
            f"Expected {expected_recording_rows} outer held-out recording evaluations"
        )

    allocation = allocation_curve(per_user)
    associations = {
        "pooled_gain": rank_association(per_user, "signed_gain"),
        "pooled_rho": rank_association(per_user, "rho_signed"),
        "pooled_gain_stratified_bootstrap_spearman_95": stratified_bootstrap_spearman(
            per_user, "signed_gain", args.bootstrap, args.seed
        ),
        "pooled_rho_stratified_bootstrap_spearman_95": stratified_bootstrap_spearman(
            per_user, "rho_signed", args.bootstrap, args.seed + 1
        ),
        "by_fold": {
            outer_fold: {
                "gain": rank_association(fold_frame, "signed_gain"),
                "rho": rank_association(fold_frame, "rho_signed"),
            }
            for outer_fold, fold_frame in per_user.groupby("outer_fold", sort=True)
        },
    }
    fleet_bootstrap = {
        outer_fold: fleet_subject_bootstrap(
            fold_frame,
            args.bootstrap,
            args.seed + 10_000 + fold_index,
            budgets=(3, 6),
        )
        for fold_index, (outer_fold, fold_frame) in enumerate(
            per_user.groupby("outer_fold", sort=True), start=1
        )
    }
    decision_rows = allocation[allocation.budget.isin([3, 6])].copy()
    captures = decision_rows.error_capture_ratio.dropna()
    positive_total = float(per_user.positive_gain.sum())
    positive_count = int((per_user.signed_gain > 0).sum())
    largest_positive_share = (
        float(per_user.positive_gain.max() / positive_total) if positive_total > 0 else None
    )
    minimum_capture_by_fold = decision_rows.groupby("outer_fold").error_capture_ratio.min()
    if positive_total <= 0:
        decision_gate = "kill_no_independently_measured_positive_gain"
    elif (
        len(minimum_capture_by_fold) == 2
        and (minimum_capture_by_fold <= 0.75).all()
        and positive_count >= 6
    ):
        decision_gate = "go_empirical_candidate_requires_sensitivity"
    elif not captures.empty and (captures >= 0.90).all():
        decision_gate = "pivot_negative_or_position"
    else:
        decision_gate = "go_position_diagnostic"

    run_name = args.run_name
    output_paths = {
        "per_user": results_dir / f"{run_name}_per_user.csv",
        "recording_metrics": results_dir / f"{run_name}_recording_metrics.csv",
        "predictions": results_dir / f"{run_name}_predictions.csv",
        "loto_audit": results_dir / f"{run_name}_loto_audit.csv",
        "split_audit": results_dir / f"{run_name}_split_audit.csv",
        "candidate_audit": results_dir / f"{run_name}_candidate_audit.csv",
        "backbone_audit": results_dir / f"{run_name}_backbone_audit.csv",
        "allocation": results_dir / f"{run_name}_allocation.csv",
        "summary": results_dir / f"{run_name}_summary.json",
        "manifest": results_dir / f"{run_name}_manifest.json",
        "pdf": figures_dir / f"{run_name}.pdf",
        "png": figures_dir / f"{run_name}.png",
    }
    per_user.to_csv(output_paths["per_user"], index=False)
    recordings.to_csv(output_paths["recording_metrics"], index=False)
    predictions.to_csv(output_paths["predictions"], index=False)
    loto_audit.to_csv(output_paths["loto_audit"], index=False)
    split_audit.to_csv(output_paths["split_audit"], index=False)
    candidate_audit.to_csv(output_paths["candidate_audit"], index=False)
    backbone_audit.to_csv(output_paths["backbone_audit"], index=False)
    allocation.to_csv(output_paths["allocation"], index=False)
    make_figure(
        per_user,
        allocation,
        selected_candidates,
        args.evaluation_protocol,
        figures_dir / run_name,
    )

    source_archive = args.repo_root / "data" / "A_DeviceMotion_data.zip"
    runtime_seconds = time.perf_counter() - runtime_start

    target_evaluation_description = (
        "15-recording leave-one-recording-out cross-fitting per user"
        if args.evaluation_protocol == "loto"
        else "one fit on nine long recordings and evaluation on six short recordings per user"
    )
    summary = {
        "status": f"strict_subject_disjoint_{args.evaluation_protocol}_complete",
        "runtime_seconds": runtime_seconds,
        "runtime_definition": (
            "Wall-clock seconds through model fitting, artifact generation, and source-tree hashing; "
            "excludes final JSON serialization and output hashing."
        ),
        "decision_gate": decision_gate,
        "protocol": {
            "dataset": "MotionSense",
            "evaluation_protocol": args.evaluation_protocol,
            "repository_commit": repository_commit(args.repo_root),
            "sampling_rate_hz_nominal": 50,
            "window_samples": args.window_samples,
            "step_samples": args.step_samples,
            "trim_samples_each_edge": args.trim_samples,
            "n_features": len(feature_columns),
            "n_windows": len(frame),
            "n_subjects": 24,
            "n_recordings": 360,
            "outer_folds": OUTER_FOLDS,
            "inner_validation_subjects": INNER_VALIDATION,
            "candidate_specs": [asdict(candidate) for candidate in candidate_specs],
            "candidate_selection": (
                "minimum mean trial/activity-balanced corrected error over all 12 development "
                "subjects; exact ties prefer identity then smaller C"
            ),
            "backbone": "development-only standardized multinomial logistic regression; C=1",
            "backbone_trial_scope": (
                "all 15 recordings for development subjects; long-to-short applies to the "
                "per-user corrector fit/test only"
            ),
            "corrector": (
                "identity or fixed 6x6+bias multinomial final-logit map on frozen backbone logits; "
                "equivalent additive residual h(z)-z"
            ),
            "target_evaluation": target_evaluation_description,
            "training_weights": (
                "equal subjects, equal activities per subject, equal trials per activity, "
                "inverse windows per recording"
            ),
            "primary_user_metric": (
                "mean recording error within activity, then equal mean over six activities"
            ),
        },
        "hard_audit": {
            "all_source_sessions_present": len(split_audit) == 720,
            "all_target_windows_predicted_once": (
                len(predictions) == len(expected_evaluation_frame)
                and not predictions.window_id.duplicated().any()
                and set(predictions.window_id) == set(expected_evaluation_frame.window_id)
            ),
            "all_heldout_sessions_absent_from_fit": bool(loto_audit.heldout_absent_from_fit.all()),
            "all_session_overlaps_zero": bool((loto_audit.session_overlap_count == 0).all()),
            "all_window_overlaps_zero": bool((loto_audit.window_overlap_count == 0).all()),
            "all_correctors_converged": bool(loto_audit.corrector_converged.all()),
            "all_backbones_converged": bool(backbone_audit.backbone_converged.all()),
            "target_labels_used_for_candidate_selection": False,
            "target_test_labels_used_for_candidate_selection_or_corrector_fit": False,
            "target_fit_recording_labels_used_for_corrector_fit": True,
        },
        "folds": fold_summaries,
        "associations": associations,
        "fleet_subject_bootstrap": fleet_bootstrap,
        "pooled_descriptives": {
            "mean_baseline_error": float(per_user.baseline_error.mean()),
            "mean_corrected_error": float(per_user.corrected_error.mean()),
            "mean_signed_gain": float(per_user.signed_gain.mean()),
            "median_signed_gain": float(per_user.signed_gain.median()),
            "mean_rho_signed": float(per_user.rho_signed.mean()),
            "positive_zero_negative_users": {
                "positive": positive_count,
                "zero": int((per_user.signed_gain == 0).sum()),
                "negative": int((per_user.signed_gain < 0).sum()),
            },
            "largest_positive_gain_share": largest_positive_share,
        },
        "decision_budgets": decision_rows.to_dict(orient="records"),
        "limitations": [
            "MotionSense provides nominal sample order but no timestamps, so actual sampling jitter is not auditable.",
            "There are only 24 user-level inference units; windows are not treated as independent users.",
            (
                "LOTO estimates a data-rich cross-recording empirical ceiling for this corrector, not Bayes reducibility."
                if args.evaluation_protocol == "loto"
                else "Long-to-short gain is specific to the available long-trial fit budget and short-trial test direction."
            ),
            "Each outer fold has a separately fitted backbone; pooled associations may contain fold effects.",
            (
                "The target-user labels in the other 14 recordings are assumed available for each LOTO corrector fit."
                if args.evaluation_protocol == "loto"
                else "The target-user labels in nine long recordings are assumed available; all six short-test labels remain evaluation-only."
            ),
            "Trial IDs encode a shared collection protocol across subjects; this primary evaluates new users in that protocol.",
            "Fleet bootstrap repeats sampled user rows as fleet slots and conditions on fitted models; it does not re-fit models or quantify uncertainty for the fixed unique-user ranking.",
        ],
        "environment": serializable_environment(),
    }
    strict_summary = json_safe(summary)
    output_paths["summary"].write_text(
        json.dumps(strict_summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )

    hashed_outputs = {}
    for name, path in output_paths.items():
        if name == "manifest":
            continue
        hashed_outputs[name] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "runtime_seconds": runtime_seconds,
        "runtime_definition": strict_summary["runtime_definition"],
        "command": [sys.executable, str(Path(__file__).relative_to(PROJECT_ROOT)), *sys.argv[1:]],
        "script": {
            "path": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "size_bytes": Path(__file__).stat().st_size,
            "sha256": sha256(Path(__file__)),
        },
        "feature_cache": {
            "path": str(feature_cache.relative_to(PROJECT_ROOT)),
            "size_bytes": feature_cache.stat().st_size,
            "sha256": sha256(feature_cache),
            "provenance_path": str(feature_provenance.relative_to(PROJECT_ROOT)),
            "provenance_sha256": sha256(feature_provenance),
        },
        "source": {
            "repository_commit": repository_commit(args.repo_root),
            "recording_count": len(source_paths),
            "recording_tree_sha256": source_tree_digest,
            "archive": (
                {
                    "path": str(source_archive.relative_to(PROJECT_ROOT)),
                    "size_bytes": source_archive.stat().st_size,
                    "sha256": sha256(source_archive),
                }
                if source_archive.exists()
                else None
            ),
            "subject_info": {
                "path": str(subject_info_path.relative_to(PROJECT_ROOT)),
                "size_bytes": subject_info_path.stat().st_size,
                "sha256": sha256(subject_info_path),
            },
        },
        "selected_candidates": {
            fold: asdict(candidate) for fold, candidate in selected_candidates.items()
        },
        "outputs": hashed_outputs,
        "hard_audit": strict_summary["hard_audit"],
    }
    output_paths["manifest"].write_text(
        json.dumps(json_safe(manifest), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(strict_summary, indent=2, ensure_ascii=False, allow_nan=False))


def assert_subject_role_disjoint(
    population_training_subjects: Iterable[int],
    development_selection_subjects: Iterable[int],
    target_subjects: Iterable[int],
) -> None:
    """Enforce the two subject-level leakage boundaries used by the audit."""
    population = set(map(int, population_training_subjects))
    development = set(map(int, development_selection_subjects))
    target = set(map(int, target_subjects))
    leaked_population = population & target
    leaked_selection = development & target
    if leaked_population:
        raise AssertionError(
            f"Target subject entered population-model training: {sorted(leaked_population)}"
        )
    if leaked_selection:
        raise AssertionError(
            f"Target subject entered development selection: {sorted(leaked_selection)}"
        )


def assert_recording_contract(
    enrollment_recordings: Iterable[str],
    future_recordings: Iterable[str],
    corrector_fit_recordings: Iterable[str],
) -> None:
    """Enforce recording-disjoint proxy, fit, and future outcome sets."""
    enrollment = set(map(str, enrollment_recordings))
    future = set(map(str, future_recordings))
    corrector_fit = set(map(str, corrector_fit_recordings))
    overlap = enrollment & future
    if overlap:
        raise AssertionError(
            f"Enrollment recording entered future evaluation: {sorted(overlap)}"
        )
    future_fit_overlap = corrector_fit & future
    if future_fit_overlap:
        raise AssertionError(
            f"Future recording entered corrector fit: {sorted(future_fit_overlap)}"
        )
    if corrector_fit and corrector_fit != enrollment:
        raise AssertionError("Corrector fit recordings must equal the labeled enrollment set")


def assert_prediction_once(
    observed_window_ids: Iterable[str], expected_window_ids: Iterable[str]
) -> None:
    observed = list(map(str, observed_window_ids))
    expected = set(map(str, expected_window_ids))
    if len(observed) != len(set(observed)):
        raise AssertionError("A held-out future prediction was generated more than once")
    observed_set = set(observed)
    if observed_set != expected:
        missing = sorted(expected - observed_set)[:5]
        unexpected = sorted(observed_set - expected)[:5]
        raise AssertionError(
            f"Future prediction coverage mismatch; missing={missing}, unexpected={unexpected}"
        )


def activity_balanced_error(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    labels = frame["label_id"].to_numpy(dtype=int)
    predictions = np.asarray(prediction, dtype=int)
    if len(frame) != len(predictions):
        raise AssertionError("Prediction length does not match evaluation frame")
    weights = activity_trial_weights(frame)
    return float(np.average(predictions != labels, weights=weights))


def make_inner_validation_groups(
    development_subjects: Sequence[int],
    subject_info: pd.DataFrame,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Create three deterministic subject-disjoint inner folds for 12 users."""
    ids = np.asarray(sorted(map(int, development_subjects)), dtype=int)
    if len(ids) != 12 or len(set(ids.tolist())) != 12:
        raise AssertionError("Inner selection requires 12 unique development subjects")
    gender = (
        subject_info.set_index("code").loc[ids, "gender"].to_numpy(dtype=int)
    )
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(seed))
    groups: list[tuple[int, ...]] = []
    for _, validation_positions in splitter.split(ids, gender):
        groups.append(tuple(sorted(ids[validation_positions].astype(int).tolist())))
    flattened = [subject for group in groups for subject in group]
    if len(flattened) != 12 or set(flattened) != set(ids.tolist()):
        raise AssertionError("Generated inner validation folds do not partition development")
    if any(len(group) != 4 for group in groups):
        raise AssertionError("Every inner validation group must contain four subjects")
    return tuple(groups)


def evaluate_subject_enrollment_future(
    split_id: str,
    subject_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    backbone: BackboneBundle,
    candidate: CandidateSpec,
    selection_lock_sha256: str,
    seed: int,
    collect_predictions: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object]]:
    """Estimate labeled-enrollment error and disjoint future held-out gain."""
    subject_ids = subject_frame["subject_id"].unique().astype(int)
    if len(subject_ids) != 1:
        raise AssertionError("Subject evaluator received more than one user")
    subject_id = int(subject_ids[0])
    if subject_id in backbone.training_subjects:
        raise AssertionError("Target subject entered population-model training")
    if subject_frame["session_key"].nunique() != 15:
        raise AssertionError("Every evaluated user must contain all 15 recordings")

    enrollment_mask = protocol_trial_mask(subject_frame, LONG_TRIALS).to_numpy(dtype=bool)
    future_mask = protocol_trial_mask(subject_frame, SHORT_TRIALS).to_numpy(dtype=bool)
    if np.any(enrollment_mask & future_mask) or not np.all(enrollment_mask | future_mask):
        raise AssertionError("Long/short recording maps do not partition this user")
    enrollment_frame = subject_frame.loc[enrollment_mask].copy()
    future_frame = subject_frame.loc[future_mask].copy()
    enrollment_sessions = sorted(enrollment_frame.session_key.unique().astype(str).tolist())
    future_sessions = sorted(future_frame.session_key.unique().astype(str).tolist())
    if len(enrollment_sessions) != 9 or len(future_sessions) != 6:
        raise AssertionError("Expected nine enrollment and six future recordings")
    if set(enrollment_frame.activity) != set(ACTIVITIES):
        raise AssertionError("Enrollment set does not contain all six activities")
    if set(future_frame.activity) != set(ACTIVITIES):
        raise AssertionError("Future set does not contain all six activities")
    corrector_fit_sessions = enrollment_sessions if candidate.kind != "identity" else []
    assert_recording_contract(
        enrollment_sessions, future_sessions, corrector_fit_sessions
    )
    if set(enrollment_frame.window_id) & set(future_frame.window_id):
        raise AssertionError("Enrollment and future window IDs overlap")

    inputs = subject_frame[list(feature_columns)].to_numpy(dtype=float)
    base_logits = decision_logits(backbone.model, backbone.scaler.transform(inputs))
    labels = subject_frame.label_id.to_numpy(dtype=int)
    enrollment_base_predictions = labels_from_logits(
        base_logits[enrollment_mask], backbone.model.classes_
    )
    future_base_predictions = labels_from_logits(
        base_logits[future_mask], backbone.model.classes_
    )
    enrollment_error = activity_balanced_error(
        enrollment_frame, enrollment_base_predictions
    )

    corrector_n_iter = 0
    corrector_state_sha256 = hashlib.sha256(b"identity").hexdigest()
    if candidate.kind == "identity":
        future_corrected_logits = base_logits[future_mask]
        future_corrected_predictions = future_base_predictions.copy()
        used_fit_recordings = 0
        used_fit_windows = 0
    else:
        if candidate.c_value is None:
            raise AssertionError("Linear corrector is missing C")
        corrector, logit_scaler, corrector_n_iter, raw_coef, raw_intercept = (
            fit_linear_corrector(
                base_logits[enrollment_mask],
                labels[enrollment_mask],
                enrollment_frame,
                candidate.c_value,
                seed + subject_id * 1000,
            )
        )
        future_corrected_logits = decision_logits(
            corrector, logit_scaler.transform(base_logits[future_mask])
        )
        absorbed_logits = base_logits[future_mask] @ raw_coef.T + raw_intercept
        if not np.allclose(
            future_corrected_logits, absorbed_logits, rtol=1e-10, atol=1e-10
        ):
            raise AssertionError("Corrector is not the declared raw-logit affine map")
        future_corrected_predictions = labels_from_logits(
            future_corrected_logits, corrector.classes_
        )
        used_fit_recordings = len(enrollment_sessions)
        used_fit_windows = len(enrollment_frame)
        corrector_state_sha256 = array_digest(
            raw_coef, raw_intercept, np.asarray(corrector.classes_)
        )

    future_baseline_error = activity_balanced_error(
        future_frame, future_base_predictions
    )
    future_corrected_error = activity_balanced_error(
        future_frame, future_corrected_predictions
    )
    future_gain = future_baseline_error - future_corrected_error
    if future_gain > future_baseline_error + 1e-12:
        raise AssertionError("Future absolute gain exceeds future baseline-error headroom")

    prediction_parts: list[pd.DataFrame] = []
    if collect_predictions:
        enrollment_part = enrollment_frame[
            [
                "window_id",
                "session_key",
                "subject_id",
                "activity",
                "label_id",
                "trial",
                "window_index",
                "raw_start",
                "raw_end_exclusive",
                "source_file",
            ]
        ].copy()
        enrollment_part.insert(0, "split_id", split_id)
        enrollment_part["evaluation_role"] = "labeled_enrollment_proxy"
        enrollment_part["candidate_name"] = candidate.name
        enrollment_part["candidate_c"] = candidate.c_value
        enrollment_part["selection_lock_sha256"] = selection_lock_sha256
        enrollment_part["baseline_prediction"] = enrollment_base_predictions
        enrollment_part["corrected_prediction"] = pd.array(
            [pd.NA] * len(enrollment_part), dtype="Int64"
        )
        prediction_parts.append(enrollment_part)

        future_part = future_frame[
            [
                "window_id",
                "session_key",
                "subject_id",
                "activity",
                "label_id",
                "trial",
                "window_index",
                "raw_start",
                "raw_end_exclusive",
                "source_file",
            ]
        ].copy()
        future_part.insert(0, "split_id", split_id)
        future_part["evaluation_role"] = "future_heldout_outcome"
        future_part["candidate_name"] = candidate.name
        future_part["candidate_c"] = candidate.c_value
        future_part["selection_lock_sha256"] = selection_lock_sha256
        future_part["baseline_prediction"] = future_base_predictions
        future_part["corrected_prediction"] = future_corrected_predictions
        prediction_parts.append(future_part)
    predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        if prediction_parts
        else pd.DataFrame()
    )

    recording_rows: list[dict[str, object]] = []
    for evaluation_role, role_frame, baseline_prediction, corrected_prediction in (
        (
            "labeled_enrollment_proxy",
            enrollment_frame,
            enrollment_base_predictions,
            None,
        ),
        (
            "future_heldout_outcome",
            future_frame,
            future_base_predictions,
            future_corrected_predictions,
        ),
    ):
        role_work = role_frame[
            ["activity", "trial", "session_key", "label_id", "window_id"]
        ].copy()
        role_work["baseline_prediction"] = baseline_prediction
        role_work["baseline_incorrect"] = (
            role_work.baseline_prediction != role_work.label_id
        )
        if corrected_prediction is not None:
            role_work["corrected_prediction"] = corrected_prediction
            role_work["corrected_incorrect"] = (
                role_work.corrected_prediction != role_work.label_id
            )
        for (activity, trial, session_key), recording in role_work.groupby(
            ["activity", "trial", "session_key"], sort=True
        ):
            baseline_risk = float(recording.baseline_incorrect.mean())
            corrected_risk = (
                float(recording.corrected_incorrect.mean())
                if corrected_prediction is not None
                else None
            )
            recording_rows.append(
                {
                    "split_id": split_id,
                    "subject_id": subject_id,
                    "evaluation_role": evaluation_role,
                    "activity": activity,
                    "trial": int(trial),
                    "session_key": session_key,
                    "n_windows": len(recording),
                    "baseline_error": baseline_risk,
                    "corrected_error": corrected_risk,
                    "signed_gain": (
                        baseline_risk - corrected_risk
                        if corrected_risk is not None
                        else None
                    ),
                }
            )

    user_row: dict[str, object] = {
        "split_id": split_id,
        "subject_id": subject_id,
        "candidate_name": candidate.name,
        "candidate_c": candidate.c_value,
        "selection_lock_sha256": selection_lock_sha256,
        "enrollment_recordings": len(enrollment_sessions),
        "future_recordings": len(future_sessions),
        "enrollment_windows": len(enrollment_frame),
        "future_windows": len(future_frame),
        "enrollment_error": enrollment_error,
        "future_baseline_error": future_baseline_error,
        "future_corrected_error": future_corrected_error,
        "future_gain": future_gain,
        "positive_future_gain": max(0.0, future_gain),
        "corrector_n_iter": corrector_n_iter,
        "corrector_converged": corrector_n_iter < MAX_ITER,
        "corrector_state_sha256": corrector_state_sha256,
    }
    audit_row: dict[str, object] = {
        "split_id": split_id,
        "subject_id": subject_id,
        "population_training_subjects": "|".join(map(str, backbone.training_subjects)),
        "target_absent_from_population_training": subject_id not in backbone.training_subjects,
        "enrollment_recordings": "|".join(enrollment_sessions),
        "future_recordings": "|".join(future_sessions),
        "corrector_fit_recordings": "|".join(corrector_fit_sessions),
        "recording_overlap_count": len(set(enrollment_sessions) & set(future_sessions)),
        "future_fit_overlap_count": len(set(corrector_fit_sessions) & set(future_sessions)),
        "window_overlap_count": len(
            set(enrollment_frame.window_id) & set(future_frame.window_id)
        ),
        "used_fit_recordings": used_fit_recordings,
        "used_fit_windows": used_fit_windows,
        "future_prediction_count": len(future_frame),
        "future_unique_window_count": future_frame.window_id.nunique(),
        "selection_lock_sha256": selection_lock_sha256,
        "candidate_name": candidate.name,
        "candidate_c": candidate.c_value,
    }
    return predictions, pd.DataFrame(recording_rows), user_row, audit_row


def select_candidate_development_only(
    candidate_rows: pd.DataFrame,
    development_subjects: Sequence[int],
    forbidden_target_subjects: Sequence[int],
) -> tuple[CandidateSpec, pd.DataFrame]:
    """Select from development rows and reject any target-contaminated input."""
    expected = set(map(int, development_subjects))
    forbidden = set(map(int, forbidden_target_subjects))
    observed = set(candidate_rows.validation_subject_id.astype(int))
    if observed != expected:
        raise AssertionError(
            f"Candidate selection subjects differ from development set: {sorted(observed)}"
        )
    contamination = observed & forbidden
    if contamination:
        raise AssertionError(
            f"Target-fleet outcome entered candidate selection: {sorted(contamination)}"
        )
    aggregate = candidate_rows.groupby(
        ["split_id", "candidate_kind", "candidate_name", "candidate_c"],
        dropna=False,
        as_index=False,
        sort=True,
    ).agg(
        validation_users=("validation_subject_id", "nunique"),
        mean_future_baseline_error=("future_baseline_error", "mean"),
        mean_future_corrected_error=("future_corrected_error", "mean"),
        mean_future_gain=("future_gain", "mean"),
        median_future_gain=("future_gain", "median"),
        positive_users=("future_gain", lambda values: int((values > 0).sum())),
        negative_users=("future_gain", lambda values: int((values < 0).sum())),
        all_recording_disjoint=("recording_disjoint", "all"),
        all_subject_disjoint=("subject_disjoint", "all"),
        all_predictions_once=("predictions_once", "all"),
        all_correctors_converged=("corrector_converged", "all"),
    )
    if not (aggregate.validation_users == len(expected)).all():
        raise AssertionError("Every candidate must be evaluated on all development users")
    hard_columns = [
        "all_recording_disjoint",
        "all_subject_disjoint",
        "all_predictions_once",
        "all_correctors_converged",
    ]
    if not aggregate[hard_columns].all(axis=None):
        raise AssertionError("A development candidate failed a hard leakage/convergence audit")
    aggregate["complexity_order"] = np.where(
        aggregate.candidate_kind == "identity", 0, 1
    )
    aggregate["c_tiebreak"] = aggregate.candidate_c.fillna(0.0)
    ranked = aggregate.sort_values(
        ["mean_future_corrected_error", "complexity_order", "c_tiebreak"],
        ascending=[True, True, True],
        kind="stable",
    )
    row = ranked.iloc[0]
    selected = CandidateSpec(
        kind=str(row.candidate_kind),
        c_value=None if row.candidate_kind == "identity" else float(row.candidate_c),
    )
    aggregate["selected"] = aggregate.candidate_name == selected.name
    return selected, aggregate


def tune_decision_corrector_on_development(
    split_id: str,
    development_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    development_subjects: Sequence[int],
    target_subjects: Sequence[int],
    validation_groups: Sequence[Sequence[int]],
    candidate_specs: Sequence[CandidateSpec],
    seed: int,
) -> tuple[CandidateSpec, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    development_ids = set(map(int, development_subjects))
    target_ids = set(map(int, target_subjects))
    actual_ids = set(development_frame.subject_id.unique().astype(int))
    if actual_ids != development_ids:
        raise AssertionError("Development frame contains unexpected subjects")
    assert_subject_role_disjoint(development_ids, development_ids, target_ids)
    group_sets = [set(map(int, group)) for group in validation_groups]
    if set.union(*group_sets) != development_ids:
        raise AssertionError("Inner validation groups do not cover development subjects")
    if sum(map(len, group_sets)) != len(set.union(*group_sets)):
        raise AssertionError("Inner validation groups overlap")

    subject_rows: list[dict[str, object]] = []
    backbone_rows: list[dict[str, object]] = []
    for inner_index, validation_ids in enumerate(group_sets, start=1):
        training_ids = sorted(development_ids - validation_ids)
        if len(training_ids) != 8 or validation_ids & set(training_ids):
            raise AssertionError("Inner fold must be an 8/4 subject-disjoint split")
        inner_train = development_frame[
            development_frame.subject_id.isin(training_ids)
        ]
        backbone = fit_backbone(
            inner_train, feature_columns, training_ids, seed + inner_index
        )
        backbone_rows.append(
            {
                "split_id": split_id,
                "stage": f"inner_{inner_index}",
                "training_subjects": "|".join(map(str, training_ids)),
                "validation_subjects": "|".join(map(str, sorted(validation_ids))),
                "target_subjects": "|".join(map(str, sorted(target_ids))),
                "training_windows": len(inner_train),
                "backbone_n_iter": backbone.n_iter,
                "backbone_converged": backbone.n_iter < MAX_ITER,
                "backbone_state_sha256": backbone.state_sha256,
                "target_overlap_count": len(set(backbone.training_subjects) & target_ids),
            }
        )
        for subject_id in sorted(validation_ids):
            subject_frame = development_frame[
                development_frame.subject_id == subject_id
            ]
            for candidate in candidate_specs:
                temporary_lock = hashlib.sha256(
                    f"{split_id}|inner|{inner_index}|{candidate.name}".encode()
                ).hexdigest()
                predictions, _, user, audit = evaluate_subject_enrollment_future(
                    split_id=f"{split_id}_inner_{inner_index}",
                    subject_frame=subject_frame,
                    feature_columns=feature_columns,
                    backbone=backbone,
                    candidate=candidate,
                    selection_lock_sha256=temporary_lock,
                    seed=seed + inner_index * 100_000,
                    collect_predictions=True,
                )
                future = predictions[
                    predictions.evaluation_role == "future_heldout_outcome"
                ]
                expected_future = subject_frame.loc[
                    protocol_trial_mask(subject_frame, SHORT_TRIALS), "window_id"
                ]
                assert_prediction_once(future.window_id, expected_future)
                subject_rows.append(
                    {
                        "split_id": split_id,
                        "inner_fold": inner_index,
                        "validation_subject_id": subject_id,
                        "candidate_kind": candidate.kind,
                        "candidate_name": candidate.name,
                        "candidate_c": candidate.c_value,
                        "enrollment_error": user["enrollment_error"],
                        "future_baseline_error": user["future_baseline_error"],
                        "future_corrected_error": user["future_corrected_error"],
                        "future_gain": user["future_gain"],
                        "recording_disjoint": (
                            audit["recording_overlap_count"] == 0
                            and audit["future_fit_overlap_count"] == 0
                            and audit["window_overlap_count"] == 0
                        ),
                        "subject_disjoint": subject_id not in backbone.training_subjects,
                        "predictions_once": len(future) == future.window_id.nunique(),
                        "corrector_converged": user["corrector_converged"],
                    }
                )
    subject_audit = pd.DataFrame(subject_rows)
    selected, aggregate = select_candidate_development_only(
        subject_audit, sorted(development_ids), sorted(target_ids)
    )
    subject_audit = subject_audit.merge(
        aggregate[
            [
                "split_id",
                "candidate_name",
                "selected",
                "mean_future_corrected_error",
                "mean_future_gain",
            ]
        ],
        on=["split_id", "candidate_name"],
        how="left",
        validate="many_to_one",
    )
    candidate_audit = pd.concat(
        [
            subject_audit.assign(audit_level="subject"),
            aggregate.assign(audit_level="aggregate"),
        ],
        ignore_index=True,
        sort=False,
    )
    aggregate_csv = aggregate.sort_values("candidate_name", kind="stable").to_csv(
        index=False, lineterminator="\n"
    )
    lock_payload: dict[str, object] = {
        "protocol": "development_only_enrollment_to_future_v1",
        "split_id": split_id,
        "development_subjects": sorted(development_ids),
        "forbidden_target_subjects": sorted(target_ids),
        "inner_validation_groups": [sorted(group) for group in group_sets],
        "candidate_grid": [asdict(candidate) for candidate in candidate_specs],
        "selected_candidate": asdict(selected),
        "candidate_aggregate_sha256": hashlib.sha256(
            aggregate_csv.encode("utf-8")
        ).hexdigest(),
        "target_outcomes_available_to_selector": False,
    }
    lock_sha256 = hashlib.sha256(
        json.dumps(lock_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    lock_payload["selection_lock_sha256"] = lock_sha256
    lock_payload["locked_before_target_evaluation"] = True
    return selected, candidate_audit, pd.DataFrame(backbone_rows), lock_payload


def midrank_percentile(values: np.ndarray, observed: float, higher_is_better: bool) -> float:
    numeric = np.asarray(values, dtype=float)
    if higher_is_better:
        worse = np.sum(numeric < observed - 1e-15)
        tied = np.sum(np.isclose(numeric, observed, rtol=0.0, atol=1e-15))
    else:
        worse = np.sum(numeric > observed + 1e-15)
        tied = np.sum(np.isclose(numeric, observed, rtol=0.0, atol=1e-15))
    return float(100.0 * (worse + 0.5 * tied) / len(numeric))


def exact_budget_audit(
    per_user: pd.DataFrame, budgets: Sequence[int] = (3, 6)
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Enumerate every exactly-B subset and compare error-first with at-most-B oracle."""
    frame = per_user.sort_values("subject_id", kind="stable").reset_index(drop=True)
    if len(frame) != 12 or frame.subject_id.nunique() != 12:
        raise AssertionError("Exact audit requires one 12-user fleet")
    split_ids = frame.split_id.unique()
    if len(split_ids) != 1:
        raise AssertionError("Exact audit received multiple fleets")
    split_id = str(split_ids[0])
    subset_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    mechanism_rows: list[dict[str, object]] = []
    by_error = frame.sort_values(
        ["enrollment_error", "subject_id"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    by_gain = frame.sort_values(
        ["future_gain", "subject_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    highest_gain_user = int(by_gain.iloc[0].subject_id)
    highest_gain = float(by_gain.iloc[0].future_gain)

    for budget in budgets:
        if budget <= 0 or budget > len(frame):
            raise ValueError(f"Invalid budget {budget}")
        error_selected = by_error.head(budget)
        error_ids = tuple(sorted(error_selected.subject_id.astype(int).tolist()))
        positive = by_gain[by_gain.future_gain > 0].head(budget)
        oracle_ids = tuple(sorted(positive.subject_id.astype(int).tolist()))
        oracle_value = float(positive.future_gain.sum())
        oracle_positive = float(positive.future_gain.clip(lower=0.0).sum())
        if not np.isclose(oracle_value, oracle_positive, atol=1e-15):
            raise AssertionError("At-most-B oracle must contain positive gains only")

        for combination in itertools.combinations(range(len(frame)), budget):
            selected = frame.iloc[list(combination)]
            selected_ids = tuple(sorted(selected.subject_id.astype(int).tolist()))
            signed_value = float(selected.future_gain.sum())
            positive_value = float(selected.future_gain.clip(lower=0.0).sum())
            capture = positive_value / oracle_positive if oracle_positive > 0 else np.nan
            signed_regret = oracle_value - signed_value
            normalized_regret = (
                signed_regret / oracle_value if oracle_value > 0 else np.nan
            )
            subset_rows.append(
                {
                    "split_id": split_id,
                    "budget": int(budget),
                    "subset_subjects": "|".join(map(str, selected_ids)),
                    "is_error_first": selected_ids == error_ids,
                    "is_positive_gain_oracle": selected_ids == oracle_ids,
                    "positive_gain_capture": capture,
                    "signed_value": signed_value,
                    "signed_regret": signed_regret,
                    "normalized_signed_regret": normalized_regret,
                    "selected_positive_gain": positive_value,
                    "selected_negative_gain": float(
                        selected.loc[selected.future_gain < 0, "future_gain"].sum()
                    ),
                }
            )

        budget_subsets = pd.DataFrame(
            [
                row
                for row in subset_rows
                if row["split_id"] == split_id and row["budget"] == budget
            ]
        )
        expected_count = math.comb(12, budget)
        if len(budget_subsets) != expected_count:
            raise AssertionError(
                f"Exact enumeration expected {expected_count}, got {len(budget_subsets)}"
            )
        policy_rows = budget_subsets[budget_subsets.is_error_first]
        if len(policy_rows) != 1:
            raise AssertionError("Error-first subset was not identified exactly once")
        policy = policy_rows.iloc[0]
        defined_capture = budget_subsets.positive_gain_capture.dropna()
        capture_percentile = (
            midrank_percentile(
                budget_subsets.positive_gain_capture.dropna().to_numpy(),
                float(policy.positive_gain_capture),
                higher_is_better=True,
            )
            if pd.notna(policy.positive_gain_capture)
            else np.nan
        )
        value_percentile = midrank_percentile(
            budget_subsets.signed_value.to_numpy(),
            float(policy.signed_value),
            higher_is_better=True,
        )
        regret_performance_percentile = midrank_percentile(
            budget_subsets.signed_regret.to_numpy(),
            float(policy.signed_regret),
            higher_is_better=False,
        )
        decision_rows.append(
            {
                "split_id": split_id,
                "budget": int(budget),
                "enumerated_subsets": expected_count,
                "error_first_subjects": "|".join(map(str, error_ids)),
                "oracle_subjects_at_most_b": "|".join(map(str, oracle_ids)),
                "oracle_selected_count": len(oracle_ids),
                "positive_gain_denominator_defined": oracle_positive > 0,
                "error_first_positive_gain_capture": float(
                    policy.positive_gain_capture
                ),
                "error_first_signed_value": float(policy.signed_value),
                "oracle_signed_value_at_most_b": oracle_value,
                "error_first_signed_regret": float(policy.signed_regret),
                "error_first_normalized_signed_regret": float(
                    policy.normalized_signed_regret
                ),
                "error_first_capture_random_midrank_percentile": capture_percentile,
                "error_first_value_random_midrank_percentile": value_percentile,
                "error_first_regret_performance_random_midrank_percentile": (
                    regret_performance_percentile
                ),
                "random_capture_mean": (
                    float(defined_capture.mean()) if len(defined_capture) else np.nan
                ),
                "random_capture_q25": (
                    float(defined_capture.quantile(0.25))
                    if len(defined_capture)
                    else np.nan
                ),
                "random_capture_median": (
                    float(defined_capture.median()) if len(defined_capture) else np.nan
                ),
                "random_capture_q75": (
                    float(defined_capture.quantile(0.75))
                    if len(defined_capture)
                    else np.nan
                ),
                "random_signed_value_mean": float(budget_subsets.signed_value.mean()),
                "random_signed_value_q25": float(
                    budget_subsets.signed_value.quantile(0.25)
                ),
                "random_signed_value_median": float(
                    budget_subsets.signed_value.median()
                ),
                "random_signed_value_q75": float(
                    budget_subsets.signed_value.quantile(0.75)
                ),
                "random_signed_regret_mean": float(
                    budget_subsets.signed_regret.mean()
                ),
                "random_signed_regret_q25": float(
                    budget_subsets.signed_regret.quantile(0.25)
                ),
                "random_signed_regret_median": float(
                    budget_subsets.signed_regret.median()
                ),
                "random_signed_regret_q75": float(
                    budget_subsets.signed_regret.quantile(0.75)
                ),
            }
        )

        cutoff_selected = by_error.iloc[budget - 1]
        cutoff_unselected = by_error.iloc[budget] if budget < len(by_error) else None
        harmed_selected = error_selected[error_selected.future_gain < 0]
        miss_highest = highest_gain_user not in set(error_ids)
        top1_fraction = (
            max(0.0, highest_gain) / oracle_value if oracle_value > 0 else np.nan
        )
        below_gain = (
            float(cutoff_unselected.future_gain)
            if cutoff_unselected is not None
            else np.nan
        )
        at_gain = float(cutoff_selected.future_gain)
        if miss_highest and pd.notna(top1_fraction) and top1_fraction >= 0.5:
            mechanism_label = "missed_concentrated_highest_gain_user"
        elif len(harmed_selected) > 0:
            mechanism_label = "negative_gain_selected_user"
        elif cutoff_unselected is not None and below_gain > at_gain:
            mechanism_label = "near_cutoff_rank_inversion"
        else:
            mechanism_label = "diffuse_ranking_disagreement"
        mechanism_rows.append(
            {
                "split_id": split_id,
                "budget": int(budget),
                "highest_gain_user": highest_gain_user,
                "highest_gain": highest_gain,
                "highest_gain_fraction_of_oracle_value": top1_fraction,
                "error_first_misses_highest_gain_user": miss_highest,
                "error_rank_b_user": int(cutoff_selected.subject_id),
                "error_rank_b_enrollment_error": float(
                    cutoff_selected.enrollment_error
                ),
                "error_rank_b_future_gain": at_gain,
                "error_rank_b_plus_1_user": (
                    int(cutoff_unselected.subject_id)
                    if cutoff_unselected is not None
                    else None
                ),
                "error_rank_b_plus_1_enrollment_error": (
                    float(cutoff_unselected.enrollment_error)
                    if cutoff_unselected is not None
                    else None
                ),
                "error_rank_b_plus_1_future_gain": (
                    below_gain if cutoff_unselected is not None else None
                ),
                "below_minus_at_cutoff_gain": (
                    below_gain - at_gain if cutoff_unselected is not None else None
                ),
                "harmed_selected_users": "|".join(
                    map(str, sorted(harmed_selected.subject_id.astype(int).tolist()))
                ),
                "harmed_selected_count": len(harmed_selected),
                "harmed_selected_signed_gain": float(harmed_selected.future_gain.sum()),
                "harmed_selected_magnitude": float(-harmed_selected.future_gain.sum()),
                "missed_positive_gain": float(
                    oracle_value - float(error_selected.future_gain.clip(lower=0.0).sum())
                ),
                "mechanism_label": mechanism_label,
            }
        )
    exact = pd.DataFrame(subset_rows)
    decision = pd.DataFrame(decision_rows)
    mechanism = pd.DataFrame(mechanism_rows)
    if (exact.signed_regret < -1e-12).any():
        raise AssertionError("At-most-B oracle produced negative regret")
    return exact, decision, mechanism


def within_fleet_association(per_user: pd.DataFrame) -> dict[str, object]:
    if len(per_user) != 12:
        raise AssertionError("Within-fleet association requires 12 users")
    valid = per_user[["enrollment_error", "future_gain"]].dropna()
    if valid.nunique().min() < 2:
        return {"n": len(valid), "spearman": None, "kendall": None}
    return {
        "n": len(valid),
        "spearman": float(
            spearmanr(valid.enrollment_error, valid.future_gain).statistic
        ),
        "kendall": float(
            kendalltau(valid.enrollment_error, valid.future_gain).statistic
        ),
    }


def make_recording_split_audit(
    split_id: str,
    frame: pd.DataFrame,
    development_subjects: Sequence[int],
    target_subjects: Sequence[int],
) -> pd.DataFrame:
    sessions = frame.groupby(
        ["subject_id", "activity", "trial", "session_key", "source_file"],
        as_index=False,
        sort=True,
    ).agg(n_windows=("window_id", "size"))
    development = set(map(int, development_subjects))
    target = set(map(int, target_subjects))
    sessions.insert(0, "split_id", split_id)
    sessions["subject_role"] = np.where(
        sessions.subject_id.isin(development), "development", "target"
    )
    long_mask = protocol_trial_mask(sessions, LONG_TRIALS)
    short_mask = protocol_trial_mask(sessions, SHORT_TRIALS)
    sessions["used_for_final_backbone"] = sessions.subject_id.isin(development)
    sessions["target_labeled_enrollment_proxy"] = (
        sessions.subject_id.isin(target) & long_mask
    )
    sessions["target_corrector_fit"] = sessions.subject_id.isin(target) & long_mask
    sessions["target_future_outcome"] = sessions.subject_id.isin(target) & short_mask
    if len(sessions) != 360:
        raise AssertionError("Recording split audit must contain all 360 source recordings")
    if (
        sessions.target_labeled_enrollment_proxy & sessions.target_future_outcome
    ).any():
        raise AssertionError("A recording has both enrollment and future roles")
    return sessions


def run_decision_split(
    split_id: str,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    subject_info: pd.DataFrame,
    development_subjects: Sequence[int],
    target_subjects: Sequence[int],
    validation_groups: Sequence[Sequence[int]],
    candidate_specs: Sequence[CandidateSpec],
    seed: int,
    collect_predictions: bool,
) -> dict[str, object]:
    development_ids = tuple(sorted(map(int, development_subjects)))
    target_ids = tuple(sorted(map(int, target_subjects)))
    if set(development_ids) | set(target_ids) != set(range(1, 25)):
        raise AssertionError("Outer split is not a partition of subjects 1..24")
    assert_subject_role_disjoint(development_ids, development_ids, target_ids)
    gender = subject_info.set_index("code")["gender"].astype(int)
    for role, ids in (("development", development_ids), ("target", target_ids)):
        counts = gender.loc[list(ids)].value_counts().to_dict()
        if counts != {1: 7, 0: 5}:
            raise AssertionError(f"{split_id} {role} gender counts are {counts}")

    development_frame = frame[frame.subject_id.isin(development_ids)].copy()
    target_frame = frame[frame.subject_id.isin(target_ids)].copy()
    selected, candidate_audit, inner_backbone_audit, lock = (
        tune_decision_corrector_on_development(
            split_id=split_id,
            development_frame=development_frame,
            feature_columns=feature_columns,
            development_subjects=development_ids,
            target_subjects=target_ids,
            validation_groups=validation_groups,
            candidate_specs=candidate_specs,
            seed=seed + 1_000_000,
        )
    )
    selection_lock_sha256 = str(lock["selection_lock_sha256"])
    final_backbone = fit_backbone(
        development_frame,
        feature_columns,
        development_ids,
        seed + 10_000_000,
    )
    final_backbone_row = pd.DataFrame(
        [
            {
                "split_id": split_id,
                "stage": "final",
                "training_subjects": "|".join(map(str, development_ids)),
                "validation_subjects": "",
                "target_subjects": "|".join(map(str, target_ids)),
                "training_windows": len(development_frame),
                "backbone_n_iter": final_backbone.n_iter,
                "backbone_converged": final_backbone.n_iter < MAX_ITER,
                "backbone_state_sha256": final_backbone.state_sha256,
                "target_overlap_count": len(
                    set(final_backbone.training_subjects) & set(target_ids)
                ),
            }
        ]
    )
    backbone_audit = pd.concat(
        [inner_backbone_audit, final_backbone_row], ignore_index=True, sort=False
    )
    if (backbone_audit.target_overlap_count != 0).any():
        raise AssertionError("Target subject leaked into a fitted backbone")

    prediction_parts: list[pd.DataFrame] = []
    recording_parts: list[pd.DataFrame] = []
    user_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for subject_id in target_ids:
        subject_frame = target_frame[target_frame.subject_id == subject_id]
        predictions, recordings, user_row, audit_row = (
            evaluate_subject_enrollment_future(
                split_id=split_id,
                subject_frame=subject_frame,
                feature_columns=feature_columns,
                backbone=final_backbone,
                candidate=selected,
                selection_lock_sha256=selection_lock_sha256,
                seed=seed + 100_000_000,
                collect_predictions=collect_predictions,
            )
        )
        if collect_predictions:
            prediction_parts.append(predictions)
        recording_parts.append(recordings)
        user_rows.append(user_row)
        audit_rows.append(audit_row)
    predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        if prediction_parts
        else pd.DataFrame()
    )
    recordings = pd.concat(recording_parts, ignore_index=True)
    per_user = pd.DataFrame(user_rows).sort_values("subject_id", kind="stable")
    protocol_audit = pd.DataFrame(audit_rows).sort_values(
        "subject_id", kind="stable"
    )
    if len(per_user) != 12 or per_user.subject_id.nunique() != 12:
        raise AssertionError("Target evaluation did not produce 12 unique users")
    if not per_user.selection_lock_sha256.eq(selection_lock_sha256).all():
        raise AssertionError("Target evaluation used an unlocked candidate")
    if not (
        (protocol_audit.recording_overlap_count == 0)
        & (protocol_audit.future_fit_overlap_count == 0)
        & (protocol_audit.window_overlap_count == 0)
    ).all():
        raise AssertionError("A target user failed recording/window disjointness")
    if collect_predictions:
        observed_future = predictions[
            predictions.evaluation_role == "future_heldout_outcome"
        ]
        expected_future = target_frame.loc[
            protocol_trial_mask(target_frame, SHORT_TRIALS), "window_id"
        ]
        assert_prediction_once(observed_future.window_id, expected_future)
    exact_subsets, decision_summary, mechanism = exact_budget_audit(per_user)
    association = within_fleet_association(per_user)
    split_audit = make_recording_split_audit(
        split_id, frame, development_ids, target_ids
    )
    split_summary: dict[str, object] = {
        "split_id": split_id,
        "development_subjects": "|".join(map(str, development_ids)),
        "target_subjects": "|".join(map(str, target_ids)),
        "selected_candidate": selected.name,
        "selected_candidate_c": selected.c_value,
        "selection_lock_sha256": selection_lock_sha256,
        "within_fleet_spearman": association["spearman"],
        "within_fleet_kendall": association["kendall"],
        "mean_enrollment_error": float(per_user.enrollment_error.mean()),
        "mean_future_baseline_error": float(per_user.future_baseline_error.mean()),
        "mean_future_corrected_error": float(per_user.future_corrected_error.mean()),
        "mean_future_gain": float(per_user.future_gain.mean()),
        "positive_gain_users": int((per_user.future_gain > 0).sum()),
        "zero_gain_users": int((per_user.future_gain == 0).sum()),
        "negative_gain_users": int((per_user.future_gain < 0).sum()),
        "future_prediction_windows": int(per_user.future_windows.sum()),
    }
    for decision in decision_summary.itertuples(index=False):
        prefix = f"b{int(decision.budget)}"
        split_summary[f"{prefix}_oracle_positive_gain_defined"] = bool(
            decision.positive_gain_denominator_defined
        )
        split_summary[f"{prefix}_capture"] = float(
            decision.error_first_positive_gain_capture
        )
        split_summary[f"{prefix}_signed_value"] = float(
            decision.error_first_signed_value
        )
        split_summary[f"{prefix}_signed_regret"] = float(
            decision.error_first_signed_regret
        )
        split_summary[f"{prefix}_normalized_signed_regret"] = float(
            decision.error_first_normalized_signed_regret
        )
        split_summary[f"{prefix}_value_random_percentile"] = float(
            decision.error_first_value_random_midrank_percentile
        )
        split_summary[f"{prefix}_capture_random_percentile"] = float(
            decision.error_first_capture_random_midrank_percentile
        )
        split_summary[f"{prefix}_regret_performance_random_percentile"] = float(
            decision.error_first_regret_performance_random_midrank_percentile
        )
    for diagnostic in mechanism.itertuples(index=False):
        prefix = f"b{int(diagnostic.budget)}"
        split_summary[f"{prefix}_misses_highest_gain_user"] = bool(
            diagnostic.error_first_misses_highest_gain_user
        )
        split_summary[f"{prefix}_highest_gain_fraction"] = float(
            diagnostic.highest_gain_fraction_of_oracle_value
        )
        split_summary[f"{prefix}_harmed_selected_magnitude"] = float(
            diagnostic.harmed_selected_magnitude
        )

    return {
        "predictions": predictions,
        "recording_metrics": recordings,
        "per_user": per_user,
        "protocol_audit": protocol_audit,
        "candidate_audit": candidate_audit,
        "backbone_audit": backbone_audit,
        "selection_lock": lock,
        "exact_subsets": exact_subsets,
        "decision_summary": decision_summary,
        "mechanism": mechanism,
        "recording_split_audit": split_audit,
        "split_summary": split_summary,
    }


def generate_gender_stratified_partitions(
    subject_info: pd.DataFrame, count: int, seed: int
) -> pd.DataFrame:
    """Generate role-specific, unique, deterministic 7/5 target cohorts."""
    if count < 0:
        raise ValueError("Split-sensitivity count cannot be negative")
    if count == 0:
        return pd.DataFrame()
    info = subject_info.set_index("code")
    male = np.asarray(sorted(info[info.gender == 1].index.astype(int)), dtype=int)
    female = np.asarray(sorted(info[info.gender == 0].index.astype(int)), dtype=int)
    if len(male) != 14 or len(female) != 10:
        raise AssertionError("Expected MotionSense gender counts 14/10")
    maximum = math.comb(14, 7) * math.comb(10, 5)
    if count > maximum:
        raise ValueError(f"Requested {count} unique partitions, maximum is {maximum}")
    rng = np.random.default_rng(seed)
    observed: set[tuple[int, ...]] = set()
    rows: list[dict[str, object]] = []
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 1000:
            raise RuntimeError("Unable to generate requested unique partitions")
        target = tuple(
            sorted(
                np.concatenate(
                    [
                        rng.choice(male, size=7, replace=False),
                        rng.choice(female, size=5, replace=False),
                    ]
                )
                .astype(int)
                .tolist()
            )
        )
        if target in observed:
            continue
        observed.add(target)
        development = tuple(sorted(set(range(1, 25)) - set(target)))
        split_index = len(rows) + 1
        partition_seed = int(seed + split_index * 1_000_003)
        groups = make_inner_validation_groups(
            development, subject_info, partition_seed + 17
        )
        rows.append(
            {
                "split_index": split_index,
                "split_id": f"sensitivity_{split_index:03d}",
                "partition_seed": partition_seed,
                "development_subjects": "|".join(map(str, development)),
                "target_subjects": "|".join(map(str, target)),
                "development_gender_1": 7,
                "development_gender_0": 5,
                "target_gender_1": 7,
                "target_gender_0": 5,
                "inner_validation_1": "|".join(map(str, groups[0])),
                "inner_validation_2": "|".join(map(str, groups[1])),
                "inner_validation_3": "|".join(map(str, groups[2])),
            }
        )
    return pd.DataFrame(rows)


def parse_subject_ids(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in str(value).split("|") if item)


def summarize_split_sensitivity(split_summaries: pd.DataFrame) -> dict[str, object]:
    if split_summaries.empty:
        return {
            "status": "skipped_by_explicit_zero_count",
            "completed_partitions": 0,
            "interpretation": "No split-sensitivity analysis was requested.",
        }
    total = len(split_summaries)
    oracle_defined = split_summaries.b3_oracle_positive_gain_defined.astype(bool)
    capture = split_summaries.loc[oracle_defined, "b3_capture"].dropna()
    normalized_regret = split_summaries.loc[
        oracle_defined, "b3_normalized_signed_regret"
    ].dropna()
    eligible_miss = split_summaries.loc[
        oracle_defined, "b3_misses_highest_gain_user"
    ].astype(bool)
    capture_below_numerator = int((capture < 0.8).sum())
    capture_below_denominator = len(capture)
    miss_top1_numerator = int(eligible_miss.sum())
    miss_top1_denominator = len(eligible_miss)
    candidate_counts = {
        str(key): int(value)
        for key, value in split_summaries.selected_candidate.value_counts().items()
    }
    identity_count = int(candidate_counts.get("identity", 0))
    oracle_defined_b6 = split_summaries.b6_oracle_positive_gain_defined.astype(bool)
    capture_b6 = split_summaries.loc[oracle_defined_b6, "b6_capture"].dropna()
    normalized_regret_b6 = split_summaries.loc[
        oracle_defined_b6, "b6_normalized_signed_regret"
    ].dropna()

    def quantile(values: pd.Series, probability: float) -> float | None:
        return float(values.quantile(probability)) if len(values) else None

    spearman_values = split_summaries.within_fleet_spearman.dropna()
    return {
        "status": "complete",
        "completed_partitions": total,
        "unit": "role-specific gender-stratified 12-development/12-target partition",
        "reuses_same_24_participants": True,
        "is_population_confidence_interval": False,
        "b3_budget_fraction": 0.25,
        "identity_selected_partitions": identity_count,
        "nonidentity_selected_partitions": total - identity_count,
        "b3_oracle_positive_gain_zero_partitions": int((~oracle_defined).sum()),
        "b3_capture_valid_partitions": len(capture),
        "b3_capture_undefined_partitions": total - len(capture),
        "b3_capture_median": quantile(capture, 0.5),
        "b3_capture_q25": quantile(capture, 0.25),
        "b3_capture_q75": quantile(capture, 0.75),
        "b3_normalized_signed_regret_valid_partitions": len(normalized_regret),
        "b3_normalized_signed_regret_undefined_partitions": (
            total - len(normalized_regret)
        ),
        "b3_normalized_signed_regret_median": quantile(normalized_regret, 0.5),
        "b3_normalized_signed_regret_q25": quantile(normalized_regret, 0.25),
        "b3_normalized_signed_regret_q75": quantile(normalized_regret, 0.75),
        "b6_oracle_positive_gain_zero_partitions": int((~oracle_defined_b6).sum()),
        "b6_capture_valid_partitions": len(capture_b6),
        "b6_capture_undefined_partitions": total - len(capture_b6),
        "b6_normalized_signed_regret_valid_partitions": len(
            normalized_regret_b6
        ),
        "b6_normalized_signed_regret_undefined_partitions": (
            total - len(normalized_regret_b6)
        ),
        "b3_capture_below_0_8_numerator": capture_below_numerator,
        "b3_capture_below_0_8_denominator": capture_below_denominator,
        "proportion_b3_capture_below_0_8": (
            capture_below_numerator / capture_below_denominator
            if capture_below_denominator
            else None
        ),
        "b3_error_first_misses_highest_gain_user_numerator": miss_top1_numerator,
        "b3_error_first_misses_highest_gain_user_denominator": miss_top1_denominator,
        "proportion_b3_error_first_misses_highest_gain_user": (
            miss_top1_numerator / miss_top1_denominator
            if miss_top1_denominator
            else None
        ),
        "selected_candidate_counts": candidate_counts,
        "within_fleet_spearman_valid_partitions": len(spearman_values),
        "within_fleet_spearman_undefined_partitions": total - len(spearman_values),
        "within_fleet_spearman_median": quantile(spearman_values, 0.5),
        "within_fleet_spearman_q25": quantile(spearman_values, 0.25),
        "within_fleet_spearman_q75": quantile(spearman_values, 0.75),
        "interpretation": (
            "Split sensitivity only: each partition reuses the same 24 participants "
            "and therefore does not create independent population samples or a "
            "population confidence interval."
        ),
    }


def make_decision_figures(
    primary_per_user: pd.DataFrame,
    primary_exact: pd.DataFrame,
    primary_decisions: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    output_root: Path,
) -> list[Path]:
    plt.rcParams.update(
        {
            "font.size": 8.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    styles = {
        "fold_1": {"color": "#0072B2", "marker": "o", "label": "Fleet 1"},
        "fold_2": {"color": "#D55E00", "marker": "s", "label": "Fleet 2"},
    }
    for split_id, part in primary_per_user.groupby("split_id", sort=True):
        style = styles.get(
            split_id,
            {"color": "#555555", "marker": "o", "label": split_id},
        )
        axes[0].scatter(
            part.enrollment_error,
            part.future_gain,
            s=34,
            color=style["color"],
            marker=style["marker"],
            edgecolor="black",
            linewidth=0.35,
            label=style["label"],
        )
    axes[0].axhline(0, color="#666666", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Labeled enrollment error")
    axes[0].set_ylabel("Future held-out gain")
    axes[0].set_title("(a) Recording-disjoint association")
    axes[0].legend(frameon=False, fontsize=7)

    positions: list[float] = []
    values: list[np.ndarray] = []
    labels: list[str] = []
    policy_values: list[float] = []
    oracle_values: list[float] = []
    position = 1
    for split_id in ("fold_1", "fold_2"):
        for budget in (3, 6):
            random_values = primary_exact[
                (primary_exact.split_id == split_id)
                & (primary_exact.budget == budget)
            ].signed_value.to_numpy()
            decision = primary_decisions[
                (primary_decisions.split_id == split_id)
                & (primary_decisions.budget == budget)
            ].iloc[0]
            positions.append(position)
            values.append(random_values)
            labels.append(f"{styles[split_id]['label']}\nB={budget}")
            policy_values.append(float(decision.error_first_signed_value))
            oracle_values.append(float(decision.oracle_signed_value_at_most_b))
            position += 1
    box = axes[1].boxplot(
        values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#333333", "linewidth": 1.0},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#D9D9D9")
        patch.set_edgecolor("#777777")
    axes[1].scatter(
        positions,
        policy_values,
        color="#CC3311",
        marker="D",
        s=32,
        label="Error-first",
        zorder=3,
    )
    axes[1].scatter(
        positions,
        oracle_values,
        color="#009988",
        marker="*",
        s=52,
        label="At-most-B oracle",
        zorder=3,
    )
    axes[1].axhline(0, color="#777777", linestyle="--", linewidth=0.7)
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Signed future gain")
    axes[1].set_title("(b) Exact random-subset reference")
    axes[1].legend(frameon=False, fontsize=6.8, loc="best")
    primary_pdf = output_root.with_suffix(".pdf")
    primary_png = output_root.with_suffix(".png")
    figure.savefig(primary_pdf, bbox_inches="tight")
    figure.savefig(primary_png, dpi=300, bbox_inches="tight")
    plt.close(figure)
    outputs = [primary_pdf, primary_png]

    if not sensitivity_summary.empty:
        sensitivity_figure, sensitivity_axis = plt.subplots(
            1, 1, figsize=(4.3, 2.7), constrained_layout=True
        )
        values_capture = sensitivity_summary.b3_capture.dropna().to_numpy()
        sensitivity_axis.hist(
            values_capture,
            bins=np.linspace(0, 1, 21),
            color="#56B4E9",
            edgecolor="white",
        )
        sensitivity_axis.axvline(
            np.median(values_capture),
            color="#CC3311",
            linewidth=1.4,
            label=f"Median {np.median(values_capture):.2f}",
        )
        sensitivity_axis.axvline(
            0.8, color="#555555", linestyle="--", linewidth=1.0, label="0.80"
        )
        sensitivity_axis.set_xlim(0, 1.02)
        sensitivity_axis.set_xlabel("Error-first positive-gain capture at B=3")
        sensitivity_axis.set_ylabel("Role-specific partitions")
        sensitivity_axis.set_title(
            f"Split sensitivity across {len(sensitivity_summary)} refitted partitions"
        )
        sensitivity_axis.legend(frameon=False, fontsize=7)
        sensitivity_root = output_root.with_name(
            f"{output_root.name}_split_sensitivity"
        )
        sensitivity_pdf = sensitivity_root.with_suffix(".pdf")
        sensitivity_png = sensitivity_root.with_suffix(".png")
        sensitivity_figure.savefig(sensitivity_pdf, bbox_inches="tight")
        sensitivity_figure.savefig(sensitivity_png, dpi=300, bbox_inches="tight")
        plt.close(sensitivity_figure)
        outputs.extend([sensitivity_pdf, sensitivity_png])
    return outputs


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def relative_project_path(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def sanitized_command() -> list[str]:
    command = ["python", relative_project_path(Path(__file__))]
    for token in sys.argv[1:]:
        candidate = Path(token)
        if candidate.is_absolute():
            try:
                command.append(relative_project_path(candidate))
            except ValueError:
                command.append("<absolute-path-redacted>")
        else:
            command.append(token)
    return command


def main() -> None:
    runtime_start = time.perf_counter()
    args = parse_args()
    if args.run_name in FORBIDDEN_RUN_NAMES or not args.run_name.startswith(
        "motionsense_decision_"
    ):
        raise ValueError(
            "Run name must start with motionsense_decision_ and cannot reuse frozen names"
        )
    if args.split_sensitivity < 0:
        raise ValueError("--split-sensitivity must be nonnegative")
    results_dir = PROJECT_ROOT / "results"
    figures_dir = PROJECT_ROOT / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    subject_info_path = args.repo_root / "data" / "data_subjects_info.csv"
    subject_info = pd.read_csv(subject_info_path)
    validate_fixed_splits(subject_info)
    data_root = args.repo_root / "data" / "A_DeviceMotion_data"
    source_paths = source_csvs(data_root)
    source_tree_digest = source_tree_sha256(source_paths, args.repo_root)
    frame, feature_columns, feature_cache, feature_provenance = load_or_build_features(
        args, source_tree_digest
    )
    candidates = corrector_specs(args.c_values)
    sensitivity_candidates = corrector_specs(
        args.sensitivity_c_values
        if args.sensitivity_c_values is not None
        else args.c_values
    )

    primary_results: list[dict[str, object]] = []
    for fold_index, (split_id, split) in enumerate(OUTER_FOLDS.items(), start=1):
        primary_results.append(
            run_decision_split(
                split_id=split_id,
                frame=frame,
                feature_columns=feature_columns,
                subject_info=subject_info,
                development_subjects=split["development"],
                target_subjects=split["target"],
                validation_groups=INNER_VALIDATION[split_id],
                candidate_specs=candidates,
                seed=args.seed + fold_index * 100_000_000,
                collect_predictions=True,
            )
        )

    def concatenate(key: str, items: Sequence[dict[str, object]]) -> pd.DataFrame:
        frames = [item[key] for item in items]
        return pd.concat(frames, ignore_index=True, sort=False)  # type: ignore[arg-type]

    primary_predictions = concatenate("predictions", primary_results)
    primary_recordings = concatenate("recording_metrics", primary_results)
    primary_per_user = concatenate("per_user", primary_results)
    primary_protocol_audit = concatenate("protocol_audit", primary_results)
    primary_candidate_audit = concatenate("candidate_audit", primary_results)
    primary_backbone_audit = concatenate("backbone_audit", primary_results)
    primary_exact = concatenate("exact_subsets", primary_results)
    primary_decisions = concatenate("decision_summary", primary_results)
    primary_mechanism = concatenate("mechanism", primary_results)
    primary_recording_split = concatenate("recording_split_audit", primary_results)
    primary_split_summary = pd.DataFrame(
        [item["split_summary"] for item in primary_results]
    )
    primary_locks = pd.DataFrame(
        [
            {
                **{
                    key: (
                        "|".join(map(str, value))
                        if isinstance(value, list)
                        and all(isinstance(item, int) for item in value)
                        else json.dumps(value, sort_keys=True)
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in item["selection_lock"].items()
                }
            }
            for item in primary_results
        ]
    )
    future_primary = primary_predictions[
        primary_predictions.evaluation_role == "future_heldout_outcome"
    ]
    expected_primary_future = frame.loc[
        protocol_trial_mask(frame, SHORT_TRIALS), "window_id"
    ]
    assert_prediction_once(future_primary.window_id, expected_primary_future)

    sensitivity_manifest = generate_gender_stratified_partitions(
        subject_info, args.split_sensitivity, args.sensitivity_seed
    )
    sensitivity_results: list[dict[str, object]] = []
    for row in sensitivity_manifest.itertuples(index=False):
        development = parse_subject_ids(row.development_subjects)
        target = parse_subject_ids(row.target_subjects)
        validation_groups = (
            parse_subject_ids(row.inner_validation_1),
            parse_subject_ids(row.inner_validation_2),
            parse_subject_ids(row.inner_validation_3),
        )
        sensitivity_results.append(
            run_decision_split(
                split_id=row.split_id,
                frame=frame,
                feature_columns=feature_columns,
                subject_info=subject_info,
                development_subjects=development,
                target_subjects=target,
                validation_groups=validation_groups,
                candidate_specs=sensitivity_candidates,
                seed=int(row.partition_seed),
                collect_predictions=False,
            )
        )
        completed = len(sensitivity_results)
        if completed % 10 == 0 or completed == len(sensitivity_manifest):
            elapsed = time.perf_counter() - runtime_start
            print(
                f"split_sensitivity_progress={completed}/{len(sensitivity_manifest)} "
                f"elapsed_seconds={elapsed:.1f}",
                flush=True,
            )

    if sensitivity_results:
        sensitivity_per_user = concatenate("per_user", sensitivity_results)
        sensitivity_candidate_audit = concatenate(
            "candidate_audit", sensitivity_results
        )
        sensitivity_backbone_audit = concatenate(
            "backbone_audit", sensitivity_results
        )
        sensitivity_exact = concatenate("exact_subsets", sensitivity_results)
        sensitivity_decisions = concatenate(
            "decision_summary", sensitivity_results
        )
        sensitivity_mechanism = concatenate("mechanism", sensitivity_results)
        sensitivity_split_summary = pd.DataFrame(
            [item["split_summary"] for item in sensitivity_results]
        )
        sensitivity_locks = pd.DataFrame(
            [
                {
                    key: (
                        "|".join(map(str, value))
                        if isinstance(value, list)
                        and all(isinstance(element, int) for element in value)
                        else json.dumps(value, sort_keys=True)
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in item["selection_lock"].items()
                }
                for item in sensitivity_results
            ]
        )
    else:
        sensitivity_per_user = pd.DataFrame()
        sensitivity_candidate_audit = pd.DataFrame()
        sensitivity_backbone_audit = pd.DataFrame()
        sensitivity_exact = pd.DataFrame()
        sensitivity_decisions = pd.DataFrame()
        sensitivity_mechanism = pd.DataFrame()
        sensitivity_split_summary = pd.DataFrame()
        sensitivity_locks = pd.DataFrame()

    sensitivity_aggregate = summarize_split_sensitivity(
        sensitivity_split_summary
    )
    figure_paths = make_decision_figures(
        primary_per_user,
        primary_exact,
        primary_decisions,
        sensitivity_split_summary,
        figures_dir / args.run_name,
    )

    output_paths: dict[str, Path] = {
        "primary_predictions": results_dir / f"{args.run_name}_predictions.csv",
        "primary_recording_metrics": results_dir
        / f"{args.run_name}_recording_metrics.csv",
        "primary_per_user": results_dir / f"{args.run_name}_per_user.csv",
        "primary_protocol_audit": results_dir
        / f"{args.run_name}_protocol_audit.csv",
        "primary_candidate_audit": results_dir
        / f"{args.run_name}_candidate_audit.csv",
        "primary_backbone_audit": results_dir
        / f"{args.run_name}_backbone_audit.csv",
        "primary_exact_subsets": results_dir
        / f"{args.run_name}_exact_subsets.csv",
        "primary_decision_summary": results_dir
        / f"{args.run_name}_decision_summary.csv",
        "primary_mechanism": results_dir / f"{args.run_name}_mechanism.csv",
        "primary_recording_split": results_dir
        / f"{args.run_name}_recording_split.csv",
        "primary_split_summary": results_dir
        / f"{args.run_name}_split_summary.csv",
        "primary_selection_locks": results_dir
        / f"{args.run_name}_selection_locks.csv",
        "sensitivity_manifest": results_dir
        / f"{args.run_name}_split_sensitivity_manifest.csv",
        "sensitivity_per_user": results_dir
        / f"{args.run_name}_split_sensitivity_per_user.csv",
        "sensitivity_candidate_audit": results_dir
        / f"{args.run_name}_split_sensitivity_candidate_audit.csv",
        "sensitivity_backbone_audit": results_dir
        / f"{args.run_name}_split_sensitivity_backbone_audit.csv",
        "sensitivity_exact_subsets": results_dir
        / f"{args.run_name}_split_sensitivity_exact_subsets.csv",
        "sensitivity_decision_summary": results_dir
        / f"{args.run_name}_split_sensitivity_decision_summary.csv",
        "sensitivity_mechanism": results_dir
        / f"{args.run_name}_split_sensitivity_mechanism.csv",
        "sensitivity_split_summary": results_dir
        / f"{args.run_name}_split_sensitivity_summary.csv",
        "sensitivity_selection_locks": results_dir
        / f"{args.run_name}_split_sensitivity_selection_locks.csv",
        "summary": results_dir / f"{args.run_name}_summary.json",
        "manifest": results_dir / f"{args.run_name}_manifest.json",
    }
    csv_outputs = {
        "primary_predictions": primary_predictions,
        "primary_recording_metrics": primary_recordings,
        "primary_per_user": primary_per_user,
        "primary_protocol_audit": primary_protocol_audit,
        "primary_candidate_audit": primary_candidate_audit,
        "primary_backbone_audit": primary_backbone_audit,
        "primary_exact_subsets": primary_exact,
        "primary_decision_summary": primary_decisions,
        "primary_mechanism": primary_mechanism,
        "primary_recording_split": primary_recording_split,
        "primary_split_summary": primary_split_summary,
        "primary_selection_locks": primary_locks,
        "sensitivity_manifest": sensitivity_manifest,
        "sensitivity_per_user": sensitivity_per_user,
        "sensitivity_candidate_audit": sensitivity_candidate_audit,
        "sensitivity_backbone_audit": sensitivity_backbone_audit,
        "sensitivity_exact_subsets": sensitivity_exact,
        "sensitivity_decision_summary": sensitivity_decisions,
        "sensitivity_mechanism": sensitivity_mechanism,
        "sensitivity_split_summary": sensitivity_split_summary,
        "sensitivity_selection_locks": sensitivity_locks,
    }
    for name, output_frame in csv_outputs.items():
        write_csv(output_frame, output_paths[name])

    runtime_seconds = time.perf_counter() - runtime_start
    primary_summary_records = json_safe(
        primary_split_summary.to_dict(orient="records")
    )
    summary: dict[str, object] = {
        "status": "recording_disjoint_decision_audit_complete",
        "run_name": args.run_name,
        "runtime_seconds": runtime_seconds,
        "primary_protocol": {
            "allocation_unit": "user",
            "population_model": "subject-disjoint frozen multinomial logistic sensor classifier",
            "intervention": "fixed 6x6+bias user-specific logit corrector",
            "proxy": "activity-balanced labeled enrollment error on nine long recordings",
            "outcome": "activity-balanced future absolute risk reduction on six short recordings",
            "budget": "number of maintained user-specific correctors",
            "enrollment_future_recording_disjoint": True,
            "candidate_selection": "development subjects under analogous enrollment-to-future protocol",
            "within_fleet_association_only": True,
            "pooled_24_user_spearman_headline_prohibited": True,
        },
        "primary_fleets": primary_summary_records,
        "primary_exact_enumeration": {
            "budget_3_subsets_per_fleet": math.comb(12, 3),
            "budget_6_subsets_per_fleet": math.comb(12, 6),
            "positive_gain_capture_denominator_rule": (
                "defined only when at-most-B positive-gain oracle value is positive"
            ),
            "random_percentile_method": (
                "midrank percentile within complete exactly-B subset distribution; "
                "higher is better for capture/value and lower is better for regret"
            ),
        },
        "split_sensitivity": sensitivity_aggregate,
        "hard_audit": {
            "target_subject_in_population_training": False,
            "target_subject_in_development_selection": False,
            "enrollment_recording_in_future_evaluation": False,
            "future_recording_in_corrector_fit": False,
            "target_outcome_in_candidate_selection": False,
            "future_prediction_generated_more_than_once": False,
            "all_primary_target_users": len(primary_per_user) == 24,
            "all_primary_future_windows_predicted_once": (
                len(future_primary) == future_primary.window_id.nunique()
                == len(expected_primary_future)
            ),
            "all_primary_selection_locks_precede_target_evaluation": bool(
                primary_locks.locked_before_target_evaluation.astype(bool).all()
            ),
            "all_sensitivity_partitions_refit_and_reselect": (
                len(sensitivity_backbone_audit) == 4 * args.split_sensitivity
                and len(sensitivity_locks) == args.split_sensitivity
            ),
        },
        "limitations": [
            "One scripted smartphone inertial dataset and 24 participants.",
            "Repeated partitions reuse the same participants and are split sensitivity, not a population confidence interval.",
            "The long-to-short direction is tied to MotionSense recording duration and trial IDs.",
            "Labeled enrollment is required before allocation; future gain is retrospective and unavailable at allocation time.",
            "The design does not identify a causal personalization effect or Bayes-reducible error.",
        ],
        "environment": serializable_environment(),
    }
    output_paths["summary"].write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    hashed_outputs: dict[str, object] = {}
    for name, path in output_paths.items():
        if name == "manifest":
            continue
        hashed_outputs[name] = {
            "path": relative_project_path(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    for figure_path in figure_paths:
        hashed_outputs[f"figure_{figure_path.stem}_{figure_path.suffix[1:]}"] = {
            "path": relative_project_path(figure_path),
            "size_bytes": figure_path.stat().st_size,
            "sha256": sha256(figure_path),
        }
    manifest: dict[str, object] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "command": sanitized_command(),
        "script": {
            "path": relative_project_path(Path(__file__)),
            "size_bytes": Path(__file__).stat().st_size,
            "sha256": sha256(Path(__file__)),
        },
        "runtime_seconds": runtime_seconds,
        "feature_cache": {
            "path": relative_project_path(feature_cache),
            "size_bytes": feature_cache.stat().st_size,
            "sha256": sha256(feature_cache),
            "provenance_path": relative_project_path(feature_provenance),
            "provenance_sha256": sha256(feature_provenance),
        },
        "source": {
            "repository_path": "data/motion-sense",
            "repository_commit": repository_commit(args.repo_root),
            "recording_count": len(source_paths),
            "recording_tree_sha256": source_tree_digest,
            "subject_info_path": relative_project_path(subject_info_path),
            "subject_info_sha256": sha256(subject_info_path),
        },
        "seeds": {
            "primary": args.seed,
            "split_sensitivity": args.sensitivity_seed,
        },
        "candidate_grid": [asdict(candidate) for candidate in candidates],
        "sensitivity_candidate_grid": [
            asdict(candidate) for candidate in sensitivity_candidates
        ],
        "outputs": hashed_outputs,
        "hard_audit": summary["hard_audit"],
    }
    output_paths["manifest"].write_text(
        json.dumps(json_safe(manifest), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
