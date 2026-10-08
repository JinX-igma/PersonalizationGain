# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] - 2026-10-08

Initial reproduction package for the camera-ready paper.

- `src/repgain`: installable estimator (`RepeatabilityCheck`), participant
  bootstrap, and the affine-logit adapter.
- `experiments/`: Stage 1 feature extraction and model fitting (needs raw
  MotionSense data).
- `analysis/`: Stage 2 CPU-only analysis reading the committed `results/`
  cache -- covariance check, stratified repeatability, participant bootstrap,
  Figure 1(b) selection, and claim macros.
- `paper/claims.yaml` + `tests/test_claims.py`: every headline number in the
  abstract checked against regenerated code.
- Unit tests for the estimator, bootstrap, and adapter (property-style:
  determinism, zero-repeatability-when-identical, monotonicity in C).
- `make setup` / `make test` / `make figures` / `make full`.
