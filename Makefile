.PHONY: setup test figures full lint clean

setup:
	uv venv .venv --python 3.13
	uv pip install -e ".[dev]"

test:
	PYTHONPATH=src uv run --active pytest -q

lint:
	uv run --active ruff check src analysis paper experiments tests

# Stage 2: CPU only, reads the committed results/ cache, no dataset needed.
figures:
	PYTHONPATH=src uv run --active python analysis/covariance_check.py
	PYTHONPATH=src uv run --active python analysis/repeatability_stratified.py
	PYTHONPATH=src uv run --active python analysis/design_participant_bootstrap.py
	PYTHONPATH=src uv run --active python analysis/figure1b.py
	PYTHONPATH=src uv run --active python analysis/make_macros.py
	PYTHONPATH=src uv run --active python paper/make_figures.py
	PYTHONPATH=src uv run --active python paper/make_readme_overview.py

# Stage 1: needs raw MotionSense data and network access. See experiments/README.md.
full:
	PYTHONPATH=src uv run --active python experiments/extract_and_fit.py \
		--run-name motionsense_decision_audit --rebuild-features
	PYTHONPATH=src uv run --active python experiments/anchor_refit.py
	PYTHONPATH=src uv run --active python experiments/episode_transport.py
	PYTHONPATH=src uv run --active python experiments/enumerate_future_episode_splits.py

clean:
	rm -rf figures/*.pdf figures/*.png
	rm -f results/covariance_check.json results/design_participant_bootstrap.json \
		results/repeatability_stratified.json results/figure1b_identity_standard_design.csv \
		results/generated_macros.json
