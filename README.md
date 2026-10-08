# PersonalizationGain

**How much does a personalization gain move if you measure it on a different
episode?** This repo answers that question and reproduces the results of
*How Precise Is a Personalization Gain? Episode Repeatability in Wearable
Activity Recognition* (NeurIPS 2026 workshop, CL4MAgents). The installable
package is `repgain`.

Cohort-average personalization gains are routinely reported as evidence that
a per-user adapter works. They can be positive on average while being
individually unrepeatable: score the same fixed adapter on a second episode
of the same user and activity, and a meaningful fraction of per-user
conclusions flip.

![Left: 24 participants' personalization gains sorted against the cohort mean, most inside the measurement's own noise band. Right: the same gain measured on two separate episodes of the same user and activity, scattered against the diagonal, colored by activity.](assets/overview.png)

*(a) The cohort mean (dashed line) is positive, but score the measurement's
own noise band (shaded, derived from panel b) against each participant and
22 of 24 are not individually distinguishable from zero -- a real group
effect does not imply a stable per-user number. (b) The same (user,
activity) gain, measured by the same fixed adapter on two different held-out
episodes: a perfectly repeatable measurement would sit on the diagonal;
51/66 cells even keep the same sign. Regenerate with `make figures`.*

`repgain` is the diagnostic that catches this, packaged so any
personalization study can run it on its own results:

```python
from repgain import RepeatabilityCheck

check = RepeatabilityCheck(n_boot=2000, seed=0)
report = check.run(df)   # columns: subject, episode, gain [, group]
print(report.summary())
report.plot()
```

`df` needs two measurements of the same (subject, group) cell's gain, taken
from one **fixed** adapter scored on two separate episodes. If you only have
raw scores under two conditions, pass `condition` + `score` columns instead
of `gain` and `RepeatabilityCheck` derives it for you.

## What this measures, concretely

For MotionSense, three of six activities happen to be recorded twice per
participant. Holding out both recordings, fitting one adapter on the rest,
and scoring that same adapter on each held-out recording separately gives two
independent draws of the same (user, activity) gain. The RMS scale of their
difference is the *episode repeatability* reported in the paper:
`sigma = sqrt(mean((gain_1 - gain_2)^2) / 2)`.

## Repository layout

```
assets/              static copies of figures/*.png, committed only so they
                     render on GitHub -- regenerate with `make figures` and
                     `cp figures/{overview,repeatability}.png assets/`,
                     never hand-edit
src/repgain/        installable package: the estimator, the bootstrap, the adapter
experiments/        Stage 1 (needs raw MotionSense data): feature extraction,
                     backbone + per-user adapter fitting -> small cached CSVs in results/
analysis/            Stage 2 (CPU only, reads only the cached CSVs in results/):
                     covariance check, stratified repeatability, bootstrap,
                     Figure 1(b) correction, paper macros
paper/               claims.yaml (every reported number -> script -> expected
                     value) and make_figures.py, which regenerates figures/
results/              committed, small, frozen Stage 1 outputs (and the Stage 2
                     outputs they regenerate, gitignored so `make figures`
                     always proves it can reproduce them)
tests/               unit tests for src/repgain, and test_claims.py
```

## Reproducing the paper

```bash
make setup     # uv-managed virtualenv + locked dependencies
make test      # unit tests for src/repgain
make figures   # regenerate figures/ and results/generated_macros.json
               # from the committed results/ cache -- CPU, a few seconds,
               # no dataset download required
make full      # Stage 1: clone MotionSense, extract features, fit the
               # backbone + adapters, refresh results/ -- needs network access
               # and a few minutes on CPU (no GPU is used anywhere)
```

`make test` also runs `tests/test_claims.py`, which recomputes every macro in
`paper/claims.yaml` from the committed cache and checks it against the value
actually printed in the paper.

`make figures` also regenerates the paper's actual Figure 1, bit-for-bit:

<img src="assets/repeatability.png" alt="Figure 1 from the paper: episode variation vs. gain magnitude, and per-user gains against episode-repeatability reference scales" width="520">

*(a) |episode gain difference| vs. |activity-level gain|, over the 72 (user,
activity) cells with a repeated episode. (b) Per-user gains against the two
reference half-widths derived from panel (a)'s three-activity aggregate.*

## Data

The analysis uses one public dataset, [MotionSense](https://github.com/mmalekzadeh/motion-sense)
(MIT License, included at `experiments/MOTIONSENSE_LICENSE.txt`). Raw
recordings are not redistributed here; `make full` fetches them. The
committed files under `results/` are small, deterministic, aggregated
feature/outcome tables (per-window features, per-user gains, per-design
summaries) derived from that data under its MIT terms, not the recordings
themselves.

## Citation

See `CITATION.cff`.

## License

Code is MIT licensed (`LICENSE`). This is independent of the MotionSense
dataset's own MIT license noted above.
