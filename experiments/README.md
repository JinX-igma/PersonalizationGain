# Stage 1: feature extraction and model fitting

Needs raw data and network access; CPU only (nothing here uses a GPU). Run
in this order from the repo root, with the virtualenv from `make setup`
active:

```bash
git clone https://github.com/mmalekzadeh/motion-sense.git data/motion-sense
git -C data/motion-sense checkout 261cb0d60611a049ab38127738e42cc510ed50f4
unzip data/motion-sense/data/A_DeviceMotion_data.zip -d data/motion-sense/data

python experiments/extract_and_fit.py \
  --run-name motionsense_decision_audit --rebuild-features
python experiments/anchor_refit.py
python experiments/episode_transport.py
python experiments/enumerate_future_episode_splits.py
```

`extract_and_fit.py` builds the deterministic, hash-verified window-feature
cache (`results/motionsense_decision_windows_w50_s50_t100.csv`, ~66MB, not
committed) and the recording split, backbone, and per-user gain outputs. The
other three scripts refit the per-user adapter many times against that
cache (fast -- seconds to low minutes on CPU, since every fit here is a
small affine map on 141-dimensional frozen logits, not a deep network).

Their outputs (`results/episode_transport.csv`,
`results/episode_split_enumeration.csv`,
`results/episode_split_peruser.csv`,
`results/motionsense_decision_audit_per_user.csv`,
`results/motionsense_decision_audit_recording_split.csv`) are committed to
this repository as the frozen Stage 2 input cache, so `make figures` and
`make test` never need to run any of this. Rerun Stage 1 to verify
reproducibility end-to-end, or after changing the feature extraction or
fitting logic.

`MOTIONSENSE_LICENSE.txt` is the upstream dataset's own MIT license.
