"""Select each user's gain under the shipped (standard) outcome design from
the full 216-design enumeration, for use in Figure 1(b) and the headline
per-user counts (NPos, NNeg, median/mean gain).

The shipped protocol -- one short recording per activity -- is design index
`is_shipped` inside `results/episode_split_enumeration.csv`. This script
only selects that design's rows out of the per-user enumeration; it does not
refit anything.

Stage 2: reads only the committed `results/episode_split_enumeration.csv`
and `results/episode_split_peruser.csv`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main() -> pd.DataFrame:
    enum = pd.read_csv(RES / "episode_split_enumeration.csv")
    peruser = pd.read_csv(RES / "episode_split_peruser.csv")

    selected = enum.loc[enum.is_shipped, ["fold", "design"]]
    shipped = peruser.merge(selected, on=["fold", "design"], validate="many_to_one")
    assert len(shipped) == 24 and shipped["subject"].nunique() == 24, (
        "expected exactly one shipped-design row per of the 24 target subjects"
    )

    out_path = RES / "figure1b_identity_standard_design.csv"
    shipped.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(shipped)} rows)")
    return shipped


if __name__ == "__main__":
    main()
