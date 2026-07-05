"""Create the single permanent journey-level train/val/test split (70/15/15).

Reads  : data/processed/station_delays_model.csv
Writes : data/processed/journey_split.csv        (journey_id, split)
         data/processed/journey_split_report.txt

Design:
- The unit of splitting is the JOURNEY (journey_id = train_number + date).
  Station rows are never split independently; every row inherits its
  journey's assignment, so no journey can leak across splits.
- Stratified by journey-level target_destination_delay bins. The target is
  heavily skewed (42% exact zeros, only 15 journeys > 300 min), so bins are
  chosen to force the rare severe-delay journeys to spread across all three
  splits instead of landing in one by chance:
      on_time   <= 0
      minor     (0, 15]
      moderate  (15, 60]
      high      (60, 300]
      severe    (300, 600]
      extreme   > 600
- Two-stage stratified split: 70% train vs 30% holdout, then holdout split
  50/50 into validation and test, both stratified on the same bins.
- Deterministic: BASE_SEED is tried first; if any train number ends up with
  zero journeys in some split, the next seed is tried (recorded in the
  report). The search itself is deterministic, so the result is fully
  reproducible.

Every model in this project (XGBoost, LSTM, baselines) must obtain its
rows via merge_split() below and must never re-split.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CSV = PROJECT_ROOT / "data" / "processed" / "station_delays_model.csv"
SPLIT_CSV = PROJECT_ROOT / "data" / "processed" / "journey_split.csv"
REPORT_TXT = PROJECT_ROOT / "data" / "processed" / "journey_split_report.txt"

BASE_SEED = 42
BIN_EDGES = [-np.inf, 0, 15, 60, 300, 600, np.inf]
BIN_LABELS = ["on_time(<=0)", "minor(0-15]", "moderate(15-60]",
              "high(60-300]", "severe(300-600]", "extreme(>600)"]


def journey_table() -> pd.DataFrame:
    """One row per journey: journey_id, train_number, target, delay bin."""
    df = pd.read_csv(MODEL_CSV)
    j = (df[df["is_destination"] == 1]
         [["journey_id", "train_number", "target_destination_delay"]]
         .reset_index(drop=True))
    assert j["journey_id"].is_unique and len(j) == df["journey_id"].nunique()
    j["delay_bin"] = pd.cut(j["target_destination_delay"],
                            bins=BIN_EDGES, labels=BIN_LABELS)
    assert j["delay_bin"].notna().all()
    return j


def try_split(j: pd.DataFrame, seed: int) -> pd.Series:
    """Stratified 70/15/15 on delay bins; returns split label per journey."""
    train_ids, rest_ids = train_test_split(
        j["journey_id"], test_size=0.30, random_state=seed,
        stratify=j["delay_bin"])
    rest = j[j["journey_id"].isin(rest_ids)]
    val_ids, test_ids = train_test_split(
        rest["journey_id"], test_size=0.50, random_state=seed,
        stratify=rest["delay_bin"])
    split = pd.Series(index=j["journey_id"], dtype=object, name="split")
    split.loc[train_ids] = "train"
    split.loc[val_ids] = "val"
    split.loc[test_ids] = "test"
    return split


def full_train_coverage(j: pd.DataFrame, split: pd.Series) -> bool:
    """Every train number must have at least one journey in every split."""
    cov = (j.assign(split=split.loc[j["journey_id"]].values)
           .groupby(["train_number", "split"], observed=True).size()
           .unstack(fill_value=0))
    return (cov[["train", "val", "test"]] > 0).all().all()


def make_split() -> tuple[pd.DataFrame, int]:
    j = journey_table()
    for seed in range(BASE_SEED, BASE_SEED + 100):
        split = try_split(j, seed)
        if full_train_coverage(j, split):
            out = split.reset_index()
            return out, seed
    raise RuntimeError("no seed in range gave full train coverage")


def merge_split(model_df: pd.DataFrame | None = None,
                split_path: Path = SPLIT_CSV) -> pd.DataFrame:
    """Merge the permanent split assignment onto the modeling dataset.

    Usage from any model script:
        from create_journey_split import merge_split
        df = merge_split()
        train_df = df[df["split"] == "train"]
    """
    if model_df is None:
        model_df = pd.read_csv(MODEL_CSV)
    split = pd.read_csv(split_path)
    merged = model_df.merge(split, on="journey_id", how="left", validate="m:1")
    assert merged["split"].notna().all(), "journey without split assignment"
    assert len(merged) == len(model_df), "merge changed row count"
    return merged


def build_report(j: pd.DataFrame, assign: pd.DataFrame, seed: int,
                 merged: pd.DataFrame) -> str:
    jj = j.merge(assign, on="journey_id", validate="1:1")
    lines = []
    add = lines.append
    add("JOURNEY SPLIT REPORT - journey_split.csv")
    add(f"unit: journey_id | stratified on delay bins | seed used: {seed}"
        + ("" if seed == BASE_SEED else f" (seeds {BASE_SEED}..{seed-1} failed train coverage)"))
    add("")

    add("sizes:")
    for s in ["train", "val", "test"]:
        nj = (jj["split"] == s).sum()
        nr = (merged["split"] == s).sum()
        add(f"  {s:<6} journeys={nj:>4} ({100*nj/len(jj):.1f}%)   station rows={nr:>6} ({100*nr/len(merged):.1f}%)")
    add("")

    add("target_destination_delay per split (journey level):")
    stats = jj.groupby("split")["target_destination_delay"].agg(
        ["count", "mean", "median", "std", "min", "max"]).round(1)
    add(stats.reindex(["train", "val", "test"]).to_string())
    add("")

    add("delay-bin counts per split:")
    tab = (jj.groupby(["delay_bin", "split"], observed=True).size()
           .unstack(fill_value=0)[["train", "val", "test"]])
    add(tab.to_string())
    add("")

    add("severe/extreme journeys per split:")
    for s in ["train", "val", "test"]:
        sub = jj[jj["split"] == s]["target_destination_delay"]
        add(f"  {s:<6} >300 min: {(sub > 300).sum():>2}   >600 min: {(sub > 600).sum():>2}")
    add("")

    add("train-number coverage (journeys per split):")
    cov = (jj.groupby(["train_number", "split"], observed=True).size()
           .unstack(fill_value=0)[["train", "val", "test"]])
    add(cov.to_string())
    add(f"  trains present in all three splits: {(cov > 0).all(axis=1).sum()} / {len(cov)}")
    add("")

    n_multi = (assign.groupby("journey_id")["split"].nunique() > 1).sum()
    add("integrity proofs:")
    add(f"  journeys assigned exactly once: {assign['journey_id'].is_unique} "
        f"({len(assign)} assignments for {j['journey_id'].nunique()} journeys)")
    add(f"  journeys in more than one split: {n_multi}")
    add(f"  set(train) & set(val) & set(test) pairwise overlap: "
        f"{len(set(assign[assign.split=='train'].journey_id) & set(assign[assign.split=='val'].journey_id)) + len(set(assign[assign.split=='train'].journey_id) & set(assign[assign.split=='test'].journey_id)) + len(set(assign[assign.split=='val'].journey_id) & set(assign[assign.split=='test'].journey_id))}")
    add(f"  every station row inherits its journey's split: "
        f"{(merged.groupby('journey_id')['split'].nunique() == 1).all()}")
    add(f"  rows train+val+test = {(merged['split'].isin(['train','val','test'])).sum()} of {len(merged)}")
    return "\n".join(lines)


def main():
    j = journey_table()
    assign, seed = make_split()

    # hard integrity assertions before saving
    assert len(assign) == len(j) == 796 or len(assign) == len(j), \
        "assignment count != journey count"
    assert assign["journey_id"].is_unique, "journey assigned more than once"
    assert set(assign["journey_id"]) == set(j["journey_id"]), \
        "assignment set != journey set"
    assert set(assign["split"]) == {"train", "val", "test"}

    assign.to_csv(SPLIT_CSV, index=False)

    merged = merge_split()
    report = build_report(j, assign, seed, merged)
    REPORT_TXT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nsaved split : {SPLIT_CSV}")
    print(f"saved report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
