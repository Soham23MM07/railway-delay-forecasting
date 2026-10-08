"""Build the trustworthy modeling dataset from the EXPANDED raw source
(original 796 journeys + new June21-July20 collection, minus 53 flagged
missing-GPS-tracking pairs). Parallel "_20k" pipeline - the original
prepare_modeling_dataset.py / frozen 796-journey results are untouched.

Reads  : data/raw/station_delays_clean_20k+.csv   (never modified)
Writes : data/processed/station_delays_model_20k+.csv
         data/processed/preparation_report_20k+.txt

Rules (approved):
- Original arrival_delay_minutes / departure_delay_minutes are kept untouched.
- Model-ready columns arrival_delay_model / departure_delay_model are created
  with strictly CAUSAL imputation: a value at station k may only be derived
  from station k or earlier stations of the same journey. No dataset-level or
  train-level statistics are used.
- Prediction-point semantics: a sample at station k represents the state
  AFTER departing station k, so arrival_k and departure_k are both known.
- Imputation priority for a missing interior value at station k:
    1. imputed_same_station : copy the observed counterpart at station k
    2. imputed_locf         : most recent OBSERVED event at an earlier station
                              (departure_j preferred over arrival_j, walking back)
    3. imputed_default      : 0 (nothing observed earlier in the journey)
- Origin arrival and destination departure are structurally undefined:
  filled with 0 and flagged 'structural' (unless the API actually recorded a
  value, which is then kept as 'observed').
- Origin departure: kept when recorded; otherwise 0 + 'imputed_default'.
- Negative delays (early arrivals/departures) are preserved as-is.
- target_destination_delay comes from the ORIGINAL raw arrival delay of the
  destination row, never from a model column.
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "station_delays_clean_20k+.csv"

OUT_PATH = PROJECT_ROOT / "data" / "processed" / "station_delays_model_20k+.csv"
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "preparation_report_20k+.txt"

OBSERVED = "observed"
SAME_STATION = "imputed_same_station"
LOCF = "imputed_locf"
DEFAULT = "imputed_default" 
STRUCTURAL = "structural"


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df["train_number"] = pd.to_numeric(df["train_number"]).astype(int)
    # DD-MM-YYYY -> ISO so lexicographic order == chronological order
    parsed = pd.to_datetime(df["journey_date"], format="%d-%m-%Y", errors="raise")
    df["journey_date"] = parsed.dt.strftime("%Y-%m-%d")
    df["journey_id"] = df["train_number"].astype(str) + "_" + df["journey_date"]
    df = df.sort_values(
        ["train_number", "journey_date", "station_sequence"], kind="mergesort"
    ).reset_index(drop=True)
    return df


def add_row_flags(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("journey_id", sort=False)
    df["is_origin"] = (grp.cumcount() == 0).astype(int)
    df["is_destination"] = (grp.cumcount(ascending=False) == 0).astype(int)
    return df


def last_observed_before(arr: np.ndarray, dep: np.ndarray, i: int) -> float:
    """Most recent observed event strictly before statio n i of one journey:
    departure of station j is preferred over arrival of station j, walking
    backwards from j = i-1. Returns np.nan if nothing was ever observed."""
    for j in range(i - 1, -1, -1):
        if not np.isnan(dep[j]):
            return dep[j]
        if not np.isnan(arr[j]):
            return arr[j]
    return np.nan


def impute_journey(arr: np.ndarray, dep: np.ndarray):
    """Causal imputation for one journey (arrays ordered by station_sequence).
    Returns model values and status labels for both columns."""
    n = len(arr)
    arr_model, dep_model = arr.copy(), dep.copy()
    arr_status = np.array([OBSERVED] * n, dtype=object)
    dep_status = np.array([OBSERVED] * n, dtype=object)

    for i in range(n):
        is_origin, is_dest = i == 0, i == n - 1

        if np.isnan(arr[i]):
            if is_origin:
                arr_model[i], arr_status[i] = 0.0, STRUCTURAL
            else:
                # destination arrival must never need imputation; asserted later
                if not np.isnan(dep[i]):
                    arr_model[i], arr_status[i] = dep[i], SAME_STATION
                else:
                    prev = last_observed_before(arr, dep, i)
                    if not np.isnan(prev):
                        arr_model[i], arr_status[i] = prev, LOCF
                    else:
                        arr_model[i], arr_status[i] = 0.0, DEFAULT

        if np.isnan(dep[i]):
            if is_dest:
                dep_model[i], dep_status[i] = 0.0, STRUCTURAL
            elif is_origin:
                dep_model[i], dep_status[i] = 0.0, DEFAULT
            else:
                if not np.isnan(arr[i]):
                    dep_model[i], dep_status[i] = arr[i], SAME_STATION
                else:
                    prev = last_observed_before(arr, dep, i)
                    if not np.isnan(prev):
                        dep_model[i], dep_status[i] = prev, LOCF
                    else:
                        dep_model[i], dep_status[i] = 0.0, DEFAULT

    return arr_model, dep_model, arr_status, dep_status


def impute(df: pd.DataFrame) -> pd.DataFrame:
    arr_model = np.empty(len(df))
    dep_model = np.empty(len(df))
    arr_status = np.empty(len(df), dtype=object)
    dep_status = np.empty(len(df), dtype=object)

    arr_all = df["arrival_delay_minutes"].to_numpy(dtype=float)
    dep_all = df["departure_delay_minutes"].to_numpy(dtype=float)

    for jid, idx in df.groupby("journey_id", sort=False).indices.items():
        a, d, sa, sd = impute_journey(arr_all[idx], dep_all[idx])
        arr_model[idx], dep_model[idx] = a, d
        arr_status[idx], dep_status[idx] = sa, sd

    df["arrival_delay_model"] = arr_model
    df["departure_delay_model"] = dep_model
    df["arrival_delay_status"] = arr_status
    df["departure_delay_status"] = dep_status
    df["arrival_was_imputed"] = df["arrival_delay_status"].isin(
        [SAME_STATION, LOCF, DEFAULT]
    ).astype(int)
    df["departure_was_imputed"] = df["departure_delay_status"].isin(
        [SAME_STATION, LOCF, DEFAULT]
    ).astype(int)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    # Target is taken from the ORIGINAL raw column at the destination row.
    dest = df[df["is_destination"] == 1]
    target_map = dest.set_index("journey_id")["arrival_delay_minutes"]
    df["target_destination_delay"] = df["journey_id"].map(target_map)
    return df


def run_assertions(df: pd.DataFrame, n_rows_raw: int, n_journeys_raw: int) -> dict:
    """Hard assertions; raises AssertionError on any violation.
    Returns counters used in the report."""
    grp = df.groupby("journey_id", sort=False)

    # 4. row and journey counts unchanged
    assert len(df) == n_rows_raw, "row count changed"
    assert df["journey_id"].nunique() == n_journeys_raw, "journey count changed"

    # 5/6/7. exactly one origin and one destination per journey, flagged
    assert (grp["is_origin"].sum() == 1).all(), "journey without exactly one origin"
    assert (grp["is_destination"].sum() == 1).all(), "journey without exactly one destination"
    origin_rows = df[df["is_origin"] == 1].set_index("journey_id")["station_sequence"]
    dest_rows = df[df["is_destination"] == 1].set_index("journey_id")["station_sequence"]
    assert (origin_rows == grp["station_sequence"].min()).all(), "origin is not min sequence"
    assert (dest_rows == grp["station_sequence"].max()).all(), "destination is not max sequence"

    # 1. no destination arrival delay is imputed
    dest_status = df.loc[df["is_destination"] == 1, "arrival_delay_status"]
    assert (dest_status == OBSERVED).all(), "a destination arrival delay was imputed"

    # 2. every target equals the raw destination arrival delay
    dest = df[df["is_destination"] == 1]
    assert dest["arrival_delay_minutes"].notna().all(), "missing raw destination arrival"
    tgt_check = dest["target_destination_delay"] == dest["arrival_delay_minutes"]
    n_target_mismatch = int((~tgt_check).sum())
    assert n_target_mismatch == 0, f"{n_target_mismatch} target mismatches"
    assert df["target_destination_delay"].notna().all(), "journey without target"
    # target must also equal the raw arrival delay at the max-sequence row
    max_seq_arr = df.loc[
        df.groupby("journey_id", sort=False)["station_sequence"].idxmax(),
        ["journey_id", "arrival_delay_minutes"],
    ].set_index("journey_id")["arrival_delay_minutes"]
    per_journey_tgt = grp["target_destination_delay"].first()
    assert per_journey_tgt.equals(max_seq_arr.reindex(per_journey_tgt.index).rename(
        "target_destination_delay")) or (
        per_journey_tgt.values == max_seq_arr.reindex(per_journey_tgt.index).values
    ).all(), "target != final station raw arrival delay"

    # 3. no imputed value derives from a later station (independent re-derivation)
    arr_raw = df["arrival_delay_minutes"].to_numpy(dtype=float)
    dep_raw = df["departure_delay_minutes"].to_numpy(dtype=float)
    for jid, idx in df.groupby("journey_id", sort=False).indices.items():
        a_raw, d_raw = arr_raw[idx], dep_raw[idx]
        a_mod = df["arrival_delay_model"].to_numpy()[idx]
        d_mod = df["departure_delay_model"].to_numpy()[idx]
        a_st = df["arrival_delay_status"].to_numpy()[idx]
        d_st = df["departure_delay_status"].to_numpy()[idx]
        for i in range(len(idx)):
            for st, mod, raw_own, raw_other in (
                (a_st[i], a_mod[i], a_raw[i], d_raw[i]),
                (d_st[i], d_mod[i], d_raw[i], a_raw[i]),
            ):
                if st == OBSERVED:
                    assert not np.isnan(raw_own) and mod == raw_own, \
                        f"observed value altered at {jid} pos {i}"
                elif st == SAME_STATION:
                    assert mod == raw_other, f"same-station mismatch at {jid} pos {i}"
                elif st == LOCF:
                    expect = last_observed_before(a_raw, d_raw, i)
                    assert not np.isnan(expect) and mod == expect, \
                        f"LOCF used non-causal source at {jid} pos {i}"
                elif st in (DEFAULT, STRUCTURAL):
                    assert mod == 0.0, f"non-zero default at {jid} pos {i}"

    # model columns are complete
    assert df["arrival_delay_model"].notna().all(), "NaN left in arrival_delay_model"
    assert df["departure_delay_model"].notna().all(), "NaN left in departure_delay_model"

    # duplicates
    n_dup_code = int(df.duplicated(["train_number", "journey_date", "station_code"]).sum())
    n_dup_seq = int(df.duplicated(["train_number", "journey_date", "station_sequence"]).sum())
    assert n_dup_code == 0 and n_dup_seq == 0, "duplicate station records found"

    return {"n_target_mismatch": n_target_mismatch,
            "n_dup_code": n_dup_code, "n_dup_seq": n_dup_seq}


def build_report(df: pd.DataFrame, raw: pd.DataFrame, checks: dict) -> str:
    lines = []
    add = lines.append
    add("PREPARATION REPORT - station_delays_model.csv")
    add(f"source: {RAW_PATH.name} (untouched)")
    add("")
    add(f"journeys : {df['journey_id'].nunique()}")
    add(f"rows     : {len(df)}  (raw: {len(raw)})")
    add(f"trains   : {df['train_number'].nunique()}")
    add(f"dates    : {df['journey_date'].nunique()}  "
        f"({df['journey_date'].min()} .. {df['journey_date'].max()})")
    add("")
    add("missing values BEFORE (raw columns, preserved as-is):")
    for col in raw.columns:
        n = int(raw[col].isna().sum())
        if n:
            add(f"  {col:.<28} {n}")
    add("")
    add("missing values AFTER in model columns:")
    add(f"  arrival_delay_model....... {int(df['arrival_delay_model'].isna().sum())}")
    add(f"  departure_delay_model..... {int(df['departure_delay_model'].isna().sum())}")
    add("")
    add("imputation breakdown (arrival_delay_status / departure_delay_status):")
    a = df["arrival_delay_status"].value_counts()
    d = df["departure_delay_status"].value_counts()
    for k in [OBSERVED, SAME_STATION, LOCF, DEFAULT, STRUCTURAL]:
        add(f"  {k:.<24} arrival: {int(a.get(k, 0)):>6}   departure: {int(d.get(k, 0)):>6}")
    add(f"  total imputed (arrival).... {int(df['arrival_was_imputed'].sum())}")
    add(f"  total imputed (departure). {int(df['departure_was_imputed'].sum())}")
    add("")
    add("target validation:")
    add(f"  target mismatches vs raw destination arrival: {checks['n_target_mismatch']}")
    add("  destination arrival imputed anywhere: 0 (hard-asserted)")
    add("")
    add("duplicate checks:")
    add(f"  duplicate (train, date, station_code): {checks['n_dup_code']}")
    add(f"  duplicate (train, date, station_sequence): {checks['n_dup_seq']}")
    add("")
    per_journey = df[df["is_destination"] == 1]["target_destination_delay"]
    add("target / extreme-delay journeys:")
    add(f"  target == 0 ............. {int((per_journey == 0).sum())}")
    add(f"  target < 0 (early) ...... {int((per_journey < 0).sum())}")
    add(f"  target > 60 ............. {int((per_journey > 60).sum())}")
    add(f"  target > 300 ............ {int((per_journey > 300).sum())}")
    add(f"  target > 600 ............ {int((per_journey > 600).sum())}")
    add(f"  target max .............. {per_journey.max():.0f}")
    add("")
    neg_raw = int((raw["arrival_delay_minutes"] < 0).sum()
                  + (raw["departure_delay_minutes"] < 0).sum())
    neg_obs = int(((df["arrival_delay_model"] < 0)
                   & (df["arrival_delay_status"] == OBSERVED)).sum()
                  + ((df["departure_delay_model"] < 0)
                     & (df["departure_delay_status"] == OBSERVED)).sum())
    neg_imp = int(((df["arrival_delay_model"] < 0)
                   & (df["arrival_was_imputed"] == 1)).sum()
                  + ((df["departure_delay_model"] < 0)
                     & (df["departure_was_imputed"] == 1)).sum())
    add(f"negative observed delays preserved: raw={neg_raw}, model columns={neg_obs} "
        f"(+{neg_imp} negative imputed value inherited from a negative causal source)")
    add("")
    add("all hard assertions passed.")
    return "\n".join(lines)


def main():
    raw = pd.read_csv(RAW_PATH)  # pristine copy for before/after comparison
    df = load_raw()
    n_rows_raw = len(df)
    n_journeys_raw = df["journey_id"].nunique()

    df = add_row_flags(df)
    df = impute(df)
    df = add_target(df)

    checks = run_assertions(df, n_rows_raw, n_journeys_raw)
    report = build_report(df, raw, checks)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nsaved dataset: {OUT_PATH}")
    print(f"saved report : {REPORT_PATH}")
    print(f"\ncolumns ({len(df.columns)}):")
    for c in df.columns:
        print(f"  {c} ({df[c].dtype})")


if __name__ == "__main__":
    main()
