"""Corrected XGBoost baseline on the frozen journey-level split.

Reads  : data/processed/station_delays_model.csv
         data/processed/journey_split.csv          (frozen - never regenerated)
Writes : data/processed/xgboost_baseline_report.txt

Prediction semantics: at station k, immediately AFTER departure from k,
predict the journey's final destination arrival delay.

Rules:
- Destination rows are never model inputs (is_destination == 1 excluded).
- Only causally available features: current-state *_model columns and their
  imputation flags, static route geometry (known from the schedule), and
  lags strictly within the journey (never across journey boundaries).
- Early lags that don't exist get 0 plus a *_missing indicator column, so a
  filled 0 is distinguishable from a genuine 0-minute delay.
- train -> fit, val -> early stopping, test -> single final evaluation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from create_journey_split_20k import merge_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_TXT = PROJECT_ROOT / "data" / "processed" / "xgboost_20k+_report.txt"

TARGET = "target_destination_delay"
SEED = 42

LAG_SOURCES = {"prev_arrival": "arrival_delay_model",
               "prev_departure": "departure_delay_model"}
N_LAGS = 2


def build_features() -> tuple[pd.DataFrame, list[str]]:
    df = merge_split()
    df = df.sort_values(["train_number", "journey_date", "station_sequence"],
                        kind="mergesort").reset_index(drop=True)
    grp = df.groupby("journey_id", sort=False)

    # Route geometry (static schedule knowledge, causally available anywhere
    # on the route). total_distance = distance_km of the destination row.
    dest_dist = df.loc[df["is_destination"] == 1].set_index("journey_id")["distance_km"]
    df["total_distance"] = df["journey_id"].map(dest_dist)
    df["remaining_distance"] = df["total_distance"] - df["distance_km"]
    df["journey_completion"] = df["distance_km"] / df["total_distance"]

    # Position fraction (for bucketed evaluation only, not a feature):
    # 0 = origin, 1 = destination.
    n = grp["station_code"].transform("size")
    df["position_frac"] = grp.cumcount() / (n - 1)

    # Lags strictly inside each journey
    lag_features, lag_flags = [], []
    for name, src in LAG_SOURCES.items():
        for k in range(1, N_LAGS + 1):
            col, flag = f"{name}_{k}", f"{name}_{k}_missing"
            df[col] = grp[src].shift(k)
            df[flag] = df[col].isna().astype(int)
            df[col] = df[col].fillna(0.0)
            lag_features.append(col)
            lag_flags.append(flag)

    features = [
        "arrival_delay_model", "departure_delay_model",
        "arrival_was_imputed", "departure_was_imputed",
        "station_sequence", "distance_km",
        "total_distance", "remaining_distance", "journey_completion",
        *lag_features, *lag_flags,
    ]

    # destination rows are inputs never - drop AFTER all derivations
    df = df[df["is_destination"] == 0].reset_index(drop=True)
    assert df[features + [TARGET]].notna().all().all()
    return df, features


def metrics(y_true, y_pred) -> dict:
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred)}


def bucket_mae(sub: pd.DataFrame, preds: dict) -> pd.DataFrame:
    bins = [0, .2, .4, .6, .8, 1.0]
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    b = pd.cut(sub["position_frac"], bins=bins, labels=labels, include_lowest=True)
    rows = {}
    for name, p in preds.items():
        err = (sub[TARGET].to_numpy() - p)
        rows[name] = pd.Series(np.abs(err)).groupby(b.reset_index(drop=True), observed=True).mean()
    out = pd.DataFrame(rows)
    out["n"] = b.value_counts().reindex(labels)
    return out.round(2)


def per_train_mae(sub: pd.DataFrame, preds: dict) -> pd.DataFrame:
    rows = {}
    for name, p in preds.items():
        err = pd.Series(np.abs(sub[TARGET].to_numpy() - p), index=sub.index)
        rows[name] = err.groupby(sub["train_number"]).mean()
    out = pd.DataFrame(rows)
    out["n"] = sub.groupby("train_number").size()
    return out.round(2)


def main():
    df, features = build_features()
    parts = {s: df[df["split"] == s] for s in ["train", "val", "test"]}
    X = {s: p[features] for s, p in parts.items()}
    y = {s: p[TARGET] for s, p in parts.items()}

    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:absoluteerror",
        eval_metric="mae",
        early_stopping_rounds=50,
        random_state=SEED,
        n_jobs=-1,
    )
    
    model.fit(X["train"], y["train"],
              eval_set=[(X["val"], y["val"])], verbose=False)

    preds = {}
    for s in ["val", "test"]:
        preds[s] = {
            "A_naive_arrival": parts[s]["arrival_delay_model"].to_numpy(),
            "B_naive_departure": parts[s]["departure_delay_model"].to_numpy(),
            "C_xgboost": model.predict(X[s]),
        }

    lines = []
    add = lines.append
    add("XGBOOST BASELINE REPORT (frozen split, causal features)")
    add(f"rows: train={len(parts['train'])} val={len(parts['val'])} test={len(parts['test'])}"
        f" | best_iteration={model.best_iteration} of {model.n_estimators}")
    add("")
    add(f"features ({len(features)}): {', '.join(features)}")
    add("")

    for s in ["val", "test"]:
        add(f"--- {s.upper()} metrics ---")
        for name, p in preds[s].items():
            m = metrics(y[s], p)
            add(f"  {name:<18} MAE={m['MAE']:7.2f}  RMSE={m['RMSE']:7.2f}  R2={m['R2']:7.4f}")
        add("")

    for s in ["val", "test"]:
        add(f"--- {s.upper()} MAE by journey-position bucket ---")
        add(bucket_mae(parts[s], preds[s]).to_string())
        add("")

    for s in ["val", "test"]:
        add(f"--- {s.upper()} MAE per train number ---")
        add(per_train_mae(parts[s], preds[s]).to_string())
        add("")

    imp = (pd.DataFrame({"feature": features,
                         "gain": model.feature_importances_})
           .sort_values("gain", ascending=False).reset_index(drop=True))
    add("--- feature importance (gain) ---")
    add(imp.round(4).to_string())
    add("")

    # worst validation errors -> failure-pattern inspection
    v = parts["val"].copy()
    v["pred"] = preds["val"]["C_xgboost"]
    v["abs_err"] = (v[TARGET] - v["pred"]).abs()
    worst = v.nlargest(15, "abs_err")[
        ["journey_id", "station_code", "position_frac", "arrival_delay_model",
         TARGET, "pred", "abs_err"]]
    add("--- 15 worst VALIDATION errors ---")
    add(worst.round(2).to_string(index=False))

    report = "\n".join(lines)
    REPORT_TXT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nsaved report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
