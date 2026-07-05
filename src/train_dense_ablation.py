"""Dense-only ablation: LSTM V2 minus the entire sequence/LSTM branch.

Reads  : data/processed/station_delays_model.csv
         data/processed/journey_split.csv            (frozen)
Writes : data/processed/dense_ablation_report.txt

Research question: does the full journey sequence add predictive value
beyond the current station state and route context?

This is an ablation, not a new model. Removed vs V2: sequence input,
Masking, LSTM(32), and the LSTM representation in the concatenation.
Everything else is identical to train_lstm_v2.py:

    current station k features (5, scaled) ---\
                                               +-> Concat -> Dense(16, relu)
    context: total/remaining distance (2) ----/       -> Dropout(0.1) -> Dense(1)

Same dataset, split, samples, scaling (train-only fit, flags unscaled),
Huber(delta=10), Adam(1e-3), callbacks, seed 42, batch 64, max 200 epochs.
Test set evaluated exactly once after early stopping restores best weights.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import keras
from keras import layers

from create_journey_split import merge_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_TXT = PROJECT_ROOT / "data" / "processed" / "dense_ablation_report.txt"

SEED = 42
TARGET = "target_destination_delay"

SEQ_CONT = ["arrival_delay_model", "departure_delay_model", "journey_completion"]
SEQ_FLAGS = ["arrival_was_imputed", "departure_was_imputed"]
CUR_FEATURES = SEQ_CONT + SEQ_FLAGS          # identical to V2's current-state branch
CTX_FEATURES = ["total_distance", "remaining_distance"]

# Frozen benchmarks - never recomputed.
FROZEN = {
    "val": {"C_xgboost(frozen)": {"MAE": 20.50, "RMSE": 53.68, "R2": 0.8482},
            "D_lstm_v1(frozen)": {"MAE": 20.97, "RMSE": 54.26, "R2": 0.8449},
            "E_lstm_v2(frozen)": {"MAE": 20.69, "RMSE": 53.62, "R2": 0.8485}},
    "test": {"C_xgboost(frozen)": {"MAE": 18.37, "RMSE": 53.65, "R2": 0.8320},
             "D_lstm_v1(frozen)": {"MAE": 19.34, "RMSE": 55.32, "R2": 0.8213},
             "E_lstm_v2(frozen)": {"MAE": 18.79, "RMSE": 53.54, "R2": 0.8327}},
}
FROZEN_TEST_BUCKETS = pd.DataFrame({
    "C_xgboost(frozen)": [24.35, 24.34, 17.87, 15.37, 6.98],
    "D_lstm_v1(frozen)": [25.64, 23.63, 18.51, 15.64, 10.77],
    "E_lstm_v2(frozen)": [25.74, 24.85, 18.27, 13.93, 8.21],
}, index=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])


def load_frame() -> pd.DataFrame:
    df = merge_split()
    df = df.sort_values(["train_number", "journey_date", "station_sequence"],
                        kind="mergesort").reset_index(drop=True)
    grp = df.groupby("journey_id", sort=False)
    dest_dist = df.loc[df["is_destination"] == 1].set_index("journey_id")["distance_km"]
    df["total_distance"] = df["journey_id"].map(dest_dist)
    df["remaining_distance"] = df["total_distance"] - df["distance_km"]
    df["journey_completion"] = df["distance_km"] / df["total_distance"]
    n = grp["station_code"].transform("size")
    df["position_frac"] = grp.cumcount() / (n - 1)
    return df


def fit_scalers(df: pd.DataFrame):
    fit_rows = df[(df["split"] == "train") & (df["is_destination"] == 0)]
    seq_scaler = StandardScaler().fit(fit_rows[SEQ_CONT])
    ctx_scaler = StandardScaler().fit(fit_rows[CTX_FEATURES])
    assert int(seq_scaler.n_samples_seen_) == len(fit_rows)
    assert int(ctx_scaler.n_samples_seen_) == len(fit_rows)
    return seq_scaler, ctx_scaler, len(fit_rows)


def build_samples(df, seq_scaler, ctx_scaler):
    """Identical sample generation and ordering as V2, minus padded tensors."""
    scaled = df.copy()
    scaled[SEQ_CONT] = seq_scaler.transform(scaled[SEQ_CONT])
    scaled[CTX_FEATURES] = ctx_scaler.transform(scaled[CTX_FEATURES])

    nondest = scaled[scaled["is_destination"] == 0]
    out = {}
    for split in ["train", "val", "test"]:
        curs, ctxs, ys, meta = [], [], [], []
        part = nondest[nondest["split"] == split]
        for jid, sub in part.groupby("journey_id", sort=False):
            feats = sub[CUR_FEATURES].to_numpy(dtype=np.float32)
            ctx = sub[CTX_FEATURES].to_numpy(dtype=np.float32)
            tgt = float(sub[TARGET].iloc[0])
            for k in range(len(sub)):
                curs.append(feats[k])        # same scaled station-k vector as V2
                ctxs.append(ctx[k])
                ys.append(tgt)
                meta.append((jid, int(sub["station_sequence"].iloc[k]),
                             float(sub["position_frac"].iloc[k])))
        m = pd.DataFrame(meta, columns=["journey_id", "station_sequence",
                                        "position_frac"])
        m["train_number"] = m["journey_id"].str.split("_").str[0]
        out[split] = {"X_cur": np.stack(curs), "X_ctx": np.stack(ctxs),
                      "y": np.array(ys, dtype=np.float32), "meta": m}
    return out


def naive_predictions(df, split):
    part = df[(df["split"] == split) & (df["is_destination"] == 0)]
    return (part["arrival_delay_model"].to_numpy(),
            part["departure_delay_model"].to_numpy(),
            part[TARGET].to_numpy())


def build_model() -> keras.Model:
    cur_in = keras.Input(shape=(len(CUR_FEATURES),), name="current_state")
    ctx_in = keras.Input(shape=(len(CTX_FEATURES),), name="context")
    h = layers.Concatenate()([cur_in, ctx_in])
    h = layers.Dense(16, activation="relu")(h)
    h = layers.Dropout(0.1)(h)
    out = layers.Dense(1, name="final_delay")(h)
    model = keras.Model([cur_in, ctx_in], out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=keras.losses.Huber(delta=10.0),
                  metrics=["mae"])
    return model


def metrics(y_true, y_pred) -> dict:
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred)}


def bucket_mae(meta, y, preds: dict) -> pd.DataFrame:
    bins = [0, .2, .4, .6, .8, 1.0]
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    b = pd.cut(meta["position_frac"], bins=bins, labels=labels,
               include_lowest=True)
    out = {}
    for name, p in preds.items():
        out[name] = pd.Series(np.abs(y - p)).groupby(b, observed=True).mean()
    tab = pd.DataFrame(out)
    tab["n"] = b.value_counts().reindex(labels)
    return tab.round(2)


def per_train_mae(meta, y, preds: dict) -> pd.DataFrame:
    out = {}
    for name, p in preds.items():
        out[name] = pd.Series(np.abs(y - p)).groupby(meta["train_number"]).mean()
    tab = pd.DataFrame(out)
    tab["n"] = meta.groupby("train_number").size()
    return tab.round(2)


def main():
    keras.utils.set_random_seed(SEED)

    df = load_frame()
    seq_scaler, ctx_scaler, n_fit = fit_scalers(df)
    data = build_samples(df, seq_scaler, ctx_scaler)

    model = build_model()

    # proof: no sequence branch anywhere in the graph
    layer_types = [type(l).__name__ for l in model.layers]
    assert "LSTM" not in layer_types and "Masking" not in layer_types
    assert all(len(l.output.shape) == 2 for l in model.layers), \
        "a layer carries a time dimension"

    es = keras.callbacks.EarlyStopping(monitor="val_mae", mode="min",
                                       patience=20, restore_best_weights=True)
    rlr = keras.callbacks.ReduceLROnPlateau(monitor="val_mae", mode="min",
                                            factor=0.5, patience=8,
                                            min_lr=1e-5)
    hist = model.fit(
        [data["train"]["X_cur"], data["train"]["X_ctx"]], data["train"]["y"],
        validation_data=([data["val"]["X_cur"], data["val"]["X_ctx"]],
                         data["val"]["y"]),
        epochs=200, batch_size=64, callbacks=[es, rlr], verbose=2)
    best_epoch = int(np.argmin(hist.history["val_mae"])) + 1

    preds = {}
    for s in ["val", "test"]:
        naive_arr, naive_dep, y_naive = naive_predictions(df, s)
        assert np.allclose(data[s]["y"], y_naive)
        p = model.predict([data[s]["X_cur"], data[s]["X_ctx"]],
                          verbose=0).ravel()
        preds[s] = {"A_naive_arrival": naive_arr,
                    "B_naive_departure": naive_dep,
                    "F_dense_only": p}

    lines = []
    add = lines.append
    add("DENSE-ONLY ABLATION REPORT - LSTM V2 minus the sequence/LSTM branch")
    add(f"backend: keras {keras.__version__} ({keras.backend.backend()}) | seed {SEED}")
    add("")
    add(f"current-state features (5): {', '.join(CUR_FEATURES)}  [scaled as in V2]")
    add(f"context features (2): {', '.join(CTX_FEATURES)}  [scaled as in V2]")
    add(f"prefix samples: train={len(data['train']['y'])} "
        f"val={len(data['val']['y'])} test={len(data['test']['y'])}")
    add(f"scalers fit on {n_fit} train non-destination rows only (asserted)")
    add("")
    add("proof sequence branch is absent:")
    add(f"  layers: {layer_types}")
    add("  asserted: no LSTM, no Masking, no layer with a time dimension;")
    add("  model inputs are only (5,) current-state and (2,) context vectors")
    add("")
    add(f"model parameters: {model.count_params()} "
        f"(V1: 5441, V2: 5521) | best epoch: {best_epoch} of {len(hist.history['mae'])} run")
    add("")

    for s in ["val", "test"]:
        add(f"--- {s.upper()} metrics ---")
        for name, p in preds[s].items():
            m = metrics(data[s]["y"], p)
            add(f"  {name:<18} MAE={m['MAE']:7.2f}  RMSE={m['RMSE']:7.2f}  R2={m['R2']:7.4f}")
        for name, m in FROZEN[s].items():
            add(f"  {name:<18} MAE={m['MAE']:7.2f}  RMSE={m['RMSE']:7.2f}  R2={m['R2']:7.4f}")
        add("")

    for s in ["val", "test"]:
        add(f"--- {s.upper()} MAE by journey-position bucket ---")
        tab = bucket_mae(data[s]["meta"], data[s]["y"], preds[s])
        if s == "test":
            tab = tab.join(FROZEN_TEST_BUCKETS)
        add(tab.to_string())
        add("")

    for s in ["val", "test"]:
        add(f"--- {s.upper()} MAE per train number ---")
        add(per_train_mae(data[s]["meta"], data[s]["y"], preds[s]).to_string())
        add("")

    for s in ["val", "test"]:
        m = data[s]["meta"].copy()
        m["y"] = data[s]["y"]
        m["pred"] = preds[s]["F_dense_only"]
        m["abs_err"] = (m["y"] - m["pred"]).abs()
        add(f"--- 12 worst {s.upper()} errors (dense-only) ---")
        add(m.nlargest(12, "abs_err")[
            ["journey_id", "station_sequence", "position_frac", "y", "pred",
             "abs_err"]].round(2).to_string(index=False))
        add("")

    report = "\n".join(lines)
    REPORT_TXT.write_text(report, encoding="utf-8")
    print(report)
    print(f"saved report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
