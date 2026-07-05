"""First Keras LSTM baseline on the frozen journey-level split.

Reads  : data/processed/station_delays_model.csv
         data/processed/journey_split.csv            (frozen)
Writes : data/processed/lstm_baseline_report.txt

Research question: can the full observed journey sequence outperform
XGBoost's engineered current-state + 2-lag representation?

Prediction semantics: for every non-destination station k, the input is the
sequence of stations 1..k (destination row never appears in any input) and
the target is the journey's final destination arrival delay.

Design notes:
- Sequence features (time-varying, 5): arrival_delay_model,
  departure_delay_model, arrival_was_imputed, departure_was_imputed,
  journey_completion. No manual lag features - history comes from the
  sequence itself.
- Context features (current-station, 2): total_distance, remaining_distance.
- Scaling: StandardScaler fit on TRAIN-split non-destination rows only.
  Binary flags are not scaled. Target stays in minutes.
- Padding/masking: post-padding to the dataset-wide max prefix length with
  PAD_VALUE = -999.0 and Masking(mask_value=-999.0). Keras masks a timestep
  only when ALL its features equal the mask value; scaled delays are bounded
  around +/-12, journey_completion around +/-2 and flags are 0/1, so no
  valid timestep can ever equal the pad vector. (Plain zero-padding would
  not carry this guarantee: a scaled timestep could in principle be all
  zeros.)
- Deliberately small model, Huber loss (delta=10 min) for the extreme tail,
  early stopping on val MAE with restore_best_weights.
- Test set is evaluated exactly once, after the configuration is finalized
  on train+validation.
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
REPORT_TXT = PROJECT_ROOT / "data" / "processed" / "lstm_baseline_report.txt"

SEED = 42
PAD_VALUE = -999.0
TARGET = "target_destination_delay"

SEQ_CONT = ["arrival_delay_model", "departure_delay_model", "journey_completion"]
SEQ_FLAGS = ["arrival_was_imputed", "departure_was_imputed"]
SEQ_FEATURES = SEQ_CONT + SEQ_FLAGS          # order = channels in the tensor
CTX_FEATURES = ["total_distance", "remaining_distance"]

# Frozen benchmark numbers (from xgboost_baseline_report.txt) - not recomputed.
XGB_FROZEN = {"val": {"MAE": 20.50, "RMSE": 53.68, "R2": 0.8482},
              "test": {"MAE": 18.37, "RMSE": 53.65, "R2": 0.8320}}


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
    df["position_frac"] = grp.cumcount() / (n - 1)   # 0 = origin, 1 = destination
    return df


def fit_scalers(df: pd.DataFrame):
    """Scalers are fit on TRAIN-split non-destination rows only."""
    fit_rows = df[(df["split"] == "train") & (df["is_destination"] == 0)]
    seq_scaler = StandardScaler().fit(fit_rows[SEQ_CONT])
    ctx_scaler = StandardScaler().fit(fit_rows[CTX_FEATURES])
    assert int(seq_scaler.n_samples_seen_) == len(fit_rows)
    assert int(ctx_scaler.n_samples_seen_) == len(fit_rows)
    return seq_scaler, ctx_scaler, len(fit_rows)


def build_samples(df, seq_scaler, ctx_scaler):
    """Every prefix [1..k] for each non-destination station k of each journey."""
    scaled = df.copy()
    scaled[SEQ_CONT] = seq_scaler.transform(scaled[SEQ_CONT])
    scaled[CTX_FEATURES] = ctx_scaler.transform(scaled[CTX_FEATURES])

    nondest = scaled[scaled["is_destination"] == 0]
    max_len = int(nondest.groupby("journey_id").size().max())

    out = {}
    for split in ["train", "val", "test"]:
        seqs, ctxs, ys, meta = [], [], [], []
        part = nondest[nondest["split"] == split]
        for jid, sub in part.groupby("journey_id", sort=False):
            feats = sub[SEQ_FEATURES].to_numpy(dtype=np.float32)
            ctx = sub[CTX_FEATURES].to_numpy(dtype=np.float32)
            tgt = float(sub[TARGET].iloc[0])
            raw_arr = sub["arrival_delay_model"].to_numpy()   # note: scaled copy
            for k in range(len(sub)):
                pad = np.full((max_len, len(SEQ_FEATURES)), PAD_VALUE,
                              dtype=np.float32)
                pad[: k + 1] = feats[: k + 1]
                seqs.append(pad)
                ctxs.append(ctx[k])
                ys.append(tgt)
                meta.append((jid, int(sub["station_sequence"].iloc[k]),
                             float(sub["position_frac"].iloc[k])))
        m = pd.DataFrame(meta, columns=["journey_id", "station_sequence",
                                        "position_frac"])
        m["train_number"] = m["journey_id"].str.split("_").str[0]
        out[split] = {"X_seq": np.stack(seqs), "X_ctx": np.stack(ctxs),
                      "y": np.array(ys, dtype=np.float32), "meta": m}
    return out, max_len


def naive_predictions(df, split):
    """Naive baselines use the UNSCALED current-station model delays."""
    part = df[(df["split"] == split) & (df["is_destination"] == 0)]
    return (part["arrival_delay_model"].to_numpy(),
            part["departure_delay_model"].to_numpy(),
            part[TARGET].to_numpy())


def build_model(max_len: int) -> keras.Model:
    seq_in = keras.Input(shape=(max_len, len(SEQ_FEATURES)), name="sequence")
    x = layers.Masking(mask_value=PAD_VALUE)(seq_in)
    x = layers.LSTM(32, dropout=0.1)(x)
    ctx_in = keras.Input(shape=(len(CTX_FEATURES),), name="context")
    h = layers.Concatenate()([x, ctx_in])
    h = layers.Dense(16, activation="relu")(h)
    h = layers.Dropout(0.1)(h)
    out = layers.Dense(1, name="final_delay")(h)
    model = keras.Model([seq_in, ctx_in], out)
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
    data, max_len = build_samples(df, seq_scaler, ctx_scaler)

    model = build_model(max_len)

    es = keras.callbacks.EarlyStopping(monitor="val_mae", mode="min",
                                       patience=20, restore_best_weights=True)
    rlr = keras.callbacks.ReduceLROnPlateau(monitor="val_mae", mode="min",
                                            factor=0.5, patience=8,
                                            min_lr=1e-5)
    hist = model.fit(
        [data["train"]["X_seq"], data["train"]["X_ctx"]], data["train"]["y"],
        validation_data=([data["val"]["X_seq"], data["val"]["X_ctx"]],
                         data["val"]["y"]),
        epochs=200, batch_size=64, callbacks=[es, rlr], verbose=2)
    best_epoch = int(np.argmin(hist.history["val_mae"])) + 1

    preds = {}
    for s in ["val", "test"]:
        naive_arr, naive_dep, y_naive = naive_predictions(df, s)
        y = data[s]["y"]
        assert np.allclose(y, y_naive), "sample order mismatch vs naive rows"
        lstm_pred = model.predict(
            [data[s]["X_seq"], data[s]["X_ctx"]], verbose=0).ravel()
        preds[s] = {"A_naive_arrival": naive_arr,
                    "B_naive_departure": naive_dep,
                    "D_lstm": lstm_pred}

    lines = []
    add = lines.append
    add("LSTM BASELINE REPORT (frozen split, causal prefix sequences)")
    add(f"backend: keras {keras.__version__} ({keras.backend.backend()}) | seed {SEED}")
    add("")
    add(f"sequence features ({len(SEQ_FEATURES)}): {', '.join(SEQ_FEATURES)}")
    add(f"context features  ({len(CTX_FEATURES)}): {', '.join(CTX_FEATURES)}")
    add(f"prefix samples: train={len(data['train']['y'])} "
        f"val={len(data['val']['y'])} test={len(data['test']['y'])}")
    add(f"max sequence length: {max_len} | padding: post, PAD_VALUE={PAD_VALUE}, "
        f"Masking(mask_value={PAD_VALUE})")
    add("")
    add("scaling proof (StandardScaler fit on train non-destination rows only):")
    add(f"  rows used to fit both scalers: {n_fit} "
        f"(= train-split non-destination rows)")
    add(f"  seq scaler mean : {np.round(seq_scaler.mean_, 3).tolist()}  ({SEQ_CONT})")
    add(f"  seq scaler scale: {np.round(seq_scaler.scale_, 3).tolist()}")
    add(f"  ctx scaler mean : {np.round(ctx_scaler.mean_, 3).tolist()}  ({CTX_FEATURES})")
    add(f"  ctx scaler scale: {np.round(ctx_scaler.scale_, 3).tolist()}")
    add(f"  flags {SEQ_FLAGS} passed through unscaled")
    add("")
    add(f"model parameters: {model.count_params()} | best epoch: {best_epoch} "
        f"(early stop patience 20, restore_best_weights=True)")
    add("")

    for s in ["val", "test"]:
        add(f"--- {s.upper()} metrics ---")
        for name, p in preds[s].items():
            m = metrics(data[s]["y"], p)
            add(f"  {name:<18} MAE={m['MAE']:7.2f}  RMSE={m['RMSE']:7.2f}  R2={m['R2']:7.4f}")
        xf = XGB_FROZEN[s]
        add(f"  {'C_xgboost(frozen)':<18} MAE={xf['MAE']:7.2f}  RMSE={xf['RMSE']:7.2f}  R2={xf['R2']:7.4f}")
        add("")

    for s in ["val", "test"]:
        add(f"--- {s.upper()} MAE by journey-position bucket ---")
        add(bucket_mae(data[s]["meta"], data[s]["y"], preds[s]).to_string())
        add("")

    for s in ["val", "test"]:
        add(f"--- {s.upper()} MAE per train number ---")
        add(per_train_mae(data[s]["meta"], data[s]["y"], preds[s]).to_string())
        add("")

    # failure cases
    for s in ["val", "test"]:
        m = data[s]["meta"].copy()
        m["y"] = data[s]["y"]
        m["pred"] = preds[s]["D_lstm"]
        m["abs_err"] = (m["y"] - m["pred"]).abs()
        add(f"--- 12 worst {s.upper()} errors (LSTM) ---")
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
