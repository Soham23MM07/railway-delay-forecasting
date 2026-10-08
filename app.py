"""Streamlit demo: predict a journey's final destination delay using the
frozen XGBoost baseline.

Run with:
    streamlit run app.py

Reuses build_features()/the feature list from src/train_xgboost_baseline.py
so the demo trains on the exact same 17 causal features as the frozen
benchmark - no feature-engineering logic is duplicated here.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train_xgboost_baseline import build_features, TARGET, SEED  # noqa: E402

st.set_page_config(page_title="Railway Delay Predictor", page_icon="🚆", layout="centered")


@st.cache_resource(show_spinner="Training XGBoost on the frozen split...")
def load_model_and_data():
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
    model.fit(X["train"], y["train"], eval_set=[(X["val"], y["val"])], verbose=False)
    test_mae = mean_absolute_error(y["test"], model.predict(X["test"]))
    return model, features, df, test_mae


@st.cache_data
def route_for_train(_df: pd.DataFrame, train_number: int) -> pd.DataFrame:
    jid = _df.loc[_df["train_number"] == train_number, "journey_id"].iloc[0]
    route = _df[_df["journey_id"] == jid].sort_values("station_sequence")
    return route[["station_sequence", "station_code", "station_name",
                  "distance_km", "is_destination"]].reset_index(drop=True)


model, features, df, test_mae = load_model_and_data()
trains = sorted(df["train_number"].unique().tolist())

st.title("🚆 Railway Delay Predictor")
st.caption(
    f"Predicts final destination arrival delay using the causal XGBoost "
    f"baseline (frozen-split test MAE ≈ {test_mae:.1f} min)."
)

with st.sidebar:
    st.header("Journey so far")
    train_number = st.selectbox("Train number", trains, index=0)

    route = route_for_train(df, train_number)
    stoppable = route[route["is_destination"] == 0]
    station_labels = [
        f"{r.station_sequence}. {r.station_name} ({r.station_code}) — {r.distance_km:.0f} km"
        for r in stoppable.itertuples()
    ]
    station_idx = st.selectbox(
        "Current station (just departed)",
        range(len(stoppable)),
        format_func=lambda i: station_labels[i],
        index=min(2, len(stoppable) - 1),
    )
    current = stoppable.iloc[station_idx]

    st.divider()
    arrival_delay = st.number_input("Current arrival delay (min)", value=10, step=1)
    departure_delay = st.number_input("Current departure delay (min)", value=arrival_delay, step=1)

    with st.expander("Advanced: delay history (optional)"):
        st.caption("Leave at 0 if unknown — this mirrors an early-journey station with no prior lag data.")
        prev_arrival_1 = st.number_input("Delay 1 station back (arrival)", value=arrival_delay, step=1)
        prev_arrival_2 = st.number_input("Delay 2 stations back (arrival)", value=arrival_delay, step=1)
        prev_departure_1 = st.number_input("Delay 1 station back (departure)", value=departure_delay, step=1)
        prev_departure_2 = st.number_input("Delay 2 stations back (departure)", value=departure_delay, step=1)
        have_lag_1 = st.checkbox("These lag values are real (not a guess)", value=False)

    predict_clicked = st.button("Predict final delay", type="primary", use_container_width=True)

total_distance = route["distance_km"].max()
remaining_distance = total_distance - current["distance_km"]
journey_completion = current["distance_km"] / total_distance

st.subheader("Selected journey state")
c1, c2, c3 = st.columns(3)
c1.metric("Station", f"{current['station_code']}")
c2.metric("Journey completion", f"{journey_completion*100:.0f}%")
c3.metric("Remaining distance", f"{remaining_distance:.0f} km")

if predict_clicked:
    row = pd.DataFrame([{
        "arrival_delay_model": arrival_delay,
        "departure_delay_model": departure_delay,
        "arrival_was_imputed": 0,
        "departure_was_imputed": 0,
        "station_sequence": current["station_sequence"],
        "distance_km": current["distance_km"],
        "total_distance": total_distance,
        "remaining_distance": remaining_distance,
        "journey_completion": journey_completion,
        "prev_arrival_1": prev_arrival_1,
        "prev_arrival_2": prev_arrival_2,
        "prev_departure_1": prev_departure_1,
        "prev_departure_2": prev_departure_2,
        "prev_arrival_1_missing": 0 if have_lag_1 else 1,
        "prev_arrival_2_missing": 0 if have_lag_1 else 1,
        "prev_departure_1_missing": 0 if have_lag_1 else 1,
        "prev_departure_2_missing": 0 if have_lag_1 else 1,
    }])[features]

    prediction = float(model.predict(row)[0])
    naive = arrival_delay

    st.subheader("Prediction")
    p1, p2 = st.columns(2)
    p1.metric("XGBoost predicted final delay", f"{prediction:.0f} min")
    p2.metric("Naive guess (current delay stays same)", f"{naive:.0f} min",
              delta=f"{prediction - naive:+.0f} min vs naive")

    st.caption(
        "This is a demo built on a frozen, previously-trained benchmark. "
        "Predictions for trains/routes far from the training distribution "
        "should be treated with caution (see the project's README limitations)."
    )
else:
    st.info("Set the journey details in the sidebar, then click **Predict final delay**.")
