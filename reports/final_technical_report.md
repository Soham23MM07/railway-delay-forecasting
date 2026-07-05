# Predicting Final Destination Arrival Delay from a Partially Observed Train Journey

**Final technical report** · All numbers in this document are taken verbatim from the frozen experiment reports in `data/processed/` and were cross-audited by `src/create_final_analysis.py` (22/22 consistency checks passed; see `reports/final_metrics_audit.txt`).

---

## 1. Problem definition

At station *k*, immediately after the train departs station *k*, predict the **final destination arrival delay** of that journey using **only information observable up to station *k***. Every non-destination station of every journey is one prediction sample (a "prefix"); the destination row itself is never a model input, because at that point the answer is already known.

This is a real operational question — "given how the journey has gone so far, how late will this train finally be?" — and it puts a hard causality constraint on every stage of the pipeline: features, imputation, and evaluation must never touch information from stations after *k*.

## 2. Dataset

Station-level running data for **20 daily long-distance trains** collected from the RailRadar API, May 1 – June 20 2026:

| property | value |
|---|---|
| trains | 20 (= **10 up/down route pairs**: Shatabdi, Rajdhani, and superfast expresses) |
| journeys (train × date) | **796** |
| station records | **13,844** |
| observed dates | **47** (4 calendar days missing) |
| route lengths | 359 km / 8 halts → 3,030 km / 44 halts |
| route structure | **fixed per train** — every journey of a train stops at the identical halt set |
| target | final destination arrival delay, minutes |

The target is **highly skewed and zero-inflated**: 42 % of journeys arrive with exactly 0 minutes delay, 60 % within 5 minutes, while a rare tail — 15 journeys above 300 minutes, 7 above 600, maximum 1,093 — carries most of the variance. Almost all extreme journeys belong to one route pair (Karnataka Express 12627/12628, including three diverted runs). Delay dynamics have a strong structural pattern: delay accumulates mid-journey and is partially recovered near the terminus through schedule padding (median journey recovers 24 minutes from its en-route peak).

## 3. The first misleading result

The initial modeling round produced impressively low errors — and could not simply be celebrated. Error analysis and a row-level audit of the processed dataset revealed two problems: the modeling CSV had been hand-edited in Excel (a stray label column and one corrupted date cell), and — far more seriously — missing delay values had been filled with **bidirectional linear interpolation**, which uses *future* stations to fill *earlier* missing values.

The clearest real example is the diverted journey `12627_2026-05-15`. The train was ~58 minutes late before the diversion; the four diverted stations reported no data; the train re-appeared 821 minutes late:

```
old leaked pattern    : 58 → 210.6 → 668.4 → 821     (smooth ramp interpolated
                                                       backwards FROM the future 821)
correct causal pattern: 58 →  58   →  58   → 821     (last known value carried
                                                       forward; the jump is a surprise)
```

The interpolated 210.6 → 668.4 ramp encodes the future disruption inside the *input features*. A model trained on it learns to "anticipate" delays it could never see in deployment — its offline metrics are inflated by information that will not exist at prediction time. This is textbook target-adjacent leakage, and it invalidated the first results. In total, 1,905 interior departure values and 144 interior arrival values (touching 81 % of journeys) had been filled this way, and the same cleaning step had also destroyed genuine origin departure delays (297 journeys) by overwriting them with zero.

## 4. Causal preprocessing (rebuild)

`src/prepare_modeling_dataset.py` rebuilds the modeling dataset from the untouched raw file with strictly causal rules:

- **Prediction-point semantics**: a sample at station *k* represents the state *after departing k*, so arrival and departure delays at stations 1..*k* are legitimately known.
- **Imputation priority** for a missing value at station *k*: (1) **same-station copy** of the observed counterpart (arrival↔departure correlate at 0.99); (2) **LOCF** — the most recent *observed* event at an earlier station; (3) default 0. No value at station *k* is ever derived from station *k+1* or later — verified by an independent re-derivation pass with hard assertions.
- **Missingness flags**: every delay value carries a status (`observed` / `imputed_same_station` / `imputed_locf` / `imputed_default` / `structural`) plus binary flags, so the model can distinguish real zeros from filled ones.
- **Structural missingness** handled explicitly: origins have no arrival, destinations no departure; recorded origin departure delays are *preserved* (they correlate 0.29 with the final delay).
- **Raw columns preserved untouched** for auditing; models train only on the `*_model` columns.
- **Target** taken exclusively from the raw destination arrival delay — never from an imputed column. All 796 destination arrivals were genuinely observed; the validation report shows **zero target mismatches**.

## 5. Frozen journey-level split

`src/create_journey_split.py` produced one permanent 70/15/15 split **by journey** (`journey_id = train_number + date`): **557 train / 119 validation / 120 test journeys**, seed 42, stratified on six delay bins (≤0, 0–15, 15–60, 60–300, 300–600, >600 minutes) so the rare severe journeys spread across splits (>600-min journeys land 5/1/1). Proofs in the split report: zero journey overlap, every journey assigned exactly once, every station row inherits its journey's split, and **all 20 trains are represented in every split**. Every model in the project consumes this same assignment file; no model ever re-splits.

## 6. Baselines

**Naive arrival carry-forward** (predict final delay = current arrival delay) is a serious baseline, not a strawman: current delay correlates 0.99 with the final delay near the end of a journey, and Indian long-distance schedules make "the train stays about this late" a strong prior. Any model that cannot beat it clearly is not learning anything useful.

| baseline | test MAE | test RMSE | test R² |
|---|---|---|---|
| Naive arrival carry-forward | 26.28 | 56.01 | 0.817 |
| Naive departure | 28.98 | 56.30 | 0.815 |

The naive baselines' main weakness is systematic: they cannot express end-of-route recovery, so they over-predict late in the journey.

## 7. XGBoost (frozen benchmark)

17 causal features: current-state (`arrival/departure_delay_model` + imputation flags), route/position context (`station_sequence`, `distance_km`, `total_distance`, `remaining_distance`, `journey_completion`), and **two-step causal lag features** with lag-missing indicators (lags never cross journey boundaries). Trained with MAE objective and early stopping on validation.

**Test: MAE 18.37, RMSE 53.65, R² 0.8320** — a ~30 % MAE improvement over naive, growing with journey position (final bucket: 6.98 vs 18.60 naive) because the model learns the recovery pattern.

Two honest caveats. First, feature importance is dominated by `total_distance` (0.29): with fixed routes, route length uniquely identifies the train pair, so this feature acts partly as a **route-identity feature** encoding per-route delay propensity rather than transferable physics. Second, the model fails on **late-onset extreme disruptions** — journeys that run near-schedule for most of their route and then collapse (e.g. `12625_2026-06-05`: ≤39 min late through 72 % of the route, then +380). No causal model can predict these from station history alone.

## 8. LSTM V1 — full journey history only

*Research question: can full journey history outperform manually engineered two-step lag features?*

Every prefix `[station 1..k]` becomes a variable-length sequence (max length 43) of 5 features (delays, imputation flags, journey completion), post-padded with a collision-free sentinel (−999) and masked; a 32-unit LSTM encodes it; `total_distance`/`remaining_distance` join as a context branch; a Dense(16) head predicts the final delay. Huber loss (δ=10 min), 5,441 parameters, scalers fit on training rows only. All sequence models (V1, V2, dense ablation) are implemented with the **Keras 3 API** (`keras.Model`, `keras.layers.LSTM`, Keras callbacks) running on the **PyTorch backend** (`KERAS_BACKEND=torch` set before import); TensorFlow is not required.

**Test: MAE 19.34, RMSE 55.32, R² 0.8213** — decisively beats naive (26.28) but loses to XGBoost, and the loss is concentrated in one place: the **80–100 % bucket (10.77 vs 6.98)**. Diagnosis: near the destination the answer is approximately *current delay minus recovery*, but V1's current-station information must pass through the recurrent hidden-state bottleneck, arriving diluted — while XGBoost consumes the current delay directly.

## 9. LSTM V2 — one controlled change

V2 adds exactly one thing: the five already-scaled **current-station features are fed directly into the dense head** (a skip connection around the LSTM). Data, split, scaling, layer sizes, loss, optimizer, callbacks, seed — all byte-identical to V1; +80 parameters (5,521).

**Test: MAE 19.34 → 18.79; the 80–100 % bucket improved 10.77 → 8.21 (−24 %)**, and the 60–80 % bucket (13.93) now *beats* XGBoost (15.37). Early buckets were unchanged — precisely the signature predicted by the bottleneck hypothesis. The experiment confirms the diagnosis: V1 was not lacking information, it was *weakening* the most important input by forcing it through the recurrence.

## 10. Dense-only ablation

*Final research question: does full sequence history actually add value beyond current state + context?*

The ablation removes the entire sequence/LSTM branch from V2 — leaving the identical 7 scaled inputs (5 current-state + 2 context) feeding the identical Dense(16) head. **145 parameters.**

**Test: MAE 19.64** vs V2's 18.79 — removing history costs **0.85 minutes MAE** (1.93 on validation). V2 beats dense-only in **all five position buckets** on both validation and test, with the largest, most consistent gap around **60–80 % journey completion** (13.93 vs 15.71) — where estimating how much accumulated delay will be recovered depends on the delay *trajectory*, not just its level.

Stated carefully: **full journey history provides measurable but modest predictive value.** The consistency of the gap (both eval sets, all buckets) makes it real rather than noise; its size (~0.9 min test MAE) makes it modest. Equally notable: a 145-parameter model on 7 numbers reaches 19.64 — current state + route context carries the large majority of the predictive signal.

## 11. Final model comparison

| model | inputs | params | test MAE | test RMSE | test R² |
|---|---|---|---|---|---|
| Naive departure | current departure delay | — | 28.98 | 56.30 | 0.815 |
| Naive arrival | current arrival delay | — | 26.28 | 56.01 | 0.817 |
| Dense-only | current state + context | 145 | 19.64 | 54.56 | 0.8262 |
| LSTM V1 | full history + context | 5,441 | 19.34 | 55.32 | 0.8213 |
| LSTM V2 | history + current-state skip + context | 5,521 | 18.79 | **53.54** | **0.8327** |
| XGBoost | current state + context + 2 lags | — | **18.37** | 53.65 | 0.8320 |

- **Best MAE: XGBoost (18.37).** **Best RMSE and R²: LSTM V2 (53.54 / 0.8327).**
- The two are effectively very close but optimize different parts of the error distribution: XGBoost (MAE objective) is slightly better on typical journeys; V2 (Huber loss) handles the extreme tail slightly better. With only 120 test journeys, the headline difference (0.42 min MAE) is within noise — a statistical tie.
- A satisfying symmetry closes the ablation ladder: V1 (history without skip) ≈ dense-only (skip without history) — each partial model loses roughly the same amount — and V2, combining both, recovers parity with XGBoost.

## 12. Engineering conclusion

- **XGBoost is the preferred practical deployment model**: equal accuracy at a fraction of the operational complexity — no scaling pipeline, no padding/masking, seconds to train, interpretable feature importances.
- **LSTM V2 proves that sequence history contains additional signal** — the dense-only ablation quantifies it at ~0.9 min test MAE, concentrated at 60–80 % journey completion.
- That additional sequence complexity buys only a **modest gain over current state + context**.
- **Two lag features are enough** for XGBoost to capture most of the useful recent history — the full 43-step sequence adds little beyond them on this dataset.
- **More complex architectures are not justified at this dataset size** (796 journeys, 20 fixed routes). The binding constraint is data, not model capacity.

## 13. Limitations

- **796 journeys** over 7 weeks — small by sequence-modeling standards.
- **20 fixed routes**: models partly memorize per-route delay propensity (`total_distance` as route identity); nothing here demonstrates generalization to unseen routes.
- **Zero-inflated target** (42 % exact zeros) biases all models toward small predictions.
- **Rare extremes**: 15 journeys > 300 min dominate RMSE; per-bucket and per-train evaluation was used to prevent pooled metrics from hiding this.
- **No external operational information**: no weather, signal failures, congestion, diversions, maintenance blocks, or network state — precisely the variables that cause the unpredictable failures.
- **Late-onset disruptions are fundamentally unpredictable** from station history alone (see `fig7_failure_case.png`); they bound every causal model's achievable error on this data.
- The **random stratified split does not test temporal drift** as strongly as a temporal (train-on-May / test-on-June) split would; delay levels demonstrably varied week to week.

## 14. Future work (grounded)

1. **Collect more journeys over a longer period** — the single highest-value action; every model comparison here is data-limited.
2. **Add operational and external features** (weather, section congestion, rolling-stock, holiday calendar) — the only route to predicting late-onset disruptions.
3. **Evaluate temporal generalization** with a strict past→future split.
4. **Test route generalization** by holding out entire trains.
5. **Uncertainty estimates** (e.g. quantile regression) so extreme-event predictions carry calibrated confidence rather than a single under-shooting point estimate.

No architecture escalation (attention, transformers, ensembles) is recommended until the dataset grows: the ablation ladder shows the current ceiling is set by information, not capacity.

---

*Figures: `reports/figures/fig1–fig7`. Source of truth: frozen reports in `data/processed/`. Audit: `reports/final_metrics_audit.txt`.*
