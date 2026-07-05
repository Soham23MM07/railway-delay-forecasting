"""Final project analysis: audit frozen reports and build comparison figures.

Reads ONLY existing frozen artifacts (no model is trained or recomputed):
    data/processed/preparation_report.txt
    data/processed/journey_split_report.txt
    data/processed/xgboost_baseline_report.txt
    data/processed/lstm_baseline_report.txt          (LSTM V1)
    data/processed/lstm_v2_report.txt
    data/processed/dense_ablation_report.txt
    data/processed/station_delays_model.csv          (journey profile for fig 7)

Writes NEW files only:
    reports/figures/fig1_test_mae.png
    reports/figures/fig2_test_rmse.png
    reports/figures/fig3_test_r2.png
    reports/figures/fig4_bucket_mae.png
    reports/figures/fig5_ablation_ladder.png
    reports/figures/fig6_per_train_mae.png
    reports/figures/fig7_failure_case.png
    reports/final_metrics_audit.txt

Phase 1 audit: every headline number is parsed from the report files and
cross-checked (a) against the frozen values quoted in later reports and
(b) against the project's frozen results table. Any mismatch raises.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROC = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
AUDIT_TXT = PROJECT_ROOT / "reports" / "final_metrics_audit.txt"

REPORTS = {
    "prep": PROC / "preparation_report.txt",
    "split": PROC / "journey_split_report.txt",
    "xgb": PROC / "xgboost_baseline_report.txt",
    "v1": PROC / "lstm_baseline_report.txt",
    "v2": PROC / "lstm_v2_report.txt",
    "dense": PROC / "dense_ablation_report.txt",
}

# The project's frozen results table (test set) - the audit target.
EXPECTED_TEST = {
    "Naive arrival": (26.28, 56.01, 0.817),
    "Naive departure": (28.98, 56.30, 0.815),
    "Dense-only": (19.64, 54.56, 0.8262),
    "LSTM V1": (19.34, 55.32, 0.8213),
    "LSTM V2": (18.79, 53.54, 0.8327),
    "XGBoost": (18.37, 53.65, 0.8320),
}

MODEL_COLORS = {
    "Naive arrival": "#9e9e9e",
    "Naive departure": "#c7c7c7",
    "Dense-only": "#8da0cb",
    "LSTM V1": "#66c2a5",
    "LSTM V2": "#1b7837",
    "XGBoost": "#d95f02",
}
BUCKETS = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]


# ----------------------------------------------------------------- parsing

def read(name: str) -> str:
    return REPORTS[name].read_text(encoding="utf-8")


def parse_metrics(text: str, section: str) -> dict:
    """Parse '  <name> MAE= x RMSE= y R2= z' lines under a metrics section."""
    out = {}
    in_sec = False
    for line in text.splitlines():
        if line.startswith(f"--- {section} metrics ---"):
            in_sec = True
            continue
        if in_sec:
            m = re.match(r"\s+(\S+)\s+MAE=\s*([-\d.]+)\s+RMSE=\s*([-\d.]+)"
                         r"\s+R2=\s*([-\d.]+)", line)
            if m:
                out[m.group(1)] = tuple(float(m.group(i)) for i in (2, 3, 4))
            elif line.strip() == "":
                break
    return out


def parse_table(text: str, header: str) -> pd.DataFrame:
    """Parse a pandas to_string() table that follows `header` in the report.
    Layout: column-names line, index-name line, data rows, blank line."""
    lines = text.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(header))
    cols = lines[i + 1].split()
    rows = []
    for line in lines[i + 3:]:
        if not line.strip():
            break
        parts = line.split()
        rows.append([parts[0]] + [float(x) for x in parts[1:]])
    df = pd.DataFrame(rows, columns=["index"] + cols).set_index("index")
    return df


def parse_worst_test(text: str, journey: str) -> pd.DataFrame:
    """Rows of the worst-TEST-errors table belonging to one journey."""
    lines = text.splitlines()
    i = next(k for k, l in enumerate(lines) if "worst TEST errors" in l)
    rows = []
    for line in lines[i + 2:]:
        if not line.strip():
            break
        p = line.split()
        if p[0] == journey:
            rows.append({"station_sequence": int(p[1]),
                         "position_frac": float(p[2]),
                         "y": float(p[3]), "pred": float(p[4])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- audit

def audit() -> tuple[dict, list[str]]:
    log = []
    xgb, v1, v2, dense = read("xgb"), read("v1"), read("v2"), read("dense")

    t = {}  # canonical parsed test metrics per model
    t["Naive arrival"] = parse_metrics(xgb, "TEST")["A_naive_arrival"]
    t["Naive departure"] = parse_metrics(xgb, "TEST")["B_naive_departure"]
    t["XGBoost"] = parse_metrics(xgb, "TEST")["C_xgboost"]
    t["LSTM V1"] = parse_metrics(v1, "TEST")["D_lstm"]
    t["LSTM V2"] = parse_metrics(v2, "TEST")["E_lstm_v2"]
    t["Dense-only"] = parse_metrics(dense, "TEST")["F_dense_only"]

    # (a) frozen copies quoted inside later reports must match the originals
    checks = [
        ("XGB frozen in V2 report", parse_metrics(v2, "TEST")["C_xgboost(frozen)"], t["XGBoost"]),
        ("XGB frozen in dense report", parse_metrics(dense, "TEST")["C_xgboost(frozen)"], t["XGBoost"]),
        ("V1 frozen in V2 report", parse_metrics(v2, "TEST")["D_lstm_v1(frozen)"], t["LSTM V1"]),
        ("V1 frozen in dense report", parse_metrics(dense, "TEST")["D_lstm_v1(frozen)"], t["LSTM V1"]),
        ("V2 frozen in dense report", parse_metrics(dense, "TEST")["E_lstm_v2(frozen)"], t["LSTM V2"]),
        ("naive arrival in V1 report", parse_metrics(v1, "TEST")["A_naive_arrival"], t["Naive arrival"]),
        ("naive arrival in V2 report", parse_metrics(v2, "TEST")["A_naive_arrival"], t["Naive arrival"]),
        ("naive arrival in dense report", parse_metrics(dense, "TEST")["A_naive_arrival"], t["Naive arrival"]),
        ("naive departure in V1 report", parse_metrics(v1, "TEST")["B_naive_departure"], t["Naive departure"]),
    ]
    for name, got, ref in checks:
        ok = all(abs(g - r) <= 0.011 for g, r in zip(got, ref))
        log.append(f"  [{'OK' if ok else 'MISMATCH'}] {name}: {got} vs {ref}")
        assert ok, f"cross-report mismatch: {name}"

    # (b) parsed values must match the project's frozen results table
    for model, (mae, rmse, r2) in EXPECTED_TEST.items():
        got = t[model]
        ok = (abs(got[0] - mae) <= 0.005 and abs(got[1] - rmse) <= 0.005
              and abs(got[2] - r2) <= 0.0006)
        log.append(f"  [{'OK' if ok else 'MISMATCH'}] frozen table {model}: "
                   f"report={got} expected=({mae}, {rmse}, {r2})")
        assert ok, f"frozen-table mismatch: {model}"

    # (c) pipeline-level invariants from prep and split reports
    prep, split = read("prep"), read("split")
    inv = [
        ("796 journeys (prep)", "journeys : 796" in prep),
        ("13844 rows (prep)", "rows     : 13844" in prep),
        ("0 target mismatches (prep)", "target mismatches vs raw destination arrival: 0" in prep),
        ("assertions passed (prep)", "all hard assertions passed." in prep),
        ("split sizes 557/119/120", all(s in split for s in
         ["train  journeys= 557", "val    journeys= 119", "test   journeys= 120"])),
        ("zero split overlap", "journeys in more than one split: 0" in split),
        ("20/20 train coverage", "trains present in all three splits: 20 / 20" in split),
    ]
    for name, ok in inv:
        log.append(f"  [{'OK' if ok else 'MISMATCH'}] {name}")
        assert ok, f"invariant failed: {name}"

    return t, log


# ------------------------------------------------------------------ figures

def annotate_bars(ax, bars, fmt="{:.2f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()),
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)


def bar_figure(test, key_idx, title, ylabel, fname, fmt="{:.2f}", ylim=None):
    models = list(EXPECTED_TEST)  # fixed order: naive -> ... -> xgboost
    vals = [test[m][key_idx] for m in models]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(models, vals, color=[MODEL_COLORS[m] for m in models])
    annotate_bars(ax, bars, fmt)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ylim or max(vals) * 1.15)
    ax.tick_params(axis="x", labelrotation=15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def bucket_figure():
    xgb_tab = parse_table(read("xgb"), "--- TEST MAE by journey-position bucket ---")
    v1_tab = parse_table(read("v1"), "--- TEST MAE by journey-position bucket ---")
    v2_tab = parse_table(read("v2"), "--- TEST MAE by journey-position bucket ---")
    dn_tab = parse_table(read("dense"), "--- TEST MAE by journey-position bucket ---")
    series = {
        "Naive arrival": xgb_tab["A_naive_arrival"],
        "XGBoost": xgb_tab["C_xgboost"],
        "LSTM V1": v1_tab["D_lstm"],
        "LSTM V2": v2_tab["E_lstm_v2"],
        "Dense-only": dn_tab["F_dense_only"],
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, s in series.items():
        ax.plot(BUCKETS, s.reindex(BUCKETS).values, marker="o",
                label=name, color=MODEL_COLORS[name],
                linewidth=2.2 if name in ("XGBoost", "LSTM V2") else 1.4)
    ax.set_title("Test MAE by journey-position bucket\n(all models, frozen results)")
    ax.set_xlabel("position of current station within journey")
    ax.set_ylabel("test MAE (minutes)")
    ax.set_ylim(0, None)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_bucket_mae.png", dpi=150)
    plt.close(fig)
    return series


def ladder_figure(test):
    order = ["Dense-only", "LSTM V1", "LSTM V2", "XGBoost"]
    desc = {"Dense-only": "current state + context only (145 params)",
            "LSTM V1": "full history only (5,441 params)",
            "LSTM V2": "history + current-state skip (5,521 params)",
            "XGBoost": "current state + context + 2 lags"}
    vals = [test[m][0] for m in order]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(order[::-1], vals[::-1],
                   color=[MODEL_COLORS[m] for m in order[::-1]])
    for b, m in zip(bars, order[::-1]):
        ax.annotate(f"{b.get_width():.2f}  ({desc[m]})",
                    (b.get_width() + 0.15, b.get_y() + b.get_height() / 2),
                    va="center", fontsize=9)
    ax.set_title("Ablation ladder - test MAE (minutes), lower is better")
    ax.set_xlabel("test MAE (minutes)")
    ax.set_xlim(0, max(vals) * 1.55)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_ablation_ladder.png", dpi=150)
    plt.close(fig)


def per_train_figure():
    xgb = parse_table(read("xgb"), "--- TEST MAE per train number ---")["C_xgboost"]
    v2 = parse_table(read("v2"), "--- TEST MAE per train number ---")["E_lstm_v2"]
    dn = parse_table(read("dense"), "--- TEST MAE per train number ---")["F_dense_only"]
    trains = sorted(xgb.index)
    y = np.arange(len(trains))
    h = 0.27
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(y + h, xgb.reindex(trains), height=h, label="XGBoost",
            color=MODEL_COLORS["XGBoost"])
    ax.barh(y, v2.reindex(trains), height=h, label="LSTM V2",
            color=MODEL_COLORS["LSTM V2"])
    ax.barh(y - h, dn.reindex(trains), height=h, label="Dense-only",
            color=MODEL_COLORS["Dense-only"])
    ax.set_yticks(y, trains)
    ax.invert_yaxis()
    ax.set_title("Per-train test MAE: XGBoost vs LSTM V2 vs Dense-only")
    ax.set_xlabel("test MAE (minutes)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_per_train_mae.png", dpi=150)
    plt.close(fig)


def failure_figure():
    """Journey 12625_2026-06-05: on time until ~72% completion, then +380 min.
    Observed profile from the modeling dataset; model predictions taken from
    the frozen worst-TEST-error tables (no model is run)."""
    jid = "12625_2026-06-05"
    df = pd.read_csv(PROC / "station_delays_model.csv")
    j = df[df["journey_id"] == jid].sort_values("station_sequence")
    target = float(j["target_destination_delay"].iloc[0])
    n = len(j)
    frac = np.arange(n) / (n - 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frac, j["arrival_delay_model"], marker=".", color="#333333",
            label="observed arrival delay along journey")
    ax.axhline(target, color="#b2182b", linestyle="--",
               label=f"actual final delay = {target:.0f} min")

    seqs = list(j["station_sequence"])
    for name, key in [("LSTM V1", "v1"), ("LSTM V2", "v2"),
                      ("Dense-only", "dense")]:
        w = parse_worst_test(read(key), jid)
        if len(w):
            x = [frac[seqs.index(s)] for s in w["station_sequence"]]
            ax.scatter(x, w["pred"], color=MODEL_COLORS[name], zorder=5,
                       label=f"{name} prediction at that station (frozen report)")
    ax.set_title(f"Why every model fails on {jid}\n"
                 "the 380-minute disruption begins only after ~72% of the journey")
    ax.set_xlabel("journey completion (fraction of stations passed)")
    ax.set_ylabel("delay (minutes)")
    ax.set_ylim(-20, None)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_failure_case.png", dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    test, log = audit()
    print("PHASE 1 AUDIT - all parsed from frozen reports")
    print("\n".join(log))
    print("\naudit passed: every number consistent across reports and with "
          "the frozen results table.\n")

    bar_figure(test, 0, "Final test MAE across all models (frozen results)",
               "test MAE (minutes)", "fig1_test_mae.png")
    bar_figure(test, 1, "Final test RMSE across all models (frozen results)",
               "test RMSE (minutes)", "fig2_test_rmse.png")
    bar_figure(test, 2, "Final test R² across all models (frozen results)",
               "test R²", "fig3_test_r2.png", fmt="{:.4f}", ylim=1.0)
    bucket_series = bucket_figure()
    ladder_figure(test)
    per_train_figure()
    failure_figure()

    audit_text = ["FINAL METRICS AUDIT (parsed from frozen reports only)", ""]
    audit_text += log
    audit_text += ["", "canonical test metrics (MAE, RMSE, R2):"]
    for m, v in test.items():
        audit_text.append(f"  {m:<16} {v}")
    audit_text += ["", "test bucket MAE:"]
    audit_text.append(pd.DataFrame(bucket_series).reindex(BUCKETS).to_string())
    AUDIT_TXT.write_text("\n".join(audit_text), encoding="utf-8")

    print(f"figures written to {FIG_DIR}")
    print(f"audit written to {AUDIT_TXT}")


if __name__ == "__main__":
    main()
