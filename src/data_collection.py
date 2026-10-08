"""Collect station-level running data from the RailRadar API.

Requires one or more personal API keys, set via either:
    RAILRADAR_API_KEYS="key1,key2,key3"   (comma-separated, rotates between them)
    RAILRADAR_API_KEY="key1"              (single key, backward compatible)
in a .env file (loaded automatically) or the environment directly.

Each key gets its own MAX_REQUESTS_PER_RUN budget for this run; once a key's
budget is used up, the script rotates to the next key automatically.

Output CSVs are written to data/raw/ relative to the project root.
"""

import requests
import pandas as pd
import os
import sys
import time
import json
import hashlib
from datetime import datetime, date
from pathlib import Path
from dotenv import load_dotenv

# Windows' default console encoding (cp1252) can't print the emoji used in
# this script's progress logs - force UTF-8 stdout so it doesn't crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ==========================
# CONFIG
# ==========================

load_dotenv()

_keys_csv = os.getenv("RAILRADAR_API_KEYS")
if _keys_csv:
    API_KEYS = [k.strip() for k in _keys_csv.split(",") if k.strip()]
else:
    _single = os.getenv("RAILRADAR_API_KEY")
    API_KEYS = [_single] if _single else []

if not API_KEYS:
    raise SystemExit(
        "Set RAILRADAR_API_KEYS (comma-separated, for multiple keys) or "
        "RAILRADAR_API_KEY (single key) in your .env file or environment."
    )

BASE_URL = "https://api.railradar.in/v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_NUMBERS = [
    "12951", "12952",
    "12301", "12302",
    "12621", "12622",
    "12627", "12628",
    "12009", "12010",
    "12431", "12432",
    "12723", "12724",
    "12657", "12658",
    "12625", "12626",
    "12001", "12002"
]

DATES = [
    # --- previously collected (2026-05-01 .. 2026-06-20), kept for reference ---
    # "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05",
    # "2026-05-06", "2026-05-07", "2026-05-08", "2026-05-09", "2026-05-10",
    # "2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15",
    # "2026-05-16", "2026-05-17", "2026-05-18", "2026-05-19", "2026-05-21",
    # "2026-05-22", "2026-05-23", "2026-05-24", "2026-05-25", "2026-05-26",
    # "2026-05-27", "2026-05-28", "2026-05-29", "2026-05-30", "2026-05-31",
    # "2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05",
    # "2026-06-06", "2026-06-07", "2026-06-08", "2026-06-09", "2026-06-10",
    # "2026-06-11", "2026-06-12", "2026-06-14", "2026-06-15", "2026-06-16",
    # "2026-06-17", "2026-06-18", "2026-06-19", "2026-06-20",

    # --- collected (2026-06-21 .. 2026-07-16), kept for reference ---
    # "2026-06-21", "2026-06-22", "2026-06-23", "2026-06-24", "2026-06-25",
    # "2026-06-26", "2026-06-27", "2026-06-28", "2026-06-29", "2026-06-30",
    # "2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04", "2026-07-05",
    # "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09", "2026-07-10",
    # "2026-07-11", "2026-07-12", "2026-07-13", "2026-07-14", "2026-07-15",
    # "2026-07-16",

    # --- NEW window: never collected before (2026-07-17 .. 2026-07-20) ---
    "2026-07-17", "2026-07-18", "2026-07-19", "2026-07-20",
]

OUTPUT_FILE = str(PROJECT_ROOT / "data" / "raw" / "station_delays_may_start.csv")

# RailRadar free-tier limits: 50 requests/day, burst-limited to 10/minute.
# MAX_REQUESTS_PER_RUN keeps one script run safely under the daily cap even
# if TRAIN_NUMBERS x DATES covers far more (train, date) pairs than that -
# the run stops early and whatever's left over resumes correctly next time,
# since load_already_collected()/save_records() already skip/dedupe by
# (train_number, journey_date).
MAX_REQUESTS_PER_KEY = 45   # a few under 50/day, per key, as a safety margin
SECONDS_BETWEEN_REQUESTS = 6.5   # > 6s, so we never exceed 10 requests/min
MAX_REQUESTS_PER_RUN = MAX_REQUESTS_PER_KEY * len(API_KEYS)

# ==========================
# HELPERS
# ==========================

LOG_FILE = PROJECT_ROOT / "data" / "raw" / "collection_log.txt"
KEY_USAGE_FILE = PROJECT_ROOT / "data" / "raw" / "key_usage.json"


def _key_id(key: str) -> str:
    """A short, stable, non-reversible identifier for a key - so the usage
    file never stores the actual secret, only enough to recognize 'this is
    the same key as before' across separate script runs."""
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def load_key_usage() -> dict:
    """Requests used TODAY, per key, persisted ACROSS script runs (not just
    within one run) - this is what prevents the bug where a key that was
    already exhausted in an earlier run today gets treated as having a
    fresh budget just because the script restarted."""
    today = str(date.today())
    if KEY_USAGE_FILE.exists():
        data = json.loads(KEY_USAGE_FILE.read_text(encoding="utf-8"))
        if data.get("date") == today:
            return data
    return {"date": today, "usage": {}}


def save_key_usage(usage_data: dict) -> None:
    KEY_USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    KEY_USAGE_FILE.write_text(json.dumps(usage_data, indent=2), encoding="utf-8")


def log(msg: str) -> None:
    """Print with a timestamp AND append to a persistent log file, so a run
    can be monitored live and reviewed afterward even if the terminal
    scrollback is gone."""
    stamped = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(stamped)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(stamped + "\n")


def calculate_delay(scheduled, actual):
    if not scheduled or not actual:
        return None
    try:
        sch   = datetime.fromisoformat(scheduled)
        act   = datetime.fromisoformat(actual)
        delay = round((act - sch).total_seconds() / 60)
        if delay < -30 or delay > 600:
            return None
        return delay
    except Exception:
        return None


def normalize_dates(date_series: pd.Series) -> pd.Series:
    """Return dates as YYYY-MM-DD strings, regardless of whether the input
    is already ISO (YYYY-MM-DD) or has been reformatted to DD-MM-YYYY (e.g.
    by Excel silently reformatting the column if the CSV was opened/saved
    there mid-run - format='mixed' handles both transparently)."""
    parsed = pd.to_datetime(date_series, format="mixed", dayfirst=True)
    return parsed.dt.strftime("%Y-%m-%d")


def load_already_collected():
    if not os.path.exists(OUTPUT_FILE):
        return set()
    df = pd.read_csv(OUTPUT_FILE)
    return set(zip(
        df["train_number"].astype(str),
        normalize_dates(df["journey_date"])
    ))


def save_records(records, max_retries=5, retry_delay=10):
    """Write records to OUTPUT_FILE, retrying on PermissionError (e.g. the
    CSV briefly open in Excel/another program) instead of crashing the
    whole collection run over a transient file lock."""
    if not records:
        return
    df_new = pd.DataFrame(records)

    for attempt in range(1, max_retries + 1):
        try:
            if os.path.exists(OUTPUT_FILE):
                df_existing = pd.read_csv(OUTPUT_FILE)
                # Re-normalize on every save: self-heals if Excel (or anything
                # else) silently reformatted journey_date since the last write.
                df_existing["journey_date"] = normalize_dates(df_existing["journey_date"])
                df_new["journey_date"] = normalize_dates(df_new["journey_date"])
                df_combined = pd.concat(
                    [df_existing, df_new],
                    ignore_index=True
                )
                df_combined.drop_duplicates(
                    subset=["train_number", "journey_date", "station_code"],
                    inplace=True
                )
                df_combined.to_csv(OUTPUT_FILE, index=False)
            else:
                df_new["journey_date"] = normalize_dates(df_new["journey_date"])
                df_new.to_csv(OUTPUT_FILE, index=False)
            return
        except PermissionError:
            log(f"  WARNING: {Path(OUTPUT_FILE).name} is locked (probably "
                f"open in Excel/another program) - retry {attempt}/"
                f"{max_retries} in {retry_delay}s. Close the file to let "
                f"this succeed.")
            time.sleep(retry_delay)

    raise PermissionError(
        f"Could not write to {OUTPUT_FILE} after {max_retries} retries - "
        f"make sure it's closed in Excel/other programs, then rerun."
    )


# ==========================
# FETCH
# ==========================

def fetch_train_on_date(train_number, date, headers):
    """Returns (records, hit_429). hit_429=True means the CALLING key is
    exhausted for today - the caller should immediately mark that key as
    used-up and rotate to the next one, instead of retrying the same dead
    key after a pointless wait."""
    url = (
        f"{BASE_URL}/trains/{train_number}"
        f"/live?date={date}&haltsOnly=true"
    )
    try:
        r = requests.get(url, headers=headers, timeout=15)

        if r.status_code == 429:
            return [], True

        if r.status_code != 200:
            print(f"  ❌ {train_number} {date} → {r.status_code}")
            return [], False

        payload = r.json()
        if not payload.get("success"):
            return [], False



        data       = payload.get("data", {})
        train_info = data.get("train", {})
        status     = data.get("status", "")

        # Skip if journey not completed
        if status not in ["completed", "running"]:
            return [], False

        records = []
        for s in data.get("route", []):
            if not s.get("isHalt"):
                continue

            arr_delay = s.get("delayArrival")

            if arr_delay is None:
                arr_delay = calculate_delay(
                    s.get("scheduledArrival"),
                    s.get("actualArrival")
                )

            dep_delay = s.get("delayDeparture")

            if dep_delay is None:
                dep_delay = calculate_delay(
                    s.get("scheduledDeparture"),
                    s.get("actualDeparture")
                )

            records.append({
                "journey_date":            date,
                "train_number":            train_number,
                "train_name":              data.get("trainName"),
                "train_type":              train_info.get("type"),
                "category":                train_info.get("category"),
                "source_station":          train_info.get("source", {}).get("code"),
                "destination_station":     train_info.get("destination", {}).get("code"),
                "station_sequence":        s.get("sequence"),
                "station_code":            s.get("stationCode"),
                "station_name":            s.get("stationName"),
                "distance_km":             s.get("distance"),
                "platform":                s.get("platform"),
                "status":                  s.get("status"),
                "scheduled_arrival":       s.get("scheduledArrival"),
                "actual_arrival":          s.get("actualArrival"),
                "scheduled_departure":     s.get("scheduledDeparture"),
                "actual_departure":        s.get("actualDeparture"),
                "arrival_delay_minutes":   arr_delay,
                "departure_delay_minutes": dep_delay,
            })

        return records, False

    except Exception as e:
        print(f"  ❌ Error {train_number} {date}: {e}")
        return [], False


# ==========================
# MAIN
# ==========================

def pick_available_key(key_usage: dict):
    """Return (key_index, key_id) for the first key that still has budget
    left TODAY, based on PERSISTED usage (survives across separate script
    runs) - or None if every key is exhausted for today."""
    for idx, key in enumerate(API_KEYS):
        kid = _key_id(key)
        used = key_usage["usage"].get(kid, 0)
        if used < MAX_REQUESTS_PER_KEY:
            return idx, kid
    return None


def main():
    os.makedirs(PROJECT_ROOT / "data" / "raw", exist_ok=True)
    run_start = time.time()

    already_collected = load_already_collected()
    log(f"Already collected: {len(already_collected)} train-date pairs")

    key_usage = load_key_usage()
    per_key_used = [key_usage["usage"].get(_key_id(k), 0) for k in API_KEYS]
    total_used_today = sum(per_key_used)
    total_budget_today = MAX_REQUESTS_PER_KEY * len(API_KEYS)

    total_requests = 0
    total_records  = 0
    batch          = []

    remaining_pairs = [
        (t, d) for d in DATES for t in TRAIN_NUMBERS
        if (t, d) not in already_collected
    ]
    remaining_budget_today = total_budget_today - total_used_today
    n_this_run = min(len(remaining_pairs), remaining_budget_today)
    eta_seconds = max(n_this_run, 0) * SECONDS_BETWEEN_REQUESTS

    log("=" * 60)
    log(f"COLLECTING {len(TRAIN_NUMBERS)} trains x {len(DATES)} dates")
    log(f"Total (train, date) pairs configured : {len(TRAIN_NUMBERS) * len(DATES)}")
    log(f"Pairs still needed (not yet collected): {len(remaining_pairs)}")
    log(f"API keys available                    : {len(API_KEYS)}")
    log(f"Per-key usage TODAY so far (persisted): "
        + ", ".join(f"key#{i+1}={u}/{MAX_REQUESTS_PER_KEY}" for i, u in enumerate(per_key_used)))
    log(f"Remaining budget today across all keys : {remaining_budget_today} requests")
    log(f"Estimated time for this run            : ~{eta_seconds/60:.1f} minutes")
    log(f"Live log also being written to         : {LOG_FILE}")
    log("=" * 60)

    if remaining_budget_today <= 0:
        log("STOPPING: every key has already used its daily budget today "
            "(persisted from earlier runs) - wait for the daily reset, then rerun.")
        return

    stopped_early = False
    current_key_idx = None
    for date in DATES:
        if stopped_early:
            break
        log(f"--- {date} ---")

        for train in TRAIN_NUMBERS:

            # Skip already collected
            if (train, date) in already_collected:
                log(f"  skip   {train} {date} (already collected)")
                continue

            # Retry with a fresh key (immediately, no wasted sleep) if the
            # current key turns out to already be exhausted (a 429) -
            # tries every key at most once for this single (train, date).
            records, delayed, fetched_ok = [], 0, False
            for _attempt in range(len(API_KEYS)):
                picked = pick_available_key(key_usage)
                if picked is None:
                    log(f"STOPPING: all {len(API_KEYS)} key(s) have used their "
                        f"persisted daily budget - rerun after the daily reset.")
                    stopped_early = True
                    break
                key_idx, kid = picked
                if key_idx != current_key_idx:
                    current_key_idx = key_idx
                    log(f"  key    switching to API key #{key_idx + 1}/{len(API_KEYS)} "
                        f"(already used {key_usage['usage'].get(kid, 0)}/{MAX_REQUESTS_PER_KEY} today)")
                headers = {"Authorization": f"Bearer {API_KEYS[current_key_idx]}"}

                records, hit_429 = fetch_train_on_date(train, date, headers)
                if hit_429:
                    # This key is actually exhausted right now - mark it as
                    # fully used immediately and try the next key, with NO
                    # sleep (unlike the old behavior).
                    key_usage["usage"][kid] = MAX_REQUESTS_PER_KEY
                    save_key_usage(key_usage)
                    log(f"  429    key#{current_key_idx + 1} is exhausted "
                        f"(discovered live) - marking used and retrying "
                        f"with the next key")
                    continue

                total_requests += 1
                total_records  += len(records)
                batch.extend(records)
                key_usage["usage"][kid] = key_usage["usage"].get(kid, 0) + 1
                save_key_usage(key_usage)
                delayed = sum(
                    1 for r in records
                    if r["arrival_delay_minutes"] not in [None, 0]
                )
                elapsed = time.time() - run_start
                log(f"  fetch  [{total_requests}] "
                    f"key#{current_key_idx + 1} ({key_usage['usage'][kid]}/{MAX_REQUESTS_PER_KEY}) "
                    f"{train} {date} "
                    f"| stations={len(records):>2} delayed={delayed:>2} "
                    f"| elapsed={elapsed/60:.1f}min")
                fetched_ok = True
                break

            if stopped_early:
                break
            if not fetched_ok:
                # every key was exhausted mid-attempt for this specific pair
                continue

            # Progress checkpoint every 10 requests
            if total_requests % 10 == 0:
                used_now = sum(key_usage["usage"].values())
                remaining_now = total_budget_today - used_now
                log(f"  >>> progress: {total_requests} fetched this run "
                    f"| {used_now}/{total_budget_today} used today across all keys "
                    f"| records so far={total_records}")

            # Save every 50 records
            if len(batch) >= 50:
                save_records(batch)
                log(f"  saved  {len(batch)} records to {Path(OUTPUT_FILE).name}")
                batch = []

            time.sleep(SECONDS_BETWEEN_REQUESTS)  # stay under 10 requests/min per key

    # Save remaining
    if batch:
        save_records(batch)
        log(f"  saved  {len(batch)} final records to {Path(OUTPUT_FILE).name}")

    # Final summary
    total_elapsed = time.time() - run_start
    log("=" * 60)
    log("STOPPED EARLY (rate-limit safeguard)" if stopped_early else "DONE")
    log(f"Total time elapsed : {total_elapsed/60:.1f} minutes")
    log(f"API requests used  : {total_requests}")
    log(f"Records collected  : {total_records}")

    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        log(f"Total in CSV       : {len(df)}")
        log(f"Trains             : {df['train_number'].nunique()}")
        log(f"Dates              : {df['journey_date'].nunique()}")
        log(f"Unique stations    : {df['station_code'].nunique()}")
        log("Delay stats: " + df["arrival_delay_minutes"].describe().to_string().replace("\n", " | "))
    if stopped_early:
        log(f"NEXT STEP: rerun this script again (later, respecting the daily "
            f"cap) to continue collecting the remaining "
            f"{len(remaining_pairs) - total_requests} pairs.")


if __name__ == "__main__":
    main()