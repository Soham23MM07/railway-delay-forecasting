"""Collect station-level running data from the RailRadar API.

Requires a personal API key in the RAILRADAR_API_KEY environment variable:
    export RAILRADAR_API_KEY="your-key"        (bash)
    $env:RAILRADAR_API_KEY = "your-key"        (PowerShell)

Output CSVs are written to data/raw/ relative to the project root.
"""

import requests
import pandas as pd
import os
import time
from datetime import datetime
from pathlib import Path

# ==========================
# CONFIG
# ==========================

API_KEY = os.environ.get("RAILRADAR_API_KEY")
if not API_KEY:
    raise SystemExit("Set the RAILRADAR_API_KEY environment variable first.")
BASE_URL = "https://api.railradar.in/v1"
HEADERS  = {"Authorization": f"Bearer {API_KEY}"}

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

    # "2026-05-01",
    # "2026-05-02",
    # "2026-05-03",
    # "2026-05-04",
    # "2026-05-05",
    # "2026-05-06",
    # "2026-05-07",
    # "2026-05-08",
    # "2026-05-09",
    # "2026-05-10"
    "2026-05-11",
    "2026-05-12",
    "2026-05-13",
    "2026-05-14"

    # "2026-05-15",
    # "2026-05-16"
    # "2026-05-17",
    # "2026-05-18",
    # "2026-05-19",
    # "2026-05-21",
    # "2026-05-22",
    # "2026-05-23"
    # "2026-05-24"
    # "2026-05-25",
    # "2026-05-26"
    # "2026-05-27",
    # "2026-05-28",
    # "2026-05-29",
    # "2026-05-30",
    # "2026-05-31"

    # "2026-06-01",
    # "2026-06-02",
    # "2026-06-03",
    # "2026-06-04",
    # "2026-06-05",
    # "2026-06-06",
    # "2026-06-07",
    # "2026-06-08",
    # "2026-06-09",
    # "2026-06-10",
    # "2026-06-11",
    # "2026-06-12"
    # "2026-06-14",
    # "2026-06-15",
    # "2026-06-16"
    # "2026-06-17",
    # "2026-06-18",
    # "2026-06-19"
    # "2026-06-20"

]

OUTPUT_FILE = str(PROJECT_ROOT / "data" / "raw" / "station_delays_may_start.csv")

# ==========================
# HELPERS
# ==========================

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


def load_already_collected():
    if not os.path.exists(OUTPUT_FILE):
        return set()
    df = pd.read_csv(OUTPUT_FILE)
    return set(zip(
        df["train_number"].astype(str),
        df["journey_date"]
    ))


def save_records(records):
    if not records:
        return
    df_new = pd.DataFrame(records)
    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
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
        df_new.to_csv(OUTPUT_FILE, index=False)


# ==========================
# FETCH
# ==========================

def fetch_train_on_date(train_number, date):
    url = (
        f"{BASE_URL}/trains/{train_number}"
        f"/live?date={date}&haltsOnly=true"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)

        if r.status_code == 429:
            print("⚠️  Rate limit hit — waiting 60 seconds")
            time.sleep(60)
            return []

        if r.status_code != 200:
            print(f"  ❌ {train_number} {date} → {r.status_code}")
            return []

        payload = r.json()
        if not payload.get("success"):
            return []



        data       = payload.get("data", {})
        train_info = data.get("train", {})
        status     = data.get("status", "")

        # Skip if journey not completed
        if status not in ["completed", "running"]:
            return []

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

        return records

    except Exception as e:
        print(f"  ❌ Error {train_number} {date}: {e}")
        return []


# ==========================
# MAIN
# ==========================

def main():
    os.makedirs(PROJECT_ROOT / "data" / "raw", exist_ok=True)

    already_collected = load_already_collected()
    print(f"Already collected: {len(already_collected)} train-date pairs")

    total_requests = 0
    total_records  = 0
    batch          = []

    print("\n" + "=" * 60)
    print(f"COLLECTING {len(TRAIN_NUMBERS)} trains × {len(DATES)} dates")
    print(f"Total API requests: {len(TRAIN_NUMBERS) * len(DATES)}")
    print("=" * 60)

    for date in DATES:
        print(f"\n📅 {date}")
        print("-" * 40)

        for train in TRAIN_NUMBERS:

            # Skip already collected
            if (train, date) in already_collected:
                print(f"  ⏭️  {train} already collected")
                continue

            records = fetch_train_on_date(train, date)
            total_requests += 1
            total_records  += len(records)
            batch.extend(records)

            delayed = sum(
                1 for r in records
                if r["arrival_delay_minutes"] not in [None, 0]
            )
            print(f"  ✅ {train} | stations={len(records)} | delayed={delayed}")

            # Save every 50 records
            if len(batch) >= 50:
                save_records(batch)
                batch = []

            time.sleep(0.5)  # respectful delay

    # Save remaining
    if batch:
        save_records(batch)

    # Final summary
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"API requests used : {total_requests}")
    print(f"Records collected : {total_records}")

    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        print(f"Total in CSV      : {len(df)}")
        print(f"Trains            : {df['train_number'].nunique()}")
        print(f"Dates             : {df['journey_date'].nunique()}")
        print(f"Unique stations   : {df['station_code'].nunique()}")
        print(f"\nDelay stats:")
        print(df["arrival_delay_minutes"].describe())


if __name__ == "__main__":
    main()