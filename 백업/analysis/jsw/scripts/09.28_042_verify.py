"""Independently recount high slots and runs from the untouched source CSV."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd


BASE = Path(__file__).parent.parent
SOURCE = BASE / "../../data/origin/okm_augumented_2021.csv"
TABLES = BASE / "tables"
SLOTS = ["15분", "30분", "45분", "60분"]


def main() -> None:
    config = json.loads((TABLES / "09.28_042_frozen.json").read_text(encoding="utf-8"))
    assert sha256(SOURCE.read_bytes()).hexdigest() == config["source_sha256"]
    raw = pd.read_csv(SOURCE, encoding="utf-8-sig", usecols=["날짜", "시간", *SLOTS])
    assert len(raw) == 6168
    scoped = raw.loc[raw["날짜"].lt(20210901)]
    assert len(scoped) == 5832 and len(scoped.loc[~scoped["시간"].between(0, 23)]) == 48
    valid = scoped.loc[scoped["시간"].between(0, 23)].copy()
    valid["key"] = pd.to_datetime(valid["날짜"].astype(str), format="%Y%m%d") + pd.to_timedelta(valid["시간"], unit="h")
    valid = valid.sort_values("key")
    assert len(valid) == 5784 and valid["key"].is_unique

    # Streaming recount, separate from the vectorized production implementation.
    counters = {p: {"hours": 0, "slots": 0, "high_hours": 0, "high_slots": 0,
                    "hour_events": 0, "slot_events": 0, "slot_lengths": [], "hour_lengths": []}
                for p in ("Jan-Jun", "Jul-Aug")}
    last_hour = last_slot = None
    prior_hour_high = prior_slot_high = False
    current_hour_length = current_slot_length = 0
    for _, row in valid.iterrows():
        period = "Jan-Jun" if row["key"] < pd.Timestamp("2021-07-01") else "Jul-Aug"
        c = counters[period]
        c["hours"] += 1
        values = [row[s] for s in SLOTS]
        high_hour = max(values) >= 182
        if high_hour:
            c["high_hours"] += 1
            if not prior_hour_high or row["key"] - last_hour != pd.Timedelta(hours=1):
                c["hour_events"] += 1
                if current_hour_length:
                    previous_period = "Jan-Jun" if last_hour < pd.Timestamp("2021-07-01") else "Jul-Aug"
                    counters[previous_period]["hour_lengths"].append(current_hour_length)
                current_hour_length = 0
            current_hour_length += 1
        elif current_hour_length:
            previous_period = "Jan-Jun" if last_hour < pd.Timestamp("2021-07-01") else "Jul-Aug"
            counters[previous_period]["hour_lengths"].append(current_hour_length)
            current_hour_length = 0
        prior_hour_high = high_hour
        last_hour = row["key"]
        for i, value in enumerate(values, 1):
            slot_key = row["key"] + pd.Timedelta(minutes=15 * i)
            c["slots"] += 1
            high_slot = value >= 182
            if high_slot:
                c["high_slots"] += 1
                if not prior_slot_high or slot_key - last_slot != pd.Timedelta(minutes=15):
                    c["slot_events"] += 1
                    if current_slot_length:
                        previous_period = "Jan-Jun" if last_slot < pd.Timestamp("2021-07-01") else "Jul-Aug"
                        counters[previous_period]["slot_lengths"].append(current_slot_length)
                    current_slot_length = 0
                current_slot_length += 1
            elif current_slot_length:
                previous_period = "Jan-Jun" if last_slot < pd.Timestamp("2021-07-01") else "Jul-Aug"
                counters[previous_period]["slot_lengths"].append(current_slot_length)
                current_slot_length = 0
            prior_slot_high = high_slot
            last_slot = slot_key
    if current_hour_length:
        counters["Jul-Aug"]["hour_lengths"].append(current_hour_length)
    if current_slot_length:
        counters["Jul-Aug"]["slot_lengths"].append(current_slot_length)

    # Recount the central finding directly from consecutive high-hour rows.
    boundary_audit = {p: {"multi_hour": 0, "no_cross": 0} for p in counters}
    episode_rows = []
    previous_key = None
    previous_high = False
    def close_episode() -> None:
        if len(episode_rows) < 2:
            return
        period = "Jan-Jun" if episode_rows[0][0] < pd.Timestamp("2021-07-01") else "Jul-Aug"
        boundary_audit[period]["multi_hour"] += 1
        crosses = any(left[2] >= 182 and right[1] >= 182 for left, right in zip(episode_rows, episode_rows[1:]))
        boundary_audit[period]["no_cross"] += int(not crosses)

    for _, row in valid.iterrows():
        key = row["key"]
        high = max(row[s] for s in SLOTS) >= 182
        if not high or not previous_high or key - previous_key != pd.Timedelta(hours=1):
            close_episode()
            episode_rows = []
        if high:
            episode_rows.append((key, row["15분"], row["60분"]))
        previous_key, previous_high = key, high
    close_episode()

    summary = pd.read_csv(TABLES / "09.28_042_summary.csv", encoding="utf-8-sig").set_index("period")
    lengths = pd.read_csv(TABLES / "09.28_042_run_lengths.csv", encoding="utf-8-sig")
    starts = pd.read_csv(TABLES / "09.28_042_start_slots.csv", encoding="utf-8-sig")
    comparison = pd.read_csv(TABLES / "09.28_042_hour_event_comparison.csv", encoding="utf-8-sig")
    for period, c in counters.items():
        for key, column in (("hours", "valid_hours"), ("slots", "valid_slots"), ("high_hours", "high_hours"),
                            ("high_slots", "high_slots"), ("hour_events", "hour_events"), ("slot_events", "slot_events")):
            assert c[key] == int(summary.loc[period, column]), (period, key)
        observed = lengths.loc[lengths["period"].eq(period)].set_index("slots")["slot_events"].to_dict()
        expected = pd.Series(c["slot_lengths"]).value_counts().to_dict()
        assert observed == expected
        assert len(c["hour_lengths"]) == c["hour_events"]
        assert sum(c["hour_lengths"]) == c["high_hours"]
        assert int(starts.loc[starts["period"].eq(period), "slot_events"].sum()) == c["slot_events"]
        part = comparison.loc[comparison["period"].eq(period)]
        assert len(part) == c["hour_events"] and int(part["high_slots"].sum()) == c["high_slots"]
        assert sorted(part["hours"].tolist()) == sorted(c["hour_lengths"])
        assert boundary_audit[period]["multi_hour"] == int(summary.loc[period, "multi_hour_events"])
        assert boundary_audit[period]["no_cross"] == int(summary.loc[period, "multi_hour_with_no_cross_hour_run"])
    print(json.dumps({"status": "passed", "source_rows": len(raw), "valid_hours": len(valid),
                      "reconciled_periods": list(counters), "slot_events": sum(x["slot_events"] for x in counters.values()),
                      "boundary_audit": boundary_audit}))


if __name__ == "__main__":
    main()
