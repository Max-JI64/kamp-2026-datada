"""Compare 182+ quarter-slot runs with the established high-hour episodes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_common import SLOTS, TABLES, events, finish, load, save


PREFIX = "09.28_042"
FROZEN = TABLES / f"{PREFIX}_frozen.json"


def main() -> None:
    config = json.loads(FROZEN.read_text(encoding="utf-8"))
    threshold = config["threshold"]
    assert threshold == 182 and config["slot_order"] == SLOTS
    _, data = load()
    high_hours, _, _, hour_events = events(data, threshold)
    old = pd.read_csv(TABLES / "09.27_002_period_summary.csv", encoding="utf-8-sig").set_index("period")
    for period, part in data.groupby("period"):
        assert len(part) == int(old.loc[period, "valid_hours"])
        assert int(high_hours.loc[part.index].sum()) == int(old.loc[period, "high_hours"])
        assert len(hour_events.loc[hour_events["period"].eq(period)]) == int(old.loc[period, "onset_hours"])

    # One key per listed quarter-slot. Absolute clock meaning remains unknown,
    # but these keys preserve within-hour order and exact next-hour adjacency.
    slots = data[SLOTS].stack().rename("value").reset_index()
    slots.columns = ["hour_key", "slot", "value"]
    slots["slot_number"] = slots["slot"].map({name: i + 1 for i, name in enumerate(SLOTS)})
    assert slots["slot_number"].notna().all()
    slots["slot_key"] = slots["hour_key"] + pd.to_timedelta(slots["slot_number"] * 15, unit="min")
    slots = slots.sort_values("slot_key").reset_index(drop=True)
    assert len(slots) == 4 * len(data) and slots["slot_key"].is_unique
    slots["period"] = np.where(slots["hour_key"].lt(pd.Timestamp("2021-07-01")), "Jan-Jun", "Jul-Aug")
    slots["high"] = slots["value"].ge(threshold)
    adjacent = slots["slot_key"].diff().eq(pd.Timedelta(minutes=15))
    previous_high = slots["high"].shift(fill_value=False)
    slots["run_start"] = slots["high"] & ~(adjacent & previous_high)
    slots["slot_event_id"] = slots["run_start"].cumsum().where(slots["high"])

    # Match each high slot to the A002 high-hour event containing its hour.
    starts = high_hours & ~data["prev_peak"].ge(threshold)
    hour_ids = starts.cumsum().where(high_hours)
    slots["hour_event_id"] = slots["hour_key"].map(hour_ids).where(slots["high"])
    assert slots.loc[slots["high"], "hour_event_id"].notna().all()
    assert slots.loc[~slots["high"], "hour_event_id"].isna().all()

    runs = slots.loc[slots["high"]].groupby("slot_event_id", sort=True).agg(
        period=("period", "first"), start=("slot_key", "first"), end=("slot_key", "last"),
        start_slot=("slot", "first"), slots=("high", "size"),
        hour_event_id=("hour_event_id", "first"), hours_touched=("hour_key", "nunique")
    ).reset_index()
    assert runs["hour_event_id"].notna().all()
    assert (runs["end"] - runs["start"]).eq(pd.to_timedelta((runs["slots"] - 1) * 15, unit="min")).all()
    assert int(runs["slots"].sum()) == int(slots["high"].sum())
    assert len(runs) == int(slots["run_start"].sum())
    assert runs["period"].eq(np.where(runs["start"].lt(pd.Timestamp("2021-07-01")), "Jan-Jun", "Jul-Aug")).all()

    slot_counts = slots.loc[slots["high"]].groupby("hour_event_id").agg(high_slots=("high", "size"))
    run_counts = runs.groupby("hour_event_id").agg(slot_runs=("slots", "size"), longest_run=("slots", "max"),
                                                     runs_crossing_hours=("hours_touched", lambda x: int(x.gt(1).sum())))
    comparison = hour_events[["event_id", "period", "start", "end", "hours"]].copy()
    comparison = comparison.join(slot_counts, on="event_id").join(run_counts, on="event_id")
    assert comparison[["high_slots", "slot_runs", "longest_run"]].notna().all().all()
    comparison[["high_slots", "slot_runs", "longest_run", "runs_crossing_hours"]] = comparison[["high_slots", "slot_runs", "longest_run", "runs_crossing_hours"]].astype(int)
    comparison["multi_hour"] = comparison["hours"].ge(2)
    comparison["no_cross_hour_run"] = comparison["runs_crossing_hours"].eq(0)
    comparison["longest_run_le4"] = comparison["longest_run"].le(4)
    assert int(comparison["high_slots"].sum()) == int(slots["high"].sum())
    assert int(comparison["slot_runs"].sum()) == len(runs)
    assert len(comparison) == len(hour_events)

    summary = []
    for period, part in data.groupby("period"):
        period_slots = slots.loc[slots["period"].eq(period)]
        period_runs = runs.loc[runs["period"].eq(period)]
        period_events = comparison.loc[comparison["period"].eq(period)]
        multi = period_events.loc[period_events["multi_hour"]]
        row = {"period": period, "valid_hours": len(part), "valid_slots": len(period_slots),
               "high_hours": int(high_hours.loc[part.index].sum()), "high_slots": int(period_slots["high"].sum()),
               "hour_events": len(period_events), "slot_events": len(period_runs),
               "multi_hour_events": len(multi), "multi_hour_with_no_cross_hour_run": int(multi["no_cross_hour_run"].sum()),
               "multi_hour_longest_run_le4": int(multi["longest_run_le4"].sum()),
               "slot_events_one_slot": int(period_runs["slots"].eq(1).sum()),
               "slot_events_ge4": int(period_runs["slots"].ge(4).sum()),
               "slot_events_crossing_hours": int(period_runs["hours_touched"].gt(1).sum()),
               "max_slot_event_slots": int(period_runs["slots"].max())}
        assert row["high_hours"] == int(old.loc[period, "high_hours"])
        assert row["hour_events"] == int(old.loc[period, "onset_hours"])
        assert row["high_slots"] == int(period_runs["slots"].sum())
        summary.append(row)

    length_distribution = runs.groupby(["period", "slots"], sort=True).size().rename("slot_events").reset_index()
    start_distribution = runs.groupby(["period", "start_slot"], sort=True).size().rename("slot_events").reset_index()
    event_shape = comparison.groupby(["period", "multi_hour", "hours", "slot_runs", "longest_run", "no_cross_hour_run"], sort=True).size().rename("hour_events").reset_index()
    save(pd.DataFrame(summary), PREFIX, "summary")
    save(length_distribution, PREFIX, "run_lengths")
    save(start_distribution, PREFIX, "start_slots")
    save(event_shape, PREFIX, "hour_event_shapes")
    save(comparison, PREFIX, "hour_event_comparison")
    finish(PREFIX, {"threshold": threshold, "summary": summary, "run_length_total": int(length_distribution["slot_events"].sum()),
                    "start_slot_total": int(start_distribution["slot_events"].sum()), "hour_event_total": len(comparison)})


if __name__ == "__main__":
    main()
