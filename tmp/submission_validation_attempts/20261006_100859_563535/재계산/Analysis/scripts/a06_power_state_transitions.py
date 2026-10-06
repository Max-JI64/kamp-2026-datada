"""Descriptive power transitions for existing manuscripts; no model fitting.

Question: do the low-power periods seen in EDA correspond to production
transitions, and how much of the observed hour-to-hour change occurs there?
Reuse EDA's rounded-mean 20..26 definition; zero is a separate state.
Positive values below 20 are separated from above-26 values after the first
pass found weekend 19 values. Crossing either boundary is not a peak label.
"""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/origin/okm_augumented_2021.csv"
OUT = ROOT / "Analysis/tables/a06_power_state_transitions"
SLOTS = ["15분", "30분", "45분", "60분"]


def load():
    raw = pd.read_csv(SOURCE, encoding="utf-8-sig")
    dates = pd.to_datetime(raw["날짜"].astype(str), format="%Y%m%d")
    keep = dates.dt.month.between(1, 8) & raw["시간"].between(0, 23)
    d = raw.loc[keep].copy()
    d["timestamp"] = dates[keep] + pd.to_timedelta(d["시간"], unit="h")
    d = d.sort_values("timestamp").set_index("timestamp")
    assert d.index.is_unique and len(d) == 5784
    assert d[SLOTS].ge(0).all().all()
    d["mean"] = d[SLOTS].mean(axis=1)
    d["maximum"] = d[SLOTS].max(axis=1)
    d["state"] = np.select([d.maximum.eq(0), d["평균"].lt(20), d["평균"].between(20, 26)],
                           ["zero", "below20", "low"], default="above26")
    d["month"] = d.index.month
    d["hour"] = d.index.hour
    d["daytype"] = np.where(d.index.dayofweek < 5, "weekday", "weekend")
    for col in ["mean", "maximum", "state", "생산량", "60분"]:
        d["prior_" + col] = d[col].reindex(d.index - pd.Timedelta(hours=1)).to_numpy()
    d["delta_mean"] = d["mean"] - d.prior_mean
    d["delta_maximum"] = d.maximum - d.prior_maximum
    d["prior_last_minus_mean"] = d["prior_60분"] - d.prior_mean
    return d


def independent_verify(d, transitions):
    # Separate stdlib reconstruction checks all rows and exact-hour pairs.
    records = {}
    with SOURCE.open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            day = datetime.strptime(r["날짜"], "%Y%m%d")
            hour = float(r["시간"])
            if day.month > 8 or not 0 <= hour <= 23:
                continue
            ts = day + timedelta(hours=hour)
            values = [float(r[s]) for s in SLOTS]
            state = "zero" if max(values) == 0 else "below20" if float(r["평균"]) < 20 else "low" if float(r["평균"]) <= 26 else "above26"
            records[ts] = (sum(values) / 4, max(values), state)
    counts = Counter()
    for ts, row in d.iterrows():
        m, maximum, state = records[ts.to_pydatetime()]
        assert (row["mean"], row.maximum, row.state) == (m, maximum, state)
        prior = records.get(ts.to_pydatetime() - timedelta(hours=1))
        if prior is not None:
            counts[(prior[2], state)] += 1
            assert row.delta_mean == m - prior[0]
            assert row.delta_maximum == maximum - prior[1]
    assert counts == Counter(transitions.groupby(["prior_state", "state"]).size().to_dict())
    assert sum(counts.values()) == 5781
    # The existing figures use M01's full valid timeline, not its model subsets.
    frame = pd.read_csv(ROOT / "Modeling/tables/m01/hourly_frame.csv", encoding="utf-8-sig")
    assert pd.to_datetime(frame.timestamp).tolist() == list(d.index)
    assert np.array_equal(frame.target_mean, d["mean"])
    assert np.array_equal(frame.target_maximum, d.maximum)
    return {"raw_rows_checked": len(records), "exact_hour_pairs_checked": sum(counts.values()),
            "existing_plot_frame_matches_raw": True}


def episodes(d):
    contiguous = d.index.to_series().diff().eq(pd.Timedelta(hours=1))
    group = (~(contiguous & d.state.eq(d.state.shift()))).cumsum()
    rows = []
    for _, part in d.groupby(group):
        if part.state.iloc[0] not in ("low", "zero"):
            continue
        start, end = part.index[0], part.index[-1]
        rows.append({"state": part.state.iloc[0], "start": start, "end": end,
                     "hours": len(part), "left_complete": start - pd.Timedelta(hours=1) in d.index,
                     "right_complete": end + pd.Timedelta(hours=1) in d.index,
                     "positive_production_hours": int(part["생산량"].gt(0).sum())})
    return pd.DataFrame(rows)


def summarize(d):
    t = d.loc[d.prior_state.notna()].copy()
    rows = []
    abs_total = t.delta_maximum.abs().sum()
    for (prior, state), g in t.groupby(["prior_state", "state"]):
        rows.append({"prior_state": prior, "state": state, "hours": len(g),
                     "dates": g.index.normalize().nunique(),
                     "median_delta_mean": g.delta_mean.median(),
                     "median_delta_maximum": g.delta_maximum.median(),
                     "negative_delta_maximum_hours": int(g.delta_maximum.lt(0).sum()),
                     "positive_delta_maximum_hours": int(g.delta_maximum.gt(0).sum()),
                     "mean_abs_delta_maximum": g.delta_maximum.abs().mean(),
                     "share_total_abs_delta_maximum": g.delta_maximum.abs().sum() / abs_total,
                     "minimum_delta_maximum": g.delta_maximum.min(),
                     "maximum_delta_maximum": g.delta_maximum.max(),
                     "production_zero_to_positive": int((g["prior_생산량"].eq(0) & g["생산량"].gt(0)).sum()),
                     "production_positive_to_zero": int((g["prior_생산량"].gt(0) & g["생산량"].eq(0)).sum()),
                     "prior_last_minus_mean_median": g.prior_last_minus_mean.median()})
    return t, pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    t, contrasts = summarize(d)
    verified = independent_verify(d, t)
    e = episodes(d)
    assert int(e.hours.sum()) == int(d.state.isin(["low", "zero"]).sum())
    assert len(e.loc[e.state.eq("zero")]) == 1
    tables = {
        "transition_summary": contrasts,
        "state_counts": d.groupby(["daytype", "state"]).size().rename("hours").reset_index(),
        "monthly_state_counts": d.groupby(["month", "state"]).size().rename("hours").reset_index(),
        "transition_hours": t.groupby(["prior_state", "state", "daytype", "hour"]).size().rename("hours").reset_index(),
        "episodes": e,
        "boundary_records": t.loc[t.prior_state.ne(t.state)].reset_index(),
        "largest_changes": pd.concat([t.nsmallest(5, "delta_maximum").assign(direction="drop"),
                                       t.nlargest(5, "delta_maximum").assign(direction="rise")]).reset_index(),
    }
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    episode_summary = {}
    for state, g in e.groupby("state"):
        complete = g.loc[g.left_complete & g.right_complete]
        episode_summary[state] = {"observed_runs": len(g), "complete_runs": len(complete),
                                  "observed_hours": int(g.hours.sum()),
                                  "complete_duration_min": int(complete.hours.min()),
                                  "complete_duration_median": float(complete.hours.median()),
                                  "complete_duration_max": int(complete.hours.max())}
    summary = {"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               "rows": len(d), "pairs": len(t), "state_counts": d.state.value_counts().to_dict(),
               "episodes": episode_summary, "verification": verified,
               "largest_changes_both_production_zero": bool((tables["largest_changes"]["생산량"].eq(0) & tables["largest_changes"]["prior_생산량"].eq(0)).all()),
               "definition": "zero: maximum=0; below20: positive and rounded mean<20; low: rounded mean20..26; above26: rounded mean>26",
               "revision": "Initial other group mixed weekend values below20 with above26. Separated at existing lower bound20 to avoid interpreting every low-range exit as a rise; no records excluded.",
               "scope": "all valid Jan-Aug hours, no model split, no new peak threshold, no model fit",
               "inference": "descriptive counts only; temporal and repeated-profile dependence retained",
               "outputs_sha256": {f"{name}.csv": hashlib.sha256((OUT / f"{name}.csv").read_bytes()).hexdigest() for name in tables}}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["rows", "pairs", "state_counts", "episodes", "verification"]}, ensure_ascii=False))
    print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
