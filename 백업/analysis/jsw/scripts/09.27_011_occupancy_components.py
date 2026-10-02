"""Separate observed high occupancy into event frequency and duration."""
import json
import numpy as np
import pandas as pd
from analysis_common import TABLES, load, events, save, finish

PREFIX = "09.27_011"
cfg = json.loads((TABLES / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
_, data = load(include_september=True)
high, eligible, onset, ev = events(data, cfg["threshold"])
assert not ev[["left_censored", "right_censored", "crosses_period"]].any().any()
old = pd.read_csv(TABLES / "09.27_009_extended_periods.csv", encoding="utf-8-sig")
old = old.loc[old["threshold"].eq(cfg["threshold"])].set_index("period")
rows, transitions = [], []
for period in cfg["periods"]:
    part = data.loc[data["period"].eq(period)]
    e = ev.loc[ev["period"].eq(period)]
    h, n, m = int(high.loc[part.index].sum()), len(part), len(e)
    freq, length = m / n, e["hours"].mean()
    assert h == int(e["hours"].sum()) and np.isclose(freq * length, h / n)
    assert h == old.loc[period, "high_hours"] and m == old.loc[period, "events_starting"]
    rows.append({"period": period, "observed_hours": n, "events": m, "high_hours": h,
                 "events_per_hour": freq, "mean_duration": length, "high_fraction": h / n,
                 "eligible": int(eligible.loc[part.index].sum()), "onsets": int(onset.loc[part.index].sum()),
                 "onset_eligible_rate": onset.loc[part.index].sum() / eligible.loc[part.index].sum()})
    for prev_high in (False, True):
        idx = part.index[part["prev_peak"].notna() & part["prev_peak"].ge(cfg["threshold"]).eq(prev_high)]
        transitions.append({"period": period, "previous_high": prev_high, "n": len(idx),
                            "current_high": int(high.loc[idx].sum()), "rate": high.loc[idx].mean()})
summary = pd.DataFrame(rows).set_index("period")
contributions = []
for earlier, later in (("Jan-Jun", "Jul-Aug"), ("Jul-Aug", "Sep")):
    a, b = summary.loc[earlier], summary.loc[later]
    frequency = (b.events_per_hour - a.events_per_hour) * (a.mean_duration + b.mean_duration) / 2
    duration = (b.mean_duration - a.mean_duration) * (a.events_per_hour + b.events_per_hour) / 2
    delta = b.high_fraction - a.high_fraction
    assert np.isclose(frequency + duration, delta)
    contributions.append({"earlier": earlier, "later": later, "high_fraction_difference": delta,
                          "frequency_component": frequency, "duration_component": duration,
                          "frequency_share_of_difference": frequency / delta,
                          "duration_share_of_difference": duration / delta})
save(summary.reset_index(), PREFIX, "occupancy")
save(pd.DataFrame(transitions), PREFIX, "transitions")
save(pd.DataFrame(contributions), PREFIX, "contributions")
finish(PREFIX, {"occupancy": rows, "contributions": contributions, "censored_or_cross_period_events": 0})
