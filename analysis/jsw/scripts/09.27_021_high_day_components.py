"""Separate high-day prevalence from high-hour burden within high days."""
import json
import numpy as np
import pandas as pd
from analysis_common import TABLES, save, finish

PREFIX = "09.27_021"
cfg = json.loads((TABLES / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
daily = pd.read_csv(TABLES / cfg["input"], encoding="utf-8-sig")
rows, contributions = [], []
for weighted in (False, True):
    for period, group in daily.groupby("period"):
        w = group.profile_weight.to_numpy(copy=True) if weighted else np.ones(len(group))
        w /= w.sum()
        high = group.high_hours.gt(0).to_numpy()
        probability = w[high].sum()
        burden = np.average(group.loc[high, "high_hours"], weights=w[high])
        fraction = np.average(group.high_hours, weights=w) / 24
        assert np.isclose(probability * burden / 24, fraction)
        rows.append({"profile_weighted": weighted, "period": period, "days": len(group),
                     "high_days": int(high.sum()), "high_day_fraction": probability,
                     "mean_high_hours_per_high_day": burden, "high_fraction": fraction,
                     "high_hours": int(group.high_hours.sum()),
                     "median_high_hours_per_high_day": group.loc[high, "high_hours"].median()})
    table = pd.DataFrame(rows).loc[lambda x: x.profile_weighted.eq(weighted)].set_index("period")
    a, b = table.loc["Jan-Jun"], table.loc["Jul-Aug"]
    extensive = (b.high_day_fraction - a.high_day_fraction) * (a.mean_high_hours_per_high_day + b.mean_high_hours_per_high_day) / 48
    intensive = (b.mean_high_hours_per_high_day - a.mean_high_hours_per_high_day) * (a.high_day_fraction + b.high_day_fraction) / 48
    delta = b.high_fraction - a.high_fraction
    assert np.isclose(extensive + intensive, delta)
    contributions.append({"profile_weighted": weighted, "high_fraction_difference": delta,
                          "high_day_frequency_component": extensive, "high_hours_within_high_day_component": intensive})
save(pd.DataFrame(rows), PREFIX, "periods")
save(pd.DataFrame(contributions), PREFIX, "contributions")
save(daily.groupby(["period", "high_hours"]).size().rename("days").reset_index(), PREFIX, "burden_distribution")
finish(PREFIX, {"periods": rows, "contributions": contributions})
