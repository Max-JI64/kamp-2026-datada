"""Compare duration within common observed event-start hours."""
import json
import numpy as np
import pandas as pd
from analysis_common import TABLES, load, events, save, finish

PREFIX = "09.27_013"
cfg = json.loads((TABLES / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
_, data = load(include_september=True)
_, _, _, ev = events(data, cfg["threshold"])
assert not ev[["left_censored", "right_censored", "crosses_period"]].any().any()
ev["hour_group"] = np.select([ev.hour.eq(8), ev.hour.eq(13)], ["08h", "13h"], default="other")
group_rows = []
for (period, group), part in ev.groupby(["period", "hour_group"]):
    group_rows.append({"period": period, "hour_group": group, "events": len(part),
                       "dates": part["date"].nunique(), "mean_duration": part.hours.mean(),
                       "median_duration": part.hours.median(),
                       **{f"atleast{k}_n": int(part.hours.ge(k).sum()) for k in cfg["duration_survival_hours"]},
                       **{f"atleast{k}_fraction": part.hours.ge(k).mean() for k in cfg["duration_survival_hours"]}})
cells, results = [], []
for earlier, later in cfg["comparisons"]:
    pair = ev.loc[ev.period.isin([earlier, later])]
    rows = []
    for hour, group in pair.groupby("hour"):
        row = {"earlier": earlier, "later": later, "hour": hour}
        for period, tag in ((earlier, "earlier"), (later, "later")):
            part = group.loc[group.period.eq(period)]
            row[f"{tag}_n"] = len(part)
            row[f"{tag}_dates"] = part.date.nunique()
            row[f"{tag}_mean_duration"] = part.hours.mean()
            for k in cfg["duration_survival_hours"]:
                row[f"{tag}_atleast{k}_fraction"] = part.hours.ge(k).mean()
        row["common"] = row["earlier_n"] > 0 and row["later_n"] > 0
        rows.append(row)
    table = pd.DataFrame(rows)
    common = table.loc[table.common].copy()
    weights = (common.earlier_n + common.later_n)
    weights = weights / weights.sum()
    result = {"earlier": earlier, "later": later, "common_hours": len(common),
              "total_hours": len(table), "small_cells_either_n_le2": int(common[["earlier_n", "later_n"]].min(axis=1).le(2).sum())}
    for period, tag in ((earlier, "earlier"), (later, "later")):
        part = pair.loc[pair.period.eq(period)]
        result[f"{tag}_all_events"] = len(part)
        result[f"{tag}_supported_events"] = int(common[f"{tag}_n"].sum())
        result[f"{tag}_excluded_events"] = len(part) - result[f"{tag}_supported_events"]
        for metric in ("mean_duration", *[f"atleast{k}_fraction" for k in cfg["duration_survival_hours"]]):
            result[f"{tag}_standardized_{metric}"] = float(np.sum(weights * common[f"{tag}_{metric}"]))
    assert np.isclose(weights.sum(), 1)
    cells.extend(rows)
    results.append(result)
save(pd.DataFrame(group_rows), PREFIX, "duration_groups")
save(pd.DataFrame(cells), PREFIX, "hour_cells")
save(pd.DataFrame(results), PREFIX, "standardization")
finish(PREFIX, {"standardization": results, "event_counts": ev.groupby("period").size().to_dict()})
