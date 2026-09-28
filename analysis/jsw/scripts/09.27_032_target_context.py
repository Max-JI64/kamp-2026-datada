"""Compare the target within observed production context and audit weather support."""
from hashlib import sha256
import json
import numpy as np
import pandas as pd
from analysis_common import TABLES, SOURCE, SOURCE_HASH, PROFILES, PROFILE_HASH, load, save, finish

PREFIX = "09.27_032"
assert sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH
assert sha256(PROFILES.read_bytes()).hexdigest() == PROFILE_HASH
setting = json.loads((TABLES / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
assert "24 exact hour" in setting["primary"]
_, data = load()
data["high"] = data.peak.ge(182)
assert data.groupby("period").high.sum().to_dict() == {"Jan-Jun": 235, "Jul-Aug": 220}

raw_rows = []
for key, part in data.groupby(["period", "weekend", "production_positive"]):
    period, weekend, positive = key
    raw_rows.append({"period": period, "weekend": int(weekend), "production_positive": int(positive),
                     "hours": len(part), "dates": part.date.nunique(), "high_hours": int(part.high.sum()),
                     "high_rate": float(part.high.mean()),
                     "profile_weighted_high_rate": float(np.average(part.high, weights=part.profile_weight)),
                     "temperature_min": float(part["기온"].min()),
                     "temperature_median": float(part["기온"].median()),
                     "temperature_max": float(part["기온"].max())})
raw = pd.DataFrame(raw_rows).sort_values(["period", "weekend", "production_positive"])
assert raw.groupby("period").hours.sum().to_dict() == {"Jan-Jun": 4344, "Jul-Aug": 1440}
assert raw.groupby("period").high_hours.sum().to_dict() == {"Jan-Jun": 235, "Jul-Aug": 220}
save(raw, PREFIX, "production_weekend_context")

positive_weekday = data.loc[data.weekend.eq(0) & data.production_positive.eq(1)].copy()
def hour_standardization(frame, weighted, min_cell_n=1):
    records = []
    for hour in range(24):
        row = {"hour": hour, "profile_weighted": weighted}
        for period, tag in (("Jan-Jun", "earlier"), ("Jul-Aug", "later")):
            part = frame.loc[(frame.period == period) & (frame.hour == hour)]
            w = part.profile_weight.to_numpy() if weighted else np.ones(len(part))
            row[f"{tag}_n"] = len(part)
            row[f"{tag}_high"] = int(part.high.sum())
            row[f"{tag}_weight"] = float(w.sum())
            row[f"{tag}_high_weight"] = float(w[part.high.to_numpy()].sum())
            row[f"{tag}_dates"] = int(part.date.nunique())
        row["common"] = row["earlier_n"] >= min_cell_n and row["later_n"] >= min_cell_n
        records.append(row)
    cells = pd.DataFrame(records)
    support = cells.loc[cells.common].copy()
    outcome = {"profile_weighted": weighted, "common_cells": len(support),
               "excluded_earlier_n": int(cells.loc[~cells.common, "earlier_n"].sum()),
               "excluded_later_n": int(cells.loc[~cells.common, "later_n"].sum())}
    if len(support):
        ref = support.earlier_weight + support.later_weight
        ref = ref / ref.sum()
        for tag in ("earlier", "later"):
            outcome[f"{tag}_n"] = int(cells[f"{tag}_n"].sum())
            outcome[f"{tag}_high"] = int(cells[f"{tag}_high"].sum())
            outcome[f"{tag}_raw_rate"] = float(cells[f"{tag}_high_weight"].sum() / cells[f"{tag}_weight"].sum())
            outcome[f"{tag}_standardized_rate"] = float((ref * support[f"{tag}_high_weight"] / support[f"{tag}_weight"]).sum())
            outcome[f"{tag}_min_n"] = int(support[f"{tag}_n"].min())
        outcome["standardized_difference"] = outcome["later_standardized_rate"] - outcome["earlier_standardized_rate"]
    return cells, outcome

primary = []
for weighted in (False, True):
    cells, summary = hour_standardization(positive_weekday, weighted)
    assert summary["common_cells"] == 24
    assert summary["excluded_earlier_n"] == summary["excluded_later_n"] == 0
    primary.append(summary)
    if not weighted:
        save(cells, PREFIX, "positive_weekday_hour_cells")

temperature_ranges = []
for period, part in positive_weekday.groupby("period"):
    q5, q95 = part["기온"].quantile([0.05, 0.95]).tolist()
    temperature_ranges.append({"period": period, "q05": float(q5), "q95": float(q95),
                               "hours": len(part), "dates": int(part.date.nunique())})
lower = max(row["q05"] for row in temperature_ranges)
upper = min(row["q95"] for row in temperature_ranges)
assert lower <= upper
overlap = positive_weekday.loc[positive_weekday["기온"].between(lower, upper)].copy()
temperature_support = {"lower": float(lower), "upper": float(upper),
                       "period_ranges": temperature_ranges,
                       "period_exposure": [{"period": period, "hours": len(part), "dates": int(part.date.nunique()),
                                            "high_hours": int(part.high.sum()), "high_rate": float(part.high.mean())}
                                           for period, part in overlap.groupby("period")]}
temp_cells, temp_summary = hour_standardization(overlap, False, min_cell_n=10)
eligible = (all(row["dates"] >= 10 for row in temperature_support["period_exposure"])
            and temp_summary["common_cells"] >= 12)
temperature_support["comparison_eligible"] = bool(eligible)
temperature_support["common_hour_cells_with_10_each"] = int(temp_summary["common_cells"])
temperature_support["common_hour_excluded_n"] = {"Jan-Jun": temp_summary["excluded_earlier_n"],
                                                  "Jul-Aug": temp_summary["excluded_later_n"]}
if eligible:
    temperature_support["comparison"] = temp_summary
save(temp_cells, PREFIX, "temperature_overlap_hour_cells")
save(pd.DataFrame(primary), PREFIX, "positive_weekday_standardization")
save(pd.DataFrame(temperature_ranges), PREFIX, "temperature_ranges")
finish(PREFIX, {"raw_context": raw_rows, "positive_weekday_standardization": primary,
                "temperature_support": temperature_support})
