"""Locate 029's descriptive period gap across existing calendar cells."""
from hashlib import sha256
import json
import numpy as np
import pandas as pd
from analysis_common import TABLES, SOURCE, SOURCE_HASH, PROFILES, PROFILE_HASH, save, finish

PREFIX = "09.27_030"
assert sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH
assert sha256(PROFILES.read_bytes()).hexdigest() == PROFILE_HASH
config = json.loads((TABLES / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
assert "08 and 13" in config["method"]
prior = json.loads((TABLES / "09.27_029_facts.json").read_text(encoding="utf-8"))["standardization"]
strata = pd.read_csv(TABLES / "09.27_029_strata.csv", encoding="utf-8-sig")
high = strata.loc[strata.metric.eq("high")].copy()
assert len(high) == 96 and high.common.all()
cell_parts, hour_parts, weekend_parts, group_parts = [], [], [], []
for weighted in (False, True):
    part = high.loc[high.profile_weighted.eq(weighted)].copy().sort_values(["hour", "weekend"])
    assert len(part) == 48
    weights = part.earlier_exposure_weight + part.later_exposure_weight
    part["reference_weight"] = weights / weights.sum()
    part["earlier_rate"] = part.earlier_event_weight / part.earlier_exposure_weight
    part["later_rate"] = part.later_event_weight / part.later_exposure_weight
    part["difference"] = part.later_rate - part.earlier_rate
    part["contribution"] = part.reference_weight * part.difference
    match = next(r for r in prior if r["metric"] == "high" and r["profile_weighted"] == weighted)
    assert np.isclose(part.contribution.sum(), match["standardized_difference"], atol=1e-12)
    assert np.isclose((part.reference_weight * part.earlier_rate).sum(), match["earlier_standardized_rate"], atol=1e-12)
    assert np.isclose((part.reference_weight * part.later_rate).sum(), match["later_standardized_rate"], atol=1e-12)
    cell_parts.append(part)
    by_hour = part.groupby("hour", as_index=False).agg(reference_weight=("reference_weight", "sum"),
        earlier_n=("earlier_n", "sum"), later_n=("later_n", "sum"),
        earlier_high=("earlier_events", "sum"), later_high=("later_events", "sum"),
        contribution=("contribution", "sum"))
    by_hour["profile_weighted"] = weighted
    hour_parts.append(by_hour)
    by_weekend = part.groupby("weekend", as_index=False).agg(reference_weight=("reference_weight", "sum"),
        earlier_n=("earlier_n", "sum"), later_n=("later_n", "sum"),
        earlier_high=("earlier_events", "sum"), later_high=("later_events", "sum"),
        contribution=("contribution", "sum"))
    by_weekend["profile_weighted"] = weighted
    weekend_parts.append(by_weekend)
    for label, mask in (("08+13", part.hour.isin([8, 13])), ("other_22", ~part.hour.isin([8, 13]))):
        subset = part.loc[mask]
        mass = subset.reference_weight.sum()
        group_parts.append({"profile_weighted": weighted, "group": label,
                            "hour_count": subset.hour.nunique(), "reference_weight": float(mass),
                            "earlier_n": int(subset.earlier_n.sum()), "later_n": int(subset.later_n.sum()),
                            "earlier_high": int(subset.earlier_events.sum()), "later_high": int(subset.later_events.sum()),
                            "earlier_conditional_rate": float((subset.reference_weight * subset.earlier_rate).sum() / mass),
                            "later_conditional_rate": float((subset.reference_weight * subset.later_rate).sum() / mass),
                            "conditional_difference": float(subset.contribution.sum() / mass),
                            "contribution": float(subset.contribution.sum()),
                            "share_of_total_gap": float(subset.contribution.sum() / part.contribution.sum())})
    assert np.isclose(sum(x["contribution"] for x in group_parts[-2:]), match["standardized_difference"], atol=1e-12)

records = pd.read_csv(TABLES / "09.27_004_records.csv", encoding="utf-8-sig")
records["month"] = pd.to_datetime(records.date).dt.month
months = []
for month, part in records.groupby("month", sort=True):
    assert len(part) % 24 == 0
    months.append({"month": int(month), "days": int(len(part) / 24), "hours": len(part),
                   "high_hours": int(part.high.sum()), "high_rate": float(part.high.mean()),
                   "weighted_high_rate": float(np.average(part.high, weights=part.profile_weight))})
assert sum(m["high_hours"] for m in months[:6]) == 235
assert sum(m["high_hours"] for m in months[6:]) == 220
save(pd.concat(cell_parts, ignore_index=True), PREFIX, "cell_contributions")
save(pd.concat(hour_parts, ignore_index=True), PREFIX, "hour_contributions")
save(pd.concat(weekend_parts, ignore_index=True), PREFIX, "weekend_contributions")
save(pd.DataFrame(group_parts), PREFIX, "predefined_hour_groups")
save(pd.DataFrame(months), PREFIX, "monthly_high_rates")
summary = []
for weighted in (False, True):
    hours = hour_parts[int(weighted)]
    groups = [g for g in group_parts if g["profile_weighted"] == weighted]
    weekend = weekend_parts[int(weighted)]
    summary.append({"profile_weighted": weighted, "gap": float(hours.contribution.sum()),
                    "positive_hours": int(hours.contribution.gt(0).sum()),
                    "zero_hours": int(hours.contribution.eq(0).sum()),
                    "negative_hours": int(hours.contribution.lt(0).sum()),
                    "top_five_hours": hours.nlargest(5, "contribution")[["hour", "contribution"]].to_dict("records"),
                    "predefined_groups": groups,
                    "weekday_contribution": float(weekend.loc[weekend.weekend.eq(0), "contribution"].iloc[0]),
                    "weekend_contribution": float(weekend.loc[weekend.weekend.eq(1), "contribution"].iloc[0])})
finish(PREFIX, {"summary": summary, "monthly_high_rates": months,
                "input_reuse": ["09.27_029_strata.csv", "09.27_004_records.csv"]})
