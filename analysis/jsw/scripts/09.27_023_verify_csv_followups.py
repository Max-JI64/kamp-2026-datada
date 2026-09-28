"""Validate the daily follow-ups against original values and exact dates."""
from hashlib import sha256
import json
import numpy as np
import pandas as pd
from analysis_common import TABLES, SOURCE, SOURCE_HASH, PROFILES, PROFILE_HASH, SLOTS

assert sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH
assert sha256(PROFILES.read_bytes()).hexdigest() == PROFILE_HASH
raw = pd.read_csv(SOURCE, encoding="utf-8-sig")
valid = raw.loc[raw["시간"].between(0, 23) & raw["날짜"].lt(20210901)].copy()
valid["date"] = pd.to_datetime(valid["날짜"].astype(str))
valid["peak"] = valid[SLOTS].max(axis=1)
source_daily = valid.groupby("date").agg(mean_peak=("peak", "mean"),
    low_hours=("peak", lambda x: x.lt(26).sum()), high_hours=("peak", lambda x: x.ge(182).sum()))
source_daily["period"] = np.where(source_daily.index < "2021-07-01", "Jan-Jun", "Jul-Aug")
source_daily["category"] = np.select([
    source_daily.low_hours.gt(0) & source_daily.high_hours.gt(0),
    source_daily.low_hours.gt(0), source_daily.high_hours.gt(0)],
    ["both_low_and_high", "low_only", "high_only"], default="neither")
profiles = pd.read_csv(PROFILES, encoding="utf-8-sig", dtype={"date": str})
profiles["date"] = pd.to_datetime(profiles.date)
profiles["period"] = np.where(profiles.date < "2021-07-01", "Jan-Jun", "Jul-Aug")
profiles["weight"] = 1 / profiles.groupby(["period", "profile_id"]).date.transform("size")
source_daily["profile_weight"] = profiles.set_index("date").weight.reindex(source_daily.index)
old = pd.read_csv(TABLES / "09.27_015_daily.csv", encoding="utf-8-sig", parse_dates=["date"]).set_index("date")
assert len(source_daily) == len(old) == 241
for col in ("mean_peak", "low_hours", "high_hours", "profile_weight"):
    assert np.allclose(source_daily[col], old.reindex(source_daily.index)[col])
assert np.array_equal(source_daily.category, old.reindex(source_daily.index).category)
checks = ["015 daily reused values and weights independently agree with source241days"]

cells = pd.read_csv(TABLES / "09.27_020_categories.csv", encoding="utf-8-sig")
for _, row in cells.iterrows():
    period = source_daily.loc[source_daily.period.eq(row.period)]
    weights = period.profile_weight.to_numpy() if row.profile_weighted else np.ones(len(period))
    mask = period.category.eq(row.category).to_numpy()
    assert mask.sum() == row.days
    assert np.isclose(weights[mask].sum() / weights.sum(), row.probability)
    assert np.isclose(np.average(period.loc[mask, "mean_peak"], weights=weights[mask]), row.mean_peak)
comp = pd.read_csv(TABLES / "09.27_020_contributions.csv", encoding="utf-8-sig")
totals = pd.read_csv(TABLES / "09.27_020_totals.csv", encoding="utf-8-sig")
assert np.allclose(comp.composition_component + comp.level_component, comp.total_mean_component)
for _, row in totals.iterrows():
    c = comp.loc[comp.profile_weighted.eq(row.profile_weighted)]
    assert np.isclose(c.total_mean_component.sum(), row.mean_difference)
    assert np.isclose(row.composition_component + row.level_component, row.mean_difference)
checks.append("020 source category counts/means/probabilities and decomposition identity")

periods = pd.read_csv(TABLES / "09.27_021_periods.csv", encoding="utf-8-sig")
for _, row in periods.iterrows():
    p = source_daily.loc[source_daily.period.eq(row.period)]
    weights = p.profile_weight.to_numpy() if row.profile_weighted else np.ones(len(p))
    assert p.high_hours.gt(0).sum() == row.high_days and p.high_hours.sum() == row.high_hours
    assert np.isclose(np.average(p.high_hours, weights=weights) / 24, row.high_fraction)
    assert np.isclose(row.high_day_fraction * row.mean_high_hours_per_high_day / 24, row.high_fraction)
contributions = pd.read_csv(TABLES / "09.27_021_contributions.csv", encoding="utf-8-sig")
assert np.allclose(contributions.high_day_frequency_component + contributions.high_hours_within_high_day_component,
                   contributions.high_fraction_difference)
checks.append("021 source high-day/hour denominators and frequency-times-burden identity")

pair_rows = []
for date, row in source_daily.iterrows():
    prior_date = date - pd.Timedelta(days=1)
    if prior_date not in source_daily.index:
        continue
    prior = source_daily.loc[prior_date]
    if prior.period != row.period:
        continue
    pair_rows.append({"date": date, "period": row.period, "previous_high_day": int(prior.high_hours > 0),
                      "high_day": int(row.high_hours > 0), "weekday": date.weekday(), "weight": row.profile_weight})
pair = pd.DataFrame(pair_rows)
assert len(pair) == 237
transitions = pd.read_csv(TABLES / "09.27_022_raw_transitions.csv", encoding="utf-8-sig")
for _, row in transitions.iterrows():
    p = pair.loc[pair.period.eq(row.period) & pair.previous_high_day.eq(row.previous_high_day)]
    weights = p.weight if row.profile_weighted else np.ones(len(p))
    assert len(p) == row.pairs and p.high_day.sum() == row.current_high_days
    assert np.isclose(np.average(p.high_day, weights=weights), row.current_high_fraction)
weekday_cells = pd.read_csv(TABLES / "09.27_022_weekday_cells.csv", encoding="utf-8-sig")
std = pd.read_csv(TABLES / "09.27_022_standardization.csv", encoding="utf-8-sig")
for _, row in weekday_cells.iterrows():
    for state in (0, 1):
        p = pair.loc[pair.period.eq(row.period) & pair.weekday.eq(row.weekday) & pair.previous_high_day.eq(state)]
        w = p.weight if row.profile_weighted else np.ones(len(p))
        assert len(p) == row[f"state{state}_n"] and p.high_day.sum() == row[f"state{state}_high"]
        assert np.isclose(np.sum(w), row[f"state{state}_exposure_weight"])
        assert np.isclose(np.sum(w * p.high_day), row[f"state{state}_high_weight"])
for _, row in std.iterrows():
    c = weekday_cells.loc[weekday_cells.period.eq(row.period) & weekday_cells.profile_weighted.eq(row.profile_weighted) & weekday_cells.common]
    refs = c.state0_exposure_weight + c.state1_exposure_weight
    refs = refs / refs.sum()
    for state in (0, 1):
        assert np.isclose(np.sum(refs * c[f"state{state}_high_weight"] / c[f"state{state}_exposure_weight"]), row[f"state{state}_standardized_fraction"])
checks.append("022 independent exact-day237pairs, weekday cells and standardized proportions")

for number in (20, 21, 22):
    prefix = f"09.27_{number:03d}"
    facts = json.loads((TABLES / f"{prefix}_facts.json").read_text(encoding="utf-8"))
    assert facts["frozen_sha256"] == sha256((TABLES / f"{prefix}_frozen.json").read_bytes()).hexdigest()
checks.append("source, profile and three frozen-setting hashes agree")
result = {"status": "passed", "checks": checks, "source_sha256": SOURCE_HASH, "profile_sha256": PROFILE_HASH,
          "reused_daily_sha256": sha256((TABLES / "09.27_015_daily.csv").read_bytes()).hexdigest()}
(TABLES / "09.27_023_verification.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({"status": "passed", "check_groups": len(checks), "days": len(source_daily), "exact_day_pairs": len(pair)}))
