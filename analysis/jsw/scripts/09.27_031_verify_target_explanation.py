"""Independently check 030-035 source arithmetic and artifact links."""
from hashlib import sha256
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from analysis_common import BASE, TABLES, SOURCE_HASH, PROFILE_HASH, load

_, data = load()
assert len(data) == 5784
assert data.groupby("period").size().to_dict() == {"Jan-Jun": 4344, "Jul-Aug": 1440}
high = data.peak.ge(182)
assert int(high.sum()) == 455
assert data.loc[high].groupby("period").size().to_dict() == {"Jan-Jun": 235, "Jul-Aug": 220}
assert data.loc[high & data.period.eq("Jul-Aug"), "weekend"].eq(0).all()

months = pd.read_csv(TABLES / "09.27_030_monthly_high_rates.csv", encoding="utf-8-sig").set_index("month")
for month, rows in data.groupby("month"):
    record = months.loc[month]
    selected = rows.peak.ge(182)
    assert len(rows) == record.hours
    assert len(rows) // 24 == record.days
    assert int(selected.sum()) == record.high_hours
    assert np.isclose(selected.mean(), record.high_rate, atol=1e-12)
    assert np.isclose(np.average(selected, weights=rows.profile_weight), record.weighted_high_rate, atol=1e-12)

saved = pd.read_csv(TABLES / "09.27_030_cell_contributions.csv", encoding="utf-8-sig")
facts = json.loads((TABLES / "09.27_030_facts.json").read_text(encoding="utf-8"))
frozen = TABLES / "09.27_030_frozen.json"
assert facts["source_sha256"] == SOURCE_HASH
assert facts["profile_sha256"] == PROFILE_HASH
assert facts["frozen_sha256"] == sha256(frozen.read_bytes()).hexdigest()
for weighted in (False, True):
    cells = saved.loc[saved.profile_weighted.eq(weighted)]
    assert len(cells) == 48
    weight = data.profile_weight if weighted else pd.Series(1.0, index=data.index)
    original = data.assign(weight=weight, high_weight=weight * high)
    fresh = original.groupby(["hour", "weekend", "period"]).agg(exposure=("weight", "sum"),
                                                                  events=("high_weight", "sum"))
    pooled = fresh.exposure.groupby(level=[0, 1]).sum()
    reference = pooled / pooled.sum()
    for row in cells.itertuples():
        a = fresh.loc[(row.hour, row.weekend, "Jan-Jun")]
        b = fresh.loc[(row.hour, row.weekend, "Jul-Aug")]
        expected = reference.loc[(row.hour, row.weekend)] * (b.events / b.exposure - a.events / a.exposure)
        assert np.isclose(expected, row.contribution, atol=1e-12)
    summary = next(x for x in facts["summary"] if x["profile_weighted"] == weighted)
    assert np.isclose(cells.contribution.sum(), summary["gap"], atol=1e-12)

# The substantive 032-034 conclusions are reconstructed from source rows.
source = data.copy()
source["high"] = source.peak.ge(182)
source["target_mean"] = source["평균"].astype(float)
source["target_max"] = source.peak.astype(float)
context = source.groupby(["period", "weekend", "production_positive"]).agg(
    hours=("high", "size"), high_hours=("high", "sum"))
saved_context = pd.read_csv(TABLES / "09.27_032_production_weekend_context.csv", encoding="utf-8-sig")
for row in saved_context.itertuples():
    match = context.loc[(row.period, row.weekend, row.production_positive)]
    assert row.hours == match.hours and row.high_hours == match.high_hours
positive = source.loc[source.weekend.eq(0) & source.production_positive.eq(1)]
assert positive.groupby("period").high.sum().to_dict() == {"Jan-Jun": 195, "Jul-Aug": 218}
saved_032 = json.loads((TABLES / "09.27_032_facts.json").read_text(encoding="utf-8"))
for weighted in (False, True):
    recorded = next(x for x in saved_032["positive_weekday_standardization"] if x["profile_weighted"] == weighted)
    check = positive.assign(weight=positive.profile_weight if weighted else 1.0)
    check["weighted_high"] = check.weight * check.high
    cells = check.groupby(["hour", "period"]).agg(weight=("weight", "sum"), high=("weighted_high", "sum"))
    pooled = cells.weight.groupby(level="hour").sum()
    pooled /= pooled.sum()
    for period, tag in (("Jan-Jun", "earlier"), ("Jul-Aug", "later")):
        part = cells.xs(period, level="period")
        expected = float((pooled * part.high / part.weight).sum())
        assert np.isclose(expected, recorded[f"{tag}_standardized_rate"], atol=1e-12)
temp = saved_032["temperature_support"]
assert np.isclose(temp["lower"], max(positive.groupby("period")["기온"].quantile(.05)))
assert np.isclose(temp["upper"], min(positive.groupby("period")["기온"].quantile(.95)))
assert temp["comparison_eligible"] is False
in_range = positive.loc[positive["기온"].between(temp["lower"], temp["upper"])]
for record in temp["period_exposure"]:
    part = in_range.loc[in_range.period.eq(record["period"])]
    assert len(part) == record["hours"] and part.date.nunique() == record["dates"]
    assert int(part.high.sum()) == record["high_hours"]

saved_033 = json.loads((TABLES / "09.27_033_facts.json").read_text(encoding="utf-8"))
for item in saved_033["distribution"]:
    part = (source if item["scope"] == "all_valid" else positive)
    part = part.loc[part.period.eq(item["period"]), item["target"]]
    assert len(part) == item["n"]
    assert np.isclose(part.mean(), item["mean"], atol=1e-12)
    assert np.isclose(part.quantile(.95), item["q95"], atol=1e-12)
for item in saved_033["standardization"]:
    check = positive.assign(weight=positive.profile_weight if item["profile_weighted"] else 1.0)
    check["weighted_target"] = check.weight * check[item["target"]]
    cells = check.groupby(["hour", "period"]).agg(weight=("weight", "sum"), target=("weighted_target", "sum"))
    pooled = cells.weight.groupby(level="hour").sum()
    pooled /= pooled.sum()
    early = cells.xs("Jan-Jun", level="period")
    late = cells.xs("Jul-Aug", level="period")
    assert np.isclose((pooled * early.target / early.weight).sum(), item["earlier_standardized_mean"], atol=1e-12)
    assert np.isclose((pooled * late.target / late.weight).sum(), item["later_standardized_mean"], atol=1e-12)

saved_034 = pd.read_csv(TABLES / "09.27_034_stratum_components.csv", encoding="utf-8-sig")
totals_034 = pd.read_csv(TABLES / "09.27_034_totals.csv", encoding="utf-8-sig")
for target in ("target_mean", "target_max"):
    rows = saved_034.loc[saved_034.target.eq(target)]
    total = totals_034.loc[totals_034.target.eq(target)].iloc[0]
    assert len(rows) == 4
    for row in rows.itertuples():
        for period, tag in (("Jan-Jun", "earlier"), ("Jul-Aug", "later")):
            part = source.loc[source.period.eq(period) & source.weekend.eq(row.weekend)
                              & source.production_positive.eq(row.production_positive), target]
            assert len(part) == getattr(row, f"{tag}_n")
            assert np.isclose(part.mean(), getattr(row, f"{tag}_mean"), atol=1e-12)
        assert np.isclose(row.mix_contribution + row.within_contribution, row.total_contribution, atol=1e-12)
    a = source.loc[source.period.eq("Jan-Jun"), target].mean()
    b = source.loc[source.period.eq("Jul-Aug"), target].mean()
    assert np.isclose(a, total.earlier_mean, atol=1e-12)
    assert np.isclose(b, total.later_mean, atol=1e-12)
    assert np.isclose(rows.total_contribution.sum(), b - a, atol=1e-12)
    assert np.isclose(rows.mix_contribution.sum() + rows.within_contribution.sum(), b - a, atol=1e-12)

saved_035 = json.loads((TABLES / "09.27_035_facts.json").read_text(encoding="utf-8"))
summer = source.loc[source.month.isin([6, 7, 8]) & source.weekend.eq(0)
                    & source.production_positive.eq(1)].copy()
summer["high"] = summer.high.astype(float)
for row in saved_035["monthly_context"]:
    part = summer.loc[summer.month.eq(row["month"])]
    assert len(part) == row["hours"] and part.date.nunique() == row["dates"]
    assert int(part.high.sum()) == row["high_hours"]
for row in saved_035["standardized_months"]:
    weighted = row["profile_weighted"]
    frame = summer.assign(weight=summer.profile_weight if weighted else 1.0)
    frame["weighted_target"] = frame.weight * frame[row["metric"]]
    cells = frame.groupby(["month", "hour"]).agg(weight=("weight", "sum"), target=("weighted_target", "sum"))
    ref = cells.weight.groupby(level="hour").sum()
    ref /= ref.sum()
    for month, name in ((6, "june"), (7, "july"), (8, "august")):
        part = cells.xs(month, level="month")
        assert np.isclose((ref * part.target / part.weight).sum(), row[name], atol=1e-12)

for number in (32, 33, 34, 35):
    prefix = f"09.27_{number:03d}"
    fact = json.loads((TABLES / f"{prefix}_facts.json").read_text(encoding="utf-8"))
    assert fact["source_sha256"] == SOURCE_HASH
    assert fact["profile_sha256"] == PROFILE_HASH
    assert fact["frozen_sha256"] == sha256((TABLES / f"{prefix}_frozen.json").read_bytes()).hexdigest()

links = 0
for report in (BASE / "reports/09.27_030_gap_location.md",
               BASE / "reports/09.27_031_target_explanation_audit.md",
               BASE / "reports/09.27_032_target_context.md",
               BASE / "reports/09.27_033_target_distribution.md",
               BASE / "reports/09.27_034_mean_reversal.md",
               BASE / "reports/09.27_035_adjacent_months.md"):
    body = report.read_text(encoding="utf-8")
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", body):
        if target.startswith(("https://", "http://", "#")):
            continue
        assert (report.parent / target.split("#", 1)[0]).resolve().exists(), (report, target)
        links += 1
result = {"status": "passed", "source_sha256": SOURCE_HASH, "profile_sha256": PROFILE_HASH,
          "source_rows": len(data), "high_hours": int(high.sum()), "monthly_rows_checked": len(months),
          "calendar_cells_checked": 96, "report_links_checked": links,
          "target_context_rows_checked": len(saved_context),
          "continuous_target_distributions_checked": len(saved_033["distribution"]),
          "reversal_strata_checked": len(saved_034),
          "adjacent_month_standardizations_checked": len(saved_035["standardized_months"]),
          "natural_language_causal_claims_automatically_verified": False}
(TABLES / "09.27_031_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"status": result["status"], "rows": result["source_rows"],
                  "calendar_cells": result["calendar_cells_checked"], "links": links}))
