"""Verify new arithmetic summaries from the source and saved tables."""
from hashlib import sha256
import json
import numpy as np
import pandas as pd
from analysis_common import TABLES, SOURCE, SOURCE_HASH, PROFILES, PROFILE_HASH, SLOTS

checks = []
assert sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH
assert sha256(PROFILES.read_bytes()).hexdigest() == PROFILE_HASH
raw = pd.read_csv(SOURCE, encoding="utf-8-sig")
valid = raw.loc[raw["시간"].between(0, 23)].copy()
valid["timestamp"] = pd.to_datetime(valid["날짜"].astype(str)) + pd.to_timedelta(valid["시간"], unit="h")
valid = valid.set_index("timestamp").sort_index()
peak = valid[SLOTS].max(axis=1)
records = pd.read_csv(TABLES / "09.27_004_records.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
assert len(records) == 5784
assert np.array_equal(records.peak, peak.reindex(records.timestamp).to_numpy())
assert np.array_equal(records["생산량"], valid["생산량"].reindex(records.timestamp).to_numpy())
checks.append("004 reused source peak and production values agree in5784 rows")

def period(ts):
    return "Jan-Jun" if ts.month <= 6 else "Jul-Aug" if ts.month <= 8 else "Sep"

# Independent consecutive high runs, without analysis_common.events.
runs = []
active, previous = None, None
for ts, value in peak.items():
    if active is not None and (value < 182 or ts - previous != pd.Timedelta(hours=1)):
        runs.append(active)
        active = None
    if value >= 182:
        if active is None:
            active = {"period": period(ts), "hour": ts.hour, "hours": 0}
        active["hours"] += 1
    previous = ts
if active is not None:
    runs.append(active)
ev = pd.DataFrame(runs)
occupancy = pd.read_csv(TABLES / "09.27_011_occupancy.csv", encoding="utf-8-sig")
for _, row in occupancy.iterrows():
    p = peak.loc[[period(ts) == row.period for ts in peak.index]]
    e = ev.loc[ev.period.eq(row.period)]
    assert len(p) == row.observed_hours and int(p.ge(182).sum()) == row.high_hours
    assert len(e) == row.events and np.isclose(e.hours.mean(), row.mean_duration)
    eligible = onsets = 0
    for ts, x in p.items():
        before = peak.get(ts - pd.Timedelta(hours=1), np.nan)
        eligible += int(np.isfinite(before) and before < 182)
        onsets += int(np.isfinite(before) and before < 182 and x >= 182)
    assert eligible == row.eligible and onsets == row.onsets
checks.append("011 independent runs, durations, exact previous exposures and onsets")
components = pd.read_csv(TABLES / "09.27_011_contributions.csv", encoding="utf-8-sig")
assert np.allclose(components.frequency_component + components.duration_component, components.high_fraction_difference)
checks.append("011 frequency plus duration equals occupancy difference")

bins = pd.read_csv(TABLES / "09.27_012_bins.csv", encoding="utf-8-sig")
diff = pd.read_csv(TABLES / "09.27_012_differences.csv", encoding="utf-8-sig")
totals = pd.read_csv(TABLES / "09.27_012_totals.csv", encoding="utf-8-sig")
old = pd.read_csv(TABLES / "09.27_005_standardized_distributions.csv", encoding="utf-8-sig")
for (weighted, p), part in bins.groupby(["profile_weighted", "period"]):
    assert np.isclose(part.standardized_probability.sum(), 1)
    assert np.isclose(part.mean_contribution.sum(), old.loc[old.profile_weighted.eq(weighted) & old.period.eq(p), "mean"].iloc[0])
for _, row in totals.iterrows():
    part = diff.loc[diff.profile_weighted.eq(row.profile_weighted)]
    assert np.isclose(part.mean_contribution_difference.sum(), row.mean_difference)
    assert np.isclose(row.below182_contribution + row.atleast182_contribution, row.mean_difference)
checks.append("012 bin mass and contribution totals reconcile with005 means")

cells = pd.read_csv(TABLES / "09.27_013_hour_cells.csv", encoding="utf-8-sig")
for _, row in cells.iterrows():
    for tag in ("earlier", "later"):
        e = ev.loc[ev.period.eq(row[tag]) & ev.hour.eq(row.hour)]
        assert len(e) == row[f"{tag}_n"]
        if len(e):
            assert np.isclose(e.hours.mean(), row[f"{tag}_mean_duration"])
            assert np.isclose(e.hours.ge(2).mean(), row[f"{tag}_atleast2_fraction"])
std = pd.read_csv(TABLES / "09.27_013_standardization.csv", encoding="utf-8-sig")
for _, row in std.iterrows():
    part = cells.loc[cells.earlier.eq(row.earlier) & cells.later.eq(row.later) & cells.common]
    w = part.earlier_n + part.later_n
    w = w / w.sum()
    for tag in ("earlier", "later"):
        for metric in ("mean_duration", "atleast2_fraction", "atleast3_fraction", "atleast4_fraction", "atleast6_fraction"):
            assert np.isclose((w * part[f"{tag}_{metric}"]).sum(), row[f"{tag}_standardized_{metric}"])
checks.append("013 independent event-hour cells and five standardized duration metrics")

bounds = pd.read_csv(TABLES / "09.27_014_bounds.csv", encoding="utf-8-sig").set_index("scenario")
fullpeak = raw[SLOTS].max(axis=1)
julaug = raw["날짜"].between(20210701, 20210831)
assert julaug.sum() == bounds.loc["all_dated_records", "later_n"]
assert fullpeak.loc[julaug].ge(182).sum() == bounds.loc["all_dated_records", "later_high"]
assert np.allclose(bounds.later_high / bounds.later_n, bounds.later_fraction)
checks.append("014 full dated1488 records and230 high records, scenario fractions")

daily = pd.read_csv(TABLES / "09.27_015_daily.csv", encoding="utf-8-sig")
source_daily = peak.loc[peak.index < "2021-09-01"].groupby(peak.loc[peak.index < "2021-09-01"].index.normalize())
for _, row in daily.iterrows():
    values = source_daily.get_group(pd.Timestamp(row.date))
    assert len(values) == 24
    assert values.lt(26).sum() == row.low_hours and values.ge(182).sum() == row.high_hours
    assert np.isclose(values.mean(), row.mean_peak) and np.isclose(values.var(ddof=0), row.within_variance)
variance = pd.read_csv(TABLES / "09.27_015_variance_partition.csv", encoding="utf-8-sig")
assert np.allclose(variance.within_day_variance + variance.between_day_variance, variance.total_variance)
coexist = pd.read_csv(TABLES / "09.27_015_coexistence.csv", encoding="utf-8-sig")
assert np.allclose(coexist.groupby(["period", "profile_weighted"]).fraction.sum(), 1)
assert (coexist.groupby(["period", "profile_weighted"]).days.sum().to_numpy() == [181, 181, 60, 60]).all()
checks.append("015 source daily tail counts/means/variances, variance identity and category totals")

input_hashes = {name: sha256((TABLES / name).read_bytes()).hexdigest() for name in
                ("09.27_004_records.csv", "09.27_005_strata.csv", "09.27_005_standardized_distributions.csv", "09.27_009_extended_periods.csv")}
for number in range(11, 16):
    prefix = f"09.27_{number:03d}"
    facts = json.loads((TABLES / f"{prefix}_facts.json").read_text(encoding="utf-8"))
    assert facts["frozen_sha256"] == sha256((TABLES / f"{prefix}_frozen.json").read_bytes()).hexdigest()
    assert facts["source_sha256"] == SOURCE_HASH and facts["profile_sha256"] == PROFILE_HASH
checks.append("source, profile and five frozen-config hashes preserved")
result = {"status": "passed", "checks": checks, "input_artifact_sha256": input_hashes,
          "source_sha256": SOURCE_HASH, "profile_sha256": PROFILE_HASH}
(TABLES / "09.27_016_verification.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({"status": "passed", "checks": len(checks), "source_rows": len(raw), "events": len(ev)}))
