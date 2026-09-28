"""Check artifact integrity and independently verify the cluster-context counterexample."""
from hashlib import sha256
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from analysis_common import BASE, TABLES, SOURCE, SOURCE_HASH, PROFILES, PROFILE_HASH, SLOTS

REPORTS = BASE / "reports"
SCRIPTS = BASE / "scripts"
assert sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH
assert sha256(PROFILES.read_bytes()).hexdigest() == PROFILE_HASH
reports = sorted(p for p in REPORTS.glob("09.*_*.md") if int(p.name.split("_")[1]) <= 29)
numbers = {int(p.name.split("_")[1]) for p in reports}
expected = set(range(1, 17)) | set(range(20, 30))
assert numbers == expected, (sorted(numbers), sorted(expected))
missing_links = []
link_count = 0
for report in reports:
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", report.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "#")):
            continue
        target = target.split("#", 1)[0]
        link_count += 1
        if not (report.parent / target).resolve().exists():
            missing_links.append(f"{report.name}: {target}")
assert not missing_links, missing_links
primary = list(range(2, 10)) + list(range(11, 16)) + [20, 21, 22, 24, 25, 27, 29]
hash_checks = []
for number in primary:
    prefix = f"09.27_{number:03d}"
    frozen = TABLES / f"{prefix}_frozen.json"
    facts = TABLES / f"{prefix}_facts.json"
    assert frozen.exists() and facts.exists(), prefix
    fact = json.loads(facts.read_text(encoding="utf-8"))
    assert fact["source_sha256"] == SOURCE_HASH
    if "profile_sha256" in fact:
        assert fact["profile_sha256"] == PROFILE_HASH
    assert fact["frozen_sha256"] == sha256(frozen.read_bytes()).hexdigest()
    assert list(SCRIPTS.glob(f"{prefix}_*.py")), prefix
    hash_checks.append(prefix)
for number in (10, 16, 23, 26):
    path = TABLES / f"09.27_{number:03d}_verification.json"
    assert path.exists()
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["source_sha256"] == SOURCE_HASH
    assert record["profile_sha256"] == PROFILE_HASH
    if number == 10:
        # The older review predates the explicit status field.
        assert record["high_hour_reconciliation"] == 455
        assert record["cluster_assignments_per_fit"] == 241
    else:
        assert record["status"] == "passed"

# Verify the strongest cluster-context claim from the original source, not from
# the EDA-derived profile table alone.
raw = pd.read_csv(SOURCE, encoding="utf-8-sig")
valid = raw.loc[raw["시간"].between(0, 23) & raw["날짜"].lt(20210901)].copy()
valid["date"] = pd.to_datetime(valid["날짜"].astype(str), format="%Y%m%d")
assert valid.groupby("date").size().eq(24).all()
by_date = valid.groupby("date", sort=True)
matrix = np.vstack([group[SLOTS].to_numpy().ravel() for _, group in by_date])
dates = pd.Index(sorted(valid.date.unique()))
production = by_date["생산량"].sum().reindex(dates)
profile = pd.read_csv(PROFILES, encoding="utf-8-sig", dtype={"date": str})
profile["date"] = pd.to_datetime(profile.date)
profile = profile.set_index("date").reindex(dates)
assert len(matrix) == len(profile) == 241
assert np.unique(matrix, axis=0).shape[0] == 126
assert np.array_equal(production.to_numpy(), profile.production_total.to_numpy())
profile_ids = profile.profile_id.to_numpy()
for profile_id in np.unique(profile_ids):
    subset = matrix[profile_ids == profile_id]
    assert np.array_equal(subset, np.repeat(subset[:1], len(subset), axis=0))
daily = pd.DataFrame({"profile_id": profile_ids, "production_total": production.to_numpy()}, index=dates)
repeated = daily.groupby("profile_id").filter(lambda part: len(part) > 1)
groups = repeated.groupby("profile_id")
different = groups.production_total.nunique().gt(1)
mixed = groups.production_total.agg(lambda values: values.eq(0).any() and values.gt(0).any())
fact = json.loads((TABLES / "09.27_027_facts.json").read_text(encoding="utf-8"))["summary"]
assert (len(different), len(repeated), int(different.sum()), int(mixed.sum())) == (
    fact["repeated_profiles"], fact["days_in_repeated_profiles"],
    fact["repeated_profiles_with_different_production"],
    fact["repeated_profiles_with_zero_and_positive"])
assert int(groups.size().loc[different].sum()) == fact["days_in_different_production_profiles"]
assert int(groups.size().loc[mixed].sum()) == fact["days_in_zero_and_positive_profiles"]

# Rebuild 029's unweighted calendar-only rates from source observations.
ordered = valid.assign(timestamp=valid.date + pd.to_timedelta(valid["시간"], unit="h"))
ordered = ordered.sort_values("timestamp").set_index("timestamp")
peak = ordered[SLOTS].max(axis=1)
previous = peak.reindex(peak.index - pd.Timedelta(hours=1)).to_numpy()
independent = pd.DataFrame({"hour": ordered.index.hour,
                            "weekend": (ordered.index.dayofweek >= 5).astype(int),
                            "period": np.where(ordered.index < pd.Timestamp("2021-07-01"), "Jan-Jun", "Jul-Aug"),
                            "high": peak.to_numpy() >= 182,
                            "eligible": np.isfinite(previous) & (previous < 182)}, index=ordered.index)
independent["onset"] = independent.high & independent.eligible
prior_004 = pd.read_csv(TABLES / "09.27_004_records.csv", encoding="utf-8-sig")
assert np.array_equal(independent.high.to_numpy(), prior_004.high.to_numpy())
assert np.array_equal(independent.eligible.to_numpy(), prior_004.eligible.to_numpy())
assert np.array_equal(independent.onset.to_numpy(), prior_004.onset.to_numpy())
calendar_facts = json.loads((TABLES / "09.27_029_facts.json").read_text(encoding="utf-8"))["standardization"]
for metric in ("high", "onset"):
    subset = independent if metric == "high" else independent.loc[independent.eligible]
    cells = subset.groupby(["hour", "weekend", "period"]).agg(n=(metric, "size"), events=(metric, "sum"))
    ref = cells.n.groupby(level=[0, 1]).sum()
    ref = ref / ref.sum()
    record = next(r for r in calendar_facts if r["metric"] == metric and not r["profile_weighted"])
    for period, tag in (("Jan-Jun", "earlier"), ("Jul-Aug", "later")):
        period_cells = cells.xs(period, level="period")
        assert int(period_cells.n.sum()) == record[f"{tag}_n"]
        assert int(period_cells.events.sum()) == record[f"{tag}_events"]
        standard = float((ref * period_cells.events / period_cells.n).sum())
        assert np.isclose(standard, record[f"{tag}_standardized_rate"], atol=1e-12)
result = {"status": "passed", "reports": len(reports), "numbered_reports": sorted(numbers),
          "conditional_unexecuted_numbers": [17, 18, 19], "local_links_checked": link_count,
          "missing_links": missing_links, "primary_hash_checks": hash_checks,
          "prior_verification_reports_checked": [10, 16, 23, 26],
          "independent_cluster_context": fact,
          "independent_calendar_metrics": ["high", "onset"],
          "source_sha256": SOURCE_HASH, "profile_sha256": PROFILE_HASH}
(TABLES / "09.27_028_audit_verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
print(json.dumps({"status": "passed", "reports": len(reports), "links": link_count,
                  "primary_hashes": len(hash_checks), "repeated_profiles": len(different),
                  "different_production_profiles": int(different.sum())}))
