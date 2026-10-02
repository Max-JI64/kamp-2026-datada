"""Independent CSV/source checks for A045, without the aggregation implementation."""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
import sysconfig
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from urllib.parse import unquote

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
TABLES = BASE / "tables"
P = "09.28_045"
TOL = 1e-8
SLOTS = ("15분", "30분", "45분", "60분")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def near(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0, abs_tol=TOL)


def check(a: float | str, b: float) -> None:
    assert near(float(a), b), (a, b)


def iso(value: str) -> str:
    return datetime.strptime(value, "%Y%m%d").strftime("%Y-%m-%d")


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def main() -> None:
    assert sys.version_info[:2] == (3, 13) and not sysconfig.get_config_var("Py_GIL_DISABLED")
    config = json.loads((TABLES / f"{P}_frozen.json").read_text(encoding="utf-8"))
    facts = json.loads((TABLES / f"{P}_facts.json").read_text(encoding="utf-8"))
    for relative, expected in config["input_sha256"].items():
        assert sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    for name, expected in facts["output_sha256"].items():
        assert sha256((TABLES / name).read_bytes()).hexdigest() == expected
    assert facts["frozen_sha256"] == sha256((TABLES / f"{P}_frozen.json").read_bytes()).hexdigest()
    source = read(ROOT / "data/origin/okm_augumented_2021.csv")
    by_day, errors = defaultdict(list), defaultdict(list)
    for row in source:
        if not 20210101 <= int(row["날짜"]) <= 20210831:
            continue
        date, hour = iso(row["날짜"]), int(row["시간"])
        cells = [float(row[column]) for column in SLOTS]
        (by_day if 0 <= hour <= 23 else errors)[date].append((hour, cells))
    assert len(by_day) == 241 and sum(map(len, by_day.values())) == 5784
    assert set(errors) == {"2021-07-13", "2021-07-15"} and sum(map(len, errors.values())) == 48
    assert not set(errors) & set(by_day)
    vectors, original, means, zeros = {}, {}, {}, {}
    for date, rows in by_day.items():
        assert sorted(hour for hour, _ in rows) == list(range(24))
        values = tuple(v for _, cells in sorted(rows) for v in cells)
        assert len(values) == 96 and all(math.isfinite(v) and v >= 0 for v in values)
        vectors[date], original[date], means[date], zeros[date] = values, max(values), statistics.mean(values), values.count(0)
    high = {date for date in by_day if original[date] >= 182}
    assert len(high) == 105 and sum(date < "2021-07-01" for date in high) == 69
    daily = {row["date"]: row for row in read(TABLES / "09.28_044_daily_concentration.csv")}
    assert set(daily) == set(by_day)
    profile_vectors, vector_profiles = {}, {}
    for date, row in daily.items():
        check(row["max_value"], original[date])
        check(row["sum_values"], sum(vectors[date]))
        check(row["mean_value"], means[date])
        check(row["mean_max_ratio"], means[date] / original[date])
        assert int(row["zero_cells"]) == zeros[date]
        profile = row["profile"]
        assert profile_vectors.setdefault(profile, vectors[date]) == vectors[date]
        assert vector_profiles.setdefault(vectors[date], profile) == profile
    assert len(profile_vectors) == 126
    eda = {iso(row["date"]): row for row in read(ROOT / "EDA/jsw/tables/09.28_024_daily_concentration.csv")}
    profiles = {iso(row["date"]): row["profile_id"] for row in read(ROOT / "EDA/jsw/tables/09.23_011_daily_profile_summary.csv")}
    assert set(eda) == set(by_day)
    for date in by_day:
        check(eda[date]["peak"], original[date])
        assert eda[date]["profile_id"] == daily[date]["profile"] == profiles[date]

    saved = read(TABLES / "09.28_044_redistribution.csv")
    sims = {(row["date"], float(row["budget_fraction"]), row["receive_zero"] == "True"): row for row in saved}
    assert len(saved) == len(sims) == 1446
    expected_keys = {(date, fraction, policy) for date in by_day for fraction in (0, .05, .1) for policy in (False, True)}
    assert set(sims) == expected_keys
    for (date, fraction, policy), row in sims.items():
        assert row["profile"] == daily[date]["profile"]
        assert row["period"] == ("Jan-Jun" if date < "2021-07-01" else "Jul-Aug")
        cap = float(row["optimal_max"])
        check(row["original_max"], original[date])
        check(row["max_reduction"], original[date] - cap)
        check(row["reduction_fraction"], (original[date] - cap) / original[date])
        check(row["budget_value_sum"], fraction * sum(vectors[date]))
        # Verify the saved bound against source removals and available receivers;
        # no new cap search or adjusted time schedule is constructed.
        removed = sum(max(v - cap, 0) for v in vectors[date])
        capacity = sum(max(cap - v, 0) for v in vectors[date] if policy or v > 0)
        assert abs(removed - float(row["moved_value_sum"])) <= 1e-7
        assert removed <= float(row["budget_value_sum"]) + 1e-7 and removed <= capacity + 1e-7
        if fraction == 0:
            check(cap, original[date])

    summaries = read(TABLES / f"{P}_group_summary.csv")
    assert len(summaries) == 108
    summary_keys = set()
    for row in summaries:
        period, group, weighting = row["period"], row["group"], row["weighting"]
        fraction, policy = float(row["budget_fraction"]), row["receive_zero"] == "True"
        key = (period, group, weighting, fraction, policy)
        assert key not in summary_keys
        summary_keys.add(key)
        candidates = sorted(date for date in by_day if period == "Jan-Aug" or daily[date]["period"] == period)
        candidates = [date for date in candidates if group == "all" or (date in high) == (group == "high_ge_182")]
        distinct = {}
        for date in candidates:
            distinct.setdefault(daily[date]["profile"], date)
        selected = candidates if weighting == "observed_days" else list(distinct.values())
        assert int(row["days"]) == len(candidates) and int(row["unique_profiles"]) == len(distinct)
        assert int(row["days_or_profiles"]) == len(selected)
        check(row["median_original_max"], statistics.median(original[date] for date in selected))
        reductions = [original[date] - float(sims[date, fraction, policy]["optimal_max"]) for date in selected]
        ratios = [reduction / original[date] for date, reduction in zip(selected, reductions)]
        check(row["mean_reduction"], statistics.mean(reductions))
        check(row["median_mean_max_ratio"], statistics.median(means[date] / original[date] for date in selected))
        for label, values in (("reduction", reductions), ("reduction_fraction", ratios)):
            for prefix, q in (("q25", .25), ("median", .5), ("q75", .75)):
                check(row[f"{prefix}_{label}"], quantile(values, q))
        assert int(row["days_or_profiles_with_reduction"]) == sum(value > 1e-7 for value in reductions)
        assert int(row["days_or_profiles_with_zero"]) == sum(zeros[date] > 0 for date in selected)
        assert int(row["undefined_reduction_fraction"]) == 0
    assert summary_keys == {(p, g, w, b, z) for p in ("Jan-Aug", "Jan-Jun", "Jul-Aug") for g in
                            ("all", "high_ge_182", "below_182") for w in ("observed_days", "unique_profiles")
                            for b in (0, .05, .1) for z in (False, True)}
    legacy = read(TABLES / "09.28_044_scenario_summary.csv")
    for row in legacy:
        new = next(r for r in summaries if r["group"] == "all" and all(r[c] == row[c] for c in
                   ("period", "weighting", "budget_fraction", "receive_zero")))
        for column in row.keys() - {"period", "weighting", "budget_fraction", "receive_zero"}:
            check(new[column], float(row[column]))

    months = read(TABLES / f"{P}_monthly_max.csv")
    actual_ties = read(TABLES / f"{P}_monthly_max_ties.csv")
    expected_ties, monthly_keys = set(), set()
    month_lookup = {}
    for row in months:
        month, fraction, policy = row["month"], float(row["budget_fraction"]), row["receive_zero"] == "True"
        key = (month, fraction, policy)
        assert key not in monthly_keys
        monthly_keys.add(key)
        dates = sorted(date for date in by_day if date.startswith(month))
        before_max = max(original[date] for date in dates)
        after_max = max(float(sims[date, fraction, policy]["optimal_max"]) for date in dates)
        before = {date for date in dates if near(original[date], before_max)}
        after = {date for date in dates if near(float(sims[date, fraction, policy]["optimal_max"]), after_max)}
        assert int(row["complete_days"]) == len(dates)
        assert int(row["unique_profiles"]) == len({daily[date]["profile"] for date in dates})
        check(row["original_month_max"], before_max)
        check(row["adjusted_month_max"], after_max)
        check(row["month_reduction"], before_max - after_max)
        check(row["month_reduction_fraction"], (before_max - after_max) / before_max)
        assert row["original_max_dates"] == "|".join(sorted(before)) and row["adjusted_max_dates"] == "|".join(sorted(after))
        assert int(row["original_tied_dates"]) == len(before) and int(row["adjusted_tied_dates"]) == len(after)
        assert row["date_set_relation"] == ("unchanged" if before == after else "partial_overlap" if before & after else "replaced")
        for column, values in (("retained_dates", before & after), ("removed_dates", before - after), ("added_dates", after - before)):
            assert row[column] == "|".join(sorted(values))
        tracked = max(float(sims[date, fraction, policy]["optimal_max"]) for date in before)
        check(row["original_date_only_adjusted_max"], tracked)
        check(row["missed_max_by_original_date_only"], after_max - tracked)
        if fraction == 0:
            assert before == after and before_max == after_max
        for phase, ties, maximum in (("original", before, before_max), ("adjusted", after, after_max)):
            for date in ties:
                expected_ties.add((month, fraction, policy, phase, date))
        month_lookup[key] = (after_max, after)
    assert monthly_keys == {(f"2021-{m:02d}", b, z) for m in range(1, 9) for b in (0, .05, .1) for z in (False, True)}
    assert len(actual_ties) == len(expected_ties)
    assert {(r["month"], float(r["budget_fraction"]), r["receive_zero"] == "True", r["phase"], r["date"])
            for r in actual_ties} == expected_ties
    for row in actual_ties:
        date = row["date"]
        assert row["profile"] == daily[date]["profile"]
        check(row["daily_max"], original[date] if row["phase"] == "original" else
              float(sims[date, float(row["budget_fraction"]), row["receive_zero"] == "True"]["optimal_max"]))
        check(row["daily_max"], float(row["month_max"]))

    error_peaks = {date: max(v for _, cells in rows for v in cells) for date, rows in errors.items()}
    assert error_peaks == {"2021-07-13": 190, "2021-07-15": 202}
    sensitivity = read(TABLES / f"{P}_excluded_date_sensitivity.csv")
    assert len(sensitivity) == 6
    assert {(float(r["budget_fraction"]), r["receive_zero"] == "True") for r in sensitivity} == {
        (b, z) for b in (0, .05, .1) for z in (False, True)}
    for row in sensitivity:
        normal, normal_dates = month_lookup["2021-07", float(row["budget_fraction"]), row["receive_zero"] == "True"]
        candidate = max(error_peaks.values())
        combined = max(normal, candidate)
        combined_dates = (normal_dates if near(normal, combined) else set()) | {
            date for date, value in error_peaks.items() if near(value, combined)}
        check(row["adjusted_normal_month_max"], normal)
        check(row["auxiliary_max_errors_unchanged"], combined)
        check(row["auxiliary_reduction_from_222"], 222 - combined)
        assert row["auxiliary_max_dates"] == "|".join(sorted(combined_dates))
        assert (row["error_candidate_exceeds_adjusted_normal"] == "True") == (candidate > normal + TOL)
    policies = read(TABLES / f"{P}_zero_policy_difference.csv")
    assert len(policies) == 723
    assert {(r["date"], float(r["budget_fraction"])) for r in policies} == {(d, b) for d in by_day for b in (0, .05, .1)}
    for row in policies:
        date, fraction = row["date"], float(row["budget_fraction"])
        allowed, forbidden = (float(sims[date, fraction, policy]["optimal_max"]) for policy in (True, False))
        check(row["allow_zero_max"], allowed)
        check(row["forbid_zero_max"], forbidden)
        check(row["forbid_minus_allow"], forbidden - allowed)

    links = 0
    for path in (BASE / "reports/09.28_045_peak_target_reaggregation.md", BASE / "reports/09.29_047_peak_management_analysis_plan.md", BASE / "README.md"):
        text = path.read_text(encoding="utf-8")
        assert "\ufffd" not in text
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            destination = unquote(target.split("#")[0].strip("<>"))
            assert (path.parent / destination).exists(), (path, target)
            links += 1
    result = {"verified_at": datetime.now().astimezone().isoformat(timespec="minutes"),
              "python": sys.version.split()[0], "executable": sys.executable, "free_threaded": False,
              "source_days": 241, "source_normal_hours": 5784, "source_error_rows": 48,
              "high_days": 105, "exact_profiles": 126, "saved_scenarios_checked": 1446,
              "group_summaries_checked": 108, "legacy_summaries_reproduced": 24,
              "monthly_scenarios_checked": 48, "calendar_tie_rows_checked": len(actual_ties),
              "excluded_date_scenarios_checked": 6, "zero_policy_pairs_checked": 723,
              "document_local_links_checked": links, "status": "passed"}
    (TABLES / f"{P}_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
