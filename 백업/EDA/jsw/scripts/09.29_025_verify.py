"""Independent standard-library verification of E025 saved descriptive tables."""

from bisect import bisect_left
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote


EDA = Path(__file__).resolve().parent.parent
ROOT = EDA.parent.parent
TABLE = EDA / "tables"
PREFIX = "09.29_025"
SLOTS = ("15분", "30분", "45분", "60분")


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position-low)


def close(actual, expected):
    assert math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-10), (actual, expected)


def weight(rows, name):
    if name == "date_hour":
        return rows
    unique = {}
    for row in rows:
        unique.setdefault((row["profile_id"], row["hour"]), row)
    return list(unique.values())


def select(rows, period, month, bin_number=0):
    return [r for r in rows if r["period"] == period and (month == "all" or r["month"] == int(month))
            and (not bin_number or r["A_bin"] == bin_number)]


def main():
    facts = json.loads((TABLE / f"{PREFIX}_facts.json").read_text(encoding="utf-8"))
    for name, digest in facts["input_sha256"].items():
        assert sha256((ROOT / facts["input_paths"][name]).read_bytes()).hexdigest() == digest, name
    profiles = {int(r["date"]): r["profile_id"] for r in read(ROOT / facts["input_paths"]["profiles"])}
    rows = []
    raw = read(ROOT / facts["input_paths"]["raw"])
    for row_number, raw_row in enumerate(raw, 1):
        date, hour = int(raw_row["날짜"]), int(raw_row["시간"])
        if date >= 20210901 or not 0 <= hour <= 23:
            continue
        values = tuple(int(raw_row[s]) for s in SLOTS)
        A = sum(values) / 4
        P = max(values)
        M = int(raw_row["평균"])
        assert M == math.floor(A+0.5) and min(values) <= A <= P
        key = datetime.strptime(str(date), "%Y%m%d") + timedelta(hours=hour)
        rows.append(dict(source_data_row=row_number, date=date, hour=hour, record_key=key,
                         month=key.month, period="Jan-Jun_design" if key.month <= 6 else "Jul-Aug_exploratory",
                         profile_id=profiles[date], values=values, M=M, A=A, P=P,
                         D=P-A, R=P-min(values), S=values[-1]-values[0], rounding_delta=M-A))
    rows.sort(key=lambda r: r["record_key"])
    assert len(rows) == 5784 and len({r["date"] for r in rows}) == 241
    assert all(n == 24 for n in Counter(r["date"] for r in rows).values())
    assert sum(r["P"] == 0 for r in rows) == 17
    by_key = {r["record_key"]: r for r in rows}
    assert len(by_key) == len(rows)
    for row in rows:
        key = row["record_key"]
        row["eligible"] = all(key - timedelta(hours=i) in by_key for i in (*range(1, 25), 168))
    assert sum(r["eligible"] for r in rows) == 5520
    manifest = read(ROOT / facts["input_paths"]["manifest"])
    for item in manifest:
        key = datetime.fromisoformat(item["record_key"])
        assert by_key[key]["eligible"] == (item["complete_past_24h"] == "True" and item["exact_lag_168h"] == "True")
    design = [r for r in rows if r["month"] <= 6]
    bounds = sorted({quantile([r["A"] for r in design], q) for q in (0, .2, .4, .6, .8, 1)})
    assert bounds == facts["A_fit_quantiles"]
    bins = read(TABLE / f"{PREFIX}_bins.csv")
    assert len(bins) == len(bounds)-1
    for number, item in enumerate(bins, 1):
        assert int(item["A_bin"]) == number and int(item["fit_hours"]) == len(design)
        assert float(item["lower_exclusive"]) == (-math.inf if number == 1 else bounds[number-1])
        assert float(item["upper_inclusive"]) == (math.inf if number == len(bins) else bounds[number])
    for row in rows:
        row["A_bin"] = bisect_left(bounds[1:-1], row["A"]) + 1
    summaries = read(TABLE / f"{PREFIX}_gap_summary.csv")
    for item in summaries:
        cell = select(rows, item["period"], item["month"], int(item["A_bin"]))
        weighted = weight(cell, item["weighting"])
        assert len(cell) == int(item["normal_hours"]) and len(weighted) == int(item["weighted_units"])
        days = len({r["date"] for r in cell})
        assert days == int(item["days"]) and len({r["profile_id"] for r in cell}) == int(item["profiles"])
        assert (item["small_cell"] == "True") == (len(weighted) < 30 or days < 5)
        for variable in ("A", "P", "D"):
            for q in (.25, .5, .75, .9, .95):
                close(item[f"{variable}_q{int(q*100):02d}"], quantile([r[variable] for r in weighted], q))
    for item in read(TABLE / f"{PREFIX}_coverage.csv"):
        part = select(rows, item["period"], item["month"])
        expected = {"normal_hours": len(part), "days": len({r["date"] for r in part}),
                    "profiles": len({r["profile_id"] for r in part}), "M001_eligible_hours": sum(r["eligible"] for r in part),
                    "M001_ineligible_hours": sum(not r["eligible"] for r in part), "zero_hours_preserved": sum(r["P"] == 0 for r in part),
                    "below_design_min_hours": sum(r["A"] < bounds[0] for r in part),
                    "above_design_max_hours": sum(r["A"] > bounds[-1] for r in part)}
        assert all(int(item[k]) == v for k, v in expected.items())
        total = [s for s in summaries if s["period"] == item["period"] and s["month"] == item["month"] and s["weighting"] == "date_hour" and int(s["A_bin"]) > 0]
        assert sum(int(s["normal_hours"]) for s in total) == len(part)
    target_keys = defaultdict(set)
    threshold_sets = defaultdict(set)
    for name in ("development", "review"):
        for item in read(ROOT / facts["input_paths"][name]):
            target_keys[item["stage"]].add(datetime.fromisoformat(item["record_key"]))
            threshold_sets[item["stage"]].add(float(item["high_threshold_from_training"]))
    assert dict(threshold_sets) == {"May": {183.0}, "Jun": {181.0}, "Jul-Aug": {182.0}}
    sets = {"legacy_period_new_A": rows, "Jan-Jun_EDA_reference": design,
            "Jul-Aug_EDA_reference": [r for r in rows if r["month"] >= 7]}
    for stage, cutoff in (("May", "2021-05-01"), ("Jun", "2021-06-01"), ("Jul-Aug", "2021-07-01")):
        train = [r for r in rows if r["eligible"] and r["record_key"] < datetime.fromisoformat(cutoff)]
        # Validate reused training quantiles without changing their saved definitions.
        close(next(iter(threshold_sets[stage])), quantile([r["P"] for r in train], .95))
        sets[f"{stage}_M001_training"] = train
        sets[f"{stage}_M001_target"] = [by_key[k] for k in sorted(target_keys[stage])]
    visibility = read(TABLE / f"{PREFIX}_visibility.csv")
    for item in visibility:
        part = sets[item["scope"]]
        weighted = weight(part, item["weighting"])
        u = float(item["threshold"])
        high = [r for r in weighted if r["P"] >= u]
        hidden = [r for r in high if r[item["mean_definition"]] < u]
        assert int(item["normal_hours"]) == len(part) and int(item["weighted_units"]) == len(weighted)
        assert int(item["hidden_units"]) == len(hidden) and int(item["peak_high_units"]) == len(high)
        close(item["hidden_rate"], len(hidden)/len(high))
        assert item["denominator_condition"] == "P>=u"
        assert item["numerator_condition"] == f'P>=u and {item["mean_definition"]}<u'
    legacy = [r for r in rows if r["P"] >= 187]
    assert len(legacy) == 287 and sum(r["M"] < 187 for r in legacy) == 194 and sum(r["A"] < 187 for r in legacy) == 202
    legacy_table = read(TABLE / f"{PREFIX}_legacy_visibility.csv")
    old_table = read(ROOT / facts["input_paths"]["legacy"])
    assert len(legacy_table) == len(old_table) == 7
    for item, old in zip(legacy_table, old_table):
        cell = [r for r in legacy if sum(v >= 187 for v in r["values"]) == int(item["high_quarters"])
                and (r["M"] < 187) == (item["mean_below_same_number"] == "True")]
        assert len(cell) == int(item["hours"]) == int(old["hours"])
        assert len({r["date"] for r in cell}) == int(item["days"]) == int(old["days"])
        close(item["median_max_mean_gap"], quantile([r["P"]-r["M"] for r in cell], .5))
        close(old["median_max_mean_gap"], float(item["median_max_mean_gap"]))
    for item in read(TABLE / f"{PREFIX}_rounding.csv"):
        part = weight(select(rows, item["period"], "all"), item["weighting"])
        assert int(item["denominator_units"]) == len(part)
        assert int(item["weighted_units"]) == sum(r["rounding_delta"] == float(item["M_minus_A"]) for r in part)
    for item in read(TABLE / f"{PREFIX}_exact_A_spread.csv"):
        part = weight(select(rows, item["period"], "all"), item["weighting"])
        groups = defaultdict(list)
        for row in part:
            groups[row["A"]].append(row["D"])
        repeated = [g for g in groups.values() if len(g) >= 2]
        varying = [g for g in repeated if len(set(g)) > 1]
        expected = {"A_values": len(groups), "repeated_A_values": len(repeated), "varying_D_A_values": len(varying),
                    "repeated_A_units": sum(map(len, repeated)), "varying_D_A_units": sum(map(len, varying))}
        assert all(int(item[k]) == v for k, v in expected.items())
        spans = [max(g)-min(g) for g in varying]
        close(item["D_span_q50_over_varying_A"], quantile(spans, .5))
        close(item["D_span_q95_over_varying_A"], quantile(spans, .95))
    examples = read(TABLE / f"{PREFIX}_examples.csv")
    for item in examples:
        pool = select(rows, item["period"], "all", int(item["A_bin"]))
        if item["selection"] == "exact_A_nearest_bin_median":
            groups = defaultdict(set)
            for row in pool:
                groups[row["A"]].add(row["D"])
            median = quantile([r["A"] for r in pool], .5)
            a = min((a for a, ds in groups.items() if len(ds) > 1), key=lambda a: (abs(a-median), a))
            pool = [r for r in pool if r["A"] == a]
        target = quantile([r["D"] for r in pool], float(item["D_quantile"]))
        chosen = min(pool, key=lambda r: (abs(r["D"]-target), r["record_key"]))
        assert chosen["source_data_row"] == int(item["source_data_row"])
        assert tuple(int(item[s]) for s in SLOTS) == chosen["values"]
        for variable in ("M", "A", "P", "D", "R", "S"):
            close(item[variable], chosen[variable])
        close(item["quantile_value"], target)
    catalog = read(ROOT / facts["input_paths"]["catalog"])
    names = {r["feature"] for r in catalog}
    assert len(catalog) == 26 and "last_slot_lag1" in names and "range_lag1" in names
    assert not any("first_slot" in n or n.startswith(("D_", "S_")) for n in names)
    assert all(int(r["latest_offset_hours"]) < 0 for r in catalog if r["group"] == "past_power")
    # Algebraic counterexample, not an observed row: order cannot be recovered from aggregates.
    x, y = (80, 90, 100, 100), (90, 80, 100, 100)
    signature = lambda v: (sum(v)/4, max(v), max(v)-min(v), v[-1])
    assert signature(x) == signature(y) and x[-1]-x[0] != y[-1]-y[0]
    overlaps = read(TABLE / f"{PREFIX}_feature_overlap.csv")
    assert len(overlaps) == 9
    overlap = {r["candidate"]: r for r in overlaps}
    assert overlap["past_S"]["classification"] == "not algebraically determined"
    assert overlap["past_D_M"]["classification"] == "fully derived"
    assert overlap["past_P"]["classification"] == "existing"
    link_count = 0
    validation_path = TABLE / f"{PREFIX}_validation.json"
    self_links = []
    for document in (EDA / "reports/09.29_025_peak_gap_eda_plan.md", EDA / "README.md"):
        text = document.read_text(encoding="utf-8")
        assert "\ufffd" not in text and "—" not in text and "–" not in text
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            if target.startswith(("https://", "http://", "#")):
                continue
            path = (document.parent / unquote(target.split("#")[0])).resolve()
            if path == validation_path.resolve():
                self_links.append(path)  # This run creates its own evidence file below.
            else:
                assert path.exists(), (document, target)
            link_count += 1
    result = {"time": datetime.now().astimezone().isoformat(timespec="seconds"), "status": "passed",
              "method": "independent stdlib CSV, exact integer quarter sums, manual linear quantiles and exact timestamp lookups",
              "raw_rows": len(raw), "normal_hours": len(rows), "summary_rows_checked": len(summaries),
              "visibility_rows_checked": len(visibility), "examples_checked": len(examples), "document_links_checked": link_count,
              "input_hashes_preserved": len(facts["input_sha256"]), "S_algebraic_counterexample": [x, y],
              "model_training_or_scoring": False}
    validation_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    assert all(path.exists() for path in self_links)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
