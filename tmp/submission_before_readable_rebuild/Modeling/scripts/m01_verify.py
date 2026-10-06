"""Independent standard-library verification of M01 saved split artifacts."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Modeling/tables/m01"


def rows(name):
    with (OUT / f"{name}.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    c = json.loads((ROOT / "Modeling/config/m01_contract.json").read_text(encoding="utf-8"))
    source = ROOT / c["source"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == c["source_sha256"]
    source_days = defaultdict(dict)
    observed, bad = {}, []
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            date, hour = int(raw["날짜"]), int(raw["시간"])
            if not c["period"][0] <= date <= c["period"][1]:
                continue
            if not 0 <= hour <= 23:
                bad.append((date, hour))
                continue
            ts = datetime.strptime(str(date), "%Y%m%d") + timedelta(hours=hour)
            slots = [float(raw[key]) for key in ("15분", "30분", "45분", "60분")]
            observed[ts] = slots
            source_days[ts.date()][hour] = slots
    profiles = {}
    for day, hours in source_days.items():
        assert set(hours) == set(range(24))
        profiles[day] = hashlib.sha256(",".join(format(value, ".17g") for hour in range(24) for value in hours[hour]).encode()).hexdigest()
    assert sorted((int(r["날짜"]), int(r["시간"])) for r in rows("excluded_rows")) == sorted(bad)
    hourly = {datetime.fromisoformat(r["timestamp"]): r for r in rows("hourly_frame")}
    assert set(hourly) == set(observed)
    for ts, r in hourly.items():
        assert float(r["target_maximum"]) == max(observed[ts])
        assert float(r["target_mean"]) == sum(observed[ts]) / 4
        for lag in c["lags_hours"]:
            assert (r[f"available_lag{lag}"] == "True") == (ts - timedelta(hours=lag) in observed)
    split_counts = {(r["split"], r["pool"]): r for r in rows("split_counts")}
    overlap = {(r["split"], r["pool"]): r for r in rows("profile_overlap")}
    diagnostics = defaultdict(list)
    for r in rows("evaluation_diagnostics"):
        diagnostics[r["split"], r["pool"]].append(r)
    checks = []
    for split in c["splits"]:
        end = datetime.fromisoformat(split["train_end"])
        start, stop = (datetime.fromisoformat(split[key]) for key in ("eval_start", "eval_end"))
        all_eval = {ts for ts in observed if start <= ts <= stop}
        for pool, lags in c["pools"].items():
            eligible = {ts for ts in observed if all(ts - timedelta(hours=lag) in observed for lag in lags)}
            train = {ts for ts in eligible if ts <= end}
            evaluation = {ts for ts in eligible if start <= ts <= stop}
            assert max(train) < min(evaluation)
            count = split_counts[split["name"], pool]
            assert int(count["train_hours"]) == len(train)
            assert int(count["eval_hours"]) == len(evaluation)
            assert int(count["eval_excluded_hours"]) == len(all_eval - evaluation)
            diag = diagnostics[split["name"], pool]
            assert {datetime.fromisoformat(r["timestamp"]) for r in diag} == evaluation
            assert len(diag) == len(evaluation)
            train_profiles = {profiles[ts.date()] for ts in train}
            eval_days = {ts.date() for ts in evaluation}
            overlaps = {day for day in eval_days if profiles[day] in train_profiles}
            record = overlap[split["name"], pool]
            assert int(record["overlap_days"]) == len(overlaps)
            assert int(record["overlap_hours"]) == sum(ts.date() in overlaps for ts in evaluation)
            complete_days = {day for day in eval_days if sum(ts.date() == day for ts in evaluation) == 24}
            assert int(count["complete_eval_days"]) == len(complete_days)
            assert int(count["incomplete_eval_days"]) == len({ts.date() for ts in all_eval} - complete_days)
            profile_counts = Counter(profiles[day] for day in eval_days)
            weight_by_day = defaultdict(float)
            for r in diag:
                ts = datetime.fromisoformat(r["timestamp"])
                assert r["diag_target_profile"] == profiles[ts.date()]
                assert (r["train_profile_overlap"] == "True") == (ts.date() in overlaps)
                assert math.isclose(float(r["profile_weight"]), 1 / profile_counts[profiles[ts.date()]], abs_tol=1e-14)
                daily_max = max(max(source_days[ts.date()][hour]) for hour in range(24))
                ties = sum(max(source_days[ts.date()][hour]) == daily_max for hour in range(24))
                expected = 1 / ties if ts.date() in complete_days and max(observed[ts]) == daily_max else 0
                assert math.isclose(float(r["daily_maximum_weight"]), expected, abs_tol=1e-14)
                weight_by_day[ts.date()] += float(r["daily_maximum_weight"])
            for day, weight in weight_by_day.items():
                assert math.isclose(weight, 1 if day in complete_days else 0, abs_tol=1e-12)
            checks.append({"split": split["name"], "pool": pool, "train_hours": len(train), "eval_hours": len(evaluation), "complete_days": len(complete_days)})
    original = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((ROOT / "Modeling/config/m01_contract.json").read_bytes()).hexdigest() == original["contract_sha256"]
    assert hashlib.sha256((ROOT / "Modeling/scripts/m01_prepare.py").read_bytes()).hexdigest() == original["script_sha256"]
    for name, digest in original["outputs_sha256"].items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest
    result = {"status": "passed", "independent_standard_library_verification": True, "hourly_rows": len(hourly),
              "split_and_daily_weight_checks": checks, "source_and_output_hashes_match": True, "model_fitting_performed": False,
              "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "independent_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "hourly_rows": len(hourly), "split_pools": len(checks), "model_fitting_performed": False}))


if __name__ == "__main__":
    main()
