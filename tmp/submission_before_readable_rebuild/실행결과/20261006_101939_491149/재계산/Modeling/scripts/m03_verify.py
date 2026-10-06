"""Independent standard-library checks of M03 inputs, metrics and paired deltas."""
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-9), (a, b)


def select(g, kind, value):
    if kind in ("all", "profile_reweighted"):
        return g
    if kind == "daily_maximum":
        return [r for r in g if float(r["daily_maximum_weight"]) > 0]
    return [r for r in g if r[kind] == value]


def weights(g, kind):
    column = {"daily_maximum": "daily_maximum_weight", "profile_reweighted": "profile_weight"}.get(kind)
    return [float(r[column]) if column else 1.0 for r in g]


def verify_metric(row, g, kind="all"):
    assert len(g) == int(row["hours"])
    w = weights(g, kind)
    total = math.fsum(w)
    close(row["weight_sum"], total)
    errors = [float(r["prediction"]) - float(r["actual"]) for r in g]
    values = {"MAE": [abs(e) for e in errors], "bias": errors,
              "mean_under": [max(-e, 0) for e in errors], "mean_over": [max(e, 0) for e in errors],
              "under_fraction": [float(e < 0) for e in errors], "over_fraction": [float(e > 0) for e in errors],
              "mean_actual": [float(r["actual"]) for r in g]}
    for name, numbers in values.items():
        close(row[name], math.fsum(x * weight for x, weight in zip(numbers, w)) / total)
    close(row["RMSE"], math.sqrt(math.fsum(e * e * weight for e, weight in zip(errors, w)) / total))
    close(row["absolute_error_sum"], math.fsum(abs(e) for e in errors))
    assert int(row["negative_prediction_hours"]) == sum(float(r["prediction"]) < 0 for r in g)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["ab", "c"], required=True)
    stage = parser.parse_args().stage
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    out = ROOT / f"Modeling/tables/m03/{stage}"
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    cpath = ROOT / f"Modeling/config/m03_{stage}_contract.json"
    c = json.loads(cpath.read_text(encoding="utf-8"))
    m1 = json.loads((ROOT / "Modeling/config/m01_contract.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed" and not run["july_august_evaluated"]
    assert sha(cpath) == run["input_hashes"]["contract"]
    assert sha(ROOT / "Modeling/config/m01_contract.json") == run["input_hashes"]["m01_contract"]
    assert sha(ROOT / m1["source"]) == run["input_hashes"]["source"] == m1["source_sha256"]
    for name, digest in run["outputs_sha256"].items():
        assert sha(out / name) == digest
    for name, digest in run["input_hashes"].items():
        if name not in ("source", "contract", "m01_contract"):
            assert sha(ROOT / name) == digest
    raw = {}
    for r in rows(ROOT / m1["source"]):
        date = int(r["날짜"])
        hour = float(r["시간"])
        if 20210101 <= date <= 20210630 and hour.is_integer() and 0 <= hour <= 23:
            ts = datetime.strptime(str(date), "%Y%m%d") + timedelta(hours=int(hour))
            slots = [float(r[k]) for k in ["15분", "30분", "45분", "60분"]]
            raw[ts] = {"last": slots[-1], "mean": math.fsum(slots) / 4, "maximum": max(slots)}
    frame = {datetime.fromisoformat(r["timestamp"]): r for r in rows(ROOT / "Modeling/tables/m01/hourly_frame.csv")}
    lag_checks = 0
    for ts, r in frame.items():
        if ts >= datetime(2021, 7, 1) or r["eligible_common"] != "True":
            continue
        for lag, names in [(1, ["last"]), (24, ["mean", "maximum"]), (168, ["mean", "maximum"])]:
            previous = ts - timedelta(hours=lag)
            assert previous in raw and previous < ts
            for name in names:
                close(r[f"lag{lag}_{name}"], raw[previous][name])
                lag_checks += 1
    p = rows(out / "predictions.csv")
    assert len(p) == len(c["groups"]) * len(c["targets"]) * 2184
    keyed = {}
    grouped = defaultdict(list)
    daily = defaultdict(list)
    for r in p:
        key = (r["target"], r["group"], r["split"], r["timestamp"])
        assert key not in keyed
        keyed[key] = r
        ts = datetime.fromisoformat(r["timestamp"])
        assert datetime(2021, 4, 1) <= ts < datetime(2021, 7, 1)
        close(r["actual"], raw[ts]["maximum" if r["target"] == "target_maximum" else "mean"])
        e = float(r["prediction"]) - float(r["actual"])
        assert math.isfinite(e)
        for name, value in {"signed_error": e, "absolute_error": abs(e), "under_amount": max(-e, 0), "over_amount": max(e, 0)}.items():
            close(r[name], value)
        grouped[(r["target"], r["group"])].append(r)
        daily[(r["target"], r["group"], r["date"])].append(r)
    expected_keys = {(r["split"], r["timestamp"]) for r in p if r["target"] == "target_maximum" and r["group"] == "A"}
    assert len(expected_keys) == 2184
    for target in c["targets"]:
        for group in c["groups"]:
            g = grouped[(target, group)]
            assert {(r["split"], r["timestamp"]) for r in g} == expected_keys and len(g) == 2184
            assert {s: sum(r["split"] == s for r in g) for s in c["splits"]} == {"dev_apr": 720, "dev_may": 744, "dev_jun": 720}
            date_weights = defaultdict(float)
            for r in g:
                date_weights[r["date"]] += float(r["daily_maximum_weight"])
            assert len(date_weights) == 91
            for date, weight in date_weights.items():
                close(weight, 1)
                day = daily[(target, group, date)]
                assert len(day) == 24
                maximum = max(float(frame[datetime.fromisoformat(r["timestamp"])]["target_maximum"]) for r in day)
                ties = [r for r in day if float(frame[datetime.fromisoformat(r["timestamp"])]["target_maximum"]) == maximum]
                for r in day:
                    close(r["daily_maximum_weight"], 1 / len(ties) if r in ties else 0)
    score_checks = 0
    for table in ["comparison", "condition_errors", "daily_errors"]:
        for r in rows(out / f"{table}.csv"):
            g = grouped[(r["target"], r["group"])]
            if table == "comparison":
                kind = "all" if r["split"] == "pooled_development" else "split"
                g = select(g, kind, r["split"])
            elif table == "condition_errors":
                kind = r["condition"]
                g = select(g, kind, r["value"])
                denominator = math.fsum(float(x["absolute_error"]) for x in grouped[(r["target"], r["group"])])
                close(r["share_total_absolute_error_unweighted"], math.fsum(float(x["absolute_error"]) for x in g) / denominator)
            else:
                kind = "all"
                g = daily[(r["target"], r["group"], r["date"])]
            verify_metric(r, g, kind)
            score_checks += 1
    pairs = rows(out / "paired_predictions.csv")
    pairgroups = defaultdict(list)
    pairdays = defaultdict(list)
    assert len(pairs) == len(c["comparisons"]) * 2 * 2184
    for r in pairs:
        for label in ["before", "after"]:
            ref = keyed[(r["target"], r[label], r["split"], r["timestamp"])]
            close(r[f"prediction_{label}"], ref["prediction"])
            close(r[f"actual_{label}"], ref["actual"])
            for k, source in [("abs", "absolute_error"), ("under", "under_amount"), ("over", "over_amount")]:
                close(r[f"{k}_{label}"], ref[source])
        for k in ["abs", "under", "over"]:
            close(r[f"delta_{k}"], float(r[f"{k}_after"]) - float(r[f"{k}_before"]))
        pairgroups[(r["target"], r["before"], r["after"])].append(r)
        pairdays[(r["target"], r["before"], r["after"], r["date"])].append(r)
    for table in ["paired_deltas", "daily_deltas"]:
        for r in rows(out / f"{table}.csv"):
            if table == "paired_deltas":
                g = select(pairgroups[(r["target"], r["before"], r["after"])], r["condition"], r["value"])
                w = weights(g, r["condition"])
                close(r["weight_sum"], math.fsum(w))
                assert int(r["improved_hours"]) == sum(float(x["delta_abs"]) < -1e-10 for x in g)
                assert int(r["worsened_hours"]) == sum(float(x["delta_abs"]) > 1e-10 for x in g)
            else:
                g = pairdays[(r["target"], r["before"], r["after"], r["date"])]
                w = [1.0] * len(g)
            assert int(r["hours"]) == len(g)
            for k in ["abs", "under", "over"]:
                close(r[f"delta_{k}"], math.fsum(float(x[f"delta_{k}"]) * weight for x, weight in zip(g, w)) / math.fsum(w))
    for r in rows(out / "model_manifest.csv"):
        assert sha(ROOT / r["file"]) == r["sha256"]
        assert r["reload_predictions_checked"] == "True"
    audits = rows(out / "training_audit.csv")
    for r in audits:
        assert datetime.fromisoformat(r["train_end"]) < datetime.fromisoformat(r["eval_start"])
        assert int(r["features"]) == len(run["features"][r["group"]])
    result = {"status": "passed", "verified_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"),
              "stage": stage, "prediction_rows": len(p), "paired_rows": len(pairs), "metric_rows": score_checks,
              "raw_source_lag_checks": lag_checks, "csv_hashes_checked": len(run["outputs_sha256"]),
              "models_checked": len(audits), "july_august_evaluated": False,
              "identical_evaluation_timestamps": len(expected_keys), "daily_maximum_dates": 91,
              "run_sha256": sha(out / "run.json"), "verifier_sha256": sha(Path(__file__))}
    (out / "independent_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
