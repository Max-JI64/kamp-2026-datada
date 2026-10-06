"""Independent raw-time, AP, chronological selection and model checks for M04 rise."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from m03_verify import close, rows, sha, verify_metric

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Modeling/tables/m04_rise"


def boolean(value):
    return value == "True"


def raw_state(record, rounded=None):
    maximum = record["maximum"]
    rounded = record["rounded"] if rounded is None else rounded
    return "zero" if maximum == 0 else "below20" if rounded < 20 else "low" if rounded <= 26 else "above26"


def ap(labels, scores):
    ordered = sorted(zip(scores, labels), reverse=True)
    tp, count, total = 0, 0, sum(labels)
    assert total > 0
    result = 0.
    grouped = defaultdict(list)
    for score, label in ordered:
        grouped[score].append(label)
    for score in sorted(grouped, reverse=True):
        group = grouped[score]
        positives = sum(group)
        tp += positives
        count += len(group)
        result += positives / total * tp / count
    return result


def classification(g, threshold):
    low = [r for r in g if boolean(r["prior_low"])]
    labels = [boolean(r["rise_event"]) for r in low]
    scores = [float(r["probability"]) for r in low]
    alarm = [p >= threshold for p in scores]
    tp = sum(a and y for a, y in zip(alarm, labels))
    fp = sum(a and not y for a, y in zip(alarm, labels))
    fn = sum(not a and y for a, y in zip(alarm, labels))
    tn = sum(not a and not y for a, y in zip(alarm, labels))
    pos = [p for p, y in zip(scores, labels) if y]
    neg = [p for p, y in zip(scores, labels) if not y]
    return {"prior_low_hours": len(low), "positive_hours": sum(labels), "prevalence": sum(labels) / len(low),
        "AP": ap(labels, scores), "AUROC": sum(float(p > n) + .5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg)),
        "Brier": math.fsum((p - y) ** 2 for p, y in zip(scores, labels)) / len(low),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": tp / (tp + fp) if tp + fp else 0.,
        "recall": tp / (tp + fn), "false_positive_fraction": fp / (fp + tn), "threshold": threshold}


def stats(g):
    errors = [float(r["prediction"]) - float(r["actual"]) for r in g]
    rise = [e for e, r in zip(errors, g) if boolean(r["rise_event"])]
    stay = [e for e, r in zip(errors, g) if r["power_transition"] == "low->low"]
    weights = [float(r["daily_maximum_weight"]) for r in g]
    return {"overall_MAE": math.fsum(abs(e) for e in errors) / len(errors),
        "daily_maximum_MAE": math.fsum(abs(e) * w for e, w in zip(errors, weights)) / math.fsum(weights),
        "rise_MAE": math.fsum(abs(e) for e in rise) / len(rise), "rise_under": math.fsum(max(-e, 0) for e in rise) / len(rise),
        "low_stay_MAE": math.fsum(abs(e) for e in stay) / len(stay), "low_stay_over": math.fsum(max(e, 0) for e in stay) / len(stay),
        "prior_low_hours": sum(boolean(r["prior_low"]) for r in g), "rise_hours": len(rise)}


def main():
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    cpath = ROOT / "Modeling/config/m04_rise_contract.json"
    c = json.loads(cpath.read_text(encoding="utf-8"))
    run = json.loads((OUT / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed" and run["no_july_august_evaluation"]
    assert sha(cpath) == run["contract_sha256"] and sha(ROOT / "Modeling/scripts/m04_rise_compare.py") == run["script_sha256"]
    for key in ["inputs_sha256", "outputs_sha256", "helper_sha256"]:
        for name, expected in run[key].items():
            path = ROOT / name if key == "inputs_sha256" else OUT / name if key == "outputs_sha256" else ROOT / "Modeling/scripts" / name
            assert sha(path) == expected, (key, name)
    parent = json.loads((ROOT / "Modeling/config/m01_contract.json").read_text(encoding="utf-8"))
    raw = {}
    for r in rows(ROOT / parent["source"]):
        if not 20210101 <= int(r["날짜"]) <= 20210630 or not 0 <= int(r["시간"]) <= 23:
            continue
        t = datetime.strptime(r["날짜"], "%Y%m%d") + timedelta(hours=int(r["시간"]))
        values = [float(r[name]) for name in ["15분", "30분", "45분", "60분"]]
        assert t not in raw
        raw[t] = {"slots": values, "mean": sum(values) / 4, "maximum": max(values), "last": values[-1],
                  "production": float(r["생산량"]), "rounded": float(r["평균"])}
    eligible = {t for t in raw if all(t - timedelta(hours=lag) in raw for lag in [1,2,24,168])}
    feature_rows = rows(OUT / "feature_frame.csv")
    frame = {datetime.fromisoformat(r["timestamp"]): r for r in feature_rows}
    assert set(frame) == eligible and len(frame) == len(feature_rows)
    feature_checks = 0
    for t, r in frame.items():
        previous, previous2 = raw[t - timedelta(hours=1)], raw[t - timedelta(hours=2)]
        prior = raw_state(previous, math.floor(previous["mean"] + .5))
        event = prior == "low" and raw_state(raw[t]) == "above26"
        assert boolean(r["prior_low"]) == (prior == "low") and boolean(r["rise_event"]) == event
        assert r["power_transition"] == prior + "->" + raw_state(raw[t])
        expected = {"month": t.month, "weekend": int(t.weekday() >= 5), "hour_sin": math.sin(2 * math.pi * t.hour / 24),
            "hour_cos": math.cos(2 * math.pi * t.hour / 24), "lag1_production": previous["production"],
            "lag1_production_zero": int(previous["production"] == 0)}
        expected.update({f"dow_{day}": int(t.weekday() == day) for day in range(7)})
        for lag, record in [(1, previous), (2, previous2)]:
            for field in ["mean", "maximum"]:
                expected[f"lag{lag}_{field}"] = record[field]
        expected["lag1_last"] = previous["last"]
        slots = previous["slots"]
        expected.update({"prev_slot_rise": slots[-1] - slots[0], "prev_slot_range": max(slots) - min(slots),
            "prev_slot_slope": math.fsum(v * w for v, w in zip(slots, [-1.5,-.5,.5,1.5])) / 5,
            "last_delta": previous["last"] - previous2["last"], "mean_delta": previous["mean"] - previous2["mean"],
            "maximum_delta": previous["maximum"] - previous2["maximum"], "production_delta": previous["production"] - previous2["production"]})
        duration, gap = 0, 0
        for lag in range(1, 7):
            earlier = raw.get(t - timedelta(hours=lag))
            if earlier is None:
                gap = 1
                break
            if raw_state(earlier) != "low":
                break
            duration += 1
        expected["low_run_length6"], expected["low_history_gap6"] = duration, gap
        assert set(expected) == set(run["features"]["dynamic"])
        for name, value in expected.items():
            close(r[name], value)
            feature_checks += 1
    outer_starts = {"dev_apr": datetime(2021,4,1), "dev_may": datetime(2021,5,1), "dev_jun": datetime(2021,6,1)}
    for r in rows(OUT / "fold_audit.csv"):
        start = datetime(2021, int(r["inner_month"]), 1)
        train = [t for t in eligible if t < start]
        val = [t for t in eligible if t.month == start.month]
        assert int(r["train_hours"]) == len(train) and int(r["validation_hours"]) == len(val)
        assert datetime.fromisoformat(r["train_last"]) == max(train) < min(val)
        assert datetime.fromisoformat(r["validation_first"]) == min(val)
        assert datetime.fromisoformat(r["validation_last"]) == max(val) < outer_starts[r["split"]]
        for name, part in [("train", train), ("validation", val)]:
            assert int(r[name + "_low"]) == sum(boolean(frame[t]["prior_low"]) for t in part)
            assert int(r[name + "_rises"]) == sum(boolean(frame[t]["rise_event"]) for t in part)
    pred, inner = rows(OUT / "predictions.csv"), rows(OUT / "inner_predictions.csv")
    index = {(r["split"], r["group"], r["timestamp"]): r for r in pred}
    assert len(index) == len(pred) == 10920
    groups = {name: [r for r in pred if r["group"] == name] for name in c["methods"]}
    expected_times = {t for t in eligible if 4 <= t.month <= 6}
    assert len(expected_times) == 2184
    parent_b = {r["timestamp"]: r for r in rows(ROOT / "Modeling/tables/m03/c/predictions.csv") if r["target"] == c["target"] and r["group"] == "B"}
    for name, g in groups.items():
        assert {datetime.fromisoformat(r["timestamp"]) for r in g} == expected_times
        for r in g:
            t = datetime.fromisoformat(r["timestamp"])
            close(r["actual"], raw[t]["maximum"])
            for field in ["prior_low", "rise_event", "power_transition"]:
                assert r[field] == frame[t][field]
            for field in ["daily_maximum_weight", "profile_weight"]:
                close(r[field], parent_b[r["timestamp"]][field])
            if name == "B":
                close(r["prediction"], parent_b[r["timestamp"]]["prediction"])
            error = float(r["prediction"]) - float(r["actual"])
            for field, value in [("signed_error", error), ("absolute_error", abs(error)), ("under_amount", max(-error,0)), ("over_amount", max(error,0))]:
                close(r[field], value)
    configs = json.loads((OUT / "selected_configurations.json").read_text(encoding="utf-8"))
    trials = rows(OUT / "trials.csv")
    assert len(trials) == 450 and all(r["state"] == "COMPLETE" for r in trials)
    for config in configs:
        split = config["split"]
        selected_trials = [("weighted", config["weighted_best_trial"], {"event_weight": config["event_weight"]})]
        for family, details in config["classifiers"].items():
            selected_trials.extend([("classifier_" + family, details["classifier_best_trial"], details["parameters"]),
                                    ("gate_" + family, details["gate_best_trial"], details["gate"])])
            for role, pool in [("outer", pred), ("inner", inner)]:
                g = [r for r in pool if r["split"] == split and r["group"] == "G_" + family]
                result = classification(g, details["gate"]["threshold"])
                saved = next(r for r in rows(OUT / "classification.csv") if (r["split"],r["role"],r["model"]) == (split,role,"G_"+family))
                for key, value in result.items():
                    close(saved[key], value)
                if role == "inner":
                    for key, value in stats(g).items():
                        close(details["gate_inner"]["metrics"][key], value)
                for r in g:
                    reference = next(x for x in pool if x["split"] == split and x["group"] == "B" and x["timestamp"] == r["timestamp"])
                    expert = next(x for x in pool if x["split"] == split and x["group"] == "W_rise_weighted" and x["timestamp"] == r["timestamp"])
                    trigger = boolean(r["prior_low"]) and float(r["probability"]) >= details["gate"]["threshold"]
                    predicted = float(reference["prediction"]) + trigger * details["gate"]["blend"] * (float(expert["prediction"]) - float(reference["prediction"]))
                    close(r["prediction"], predicted)
                    if role == "outer":
                        assert boolean(r["trigger"]) == trigger
        for family, number, parameters in selected_trials:
            part = [r for r in trials if r["split"] == split and r["family"] == family]
            assert len(part) == 30
            if family.startswith("classifier_"):
                best = min(part, key=lambda r: (-float(r["objective"]), json.loads(r["metrics"])["classification"]["Brier"], int(r["trial"])))
            else:
                best = min(part, key=lambda r: (float(r["objective"]), int(r["trial"])))
            assert int(best["trial"]) == number and json.loads(best["parameters"]) == parameters
    for r in rows(OUT / "metrics.csv"):
        g = groups[r["group"]]
        kind, value = r["kind"], r["value"]
        weight_kind = "all"
        if kind in ["daily_maximum", "profile_reweighted"]:
            weight_kind = kind
            part = [x for x in g if float(x["daily_maximum_weight"]) > 0] if kind == "daily_maximum" else g
        elif kind.startswith("peak_"):
            weight_kind = "daily_maximum"
            part = [x for x in g if float(x["daily_maximum_weight"]) > 0 and x[kind[5:]] == value]
        else:
            part = g if kind == "all" else [x for x in g if x[kind] == value]
        verify_metric(r, part, weight_kind)
    table = rows(OUT / "selection_metrics.csv")
    reference_stats = stats(groups["B"])
    for r in table:
        values = stats(groups[r["group"]])
        for key, value in values.items():
            close(r[key], value)
        requirements = []
        for key, setting in [("overall_MAE", "overall_MAE_ratio_max"), ("daily_maximum_MAE", "daily_maximum_MAE_ratio_max"),
            ("low_stay_MAE", "low_stay_MAE_ratio_max"), ("low_stay_over", "low_stay_mean_over_ratio_max"),
            ("rise_MAE", "rise_MAE_ratio_max"), ("rise_under", "rise_mean_under_ratio_max")]:
            ratio = values[key] / reference_stats[key]
            passed = ratio <= c["development_selection"][setting] + 1e-12
            close(r[key + "_ratio"], ratio)
            assert boolean(r[key + "_passed"]) == passed
            requirements.append(passed)
        monthly = all(stats([x for x in groups[r["group"]] if x["split"] == split])["rise_MAE"] <=
            stats([x for x in groups["B"] if x["split"] == split])["rise_MAE"] + 1e-9 for split in outer_starts)
        assert boolean(r["rise_monthly_passed"]) == monthly
        requirements.append(monthly)
        if r["group"].startswith("G_"):
            low = [x for x in groups[r["group"]] if boolean(x["prior_low"])]
            fp = sum(boolean(x["trigger"]) and not boolean(x["rise_event"]) for x in low)
            negatives = sum(not boolean(x["rise_event"]) for x in low)
            close(r["gate_false_positive_fraction"], fp / negatives)
            passed = fp / negatives <= .05 + 1e-12
            assert boolean(r["gate_false_positive_passed"]) == passed
            requirements.append(passed)
        assert boolean(r["eligible"]) == (all(requirements) and r["group"] != "B")
        close(r["balanced_score"], max(values[k] / reference_stats[k] for k in ["overall_MAE", "daily_maximum_MAE", "rise_MAE"]))
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    possible = [r for r in table if boolean(r["eligible"])]
    if possible:
        minimum = min(float(r["balanced_score"]) for r in possible)
        near = [r for r in possible if float(r["balanced_score"]) <= minimum * 1.01]
        chosen = min(near, key=lambda r: (c["methods"].index(r["group"]), float(r["balanced_score"])))["group"]
    else:
        chosen = "B"
    assert selection["selected"] == chosen and selection["eligible_methods"] == [r["group"] for r in possible]
    # Validate paired/daily arithmetic, not merely their stored hash.
    for r in rows(OUT / "paired_predictions.csv"):
        x, b = index[r["split"],r["group"],r["timestamp"]], index[r["split"],"B",r["timestamp"]]
        for field, label in [("absolute_error","abs"), ("under_amount","under"), ("over_amount","over")]:
            close(r["delta_" + label], float(x[field]) - float(b[field]))
    for r in rows(OUT / "daily_errors.csv"):
        g = [x for x in groups[r["group"]] if x["date"] == r["date"]]
        b = [x for x in groups["B"] if x["date"] == r["date"]]
        assert len(g) == len(b) == int(r["hours"]) == 24
        for field, label in [("absolute_error","MAE"), ("under_amount","under"), ("over_amount","over")]:
            close(r["delta_" + label], (math.fsum(float(x[field]) for x in g) - math.fsum(float(x[field]) for x in b)) / 24)
    df = pd.read_csv(OUT / "feature_frame.csv", encoding="utf-8-sig", float_precision="round_trip").set_index("timestamp")
    reloads = 0
    for r in rows(OUT / "model_manifest.csv"):
        path = ROOT / r["file"]
        assert sha(path) == r["sha256"]
        estimator = joblib.load(path)
        part = [x for x in groups["B"] if x["split"] == r["split"]]
        inputs = df.loc[[x["timestamp"] for x in part], run["features"]["dynamic"]]
        if r["name"].endswith("_classifier"):
            family = r["name"].split("_")[0]
            expected = [float(index[r["split"],"G_" + family,x["timestamp"]]["probability"]) for x in part]
            actual = np.full(len(part), estimator["constant_probability"]) if isinstance(estimator, dict) else estimator.predict_proba(inputs)[:,1]
            if family == "Logistic" and not isinstance(estimator, dict):
                training = [t for t in frame if t < outer_starts[r["split"]] and boolean(frame[t]["prior_low"])]
                matrix = df.loc[[str(t) for t in training], run["features"]["dynamic"]]
                assert np.allclose(estimator[0].mean_, matrix.mean().to_numpy(), atol=1e-9, rtol=1e-10)
        else:
            expected = [float(index[r["split"],r["name"],x["timestamp"]]["prediction"]) for x in part]
            actual = estimator.predict(inputs)
        assert np.allclose(actual, expected, atol=1e-9, rtol=0)
        reloads += 1
    result = {"status": "passed", "checked_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"),
        "run_sha256": sha(OUT / "run.json"), "script_sha256": sha(Path(__file__)), "raw_feature_values_checked": feature_checks,
        "prediction_rows": len(pred), "metric_rows": len(rows(OUT / "metrics.csv")), "trials": len(trials),
        "model_reload_checks": reloads, "selected": chosen, "no_july_august_evaluation": True,
        "AP_and_confusion_independently_recomputed": True, "training_only_scaler_checked": True}
    (OUT / "independent_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
