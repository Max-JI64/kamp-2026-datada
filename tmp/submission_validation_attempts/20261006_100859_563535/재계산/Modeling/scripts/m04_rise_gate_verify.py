"""Independently verify cached gate trials, predictions and unchanged acceptance gates."""
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from m03_verify import close, rows, sha, verify_metric
from m04_rise_verify import boolean, classification, stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Modeling/tables/m04_rise_gate"
PARENT = ROOT / "Modeling/tables/m04_rise"


def grouped(pool):
    result = defaultdict(list)
    for r in pool:
        result[r["split"], r["group"]].append(r)
    return result


def gate_rows(base, expert, probability, threshold, blend):
    result = []
    for b, w, p in zip(base, expert, probability):
        assert b["timestamp"] == w["timestamp"] == p["timestamp"]
        trigger = boolean(b["prior_low"]) and float(p["probability"]) >= threshold
        r = dict(b)
        r["prediction"] = float(b["prediction"]) + trigger * blend * (float(w["prediction"]) - float(b["prediction"]))
        r["probability"] = p["probability"]
        r["trigger"] = str(trigger)
        result.append(r)
    return result


def objective(values, reference, limits, fpr):
    penalty = math.fsum(max(values[k] / reference[k] / limits[setting] - 1, 0) for k, setting in [
        ("overall_MAE", "overall_MAE_ratio_max"), ("daily_maximum_MAE", "daily_maximum_MAE_ratio_max"),
        ("low_stay_MAE", "low_stay_MAE_ratio_max"), ("low_stay_over", "low_stay_mean_over_ratio_max")])
    penalty += max(fpr / limits["gate_false_positive_fraction_max"] - 1, 0)
    return values["rise_MAE"] / reference["rise_MAE"] + 10 * penalty


def main():
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    run = json.loads((OUT / "run.json").read_text(encoding="utf-8"))
    follow_path = ROOT / "Modeling/config/m04_rise_gate_contract.json"
    follow = json.loads(follow_path.read_text(encoding="utf-8"))
    contract = json.loads((ROOT / "Modeling/config/m04_rise_contract.json").read_text(encoding="utf-8"))
    parent = json.loads((PARENT / "run.json").read_text(encoding="utf-8"))
    independent = json.loads((PARENT / "independent_verification.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed" and run["no_new_fitting"] and run["no_july_august_evaluation"]
    assert sha(follow_path) == run["contract_sha256"]
    assert sha(ROOT / "Modeling/scripts/m04_rise_gate.py") == run["script_sha256"]
    assert independent["status"] == "passed"
    assert sha(PARENT / "run.json") == run["parent_run_sha256"] == follow["parent_run_sha256"] == independent["run_sha256"]
    for name, expected in parent["outputs_sha256"].items():
        assert sha(PARENT / name) == expected, name
    for name, expected in run["outputs_sha256"].items():
        assert sha(OUT / name) == expected, name
    pred, inn = rows(OUT / "predictions.csv"), rows(OUT / "inner_predictions.csv")
    original, cached = rows(PARENT / "predictions.csv"), rows(PARENT / "inner_predictions.csv")
    index = {(r["split"], r["group"], r["timestamp"]): r for r in pred}
    assert len(index) == len(pred) == run["prediction_rows"] == 15288
    for r in original:
        current = index[r["split"], r["group"], r["timestamp"]]
        for field in r:
            if field in ["prediction", "signed_error", "absolute_error", "under_amount", "over_amount"]:
                close(current[field], r[field])
            else:
                assert current[field] == r[field], field
    parent_groups, cached_groups = grouped(original), grouped(cached)
    inner_groups, outer_groups = grouped(inn), grouped(pred)
    configs = json.loads((OUT / "selected_configurations.json").read_text(encoding="utf-8"))
    trial_rows, alarms = rows(OUT / "trials.csv"), rows(OUT / "classification.csv")
    assert len(configs) == 6 and len(trial_rows) == run["trials"] == 300 and len(alarms) == 12
    trial_checks = 0
    for config in configs:
        split, group = config["split"], config["group"]
        family = group.removesuffix("_rise_focus")
        part = [r for r in trial_rows if r["split"] == split and r["group"] == group]
        assert len(part) == 50 and {int(r["trial"]) for r in part} == set(range(50))
        best = min(part, key=lambda r: (float(r["objective"]), int(r["trial"])))
        assert int(best["trial"]) == config["best_trial"] and json.loads(best["parameters"]) == config["parameters"]
        start = datetime(2021, {"dev_apr": 4, "dev_may": 5, "dev_jun": 6}[split], 1)
        assert max(datetime.fromisoformat(r["timestamp"]) for r in cached_groups[split, "B"]) < start
        reference = stats(cached_groups[split, "B"])
        for trial in part:
            parameters = json.loads(trial["parameters"])
            assert .01 <= parameters["threshold"] <= .8 and 0 <= parameters["blend"] <= 1
            generated = gate_rows(cached_groups[split, "B"], cached_groups[split, "W_rise_weighted"],
                                  cached_groups[split, family], **parameters)
            values = stats(generated)
            alarm = classification(generated, parameters["threshold"])
            close(trial["objective"], objective(values, reference, contract["development_selection"], alarm["false_positive_fraction"]))
            attrs = json.loads(trial["metrics"])
            for key, value in values.items():
                close(attrs["metrics"][key], value)
            for key, value in alarm.items():
                close(attrs["classification"][key], value)
            trial_checks += 1
        for role, base_groups, actual_groups in [("inner", cached_groups, inner_groups), ("outer", parent_groups, outer_groups)]:
            generated = gate_rows(base_groups[split, "B"], base_groups[split, "W_rise_weighted"],
                                  base_groups[split, family], **config["parameters"])
            actual = actual_groups[split, group]
            assert len(actual) == len(generated)
            for r, expected in zip(actual, generated):
                assert r["timestamp"] == expected["timestamp"] and r["trigger"] == expected["trigger"]
                close(r["prediction"], expected["prediction"])
                close(r["probability"], expected["probability"])
            alarm = classification(actual, config["parameters"]["threshold"])
            saved = next(r for r in alarms if (r["split"], r["role"], r["group"]) == (split, role, group))
            for key, value in alarm.items():
                close(saved[key], value)
            if role == "inner":
                for key, value in stats(actual).items():
                    close(config["inner_metrics"]["metrics"][key], value)
    groups = defaultdict(list)
    for r in pred:
        groups[r["group"]].append(r)
        assert datetime.fromisoformat(r["timestamp"]).month in [4, 5, 6]
        error = float(r["prediction"]) - float(r["actual"])
        for key, value in [("signed_error", error), ("absolute_error", abs(error)), ("under_amount", max(-error, 0)), ("over_amount", max(error, 0))]:
            close(r[key], value)
    expected_times = {r["timestamp"] for r in groups["B"]}
    assert len(groups) == 7 and len(expected_times) == 2184
    assert all(len(g) == 2184 and {r["timestamp"] for r in g} == expected_times for g in groups.values())
    metrics = rows(OUT / "metrics.csv")
    assert len(metrics) == run["metric_rows"] == 364
    for r in metrics:
        g, kind, value = groups[r["group"]], r["kind"], r["value"]
        weight = "all"
        if kind in ["daily_maximum", "profile_reweighted"]:
            weight = kind
            part = [x for x in g if float(x["daily_maximum_weight"]) > 0] if kind == "daily_maximum" else g
        elif kind.startswith("peak_"):
            weight = "daily_maximum"
            part = [x for x in g if float(x["daily_maximum_weight"]) > 0 and x[kind[5:]] == value]
        else:
            part = g if kind == "all" else [x for x in g if x[kind] == value]
        verify_metric(r, part, weight)
    table = rows(OUT / "selection_metrics.csv")
    reference = stats(groups["B"])
    for r in table:
        values = stats(groups[r["group"]])
        for key, value in values.items():
            close(r[key], value)
        passed = []
        for key, setting in [("overall_MAE", "overall_MAE_ratio_max"), ("daily_maximum_MAE", "daily_maximum_MAE_ratio_max"),
            ("low_stay_MAE", "low_stay_MAE_ratio_max"), ("low_stay_over", "low_stay_mean_over_ratio_max"),
            ("rise_MAE", "rise_MAE_ratio_max"), ("rise_under", "rise_mean_under_ratio_max")]:
            ratio = values[key] / reference[key]
            flag = ratio <= contract["development_selection"][setting] + 1e-12
            close(r[key + "_ratio"], ratio)
            assert boolean(r[key + "_passed"]) == flag
            passed.append(flag)
        monthly = all(stats(outer_groups[split, r["group"]])["rise_MAE"] <= stats(outer_groups[split, "B"])["rise_MAE"] + 1e-9 for split in contract["outer_splits"])
        assert boolean(r["rise_monthly_passed"]) == monthly
        passed.append(monthly)
        if r["group"].startswith("G_"):
            low = [x for x in groups[r["group"]] if boolean(x["prior_low"])]
            fp = sum(boolean(x["trigger"]) and not boolean(x["rise_event"]) for x in low)
            negatives = sum(not boolean(x["rise_event"]) for x in low)
            close(r["gate_false_positive_fraction"], fp / negatives)
            flag = fp / negatives <= contract["development_selection"]["gate_false_positive_fraction_max"] + 1e-12
            assert boolean(r["gate_false_positive_passed"]) == flag
            passed.append(flag)
        assert boolean(r["eligible"]) == (r["group"] != "B" and all(passed))
        close(r["balanced_score"], max(values[k] / reference[k] for k in ["overall_MAE", "daily_maximum_MAE", "rise_MAE"]))
    possible = [r for r in table if boolean(r["eligible"])]
    order = contract["methods"] + ["G_Logistic_rise_focus", "G_HGB_rise_focus"]
    if possible:
        minimum = min(float(r["balanced_score"]) for r in possible)
        near = [r for r in possible if float(r["balanced_score"]) <= minimum * 1.01]
        chosen = min(near, key=lambda r: (order.index(r["group"]), float(r["balanced_score"]))) ["group"]
    else:
        chosen = "B"
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    assert selection["selected"] == chosen and selection["eligible_methods"] == [r["group"] for r in possible]
    assert selection["acceptance_criteria_unchanged"] and selection["no_july_august_evaluation"]
    result = {"status": "passed", "checked_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"),
        "run_sha256": sha(OUT / "run.json"), "script_sha256": sha(Path(__file__)), "prediction_rows": len(pred),
        "metric_rows": len(metrics), "trials_independently_recomputed": trial_checks,
        "prior_five_methods_preserved": True, "AP_and_confusion_independently_recomputed": True,
        "acceptance_criteria_unchanged": True, "no_new_fitting": True, "no_july_august_evaluation": True, "selected": chosen}
    (OUT / "independent_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
