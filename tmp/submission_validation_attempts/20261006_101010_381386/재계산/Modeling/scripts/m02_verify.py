"""Independent stdlib verification of M02 saved forecasts, scores and weights."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Modeling/tables/m02"


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-9), (a, b)


def power_state(maximum, rounded):
    if float(maximum) == 0:
        return "zero"
    if float(rounded) < 20:
        return "below20"
    return "low" if float(rounded) <= 26 else "above26"


def main():
    run = json.loads((OUT / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed" and run["fits_completed"] == 150
    assert run["july_august_evaluated"] is False and run["last_slot_used"] is False
    parent_path = ROOT / "Modeling/config/m01_contract.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    source = ROOT / parent["source"]
    for key, path in {"m01_contract": parent_path, "m02_contract": ROOT / "Modeling/config/m02_contract.json", "source": source,
                      **{n: ROOT / "Modeling/tables/m01" / n for n in ["hourly_frame.csv", "evaluation_diagnostics.csv", "split_counts.csv"]}}.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == run["input_hashes"][key]
    assert run["features"] == parent["calendar_features"] + parent["feature_groups"]["A"]
    for name, digest in run["outputs_sha256"].items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest
    f = {r["timestamp"]: r for r in rows(ROOT / "Modeling/tables/m01/hourly_frame.csv")}
    diags = {(r["split"], r["timestamp"]): r for r in rows(ROOT / "Modeling/tables/m01/evaluation_diagnostics.csv") if r["pool"] == "common"}
    c = json.loads((ROOT / "Modeling/config/m02_contract.json").read_text(encoding="utf-8"))
    pred = rows(OUT / "predictions.csv")
    groups, membership = defaultdict(list), defaultdict(set)
    for r in pred:
        own = f[r["timestamp"]]
        assert own["eligible_common"] == "True"
        assert datetime.fromisoformat(r["timestamp"]) < datetime(2021, 7, 1)
        close(r["actual"], own[r["target"]])
        prior_state = power_state(own["lag1_maximum"], math.floor(float(own["lag1_mean"]) + 0.5))
        current_state = power_state(own["target_maximum"], own["diag_target_rounded_mean"])
        assert r["power_transition"] == prior_state + "->" + current_state
        assert r["diag_production_transition"] == own["diag_production_transition"]
        error = float(r["prediction"]) - float(r["actual"])
        close(error, r["signed_error"])
        close(abs(error), r["absolute_error"])
        close(max(-error, 0), r["under_amount"])
        close(max(error, 0), r["over_amount"])
        diag = diags[(r["split"], r["timestamp"])]
        for col in ["profile_weight", "daily_maximum_weight"]:
            close(r[col], diag[col])
        assert r["train_profile_overlap"] == diag["train_profile_overlap"]
        if r["model"] in c["baselines"]:
            lag = c["baselines"][r["model"]]
            metric_name = "maximum" if r["target"] == "target_maximum" else "mean"
            close(r["prediction"], own[f"lag{lag}_{metric_name}"])
        key = (r["target"], r["model"])
        groups[key].append(r)
        membership[key].add(r["timestamp"])
    common_times = next(iter(membership.values()))
    assert len(common_times) == 2184
    assert all(v == common_times for v in membership.values())
    assert all(len(v) == 2184 for v in groups.values())
    checked_scores = 0
    for s in rows(OUT / "all_candidate_metrics.csv"):
        g = groups[(s["target"], s["model"])]
        if s["split"] != "pooled_development":
            g = [r for r in g if r["split"] == s["split"]]
        errors = [float(r["prediction"]) - float(r["actual"]) for r in g]
        assert len(g) == int(s["hours"])
        close(sum(abs(e) for e in errors) / len(g), s["MAE"])
        close(math.sqrt(sum(e * e for e in errors) / len(g)), s["RMSE"])
        close(sum(max(-e, 0) for e in errors) / len(g), s["mean_under"])
        close(sum(max(e, 0) for e in errors) / len(g), s["mean_over"])
        checked_scores += 1
    peaks = 0
    for s in rows(OUT / "condition_errors.csv"):
        if s["condition"] != "daily_maximum":
            continue
        g = [r for r in groups[(s["target"], s["model"])] if float(r["daily_maximum_weight"]) > 0]
        w = [float(r["daily_maximum_weight"]) for r in g]
        close(sum(w), 91)
        close(sum(wi * abs(float(r["prediction"]) - float(r["actual"])) for wi, r in zip(w, g)) / sum(w), s["MAE"])
        close(sum(wi * max(float(r["actual"]) - float(r["prediction"]), 0) for wi, r in zip(w, g)) / sum(w), s["mean_under"])
        peaks += 1
    for r in rows(OUT / "model_manifest.csv"):
        assert hashlib.sha256((ROOT / r["file"]).read_bytes()).hexdigest() == r["sha256"]
        assert r["reload_predictions_checked"] == "True"
    for r in rows(OUT / "training_audit.csv"):
        assert datetime.fromisoformat(r["train_last_timestamp"]) < datetime.fromisoformat(r["eval_first_timestamp"])
        assert r["training_only_scalers_verified"] == "True"
        assert json.loads(r["warnings"]) == []
    result = {"status": "passed", "prediction_rows_checked": len(pred), "metrics_recomputed": checked_scores,
              "same_evaluation_timestamps": len(common_times), "daily_maximum_metrics_recomputed": peaks,
              "daily_maximum_weighted_dates": 91, "later_period_absent": True,
              "baseline_predictions_checked": True, "errors_and_input_labels_checked": True,
              "outputs_and_model_hashes_checked": True}
    (OUT / "independent_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
