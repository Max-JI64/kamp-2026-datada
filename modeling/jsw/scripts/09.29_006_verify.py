"""Independent source/key/metric verification of M006, without importing M006."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import re
from statistics import fmean, pstdev

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent.parent
P = "09.29_006"
BLEND = "hgb_lag1_high_blend05"
Q75 = "hgb_q75_calendar"


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def close(a, b, label=""):
    assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-8), (label, a, b)


def boolean(text):
    assert text in ["True", "False"]
    return text == "True"


def dt(text):
    return datetime.fromisoformat(text)


def main():
    cfg = json.loads((HERE / "tables" / f"{P}_frozen.json").read_text(encoding="utf-8"))
    protected = cfg["protected_sha256"]
    for name, value in protected.items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == value, name
    raw = read(ROOT / "data/origin/okm_augumented_2021.csv")
    source = {}
    for seq, row in enumerate(raw, 1):
        hour = int(row["시간"])
        if not 0 <= hour <= 23 or int(row["날짜"]) >= 20210901:
            continue
        t = datetime.strptime(row["날짜"], "%Y%m%d") + timedelta(hours=hour)
        slots = [float(row[s]) for s in ["15분", "30분", "45분", "60분"]]
        source[t] = {"mean": float(row["평균"]), "peak": max(slots), "range": max(slots) - min(slots),
                     "last_slot": slots[-1], "source_row": seq, "production": float(row["생산량"])}
    assert len(source) == 5784
    # Independently recover original eligibility and training quantiles.
    source_eligible = [t for t in sorted(source) if all(t - timedelta(hours=i) in source for i in [*range(1, 25), 168])]
    assert len(source_eligible) == 5520
    for fold in cfg["folds"]:
        cutoff = dt(fold["cutoff"])
        training = sorted(source[t]["peak"] for t in source_eligible if t < cutoff)
        assert len(training) == fold["train_n"]
        position = .95 * (len(training) - 1)
        lower = math.floor(position)
        q95 = training[lower] + (training[math.ceil(position)] - training[lower]) * (position - lower)
        close(q95, fold["threshold"])
    base = {}
    for part in ["development", "review"]:
        for r in read(HERE / "predictions" / f"09.26_001_{part}_predictions.csv"):
            base[(r["stage"], r["target"], r["method"], dt(r["record_key"]))] = r
    for r in read(HERE / "predictions/09.26_002_development_predictions.csv"):
        if r["method"] == Q75:
            base[(r["stage"], r["target"], Q75, dt(r["record_key"]))] = r
    coverage = read(HERE / "predictions" / f"{P}_target_coverage.csv")
    assert len(coverage) == 2904 and len({r["record_key"] for r in coverage}) == 2904
    eligible = {r["record_key"] for r in coverage if boolean(r["eligible"])}
    assert len(eligible) == 2808
    by_source = {r["record_key"]: r for r in coverage}
    predictions = []
    for part, count in [("development", 14640), ("review", 13440)]:
        rows = read(HERE / "predictions" / f"{P}_{part}_predictions.csv")
        assert len(rows) == count
        predictions.extend(rows)
    keys, by_key, fold_keys = set(), {}, {}
    day_max = {}
    for t, s in source.items():
        day_max[t.date()] = max(day_max.get(t.date(), -math.inf), s["peak"])
    for r in predictions:
        t = dt(r["record_key"])
        key = (r["stage"], r["target"], r["method"], t)
        assert key not in keys
        keys.add(key)
        by_key[key] = r
        fold_keys.setdefault(key[:3], set()).add(r["record_key"])
        assert r["record_key"] in eligible and boolean(r["input_available"])
        cutoff = dt(r["train_cutoff_exclusive"])
        assert dt(r["training_end"]) < cutoff <= t
        assert dt(r["forecast_after_completed_record"]) == t - timedelta(hours=1)
        assert r["evaluation_kind"] == ("exploratory_development" if t.month < 7 else "exploratory_followup")
        assert t.month in [5, 6, 7, 8] and r["policy_id"] == cfg["policy_id"]
        actual, prediction, u = float(r["actual"]), float(r["prediction"]), float(r["high_threshold_from_training"])
        s, previous = source[t], source[t - timedelta(hours=1)]
        close(actual, s[r["target"]])
        assert int(r["source_data_row"]) == s["source_row"]
        close(r["previous_peak"], previous["peak"])
        assert u == {"May": 183, "Jun": 181, "Jul-Aug": 182}[r["stage"]]
        state = "high_continued" if s["peak"] >= u and previous["peak"] >= u else "high_onset" if s["peak"] >= u else "descending" if previous["peak"] >= u else "ordinary"
        assert r["state"] == state and boolean(r["gate_previous_high"]) == (previous["peak"] >= u)
        assert boolean(r["is_daily_peak"]) == (s["peak"] == day_max[t.date()])
        assert prediction >= 0 and math.isfinite(prediction)
        if r["method"] == BLEND:
            old = float(base[(r["stage"], "peak", "hgb_calendar", t)]["prediction"])
            close(prediction, (old + previous["peak"]) / 2 if previous["peak"] >= u else old)
        elif r["method"].startswith("lag"):
            hours = int(r["method"][3:])
            close(prediction, source[t - timedelta(hours=hours)][r["target"]])
        elif r["method"] != Q75 or t.month < 7:
            close(prediction, base[key]["prediction"])
        error = prediction - actual
        for name, expected in [("error", error), ("abs_error", abs(error)), ("shortfall", max(-error, 0)), ("over", max(error, 0))]:
            close(r[name], expected)
    for (stage, target, method), values in fold_keys.items():
        expected = {r["record_key"] for r in coverage if r["stage"] == stage and boolean(r["eligible"])}
        assert values == expected, (stage, target, method)
    assert len(fold_keys) == 30
    # Reconstruct all 26 model inputs from original source keys independently.
    qrows = sorted([r for r in predictions if r["method"] == Q75 and int(r["month"]) >= 7], key=lambda r: r["record_key"])
    feature_rows = []
    for row in qrows:
        t = dt(row["record_key"])
        f = {}
        for variable in ["mean", "peak"]:
            for lag in [1, 2, 3, 24, 168]:
                f[f"{variable}_lag{lag}"] = source[t - timedelta(hours=lag)][variable]
        for variable in ["last_slot", "range"]:
            f[f"{variable}_lag1"] = source[t - timedelta(hours=1)][variable]
        for variable, functions in [("mean", ["mean", "std", "min", "max"]), ("peak", ["mean", "max"])]:
            v = [source[t - timedelta(hours=i)][variable] for i in range(1, 25)]
            options = {"mean": fmean(v), "std": pstdev(v), "min": min(v), "max": max(v)}
            for name in functions:
                f[f"{variable}_past24_{name}"] = options[name]
        f.update(hour=t.hour, weekday=t.weekday(), month=t.month,
                 hour_sin=np.sin(2 * np.pi * t.hour / 24), hour_cos=np.cos(2 * np.pi * t.hour / 24),
                 weekday_sin=np.sin(2 * np.pi * t.weekday() / 7), weekday_cos=np.cos(2 * np.pi * t.weekday() / 7),
                 weekend=int(t.weekday() >= 5))
        feature_rows.append(f)
    bundle = joblib.load(HERE / "models" / f"{P}_q75_june_end.joblib")
    assert bundle["feature_order"] == cfg["feature_order"] and bundle["settings"] == cfg["existing_comparator"]["settings"]
    for name, expected in bundle["settings"].items():
        assert bundle["estimator"].get_params()[name] == expected
    run_facts = json.loads((HERE / "tables" / f"{P}_facts.json").read_text(encoding="utf-8"))
    assert sha256((HERE / "models" / f"{P}_q75_june_end.joblib").read_bytes()).hexdigest() == run_facts["model_sha256"]
    features = pd.DataFrame(feature_rows)[bundle["feature_order"]]
    with threadpool_limits(limits=2):
        qpred = np.maximum(bundle["estimator"].predict(features), 0)
    np.testing.assert_allclose(qpred, [float(r["prediction"]) for r in qrows], rtol=1e-11, atol=1e-8)
    checked_scores = 0
    for table in ["scores", "condition_scores"]:
        for score in read(HERE / "tables" / f"{P}_{table}.csv"):
            months = [5, 6] if score["period"] == "development" else [7, 8] if score["period"] == "review" else [int(score["period"])]
            group = [r for r in predictions if int(r["month"]) in months and r["target"] == score["target"] and r["method"] == score["method"]]
            c = score.get("condition")
            if c in ["high_onset", "high_continued", "descending", "ordinary"]:
                group = [r for r in group if r["state"] == c]
            elif c == "high":
                group = [r for r in group if float(r["peak"]) >= float(r["high_threshold_from_training"])]
            elif c == "zero":
                group = [r for r in group if float(r["actual"]) == 0]
            elif c == "daily_peak":
                group = [r for r in group if boolean(r["is_daily_peak"])]
            elif c == "gate_true":
                group = [r for r in group if boolean(r["gate_previous_high"])]
            elif c and c.startswith("profile_"):
                group = [r for r in group if r["profile_seen_contributed"] == c[8:]]
            elif c and c.startswith("production_"):
                group = [r for r in group if r["production_condition"] == c[11:]]
            assert len(group) == int(score["n"])
            error = [float(r["prediction"]) - float(r["actual"]) for r in group]
            expected = {"mae": fmean(abs(e) for e in error), "rmse": math.sqrt(fmean(e * e for e in error)),
                        "bias": fmean(error), "mean_shortfall": fmean(max(-e, 0) for e in error),
                        "mean_over": fmean(max(e, 0) for e in error), "under_rate": fmean(e < 0 for e in error),
                        "pinball_075": fmean(max(-.75 * e, .25 * e) for e in error),
                        "actual_le_prediction_rate": fmean(e >= 0 for e in error)}
            for name, value in expected.items():
                close(score[name], value, (table, score["period"], c, score["method"], name))
            assert int(score["under_n"]) == sum(e < 0 for e in error)
            checked_scores += 1
    # Every paired row refers to the identical original target and baseline.
    differences = read(HERE / "predictions" / f"{P}_row_differences.csv")
    for r in differences:
        t = dt(r["record_key"])
        a = by_key[(r["stage"], r["target"], r["method"], t)]
        b = by_key[(r["stage"], r["target"], "hgb_calendar", t)]
        for field in ["abs_error", "shortfall", "over"]:
            close(r[f"delta_{field}"], float(a[field]) - float(b[field]))
    for r in read(HERE / "tables" / f"{P}_sensitivity.csv"):
        months = [5, 6] if r["period"] == "development" else [7, 8] if r["period"] == "review" else [int(r["period"])]
        group = [d for d in differences if d["target"] == "peak" and d["method"] == r["method"] and int(d["month"]) in months
                 and d["state"] == "high_continued" and d[r["unit"]] != r["excluded"]]
        assert len(group) == int(r["remaining_n"])
        close(r["delta_shortfall"], fmean(float(d["delta_shortfall"]) for d in group))
    daily = read(HERE / "predictions" / f"{P}_daily_peak_dates.csv")
    for r in daily:
        full = [v for v in coverage if v["date"] == r["date"] and boolean(v["is_daily_peak"])]
        group = [by_key[(v["stage"], "peak", r["method"], dt(v["record_key"]))] for v in full if boolean(v["eligible"])]
        assert len(full) == int(r["original_peak_hours"]) and len(group) == int(r["available_peak_hours"])
        if group:
            close(r["day_mean_shortfall_available"], fmean(float(v["shortfall"]) for v in group))
        else:
            assert r["day_mean_shortfall_available"] == ""
    for r in read(HERE / "tables" / f"{P}_fixed_threshold_diagnostics.csv"):
        g = [v for v in predictions if v["target"] == "peak" and v["method"] == r["method"] and v["month"] == r["month"]]
        assert int(r["alarm_hours"]) == sum(float(v["prediction"]) >= float(v["high_threshold_from_training"]) for v in g)
        assert int(r["false_alarm_hours"]) == sum(float(v["prediction"]) >= float(v["high_threshold_from_training"]) > float(v["actual"]) for v in g)
    events = read(HERE / "predictions" / f"{P}_event_audit.csv")
    for r in events:
        start, end, u = dt(r["start"]), dt(r["end"]), float(r["threshold"])
        members = []
        t = start
        while t <= end:
            assert source[t]["peak"] >= u
            members.append(by_key.get((r["stage"], "peak", r["method"], t)))
            t += timedelta(hours=1)
        assert source[start - timedelta(hours=1)]["peak"] < u and source[end + timedelta(hours=1)]["peak"] < u
        hits = [v for v in members if v and float(v["prediction"]) >= u]
        expected = "start_unpredictable" if members[0] is None else "pre_start_warning" if members[0] in hits else "late_capture" if hits else "missed"
        assert r["outcome"] == expected
        if hits:
            first = dt(hits[0]["record_key"])
            assert dt(r["first_warning_target"]) == first
            assert dt(r["warning_after_completed_record"]) == first - timedelta(hours=1)
        assert int(r["missing_prediction_hours"]) == sum(v is None for v in members)
    handoff = json.loads((HERE / "tables" / f"{P}_m007_handoff.json").read_text(encoding="utf-8"))
    assert handoff["basic_point_models"] == {"mean": "hgb_calendar", "peak": "hgb_calendar"}
    scored = read(HERE / "tables" / f"{P}_scores.csv")
    conditioned = read(HERE / "tables" / f"{P}_condition_scores.csv")
    sens = read(HERE / "tables" / f"{P}_sensitivity.csv")
    def metric(table, period, method, field, condition=None):
        matches = [r for r in table if r["period"] == period and r["target"] == "peak" and r["method"] == method
                   and (condition is None or r["condition"] == condition)]
        assert len(matches) == 1
        return float(matches[0][field])
    for r in read(HERE / "tables" / f"{P}_selection.csv"):
        method = r["method"]
        gain = metric(conditioned, "development", method, "mean_shortfall", "high_continued") < metric(conditioned, "development", "hgb_calendar", "mean_shortfall", "high_continued") - 1e-10
        nonworse = all(metric(conditioned, m, method, "mean_shortfall", "high_continued") <= metric(conditioned, m, "hgb_calendar", "mean_shortfall", "high_continued") + 1e-10 for m in ["5", "6"])
        robust = all(float(s["delta_shortfall"]) <= 1e-10 for s in sens if s["period"] == "development" and s["method"] == method)
        dev_mae = all(metric(scored, m, method, "mae") <= metric(scored, m, "hgb_calendar", "mae") + 1e-10 for m in ["5", "6"])
        review_gain = all(metric(conditioned, m, method, "mean_shortfall", "high_continued") < metric(conditioned, m, "hgb_calendar", "mean_shortfall", "high_continued") - 1e-10 for m in ["7", "8"])
        review_mae = all(metric(scored, m, method, "mae") <= metric(scored, m, "hgb_calendar", "mae") + 1e-10 for m in ["7", "8"])
        for field, expected in [("dev_primary_improves", gain), ("dev_month_primary_nonworse", nonworse), ("dev_date_week_robust", robust),
                                ("dev_month_mae_nonworse", dev_mae), ("review_month_primary_improves", review_gain), ("review_month_mae_nonworse", review_mae)]:
            assert boolean(r[field]) == expected
        role = "basic_replacement" if gain and nonworse and robust and dev_mae and review_gain and review_mae else "peak_only_for_M007" if gain and nonworse and robust and review_gain else "discard"
        assert r["decision"] == role
        assert next(d for d in handoff["candidate_decisions"] if d["method"] == method)["decision"] == role
    for method in handoff["peak_methods_to_compare"] + handoff["auxiliary_baselines"]:
        for stage in ["May", "Jun", "Jul-Aug"]:
            assert (stage, "peak", method) in fold_keys
    facts = {"time": datetime.now().astimezone().isoformat(timespec="seconds"), "python": platform.python_version(),
             "exit_code_on_success": 0, "protected_sha256_checked": len(protected),
             "prediction_rows_checked": len(predictions), "unique_target_hours": len(eligible),
             "paired_rows_checked": len(differences), "score_groups_checked": checked_scores,
             "daily_peak_date_rows_checked": len(daily), "event_rows_checked": len(events),
             "q75_review_predictions_reconstructed_from_source": len(qrows),
             "method": "independent csv/source-key loops, source-based feature reconstruction, serialized q75 inference; no M006 imports"}
    path = HERE / "tables" / f"{P}_verification.json"
    path.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    links = 0
    for file in [HERE / "README.md", HERE / "reports" / f"{P}_targeted_model_plan.md"]:
        for target in re.findall(r"\]\(([^)]+)\)", file.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            local = target.split("#")[0]
            assert (file.parent / local).resolve().exists(), (file, target)
            links += 1
    facts["local_links_checked"] = links
    path.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(facts), flush=True)


if __name__ == "__main__":
    main()
