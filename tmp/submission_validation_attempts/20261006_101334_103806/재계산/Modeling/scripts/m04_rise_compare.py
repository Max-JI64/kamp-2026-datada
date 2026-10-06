"""M04 rise development: dynamics, weighted HGB, probability-gated expert; chronological Optuna."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from m01_prepare import ROOT, SLOTS, feature_columns, read_contract, sha
from m02_compare import metric, state
from m02_rerun import peak_weights
from m04_compare import subsets

OUT = ROOT / "Modeling/tables/m04_rise"
MODELS = ROOT / "Modeling/models/m04_rise"
CONTRACT = ROOT / "Modeling/config/m04_rise_contract.json"


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes")


def read_csv(path, dates=None):
    return pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip", parse_dates=dates)


def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def build_frame(c):
    parent = read_contract()
    source = ROOT / parent["source"]
    frame_path = ROOT / "Modeling/tables/m01/hourly_frame.csv"
    assert sha(source) == c["source_sha256"] == parent["source_sha256"]
    assert sha(frame_path) == c["frame_sha256"]
    f = read_csv(frame_path, ["timestamp", "date"])
    f = f.loc[f.timestamp.lt("2021-07-01")].copy()
    raw = read_csv(source)
    raw = raw.loc[raw["날짜"].between(20210101, 20210630) & raw["시간"].between(0, 23)].copy()
    raw["timestamp"] = pd.to_datetime(raw["날짜"].astype(str), format="%Y%m%d") + pd.to_timedelta(raw["시간"], unit="h")
    raw = raw.set_index("timestamp")
    assert raw.index.is_unique
    slots = raw[SLOTS].reindex(pd.DatetimeIndex(f.timestamp - pd.Timedelta(hours=1))).to_numpy()
    f["prev_slot_rise"] = slots[:, 3] - slots[:, 0]
    f["prev_slot_range"] = np.max(slots, axis=1) - np.min(slots, axis=1)
    f["prev_slot_slope"] = slots @ np.array([-1.5, -.5, .5, 1.5]) / 5
    f["last_delta"] = f.lag1_last - f.lag2_last
    f["mean_delta"] = f.lag1_mean - f.lag2_mean
    f["maximum_delta"] = f.lag1_maximum - f.lag2_maximum
    f["production_delta"] = f.lag1_production - f.lag2_production
    f["prior_state"] = state(f.lag1_maximum, np.floor(f.lag1_mean + .5))
    f["power_state"] = state(f.target_maximum, f.diag_target_rounded_mean)
    f["power_transition"] = f.prior_state + "->" + f.power_state
    f["prior_low"] = f.prior_state.eq("low")
    f["rise_event"] = f.power_transition.eq("low->above26")
    full_state = dict(zip(f.timestamp, f.power_state))
    durations, gaps = [], []
    for timestamp in f.timestamp:
        duration, gap = 0, 0
        for lag in range(1, 7):
            previous = full_state.get(timestamp - pd.Timedelta(hours=lag))
            if previous is None:
                gap = 1
                break
            if previous != "low":
                break
            duration += 1
        durations.append(duration)
        gaps.append(gap)
    f["low_run_length6"], f["low_history_gap6"] = durations, gaps
    f = f.loc[f.eligible_common].copy().reset_index(drop=True)
    f["hour"] = f.timestamp.dt.hour
    base = feature_columns("B", parent)
    dynamic = base + c["features"]["add"]
    assert len(base) == 18 and len(dynamic) == 27 and f[dynamic].notna().all().all()
    assert not any(x.startswith(("target_", "diag_")) for x in dynamic)
    return f, base, dynamic


def stats(frame, prediction):
    e = np.asarray(prediction) - frame.target_maximum.to_numpy()
    low = frame.prior_low.to_numpy()
    event = frame.rise_event.to_numpy()
    stay = frame.power_transition.eq("low->low").to_numpy()
    peak = peak_weights(frame)
    assert event.any() and stay.any() and peak.sum() > 0
    return {"overall_MAE": float(np.abs(e).mean()), "daily_maximum_MAE": float(np.average(np.abs(e), weights=peak)),
        "rise_MAE": float(np.abs(e[event]).mean()), "rise_under": float(np.maximum(-e[event], 0).mean()),
        "low_stay_MAE": float(np.abs(e[stay]).mean()), "low_stay_over": float(np.maximum(e[stay], 0).mean()),
        "prior_low_hours": int(low.sum()), "rise_hours": int(event.sum())}


def score(values, reference, limits, false_positive_fraction=None):
    ratios = {key: values[key] / reference[key] for key in ["overall_MAE", "daily_maximum_MAE", "rise_MAE", "low_stay_MAE", "low_stay_over"]}
    penalty = sum(max(ratios[key] / limit - 1, 0) for key, limit in [
        ("overall_MAE", limits["overall_MAE_ratio_max"]), ("daily_maximum_MAE", limits["daily_maximum_MAE_ratio_max"]),
        ("low_stay_MAE", limits["low_stay_MAE_ratio_max"]), ("low_stay_over", limits["low_stay_mean_over_ratio_max"])])
    if false_positive_fraction is not None:
        penalty += max(false_positive_fraction / limits["gate_false_positive_fraction_max"] - 1, 0)
    return max(ratios[key] for key in ["overall_MAE", "daily_maximum_MAE", "rise_MAE"]) + 10 * penalty


def classifier(family, params):
    if family == "Logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=params["C"], max_iter=2000, random_state=42))
    return HistGradientBoostingClassifier(max_leaf_nodes=params["max_leaf_nodes"], min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"], max_iter=150, learning_rate=.05, early_stopping=False, random_state=42)


def fit_probability(family, params, tr, ev, features):
    low = tr.loc[tr.prior_low]
    assert len(low) > 0
    if low.rise_event.nunique() < 2:
        probability = np.full(len(ev), float(low.rise_event.mean()))
        return {"constant_probability": float(low.rise_event.mean())}, probability, True
    estimator = classifier(family, params)
    estimator.fit(low[features], low.rise_event.astype(int))
    probability = estimator.predict_proba(ev[features])[:, 1]
    return estimator, probability, False


def probability(estimator, ev, features):
    if isinstance(estimator, dict):
        return np.full(len(ev), estimator["constant_probability"])
    return estimator.predict_proba(ev[features])[:, 1]


def alarm_stats(frame, probability_values, threshold):
    low = frame.prior_low.to_numpy()
    y = frame.rise_event.to_numpy()[low]
    p = np.asarray(probability_values)[low]
    alarm = p >= threshold
    tp, fp, fn, tn = [int(x.sum()) for x in [alarm & y, alarm & ~y, ~alarm & y, ~alarm & ~y]]
    return {"prior_low_hours": len(y), "positive_hours": int(y.sum()), "prevalence": float(y.mean()),
        "AP": float(average_precision_score(y, p)), "AUROC": float(roc_auc_score(y, p)),
        "Brier": float(brier_score_loss(y, p)), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": tp / (tp + fp) if tp + fp else 0., "recall": tp / (tp + fn),
        "false_positive_fraction": fp / (fp + tn), "threshold": threshold}


def gated(base, expert, p, low, threshold, blend):
    trigger = np.asarray(low) & (np.asarray(p) >= threshold)
    return np.asarray(base) + trigger * blend * (np.asarray(expert) - np.asarray(base)), trigger


def main():
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    assert not (OUT / "run.json").exists(), "Preserve previous run; create a versioned comparison."
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent = read_contract()
    old_path = ROOT / "Modeling/tables/m03/c/predictions.csv"
    assert sha(old_path) == c["parent_predictions_sha256"]
    old_verify = json.loads((ROOT / "Modeling/tables/m03/c/independent_verification.json").read_text(encoding="utf-8"))
    assert old_verify["status"] == "passed" and old_verify["run_sha256"] == sha(ROOT / "Modeling/tables/m03/c/run.json")
    old = read_csv(old_path, ["timestamp", "date"])
    old = old.loc[old.target.eq(c["target"]) & old.group.eq("B")].copy()
    old_models = read_csv(ROOT / "Modeling/tables/m03/ab/model_manifest.csv")
    f, base_features, features = build_frame(c)
    save(f[["timestamp", "prior_low", "rise_event", "power_transition"] + features], "feature_frame.csv")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    started = time.perf_counter()
    run = {"status": "running", "started": now(), "contract_sha256": sha(CONTRACT),
        "inputs_sha256": {"Modeling/tables/m01/hourly_frame.csv": c["frame_sha256"], parent["source"]: c["source_sha256"],
                          "Modeling/tables/m03/c/predictions.csv": c["parent_predictions_sha256"]},
        "features": {"B": base_features, "dynamic": features}, "no_july_august_evaluation": True,
        "runtime": {"executable": sys.executable, "version": sys.version}, "script_sha256": sha(Path(__file__))}
    (OUT / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    trials, audits, manifests, configs, predictions, inner_predictions, classification, counts = [], [], [], [], [], [], [], []
    fit_counter = {"inner_regression": 0, "inner_classifier": 0, "outer": 0}
    warning_messages = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for outer in parent["splits"]:
            split = outer["name"]
            if split not in c["outer_splits"]:
                continue
            tr = f.loc[f.timestamp.le(outer["train_end"])].copy()
            ev = f.loc[f.timestamp.between(outer["eval_start"], outer["eval_end"])].copy()
            template = old.loc[old.split.eq(split)].sort_values("timestamp").copy()
            assert template.timestamp.tolist() == ev.timestamp.tolist() and np.array_equal(template.actual, ev.target_maximum)
            for role, part in [("train", tr), ("eval", ev)]:
                event = part.loc[part.rise_event]
                counts.append({"split": split, "role": role, "hours": len(part), "prior_low_hours": int(part.prior_low.sum()),
                    "rise_hours": len(event), "rise_dates": event.date.nunique()})
            folds, inner_parts, inner_b, inner_d = [], [], [], []
            for month in c["inner_months"][split]:
                start = pd.Timestamp(2021, month, 1)
                t = tr.loc[tr.timestamp.lt(start)]
                v = tr.loc[tr.timestamp.between(start, start + pd.offsets.MonthBegin(1) - pd.Timedelta(hours=1))]
                assert t.timestamp.max() < v.timestamp.min() <= v.timestamp.max() < ev.timestamp.min()
                folds.append((t, v))
                audits.append({"split": split, "inner_month": month, "train_hours": len(t), "validation_hours": len(v),
                    "train_last": str(t.timestamp.max()), "validation_first": str(v.timestamp.min()), "validation_last": str(v.timestamp.max()),
                    "outer_first": str(ev.timestamp.min()), "train_low": int(t.prior_low.sum()), "train_rises": int(t.rise_event.sum()),
                    "validation_low": int(v.prior_low.sum()), "validation_rises": int(v.rise_event.sum())})
                for cols, container in [(base_features, inner_b), (features, inner_d)]:
                    estimator = HistGradientBoostingRegressor(**c["hgb_parameters"])
                    estimator.fit(t[cols], t.target_maximum)
                    container.append(estimator.predict(v[cols]))
                    fit_counter["inner_regression"] += 1
                inner_parts.append(v)
            iv = pd.concat(inner_parts)
            ib, idynamic = np.concatenate(inner_b), np.concatenate(inner_d)
            reference = stats(iv, ib)
            selected = {"B": ib, "D_dynamic": idynamic}
            def study(direction="minimize"):
                return optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
                                          pruner=optuna.pruners.NopPruner())
            wcache = {}
            def weighted_objective(trial):
                weight = trial.suggest_float("event_weight", *c["weighted"]["weight_range"], log=True)
                parts = []
                for t, v in folds:
                    estimator = HistGradientBoostingRegressor(**c["hgb_parameters"])
                    estimator.fit(t[features], t.target_maximum, sample_weight=np.where(t.rise_event, weight, 1.))
                    parts.append(estimator.predict(v[features]))
                    fit_counter["inner_regression"] += 1
                result = np.concatenate(parts)
                values = stats(iv, result)
                trial.set_user_attr("metrics", values)
                wcache[trial.number] = result
                return score(values, reference, c["development_selection"])
            ws = study()
            for weight in c["weighted"]["enqueue"]:
                ws.enqueue_trial({"event_weight": weight})
            ws.optimize(weighted_objective, n_trials=c["weighted"]["trials"])
            wt = min(ws.trials, key=lambda trial: (trial.value, trial.number))
            selected["W_rise_weighted"] = wcache[wt.number]
            split_config = {"split": split, "event_weight": wt.params["event_weight"], "weighted_best_trial": wt.number,
                            "weighted_inner_metrics": wt.user_attrs["metrics"], "classifiers": {}}
            def log_trials(study_object, family):
                for trial in study_object.trials:
                    trials.append({"split": split, "family": family, "trial": trial.number, "state": trial.state.name,
                        "objective": trial.value, "parameters": json.dumps(trial.params),
                        "metrics": json.dumps(trial.user_attrs), "seconds": trial.duration.total_seconds()})
            log_trials(ws, "weighted")
            parent_row = old_models.loc[old_models.split.eq(split) & old_models.target.eq(c["target"]) & old_models.group.eq("B")].iloc[0]
            parent_path = ROOT / parent_row.file
            assert sha(parent_path) == parent_row.sha256
            bmodel = joblib.load(parent_path)
            assert np.allclose(bmodel.predict(ev[base_features]), template.prediction, atol=1e-9, rtol=0)
            outer_predictions = {"B": template.prediction.to_numpy()}
            for name, weight in [("D_dynamic", 1.), ("W_rise_weighted", wt.params["event_weight"])]:
                estimator = HistGradientBoostingRegressor(**c["hgb_parameters"])
                estimator.fit(tr[features], tr.target_maximum, sample_weight=None if weight == 1 else np.where(tr.rise_event, weight, 1.))
                predicted = estimator.predict(ev[features])
                path = MODELS / f"{split}_{name}.joblib"
                joblib.dump(estimator, path, compress=3)
                assert np.allclose(joblib.load(path).predict(ev[features]), predicted, atol=1e-9, rtol=0)
                manifests.append({"split": split, "name": name, "file": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "features": 27})
                outer_predictions[name] = predicted
                fit_counter["outer"] += 1
            for family in ["Logistic", "HGB"]:
                pcache = {}
                def classifier_objective(trial):
                    if family == "Logistic":
                        params = {"C": trial.suggest_float("C", .01, 100., log=True)}
                    else:
                        params = {"max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes", [3,7,15]),
                            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [10,20,40]),
                            "l2_regularization": trial.suggest_float("l2_regularization", .1, 10., log=True)}
                    parts, constants = [], 0
                    for t, v in folds:
                        estimator, prob, constant = fit_probability(family, params, t, v, features)
                        parts.append(prob)
                        constants += int(constant)
                        fit_counter["inner_classifier"] += int(not constant)
                    prob = np.concatenate(parts)
                    values = alarm_stats(iv, prob, .5)
                    trial.set_user_attr("classification", values)
                    trial.set_user_attr("constant_folds", constants)
                    pcache[trial.number] = prob
                    return values["AP"]
                cs = study("maximize")
                cs.optimize(classifier_objective, n_trials=c["classifier"]["trials_per_family"])
                ct = min(cs.trials, key=lambda trial: (-trial.value, trial.user_attrs["classification"]["Brier"], trial.number))
                ip = pcache[ct.number]
                def gate_objective(trial):
                    threshold = trial.suggest_float("threshold", *c["gate"]["threshold"])
                    blend = trial.suggest_float("blend", *c["gate"]["blend"])
                    pred, trigger = gated(ib, selected["W_rise_weighted"], ip, iv.prior_low, threshold, blend)
                    values = stats(iv, pred)
                    alarm = alarm_stats(iv, ip, threshold)
                    trial.set_user_attr("metrics", values)
                    trial.set_user_attr("classification", alarm)
                    return score(values, reference, c["development_selection"], alarm["false_positive_fraction"])
                gs = study()
                for params in c["gate"]["enqueue"]:
                    gs.enqueue_trial(params)
                gs.optimize(gate_objective, n_trials=c["gate"]["trials_per_family"])
                gt = min(gs.trials, key=lambda trial: (trial.value, trial.number))
                name = "G_" + family
                selected[name], _ = gated(ib, selected["W_rise_weighted"], ip, iv.prior_low, **gt.params)
                estimator, op, constant = fit_probability(family, ct.params, tr, ev, features)
                path = MODELS / f"{split}_{family}_classifier.joblib"
                joblib.dump(estimator, path, compress=3)
                assert np.allclose(probability(joblib.load(path), ev, features), op, atol=1e-9, rtol=0)
                manifests.append({"split": split, "name": family + "_classifier", "file": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "features": 27})
                fit_counter["outer"] += int(not constant)
                outer_predictions[name], trigger = gated(outer_predictions["B"], outer_predictions["W_rise_weighted"], op, ev.prior_low, **gt.params)
                split_config["classifiers"][family] = {"parameters": ct.params, "classifier_best_trial": ct.number,
                    "classifier_inner": ct.user_attrs["classification"], "gate": gt.params, "gate_best_trial": gt.number,
                    "gate_inner": gt.user_attrs, "constant_outer": constant}
                log_trials(cs, "classifier_" + family)
                log_trials(gs, "gate_" + family)
                for role, part, prob in [("inner", iv, ip), ("outer", ev, op)]:
                    classification.append({"split": split, "role": role, "model": name,
                                           **alarm_stats(part, prob, gt.params["threshold"])})
                selected[name + "_probability"] = ip
                outer_predictions[name + "_probability"] = op
                outer_predictions[name + "_trigger"] = trigger
                print(f"{split} {name}: inner AP={ct.value:.3f}, gate={gt.params}", flush=True)
            for name in c["methods"]:
                record = template.copy()
                record["group"] = name
                record["prediction"] = outer_predictions[name]
                record["prior_low"] = ev.prior_low.to_numpy()
                record["rise_event"] = ev.rise_event.to_numpy()
                record["prior_state"] = ev.prior_state.to_numpy()
                record["probability"] = outer_predictions.get(name + "_probability", np.full(len(ev), np.nan))
                record["trigger"] = outer_predictions.get(name + "_trigger", np.zeros(len(ev), dtype=bool))
                predictions.append(record)
                record_inner = iv[["timestamp", "date", "prior_low", "rise_event", "power_transition"]].copy()
                record_inner["split"], record_inner["group"] = split, name
                record_inner["actual"] = iv.target_maximum.to_numpy()
                record_inner["prediction"] = selected[name]
                record_inner["daily_maximum_weight"] = peak_weights(iv)
                record_inner["probability"] = selected.get(name + "_probability", np.full(len(iv), np.nan))
                inner_predictions.append(record_inner)
            configs.append(split_config)
            save(pd.DataFrame(trials), "trials.csv")
            save(pd.DataFrame(audits), "fold_audit.csv")
            (OUT / "selected_configurations.json").write_text(json.dumps(configs, indent=2), encoding="utf-8")
            print(f"completed {split}: {len(trials)} cumulative trials", flush=True)
        warning_messages = sorted(set(str(w.message) for w in caught))
    p = pd.concat(predictions, ignore_index=True)
    p["signed_error"] = p.prediction - p.actual
    p["absolute_error"] = p.signed_error.abs()
    p["under_amount"] = (-p.signed_error).clip(lower=0)
    p["over_amount"] = p.signed_error.clip(lower=0)
    assert p.timestamp.max() < pd.Timestamp("2021-07-01") and np.isfinite(p.prediction).all()
    save(p, "predictions.csv")
    save(pd.concat(inner_predictions, ignore_index=True), "inner_predictions.csv")
    save(pd.DataFrame(manifests), "model_manifest.csv")
    save(pd.DataFrame(classification), "classification.csv")
    save(pd.DataFrame(counts), "sample_counts.csv")
    metrics, selection_rows, paired, daily = [], [], [], []
    reference_pred = p.loc[p.group.eq("B")]
    reference_frame = f.loc[f.timestamp.isin(reference_pred.timestamp)].sort_values("timestamp")
    reference_stats = stats(reference_frame, reference_pred.sort_values("timestamp").prediction)
    for name, part in p.groupby("group"):
        for kind, value, subset, weights in subsets(part):
            metrics.append({"group": name, "kind": kind, "value": value, **metric(subset, weights)})
        values = stats(reference_frame, part.sort_values("timestamp").prediction)
        row = {"group": name, **values, "balanced_score": max(values[k] / reference_stats[k] for k in ["overall_MAE", "daily_maximum_MAE", "rise_MAE"])}
        requirements = []
        for key, setting in [("overall_MAE", "overall_MAE_ratio_max"), ("daily_maximum_MAE", "daily_maximum_MAE_ratio_max"),
            ("low_stay_MAE", "low_stay_MAE_ratio_max"), ("low_stay_over", "low_stay_mean_over_ratio_max"),
            ("rise_MAE", "rise_MAE_ratio_max"), ("rise_under", "rise_mean_under_ratio_max")]:
            row[key + "_ratio"] = values[key] / reference_stats[key]
            row[key + "_passed"] = bool(row[key + "_ratio"] <= c["development_selection"][setting] + 1e-12)
            requirements.append(row[key + "_passed"])
        month_ok = True
        for split, month_part in part.groupby("split"):
            target_rise = month_part.loc[month_part.rise_event]
            old_rise = reference_pred.loc[reference_pred.split.eq(split) & reference_pred.rise_event]
            month_ok &= target_rise.absolute_error.mean() <= old_rise.absolute_error.mean() * c["development_selection"]["rise_monthly_MAE_ratio_max"] + 1e-9
        row["rise_monthly_passed"] = bool(month_ok)
        requirements.append(bool(month_ok))
        if name.startswith("G_"):
            alarm = alarm_stats(reference_frame, part.sort_values("timestamp").probability, .5)
            # Threshold differs by split; use actual stored trigger for pooled confusion counts.
            low = part.loc[part.prior_low]
            fp = int((low.trigger & ~low.rise_event).sum())
            tn = int((~low.trigger & ~low.rise_event).sum())
            row["gate_false_positive_fraction"] = fp / (fp + tn)
            row["gate_false_positive_passed"] = bool(row["gate_false_positive_fraction"] <= .05 + 1e-12)
            requirements.append(row["gate_false_positive_passed"])
        row["eligible"] = all(requirements) and name != "B"
        selection_rows.append(row)
        q = part[["timestamp", "date", "split", "power_transition", "prediction", "absolute_error", "under_amount", "over_amount"]].merge(
            reference_pred[["timestamp", "prediction", "absolute_error", "under_amount", "over_amount"]], on="timestamp", suffixes=("", "_B"), validate="one_to_one")
        q["group"] = name
        for key, label in [("absolute_error", "abs"), ("under_amount", "under"), ("over_amount", "over")]:
            q["delta_" + label] = q[key] - q[key + "_B"]
        paired.append(q)
        d = q.groupby(["split", "date"], as_index=False).agg(hours=("timestamp", "size"), delta_MAE=("delta_abs", "mean"), delta_under=("delta_under", "mean"), delta_over=("delta_over", "mean"))
        d["group"] = name
        daily.append(d)
    table = pd.DataFrame(selection_rows)
    eligible = table.loc[table.eligible].copy()
    if len(eligible):
        near = eligible.loc[eligible.balanced_score.le(eligible.balanced_score.min() * 1.01)].copy()
        near["preference"] = near.group.map({name: i for i, name in enumerate(c["methods"])})
        chosen = near.sort_values(["preference", "balanced_score"]).iloc[0].group
    else:
        chosen = "B"
    selection = {"selected": chosen, "eligible_methods": eligible.group.tolist(), "development_only": True,
                 "rule": c["development_selection"], "no_july_august_evaluation": True}
    save(pd.DataFrame(metrics), "metrics.csv")
    save(table, "selection_metrics.csv")
    save(pd.concat(paired, ignore_index=True), "paired_predictions.csv")
    save(pd.concat(daily, ignore_index=True), "daily_errors.csv")
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    run.update({"status": "completed", "finished": now(), "seconds": time.perf_counter() - started,
        "fits": fit_counter, "trials": len(trials), "prediction_rows": len(p), "metric_rows": len(metrics),
        "warnings": warning_messages, "outputs_sha256": {path.name: sha(path) for path in OUT.iterdir() if path.name != "run.json"},
        "helper_sha256": {name: sha(ROOT / "Modeling/scripts" / name) for name in ["m01_prepare.py", "m02_compare.py", "m02_rerun.py", "m04_compare.py"]}})
    (OUT / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(table[["group", "overall_MAE", "daily_maximum_MAE", "rise_MAE", "rise_under", "low_stay_MAE", "low_stay_over", "eligible"]].to_string(index=False), flush=True)
    print(json.dumps(selection), flush=True)


if __name__ == "__main__":
    main()
