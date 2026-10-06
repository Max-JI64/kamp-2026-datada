"""M02-2: nested expanding-month Optuna and honest outer-month ensembles."""
import json
import os
import sys
import time
from pathlib import Path

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(key, "4")
import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from m01_prepare import ROOT, feature_columns, read_contract, sha
from m02_compare import metric
from m03_ablation import now, subsets

OUT = ROOT / "Modeling/tables/m022"
MODELS = ROOT / "Modeling/models/m022"
CONTRACT = ROOT / "Modeling/config/m022_contract.json"


def save(df, name, directory=OUT):
    df.to_csv(directory / f"{name}.csv", index=False, encoding="utf-8-sig")


def suggest(trial, family, c):
    result = {}
    categorical = {"max_leaf_nodes", "num_leaves", "max_depth"} if family == "ExtraTrees" else {"max_leaf_nodes", "num_leaves"}
    integers = {"max_iter", "n_estimators", "iterations", "max_depth", "depth", "min_samples_leaf", "min_child_samples"}
    for name, spec in c["search_spaces"][family].items():
        if name in categorical:
            result[name] = trial.suggest_categorical(name, spec)
        elif name in integers:
            result[name] = trial.suggest_int(name, spec[0], spec[1], step=spec[2] if len(spec) == 3 else 1)
        else:
            result[name] = trial.suggest_float(name, spec[0], spec[1], log=len(spec) == 3 and spec[2] == "log")
    return result


def make_model(family, params):
    if family == "HGB":
        return HistGradientBoostingRegressor(**params, early_stopping=False, random_state=42)
    if family == "XGBoost":
        return XGBRegressor(**params, objective="reg:squarederror", tree_method="hist", random_state=42, n_jobs=4)
    if family == "LightGBM":
        return LGBMRegressor(**params, objective="regression", subsample_freq=1, random_state=42, n_jobs=4,
                             deterministic=True, force_col_wise=True, verbosity=-1)
    if family == "CatBoost":
        return CatBoostRegressor(**params, loss_function="RMSE", random_seed=42, thread_count=4,
                                 verbose=False, allow_writing_files=False, task_type="CPU")
    if family == "ExtraTrees":
        return ExtraTreesRegressor(**params, random_state=42, n_jobs=4, bootstrap=False)
    raise ValueError(family)


def study_rows(study, split, family):
    return [{"split": split, "family": family, "trial": t.number, "state": t.state.name,
             "value": t.value, "parameters": json.dumps(t.params), "seconds": t.duration.total_seconds() if t.duration else None,
             "best_so_far": min(x.value for x in study.trials[:t.number+1] if x.value is not None)} for t in study.trials]


def summarize(p, directory=OUT):
    comparison, conditions, daily = [], [], []
    for (target, model), g in p.groupby(["target", "model"]):
        comparison.append({"target": target, "model": model, "split": "pooled_development", **metric(g)})
        comparison.extend({"target": target, "model": model, "split": s, **metric(part)} for s, part in g.groupby("split"))
        for kind, value, part, w in subsets(g):
            if len(part):
                conditions.append({"target": target, "model": model, "condition": kind, "value": value, **metric(part, w),
                                   "share_total_absolute_error_unweighted": float(part.absolute_error.sum() / g.absolute_error.sum())})
        daily.extend({"target": target, "model": model, "date": date, **metric(part)} for date, part in g.groupby("date"))
    comparison = pd.DataFrame(comparison)
    save(comparison, "comparison", directory)
    save(pd.DataFrame(conditions), "condition_errors", directory)
    save(pd.DataFrame(daily), "daily_errors", directory)
    return comparison


def add_errors(p):
    p["signed_error"] = p.prediction - p.actual
    p["absolute_error"] = p.signed_error.abs()
    p["under_amount"] = (-p.signed_error).clip(lower=0)
    p["over_amount"] = p.signed_error.clip(lower=0)
    return p


def main():
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent = read_contract()
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    v1 = json.loads((ROOT / "Modeling/tables/m01/verification.json").read_text(encoding="utf-8"))
    olddir = ROOT / "Modeling/tables/m03/c"
    oldrun = json.loads((olddir / "run.json").read_text(encoding="utf-8"))
    oldverify = json.loads((olddir / "independent_verification.json").read_text(encoding="utf-8"))
    assert oldverify["status"] == "passed" and oldverify["run_sha256"] == sha(olddir / "run.json")
    assert sha(ROOT / parent["source"]) == parent["source_sha256"]
    inputs = {"Modeling/config/m022_contract.json": sha(CONTRACT), parent["source"]: sha(ROOT / parent["source"])}
    for rel, expected in [("Modeling/tables/m01/hourly_frame.csv", v1["outputs_sha256"]["hourly_frame.csv"]),
                          ("Modeling/tables/m03/c/predictions.csv", oldrun["outputs_sha256"]["predictions.csv"])]:
        assert sha(ROOT / rel) == expected
        inputs[rel] = expected
    f = pd.read_csv(ROOT / "Modeling/tables/m01/hourly_frame.csv", encoding="utf-8-sig", float_precision="round_trip", parse_dates=["timestamp"])
    f = f.loc[f.eligible_common & f.timestamp.lt("2021-07-01")].copy()
    old = pd.read_csv(olddir / "predictions.csv", encoding="utf-8-sig", float_precision="round_trip", parse_dates=["timestamp"])
    old = old.loc[old.group.eq("B") & old.target.eq(c["target"])].copy()
    features = feature_columns("B", parent)
    assert len(features) == 18 and not any(x.startswith(("target_", "diag_")) for x in features)
    run = {"status": "running", "started": now(), "input_hashes": inputs, "features": features,
           "environment": json.loads((OUT / "environment.json").read_text(encoding="utf-8")),
           "july_august_evaluated": False, "selection_scope": "outer development family choice; not final test"}
    (OUT / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    all_predictions, trials, fold_scores, audit, configs, manifest, inner_predictions, correlations = [], [], [], [], [], [], [], []
    started = time.perf_counter()
    for outer in parent["splits"]:
        split = outer["name"]
        if split not in c["outer_splits"]:
            continue
        tr = f.loc[f.timestamp.le(outer["train_end"])]
        ev = f.loc[f.timestamp.between(outer["eval_start"], outer["eval_end"])]
        base = old.loc[old.split.eq(split)].copy().drop(columns=["group"])
        assert len(ev) == len(base) and base.timestamp.tolist() == ev.timestamp.tolist()
        assert np.array_equal(base.actual, ev[c["target"]]) and tr.timestamp.max() < ev.timestamp.min()
        for name, values in [("HGB_B_m03", base.prediction.to_numpy()), ("previous_hour", ev.lag1_maximum.to_numpy()),
                             ("previous_day", ev.lag24_maximum.to_numpy()), ("previous_week", ev.lag168_maximum.to_numpy())]:
            record = base.copy()
            record["model"], record["prediction"] = name, values
            all_predictions.append(record)
        folds = []
        for month in c["inner_validation_months"][split]:
            start = pd.Timestamp(2021, month, 1)
            end = start + pd.offsets.MonthBegin(1)
            train = tr.loc[tr.timestamp.lt(start)]
            val = tr.loc[tr.timestamp.ge(start) & tr.timestamp.lt(end)]
            assert len(train) > 300 and len(val) > 300
            assert train.timestamp.max() < val.timestamp.min() and val.timestamp.max() < ev.timestamp.min()
            folds.append((month, train, val))
            audit.append({"outer_split": split, "kind": "inner", "month": month, "train_hours": len(train), "validation_hours": len(val),
                          "train_last": str(train.timestamp.max()), "validation_first": str(val.timestamp.min()), "validation_last": str(val.timestamp.max()),
                          "outer_evaluation_first": str(ev.timestamp.min())})
        chosen_inner, chosen_outer, inner_maes = {}, {}, {}
        inner_actual = np.concatenate([val[c["target"]].to_numpy() for _, _, val in folds])
        inner_times = np.concatenate([val.timestamp.to_numpy() for _, _, val in folds])
        for family in c["families"]:
            cache = {}
            def objective(trial):
                params = suggest(trial, family, c)
                predictions = []
                for month, train, val in folds:
                    model = make_model(family, params)
                    model.fit(train[features], train[c["target"]])
                    predicted = model.predict(val[features])
                    assert np.isfinite(predicted).all()
                    error = np.abs(predicted - val[c["target"]].to_numpy())
                    fold_scores.append({"split": split, "family": family, "trial": trial.number, "month": month,
                                        "train_hours": len(train), "validation_hours": len(val), "MAE": float(error.mean()), "absolute_error_sum": float(error.sum())})
                    predictions.append(predicted)
                pred = np.concatenate(predictions)
                value = float(np.mean(np.abs(pred - inner_actual)))
                cache[trial.number] = pred
                return value
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10), pruner=optuna.pruners.NopPruner())
            if family == "HGB":
                study.enqueue_trial({"max_leaf_nodes": 15, "max_iter": 150, "learning_rate": 0.05, "min_samples_leaf": 20, "l2_regularization": 1.0})
            study.optimize(objective, n_trials=c["optuna"]["base_trials_per_study"], n_jobs=1)
            first20 = min(t.value for t in study.trials[:20])
            extended = study.best_value <= first20 * .99
            if extended:
                study.optimize(objective, n_trials=c["optuna"]["max_trials_per_study"] - len(study.trials), n_jobs=1)
            params = study.best_params
            model = make_model(family, params)
            model.fit(tr[features], tr[c["target"]])
            pred = model.predict(ev[features])
            model_path = MODELS / f"{split}_{family}.joblib"
            joblib.dump(model, model_path, compress=3)
            assert np.allclose(joblib.load(model_path).predict(ev[features]), pred, atol=1e-9, rtol=0)
            manifest.append({"split": split, "model": family, "file": str(model_path.relative_to(ROOT)), "sha256": sha(model_path), "reload_predictions_checked": True})
            chosen_inner[family], chosen_outer[family] = cache[study.best_trial.number], pred
            inner_maes[family] = study.best_value
            inner_predictions.append(pd.DataFrame({"outer_split": split, "model": family, "timestamp": inner_times, "actual": inner_actual, "prediction": chosen_inner[family]}))
            configs.append({"split": split, "model": family, "best_trial": study.best_trial.number, "trials": len(study.trials), "extended": extended,
                            "inner_MAE": study.best_value, "parameters": params, "train_hours": len(tr), "eval_hours": len(ev)})
            trials.extend(study_rows(study, split, family))
            record = base.copy()
            record["model"], record["prediction"] = family, pred
            all_predictions.append(record)
            save(pd.DataFrame(trials), "trials")
            save(pd.DataFrame(fold_scores), "trial_fold_scores")
            (OUT / "selected_configurations.json").write_text(json.dumps(configs, indent=2), encoding="utf-8")
            print(f"{split} {family}: trials={len(study.trials)} inner_MAE={study.best_value:.4f} outer_MAE={np.mean(np.abs(pred-base.actual)):.4f}", flush=True)
        mean_pred = np.mean(list(chosen_outer.values()), axis=0)
        record = base.copy()
        record["model"], record["prediction"] = "mean5", mean_pred
        all_predictions.append(record)
        configs.append({"split": split, "model": "mean5", "members": c["families"], "weights": [0.2] * 5})
        members = sorted(inner_maes, key=inner_maes.get)[:3]
        matrix = np.column_stack([chosen_inner[x] for x in members])
        def weight_objective(trial):
            w = np.array([trial.suggest_float(f"weight_{i}", 0, 1) for i in range(3)])
            w = w / w.sum() if w.sum() else np.ones(3) / 3
            return float(np.mean(np.abs(matrix @ w - inner_actual)))
        ws = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.NopPruner())
        ws.enqueue_trial({f"weight_{i}": 1/3 for i in range(3)})
        for member in range(3):
            ws.enqueue_trial({f"weight_{i}": float(i == member) for i in range(3)})
        ws.optimize(weight_objective, n_trials=50, n_jobs=1)
        w = np.array([ws.best_params[f"weight_{i}"] for i in range(3)])
        w = w / w.sum() if w.sum() else np.ones(3) / 3
        pred = np.column_stack([chosen_outer[x] for x in members]) @ w
        record = base.copy()
        record["model"], record["prediction"] = "weighted_top3", pred
        all_predictions.append(record)
        configs.append({"split": split, "model": "weighted_top3", "members": members, "weights": w.tolist(), "inner_MAE": ws.best_value})
        trials.extend(study_rows(ws, split, "weighted_top3"))
        corr = pd.DataFrame({name: values - inner_actual for name, values in chosen_inner.items()}).corr()
        for i, a in enumerate(c["families"]):
            for b in c["families"][i+1:]:
                correlations.append({"split": split, "model_a": a, "model_b": b, "inner_residual_correlation": corr.loc[a, b]})
        save(add_errors(pd.concat(all_predictions, ignore_index=True)), "predictions")
        save(pd.DataFrame(trials), "trials")
        save(pd.DataFrame(audit), "fold_audit")
        save(pd.DataFrame(manifest), "model_manifest")
        save(pd.concat(inner_predictions, ignore_index=True), "inner_selected_predictions")
        save(pd.DataFrame(correlations), "inner_residual_correlations")
        (OUT / "selected_configurations.json").write_text(json.dumps(configs, indent=2), encoding="utf-8")
    p = add_errors(pd.concat(all_predictions, ignore_index=True))
    assert p.timestamp.max() < pd.Timestamp("2021-07-01")
    comparison = summarize(p)
    pooled = comparison.loc[comparison.split.eq("pooled_development")].copy()
    near = pooled.loc[pooled.MAE.le(pooled.MAE.min() * 1.01)].copy()
    preference = {"HGB_B_m03": 0, "HGB": 1, "XGBoost": 2, "LightGBM": 3, "CatBoost": 4, "ExtraTrees": 5, "mean5": 6, "weighted_top3": 7,
                  "previous_hour": 0, "previous_day": 0, "previous_week": 0}
    near["preference"] = near.model.map(preference)
    selected = near.sort_values(["preference", "MAE"]).iloc[0].to_dict()
    oldmae = float(pooled.loc[pooled.model.eq("HGB_B_m03"), "MAE"].iloc[0])
    selection = {"selected": selected, "point_best": pooled.sort_values("MAE").iloc[0].to_dict(),
                 "old_HGB_B_MAE": oldmae, "relative_MAE_improvement": 1 - selected["MAE"] / oldmae,
                 "M03_repeat_required": selected["model"] != "HGB_B_m03", "development_only": True}
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    run.update({"status": "completed", "finished": now(), "seconds": time.perf_counter()-started,
                "tuning_fits": len(fold_scores), "outer_model_fits": len(manifest), "trial_count_including_ensemble": len(trials),
                "prediction_rows": len(p), "outer_hours_per_method": 2184,
                "outputs_sha256": {x.name: sha(x) for x in OUT.glob("*.csv")},
                "configuration_sha256": sha(OUT / "selected_configurations.json"), "selection_sha256": sha(OUT / "selection.json"),
                "script_sha256": sha(Path(__file__))})
    assert inputs["Modeling/config/m022_contract.json"] == sha(CONTRACT)
    (OUT / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(pooled[["model", "MAE", "RMSE", "mean_under", "mean_over"]].sort_values("MAE").to_string(index=False), flush=True)
    print(json.dumps(selection), flush=True)


if __name__ == "__main__":
    main()
