"""M02: versioned development comparison and first M04 error diagnostics.

Regular CPython 3.13; run from project root. No July/August evaluation.
Only A features. Standardization is fit on each training split.
"""
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Bound numerical worker counts before importing numerical libraries.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from m01_prepare import ROOT, feature_columns, read_contract, sha

CONTRACT = ROOT / "Modeling/config/m02_contract.json"
OUT = ROOT / "Modeling/tables/m02"
MODELS = ROOT / "Modeling/models/m02"


def candidates(c):
    classes = {"Ridge": Ridge, "ElasticNet": ElasticNet, "SVR": SVR, "HGB": HistGradientBoostingRegressor}
    result = []
    for family, spec in c["grids"].items():
        varying = {k: v for k, v in spec.items() if isinstance(v, list)}
        fixed = {k: v for k, v in spec.items() if not isinstance(v, list)}
        for order, values in enumerate(itertools.product(*varying.values())):
            params = {**fixed, **dict(zip(varying, values))}
            estimator = classes[family](**params)
            if family != "HGB":
                estimator = make_pipeline(StandardScaler(), estimator)
            if family in ("ElasticNet", "SVR"):
                estimator = TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler())
            result.append((f"{family.lower()}_{order:02d}", family, order, params, estimator))
    return result


def state(maximum, rounded):
    return np.select([maximum.eq(0), rounded.lt(20), rounded.between(20, 26)],
                     ["zero", "below20", "low"], default="above26")


def metric(g, weights=None):
    w = np.ones(len(g)) if weights is None else np.asarray(weights, dtype=float)
    assert len(g) and w.sum() > 0 and np.isfinite(w).all()
    error = g.prediction.to_numpy() - g.actual.to_numpy()
    absolute = np.abs(error)
    under, over = np.maximum(-error, 0), np.maximum(error, 0)
    return {"hours": len(g), "weight_sum": float(w.sum()),
            "MAE": float(np.average(absolute, weights=w)),
            "RMSE": float(np.sqrt(np.average(error ** 2, weights=w))),
            "bias": float(np.average(error, weights=w)),
            "mean_under": float(np.average(under, weights=w)),
            "mean_over": float(np.average(over, weights=w)),
            "under_fraction": float(np.average(error < 0, weights=w)),
            "over_fraction": float(np.average(error > 0, weights=w)),
            "under_when_present": float(np.average(under[under > 0], weights=w[under > 0])) if (under > 0).any() else 0.0,
            "over_when_present": float(np.average(over[over > 0], weights=w[over > 0])) if (over > 0).any() else 0.0,
            "absolute_error_sum": float(absolute.sum()),
            "mean_actual": float(np.average(g.actual, weights=w)),
            "negative_prediction_hours": int((g.prediction < 0).sum())}


def save(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")


def main():
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent = read_contract()
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    verification = json.loads((ROOT / "Modeling/tables/m01/verification.json").read_text(encoding="utf-8"))
    assert sha(ROOT / parent["source"]) == parent["source_sha256"]
    files = ["hourly_frame.csv", "evaluation_diagnostics.csv", "split_counts.csv"]
    for name in files:
        assert sha(ROOT / "Modeling/tables/m01" / name) == verification["outputs_sha256"][name]
    f = pd.read_csv(ROOT / "Modeling/tables/m01/hourly_frame.csv", encoding="utf-8-sig", float_precision="round_trip", parse_dates=["timestamp", "date"])
    diagnostics = pd.read_csv(ROOT / "Modeling/tables/m01/evaluation_diagnostics.csv", encoding="utf-8-sig", parse_dates=["timestamp", "date"])
    expected = pd.read_csv(ROOT / "Modeling/tables/m01/split_counts.csv", encoding="utf-8-sig")
    features = feature_columns("A", parent)
    assert "lag1_last" not in features and not any(x.startswith(("target_", "diag_")) for x in features)
    f["power_state"] = state(f.target_maximum, f.diag_target_rounded_mean)
    # M01 verified integer half-up averages, so rounding the lag arithmetic mean
    # reproduces the prior CSV 'average' used by A06.
    f["prior_power_state"] = state(f.lag1_maximum, np.floor(f.lag1_mean + 0.5))
    f["power_transition"] = f.prior_power_state + "->" + f.power_state
    specs = candidates(c)
    descriptions = [{"model": name, "family": family, "order": order, "parameters": json.dumps(params, ensure_ascii=False)} for name, family, order, params, _ in specs]
    descriptions += [{"model": name, "family": "Baseline", "order": i, "parameters": json.dumps({"lag": lag})} for i, (name, lag) in enumerate(c["baselines"].items())]
    save(pd.DataFrame(descriptions), "candidates")
    inputs = {"m01_contract": sha(ROOT / c["parent"]), "m02_contract": sha(CONTRACT), "source": sha(ROOT / parent["source"]),
              **{name: sha(ROOT / "Modeling/tables/m01" / name) for name in files}}
    run = {"started": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"), "status": "running", "input_hashes": inputs,
           "features": features, "candidate_settings": len(specs), "planned_fits": len(specs) * len(c["targets"]) * len(c["splits"]),
           "runtime": {"executable": sys.executable, "version": sys.version, "platform": platform.platform(),
                       "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "joblib"]}}}
    (OUT / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions, audits, fitted = [], [], {}
    total_started = time.perf_counter()
    for split in parent["splits"]:
        if split["name"] not in c["splits"]:
            continue
        train = f.loc[f.eligible_common & (f.timestamp <= pd.Timestamp(split["train_end"]))]
        ev = f.loc[f.eligible_common & f.timestamp.between(pd.Timestamp(split["eval_start"]), pd.Timestamp(split["eval_end"]))]
        expect = expected.loc[expected.split.eq(split["name"]) & expected.pool.eq("common")].iloc[0]
        assert len(train) == expect.train_hours and len(ev) == expect.eval_hours
        assert train.timestamp.max() < ev.timestamp.min()
        assert train[features].notna().all().all() and ev[features].notna().all().all()
        diag = diagnostics.loc[diagnostics.split.eq(split["name"]) & diagnostics.pool.eq("common")]
        cols = ["timestamp", "date", "month", "weekend", "power_transition", "diag_production_transition", "diag_target_profile"]
        base = ev[cols].merge(diag[["timestamp", "train_profile_overlap", "profile_weight", "daily_maximum_weight"]], on="timestamp", validate="one_to_one")
        base["hour"] = base.timestamp.dt.hour
        assert base.timestamp.tolist() == ev.timestamp.tolist()
        for target in c["targets"]:
            for name, lag in c["baselines"].items():
                record = base.copy()
                record["split"], record["target"], record["model"], record["family"] = split["name"], target, name, "Baseline"
                record["actual"] = ev[target].to_numpy()
                metric_name = "maximum" if target == "target_maximum" else "mean"
                record["prediction"] = ev[f"lag{lag}_{metric_name}"].to_numpy()
                predictions.append(record)
            for name, family, order, params, estimator in candidates(c):
                start = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    estimator.fit(train[features], train[target])
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    raise RuntimeError(f"Convergence failure: {name}, {split['name']}: {[str(w.message) for w in caught]}")
                predicted = estimator.predict(ev[features])
                assert np.isfinite(predicted).all()
                # Directly verify fitted scaler statistics against training rows.
                reg = estimator.regressor_ if isinstance(estimator, TransformedTargetRegressor) else estimator
                if family != "HGB":
                    scaler = reg.steps[0][1]
                    assert scaler.n_samples_seen_ == len(train)
                    assert np.allclose(scaler.mean_, train[features].mean().to_numpy())
                if isinstance(estimator, TransformedTargetRegressor):
                    assert np.allclose(estimator.transformer_.mean_, [train[target].mean()])
                    assert estimator.transformer_.n_samples_seen_ == len(train)
                elapsed = time.perf_counter() - start
                record = base.copy()
                record["split"], record["target"], record["model"], record["family"] = split["name"], target, name, family
                record["actual"], record["prediction"] = ev[target].to_numpy(), predicted
                predictions.append(record)
                audits.append({"split": split["name"], "target": target, "model": name, "family": family, "train_hours": len(train), "eval_hours": len(ev),
                               "train_last_timestamp": str(train.timestamp.max()), "eval_first_timestamp": str(ev.timestamp.min()),
                               "seconds": elapsed, "warnings": json.dumps([str(w.message) for w in caught]),
                               "training_only_scalers_verified": True, "minimum_prediction": float(predicted.min())})
                fitted[(split["name"], target, name)] = estimator
            print(f"completed {split['name']} {target}: {len(specs)} fits; {len(ev)} evaluation hours", flush=True)
    p = pd.concat(predictions, ignore_index=True)
    p["signed_error"] = p.prediction - p.actual
    p["absolute_error"] = p.signed_error.abs()
    p["under_amount"] = (-p.signed_error).clip(lower=0)
    p["over_amount"] = p.signed_error.clip(lower=0)
    save(p, "predictions")
    save(pd.DataFrame(audits), "training_audit")
    scores = []
    for (target, model, family), g in p.groupby(["target", "model", "family"]):
        scores.append({"target": target, "model": model, "family": family, "split": "pooled_development", **metric(g)})
        for split, part in g.groupby("split"):
            scores.append({"target": target, "model": model, "family": family, "split": split, **metric(part)})
    scores = pd.DataFrame(scores)
    save(scores, "all_candidate_metrics")
    selected = []
    order_map = {name: order for name, _, order, _, _ in specs}
    for (target, family), group in scores.loc[scores.split.eq("pooled_development")].groupby(["target", "family"]):
        if family == "Baseline":
            chosen = group.sort_values(["MAE", "model"]).iloc[0]
        else:
            near = group.loc[group.MAE <= group.MAE.min() * 1.01].copy()
            near["preference"] = near.model.map(order_map)
            chosen = near.sort_values("preference").iloc[0]
        selected.append({**chosen.to_dict(), "family_best_MAE": float(group.MAE.min())})
    selected = pd.DataFrame(selected)
    save(selected, "family_selection")
    # Retain all three baselines in readable comparison and condition tables.
    selected_pairs = set(zip(selected.target, selected.model))
    q = p.loc[[family == "Baseline" or (target, model) in selected_pairs for target, model, family in zip(p.target, p.model, p.family)]].copy()
    save(q, "selected_predictions")
    comparison = scores.loc[[family == "Baseline" or (target, model) in selected_pairs for target, model, family in zip(scores.target, scores.model, scores.family)]]
    save(comparison, "comparison")
    condition_rows = []
    for (target, model, family), g in q.groupby(["target", "model", "family"]):
        total_error = g.absolute_error.sum()
        subsets = [("all", "all", g, None), ("profile_reweighted", "all", g, g.profile_weight)]
        peak = g.loc[g.daily_maximum_weight > 0]
        subsets.append(("daily_maximum", "all", peak, peak.daily_maximum_weight))
        for column in ["split", "month", "weekend", "hour", "power_transition", "diag_production_transition", "train_profile_overlap"]:
            for value, part in g.groupby(column):
                subsets.append((column, str(value), part, None))
        for kind, label, subset, weights in subsets:
            if subset.empty:
                continue
            condition_rows.append({"target": target, "model": model, "family": family, "condition": kind, "value": label,
                                   **metric(subset, weights), "share_total_absolute_error_unweighted": float(subset.absolute_error.sum() / total_error)})
    condition = pd.DataFrame(condition_rows)
    save(condition, "condition_errors")
    daily = []
    for (target, model, date), g in q.groupby(["target", "model", "date"]):
        daily.append({"target": target, "model": model, "date": date, **metric(g)})
    save(pd.DataFrame(daily), "daily_errors")
    save(q.loc[q.target.eq("target_maximum")].sort_values("absolute_error", ascending=False).groupby("model").head(20), "largest_errors")
    model_manifest = []
    for row in selected.loc[selected.family.ne("Baseline")].itertuples():
        for split in c["splits"]:
            estimator = fitted[(split, row.target, row.model)]
            path = MODELS / f"{split}_{row.target}_{row.model}.joblib"
            joblib.dump(estimator, path, compress=3)
            reloaded = joblib.load(path)
            ev = f.loc[f.eligible_common & f.timestamp.isin(q.loc[q.split.eq(split)].timestamp)]
            expected_pred = q.loc[q.target.eq(row.target) & q.model.eq(row.model) & q.split.eq(split)].prediction.to_numpy()
            assert np.allclose(reloaded.predict(ev[features]), expected_pred, rtol=0, atol=1e-10)
            model_manifest.append({"split": split, "target": row.target, "model": row.model, "file": str(path.relative_to(ROOT)), "sha256": sha(path), "reload_predictions_checked": True})
    save(pd.DataFrame(model_manifest), "model_manifest")
    best = {}
    priority = {"Baseline": 0, "Ridge": 1, "ElasticNet": 2, "HGB": 3, "SVR": 4}
    for target, group in selected.groupby("target"):
        near = group.loc[group.MAE <= group.MAE.min() * 1.01].copy()
        near["priority"] = near.family.map(priority)
        best[target] = near.sort_values("priority").iloc[0][["model", "family", "MAE", "RMSE"]].to_dict()
    assert p.timestamp.max() < pd.Timestamp("2021-07-01")
    assert sha(CONTRACT) == inputs["m02_contract"]
    run.update({"status": "completed", "finished": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"),
                "fits_completed": len(audits), "prediction_rows": len(p), "seconds": time.perf_counter() - total_started,
                "development_hours_per_method_target": 2184, "provisional_candidates": best,
                "july_august_evaluated": False, "last_slot_used": False,
                "outputs_sha256": {x.name: sha(x) for x in OUT.glob("*.csv")}, "script_sha256": sha(Path(__file__))})
    (OUT / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(selected[["target", "family", "model", "MAE", "RMSE", "mean_under", "mean_over"]].to_string(index=False), flush=True)
    print(json.dumps({"status": "completed", "fits": len(audits), "best": best, "seconds": run["seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
