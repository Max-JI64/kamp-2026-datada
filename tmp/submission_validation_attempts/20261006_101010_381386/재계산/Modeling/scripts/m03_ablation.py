"""M03 fixed-HGB input ablation; verified parent predictions reused, dev only."""
import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from m01_prepare import ROOT, feature_columns, read_contract, sha
from m02_compare import metric, state


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes")


def subsets(g):
    result = [("all", "all", g, None), ("profile_reweighted", "all", g, g.profile_weight)]
    peak = g.loc[g.daily_maximum_weight.gt(0)]
    result.append(("daily_maximum", "all", peak, peak.daily_maximum_weight))
    for col in ["split", "month", "weekend", "hour", "power_transition", "diag_production_transition", "train_profile_overlap"]:
        result.extend((col, str(value), part, None) for value, part in g.groupby(col))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["ab", "c"], required=True)
    stage = parser.parse_args().stage
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    contract_path = ROOT / f"Modeling/config/m03_{stage}_contract.json"
    c = json.loads(contract_path.read_text(encoding="utf-8"))
    parent = read_contract()
    out = ROOT / f"Modeling/tables/m03/{stage}"
    model_dir = ROOT / f"Modeling/models/m03/{stage}"
    out.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    def save(df, name):
        df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    v1 = json.loads((ROOT / "Modeling/tables/m01/verification.json").read_text(encoding="utf-8"))
    assert sha(ROOT / parent["source"]) == parent["source_sha256"]
    inputs = {"source": sha(ROOT / parent["source"]), "contract": sha(contract_path),
              "m01_contract": sha(ROOT / "Modeling/config/m01_contract.json")}
    for name in ["hourly_frame.csv", "evaluation_diagnostics.csv", "split_counts.csv"]:
        path = ROOT / "Modeling/tables/m01" / name
        assert sha(path) == v1["outputs_sha256"][name]
        inputs[str(path.relative_to(ROOT))] = sha(path)
    reuse_dir = ROOT / ("Modeling/tables/m02" if stage == "ab" else "Modeling/tables/m03/ab")
    reuse_run = json.loads((reuse_dir / "run.json").read_text(encoding="utf-8"))
    assert reuse_run["status"] == "completed"
    verification = json.loads((reuse_dir / "independent_verification.json").read_text(encoding="utf-8"))
    assert verification["status"] == "passed"
    reuse_name = "selected_predictions.csv" if stage == "ab" else "predictions.csv"
    for name in [reuse_name, "model_manifest.csv"]:
        path = reuse_dir / name
        assert sha(path) == reuse_run["outputs_sha256"][name]
        inputs[str(path.relative_to(ROOT))] = sha(path)
    inputs[str((reuse_dir / "run.json").relative_to(ROOT))] = sha(reuse_dir / "run.json")
    if stage == "ab":
        assert inputs["m01_contract"] == reuse_run["input_hashes"]["m01_contract"]
        assert sha(ROOT / c["parent"]) == reuse_run["input_hashes"]["m02_contract"]
        assert c["hgb_parameters"] == next(x[3] for x in __import__("m02_compare").candidates(json.loads((ROOT / c["parent"]).read_text(encoding="utf-8"))) if x[0] == "hgb_01")
    else:
        assert c["hgb_parameters"] == json.loads((ROOT / c["parent"]).read_text(encoding="utf-8"))["hgb_parameters"]
    f = pd.read_csv(ROOT / "Modeling/tables/m01/hourly_frame.csv", encoding="utf-8-sig", float_precision="round_trip", parse_dates=["timestamp", "date"])
    diag = pd.read_csv(ROOT / "Modeling/tables/m01/evaluation_diagnostics.csv", encoding="utf-8-sig", parse_dates=["timestamp", "date"])
    expected = pd.read_csv(ROOT / "Modeling/tables/m01/split_counts.csv", encoding="utf-8-sig")
    reused = pd.read_csv(reuse_dir / reuse_name, encoding="utf-8-sig", float_precision="round_trip", parse_dates=["timestamp", "date"])
    manifest = pd.read_csv(reuse_dir / "model_manifest.csv", encoding="utf-8-sig")
    if stage == "ab":
        reused = reused.loc[reused.model.eq("hgb_01")].copy()
        reused["group"] = "A"
    f["power_state"] = state(f.target_maximum, f.diag_target_rounded_mean)
    f["prior_power_state"] = state(f.lag1_maximum, np.floor(f.lag1_mean + 0.5))
    f["power_transition"] = f.prior_power_state + "->" + f.power_state
    features = {group: feature_columns(group, parent) for group in c["groups"]}
    assert features["B"] == features["A"] + ["lag1_last"]
    for group, cols in features.items():
        assert not any(x.startswith(("target_", "diag_")) for x in cols)
    run = {"status": "running", "started": now(), "stage": stage, "input_hashes": inputs,
           "features": features, "parameters": c["hgb_parameters"],
           "runtime": {"executable": sys.executable, "version": sys.version, "platform": platform.platform(),
                       "packages": {x: importlib.metadata.version(x) for x in ["numpy", "pandas", "scikit-learn", "joblib"]}}}
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    begin = time.perf_counter()
    records, audits, models = [], [], []
    for split in parent["splits"]:
        if split["name"] not in c["splits"]:
            continue
        tr = f.loc[f.eligible_common & (f.timestamp <= pd.Timestamp(split["train_end"]))]
        ev = f.loc[f.eligible_common & f.timestamp.between(pd.Timestamp(split["eval_start"]), pd.Timestamp(split["eval_end"]))]
        count = expected.loc[expected.split.eq(split["name"]) & expected.pool.eq("common")].iloc[0]
        assert len(tr) == count.train_hours and len(ev) == count.eval_hours
        assert tr.timestamp.max() < ev.timestamp.min()
        d = diag.loc[diag.split.eq(split["name"]) & diag.pool.eq("common")]
        base = ev[["timestamp", "date", "month", "weekend", "power_transition", "diag_production_transition", "diag_target_profile"]].merge(
            d[["timestamp", "train_profile_overlap", "profile_weight", "daily_maximum_weight"]], on="timestamp", validate="one_to_one")
        base["hour"] = base.timestamp.dt.hour
        assert base.timestamp.tolist() == ev.timestamp.tolist()
        for target in c["targets"]:
            for group in c["groups"]:
                cols = features[group]
                assert tr[cols].notna().all().all() and ev[cols].notna().all().all()
                start = time.perf_counter()
                if group in c["new_groups"]:
                    estimator = HistGradientBoostingRegressor(**c["hgb_parameters"])
                    estimator.fit(tr[cols], tr[target])
                    pred = estimator.predict(ev[cols])
                    path = model_dir / f"{split['name']}_{target}_{group}.joblib"
                    joblib.dump(estimator, path, compress=3)
                    assert np.allclose(joblib.load(path).predict(ev[cols]), pred, rtol=0, atol=1e-10)
                    source = "new_fit"
                else:
                    mr = manifest.loc[manifest.split.eq(split["name"]) & manifest.target.eq(target) &
                                      (manifest.model.eq("hgb_01") if stage == "ab" else manifest.group.eq(group))].iloc[0]
                    path = ROOT / mr.file
                    assert sha(path) == mr.sha256
                    estimator = joblib.load(path)
                    pred = estimator.predict(ev[cols])
                    old = reused.loc[reused.split.eq(split["name"]) & reused.target.eq(target) & reused.group.eq(group)]
                    assert old.timestamp.tolist() == ev.timestamp.tolist()
                    assert np.array_equal(old.actual.to_numpy(), ev[target].to_numpy())
                    assert np.allclose(old.prediction.to_numpy(), pred, rtol=0, atol=1e-10)
                    source = "verified_parent_model"
                assert list(estimator.feature_names_in_) == cols
                assert all(estimator.get_params()[k] == v for k, v in c["hgb_parameters"].items())
                assert np.isfinite(pred).all()
                rec = base.copy()
                rec["split"], rec["target"], rec["group"] = split["name"], target, group
                rec["actual"], rec["prediction"] = ev[target].to_numpy(), pred
                records.append(rec)
                models.append({"split": split["name"], "target": target, "group": group, "file": str(path.relative_to(ROOT)),
                               "sha256": sha(path), "source": source, "reload_predictions_checked": True})
                audits.append({"split": split["name"], "target": target, "group": group, "train_hours": len(tr), "eval_hours": len(ev),
                               "features": len(cols), "source": source, "train_end": str(tr.timestamp.max()), "eval_start": str(ev.timestamp.min()),
                               "seconds": time.perf_counter() - start})
        print(f"completed {stage}: {split['name']}", flush=True)
    p = pd.concat(records, ignore_index=True)
    p["signed_error"] = p.prediction - p.actual
    p["absolute_error"] = p.signed_error.abs()
    p["under_amount"] = (-p.signed_error).clip(lower=0)
    p["over_amount"] = p.signed_error.clip(lower=0)
    assert p.timestamp.max() < pd.Timestamp("2021-07-01")
    save(p, "predictions")
    save(pd.DataFrame(audits), "training_audit")
    save(pd.DataFrame(models), "model_manifest")
    scores, conditions, daily = [], [], []
    for (target, group), g in p.groupby(["target", "group"]):
        scores.append({"target": target, "group": group, "split": "pooled_development", **metric(g)})
        scores.extend({"target": target, "group": group, "split": s, **metric(part)} for s, part in g.groupby("split"))
        for kind, value, part, weights in subsets(g):
            if len(part):
                conditions.append({"target": target, "group": group, "condition": kind, "value": value, **metric(part, weights),
                                   "share_total_absolute_error_unweighted": float(part.absolute_error.sum() / g.absolute_error.sum())})
        daily.extend({"target": target, "group": group, "date": date, **metric(part)} for date, part in g.groupby("date"))
    scores = pd.DataFrame(scores)
    save(scores, "comparison")
    save(pd.DataFrame(conditions), "condition_errors")
    save(pd.DataFrame(daily), "daily_errors")
    pairs, deltas, daily_deltas = [], [], []
    keys = ["split", "target", "timestamp"]
    for before, after in c["comparisons"]:
        a = p.loc[p.group.eq(before)].drop(columns="group")
        b = p.loc[p.group.eq(after), keys + ["actual", "prediction"]]
        pair = a.merge(b, on=keys, suffixes=("_before", "_after"), validate="one_to_one")
        assert np.array_equal(pair.actual_before, pair.actual_after)
        pair["before"], pair["after"] = before, after
        for label in ["before", "after"]:
            error = pair[f"prediction_{label}"] - pair.actual_before
            pair[f"abs_{label}"] = error.abs()
            pair[f"under_{label}"] = (-error).clip(lower=0)
            pair[f"over_{label}"] = error.clip(lower=0)
        for kind in ["abs", "under", "over"]:
            pair[f"delta_{kind}"] = pair[f"{kind}_after"] - pair[f"{kind}_before"]
        pairs.append(pair)
        for target, g in pair.groupby("target"):
            for kind, value, part, weights in subsets(g):
                if part.empty:
                    continue
                w = np.ones(len(part)) if weights is None else np.asarray(weights)
                deltas.append({"target": target, "before": before, "after": after, "condition": kind, "value": value,
                               "hours": len(part), "weight_sum": float(w.sum()),
                               **{f"delta_{x}": float(np.average(part[f"delta_{x}"], weights=w)) for x in ["abs", "under", "over"]},
                               "improved_hours": int(part.delta_abs.lt(-1e-10).sum()), "worsened_hours": int(part.delta_abs.gt(1e-10).sum())})
            for date, part in g.groupby("date"):
                daily_deltas.append({"target": target, "before": before, "after": after, "date": date, "hours": len(part),
                                     **{f"delta_{x}": float(part[f"delta_{x}"].mean()) for x in ["abs", "under", "over"]}})
    save(pd.concat(pairs, ignore_index=True), "paired_predictions")
    save(pd.DataFrame(deltas), "paired_deltas")
    save(pd.DataFrame(daily_deltas), "daily_deltas")
    chosen = {}
    for target, g in scores.loc[scores.split.eq("pooled_development")].groupby("target"):
        near = g.loc[g.MAE <= g.MAE.min() * 1.01].copy()
        near["size"] = near.group.map(lambda x: len(features[x]))
        chosen[target] = near.sort_values(["size", "MAE"]).iloc[0][["group", "MAE", "RMSE"]].to_dict()
    assert sha(contract_path) == inputs["contract"]
    run.update({"status": "completed", "finished": now(), "seconds": time.perf_counter() - begin,
                "fits_completed": sum(x["source"] == "new_fit" for x in audits), "reused_models_verified": sum(x["source"] != "new_fit" for x in audits),
                "prediction_rows": len(p), "july_august_evaluated": False, "provisional_candidates": chosen,
                "outputs_sha256": {x.name: sha(x) for x in out.glob("*.csv")},
                "script_sha256": sha(Path(__file__)), "helper_sha256": {x: sha(ROOT / f"Modeling/scripts/{x}") for x in ["m01_prepare.py", "m02_compare.py"]}})
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(scores.loc[scores.split.eq("pooled_development"), ["target", "group", "MAE", "RMSE", "mean_under", "mean_over"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
