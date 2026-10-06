"""S04 fixed-HGB controlled integration of genuine chronological S03 probabilities.

freeze -> run. Uses Jan-defined labels and Feb-Jun precomputed OOF scores only.
No July/August scoring, refitting classifiers, tuning, clipping, or source edits.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Modeling/tables/regime_integration"
MODELS = ROOT / "Modeling/models/regime_integration"
FORECAST = ROOT / "Modeling/tables/regime_forecast"
DIAG = ROOT / "Modeling/tables/regime_diagnosis"
VARIANTS = ["B0", "B1", "B2_up", "B2_down", "B2_both"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def dependencies():
    names = ["Modeling/config/m01_contract.json", "Modeling/config/m03_ab_contract.json",
             "Modeling/tables/m01/hourly_frame.csv", "Modeling/tables/m01/verification.json",
             "Modeling/tables/regime_diagnosis/hourly_labels.csv", "Modeling/tables/regime_diagnosis/contract.json",
             "Modeling/tables/regime_forecast/predictions.csv", "Modeling/tables/regime_forecast/folds.csv",
             "Modeling/tables/regime_forecast/contract.json", "Modeling/tables/regime_forecast/run.json",
             "Modeling/tables/regime_forecast/independent_verification.json"]
    return {n: sha(ROOT / n) for n in names}


def definitions():
    m1 = json.loads((ROOT / "Modeling/config/m01_contract.json").read_text(encoding="utf-8"))
    s3 = json.loads((FORECAST / "contract.json").read_text(encoding="utf-8"))
    base = m1["calendar_features"] + m1["feature_groups"]["A"] + m1["feature_groups"]["B_add"]
    extra = [n for n in s3["feature_names"] if n.startswith(("prior_", "past_")) and n not in ["prior_mean", "prior_production"]]
    # Exact duplicates prior_mean/production are already lag1_mean/production.
    extended = base + extra + ["up_classifier_available", "down_classifier_available"]
    return {"B0": base, "B1": extended, "B2_up": extended + ["p_up"],
            "B2_down": extended + ["p_down"], "B2_both": extended + ["p_up", "p_down"]}


def data():
    m = pd.read_csv(ROOT / "Modeling/tables/m01/hourly_frame.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    h = pd.read_csv(DIAG / "hourly_labels.csv", encoding="utf-8-sig", parse_dates=["timestamp", "label_confirmed_at", "latest_input_timestamp"])
    p = pd.read_csv(FORECAST / "predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp", "label_confirmed_at"])
    f = h.loc[h.eligible_onset & h.persistence_known & (h.month >= 2)].copy()
    base = definitions()["B0"]
    # Use the same validated M01 common pool; Jan rows are not backfilled with a definition learned at Jan end.
    b = m.loc[m.eligible_common & (m.timestamp < "2021-07-01"), ["timestamp", "target_maximum", "target_mean"] + base]
    dynamic = [n for n in h.columns if n.startswith(("prior_", "past_"))]
    f = f[["timestamp", "label_confirmed_at", "latest_input_timestamp", "month", "profile", "onset", "sustained_onset", "level", "maximum"] + dynamic]
    f = f.drop(columns="month").merge(b, on="timestamp", how="inner", validate="one_to_one")
    assert len(f) == 3598 and np.allclose(f.maximum, f.target_maximum) and np.allclose(f.level, f.target_mean)
    assert np.allclose(f.prior_mean, f.lag1_mean) and np.allclose(f.prior_production, f.lag1_production)
    for direction, method in [("up", "logistic"), ("down", "hgb")]:
        rows = []
        for _, group in p.loc[p.direction == direction].groupby("timestamp", sort=True):
            kind = method if method in set(group.method) else "calendar"
            r = group.loc[group.method == kind].iloc[0]
            rows.append({"timestamp": r.timestamp, f"p_{direction}": r.score,
                         f"{direction}_classifier_available": int(kind == method),
                         f"{direction}_score_method": kind})
        f = f.merge(pd.DataFrame(rows), on="timestamp", validate="one_to_one")
    assert f.timestamp.min() == pd.Timestamp("2021-02-01") and len(f) == 3598
    assert (f.latest_input_timestamp == f.timestamp - pd.Timedelta(hours=1)).all()
    assert f[definitions()["B2_both"]].notna().all().all()
    return f.sort_values("timestamp").reset_index(drop=True)


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "contract.json").exists()
    verified = json.loads((FORECAST / "independent_verification.json").read_text(encoding="utf-8"))
    assert verified["status"] == "passed" and verified["run_sha256"] == sha(FORECAST / "run.json")
    frame = data()  # Availability and fixed target counts only; no model or comparative errors.
    c = {"stage": "S04", "created_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"),
         "entrypoint_sha256": sha(__file__), "inputs_sha256": dependencies(), "features": definitions(),
         "reason": "S03 persistent up 12/12 FP0; down 8/15 FP10. Test whether predicted transition probabilities add beyond identical raw causal inputs.",
         "scope": "Exploratory Feb-Jun training, Apr-Jun evaluation; initial definition estimated in Jan. No new independent final test.",
         "hgb_parameters": json.loads((ROOT / "Modeling/config/m03_ab_contract.json").read_text(encoding="utf-8"))["hgb_parameters"],
         "probability_policy": "Fixed S03 Logistic up / HGB down genuine earlier-month predictions; use saved calendar rate when ML unavailable. No in-sample rescoring, no Jan probability backfill. Availability indicators also in B1 control.",
         "training": "Same Feb-to-previous-month rows for all variants; label_confirmed_at < evaluation boundary. Monthly refit, hourly input update. Compare with newly fitted B0, not historical Jan-trained errors.",
         "evaluation": "Same Apr-Jun eligible common rows, sustained label known; last two June hours excluded. April diagnostic for upward integration: no upward ML OOF training yet. Primary up May-Jun; down Apr-Jun.",
         "primary_variant": "B2_up", "secondary_variants": ["B2_down", "B2_both"],
         "development_gate": {"event_start_mae_reduction_vs_both_B0_B1": .05,
                              "directional_error_reduction_vs_both_B0_B1": .05,
                              "overall_mae_worsening_vs_B0_at_most": .01,
                              "daily_peak_mae_worsening_vs_B0_at_most": .01,
                              "minimum_events": 5, "minimum_months_with_start_mae_improvement_vs_B1": 2},
         "directional_errors": "up under=max(actual-pred,0); down over=max(pred-actual,0). Zero baseline cannot show relative reduction; require strict absolute decrease otherwise.",
         "metrics": "MAE/RMSE/bias/under/over, primary event starts, event-equal first-three-hour MAE, monthly directions, full24-hour daily peak tie weights, seen/new daily profiles; no favorable case deletion",
         "event_windows": "t,t+1,t+2 predictions only for completely covered windows. Report exclusion counts and event-equal averages; overlaps across different event baselines possible.",
         "daily_peak": "Only full 24 prediction hours per day; all maximum ties get inverse tie-count weights; June30 excluded from daily comparison.",
         "model_selection": "Fixed ablation variants, primary up candidate predetermined; report all. No best-by-evaluation candidate treated as preselected or validated deployment model.",
         "tuning_trials": 0, "clip_predictions": False,
         "preflight_counts": {"rows": len(frame), "by_month": {str(m): len(g) for m, g in frame.groupby("month")},
                              "features": {v: len(cols) for v, cols in definitions().items()}}}
    write_json(OUT / "contract.json", c)
    print(json.dumps({"status": "frozen", **c["preflight_counts"]}), flush=True)


def measure(g, weights=None):
    err = g.prediction.to_numpy() - g.actual.to_numpy()
    w = np.ones(len(g)) if weights is None else np.asarray(weights)
    return {"hours": len(g), "weight_sum": float(w.sum()), "mae": float(np.average(abs(err), weights=w)),
            "rmse": float(np.sqrt(np.average(err ** 2, weights=w))), "bias": float(np.average(err, weights=w)),
            "under": float(np.average(np.maximum(-err, 0), weights=w)), "over": float(np.average(np.maximum(err, 0), weights=w))}


def run():
    assert not (OUT / "run.json").exists(), "Completed results must not be overwritten."
    c = json.loads((OUT / "contract.json").read_text(encoding="utf-8"))
    assert c["entrypoint_sha256"] == sha(__file__) and c["inputs_sha256"] == dependencies()
    f = data(); MODELS.mkdir(parents=True, exist_ok=True)
    predictions = []; manifest = []; folds = []
    for month in [4, 5, 6]:
        boundary = pd.Timestamp(2021, month, 1)
        tr = f.loc[(f.timestamp < boundary) & (f.label_confirmed_at < boundary)]
        te = f.loc[f.month == month]
        fold = {"month": month, "train_hours": len(tr), "train_start": str(tr.timestamp.min()),
                "train_end": str(tr.timestamp.max()), "last_confirmed": str(tr.label_confirmed_at.max()),
                "eval_hours": len(te), "train_up_ml_probability_hours": int(tr.up_classifier_available.sum()),
                "train_down_ml_probability_hours": int(tr.down_classifier_available.sum())}
        folds.append(fold)
        for variant in VARIANTS:
            cols = c["features"][variant]
            model = HistGradientBoostingRegressor(**c["hgb_parameters"])
            model.fit(tr[cols], tr.target_maximum)
            yhat = model.predict(te[cols])
            path = MODELS / f"{month:02d}_{variant}.joblib"
            assert not path.exists(); joblib.dump(model, path)
            manifest.append({**fold, "variant": variant, "features": "|".join(cols),
                             "file": path.relative_to(ROOT).as_posix(), "sha256": sha(path)})
            g = te[["timestamp", "month", "profile", "onset", "sustained_onset", "prior_mean", "level", "p_up", "p_down", "up_classifier_available", "down_classifier_available"]].copy()
            g["variant"] = variant; g["actual"] = te.target_maximum.to_numpy(); g["prediction"] = yhat
            g["seen_regression_profile"] = te.profile.isin(tr.profile).astype(int).to_numpy()
            predictions.append(g)
        print(json.dumps(fold), flush=True)
    p = pd.concat(predictions, ignore_index=True)
    p["date"] = p.timestamp.dt.date
    metrics = []; event_rows = []; coverage = []; daily_rows = []
    for variant, g in p.groupby("variant"):
        for period, gg in [("Apr-Jun", g), ("May-Jun", g.loc[g.month >= 5])]:
            metrics.append({"variant": variant, "period": period, "condition": "all", **measure(gg)})
            metrics.append({"variant": variant, "period": period, "condition": "non_persistent_start", **measure(gg.loc[gg.sustained_onset == 0])})
            for direction, sign in [("up", 1), ("down", -1)]:
                starts = gg.loc[gg.sustained_onset == sign]
                if len(starts):
                    metrics.append({"variant": variant, "period": period, "condition": f"{direction}_start", **measure(starts)})
            for seen, group in gg.groupby("seen_regression_profile"):
                metrics.append({"variant": variant, "period": period, "condition": f"profile_{int(seen)}", **measure(group)})
        for month, gg in g.groupby("month"):
            metrics.append({"variant": variant, "period": str(month), "condition": "all", **measure(gg)})
            for direction, sign in [("up", 1), ("down", -1)]:
                starts = gg.loc[gg.sustained_onset == sign]
                if len(starts):
                    metrics.append({"variant": variant, "period": str(month), "condition": f"{direction}_start", **measure(starts)})
        indexed = g.set_index("timestamp")
        for r in g.loc[g.sustained_onset != 0].itertuples():
            times = pd.date_range(r.timestamp, periods=3, freq="h")
            complete = all(t in indexed.index for t in times)
            coverage.append({"variant": variant, "timestamp": r.timestamp, "direction": "up" if r.sustained_onset == 1 else "down", "complete": complete})
            if complete:
                window = indexed.loc[times]
                event_rows.append({"variant": variant, "timestamp": r.timestamp, "month": r.month,
                                   "direction": "up" if r.sustained_onset == 1 else "down", "seen_profile": r.seen_regression_profile,
                                   "start_actual": r.actual, "start_prediction": r.prediction,
                                   "p_up": r.p_up, "p_down": r.p_down, **measure(window)})
        for date, gg in g.groupby("date"):
            if len(gg) != 24:
                continue
            peak = gg.loc[gg.actual == gg.actual.max()]
            daily_rows.append({"variant": variant, "date": str(date), "month": int(gg.month.iloc[0]),
                               "peak_actual": float(gg.actual.max()), "ties": len(peak), **measure(peak)})
    events = pd.DataFrame(event_rows); daily = pd.DataFrame(daily_rows)
    for variant, g in events.groupby("variant"):
        for period, gg in [("Apr-Jun", g), ("May-Jun", g.loc[g.month >= 5])]:
            for direction, part in gg.groupby("direction"):
                metrics.append({"variant": variant, "period": period, "condition": direction + "_window_event_equal",
                                "hours": len(part) * 3, "weight_sum": len(part),
                                **{n: float(part[n].mean()) for n in ["mae", "bias", "under", "over"]},
                                "rmse": float(np.sqrt(np.mean(part.rmse ** 2)))})
        for period, gg in [("Apr-Jun", daily.loc[daily.variant == variant]), ("May-Jun", daily.loc[(daily.variant == variant) & (daily.month >= 5)])]:
            # MAE/under/over/bias average by day; RMSE is square root of mean daily MSE.
            metrics.append({"variant": variant, "period": period, "condition": "daily_peak_day_equal", "hours": int(gg.ties.sum()),
                            "weight_sum": len(gg), **{n: float(gg[n].mean()) for n in ["mae", "bias", "under", "over"]},
                            "rmse": float(np.sqrt(np.mean(gg.rmse ** 2)))})
    table = pd.DataFrame(metrics)
    gains = []; decisions = []
    for variant in VARIANTS[2:]:
        for direction, period, directional in [("up", "May-Jun", "under"), ("down", "Apr-Jun", "over")]:
            stats = {}
            for baseline in ["B0", "B1"]:
                a = table.loc[(table.variant == variant) & (table.period == period) & (table.condition == direction + "_start")].iloc[0]
                b = table.loc[(table.variant == baseline) & (table.period == period) & (table.condition == direction + "_start")].iloc[0]
                row = {"variant": variant, "direction": direction, "period": period, "baseline": baseline,
                       "events": int(a.hours), "start_mae_reduction": 1 - a.mae / b.mae,
                       "directional_reduction": 1 - a[directional] / b[directional] if b[directional] else None}
                gains.append(row); stats[baseline] = row
            a_all = table.loc[(table.variant == variant) & (table.period == period) & (table.condition == "all")].iloc[0]
            b_all = table.loc[(table.variant == "B0") & (table.period == period) & (table.condition == "all")].iloc[0]
            a_peak = table.loc[(table.variant == variant) & (table.period == period) & (table.condition == "daily_peak_day_equal")].iloc[0]
            b_peak = table.loc[(table.variant == "B0") & (table.period == period) & (table.condition == "daily_peak_day_equal")].iloc[0]
            monthly_wins = 0
            for m in ([5, 6] if direction == "up" else [4, 5, 6]):
                aa = table.loc[(table.variant == variant) & (table.period == str(m)) & (table.condition == direction + "_start")].iloc[0]
                bb = table.loc[(table.variant == "B1") & (table.period == str(m)) & (table.condition == direction + "_start")].iloc[0]
                monthly_wins += int(aa.mae < bb.mae)
            passed = (all(s["start_mae_reduction"] >= .05 and s["directional_reduction"] is not None and s["directional_reduction"] >= .05 for s in stats.values())
                      and a_all.mae <= b_all.mae * 1.01 and a_peak.mae <= b_peak.mae * 1.01
                      and stats["B0"]["events"] >= 5 and monthly_wins >= 2)
            decisions.append({"variant": variant, "direction": direction, "period": period,
                              "development_gate_passed": bool(passed), "monthly_wins_vs_B1": monthly_wins,
                              "overall_mae_change_vs_B0": a_all.mae / b_all.mae - 1,
                              "daily_peak_mae_change_vs_B0": a_peak.mae / b_peak.mae - 1})
    outputs = {"meta_features.csv": f, "predictions.csv": p, "metrics.csv": table, "folds.csv": pd.DataFrame(folds),
               "model_manifest.csv": pd.DataFrame(manifest), "event_errors.csv": events,
               "event_coverage.csv": pd.DataFrame(coverage), "daily_peak_errors.csv": daily, "paired_gains.csv": pd.DataFrame(gains)}
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    write_json(OUT / "decision.json", {"primary": "B2_up on up May-Jun", "comparisons": decisions,
                                        "scope": "Development ablation using strategy chosen after S03; not independent validation or operating savings."})
    write_json(OUT / "run.json", {"status": "completed", "entrypoint_sha256": sha(__file__), "contract_sha256": sha(OUT / "contract.json"),
                                 "inputs_sha256": c["inputs_sha256"], "new_regression_fits": len(manifest), "new_classifier_fits": 0,
                                 "outputs_sha256": {n: sha(OUT / n) for n in list(outputs) + ["decision.json"]}, "runtime": sys.version})
    print(json.dumps({"status": "completed", "new_fits": len(manifest), "decisions": decisions}), flush=True)


if __name__ == "__main__":
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    {"freeze": freeze, "run": run}[sys.argv[1]]()
