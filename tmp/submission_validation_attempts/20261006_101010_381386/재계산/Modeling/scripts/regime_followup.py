"""S05 fixed June-end upward classifier and B0/B1 routing, July-August follow-up.

Freeze using only previous artifacts, then run once. Future labels are diagnostic.
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
from regime_diagnose import label
from regime_forecast import model, threshold, calendar_scores

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Modeling/tables/regime_followup"
MODELS = ROOT / "Modeling/models/regime_followup"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jsave(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def inputs():
    names = ["data/origin/okm_augumented_2021.csv", "Modeling/tables/m01/hourly_frame.csv", "Modeling/tables/m01/verification.json",
             "Modeling/tables/regime_diagnosis/hourly_labels.csv", "Modeling/tables/regime_diagnosis/contract.json",
             "Modeling/tables/regime_forecast/contract.json", "Modeling/tables/regime_forecast/predictions.csv",
             "Modeling/tables/regime_forecast/independent_verification.json", "Modeling/tables/regime_integration/contract.json",
             "Modeling/tables/regime_integration/meta_features.csv", "Modeling/tables/regime_integration/independent_verification.json",
             "Modeling/tables/regime_routing/contract.json", "Modeling/tables/regime_routing/decision.json",
             "Modeling/tables/regime_routing/independent_verification.json", "Modeling/scripts/regime_diagnose.py",
             "Modeling/scripts/regime_forecast.py"]
    return {n: sha(ROOT / n) for n in names}


def contracts():
    return [json.loads((ROOT / f"Modeling/tables/{folder}/contract.json").read_text(encoding="utf-8"))
            for folder in ["regime_diagnosis", "regime_forecast", "regime_integration"]]


def classification_train():
    f = pd.read_csv(ROOT / "Modeling/tables/regime_diagnosis/hourly_labels.csv", encoding="utf-8-sig", parse_dates=["timestamp", "label_confirmed_at"])
    f = f.loc[f.eligible_onset & f.persistence_known & (f.label_confirmed_at < "2021-07-01")].copy()
    return add_calendar(f)


def add_calendar(f):
    f = f.copy()
    f["hour_sin"] = np.sin(2 * np.pi * f.timestamp.dt.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * f.timestamp.dt.hour / 24)
    for i in range(7):
        f[f"dow_{i}"] = (f.timestamp.dt.dayofweek == i).astype(int)
    return f


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "contract.json").exists()
    for folder in ["regime_forecast", "regime_integration", "regime_routing"]:
        verified = json.loads((ROOT / f"Modeling/tables/{folder}/independent_verification.json").read_text(encoding="utf-8"))
        assert verified["status"] == "passed" and verified["run_sha256"] == sha(ROOT / f"Modeling/tables/{folder}/run.json")
    d, s, r = contracts()
    cp = pd.read_csv(ROOT / "Modeling/tables/regime_forecast/predictions.csv", encoding="utf-8-sig", parse_dates=["label_confirmed_at"])
    cp = cp.loc[(cp.direction == "up") & cp.month.between(4, 6) & (cp.label_confirmed_at < "2021-07-01")]
    calibrated = {}
    for method in ["logistic", "calendar"]:
        g = cp.loc[cp.method == method]
        t, tp, fp, cap = threshold(g.event.to_numpy(int), g.score.to_numpy(float))
        calibrated[method] = {"threshold": float(t) if np.isfinite(t) else "infinity", "tp": int(tp), "fp": int(fp), "fp_cap": int(cap),
                              "hours": len(g), "events": int(g.event.sum()), "latest_label_confirmation": str(g.label_confirmed_at.max())}
    ct = classification_train()
    rt = pd.read_csv(ROOT / "Modeling/tables/regime_integration/meta_features.csv", encoding="utf-8-sig", parse_dates=["timestamp", "label_confirmed_at"])
    rt = rt.loc[rt.label_confirmed_at < "2021-07-01"]
    assert len(ct) == 4336 and len(rt) == 3598
    c = {"stage": "S05", "created_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"),
         "entrypoint_sha256": sha(__file__), "inputs_sha256": inputs(), "onset_definition": d,
         "classification_features": s["feature_names"], "logistic": s["logistic"],
         "regression_features": {name: r["features"][name] for name in ["B0", "B1"]}, "hgb_parameters": r["hgb_parameters"],
         "training": "Classification Jan-Jun4336 matured labels; regression Feb-Jun3598 common genuine previous-month probability rows. One fit per model before July; no July/August fitting or parameter changes.",
         "models": ["up_logistic", "B0", "B1"], "classification_thresholds": calibrated,
         "threshold_policy": "Apply existing S03 threshold algorithm once to Apr-Jun previous-month predictions, FPR<=1%, fixed for both July/August. No in-sample calibration.",
         "routing": "G1 selects B1 only when fixed Logistic score>=fixed threshold, else B0. Compare B0,B1_all,G1,hour08,simple_gate.",
         "regression_flags": "Both classifier-availability flags1 at July/August forecast origin; June training had sufficient support. No down classifier prediction is needed by chosen B1/routing.",
         "scope": "Already observed July/August follow-up, not a new independent test. Do not retune from these results.",
         "availability": "Classification requires six exact earlier valid hourly records; regression additionally M01 common lags1/2/24/168. Predict whenever past inputs complete and current target valid; future persistence missing does not remove regression prediction. Classification metrics separately require observed t,t+1,t+2.",
         "classification_gate": {"min_events": 5, "recall_at_least": .8, "precision_at_least": .8, "fpr_at_most": .01},
         "routing_gate": {"min_events": 5, "start_mae_reduction_vs_B0_at_least": .05, "up_under_reduction_vs_B0_at_least": .05,
                          "overall_mae_worsening_vs_B0_at_most": .01, "daily_peak_mae_worsening_vs_B0_at_most": .01,
                          "monthly_results": "Diagnostic, no demand for every-month win; always report."},
         "coverage": "Report normal/invalid/gap hours, six-history vs common counts, known/unknown labels and excluded persistent events. Evaluate both classification full-six-history and primary common-regression scope.",
         "metrics": "AP with zero-event months undefined; TP/FP/FN/TN/precision/recall, onset/window MAE/under/over, full24-hour day-equal peak MAE, month/profile conditions, same-target paired errors and routing counts.",
         "preflight": {"classification_train": len(ct), "regression_train": len(rt), "training_up_events": int((ct.sustained_onset == 1).sum())},
         "no_tuning": True, "no_clipping": True, "no_root_readme_or_manuscript_edit": True}
    jsave(OUT / "contract.json", c)
    print(json.dumps({"status": "frozen", "preflight": c["preflight"], "thresholds": calibrated}), flush=True)


def extended(c):
    raw = pd.read_csv(ROOT / "data/origin/okm_augumented_2021.csv", encoding="utf-8-sig")
    raw = raw.loc[raw["날짜"].between(20210101, 20210831) & raw["시간"].between(0, 23)].copy()
    raw["timestamp"] = pd.to_datetime(raw["날짜"].astype(str), format="%Y%m%d") + pd.to_timedelta(raw["시간"], unit="h")
    raw = raw.sort_values("timestamp").set_index("timestamp")
    assert raw.index.is_unique
    f = raw.reindex(pd.date_range("2021-01-01", "2021-08-31 23:00", freq="h")); f.index.name = "timestamp"
    slots = ["15분", "30분", "45분", "60분"]
    f["valid"] = f[slots].notna().all(axis=1); f["level"] = f[slots].mean(axis=1).where(f.valid)
    f["maximum"] = f[slots].max(axis=1).where(f.valid)
    f["date"] = f.index.normalize(); f["month"] = f.index.month; f["hour"] = f.index.hour
    f["weekend"] = (f.index.dayofweek >= 5).astype(int)
    profiles = {date: hashlib.sha256(g[slots].to_numpy(dtype=np.int64).tobytes()).hexdigest() for date, g in raw.groupby(raw.index.normalize())}
    f["profile"] = f.date.map(profiles)
    h = pd.concat([f.level.shift(k).rename(str(k)) for k in range(1, 7)], axis=1)
    f["eligible_onset"] = f.valid & h.notna().all(axis=1); f["prior_mean"] = h["1"]
    f["past_median6"] = h.median(axis=1).where(f.eligible_onset)
    f["past_iqr6"] = (h.quantile(.75, axis=1) - h.quantile(.25, axis=1)).where(f.eligible_onset)
    f["past_range6"] = (h.max(axis=1) - h.min(axis=1)).where(f.eligible_onset)
    f["past_slope6"] = ((h["1"] - h["6"]) / 5).where(f.eligible_onset)
    f["prior_delta"] = f.level.shift(1) - f.level.shift(2)
    f["prior_last_minus_mean"] = f["60분"].shift(1) - f.prior_mean
    f["prior_slot_range"] = (f[slots].max(axis=1) - f[slots].min(axis=1)).shift(1).where(f.valid.shift(1, fill_value=False))
    f["prior_production"] = f["생산량"].shift(1); f["prior_production_delta"] = f["생산량"].shift(1) - f["생산량"].shift(2)
    f["delta"] = f.level - f.prior_mean; f["deviation"] = f.level - f.past_median6
    z, events = label(f, c["onset_definition"]["thresholds"])
    return add_calendar(z.reset_index()), events


def metrics(g):
    if not len(g):
        return {"hours": 0, **{n: None for n in ["mae", "rmse", "bias", "under", "over"]}}
    err = g.prediction.to_numpy() - g.actual.to_numpy()
    return {"hours": len(g), "mae": float(np.mean(abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(err)), "under": float(np.mean(np.maximum(-err, 0))), "over": float(np.mean(np.maximum(err, 0)))}


def detection(g):
    from sklearn.metrics import average_precision_score
    y, a = g.event.to_numpy(int), g.alarm.to_numpy(int)
    tp = int(((y == 1) & (a == 1)).sum()); fp = int(((y == 0) & (a == 1)).sum())
    fn = int(y.sum()) - tp; tn = len(g) - int(y.sum()) - fp
    return {"hours": len(g), "events": int(y.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": tp / (tp + fn) if tp + fn else None, "precision": tp / (tp + fp) if tp + fp else None,
            "fpr": fp / (fp + tn) if fp + tn else None, "ap": float(average_precision_score(y, g.score)) if y.sum() else None}


def run():
    assert not (OUT / "run.json").exists()
    c = json.loads((OUT / "contract.json").read_text(encoding="utf-8"))
    assert c["entrypoint_sha256"] == sha(__file__) and c["inputs_sha256"] == inputs()
    z, events = extended(c)
    ct = classification_train(); rt = pd.read_csv(ROOT / "Modeling/tables/regime_integration/meta_features.csv", encoding="utf-8-sig", parse_dates=["label_confirmed_at"])
    rt = rt.loc[rt.label_confirmed_at < "2021-07-01"]
    s = json.loads((ROOT / "Modeling/tables/regime_forecast/contract.json").read_text(encoding="utf-8"))
    y = (ct.sustained_onset == 1).astype(int).to_numpy(); classifier = model("logistic", s)
    classifier.fit(ct[c["classification_features"]], y)
    MODELS.mkdir(parents=True, exist_ok=True); manifest = []
    def keep(fitted, name, count):
        path = MODELS / (name + ".joblib"); assert not path.exists(); joblib.dump(fitted, path)
        manifest.append({"name": name, "file": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "train_hours": count, "trained_before": "2021-07-01"})
    keep(classifier, "up_logistic", len(ct))
    class_eval = z.loc[z.month >= 7].copy()
    class_eval = class_eval.loc[class_eval.eligible_onset]
    score = classifier.predict_proba(class_eval[c["classification_features"]])[:, 1]
    cal_score = calendar_scores(ct, class_eval, y)
    for method, sc in [("logistic", score), ("calendar", cal_score)]:
        class_eval["score_" + method] = sc
        t = c["classification_thresholds"][method]["threshold"]
        class_eval["alarm_" + method] = (sc >= (np.inf if t == "infinity" else t)).astype(int)
    class_eval["event"] = np.where(class_eval.persistence_known, (class_eval.sustained_onset == 1).astype(int), np.nan)
    class_eval["seen_classification_profile"] = class_eval.profile.isin(ct.profile).astype(int)
    m = pd.read_csv(ROOT / "Modeling/tables/m01/hourly_frame.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    base = m.loc[m.eligible_common & (m.month >= 7), ["timestamp", "target_maximum"] + c["regression_features"]["B0"]]
    extra = [n for n in c["regression_features"]["B1"] if n not in c["regression_features"]["B0"] and not n.endswith("classifier_available")]
    f = class_eval[["timestamp", "label_confirmed_at", "latest_input_timestamp", "profile", "level", "maximum", "onset", "sustained_onset", "persistence_known", "event", "score_logistic", "score_calendar", "alarm_logistic", "alarm_calendar"] + extra].merge(base, on="timestamp", validate="one_to_one")
    f["up_classifier_available"] = 1; f["down_classifier_available"] = 1
    f["seen_regression_profile"] = f.profile.isin(rt.profile).astype(int)
    assert np.allclose(f.maximum, f.target_maximum)
    fitted_reg = {}
    for name in ["B0", "B1"]:
        fitted = HistGradientBoostingRegressor(**c["hgb_parameters"])
        fitted.fit(rt[c["regression_features"][name]], rt.target_maximum); keep(fitted, name, len(rt))
        fitted_reg[name] = fitted.predict(f[c["regression_features"][name]])
    b0, b1 = fitted_reg["B0"], fitted_reg["B1"]
    options = {"B0": b0, "B1_all": b1, "G1": np.where(f.alarm_logistic == 1, b1, b0),
               "hour08": np.where(f.timestamp.dt.hour == 8, b1, b0), "simple_gate": np.where(f.alarm_calendar == 1, b1, b0)}
    masks = {"B0": np.zeros(len(f)), "B1_all": np.ones(len(f)), "G1": f.alarm_logistic,
             "hour08": (f.timestamp.dt.hour == 8).astype(int), "simple_gate": f.alarm_calendar}
    pred = []; metric_rows = []; event_rows = []; window_coverage = []; daily_rows = []; burden = []
    for variant, yhat in options.items():
        g = f.copy(); g["variant"] = variant; g["actual"] = g.target_maximum; g["prediction"] = yhat
        g["routed"] = np.asarray(masks[variant], int); g["date"] = g.timestamp.dt.strftime("%Y-%m-%d"); pred.append(g)
        for period, part in [("Jul-Aug", g), ("7", g.loc[g.month == 7]), ("8", g.loc[g.month == 8])]:
            for condition, gg in [("all", part), ("up_start", part.loc[part.sustained_onset == 1]),
                                  ("down_start", part.loc[part.sustained_onset == -1]), ("other_known_hours", part.loc[part.persistence_known & (part.sustained_onset != 1)])]:
                metric_rows.append({"variant": variant, "period": period, "condition": condition, **metrics(gg)})
            for seen, gg in part.groupby("seen_regression_profile"):
                metric_rows.append({"variant": variant, "period": period, "condition": f"profile_{seen}", **metrics(gg)})
            burden.append({"variant": variant, "period": period, "hours": len(part), "routed": int(part.routed.sum()),
                           "routed_up_starts": int(((part.routed == 1) & (part.sustained_onset == 1)).sum()),
                           "routed_non_up_known": int(((part.routed == 1) & part.persistence_known & (part.sustained_onset != 1)).sum()),
                           "routed_unknown": int(((part.routed == 1) & ~part.persistence_known).sum())})
        for r in g.loc[g.sustained_onset.isin([1, -1])].itertuples():
            times = pd.date_range(r.timestamp, periods=3, freq="h")
            window = g.loc[g.timestamp.isin(times)]; complete = len(window) == 3
            window_coverage.append({"variant": variant, "timestamp": r.timestamp, "direction": "up" if r.sustained_onset == 1 else "down", "complete": complete})
            if complete:
                event_rows.append({"variant": variant, "timestamp": r.timestamp, "month": r.month, "direction": "up" if r.sustained_onset == 1 else "down", **metrics(window)})
        for date, gg in g.groupby("date"):
            if len(gg) == 24:
                peak = gg.loc[gg.actual == gg.actual.max()]
                daily_rows.append({"variant": variant, "date": date, "month": int(gg.month.iloc[0]), "ties": len(peak), **metrics(peak)})
    p = pd.concat(pred, ignore_index=True); daily = pd.DataFrame(daily_rows); ew = pd.DataFrame(event_rows)
    for variant in options:
        for period in ["Jul-Aug", "7", "8"]:
            for condition, source in [("daily_peak_day_equal", daily.loc[daily.variant == variant]),
                                      ("up_window_event_equal", ew.loc[(ew.variant == variant) & (ew.direction == "up")])]:
                gg = source if period == "Jul-Aug" else source.loc[source.month == int(period)]
                result = {"hours": int(gg.ties.sum()) if condition.startswith("daily") else len(gg) * 3, "units": len(gg)}
                result.update({n: float(gg[n].mean()) if len(gg) else None for n in ["mae", "under", "over", "bias"]})
                result["rmse"] = float(np.sqrt(np.mean(gg.rmse ** 2))) if len(gg) else None
                metric_rows.append({"variant": variant, "period": period, "condition": condition, **result})
    det = []
    for scope, part in [("six_history", class_eval), ("common_regression", class_eval.loc[class_eval.timestamp.isin(f.timestamp)])]:
        part = part.loc[part.persistence_known]
        for period, gg in [("Jul-Aug", part), ("7", part.loc[part.month == 7]), ("8", part.loc[part.month == 8])]:
            for method in ["logistic", "calendar"]:
                det.append({"scope": scope, "period": period, "method": method,
                            **detection(gg.assign(score=gg["score_" + method], alarm=gg["alarm_" + method]))})
    table = pd.DataFrame(metric_rows); detection_table = pd.DataFrame(det)
    def get(v, condition):
        return table.loc[(table.variant == v) & (table.period == "Jul-Aug") & (table.condition == condition)].iloc[0]
    a, b = get("G1", "up_start"), get("B0", "up_start")
    gain = 1 - a.mae / b.mae if b.mae else None; under_gain = 1 - a.under / b.under if b.under else None
    all_change = get("G1", "all").mae / get("B0", "all").mae - 1
    peak_change = get("G1", "daily_peak_day_equal").mae / get("B0", "daily_peak_day_equal").mae - 1
    dr = detection_table.loc[(detection_table.scope == "common_regression") & (detection_table.period == "Jul-Aug") & (detection_table.method == "logistic")].iloc[0]
    detection_pass = bool(dr.events >= 5 and dr.recall is not None and dr.recall >= .8 and dr.precision is not None and dr.precision >= .8 and dr.fpr <= .01)
    routing_pass = bool(a.hours >= 5 and gain is not None and gain >= .05 and under_gain is not None and under_gain >= .05 and all_change <= .01 and peak_change <= .01)
    coverage_rows = []
    for month in [7, 8]:
        zz = z.loc[z.month == month]; hh = class_eval.loc[class_eval.month == month]; ff = f.loc[f.month == month]
        ev = events.loc[(events.month == month) & (events.direction == "up") & events.sustained3.eq(True)]
        coverage_rows.append({"month": month, "calendar_hours": len(zz), "valid_hours": int(zz.valid.sum()), "missing_hours": int((~zz.valid).sum()),
                              "six_history_prediction_hours": len(hh), "six_history_known_label_hours": int(hh.persistence_known.sum()),
                              "common_prediction_hours": len(ff), "common_known_label_hours": int(ff.persistence_known.sum()),
                              "all_observed_persistent_up_events": len(ev), "persistent_up_events_in_common": int((ff.sustained_onset == 1).sum()),
                              "excluded_persistent_up_events": int((~ev.timestamp.isin(ff.timestamp)).sum())})
    outputs = {"hourly_extended.csv": z, "events_extended.csv": events, "classification_predictions.csv": class_eval,
               "evaluation_frame.csv": f, "predictions.csv": p, "metrics.csv": table, "detection_metrics.csv": detection_table,
               "coverage.csv": pd.DataFrame(coverage_rows), "routing_burden.csv": pd.DataFrame(burden),
               "event_windows.csv": ew, "event_window_coverage.csv": pd.DataFrame(window_coverage), "daily_peak_errors.csv": daily,
               "model_manifest.csv": pd.DataFrame(manifest)}
    decision = {"classification_gate_passed": detection_pass, "routing_gate_passed": routing_pass,
                "start_mae_reduction": gain, "up_under_reduction": under_gain,
                "overall_mae_change": all_change, "daily_peak_mae_change": peak_change,
                "scope": c["scope"], "no_retuning": True}
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    jsave(OUT / "decision.json", decision)
    jsave(OUT / "run.json", {"status": "completed", "script_sha256": sha(__file__), "contract_sha256": sha(OUT / "contract.json"),
                             "inputs_sha256": c["inputs_sha256"], "new_fits": len(manifest), "outputs_sha256": {n: sha(OUT / n) for n in list(outputs) + ["decision.json"]}, "runtime": sys.version})
    print(json.dumps({"status": "completed", "coverage": coverage_rows, "decision": decision}), flush=True)


if __name__ == "__main__":
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    {"freeze": freeze, "run": run}[sys.argv[1]]()
