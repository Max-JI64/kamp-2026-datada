"""Independent raw-source/state/timing/model/error verification of fixed S05."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from regime_diagnose_verify import reconstruct
from regime_forecast_verify import average_precision

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Modeling/tables/regime_followup"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def close(a, b):
    if pd.isna(a) or b is None:
        assert pd.isna(a) and b is None, (a, b)
    else:
        assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-8), (a, b)


def stats(g):
    err = [float(r.prediction) - float(r.actual) for r in g.itertuples()]
    if not err:
        return {"hours": 0, **{n: None for n in ["mae", "rmse", "bias", "under", "over"]}}
    return {"hours": len(err), "mae": math.fsum(abs(e) for e in err) / len(err),
            "rmse": math.sqrt(math.fsum(e * e for e in err) / len(err)), "bias": math.fsum(err) / len(err),
            "under": math.fsum(max(-e, 0) for e in err) / len(err), "over": math.fsum(max(e, 0) for e in err) / len(err)}


def main():
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    c = json.loads((OUT / "contract.json").read_text(encoding="utf-8"))
    run = json.loads((OUT / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed" and run["contract_sha256"] == sha(OUT / "contract.json")
    assert c["entrypoint_sha256"] == run["script_sha256"] == sha(ROOT / "Modeling/scripts/regime_followup.py")
    for name, digest in c["inputs_sha256"].items():
        assert sha(ROOT / name) == digest
    for name, digest in run["outputs_sha256"].items():
        assert sha(OUT / name) == digest
    raw = {}
    with (ROOT / "data/origin/okm_augumented_2021.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            date, hour = int(row["날짜"]), int(row["시간"])
            if not (20210101 <= date <= 20210831 and 0 <= hour < 24):
                continue
            t = datetime.strptime(str(date), "%Y%m%d") + timedelta(hours=hour)
            assert t not in raw
            slots = [float(row[n]) for n in ["15분", "30분", "45분", "60분"]]
            raw[t] = {"slots": slots, "mean": math.fsum(slots) / 4, "maximum": max(slots), "production": float(row["생산량"])}
    times = [datetime(2021, 1, 1) + timedelta(hours=i) for i in range(243 * 24)]
    rebuilt, starts = reconstruct(raw, times, c["onset_definition"]["thresholds"])
    z = pd.read_csv(OUT / "hourly_extended.csv", encoding="utf-8-sig", parse_dates=["timestamp", "label_confirmed_at", "latest_input_timestamp"])
    assert z.timestamp.tolist() == times
    for r in z.itertuples():
        t = r.timestamp.to_pydatetime(); expect = rebuilt[t]
        assert r.valid == (t in raw) and r.eligible_onset == expect["eligible"]
        assert r.candidate == expect["candidate"] and r.onset == expect["onset"]
        assert r.prior_state_age == expect["state"]["age"] and r.prior_state_direction == expect["state"]["direction"]
        assert r.prior_state_left_censored == expect["state"]["censored"]
        assert r.latest_input_timestamp == t - timedelta(hours=1) and r.label_confirmed_at == t + timedelta(hours=2)
        known = all(t + timedelta(hours=k) in raw for k in [0, 1, 2])
        assert r.persistence_known == known
        target = None if not known else 0
        if known and expect["onset"]:
            sign = expect["onset"]; baseline = expect["features"]["past_median6"]
            boundary = c["onset_definition"]["thresholds"]["up" if sign == 1 else "down"] / 2
            target = sign if all(sign * (raw[t + timedelta(hours=k)]["mean"] - baseline) >= boundary for k in [0, 1, 2]) else 0
        close(r.sustained_onset, target)
        if t in raw:
            close(r.level, raw[t]["mean"]); close(r.maximum, raw[t]["maximum"])
        if expect["eligible"]:
            for feature, value in expect["features"].items():
                close(getattr(r, feature), value)
    old = pd.read_csv(ROOT / "Modeling/tables/regime_diagnosis/hourly_labels.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    joined = old.merge(z, on="timestamp", suffixes=("_old", "_new"))
    for name in ["onset", "candidate", "prior_state_age", "prior_state_direction", "prior_state_left_censored"]:
        assert joined[name + "_old"].tolist() == joined[name + "_new"].tolist()
    cls = pd.read_csv(OUT / "classification_predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    frame = pd.read_csv(OUT / "evaluation_frame.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    ct = z.loc[z.eligible_onset & z.persistence_known & (z.label_confirmed_at < "2021-07-01")]
    assert len(ct) == 4336 and ct.timestamp.max() < pd.Timestamp("2021-07-01")
    assert cls.timestamp.tolist() == z.loc[(z.month >= 7) & z.eligible_onset, "timestamp"].tolist()
    rt = pd.read_csv(ROOT / "Modeling/tables/regime_integration/meta_features.csv", encoding="utf-8-sig", parse_dates=["timestamp", "label_confirmed_at"])
    rt = rt.loc[rt.label_confirmed_at < "2021-07-01"]
    m = pd.read_csv(ROOT / "Modeling/tables/m01/hourly_frame.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    expected_times = sorted(set(cls.timestamp) & set(m.loc[(m.month >= 7) & m.eligible_common, "timestamp"]))
    assert frame.timestamp.tolist() == expected_times and len(rt) == 3598
    original = m.set_index("timestamp").loc[frame.timestamp]
    for name in c["regression_features"]["B0"] + ["target_maximum"]:
        np.testing.assert_allclose(frame[name], original[name], rtol=0, atol=1e-8)
    assert (frame.up_classifier_available == 1).all() and (frame.down_classifier_available == 1).all()
    for features in [c["classification_features"], *c["regression_features"].values()]:
        assert not set(features) & {"event", "target_maximum", "maximum", "level", "onset", "sustained_onset", "profile", "delta", "deviation"}
    manifest = pd.read_csv(OUT / "model_manifest.csv", encoding="utf-8-sig")
    models = {}; fresh_predictions = {}; refits = 0
    for r in manifest.itertuples():
        assert sha(ROOT / r.file) == r.sha256 and r.trained_before == "2021-07-01"
        fitted = joblib.load(ROOT / r.file); models[r.name] = fitted
        if r.name == "up_logistic":
            features = c["classification_features"]; y = (ct.sustained_onset == 1).astype(int)
            np.testing.assert_allclose(cls.score_logistic, fitted.predict_proba(cls[features])[:, 1], atol=1e-8, rtol=1e-9)
            fresh = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=3000, random_state=42))
            fresh.fit(ct[features], y)
            np.testing.assert_allclose(cls.score_logistic, fresh.predict_proba(cls[features])[:, 1], atol=1e-8, rtol=1e-9)
            assert r.train_hours == len(ct)
        else:
            features = c["regression_features"][r.name]; fresh = HistGradientBoostingRegressor(**c["hgb_parameters"])
            fresh.fit(rt[features], rt.target_maximum)
            fresh_predictions[r.name] = fresh.predict(frame[features])
            np.testing.assert_allclose(fresh_predictions[r.name], fitted.predict(frame[features]), atol=1e-8, rtol=1e-9)
            assert r.train_hours == len(rt)
        refits += 1
    cal = pd.read_csv(ROOT / "Modeling/tables/regime_forecast/predictions.csv", encoding="utf-8-sig", parse_dates=["label_confirmed_at"])
    cal = cal.loc[(cal.direction == "up") & cal.month.between(4, 6) & (cal.label_confirmed_at < "2021-07-01")]
    for method in ["logistic", "calendar"]:
        mc = cal.loc[cal.method == method]; cap = int((len(mc) - mc.event.sum()) // 100)
        candidates = []
        for threshold in [math.inf] + sorted(mc.score.unique(), reverse=True):
            al = mc.loc[mc.score >= threshold]; tp = int(al.event.sum()); fp = len(al) - tp
            if fp <= cap:
                candidates.append((tp, -fp, threshold))
        tp, negfp, th = max(candidates); rec = c["classification_thresholds"][method]
        actual_th = math.inf if rec["threshold"] == "infinity" else rec["threshold"]
        close(th, actual_th); assert (tp, -negfp, cap, len(mc)) == (rec["tp"], rec["fp"], rec["fp_cap"], rec["hours"])
        assert cls["alarm_" + method].tolist() == (cls["score_" + method] >= actual_th).astype(int).tolist()
    # Calendar probabilities use only the independently rebuilt Jan-Jun labels.
    for r in cls.itertuples():
        group = ct.loc[(ct.hour == r.hour) & (ct.weekend == r.weekend)]
        if len(group) < 20:
            group = ct.loc[ct.hour == r.hour]
        if len(group) < 20:
            group = ct
        close(r.score_calendar, ((group.sustained_onset == 1).sum() + .5) / (len(group) + 1))
    p = pd.read_csv(OUT / "predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    assert len(p) == 5 * len(frame) and not p.duplicated(["timestamp", "variant"]).any()
    for variant, g in p.groupby("variant", sort=False):
        assert g.timestamp.tolist() == frame.timestamp.tolist()
        b0, b1 = fresh_predictions["B0"], fresh_predictions["B1"]
        mask = {"B0": np.zeros(len(frame), bool), "B1_all": np.ones(len(frame), bool), "G1": frame.alarm_logistic == 1,
                "hour08": frame.timestamp.dt.hour == 8, "simple_gate": frame.alarm_calendar == 1}[variant]
        np.testing.assert_allclose(g.prediction, np.where(mask, b1, b0), atol=1e-8, rtol=1e-9)
        assert g.routed.tolist() == np.asarray(mask, int).tolist()
    daily = pd.read_csv(OUT / "daily_peak_errors.csv", encoding="utf-8-sig")
    windows = pd.read_csv(OUT / "event_windows.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    for r in daily.itertuples():
        g = p.loc[(p.variant == r.variant) & (p.date == r.date)]; assert len(g) == 24
        peak = g.loc[g.actual == g.actual.max()]; assert len(peak) == r.ties
        for name, value in stats(peak).items():
            close(getattr(r, name), value)
    for r in windows.itertuples():
        g = p.loc[(p.variant == r.variant) & p.timestamp.isin(pd.date_range(r.timestamp, periods=3, freq="h"))]; assert len(g) == 3
        for name, value in stats(g).items():
            close(getattr(r, name), value)
    table = pd.read_csv(OUT / "metrics.csv", encoding="utf-8-sig", dtype={"period": str})
    for r in table.itertuples():
        g = p.loc[p.variant == r.variant]
        if r.period != "Jul-Aug":
            g = g.loc[g.month == int(r.period)]
        if r.condition == "up_start":
            g = g.loc[g.sustained_onset == 1]
        elif r.condition == "down_start":
            g = g.loc[g.sustained_onset == -1]
        elif r.condition == "other_known_hours":
            g = g.loc[g.persistence_known & (g.sustained_onset != 1)]
        elif r.condition.startswith("profile_"):
            g = g.loc[g.seen_regression_profile == int(r.condition[-1])]
        if r.condition.endswith("_equal"):
            source = daily if r.condition.startswith("daily") else windows.loc[windows.direction == "up"]
            g = source.loc[source.variant == r.variant]
            if r.period != "Jul-Aug":
                g = g.loc[g.month == int(r.period)]
            expected = {n: math.fsum(g[n]) / len(g) if len(g) else None for n in ["mae", "bias", "under", "over"]}
            expected["rmse"] = math.sqrt(math.fsum(v * v for v in g.rmse) / len(g)) if len(g) else None
            expected["hours"] = int(g.ties.sum()) if r.condition.startswith("daily") else 3 * len(g)
            assert r.units == len(g)
        else:
            expected = stats(g)
        for name, value in expected.items():
            close(getattr(r, name), value)
    dt = pd.read_csv(OUT / "detection_metrics.csv", encoding="utf-8-sig", dtype={"period": str})
    for r in dt.itertuples():
        g = cls.loc[cls.persistence_known]
        if r.scope == "common_regression":
            g = g.loc[g.timestamp.isin(frame.timestamp)]
        if r.period != "Jul-Aug":
            g = g.loc[g.month == int(r.period)]
        al = g["alarm_" + r.method]; y = g.event.astype(int)
        tp = int(((y == 1) & (al == 1)).sum()); fp = int(((y == 0) & (al == 1)).sum()); fn = int(y.sum()) - tp; tn = len(g) - int(y.sum()) - fp
        assert (r.hours, r.events, r.tp, r.fp, r.fn, r.tn) == (len(g), y.sum(), tp, fp, fn, tn)
        close(r.recall, tp / (tp + fn) if tp + fn else None); close(r.precision, tp / (tp + fp) if tp + fp else None)
        close(r.fpr, fp / (fp + tn) if fp + tn else None)
        close(r.ap, average_precision(g.assign(score=g["score_" + r.method])))
    for r in pd.read_csv(OUT / "routing_burden.csv", encoding="utf-8-sig", dtype={"period": str}).itertuples():
        g = p.loc[p.variant == r.variant]
        if r.period != "Jul-Aug":
            g = g.loc[g.month == int(r.period)]
        assert r.hours == len(g) and r.routed == int(g.routed.sum())
        assert r.routed_up_starts == int(((g.routed == 1) & (g.sustained_onset == 1)).sum())
        assert r.routed_non_up_known == int(((g.routed == 1) & g.persistence_known & (g.sustained_onset != 1)).sum())
        assert r.routed_unknown == int(((g.routed == 1) & ~g.persistence_known).sum())
    cov = pd.read_csv(OUT / "coverage.csv", encoding="utf-8-sig")
    for r in cov.itertuples():
        full = z.loc[z.month == r.month]; own = frame.loc[frame.month == r.month]; six = cls.loc[cls.month == r.month]
        assert r.calendar_hours == len(full) and r.valid_hours == full.valid.sum() and r.missing_hours == (~full.valid).sum()
        assert r.common_prediction_hours == len(own) and r.common_known_label_hours == own.persistence_known.sum()
        assert r.six_history_prediction_hours == len(six) and r.six_history_known_label_hours == six.persistence_known.sum()
        all_events = full.loc[full.sustained_onset == 1]
        assert r.all_observed_persistent_up_events == len(all_events) and r.persistent_up_events_in_common == (own.sustained_onset == 1).sum()
        assert r.excluded_persistent_up_events == (~all_events.timestamp.isin(own.timestamp)).sum()
    d = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    def get(v, condition):
        return table.loc[(table.variant == v) & (table.period == "Jul-Aug") & (table.condition == condition)].iloc[0]
    gain = 1 - get("G1", "up_start").mae / get("B0", "up_start").mae
    under = 1 - get("G1", "up_start").under / get("B0", "up_start").under
    overall = get("G1", "all").mae / get("B0", "all").mae - 1
    peak = get("G1", "daily_peak_day_equal").mae / get("B0", "daily_peak_day_equal").mae - 1
    for name, value in [("start_mae_reduction", gain), ("up_under_reduction", under), ("overall_mae_change", overall), ("daily_peak_mae_change", peak)]:
        close(d[name], value)
    dr = dt.loc[(dt.scope == "common_regression") & (dt.period == "Jul-Aug") & (dt.method == "logistic")].iloc[0]
    assert d["classification_gate_passed"] == bool(dr.events >= 5 and dr.recall >= .8 and dr.precision >= .8 and dr.fpr <= .01)
    assert d["routing_gate_passed"] == bool(get("G1", "up_start").hours >= 5 and gain >= .05 and under >= .05 and overall <= .01 and peak <= .01)
    prefix_tests = 0
    for cut in [datetime(2021, 7, 10), datetime(2021, 8, 10)]:
        trimmed, _ = reconstruct({t: v for t, v in raw.items() if t < cut}, [t for t in times if t < cut], c["onset_definition"]["thresholds"])
        for t, item in trimmed.items():
            assert item["state"] == rebuilt[t]["state"] and item["onset"] == rebuilt[t]["onset"]
        prefix_tests += 1
    # Diagnostic only: linear logit contribution and range of features at each known-label alarm.
    scaler, fitted = models["up_logistic"].steps[0][1], models["up_logistic"].steps[1][1]
    features = c["classification_features"]
    alarm_cases = cls.loc[cls.persistence_known & (cls.alarm_logistic == 1)].copy()
    contrib = scaler.transform(alarm_cases[features]) * fitted.coef_[0]
    age_index = features.index("prior_state_age")
    age_min, age_max = float(ct.prior_state_age.min()), float(ct.prior_state_age.max())
    alarm_cases["age_logit_contribution"] = contrib[:, age_index]
    alarm_cases["largest_positive_logit_feature"] = [features[i] for i in contrib.argmax(axis=1)]
    alarm_cases["age_above_training_max"] = alarm_cases.prior_state_age > age_max
    alarm_cases["classification_training_age_max"] = age_max
    # No counterfactual prediction, re-fit, threshold change, or result cherry-picking.
    alarm_cases.to_csv(OUT / "verified_alarm_cases.csv", index=False, encoding="utf-8-sig")
    contribution_summary = []
    for (month, event), g in alarm_cases.groupby(["month", "event"]):
        contribution_summary.append({"month": int(month), "true_event": int(event), "alarms": len(g),
                                     "prior_state_age_min": float(g.prior_state_age.min()), "prior_state_age_max": float(g.prior_state_age.max()),
                                     "age_above_training_max": int(g.age_above_training_max.sum()),
                                     "mean_age_logit_contribution": float(g.age_logit_contribution.mean()),
                                     "age_largest_positive_feature": int((g.largest_positive_logit_feature == "prior_state_age").sum())})
    pd.DataFrame(contribution_summary).to_csv(OUT / "verified_alarm_conditions.csv", index=False, encoding="utf-8-sig")
    result = {"status": "passed", "run_sha256": sha(OUT / "run.json"), "verifier_sha256": sha(__file__),
              "raw_hours_checked": len(raw), "hourly_state_rows_checked": len(z), "regression_predictions_checked": len(p),
              "classification_prediction_hours": len(cls), "models_reloaded": len(models), "independent_refits": refits,
              "metrics_checked": len(table), "detection_metrics_checked": len(dt), "daily_rows_checked": len(daily),
              "prefix_tests": prefix_tests, "no_july_august_training": True,
              "extra_outputs_sha256": {n: sha(OUT / n) for n in ["verified_alarm_cases.csv", "verified_alarm_conditions.csv"]}}
    (OUT / "independent_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
