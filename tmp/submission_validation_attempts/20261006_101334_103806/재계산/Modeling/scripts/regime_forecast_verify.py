"""Independent S03 chronology, tie-aware AP, thresholds, selection and model checks."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import hashlib
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Modeling/tables/regime_forecast"
PARENT = ROOT / "Modeling/tables/regime_diagnosis"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def close(a, b):
    if pd.isna(a) or b is None:
        assert pd.isna(a) and b is None, (a, b)
    else:
        assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-9), (a, b)


def average_precision(rows):
    pairs = sorted(zip(rows.score, rows.event), reverse=True)
    positives = sum(y for _, y in pairs)
    if not positives:
        return None
    tp = total = 0
    numerator = 0.
    for score in sorted(set(s for s, _ in pairs), reverse=True):
        group = [y for s, y in pairs if s == score]
        new_tp = sum(group)
        total += len(group); tp += new_tp
        numerator += new_tp * tp / total
    return numerator / positives


def metrics(g):
    result = {"hours": len(g), "events": int(g.event.sum())}
    if g.alarm.notna().all():
        tp = fp = fn = tn = short_fp = 0
        for r in g.itertuples():
            if r.event:
                tp += int(r.alarm == 1); fn += int(r.alarm == 0)
            else:
                fp += int(r.alarm == 1); tn += int(r.alarm == 0)
                short_fp += int(r.alarm == 1 and r.short_event == 1)
        result.update(tp=tp, fp=fp, fn=fn, tn=tn, recall=tp / (tp + fn) if tp + fn else None,
                      precision=tp / (tp + fp) if tp + fp else None, fp_per100h=100 * fp / len(g),
                      fpr=fp / (fp + tn) if fp + tn else None, short_event_fp=short_fp)
    return result


def check_stats(record, expected):
    for field, value in expected.items():
        close(getattr(record, field), value)


def independent_threshold(g):
    cap = int((len(g) - g.event.sum()) // 100)
    choices = []
    for threshold in [math.inf] + sorted(g.score.unique().tolist(), reverse=True):
        alerted = g.loc[g.score >= threshold]
        tp = int(alerted.event.sum()); fp = len(alerted) - tp
        if fp <= cap:
            choices.append((tp, -fp, threshold))
    best = max(choices)
    return best[2], best[0], -best[1], cap


def main():
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    c = json.loads((OUT / "contract.json").read_text(encoding="utf-8"))
    run = json.loads((OUT / "run.json").read_text(encoding="utf-8"))
    assert c["entrypoint_sha256"] == run["script_sha256"] == sha(ROOT / "Modeling/scripts/regime_forecast.py")
    assert run["contract_sha256"] == sha(OUT / "contract.json") and run["status"] == "completed"
    for name, digest in {**c["inputs_sha256"], **run["inputs_sha256"]}.items():
        assert sha(ROOT / name) == digest
    for name, digest in run["outputs_sha256"].items():
        assert sha(OUT / name) == digest
    parent_run = json.loads((PARENT / "run.json").read_text(encoding="utf-8"))
    parent_verify = json.loads((PARENT / "independent_verification.json").read_text(encoding="utf-8"))
    assert parent_verify["status"] == "passed" and parent_verify["run_sha256"] == sha(PARENT / "run.json")
    for name, digest in parent_run["outputs_sha256"].items():
        assert sha(PARENT / name) == digest
    parent_contract = json.loads((PARENT / "contract.json").read_text(encoding="utf-8"))
    assert sha(ROOT / "data/origin/okm_augumented_2021.csv") == parent_contract["source_sha256"]
    full_frame = pd.read_csv(PARENT / "hourly_labels.csv", encoding="utf-8-sig", parse_dates=["timestamp", "label_confirmed_at", "latest_input_timestamp"])
    f = full_frame.loc[full_frame.eligible_onset & full_frame.persistence_known].copy()
    features = c["feature_names"]
    assert not set(features) & set(c["forbidden_inputs"])
    f["hour_sin"] = [math.sin(2 * math.pi * int(h) / 24) for h in f.hour]
    f["hour_cos"] = [math.cos(2 * math.pi * int(h) / 24) for h in f.hour]
    for i in range(7):
        f[f"dow_{i}"] = (f.timestamp.dt.weekday == i).astype(int)
    p = pd.read_csv(OUT / "predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp", "label_confirmed_at", "latest_input_timestamp"])
    assert not p.duplicated(["direction", "method", "timestamp"]).any()
    assert (p.latest_input_timestamp < p.timestamp).all() and p.timestamp.max() < pd.Timestamp("2021-07-01")
    folds = pd.read_csv(OUT / "folds.csv", encoding="utf-8-sig")
    manifest = pd.read_csv(OUT / "model_manifest.csv", encoding="utf-8-sig")
    model_reloads = 0; refits = 0
    # Every fold/model is reloaded and its probabilities are checked against a separately built frame.
    for fold in folds.itertuples():
        sign = 1 if fold.direction == "up" else -1
        boundary = pd.Timestamp(2021, fold.month, 1)
        tr = f.loc[(f.timestamp < boundary) & (f.label_confirmed_at < boundary)]
        te = f.loc[f.month == fold.month]
        y = (tr.sustained_onset == sign).astype(int).to_numpy()
        assert len(tr) == fold.train_hours and int(y.sum()) == fold.train_events
        assert tr.loc[y == 1, "date"].nunique() == fold.positive_dates
        assert str(tr.label_confirmed_at.max()) == fold.train_latest_confirmed
        assert tr.label_confirmed_at.max() < boundary and len(te) == fold.test_hours
        available = int(y.sum()) >= 8 and fold.positive_dates >= 5 and (y == 0).any()
        assert available == fold.ml_available
        g = p.loc[(p.direction == fold.direction) & (p.month == fold.month)]
        assert g.method.nunique() == (5 if available else 3)
        for method, gg in g.groupby("method"):
            assert gg.timestamp.tolist() == te.timestamp.tolist()
            assert gg.event.tolist() == (te.sustained_onset == sign).astype(int).tolist()
            assert gg.short_event.tolist() == ((te.onset == sign) & (te.sustained_onset != sign)).astype(int).tolist()
            assert gg.label_confirmed_at.tolist() == te.label_confirmed_at.tolist()
            if method == "last_delta":
                np.testing.assert_allclose(gg.score, sign * te.prior_delta, rtol=0, atol=1e-9)
            elif method == "slope6":
                np.testing.assert_allclose(gg.score, sign * te.past_slope6, rtol=0, atol=1e-9)
            elif method == "calendar":
                values = []
                for r in te.itertuples():
                    group = tr.loc[(tr.hour == r.hour) & (tr.weekend == r.weekend)]
                    if len(group) < 20:
                        group = tr.loc[tr.hour == r.hour]
                    if len(group) < 20:
                        group = tr
                    values.append(((group.sustained_onset == sign).sum() + .5) / (len(group) + 1))
                np.testing.assert_allclose(gg.score, values, rtol=0, atol=1e-9)
            else:
                record = manifest.loc[(manifest.direction == fold.direction) & (manifest.month == fold.month) & (manifest.method == method)].iloc[0]
                assert sha(ROOT / record.file) == record.sha256 and record.features == "|".join(features)
                fitted = joblib.load(ROOT / record.file)
                np.testing.assert_allclose(gg.score, fitted.predict_proba(te[features])[:, 1], rtol=1e-8, atol=1e-9)
                model_reloads += 1
                if (fold.direction, fold.month, method) in [("up", 5, "logistic"), ("down", 6, "hgb")]:
                    fresh = (make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=3000, random_state=42))
                             if method == "logistic" else HistGradientBoostingClassifier(max_leaf_nodes=7, max_iter=100,
                                 learning_rate=.05, min_samples_leaf=20, l2_regularization=1., early_stopping=False, random_state=42))
                    fresh.fit(tr[features], y)
                    np.testing.assert_allclose(gg.score, fresh.predict_proba(te[features])[:, 1], rtol=1e-8, atol=1e-9)
                    refits += 1
    selection = pd.read_csv(OUT / "selection.csv", encoding="utf-8-sig").fillna({"selected_simple": "", "selected_ml": "", "calibration_latest_confirmed": ""})
    thresholds = pd.read_csv(OUT / "calibration.csv", encoding="utf-8-sig")
    thresholds_checked = 0
    for r in selection.itertuples():
        boundary = pd.Timestamp(2021, r.month, 1)
        prior = p.loc[(p.direction == r.direction) & (p.month < r.month) & (p.label_confirmed_at < boundary)]
        complete_times = prior.groupby("timestamp").method.nunique()
        cal = prior.loc[prior.timestamp.isin(complete_times.index[complete_times == 5])]
        n = len(cal.loc[cal.method == "calendar"])
        positives = int(cal.loc[cal.method == "calendar", "event"].sum())
        assert n == r.calibration_hours and positives == r.calibration_events
        if n:
            assert str(cal.label_confirmed_at.max()) == r.calibration_latest_confirmed
        if r.calibration_available:
            assert positives > 0
            for family, methods in [("simple", ["calendar", "last_delta", "slope6"]), ("ml", ["logistic", "hgb"])]:
                weighted = []
                for method in methods:
                    mc = cal.loc[cal.method == method]
                    terms = [(int(gg.event.sum()), average_precision(gg)) for _, gg in mc.groupby("month")]
                    score = sum(k * value for k, value in terms if k) / sum(k for k, _ in terms)
                    weighted.append(score)
                    threshold, tp, fp, cap = independent_threshold(mc)
                    rec = thresholds.loc[(thresholds.direction == r.direction) & (thresholds.month == r.month) & (thresholds.method == method)].iloc[0]
                    close(rec.threshold, threshold); close(rec.weighted_monthly_ap, score)
                    assert (rec.tp, rec.fp, rec.fp_cap, rec.hours, rec.events) == (tp, fp, cap, n, positives)
                    evaluated = p.loc[(p.direction == r.direction) & (p.month == r.month) & (p.method == method)]
                    assert (evaluated.threshold == threshold).all()
                    assert evaluated.alarm.tolist() == (evaluated.score >= threshold).astype(int).tolist()
                    thresholds_checked += 1
                picked = methods[max(range(len(methods)), key=lambda i: (weighted[i], -i))]
                assert picked == getattr(r, "selected_" + family)
        else:
            g = p.loc[(p.direction == r.direction) & (p.month == r.month)]
            assert g.alarm.isna().all() and g.threshold.isna().all()
    for r in pd.read_csv(OUT / "monthly_metrics.csv", encoding="utf-8-sig").itertuples():
        g = p.loc[(p.direction == r.direction) & (p.method == r.method) & (p.month == r.month)]
        check_stats(r, {**metrics(g), "ap": average_precision(g)})
    selected = p.loc[(p.month >= 4) & p.selected_family.notna()].copy()
    saved_selected = pd.read_csv(OUT / "selected_predictions.csv", encoding="utf-8-sig")
    assert saved_selected.timestamp.tolist() == selected.timestamp.astype(str).tolist()
    summary = pd.read_csv(OUT / "selected_summary.csv", encoding="utf-8-sig")
    for r in summary.itertuples():
        g = selected.loc[(selected.direction == r.direction) & (selected.selected_family == r.family)]
        check_stats(r, metrics(g))
    for r in pd.read_csv(OUT / "profile_conditions.csv", encoding="utf-8-sig").itertuples():
        g = selected.loc[(selected.direction == r.direction) & (selected.selected_family == r.family) & (selected.seen_profile == r.seen_profile)]
        check_stats(r, metrics(g))
    decisions = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    for r in decisions["directions"]:
        a = summary.loc[(summary.direction == r["direction"]) & (summary.family == "ml")].iloc[0]
        b = summary.loc[(summary.direction == r["direction"]) & (summary.family == "simple")].iloc[0]
        extra = int(a.tp - b.tp); relative = extra / b.tp if b.tp else None
        close(r["extra_tp"], extra); close(r["relative_extra_tp"], relative)
        close(r["extra_fp_per100h"], a.fp_per100h - b.fp_per100h)
        assert r["development_gate_passed"] == (extra >= 2 and (relative is None or relative >= .20) and a.fp_per100h - b.fp_per100h <= .5)
    # Check full saved event/late-alarm diagnosis separately, and expose case concentration.
    ep = pd.read_csv(OUT / "event_predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    assert len(ep) == len(p.loc[(p.month >= 4) & (p.event == 1)])
    keyed = {(r.direction, r.method, r.timestamp): r for r in p.itertuples()}
    for r in ep.itertuples():
        assert (r.direction, r.method, r.timestamp) in keyed and keyed[(r.direction, r.method, r.timestamp)].event == 1
        for k in [1, 2]:
            other = keyed.get((r.direction, r.method, r.timestamp + pd.Timedelta(hours=k)))
            actual = getattr(r, "alarm_one_hour_late" if k == 1 else "alarm_two_hours_late")
            expected = None if other is None or pd.isna(other.alarm) else other.alarm
            close(actual, expected)
    cases = []
    selected_ml = selected.loc[(selected.selected_family == "ml") & ((selected.event == 1) | (selected.alarm == 1))]
    dynamic = [x for x in features if x.startswith("prior_") or x.startswith("past_")]
    for r in selected_ml.itertuples():
        row = f.loc[f.timestamp == r.timestamp].iloc[0]
        tr = f.loc[(f.timestamp < pd.Timestamp(2021, r.month, 1)) & (f.label_confirmed_at < pd.Timestamp(2021, r.month, 1))]
        identical = np.isclose(tr[dynamic].to_numpy(float), row[dynamic].to_numpy(float), rtol=0, atol=1e-9).all(axis=1)
        future = full_frame.loc[full_frame.timestamp.between(r.timestamp, r.timestamp + pd.Timedelta(hours=2))]
        assert len(future) == 3
        cases.append({"timestamp": r.timestamp, "direction": r.direction, "method": r.method, "event": r.event,
                      "alarm": r.alarm, "score": r.score, "threshold": r.threshold, "seen_daily_profile": r.seen_profile,
                      "identical_dynamic_input_training_rows": int(identical.sum()), "prior_mean": row.prior_mean,
                      "start_mean": row.level, "start_maximum": row.maximum, "observed_next3_mean": float(future.level.mean()),
                      "prior_delta": row.prior_delta, "prior_last_minus_mean": row.prior_last_minus_mean,
                      "past_slope6": row.past_slope6, "prior_state_age": row.prior_state_age, "profile": row.profile})
    cases = pd.DataFrame(cases)
    cases.to_csv(OUT / "verified_cases.csv", index=False, encoding="utf-8-sig")
    support = []
    for direction, gg in cases.loc[cases.event == 1].groupby("direction"):
        support.append({"direction": direction, "events": len(gg), "detected": int(gg.alarm.sum()),
                        "positive_dates": gg.timestamp.dt.date.nunique(), "distinct_daily_profiles": gg.profile.nunique(),
                        "events_with_identical_dynamic_training_input": int((gg.identical_dynamic_input_training_rows > 0).sum()),
                        "new_daily_profile_events": int((gg.seen_daily_profile == 0).sum())})
    pd.DataFrame(support).to_csv(OUT / "verified_support.csv", index=False, encoding="utf-8-sig")
    result = {"status": "passed", "run_sha256": sha(OUT / "run.json"), "verifier_sha256": sha(__file__),
              "prediction_rows_checked": len(p), "folds_checked": len(folds), "thresholds_checked": thresholds_checked,
              "models_reloaded": model_reloads, "independent_refits": refits, "event_rows_checked": len(ep),
              "forbidden_features_absent": True, "mature_labels_only": True,
              "extra_outputs_sha256": {n: sha(OUT / n) for n in ["verified_cases.csv", "verified_support.csv"]}}
    (OUT / "independent_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
