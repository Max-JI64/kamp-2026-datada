"""Independent checks for the single-feature ablation and reused HGB routing."""
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from regime_forecast_verify import average_precision, independent_threshold

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Modeling/tables/regime_age_ablation"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def close(a, b):
    if b is None or pd.isna(b):
        assert pd.isna(a), (a, b)
    else:
        assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-8), (a, b)


def calendar(f):
    f = f.copy()
    f["hour_sin"] = np.sin(2 * np.pi * f.timestamp.dt.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * f.timestamp.dt.hour / 24)
    for i in range(7):
        f[f"dow_{i}"] = (f.timestamp.dt.dayofweek == i).astype(int)
    return f


def errstats(g):
    errors = [float(r.prediction) - float(r.actual) for r in g.itertuples()]
    if not errors:
        return {n: None for n in ["mae", "rmse", "under", "over", "bias"]}
    n = len(errors)
    return {"mae": math.fsum(abs(e) for e in errors) / n,
            "rmse": math.sqrt(math.fsum(e * e for e in errors) / n),
            "under": math.fsum(max(-e, 0) for e in errors) / n,
            "over": math.fsum(max(e, 0) for e in errors) / n,
            "bias": math.fsum(errors) / n}


def main():
    c = json.loads((OUT / "contract.json").read_text(encoding="utf-8"))
    run = json.loads((OUT / "run.json").read_text(encoding="utf-8"))
    decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    assert c["entrypoint_sha256"] == run["script_sha256"] == sha(ROOT / "Modeling/scripts/regime_age_ablation.py")
    assert run["contract_sha256"] == sha(OUT / "contract.json") and run["new_regression_fits"] == 0
    for name, expected in c["inputs_sha256"].items():
        assert sha(ROOT / name) == expected
    for name, expected in run["outputs_sha256"].items():
        assert sha(OUT / name) == expected
    original = json.loads((ROOT / "Modeling/tables/regime_forecast/contract.json").read_text(encoding="utf-8"))
    assert c["features"] == [x for x in original["feature_names"] if x != "prior_state_age"]
    assert len(c["features"]) == 23 and c["logistic"] == original["logistic"]
    assert not set(c["features"]) & set(original["forbidden_inputs"])
    f = pd.read_csv(ROOT / "Modeling/tables/regime_diagnosis/hourly_labels.csv", encoding="utf-8-sig", parse_dates=["timestamp", "label_confirmed_at"])
    f = calendar(f.loc[f.eligible_onset & f.persistence_known].copy())
    dev = pd.read_csv(OUT / "development_classification.csv", encoding="utf-8-sig", parse_dates=["timestamp", "label_confirmed_at"])
    late = pd.read_csv(OUT / "followup_classification.csv", encoding="utf-8-sig", parse_dates=["timestamp", "label_confirmed_at"]) if decision["followup_executed"] else None
    frames = {"development": dev}
    if late is not None:
        frames["followup"] = late
    full = pd.read_csv(ROOT / "Modeling/tables/regime_followup/classification_predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp", "label_confirmed_at"])
    manifest = pd.read_csv(OUT / "model_manifest.csv", encoding="utf-8-sig")
    assert len(manifest) == run["new_fits"]
    refits = 0
    for r in manifest.itertuples():
        boundary = pd.Timestamp(r.boundary)
        train = f.loc[(f.timestamp < boundary) & (f.label_confirmed_at < boundary)]
        if r.name == "noage_fixed_june":
            test = full
            saved = late.loc[late.method == "noage"]
        else:
            m = int(r.name[-2:]); test = f.loc[f.month == m]
            saved = dev.loc[(dev.method == "noage") & (dev.month == m)]
        assert saved.timestamp.tolist() == test.timestamp.tolist()
        assert len(train) == r.train_hours and (train.sustained_onset == 1).sum() == r.train_events
        assert pd.Timestamp(r.train_latest_confirmed) == train.label_confirmed_at.max() < boundary
        assert sha(ROOT / r.file) == r.sha256 and r.features.split("|") == c["features"]
        loaded = joblib.load(ROOT / r.file)
        fresh = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=3000, random_state=42, class_weight=None))
        fresh.fit(train[c["features"]], (train.sustained_onset == 1).astype(int))
        for fitted in [loaded, fresh]:
            assert fitted.feature_names_in_.tolist() == c["features"]
            np.testing.assert_allclose(saved.score, fitted.predict_proba(test[c["features"]])[:, 1], rtol=1e-9, atol=1e-8)
            np.testing.assert_allclose(fitted[0].mean_, train[c["features"]].mean(), rtol=1e-9, atol=1e-8)
        refits += 1
    cal = pd.read_csv(OUT / "calibration.csv", encoding="utf-8-sig")
    for r in cal.itertuples():
        boundary = pd.Timestamp("2021-07-01") if r.period == "Jul-Aug" else pd.Timestamp(2021, int(r.period), 1)
        previous = dev.loc[(dev.method == "noage") & (dev.timestamp < boundary) & (dev.label_confirmed_at < boundary)]
        th, tp, fp, cap = independent_threshold(previous)
        close(r.threshold, th)
        assert (r.tp, r.fp, r.fp_cap, r.hours, r.events) == (tp, fp, cap, len(previous), previous.event.sum())
        assert previous.label_confirmed_at.max() == pd.Timestamp(r.latest_confirmed) < boundary
        target = late if r.period == "Jul-Aug" else dev.loc[dev.month == int(r.period)]
        target = target.loc[target.method == "noage"]
        assert target.threshold.eq(th).all()
        np.testing.assert_array_equal(target.alarm, (target.score >= th).astype(int))
    raw = {}
    with (ROOT / "data/origin/okm_augumented_2021.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            date, hour = int(row["날짜"]), int(row["시간"])
            if 20210101 <= date <= 20210831 and 0 <= hour < 24:
                t = datetime.strptime(str(date), "%Y%m%d") + timedelta(hours=hour)
                raw[t] = max(float(row[n]) for n in ["15분", "30분", "45분", "60분"])
    class_checks = regression_checks = 0
    for stage, cp in frames.items():
        period = "May-Jun" if stage == "development" else "Jul-Aug"
        pp = pd.read_csv(OUT / f"{stage}_predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
        parentdir = "regime_routing" if stage == "development" else "regime_followup"
        parent = pd.read_csv(ROOT / f"Modeling/tables/{parentdir}/predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
        candidate = cp.loc[cp.method == "noage"].set_index("timestamp")
        b0 = parent.loc[parent.variant == "B0"].set_index("timestamp")
        b1 = parent.loc[parent.variant == "B1_all"].set_index("timestamp")
        for variant, g in pp.groupby("variant"):
            assert g.timestamp.tolist() == b0.index.tolist()
            for r in g.itertuples():
                close(r.actual, raw[r.timestamp.to_pydatetime()])
                if variant == "G1_noage":
                    assert r.routed == int(candidate.loc[r.timestamp, "alarm"])
                    close(r.prediction, (b1 if r.routed else b0).loc[r.timestamp, "prediction"])
                else:
                    name = "G1" if variant == "G1_old" else variant
                    saved = parent.loc[(parent.variant == name) & (parent.timestamp == r.timestamp)].iloc[0]
                    close(r.prediction, saved.prediction); assert r.routed == saved.routed
        dt = pd.read_csv(OUT / f"{stage}_detection.csv", encoding="utf-8-sig")
        for r in dt.itertuples():
            g = cp.loc[(cp.method == r.method) & cp.event.notna() & cp.alarm.notna()]
            if stage == "development":
                g = g.loc[g.month >= 5]
            if r.scope == "common_regression":
                g = g.loc[g.timestamp.isin(b0.index)]
            if r.period != period:
                g = g.loc[g.month == int(r.period)]
            tp = fp = fn = tn = 0
            for rr in g.itertuples():
                if rr.event == 1:
                    tp += int(rr.alarm == 1); fn += int(rr.alarm == 0)
                else:
                    fp += int(rr.alarm == 1); tn += int(rr.alarm == 0)
            assert (r.tp, r.fp, r.fn, r.tn, r.hours, r.events) == (tp, fp, fn, tn, len(g), tp + fn)
            for name, value in {"recall": tp / (tp + fn), "precision": tp / (tp + fp) if tp + fp else None,
                                "fpr": fp / (fp + tn), "ap": average_precision(g)}.items():
                close(getattr(r, name), value)
            class_checks += 1
        mt = pd.read_csv(OUT / f"{stage}_metrics.csv", encoding="utf-8-sig")
        daily = pd.read_csv(OUT / f"{stage}_daily.csv", encoding="utf-8-sig")
        windows = pd.read_csv(OUT / f"{stage}_windows.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
        for r in daily.itertuples():
            g = pp.loc[(pp.variant == r.variant) & (pp.date == r.date)]
            assert len(g) == 24
            peak = g.loc[g.actual == g.actual.max()]
            assert r.ties == len(peak)
            for n, v in errstats(peak).items():
                close(getattr(r, n), v)
        for r in windows.itertuples():
            g = pp.loc[(pp.variant == r.variant) & pp.timestamp.isin(pd.date_range(r.timestamp, periods=3, freq="h"))]
            assert len(g) == 3
            for n, v in errstats(g).items():
                close(getattr(r, n), v)
        for r in mt.itertuples():
            g = pp.loc[pp.variant == r.variant]
            if r.period != period:
                g = g.loc[g.month == int(r.period)]
            if r.condition in ["all", "up_start", "down_start"]:
                if r.condition != "all":
                    g = g.loc[g.sustained_onset == (1 if r.condition == "up_start" else -1)]
                assert len(g) == r.hours
                stat = errstats(g)
            else:
                source = daily if r.condition == "daily_peak_day_equal" else windows
                z = source.loc[source.variant == r.variant]
                if r.period != period:
                    z = z.loc[z.month == int(r.period)]
                stat = {n: math.fsum(z[n]) / len(z) if len(z) else None for n in ["mae", "under", "over", "bias"]}
                stat["rmse"] = math.sqrt(math.fsum(z.rmse ** 2) / len(z)) if len(z) else None
                assert r.units == len(z)
            for name, value in stat.items():
                close(getattr(r, name), value)
            regression_checks += 1
        def get(variant, condition):
            return mt.loc[(mt.variant == variant) & (mt.period == period) & (mt.condition == condition)].iloc[0]
        gain = 1 - get("G1_noage", "up_start").mae / get("B0", "up_start").mae
        under = 1 - get("G1_noage", "up_start").under / get("B0", "up_start").under
        overall = get("G1_noage", "all").mae / get("B0", "all").mae - 1
        peak = get("G1_noage", "daily_peak_day_equal").mae / get("B0", "daily_peak_day_equal").mae - 1
        dr = dt.loc[(dt.scope == "common_regression") & (dt.period == period) & (dt.method == "noage")].iloc[0]
        expected = {"classification_passed": bool(dr.events >= 5 and dr.recall >= .8 and dr.precision >= .8 and dr.fpr <= .01),
                    "routing_passed": bool(dr.events >= 5 and gain >= .05 and under >= .05 and overall <= .01 and peak <= .01),
                    "start_mae_reduction": gain, "under_reduction": under,
                    "overall_mae_change": overall, "daily_peak_mae_change": peak}
        for name, value in expected.items():
            close(decision[stage][name], value)
    assert decision["followup_executed"] == (decision["development"]["classification_passed"] and decision["development"]["routing_passed"])
    assert c["root_readme_sha256"] == sha(ROOT / "README.md") and c["manuscript_sha256"] == sha(ROOT / "Modeling/04_Modeling_원고.md")
    result = {"status": "passed", "run_sha256": sha(OUT / "run.json"), "verifier_sha256": sha(__file__),
              "models_reloaded": len(manifest), "independent_classifier_refits": refits,
              "classifier_metrics_checked": class_checks, "regression_metrics_checked": regression_checks,
              "new_regression_fits": 0, "raw_valid_hours": len(raw), "extra_outputs_sha256": {}}
    (OUT / "independent_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    main()
