"""Independent S04 joining, OOF chronology, fixed model and scalar error verification."""
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
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Modeling/tables/regime_integration"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-8), (a, b)


def stats(g):
    errors = [float(r.prediction) - float(r.actual) for r in g.itertuples()]
    return {"hours": len(errors), "weight_sum": len(errors), "mae": math.fsum(abs(e) for e in errors) / len(errors),
            "rmse": math.sqrt(math.fsum(e * e for e in errors) / len(errors)),
            "bias": math.fsum(errors) / len(errors), "under": math.fsum(max(-e, 0) for e in errors) / len(errors),
            "over": math.fsum(max(e, 0) for e in errors) / len(errors)}


def main():
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    c = json.loads((OUT / "contract.json").read_text(encoding="utf-8"))
    run = json.loads((OUT / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed" and run["contract_sha256"] == sha(OUT / "contract.json")
    assert run["entrypoint_sha256"] == c["entrypoint_sha256"] == sha(ROOT / "Modeling/scripts/regime_integrate.py")
    for name, digest in c["inputs_sha256"].items():
        assert sha(ROOT / name) == digest
    for name, digest in run["outputs_sha256"].items():
        assert sha(OUT / name) == digest
    m1c = json.loads((ROOT / "Modeling/config/m01_contract.json").read_text(encoding="utf-8"))
    assert sha(ROOT / m1c["source"]) == m1c["source_sha256"]
    s3verify = json.loads((ROOT / "Modeling/tables/regime_forecast/independent_verification.json").read_text(encoding="utf-8"))
    assert s3verify["status"] == "passed" and s3verify["run_sha256"] == sha(ROOT / "Modeling/tables/regime_forecast/run.json")
    meta = pd.read_csv(OUT / "meta_features.csv", encoding="utf-8-sig", parse_dates=["timestamp", "label_confirmed_at", "latest_input_timestamp"])
    h = pd.read_csv(ROOT / "Modeling/tables/regime_diagnosis/hourly_labels.csv", encoding="utf-8-sig", parse_dates=["timestamp", "label_confirmed_at"])
    m = pd.read_csv(ROOT / "Modeling/tables/m01/hourly_frame.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    s = pd.read_csv(ROOT / "Modeling/tables/regime_forecast/predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    folds = pd.read_csv(ROOT / "Modeling/tables/regime_forecast/folds.csv", encoding="utf-8-sig")
    expected_times = sorted(set(h.loc[h.eligible_onset & h.persistence_known & (h.month >= 2), "timestamp"]) & set(m.loc[m.eligible_common & (m.timestamp < "2021-07-01"), "timestamp"]))
    assert meta.timestamp.tolist() == expected_times and len(meta) == 3598
    assert (meta.latest_input_timestamp == meta.timestamp - pd.Timedelta(hours=1)).all()
    for variant, features in c["features"].items():
        assert not set(features) & {"actual", "level", "maximum", "target_maximum", "sustained_onset", "onset", "profile", "delta", "deviation"}
        assert len(features) == len(set(features))
    original = m.set_index("timestamp").loc[meta.timestamp]
    parent = h.set_index("timestamp").loc[meta.timestamp]
    for feature in c["features"]["B0"] + ["target_maximum", "target_mean"]:
        np.testing.assert_allclose(meta[feature], original[feature], rtol=0, atol=1e-9)
    for feature in [n for n in c["features"]["B1"] if n.startswith(("prior_", "past_"))]:
        np.testing.assert_allclose(meta[feature], parent[feature], rtol=0, atol=1e-9)
    assert meta.sustained_onset.tolist() == parent.sustained_onset.tolist()
    for direction, family in [("up", "logistic"), ("down", "hgb")]:
        for month, g in meta.groupby("month"):
            available = bool(folds.loc[(folds.direction == direction) & (folds.month == month), "ml_available"].iloc[0])
            used = family if available else "calendar"
            prior = s.loc[(s.direction == direction) & (s.method == used) & s.timestamp.isin(g.timestamp)].set_index("timestamp").loc[g.timestamp]
            np.testing.assert_allclose(g[f"p_{direction}"], prior.score, rtol=0, atol=1e-9)
            assert (g[f"{direction}_classifier_available"] == int(available)).all()
            assert (g[f"{direction}_score_method"] == used).all()
    p = pd.read_csv(OUT / "predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    manifest = pd.read_csv(OUT / "model_manifest.csv", encoding="utf-8-sig")
    assert len(p) == 5 * 2182 and not p.duplicated(["timestamp", "variant"]).any()
    reloaded = refitted = 0
    for r in manifest.itertuples():
        assert sha(ROOT / r.file) == r.sha256
        boundary = pd.Timestamp(2021, r.month, 1)
        tr = meta.loc[(meta.timestamp < boundary) & (meta.label_confirmed_at < boundary)]
        te = meta.loc[meta.month == r.month]
        g = p.loc[(p.variant == r.variant) & (p.month == r.month)]
        features = c["features"][r.variant]
        assert r.features == "|".join(features) and len(tr) == r.train_hours and len(te) == r.eval_hours
        assert str(tr.timestamp.min()) == r.train_start and str(tr.timestamp.max()) == r.train_end
        assert str(tr.label_confirmed_at.max()) == r.last_confirmed and tr.label_confirmed_at.max() < boundary
        assert int(tr.up_classifier_available.sum()) == r.train_up_ml_probability_hours
        assert int(tr.down_classifier_available.sum()) == r.train_down_ml_probability_hours
        assert g.timestamp.tolist() == te.timestamp.tolist()
        np.testing.assert_allclose(g.actual, te.target_maximum, rtol=0, atol=1e-9)
        assert g.seen_regression_profile.tolist() == te.profile.isin(tr.profile).astype(int).tolist()
        fitted = joblib.load(ROOT / r.file)
        np.testing.assert_allclose(g.prediction, fitted.predict(te[features]), rtol=1e-9, atol=1e-8)
        reloaded += 1
        if (r.month, r.variant) in [(5, "B0"), (6, "B1"), (6, "B2_up")]:
            fresh = HistGradientBoostingRegressor(**c["hgb_parameters"])
            fresh.fit(tr[features], tr.target_maximum)
            np.testing.assert_allclose(g.prediction, fresh.predict(te[features]), rtol=1e-9, atol=1e-8)
            refitted += 1
    event_table = pd.read_csv(OUT / "event_errors.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    coverage = pd.read_csv(OUT / "event_coverage.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    daily_table = pd.read_csv(OUT / "daily_peak_errors.csv", encoding="utf-8-sig")
    assert len(coverage) == 5 * 33
    assert coverage.complete.all() and len(event_table) == len(coverage)
    for r in event_table.itertuples():
        times = pd.date_range(r.timestamp, periods=3, freq="h")
        g = p.loc[(p.variant == r.variant) & p.timestamp.isin(times)]
        assert len(g) == 3
        for name, value in stats(g).items():
            close(getattr(r, name), value)
    for r in daily_table.itertuples():
        g = p.loc[(p.variant == r.variant) & (p.timestamp.dt.strftime("%Y-%m-%d") == r.date)]
        assert len(g) == 24
        peak = g.loc[g.actual == g.actual.max()]
        assert len(peak) == r.ties
        for name, value in stats(peak).items():
            close(getattr(r, name), value)
    table = pd.read_csv(OUT / "metrics.csv", encoding="utf-8-sig", dtype={"period": str})
    for r in table.itertuples():
        gg = p.loc[p.variant == r.variant]
        if r.period == "May-Jun":
            gg = gg.loc[gg.month >= 5]
        elif r.period != "Apr-Jun":
            gg = gg.loc[gg.month == int(r.period)]
        if r.condition == "all":
            expected = stats(gg)
        elif r.condition == "non_persistent_start":
            expected = stats(gg.loc[gg.sustained_onset == 0])
        elif r.condition.startswith("profile_"):
            expected = stats(gg.loc[gg.seen_regression_profile == int(r.condition[-1])])
        elif r.condition.endswith("_start"):
            expected = stats(gg.loc[gg.sustained_onset == (1 if r.condition.startswith("up") else -1)])
        else:
            origin = event_table if "window" in r.condition else daily_table
            rows = origin.loc[origin.variant == r.variant]
            if r.period == "May-Jun":
                rows = rows.loc[rows.month >= 5]
            if "window" in r.condition:
                rows = rows.loc[rows.direction == ("up" if r.condition.startswith("up") else "down")]
            expected = {n: math.fsum(float(v) for v in rows[n]) / len(rows) for n in ["mae", "bias", "under", "over"]}
            expected["rmse"] = math.sqrt(math.fsum(float(v) ** 2 for v in rows.rmse) / len(rows))
            expected["hours"] = 3 * len(rows) if "window" in r.condition else int(rows.ties.sum())
            expected["weight_sum"] = len(rows)
        for name, value in expected.items():
            close(getattr(r, name), value)
    gains = pd.read_csv(OUT / "paired_gains.csv", encoding="utf-8-sig")
    for r in gains.itertuples():
        def get(v):
            return table.loc[(table.variant == v) & (table.period == r.period) & (table.condition == r.direction + "_start")].iloc[0]
        a, b = get(r.variant), get(r.baseline)
        close(r.start_mae_reduction, 1 - a.mae / b.mae)
        name = "under" if r.direction == "up" else "over"
        close(r.directional_reduction, 1 - a[name] / b[name])
    decisions = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    for r in decisions["comparisons"]:
        rows = gains.loc[(gains.variant == r["variant"]) & (gains.direction == r["direction"])]
        def get(v, condition):
            return table.loc[(table.variant == v) & (table.period == r["period"]) & (table.condition == condition)].iloc[0]
        overall = get(r["variant"], "all").mae / get("B0", "all").mae - 1
        daily = get(r["variant"], "daily_peak_day_equal").mae / get("B0", "daily_peak_day_equal").mae - 1
        close(r["overall_mae_change_vs_B0"], overall); close(r["daily_peak_mae_change_vs_B0"], daily)
        wins = 0
        for month in ([5, 6] if r["direction"] == "up" else [4, 5, 6]):
            part = table.loc[(table.period == str(month)) & (table.condition == r["direction"] + "_start")].set_index("variant")
            wins += int(part.loc[r["variant"], "mae"] < part.loc["B1", "mae"])
        assert wins == r["monthly_wins_vs_B1"]
        passed = bool((rows.start_mae_reduction >= .05).all() and (rows.directional_reduction >= .05).all()
                      and overall <= .01 and daily <= .01 and rows.events.min() >= 5 and wins >= 2)
        assert passed == r["development_gate_passed"]
    result = {"status": "passed", "run_sha256": sha(OUT / "run.json"), "verifier_sha256": sha(__file__),
              "meta_rows_checked": len(meta), "predictions_checked": len(p), "models_reloaded": reloaded,
              "independent_refits": refitted, "event_windows_checked": len(event_table), "daily_rows_checked": len(daily_table),
              "metrics_checked": len(table), "genuine_prior_month_probabilities": True}
    (OUT / "independent_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
