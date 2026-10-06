"""Single predeclared classifier age removal; reuse frozen HGB forecasts.

Development first. Follow-up is conditional and explicitly exploratory.
No completed parent artifact is edited, no hyperparameter or threshold search.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from regime_forecast import FEATURES, load_frame, model, threshold, sha, save_json
from regime_followup import detection, metrics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Modeling/tables/regime_age_ablation"
MODELS = ROOT / "Modeling/models/regime_age_ablation"
FEATURES_NOAGE = [x for x in FEATURES if x != "prior_state_age"]


def inputs():
    names = ["Modeling/scripts/regime_forecast.py", "Modeling/scripts/regime_followup.py",
             "Modeling/tables/regime_diagnosis/hourly_labels.csv",
             "Modeling/tables/regime_forecast/contract.json", "Modeling/tables/regime_forecast/predictions.csv",
             "Modeling/tables/regime_routing/predictions.csv", "Modeling/tables/regime_routing/independent_verification.json",
             "Modeling/tables/regime_followup/contract.json", "Modeling/tables/regime_followup/run.json",
             "Modeling/tables/regime_followup/independent_verification.json",
             "Modeling/tables/regime_followup/classification_predictions.csv",
             "Modeling/tables/regime_followup/predictions.csv",
             "Modeling/tables/regime_followup/verified_alarm_conditions.csv"]
    return {n: sha(ROOT / n) for n in names}


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "contract.json").exists()
    c = {
        "stage": "S06 single age ablation", "created_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"),
        "entrypoint_sha256": sha(__file__), "inputs_sha256": inputs(),
        "reason": "S05 all38 August FP had age180..227 vs training max64, age largest positive logit. Test one classifier feature removal, not proof of a physical cause.",
        "change": "Remove prior_state_age only from upward Logistic classifier; retain original24 vs candidate23 inputs. HGB B0/B1 and their age input remain unchanged to isolate routing selection.",
        "features": FEATURES_NOAGE,
        "logistic": json.loads((ROOT / "Modeling/tables/regime_forecast/contract.json").read_text(encoding="utf-8"))["logistic"],
        "definition": "Frozen S01 three-hour persistent mean-level upward onset, predicted at t-1; target max only for HGB error evaluation.",
        "development": "Apr-Jun cumulative preceding-month training with label_confirmed_at < boundary; Apr ranking, May-Jun automatic alarms. Min8 events/5 dates. Threshold from earlier matured OOF only, maximizeTP/minFP/maxthreshold under1%calibrationFPR. No method selection/tuning.",
        "development_gate": "On May-Jun min5 events, recall>=.8, precision>=.8, FPR<=.01; then G1 upMAE and under reduction>=5%vsB0, overall and daily peak worsening<=1%. Monthly results diagnostic, no all-month victory demand.",
        "conditional_followup": "Only if BOTH development gates pass: one Jan-Jun mature-label fit; threshold computed once from candidate Apr-Jun OOF scores; apply fixed model/threshold Jul-Aug without July refit. Reuse S05 HGB predictions. Same gates; already-observed exploratory reuse, no independent confirmation claim.",
        "scope": "Hypothesis selected after observing S05. Development and follow-up are exploratory; no new independent test.",
        "controls": ["original Logistic", "B0", "B1_all", "original G1", "candidate G1", "hour08"],
        "coverage": "Use exactly parent forecast rows, retain unknown future labels for regression but exclude them from classification; no record deletion, age clipping or label changes.",
        "stop": "Exactly one candidate. Preserve failed results and stop this comparison if gate fails; do not try age transforms or tune boundaries in same run.",
        "root_readme_sha256": sha(ROOT / "README.md"),
        "manuscript_sha256": sha(ROOT / "Modeling/04_Modeling_원고.md"),
    }
    save_json(OUT / "contract.json", c)
    print(json.dumps({"status": "frozen", "contract_sha256": sha(OUT / "contract.json")}), flush=True)


def classify_frame(test, score, th, old):
    z = test[["timestamp", "label_confirmed_at", "month", "profile", "sustained_onset"]].copy()
    z["event"] = (z.sustained_onset == 1).astype(int)
    z["score"] = score; z["threshold"] = th
    z["alarm"] = (score >= th).astype(int) if th is not None else np.nan
    z["method"] = "noage"
    baseline = old.set_index("timestamp").loc[z.timestamp]
    b = z.copy()
    for name in ["score", "threshold", "alarm"]:
        b[name] = baseline[name].to_numpy()
    b["method"] = "original"
    return pd.concat([z, b], ignore_index=True)


def class_summary(p, common, period):
    rows = []
    for scope, part in [("six_history", p), ("common_regression", p.loc[p.timestamp.isin(common)])]:
        part = part.loc[part.event.notna() & part.alarm.notna()]
        for subperiod, gg in [(period, part)] + [(str(m), g) for m, g in part.groupby("month")]:
            for method, g in gg.groupby("method"):
                rows.append({"scope": scope, "period": subperiod, "method": method, **detection(g)})
    return pd.DataFrame(rows)


def regress_frame(parent, cp, late=False):
    b = parent.loc[parent.variant == "B0"].copy().sort_values("timestamp")
    other = parent.loc[parent.variant == "B1_all"].set_index("timestamp").loc[b.timestamp]
    old = parent.loc[parent.variant == "G1"].set_index("timestamp").loc[b.timestamp]
    hour = parent.loc[parent.variant == "hour08"].set_index("timestamp").loc[b.timestamp]
    assert np.allclose(b.actual, other.actual)
    alarm = cp.loc[cp.method == "noage"].set_index("timestamp").loc[b.timestamp, "alarm"].to_numpy(int)
    options = {"B0": b.prediction.to_numpy(), "B1_all": other.prediction.to_numpy(),
               "G1_old": old.prediction.to_numpy(), "G1_noage": np.where(alarm, other.prediction, b.prediction),
               "hour08": hour.prediction.to_numpy()}
    masks = {"B0": np.zeros(len(b), int), "B1_all": np.ones(len(b), int),
             "G1_old": old.routed.to_numpy(int), "G1_noage": alarm,
             "hour08": hour.routed.to_numpy(int)}
    parts = []
    cols = ["timestamp", "month", "date", "profile", "sustained_onset", "actual", "seen_regression_profile"]
    for variant, values in options.items():
        g = b[cols].copy(); g["variant"] = variant; g["prediction"] = values; g["routed"] = masks[variant]
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def regress_summary(p, period):
    rows = []; daily = []; windows = []; burden = []
    for variant, g in p.groupby("variant"):
        for date, day in g.groupby("date"):
            if len(day) == 24:
                peak = day.loc[day.actual == day.actual.max()]
                daily.append({"variant": variant, "date": date, "month": int(day.month.iloc[0]), "ties": len(peak), **metrics(peak)})
        for r in g.loc[g.sustained_onset == 1].itertuples():
            w = g.loc[g.timestamp.isin(pd.date_range(r.timestamp, periods=3, freq="h"))]
            if len(w) == 3:
                windows.append({"variant": variant, "timestamp": r.timestamp, "month": r.month, **metrics(w)})
    dd = pd.DataFrame(daily); ww = pd.DataFrame(windows)
    for variant, g in p.groupby("variant"):
        for subperiod, part in [(period, g)] + [(str(m), gg) for m, gg in g.groupby("month")]:
            for condition, gg in [("all", part), ("up_start", part.loc[part.sustained_onset == 1]),
                                  ("down_start", part.loc[part.sustained_onset == -1])]:
                rows.append({"variant": variant, "period": subperiod, "condition": condition, **metrics(gg)})
            for condition, source in [("daily_peak_day_equal", dd), ("up_window_event_equal", ww)]:
                a = source.loc[source.variant == variant]
                if subperiod != period:
                    a = a.loc[a.month == int(subperiod)]
                stat = {n: float(a[n].mean()) if len(a) else None for n in ["mae", "under", "over", "bias"]}
                stat["rmse"] = float(np.sqrt(np.mean(a.rmse ** 2))) if len(a) else None
                rows.append({"variant": variant, "period": subperiod, "condition": condition,
                             "hours": int(a.ties.sum()) if condition.startswith("daily") else len(a) * 3, "units": len(a), **stat})
            burden.append({"variant": variant, "period": subperiod, "hours": len(part), "routed": int(part.routed.sum()),
                           "routed_up": int(((part.routed == 1) & (part.sustained_onset == 1)).sum()),
                           "routed_non_up_known": int(((part.routed == 1) & part.sustained_onset.notna() & (part.sustained_onset != 1)).sum()),
                           "routed_unknown": int(((part.routed == 1) & part.sustained_onset.isna()).sum())})
    return pd.DataFrame(rows), dd, ww, pd.DataFrame(burden)


def gates(det, reg, period):
    a = det.loc[(det.scope == "common_regression") & (det.period == period) & (det.method == "noage")].iloc[0]
    def get(v, cond):
        return reg.loc[(reg.variant == v) & (reg.period == period) & (reg.condition == cond)].iloc[0]
    start_gain = 1 - get("G1_noage", "up_start").mae / get("B0", "up_start").mae
    under_gain = 1 - get("G1_noage", "up_start").under / get("B0", "up_start").under
    overall = get("G1_noage", "all").mae / get("B0", "all").mae - 1
    peak = get("G1_noage", "daily_peak_day_equal").mae / get("B0", "daily_peak_day_equal").mae - 1
    return {"classification_passed": bool(a.events >= 5 and a.recall >= .8 and pd.notna(a.precision) and a.precision >= .8 and a.fpr <= .01),
            "routing_passed": bool(a.events >= 5 and start_gain >= .05 and under_gain >= .05 and overall <= .01 and peak <= .01),
            "start_mae_reduction": float(start_gain), "under_reduction": float(under_gain),
            "overall_mae_change": float(overall), "daily_peak_mae_change": float(peak)}


def run():
    assert not (OUT / "run.json").exists()
    c = json.loads((OUT / "contract.json").read_text(encoding="utf-8"))
    assert c["entrypoint_sha256"] == sha(__file__) and c["inputs_sha256"] == inputs()
    f = load_frame(); base_contract = json.loads((ROOT / "Modeling/tables/regime_forecast/contract.json").read_text(encoding="utf-8"))
    old = pd.read_csv(ROOT / "Modeling/tables/regime_forecast/predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp", "label_confirmed_at"])
    old = old.loc[(old.direction == "up") & (old.method == "logistic") & (old.month >= 4)]
    MODELS.mkdir(parents=True, exist_ok=True)
    manifest = []; calrows = []; devparts = []; history = []
    def fit(train, test, name, boundary):
        y = (train.sustained_onset == 1).astype(int)
        assert y.sum() >= 8 and train.loc[y == 1, "date"].nunique() >= 5
        assert (train.label_confirmed_at < boundary).all()
        fitted = model("logistic", base_contract); fitted.fit(train[c["features"]], y)
        path = MODELS / (name + ".joblib"); assert not path.exists(); joblib.dump(fitted, path)
        manifest.append({"name": name, "file": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
                         "boundary": str(boundary), "train_hours": len(train), "train_events": int(y.sum()),
                         "train_latest_confirmed": str(train.label_confirmed_at.max()), "features": "|".join(c["features"])})
        return fitted.predict_proba(test[c["features"]])[:, 1]
    def calibrate(cal, period):
        th, tp, fp, cap = threshold(cal.event.to_numpy(int), cal.score.to_numpy(float))
        calrows.append({"period": period, "threshold": th, "tp": tp, "fp": fp, "fp_cap": cap,
                        "hours": len(cal), "events": int(cal.event.sum()), "latest_confirmed": str(cal.label_confirmed_at.max())})
        return th
    for month in [4, 5, 6]:
        boundary = pd.Timestamp(2021, month, 1)
        train = f.loc[(f.timestamp < boundary) & (f.label_confirmed_at < boundary)]
        test = f.loc[f.month == month]
        score = fit(train, test, f"noage_{month:02d}", boundary)
        cal = pd.concat(history, ignore_index=True) if history else None
        th = calibrate(cal.loc[cal.label_confirmed_at < boundary], str(month)) if cal is not None else None
        cp = classify_frame(test, score, th, old)
        devparts.append(cp); history.append(cp.loc[cp.method == "noage"])
    dev = pd.concat(devparts, ignore_index=True)
    rp = pd.read_csv(ROOT / "Modeling/tables/regime_routing/predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    pred = regress_frame(rp, dev)
    primary = dev.loc[dev.month >= 5]
    det = class_summary(primary, pred.timestamp.unique(), "May-Jun")
    reg, daily, windows, burden = regress_summary(pred, "May-Jun")
    dg = gates(det, reg, "May-Jun")
    outputs = {"development_classification.csv": dev, "development_detection.csv": det,
               "development_predictions.csv": pred, "development_metrics.csv": reg,
               "development_daily.csv": daily, "development_windows.csv": windows, "development_burden.csv": burden}
    late_executed = dg["classification_passed"] and dg["routing_passed"]
    decision = {"development": dg, "followup_executed": late_executed, "scope": c["scope"]}
    print(json.dumps({"development": dg, "followup_executed": late_executed}), flush=True)
    if late_executed:
        # This branch is fixed before any candidate result. Late labels never select it.
        full = pd.read_csv(ROOT / "Modeling/tables/regime_followup/classification_predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp", "label_confirmed_at"])
        train = f.loc[f.label_confirmed_at < "2021-07-01"]
        score = fit(train, full, "noage_fixed_june", pd.Timestamp("2021-07-01"))
        cal = dev.loc[(dev.method == "noage") & (dev.label_confirmed_at < "2021-07-01")]
        th = calibrate(cal, "Jul-Aug")
        cp = full[["timestamp", "label_confirmed_at", "month", "profile", "sustained_onset", "event"]].copy()
        cp["score"] = score; cp["threshold"] = th; cp["alarm"] = (score >= th).astype(int); cp["method"] = "noage"
        original = cp.copy(); original["score"] = full.score_logistic; original["alarm"] = full.alarm_logistic
        original["method"] = "original"
        original["threshold"] = json.loads((ROOT / "Modeling/tables/regime_followup/contract.json").read_text(encoding="utf-8"))["classification_thresholds"]["logistic"]["threshold"]
        cp = pd.concat([cp, original], ignore_index=True)
        rp = pd.read_csv(ROOT / "Modeling/tables/regime_followup/predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
        pp = regress_frame(rp, cp, late=True)
        dd = class_summary(cp, pp.timestamp.unique(), "Jul-Aug")
        rr, daily, windows, burden = regress_summary(pp, "Jul-Aug")
        decision["followup"] = gates(dd, rr, "Jul-Aug")
        outputs.update({"followup_classification.csv": cp, "followup_detection.csv": dd,
                        "followup_predictions.csv": pp, "followup_metrics.csv": rr,
                        "followup_daily.csv": daily, "followup_windows.csv": windows, "followup_burden.csv": burden})
    outputs.update({"model_manifest.csv": pd.DataFrame(manifest), "calibration.csv": pd.DataFrame(calrows)})
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    save_json(OUT / "decision.json", decision)
    save_json(OUT / "run.json", {"status": "completed", "script_sha256": sha(__file__), "contract_sha256": sha(OUT / "contract.json"),
                                "inputs_sha256": c["inputs_sha256"], "new_fits": len(manifest), "new_regression_fits": 0,
                                "outputs_sha256": {n: sha(OUT / n) for n in list(outputs) + ["decision.json"]}, "runtime": sys.version})
    assert c["root_readme_sha256"] == sha(ROOT / "README.md")
    assert c["manuscript_sha256"] == sha(ROOT / "Modeling/04_Modeling_원고.md")
    print(json.dumps({"status": "completed", "new_fits": len(manifest), "decision": decision}), flush=True)


if __name__ == "__main__":
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    {"freeze": freeze, "run": run}[sys.argv[1]]()
