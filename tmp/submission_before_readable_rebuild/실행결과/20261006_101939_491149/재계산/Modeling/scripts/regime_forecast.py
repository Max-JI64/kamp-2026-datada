"""S03: frozen, chronological one-hour-ahead persistent transition classification.

No regression integration, held-out July/August reads, tuning, or source edits.
Run freeze before run; completed outputs are never overwritten.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import hashlib
import json
import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "Modeling/tables/regime_diagnosis"
OUT = ROOT / "Modeling/tables/regime_forecast"
MODELS = ROOT / "Modeling/models/regime_forecast"
DYNAMIC = ["prior_mean", "past_median6", "past_iqr6", "past_range6", "past_slope6",
           "prior_delta", "prior_last_minus_mean", "prior_slot_range", "prior_production",
           "prior_production_delta", "prior_state_age", "prior_state_left_censored",
           "prior_state_direction"]
CALENDAR = ["hour_sin", "hour_cos", "month", "weekend"] + [f"dow_{i}" for i in range(7)]
FEATURES = DYNAMIC + CALENDAR
BASE = ["calendar", "last_delta", "slope6"]
ML = ["logistic", "hgb"]
METHODS = BASE + ML


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def frozen_inputs():
    return {str((PARENT / name).relative_to(ROOT)).replace("\\", "/"): sha(PARENT / name)
            for name in ["hourly_labels.csv", "contract.json", "run.json", "independent_verification.json"]}


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "contract.json").exists(), "Contract already exists; verify instead of overwrite."
    parent_verify = json.loads((PARENT / "independent_verification.json").read_text(encoding="utf-8"))
    assert parent_verify["status"] == "passed" and parent_verify["run_sha256"] == sha(PARENT / "run.json")
    contract = {
        "stage": "S03", "created_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"),
        "entrypoint_sha256": sha(__file__), "inputs_sha256": frozen_inputs(),
        "scope": "Jan-Jun only; Feb-Jun chronological predictions; Apr-Jun development ranking",
        "targets": {"up": 1, "down": -1}, "horizon": "t, t+1, t+2 sustained onset; predict at end of t-1",
        "feature_names": FEATURES, "forbidden_inputs": ["level", "maximum", "delta", "deviation", "onset", "sustained_onset", "profile"],
        "training": "All earlier eligible known labels with label_confirmed_at strictly before evaluation-month boundary",
        "minimum_fit_support": {"positive_events": 8, "positive_dates": 5},
        "logistic": {"C": 1.0, "max_iter": 3000, "random_state": 42, "class_weight": None, "standardize": "training only"},
        "hgb": {"max_leaf_nodes": 7, "max_iter": 100, "learning_rate": .05, "min_samples_leaf": 20, "l2_regularization": 1.0, "early_stopping": False, "random_state": 42},
        "calendar_baseline": "Jeffreys-smoothed hour x weekend event rate; cell n<20 falls back to hour; hour n<20 to global rate, using training only",
        "other_baselines": {"last_delta": "direction * prior_delta", "slope6": "direction * past_slope6"},
        "calibration": "Earlier genuinely out-of-month predictions with matured labels, intersect available rows across all five methods, never in-sample rescoring. At least one positive required. No calibration means ranking only.",
        "selection": "Separately choose ML and simple baseline by positive-event-weighted monthly AP on common past calibration rows; tie follows fixed method order. Each direction/month selected before evaluation labels are summarized.",
        "threshold": "score >= threshold; calibration FPR <=1%; maximize TP then minimize FP then maximize threshold; include infinity for no alarms",
        "primary_alarm_scope": "Selected ML vs past-selected simple baseline, Apr-Jun where common past calibration exists. Expected up May-Jun, down Apr-Jun.",
        "development_gate": {"extra_tp_at_least": 2, "relative_extra_tp_at_least": .20, "zero_baseline": "use absolute gain only", "extra_fp_per_100_hours_at_most": .5},
        "metrics": "Monthly AP (undefined with zero events), precision, recall, TP/FN/FP/TN, false alarms per100 evaluated hours, negative FPR; no AUROC-only selection",
        "posthoc": "Short-event false positives, late alarms, seen/new daily arrays are diagnostic only; labels/features/thresholds are not revised from these results",
        "tuning_trials": 0, "regression_integration": False,
    }
    save_json(OUT / "contract.json", contract)
    print(json.dumps({"status": "frozen", "contract_sha256": sha(OUT / "contract.json")}), flush=True)


def load_frame():
    f = pd.read_csv(PARENT / "hourly_labels.csv", encoding="utf-8-sig")
    for name in ["timestamp", "label_confirmed_at", "latest_input_timestamp"]:
        f[name] = pd.to_datetime(f[name])
    f = f.loc[f.eligible_onset & f.persistence_known].copy().reset_index(drop=True)
    f["hour_sin"] = np.sin(2 * np.pi * f.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * f.hour / 24)
    for i in range(7):
        f[f"dow_{i}"] = (f.timestamp.dt.dayofweek == i).astype(int)
    assert len(f) == 4336 and f.timestamp.max() < pd.Timestamp("2021-07-01")
    assert (f.latest_input_timestamp == f.timestamp - pd.Timedelta(hours=1)).all()
    assert np.isfinite(f[FEATURES].to_numpy(float)).all()
    return f


def model(method, contract):
    if method == "logistic":
        params = {k: v for k, v in contract[method].items() if k != "standardize"}
        return make_pipeline(StandardScaler(), LogisticRegression(**params))
    return HistGradientBoostingClassifier(**contract[method])


def calendar_scores(train, test, y):
    z = train.assign(y=y)
    cell = z.groupby(["hour", "weekend"]).y.agg(["sum", "count"])
    hour = z.groupby("hour").y.agg(["sum", "count"])
    global_rate = (y.sum() + .5) / (len(y) + 1)
    values = []
    for h, w in zip(test.hour, test.weekend):
        r = cell.loc[(h, w)] if (h, w) in cell.index else None
        if r is None or r["count"] < 20:
            r = hour.loc[h] if h in hour.index else None
        values.append((r["sum"] + .5) / (r["count"] + 1) if r is not None and r["count"] >= 20 else global_rate)
    return np.array(values)


def threshold(y, score):
    # Stable descending order, evaluating all tie groups in a single pass.
    order = np.argsort(-score, kind="stable")
    yy, ss = np.asarray(y)[order], np.asarray(score)[order]
    best = (0, 0, math.inf)
    tp = fp = 0
    cap = math.floor(.01 * int((yy == 0).sum()) + 1e-10)
    for i in range(len(yy)):
        tp += int(yy[i] == 1); fp += int(yy[i] == 0)
        if i + 1 < len(yy) and ss[i + 1] == ss[i]:
            continue
        candidate = (tp, -fp, float(ss[i]))
        if fp <= cap and candidate > best:
            best = candidate
    return best[2], best[0], -best[1], cap


def ap(y, score):
    return float(average_precision_score(y, score)) if np.sum(y) else None


def measures(frame):
    y = frame.event.to_numpy(int)
    scores = frame.score.to_numpy(float)
    result = {"hours": len(frame), "events": int(y.sum()), "ap": ap(y, scores)}
    if frame.alarm.notna().all():
        a = frame.alarm.to_numpy(int)
        tp = int(((a == 1) & (y == 1)).sum()); fp = int(((a == 1) & (y == 0)).sum())
        fn = int(y.sum()) - tp; tn = len(y) - int(y.sum()) - fp
        result.update(tp=tp, fp=fp, fn=fn, tn=tn, recall=tp / (tp + fn) if tp + fn else None,
                      precision=tp / (tp + fp) if tp + fp else None, fp_per100h=100 * fp / len(y),
                      fpr=fp / (fp + tn) if fp + tn else None,
                      short_event_fp=int(((a == 1) & (frame.short_event.to_numpy(int) == 1)).sum()))
    return result


def past_ap(cal, method):
    subset = cal.loc[cal.method == method]
    terms = [(len(g.loc[g.event == 1]), ap(g.event.to_numpy(), g.score.to_numpy())) for _, g in subset.groupby("month")]
    return sum(n * value for n, value in terms if n) / sum(n for n, _ in terms) if sum(n for n, _ in terms) else -math.inf


def run():
    assert not (OUT / "run.json").exists(), "Completed run exists; verify instead of overwrite."
    c = json.loads((OUT / "contract.json").read_text(encoding="utf-8"))
    assert c["entrypoint_sha256"] == sha(__file__) and c["inputs_sha256"] == frozen_inputs()
    f = load_frame()
    MODELS.mkdir(parents=True, exist_ok=True)
    predictions = []; manifests = []; folds = []; choices = []; calibration_rows = []
    for direction, sign in c["targets"].items():
        history = []
        for month in range(2, 7):
            boundary = pd.Timestamp(2021, month, 1)
            train = f.loc[(f.timestamp < boundary) & (f.label_confirmed_at < boundary)]
            test = f.loc[f.month == month]
            y = (train.sustained_onset == sign).astype(int).to_numpy()
            dates = train.loc[y == 1, "date"].nunique()
            available = y.sum() >= 8 and dates >= 5 and (y == 0).any()
            fold = {"direction": direction, "month": month, "train_hours": len(train), "train_events": int(y.sum()),
                    "positive_dates": int(dates), "train_latest_confirmed": str(train.label_confirmed_at.max()),
                    "test_hours": len(test), "test_events": int((test.sustained_onset == sign).sum()), "ml_available": bool(available)}
            folds.append(fold)
            scores = {"calendar": calendar_scores(train, test, y), "last_delta": sign * test.prior_delta.to_numpy(),
                      "slope6": sign * test.past_slope6.to_numpy()}
            if available:
                for method in ML:
                    fitted = model(method, c)
                    fitted.fit(train[FEATURES], y)
                    scores[method] = fitted.predict_proba(test[FEATURES])[:, 1]
                    path = MODELS / f"{direction}_{month:02d}_{method}.joblib"
                    assert not path.exists()
                    joblib.dump(fitted, path)
                    manifests.append({**fold, "method": method, "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                                      "sha256": sha(path), "features": "|".join(FEATURES)})
            # Common prior OOF rows prevent comparing one family using a different calibration sample.
            cal = pd.DataFrame(history)
            if len(cal):
                cal = cal.loc[cal.label_confirmed_at < boundary]
                common = cal.groupby("timestamp").method.nunique()
                cal = cal.loc[cal.timestamp.isin(common.index[common == len(METHODS)])].copy()
            calibrated = bool(available and len(cal) and cal.loc[cal.method == "calendar", "event"].sum() > 0)
            selected = {}
            if calibrated:
                for family, members in [("simple", BASE), ("ml", ML)]:
                    selected[family] = max(members, key=lambda m: (past_ap(cal, m), -members.index(m)))
            choice = {"direction": direction, "month": month, "calibration_available": calibrated,
                      "calibration_hours": len(cal.loc[cal.method == "calendar"]) if len(cal) else 0,
                      "calibration_events": int(cal.loc[cal.method == "calendar", "event"].sum()) if len(cal) else 0,
                      "calibration_latest_confirmed": str(cal.label_confirmed_at.max()) if len(cal) else "",
                      "selected_simple": selected.get("simple", ""), "selected_ml": selected.get("ml", "")}
            choices.append(choice)
            current = []
            for method, score in scores.items():
                th = None
                if calibrated:
                    mc = cal.loc[cal.method == method]
                    th, ctp, cfp, cap = threshold(mc.event.to_numpy(int), mc.score.to_numpy(float))
                    calibration_rows.append({"direction": direction, "month": month, "method": method,
                                             "threshold": th, "tp": ctp, "fp": cfp, "fp_cap": cap,
                                             "hours": len(mc), "events": int(mc.event.sum()), "weighted_monthly_ap": past_ap(cal, method)})
                group = test[["timestamp", "label_confirmed_at", "latest_input_timestamp", "profile", "level", "maximum", "prior_mean"]].copy()
                group["direction"] = direction; group["month"] = month; group["method"] = method
                group["event"] = (test.sustained_onset.to_numpy() == sign).astype(int)
                group["short_event"] = ((test.onset.to_numpy() == sign) & (test.sustained_onset.to_numpy() != sign)).astype(int)
                group["seen_profile"] = test.profile.isin(train.profile).astype(int).to_numpy()
                group["score"] = score; group["threshold"] = th
                group["alarm"] = (score >= th).astype(int) if th is not None else np.nan
                group["selected_family"] = [next((family for family, member in selected.items() if member == method), "")] * len(group)
                current.extend(group.to_dict("records"))
            predictions.extend(current); history.extend(current)
            print(json.dumps({**fold, **choice}), flush=True)
    p = pd.DataFrame(predictions)
    monthly = [{"direction": d, "month": m, "method": method, **measures(g)} for (d, m, method), g in p.groupby(["direction", "month", "method"])]
    selected = p.loc[(p.month >= 4) & (p.selected_family != "")].copy()
    aggregate = []; decisions = []
    for direction, g in selected.groupby("direction"):
        stats = {}
        for family, gg in g.groupby("selected_family"):
            # Cross-month pooled AP is not comparable when probability scales change.
            result = measures(gg); result.pop("ap", None)
            aggregate.append({"direction": direction, "family": family, "months": "|".join(map(str, sorted(gg.month.unique()))), **result})
            stats[family] = result
        a, b = stats["ml"], stats["simple"]
        extra = a["tp"] - b["tp"]; relative = extra / b["tp"] if b["tp"] else None
        fp_extra = a["fp_per100h"] - b["fp_per100h"]
        passed = extra >= 2 and (relative is None or relative >= .20) and fp_extra <= .5 + 1e-12
        decisions.append({"direction": direction, "extra_tp": extra, "relative_extra_tp": relative,
                          "extra_fp_per100h": fp_extra, "development_gate_passed": passed, "scope": "exploratory Jan-Jun development; not independent final testing"})
    events = p.loc[(p.month >= 4) & (p.event == 1)].copy()
    # A subsequent warning is late for this onset and never counted as a true advance detection.
    idx = {(r.direction, r.method, r.timestamp): r for r in p.itertuples()}
    events["alarm_one_hour_late"] = [getattr(idx.get((r.direction, r.method, r.timestamp + pd.Timedelta(hours=1))), "alarm", np.nan) for r in events.itertuples()]
    events["alarm_two_hours_late"] = [getattr(idx.get((r.direction, r.method, r.timestamp + pd.Timedelta(hours=2))), "alarm", np.nan) for r in events.itertuples()]
    condition = []
    for (d, family, seen), gg in selected.groupby(["direction", "selected_family", "seen_profile"]):
        result = measures(gg); result.pop("ap", None)
        condition.append({"direction": d, "family": family, "seen_profile": int(seen), **result})
    outputs = {"predictions.csv": p, "monthly_metrics.csv": pd.DataFrame(monthly), "folds.csv": pd.DataFrame(folds),
               "model_manifest.csv": pd.DataFrame(manifests), "selection.csv": pd.DataFrame(choices),
               "calibration.csv": pd.DataFrame(calibration_rows), "selected_predictions.csv": selected,
               "selected_summary.csv": pd.DataFrame(aggregate), "event_predictions.csv": events,
               "profile_conditions.csv": pd.DataFrame(condition)}
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    save_json(OUT / "decision.json", {"directions": decisions, "no_claim": "No regression improvement, final-test success, physical operating cause, or all-shape transition coverage was evaluated."})
    names = list(outputs) + ["decision.json"]
    result = {"status": "completed", "script_sha256": sha(__file__), "contract_sha256": sha(OUT / "contract.json"),
              "inputs_sha256": c["inputs_sha256"], "new_fits": len(manifests), "eligible_hours": len(f),
              "feature_count": len(FEATURES), "outputs_sha256": {n: sha(OUT / n) for n in names},
              "runtime": sys.version, "packages": {n: __import__(n).__version__ for n in ["numpy", "pandas", "sklearn", "joblib"]}}
    save_json(OUT / "run.json", result)
    print(json.dumps({"status": "completed", "new_fits": len(manifests), "decisions": decisions}), flush=True)


if __name__ == "__main__":
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    {"freeze": freeze, "run": run}[sys.argv[1]]()
