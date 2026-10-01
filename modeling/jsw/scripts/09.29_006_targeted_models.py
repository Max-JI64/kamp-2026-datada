"""M006: one frozen past-high blend and the existing q75 comparator.

Run --phase freeze before --phase run. Existing predictions and labels are
read-only. Only the missing June-end q75 estimator is fitted.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent.parent
PREFIX = "09.29_006"
BLEND = "hgb_lag1_high_blend05"
Q75 = "hgb_q75_calendar"
FROZEN = HERE / "tables" / f"{PREFIX}_frozen.json"
KEY = ["stage", "target", "method", "record_key"]
FOLDS = [
    {"stage": "May", "cutoff": "2021-05-01", "end": "2021-06-01", "train_n": 2712, "n": 744, "threshold": 183.0},
    {"stage": "Jun", "cutoff": "2021-06-01", "end": "2021-07-01", "train_n": 3456, "n": 720, "threshold": 181.0},
    {"stage": "Jul-Aug", "cutoff": "2021-07-01", "end": "2021-09-01", "train_n": 4176, "n": 1344, "threshold": 182.0},
]


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read(name, rows=False, **kwargs):
    return pd.read_csv(HERE / ("predictions" if rows else "tables") / name,
                       encoding="utf-8-sig", **kwargs)


def save(frame, name, rows=False):
    frame.to_csv(HERE / ("predictions" if rows else "tables") / f"{PREFIX}_{name}.csv",
                 index=False, encoding="utf-8-sig")


def jsave(value, name):
    (HERE / "tables" / f"{PREFIX}_{name}.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def protected_files():
    paths = list(ROOT.glob("*"))
    for name in ("EDA", "analysis", "data", "modeling"):
        paths.extend((ROOT / name).rglob("*"))
    return sorted(p for p in set(paths) if p.is_file() and "__pycache__" not in p.parts
                  and not (HERE in p.parents and
                           (p.name.startswith(PREFIX) or p in (HERE / "README.md", HERE / "LOG.md"))))


def freeze():
    assert not FROZEN.exists(), "Frozen contract already exists; use --phase run without rewriting it."
    qcfg = json.loads((HERE / "tables/09.26_002_frozen.json").read_text(encoding="utf-8"))
    bcfg = json.loads((HERE / "tables/09.26_001_frozen.json").read_text(encoding="utf-8"))
    handoff = json.loads((HERE / "tables/09.29_005_m006_handoff.json").read_text(encoding="utf-8"))
    assert handoff["candidate"]["fixed_weight"] == .5
    cfg = {
        "frozen_at": now(), "question": "Reduce peak high_continued mean shortfall with one past-only policy change",
        "source_sha256": bcfg["source_sha256"], "folds": FOLDS,
        "targets": {"mean": "original CSV mean", "peak": "maximum of four original slots"},
        "primary_candidate": {"method": BLEND, "target": "peak", "weight": .5,
                              "gate": "observed peak at target minus exactly one hour >= stage training threshold",
                              "change": "prediction combination policy only; no new features, loss or update policy"},
        "baselines": ["hgb_calendar", "lag1", "lag24", "lag168"],
        "existing_comparator": {"method": Q75, "settings": qcfg["candidate_settings"],
                                "development": "reuse M002 predictions", "review": "one June-end fit; no July refit",
                                "purpose": "compare existing loss tradeoff on the same future period, not a second search"},
        "hgb_settings": bcfg["settings"]["hgb"], "feature_order": bcfg["all_features"],
        "input_time": "all previous record slots completed; target calendar known; exact hourly grid preserved",
        "forbidden_inputs_or_gate": ["target actual", "actual next state", "full current-day profile", "daily maximum", "target production", "target weather", "factory staffing", "cost"],
        "policy_id": "m001_chronological_static_june_end_observed_past_updates",
        "common_hours": {"May": 744, "Jun": 720, "July": 600, "August": 744},
        "clip": "all predictions >= 0", "event": "peak >= preserved training threshold; fixed diagnostic only",
        "primary_metric": "high_continued mean shortfall including zero shortfalls",
        "basic_adoption": "pooled development continuation shortfall strictly lower; neither May nor June overall peak MAE worse; both review monthly MAE non-worse and continuation shortfall lower",
        "peak_candidate": "pooled development continuation shortfall strictly lower; no development month continuation deterioration; leave-one-date and leave-one-week pooled primary deltas <= 0; both review month primary deltas < 0; keep only for M007 counter-effect evaluation",
        "counter_effects": ["descending mean over and MAE", "ordinary mean over", "zero MAE", "overall MAE/RMSE", "date-weighted tied daily peak shortfall", "profile seen/unseen", "production posthoc", "fixed-threshold warning burden"],
        "loss_tolerance": "no operational tolerance identified; peak-only retention is an exploratory M007 comparator, not deployment approval",
        "stop": "if primary gain or robustness fails, discard; no weight, quantile, threshold, feature or update search",
        "period_interpretation": "May-Jun exploratory development; July-August exploratory follow-up already used in M005; no September",
        "E025": "completed; retain original mean/peak definitions and existing 26 inputs; no S/D addition needed",
        "protected_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in protected_files()},
        "implementation_sha256": digest(Path(__file__)),
    }
    jsave(cfg, "frozen")
    print(json.dumps({"phase": "freeze", "time": cfg["frozen_at"], "protected_files": len(cfg["protected_sha256"]),
                      "primary": BLEND, "q75_new_fits": 1}), flush=True)


def metrics(g):
    e = g.prediction.to_numpy() - g.actual.to_numpy()
    return {"n": len(g), "dates": int(g.date.nunique()), "mae": float(np.abs(e).mean()),
            "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(e.mean()),
            "under_n": int((e < 0).sum()), "under_rate": float((e < 0).mean()),
            "mean_shortfall": float(np.maximum(-e, 0).mean()), "mean_over": float(np.maximum(e, 0).mean()),
            "pinball_075": float(np.maximum(.75 * -e, -.25 * -e).mean()),
            "actual_le_prediction_rate": float((e >= 0).mean())}


def sections(p):
    for month, g in p.groupby("month"):
        yield str(month), g
    yield "development", p[p.month.isin([5, 6])]
    yield "review", p[p.month.isin([7, 8])]


def aggregate(p):
    scores, conditions = [], []
    for period, section in sections(p):
        for (target, method), g in section.groupby(["target", "method"]):
            fields = {"period": period, "target": target, "method": method}
            scores.append({**fields, **metrics(g)})
            subsets = {s: g[g.state.eq(s)] for s in ["high_onset", "high_continued", "descending", "ordinary"]}
            subsets.update(high=g[g.peak.ge(g.high_threshold_from_training)], zero=g[g.actual.eq(0)],
                           daily_peak=g[g.is_daily_peak], gate_true=g[g.previous_peak.ge(g.high_threshold_from_training)])
            for name, column in [("profile", "profile_seen_contributed"), ("production", "production_condition")]:
                for val, sub in g.groupby(column):
                    subsets[f"{name}_{val}"] = sub
            for condition, sub in subsets.items():
                if len(sub):
                    conditions.append({**fields, "condition": condition, **metrics(sub)})
    save(pd.DataFrame(scores), "scores")
    save(pd.DataFrame(conditions), "condition_scores")
    return pd.DataFrame(scores), pd.DataFrame(conditions)


def paired(p):
    base = p[p.method.eq("hgb_calendar")].set_index(["stage", "target", "record_key"])
    rows = []
    for method, g in p[~p.method.eq("hgb_calendar")].groupby("method"):
        g = g.set_index(["stage", "target", "record_key"]).sort_index()
        b = base.reindex(g.index)
        np.testing.assert_array_equal(b.actual, g.actual)
        part = g.reset_index()[["stage", "target", "record_key", "source_data_row", "month", "date", "week", "state", "is_daily_peak"]].copy()
        part["method"] = method
        part["actual"] = g.actual.to_numpy()
        part["hgb_prediction"] = b.prediction.to_numpy()
        part["candidate_prediction"] = g.prediction.to_numpy()
        for name in ("abs_error", "shortfall", "over"):
            part[f"delta_{name}"] = g[name].to_numpy() - b[name].to_numpy()
        rows.append(part)
    d = pd.concat(rows, ignore_index=True)
    save(d, "row_differences", rows=True)
    summaries, sensitivity = [], []
    for period, section in sections(d):
        for (target, method), g in section.groupby(["target", "method"]):
            for condition, sub in [("all", g), *[(s, g[g.state.eq(s)]) for s in ["high_onset", "high_continued", "descending", "ordinary"]]]:
                if not len(sub):
                    continue
                for unit in ["date", "week"]:
                    for label, part in sub.groupby(unit):
                        summaries.append({"period": period, "target": target, "method": method, "condition": condition,
                                          "unit": unit, "label": label, "n": len(part),
                                          **{f"delta_{k}": float(part[f"delta_{k}"].mean()) for k in ["abs_error", "shortfall", "over"]}})
                    if condition == "high_continued" and target == "peak" and method in [BLEND, Q75]:
                        for label in sorted(sub[unit].unique()):
                            rest = sub[sub[unit].ne(label)]
                            if len(rest):
                                sensitivity.append({"period": period, "method": method, "unit": unit, "excluded": label,
                                                    "remaining_n": len(rest), "delta_shortfall": float(rest.delta_shortfall.mean()),
                                                    "delta_mae": float(rest.delta_abs_error.mean())})
    save(pd.DataFrame(summaries), "paired_differences")
    save(pd.DataFrame(sensitivity), "sensitivity")
    return pd.DataFrame(sensitivity)


def select(scores, conditions, sensitivity, review=False):
    def val(table, period, method, metric, condition=None):
        g = table[table.period.eq(period) & table.target.eq("peak") & table.method.eq(method)]
        if condition:
            g = g[g.condition.eq(condition)]
        assert len(g) == 1
        return float(g.iloc[0][metric])
    result = []
    for method in [BLEND, Q75]:
        gain = val(conditions, "development", method, "mean_shortfall", "high_continued") < val(conditions, "development", "hgb_calendar", "mean_shortfall", "high_continued") - 1e-10
        month_ok = all(val(conditions, m, method, "mean_shortfall", "high_continued") <= val(conditions, m, "hgb_calendar", "mean_shortfall", "high_continued") + 1e-10 for m in ["5", "6"])
        dev_sens = sensitivity[sensitivity.period.eq("development") & sensitivity.method.eq(method)]
        robust = len(dev_sens) > 0 and bool(dev_sens.delta_shortfall.le(1e-10).all())
        dev_mae = all(val(scores, m, method, "mae") <= val(scores, m, "hgb_calendar", "mae") + 1e-10 for m in ["5", "6"])
        review_gain = all(val(conditions, m, method, "mean_shortfall", "high_continued") < val(conditions, m, "hgb_calendar", "mean_shortfall", "high_continued") - 1e-10 for m in ["7", "8"]) if review else None
        review_mae = all(val(scores, m, method, "mae") <= val(scores, m, "hgb_calendar", "mae") + 1e-10 for m in ["7", "8"]) if review else None
        viable = gain and month_ok and robust and (review_gain if review else True)
        role = "basic_replacement" if viable and dev_mae and (review_mae if review else True) else ("peak_only_for_M007" if viable else "discard")
        result.append({"method": method, "dev_primary_improves": gain, "dev_month_primary_nonworse": month_ok,
                       "dev_date_week_robust": robust, "dev_month_mae_nonworse": dev_mae,
                       "review_month_primary_improves": review_gain, "review_month_mae_nonworse": review_mae,
                       "decision": role, "dev_leaveout_delta_shortfall_min": float(dev_sens.delta_shortfall.min()),
                       "dev_leaveout_delta_shortfall_max": float(dev_sens.delta_shortfall.max())})
    return result


def diagnostics(p, coverage):
    # Diagnostic labels are used only here, after all predictions have been fixed.
    cases, monthly = [], []
    for method, mp in p[p.target.eq("peak")].groupby("method"):
        pred = mp.set_index("record_key").prediction
        for date, full in coverage.groupby("date"):
            if full.stage.iloc[0] not in set(mp.stage):
                continue
            peaks = full[full.is_daily_peak]
            estimates = pred.reindex(peaks.record_key)
            available = estimates.notna()
            errors = estimates.to_numpy() - peaks.peak.to_numpy()
            cases.append({"date": date, "stage": full.stage.iloc[0], "month": int(full.month.iloc[0]), "method": method,
                          "normal_hours": len(full), "available_hours": int(full.eligible.sum()),
                          "prediction_day_status": "complete" if full.eligible.all() else ("partial" if full.eligible.any() else "none"),
                          "original_peak_hours": len(peaks), "available_peak_hours": int(available.sum()),
                          "all_original_peaks_available": bool(available.all()),
                          "day_mean_shortfall_available": float(np.maximum(-errors[available], 0).mean()) if available.any() else np.nan,
                          "day_mae_available": float(np.abs(errors[available]).mean()) if available.any() else np.nan})
        for _, row in coverage[coverage.is_monthly_peak & coverage.stage.isin(mp.stage.unique())].iterrows():
            estimate = pred.get(row.record_key, np.nan)
            monthly.append({"month": row.month, "method": method, "record_key": row.record_key, "actual": row.peak,
                            "prediction": estimate, "shortfall": max(row.peak - estimate, 0), "available": pd.notna(estimate)})
    daily = pd.DataFrame(cases)
    save(daily, "daily_peak_dates", rows=True)
    save(daily.groupby(["stage", "month", "method", "prediction_day_status"]).agg(
        dates=("date", "size"), dates_with_peak_prediction=("available_peak_hours", lambda x: int(x.gt(0).sum())),
        mean_shortfall=("day_mean_shortfall_available", "mean"), mae=("day_mae_available", "mean")).reset_index(), "daily_peak_date_scores")
    save(pd.DataFrame(monthly), "monthly_peak_cases")
    # Reuse M005 full event boundaries, including unavailable starts.
    events = read("09.29_005_event_audit.csv", rows=True, parse_dates=["start", "end"])
    events = events[events.method.eq("hgb_calendar")].copy()
    members = read("09.29_005_event_members.csv", rows=True, parse_dates=["record_key"])
    out = []
    for method, mp in p[p.target.eq("peak")].groupby("method"):
        pred = mp.set_index("record_key").prediction
        for _, event in events[events.stage.isin(mp.stage.unique())].iterrows():
            keys = members[members.event_id.eq(event.event_id)].record_key
            estimates = pred.reindex(keys)
            hit = estimates[estimates.ge(event.threshold)]
            available = pd.notna(estimates.iloc[0])
            outcome = ("start_unknown" if event.left_unknown else "start_unpredictable" if not available else
                       "pre_start_warning" if estimates.iloc[0] >= event.threshold else "late_capture" if len(hit) else "missed")
            first = hit.index[0] if len(hit) else pd.NaT
            out.append({"stage": event.stage, "month": event.month, "method": method, "event_id": event.event_id,
                        "threshold": event.threshold, "start": event.start, "end": event.end,
                        "left_unknown": event.left_unknown, "right_unknown": event.right_unknown,
                        "start_input_available": available, "outcome": outcome,
                        "first_warning_target": first, "warning_after_completed_record": first - pd.Timedelta(hours=1) if len(hit) else pd.NaT,
                        "delay_hours": (first - event.start) / pd.Timedelta(hours=1) if len(hit) else np.nan,
                        "missing_prediction_hours": int(estimates.isna().sum())})
    es = pd.DataFrame(out)
    save(es, "event_audit", rows=True)
    save(es.groupby(["stage", "month", "method", "outcome"]).size().reset_index(name="events"), "event_scores")
    rows = []
    for (month, method), g in p[p.target.eq("peak")].groupby(["month", "method"]):
        alert = g.prediction.ge(g.high_threshold_from_training)
        high = g.actual.ge(g.high_threshold_from_training)
        rows.append({"month": month, "method": method, "n": len(g), "high_n": int(high.sum()),
                     "alarm_hours": int(alert.sum()), "false_alarm_hours": int((alert & ~high).sum()),
                     "missed_high_hours": int((~alert & high).sum()), "threshold_policy": "inherited training u, diagnostic only"})
    save(pd.DataFrame(rows), "fixed_threshold_diagnostics")


def main():
    frozen_bytes = FROZEN.read_bytes()
    cfg = json.loads(frozen_bytes)
    revision = HERE / "tables" / f"{PREFIX}_implementation_revision.json"
    expected_impl = json.loads(revision.read_text(encoding="utf-8"))["implementation_sha256"] if revision.exists() else cfg["implementation_sha256"]
    assert cfg["folds"] == FOLDS and expected_impl == digest(Path(__file__))
    for name, value in cfg["protected_sha256"].items():
        assert digest(ROOT / name) == value, name
    print(json.dumps({"runtime_started": platform.python_version(), "phase": "run"}), flush=True)
    spec = importlib.util.spec_from_file_location("m001_readonly", HERE / "scripts/09.26_001_power_models.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    X, labels, _, full, source_hash = base.read_data(save_catalog=False)
    assert full == cfg["feature_order"] and base.SETTINGS["hgb"] == cfg["hgb_settings"]
    assert source_hash == cfg["source_sha256"]
    audit = read("09.29_005_row_audit.csv", rows=True, parse_dates=["record_key"])
    p = audit[audit.method.isin(["hgb_calendar", "lag1", "lag24", "lag168", Q75])].copy()
    assert not p.duplicated(KEY).any()
    p = p.drop(columns=["error", "shortfall", "over"])
    # M005 row_audit contains diagnostic methods only. Restore the frozen
    # auxiliary baselines from M001, using identical HGB diagnostic rows.
    extras = []
    saved = pd.concat([read(f"09.26_001_{part}_predictions.csv", rows=True, parse_dates=["record_key"])
                       for part in ["development", "review"]], ignore_index=True)
    for target in ["mean", "peak"]:
        template = p[p.target.eq(target) & p.method.eq("hgb_calendar")].set_index(["stage", "record_key"]).sort_index()
        for method in ["lag24", "lag168"]:
            if method in set(p.method):
                continue
            baseline = saved[saved.target.eq(target) & saved.method.eq(method)].set_index(["stage", "record_key"]).sort_index()
            assert template.index.equals(baseline.index)
            np.testing.assert_array_equal(template.actual, baseline.actual)
            extra = template.copy()
            extra["method"] = method
            extra["prediction"] = baseline.prediction
            extras.append(extra.reset_index())
    p = pd.concat([p, *extras], ignore_index=True)
    blend = p[p.target.eq("peak") & p.method.eq("hgb_calendar")].copy()
    for f in FOLDS:
        train = X.index < pd.Timestamp(f["cutoff"])
        test = (X.index >= pd.Timestamp(f["cutoff"])) & (X.index < pd.Timestamp(f["end"]))
        assert int(train.sum()) == f["train_n"] and int(test.sum()) == f["n"]
        assert X.index[train].max() < X.index[test].min()
        assert base.threshold_before(labels, pd.Timestamp(f["cutoff"])) == f["threshold"]
        for (target, method), g in p[p.stage.eq(f["stage"])].groupby(["target", "method"]):
            g = g.sort_values("record_key")
            assert list(g.record_key) == list(X.index[test])
            np.testing.assert_array_equal(g.actual, labels.loc[test, target])
            np.testing.assert_array_equal(g.source_data_row, labels.loc[test, "source_data_row"])
            np.testing.assert_array_equal(g.previous_peak, X.loc[test, "peak_lag1"])
            if method.startswith("lag"):
                np.testing.assert_array_equal(g.prediction, X.loc[test, f"{target}_{method}"])
    # The gate is the only primary change. No posthoc diagnostic label is read here.
    blend["gate_previous_high"] = blend.previous_peak.ge(blend.high_threshold_from_training)
    blend["prediction"] = np.where(blend.gate_previous_high,
                                   .5 * blend.prediction + .5 * blend.previous_peak, blend.prediction)
    blend["method"] = BLEND
    p = pd.concat([p, blend], ignore_index=True)
    for name in ["error", "shortfall", "over", "abs_error"]:
        p[name] = p.prediction - p.actual if name == "error" else (np.maximum(p.actual - p.prediction, 0) if name == "shortfall" else np.maximum(p.prediction - p.actual, 0) if name == "over" else np.abs(p.prediction - p.actual))
    p["production_condition"] = np.where(p.production_posthoc.eq(0), "zero", "positive")
    dev = p[p.month.isin([5, 6])]
    ds, dc = aggregate(dev)
    sensitivity = paired(dev)
    dev_selection = select(ds, dc, sensitivity)
    jsave({"decided_at": now(), "before_q75_review_fit": True, "candidates": dev_selection}, "development_decision")
    print(json.dumps({"development_decision": dev_selection}), flush=True)
    # q75 is a pre-existing fixed comparator. One missing review fit, no search.
    train = X.index < pd.Timestamp("2021-07-01")
    test = X.index >= pd.Timestamp("2021-07-01")
    model_path = HERE / "models" / f"{PREFIX}_q75_june_end.joblib"
    model_path.parent.mkdir(exist_ok=True)
    fitted = not model_path.exists()
    with threadpool_limits(limits=2):
        if fitted:
            model = HistGradientBoostingRegressor(**cfg["existing_comparator"]["settings"])
            model.fit(X.loc[train, full], labels.loc[train, "peak"])
            joblib.dump({"estimator": model, "feature_order": full, "train_cutoff_exclusive": "2021-07-01",
                         "settings": cfg["existing_comparator"]["settings"], "source_sha256": source_hash}, model_path)
        bundle = joblib.load(model_path)
        assert bundle["settings"] == cfg["existing_comparator"]["settings"] and bundle["feature_order"] == full
        assert bundle["train_cutoff_exclusive"] == "2021-07-01" and bundle["source_sha256"] == source_hash
        estimates = np.maximum(bundle["estimator"].predict(X.loc[test, full]), 0)
    loaded = joblib.load(model_path)
    with threadpool_limits(limits=2):
        np.testing.assert_array_equal(np.maximum(loaded["estimator"].predict(X.loc[test, loaded["feature_order"]]), 0), estimates)
    new = p[p.stage.eq("Jul-Aug") & p.target.eq("peak") & p.method.eq("hgb_calendar")].sort_values("record_key").copy()
    new["method"] = Q75
    new["prediction"] = estimates
    p = pd.concat([p, new], ignore_index=True)
    p["error"] = p.prediction - p.actual
    p["shortfall"] = np.maximum(-p.error, 0)
    p["over"] = np.maximum(p.error, 0)
    p["abs_error"] = np.abs(p.error)
    p["gate_previous_high"] = p.previous_peak.ge(p.high_threshold_from_training)
    p["policy_id"] = cfg["policy_id"]
    p["input_available"] = True
    p["train_cutoff_exclusive"] = p.stage.map({f["stage"]: f["cutoff"] for f in FOLDS})
    p["training_end"] = p.stage.map({f["stage"]: str(X.index[X.index < pd.Timestamp(f["cutoff"])].max()) for f in FOLDS})
    p["forecast_after_completed_record"] = p.record_key - pd.Timedelta(hours=1)
    p["evaluation_kind"] = np.where(p.month.isin([5, 6]), "exploratory_development", "exploratory_followup")
    p["posthoc_fields_not_inputs"] = "actual,state,production_posthoc,profile_id,profile_seen_contributed,is_daily_peak,event_id"
    assert not p.duplicated(KEY).any() and np.isfinite(p.prediction).all() and p.prediction.ge(0).all()
    assert p.groupby(["stage", "target", "method"]).size().eq(p.groupby(["stage", "target", "method"]).size().index.get_level_values("stage").map({f["stage"]: f["n"] for f in FOLDS})).all()
    p = p.sort_values(KEY)
    save(p[p.month.isin([5, 6])], "development_predictions", rows=True)
    save(p[p.month.isin([7, 8])], "review_predictions", rows=True)
    scores, conditions = aggregate(p)
    sensitivity = paired(p)
    decisions = select(scores, conditions, sensitivity, review=True)
    save(pd.DataFrame(decisions), "selection")
    coverage = read("09.29_005_target_coverage.csv", rows=True, parse_dates=["record_key"])
    save(coverage, "target_coverage", rows=True)
    cov = coverage.groupby(["stage", "month"]).agg(normal_hours=("record_key", "size"), common_hours=("eligible", "sum"),
                                                    missing_hours=("eligible", lambda v: int((~v).sum()))).reset_index()
    save(cov, "coverage")
    diagnostics(p, coverage)
    basic = next((d["method"] for d in decisions if d["decision"] == "basic_replacement"), "hgb_calendar")
    retained = [d["method"] for d in decisions if d["decision"] != "discard"]
    handoff = {"created_at": now(), "status": "M006 completed; M007 not executed", "basic_point_models": {"mean": "hgb_calendar", "peak": basic},
               "peak_methods_to_compare": list(dict.fromkeys(["hgb_calendar", "lag1", *retained])),
               "auxiliary_baselines": ["lag24", "lag168"], "candidate_decisions": decisions,
               "prediction_files": [f"predictions/{PREFIX}_development_predictions.csv", f"predictions/{PREFIX}_review_predictions.csv"],
               "unavailable_rows_file": f"predictions/{PREFIX}_target_coverage.csv", "folds": FOLDS,
               "policy_id": cfg["policy_id"], "q75_review_model": model_path.relative_to(HERE).as_posix(),
               "q75_development_models": "not serialized; saved M002 predictions, retrain using its frozen settings if needed",
               "hgb_artifact": "saved M001 predictions; exact reconstruction in scripts/09.26_001_power_models.py and tables/09.26_001_frozen.json",
               "blend_artifact": "saved HGB and lag1 plus frozen .5 past-high gate; no fitted estimator",
               "M007_contract": "choose warning policy on May-Jun only; freeze for July-August; preserve missing-start service denominator; outputs are record forecasts, not peak probabilities",
               "not_run": ["M007 warning threshold optimization", "multi-horizon forecasting", "September predictions", "coverage fallback", "new features", "July model refit"]}
    jsave(handoff, "m007_handoff")
    for name, value in cfg["protected_sha256"].items():
        assert digest(ROOT / name) == value, name
    assert FROZEN.read_bytes() == frozen_bytes
    facts = {"started_from_frozen_at": cfg["frozen_at"], "completed_at": now(), "python": platform.python_version(),
             "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__,
             "new_fits_this_execution": int(fitted), "new_fits_total_session": 1, "new_fit_method": Q75, "train_n": int(train.sum()), "review_n": int(test.sum()),
             "development_prediction_rows": int(p.month.isin([5, 6]).sum()), "review_prediction_rows": int(p.month.isin([7, 8]).sum()),
             "protected_files_unchanged": len(cfg["protected_sha256"]), "model_sha256": digest(model_path),
             "frozen_sha256": sha256(frozen_bytes).hexdigest(), "model_reload_predictions_exact": True,
             "successful_child_exit_code": 0, "decisions": decisions}
    jsave(facts, "facts")
    print(json.dumps(facts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["freeze", "run"], required=True)
    args = parser.parse_args()
    freeze() if args.phase == "freeze" else main()
