"""M007: evaluate saved one-record forecasts; never fit or adjust on review data."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
PREFIX = "09.29_007"
HOUR = pd.Timedelta(hours=1)
METHODS = ["hgb_calendar", "lag1", "hgb_lag1_high_blend05", "hgb_q75_calendar"]
BUDGETS = [1, 2, 4]
INPUTS = [
    "data/origin/okm_augumented_2021.csv",
    *[f"modeling/jsw/predictions/09.29_006_{x}.csv" for x in
      ("development_predictions", "review_predictions", "target_coverage")],
    "modeling/jsw/predictions/09.29_005_event_audit.csv",
    "modeling/jsw/predictions/09.29_005_event_members.csv",
    "modeling/jsw/tables/09.29_006_m007_handoff.json",
    "modeling/jsw/tables/09.29_006_selection.csv",
    "modeling/jsw/predictions/09.26_002_development_predictions.csv",
    "modeling/jsw/predictions/09.26_003_development_predictions.csv",
    "modeling/jsw/tables/09.26_004_selection_scores.csv",
    "modeling/jsw/tables/09.26_004_by_high_state.csv",
]


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative, **kwargs):
    return pd.read_csv(HERE / relative, encoding="utf-8-sig", **kwargs)


def save(frame, name, rows=False):
    frame.to_csv(HERE / ("predictions" if rows else "tables") / f"{PREFIX}_{name}.csv",
                 index=False, encoding="utf-8-sig")


def write_json(value, name):
    (HERE / "tables" / f"{PREFIX}_{name}.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def ratio(a, b):
    return a / b if b else np.nan


def choose_threshold(scores, days, budget):
    """Use scores and date count only. Select complete score ties, never labels."""
    scores = np.asarray(scores, dtype=float)
    assert len(scores) and np.isfinite(scores).all() and days > 0 and budget >= 0
    values, counts = np.unique(scores, return_counts=True)
    values, counts = values[::-1], counts[::-1]
    cum = np.cumsum(counts)
    feasible = np.flatnonzero(cum <= days * budget)
    if not len(feasible):
        return float(np.nextafter(values[0], np.inf)), 0
    i = feasible[-1]
    # Among candidate score boundaries with identical flags, this is maximal c.
    return float(values[i]), int(cum[i])


def confusion(actual, warning):
    a, w = np.asarray(actual, bool), np.asarray(warning, bool)
    tp, fp, fn, tn = (int((a & w).sum()), int((~a & w).sum()),
                      int((a & ~w).sum()), int((~a & ~w).sum()))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn)}


def mark_alerts(frame):
    f = frame.sort_values("record_key").copy().reset_index(drop=True)
    f["available"] = f.prediction.notna()
    # Missing warning is NA, not a normal/non-warning observation.
    f["warning"] = pd.Series(pd.NA, index=f.index, dtype="boolean")
    f.loc[f.available, "warning"] = f.loc[f.available, "prediction"].ge(f.loc[f.available, "c"])
    f["actual_high"] = f.peak.ge(f.u)
    adjacent = f.record_key.diff().eq(HOUR)
    previous_warning = f.warning.shift(1).fillna(False).astype(bool) & adjacent
    f["active_before_update"] = previous_warning
    f["new_notification"] = f.warning.fillna(False).astype(bool) & ~previous_warning
    f["active_record_hours"] = f.warning.astype("Float64")
    f["episode_id"] = ""
    seq, current = 0, ""
    for i, row in f.iterrows():
        if row.available and bool(row.warning):
            if row.new_notification:
                seq += 1
                current = f"{row.phase}|{row.method}|b{row.budget}|{seq:04d}"
            f.at[i, "episode_id"] = current
        else:
            current = ""
    return f


def bundles(f):
    lookup = f.set_index("record_key")
    rows = []
    for eid, g in f.loc[f.episode_id.ne("")].groupby("episode_id", sort=False):
        s, e = g.record_key.min(), g.record_key.max()
        nxt = lookup.loc[e + HOUR] if e + HOUR in lookup.index else None
        if nxt is not None:
            reason = "non_warning" if nxt.available else "prediction_unavailable"
        elif e == f.record_key.max():
            reason = "evaluation_boundary"
        else:
            reason = "record_gap"
        no_high = not bool(g.actual_high.any())
        rows.append({"phase": g.phase.iloc[0], "method": g.method.iloc[0],
                     "budget": g.budget.iloc[0], "episode_id": eid, "start": s, "end": e,
                     "issue_after_record": s - HOUR, "release_boundary_record": e,
                     "end_reason": reason, "complete_end_observed": reason == "non_warning",
                     "start_month": s.month, "start_date": s.strftime("%Y-%m-%d"),
                     "raw_warning_hours": len(g), "active_record_hours": len(g),
                     "high_target_hours": int(g.actual_high.sum()),
                     "false_bundle_observed": no_high,
                     "false_bundle_complete": no_high and reason == "non_warning",
                     "false_bundle_censored": no_high and reason != "non_warning"})
    return pd.DataFrame(rows, columns=["phase", "method", "budget", "episode_id", "start", "end",
        "issue_after_record", "release_boundary_record", "end_reason", "complete_end_observed",
        "start_month", "start_date", "raw_warning_hours", "active_record_hours", "high_target_hours",
        "false_bundle_observed", "false_bundle_complete", "false_bundle_censored"])


def event_outcome(start_available, left_unknown, flags):
    if left_unknown:
        return "start_unknown"
    if not start_available:
        return "start_unavailable"
    if flags[0] is True:
        return "pre_start_capture"
    if any(x is True for x in flags[1:]):
        return "late_capture"
    return "missed_observed"


def peak_capture(flags):
    available = [x for x in flags if x is not None]
    if not available:
        return "unavailable", "unavailable"
    any_result = "captured" if any(available) else (
        "partial_unknown" if len(available) != len(flags) else "not_captured")
    all_result = "unavailable" if len(available) != len(flags) else (
        "captured" if all(available) else "not_captured")
    return any_result, all_result


def self_tests():
    checks = []
    def passed(name):
        checks.append({"case": name, "passed": True})
    c, n = choose_threshold([9, 8, 8, 7], 1, 2)
    assert c == 9 and n == 1
    passed("score_ties_do_not_exceed_budget")
    c, n = choose_threshold([5, 5, 5], 1, 1)
    assert c > 5 and n == 0
    passed("tie_larger_than_budget_no_warning_boundary")
    c, n = choose_threshold([9, 8, 8, 7], 1, 4)
    assert c == 7 and n == 4
    passed("all_warning_threshold_inclusive")
    def fixture(times, preds, peaks=None):
        return mark_alerts(pd.DataFrame({"record_key": pd.to_datetime(times),
            "prediction": preds, "peak": peaks or [0] * len(times), "u": 5., "c": 5.,
            "phase": "test", "method": "manual", "budget": 2}))
    times = ["2021-05-31 23:00", "2021-06-01 00:00", "2021-06-01 01:00"]
    f = fixture(times, [6, 6, 0])
    b = bundles(f)
    assert len(b) == 1 and b.raw_warning_hours.iloc[0] == 2 and f.new_notification.sum() == 1
    passed("date_and_month_boundary_keeps_one_bundle")
    f = fixture(["2021-05-01 00:00", "2021-05-01 02:00"], [6, 6])
    assert len(bundles(f)) == 2 and f.new_notification.sum() == 2
    passed("hour_gap_breaks_bundle")
    f = fixture(times, [6, np.nan, 6])
    assert pd.isna(f.warning.iloc[1]) and len(bundles(f)) == 2
    passed("unavailable_breaks_bundle_stays_unknown")
    f = fixture(times, [0, 0, 0])
    assert len(bundles(f)) == 0 and not f.new_notification.any()
    assert np.isnan(confusion([False] * 3, [False] * 3)["precision"])
    passed("no_warning_zero_precision_denominator")
    f = fixture(times, [6, 6, 6])
    assert len(bundles(f)) == 1 and f.active_record_hours.sum() == 3
    assert bundles(f).end_reason.iloc[0] == "evaluation_boundary"
    passed("all_warning_one_censored_bundle")
    assert event_outcome(True, False, [False, True]) == "late_capture"
    assert event_outcome(False, False, [None, True]) == "start_unavailable"
    assert event_outcome(True, True, [True]) == "start_unknown"
    passed("late_missing_and_unknown_starts_not_pre_warning")
    f = fixture(times, [6, 0, 0], [0, 7, 7])
    assert f.active_before_update.iloc[1] and not f.warning.iloc[1]
    assert event_outcome(True, False, [False, False]) == "missed_observed"
    passed("previous_active_cleared_before_onset_not_capture")
    assert peak_capture([True, False]) == ("captured", "not_captured")
    assert peak_capture([True, None]) == ("captured", "unavailable")
    assert peak_capture([False, None]) == ("partial_unknown", "unavailable")
    assert peak_capture([None]) == ("unavailable", "unavailable")
    passed("daily_peak_ties_and_partial_coverage")
    assert np.isnan(confusion([False, False], [True, True])["recall"])
    passed("no_actual_high_zero_recall_denominator")
    assert confusion([], []) == {"tp": 0, "fp": 0, "fn": 0, "tn": 0,
        "precision": np.nan, "recall": np.nan, "f1": np.nan}
    passed("empty_group_denominators")
    write_json({"time": now(), "python": platform.python_version(), "checks": checks}, "self_tests")
    return checks


def load_predictions(name):
    p = read(f"predictions/09.29_006_{name}_predictions.csv", parse_dates=["record_key", "forecast_after_completed_record"])
    p = p.loc[p.target.eq("peak") & p.method.isin(METHODS)].copy()
    assert not p.duplicated(["method", "record_key"]).any()
    assert np.isfinite(p.prediction).all() and p.prediction.ge(0).all()
    assert (p.record_key - p.forecast_after_completed_record).eq(HOUR).all()
    assert pd.to_datetime(p.training_end).lt(p.forecast_after_completed_record + HOUR).all()
    ref = p.loc[p.method.eq(METHODS[0])].sort_values("record_key")
    for m in METHODS:
        g = p.loc[p.method.eq(m)].sort_values("record_key")
        assert g.record_key.tolist() == ref.record_key.tolist()
        np.testing.assert_array_equal(g.actual, ref.actual)
        np.testing.assert_array_equal(g.high_threshold_from_training, ref.high_threshold_from_training)
    return p


def freeze():
    self_tests()
    path = HERE / "tables" / f"{PREFIX}_frozen.json"
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))
        for rel, h in config["input_sha256"].items():
            assert sha(ROOT / rel) == h, rel
        print(json.dumps({"status": "reused_existing_frozen", "created_at": config["created_at"]}))
        return
    for rel in INPUTS:
        if not (ROOT / rel).is_file():
            raise FileNotFoundError(f"Required saved input missing; no training allowed: {rel}")
    p = load_predictions("development")
    assert p.record_key.dt.month.isin([5, 6]).all()
    assert len(p) == 1464 * len(METHODS)
    days = p.record_key.dt.date.nunique()
    assert days == 61
    policies, internal = [], []
    for m in METHODS:
        g = p.loc[p.method.eq(m)]
        for b in BUDGETS:
            c, n = choose_threshold(g.prediction, days, b)
            policies.append({"method": m, "budget": b, "c": c, "development_dates": days,
                             "development_warning_hours": n, "development_hours_per_day": n / days})
            may = g.loc[g.record_key.dt.month.eq(5)]
            c5, n5 = choose_threshold(may.prediction, 31, b)
            internal.append({"method": m, "budget": b, "c": c5,
                             "may_warning_hours": n5, "selection": "May_only_applied_to_June"})
    allowed = {"modeling/jsw/README.md", "modeling/jsw/LOG.md"}
    protected = {}
    for path2 in ROOT.rglob("*"):
        if not path2.is_file() or any(x in path2.parts for x in (".git", "__pycache__", ".venv")):
            continue
        rel = path2.relative_to(ROOT).as_posix()
        if rel in allowed or (rel.startswith("modeling/jsw/") and PREFIX in path2.name):
            continue
        protected[rel] = sha(path2)
    write_json({"created_at": now(), "target": "peak", "methods": METHODS, "budgets": BUDGETS,
        "primary_budget": 2, "u_by_stage": {"May": 183., "Jun": 181., "Jul-Aug": 182.},
        "development": "May-June scores only; 61 common complete dates; labels unused for c",
        "review": "July-August exploratory; c never recalibrated",
        "threshold_candidates": "distinct scores plus nextafter(max,+inf); prediction>=c; keep full ties",
        "selection_rule": "max warning hours <= budget*dates; most conservative candidate for identical flags",
        "bundle_rule": "first true=new; consecutive exact-hour true=active; false/missing/gap=break; date unchanged",
        "phase_boundary": "development/review reset; monthly/date boundaries within phase do not reset",
        "active_definition": "one nominal record interval per true flag; not measured physical response duration",
        "false_bundle_definition": "no high target observed while flags active; complete and censored separated",
        "start_definition": "only warning on first target counts pre-start; previous active reported separately",
        "decision_rule": "HGB reference; strict Pareto improvements in start rate, raw hours, new/false bundles across both review months support adoption; gains with losses=tradeoff; no gain=reject; no operational tolerance inferred",
        "policies": policies, "internal_policies": internal,
        "input_sha256": {rel: sha(ROOT / rel) for rel in INPUTS},
        "protected_sha256": protected}, "frozen")
    save(pd.DataFrame(policies), "thresholds")
    print(json.dumps({"status": "frozen_from_development_only", "policies": policies}, ensure_ascii=False))


def reproduce_m004():
    q = read("predictions/09.26_002_development_predictions.csv")
    p = read("predictions/09.26_003_development_predictions.csv")
    p = p.loc[p.target.eq("peak") & ~p.method.eq("hgb_calendar")]
    combined = pd.concat([q, p], ignore_index=True)
    rows, states = [], []
    for period in ["pooled", "May", "Jun"]:
        sub = combined if period == "pooled" else combined.loc[combined.stage.eq(period)]
        for m, g in sub.groupby("method"):
            a, w = g.actual.ge(g.high_threshold_from_training), g.prediction.ge(g.high_threshold_from_training)
            c = confusion(a, w)
            rows.append({"period": period, "method": m, "n": len(g), "high_actual_n": int(a.sum()),
                "not_high_actual_n": int((~a).sum()), "selected_n": int(w.sum()),
                "hit_n": c["tp"], "miss_n": c["fn"], "false_selection_n": c["fp"],
                "recall": c["recall"], "precision": c["precision"]})
            for state in ["high_onset", "high_continued"]:
                s = g.loc[g.peak_state.eq(state)]
                hit = int(s.prediction.ge(s.high_threshold_from_training).sum())
                states.append({"period": period, "method": m, "peak_state": state,
                               "actual_n": len(s), "hit_n": hit, "miss_n": len(s) - hit})
    comparisons = []
    for calculated, old_name, keys in [(pd.DataFrame(rows), "selection_scores", ["period", "method"]),
                                      (pd.DataFrame(states), "by_high_state", ["period", "method", "peak_state"])]:
        old = read(f"tables/09.26_004_{old_name}.csv").set_index(keys).sort_index()
        new = calculated.set_index(keys).sort_index()
        assert old.index.equals(new.index)
        for col in old.columns:
            np.testing.assert_allclose(new[col], old[col], atol=1e-12, rtol=0, equal_nan=True)
            comparisons.append({"table": old_name, "column": col, "groups": len(old),
                                "max_abs_difference": float((new[col] - old[col]).abs().max())})
    save(pd.DataFrame(rows), "m004_selection_reproduced")
    save(pd.DataFrame(states), "m004_states_reproduced")
    save(pd.DataFrame(comparisons), "m004_reproduction")


def event_rows(f, catalogue, members):
    lookup = f.set_index("record_key")
    rows = []
    for _, e in catalogue.iterrows():
        keys = members.loc[members.event_id.eq(e.event_id), "record_key"].sort_values()
        g = lookup.reindex(keys)
        flags = [bool(x) if pd.notna(x) else None for x in g.warning]
        available = int(g.prediction.notna().sum())
        firsts = g.loc[g.warning.fillna(False).astype(bool)]
        first = firsts.index[0] if len(firsts) else pd.NaT
        start = pd.Timestamp(e.start)
        assert keys.iloc[0] == start and keys.iloc[-1] == pd.Timestamp(e.end)
        start_av = flags[0] is not None
        outcome = event_outcome(start_av, bool(e.left_unknown), flags)
        row = g.iloc[0]
        rows.append({"phase": f.phase.iloc[0], "method": f.method.iloc[0], "budget": f.budget.iloc[0],
            "event_id": e.event_id, "stage": e.stage, "month": start.month, "u": e.threshold,
            "start": start, "end": e.end, "hours_observed": len(keys), "maximum": e.maximum,
            "left_unknown": e.left_unknown, "right_unknown": e.right_unknown,
            "left_reason": e.left_reason, "right_reason": e.right_reason,
            "start_available": start_av, "start_evaluable": start_av and not e.left_unknown,
            "outcome": outcome, "predicted_hours": available, "missing_prediction_hours": len(keys) - available,
            "complete_miss_confirmed": outcome == "missed_observed" and available == len(keys),
            "partial_miss_unresolved": outcome == "missed_observed" and available < len(keys),
            "any_observed_capture": bool(len(firsts)), "first_warning_target": first,
            "issue_after_completed_record": first - HOUR if len(firsts) else pd.NaT,
            "delay_target_records": float((first - start) / HOUR) if len(firsts) else np.nan,
            "active_before_start_update": bool(row.active_before_update),
            "start_target_warning": flags[0], "start_new_notification": bool(row.new_notification),
            "start_continued_warning": flags[0] is True and not bool(row.new_notification),
            "start_unavailable_reason": row.input_missing_reason if not start_av else "none"})
    return rows


def hour_summary(f, b, period):
    available = f.loc[f.available]
    d = f.groupby("date").available.agg(["count", "sum"])
    common_days = int(d["sum"].gt(0).sum())
    complete = int((d["sum"].eq(d["count"]) & d["count"].eq(24)).sum())
    raw = int(available.warning.sum())
    high_unavailable = int((~f.available & f.actual_high).sum())
    out = {"phase": f.phase.iloc[0], "period": period, "method": f.method.iloc[0],
        "budget": f.budget.iloc[0], "c": f.c.iloc[0], "u_values": ",".join(map(str, sorted(f.u.unique()))),
        "normal_hours": len(f), "available_hours": len(available), "unavailable_hours": len(f) - len(available),
        "calendar_dates": (f.record_key.max().normalize() - f.record_key.min().normalize()).days + 1,
        "normal_dates": len(d), "common_prediction_dates": common_days, "complete_prediction_dates": complete,
        "partial_prediction_dates": int(d["sum"].gt(0).sum()) - complete,
        "no_prediction_dates": int(d["sum"].eq(0).sum()), "high_unavailable_hours": high_unavailable,
        "raw_warning_hours": raw, "raw_hours_per_common_day": ratio(raw, common_days),
        "raw_hours_per_normal_day": ratio(raw, len(d)),
        "active_record_hours": raw, "active_fraction_available": ratio(raw, len(available)),
        "new_notifications": int(f.new_notification.sum()),
        "new_notifications_per_common_day": ratio(int(f.new_notification.sum()), common_days),
        "false_bundles_observed": int(b.false_bundle_observed.sum()),
        "false_bundles_complete": int(b.false_bundle_complete.sum()),
        "false_bundles_censored": int(b.false_bundle_censored.sum()),
        "bundles_started": len(b), "bundle_mean_hours": b.raw_warning_hours.mean(),
        "bundle_median_hours": b.raw_warning_hours.median(), "bundle_max_hours": b.raw_warning_hours.max()}
    out.update(confusion(available.actual_high, available.warning))
    assert sum(out[x] for x in ("tp", "fp", "fn", "tn")) == len(available)
    assert out["tp"] + out["fp"] == raw
    return out


def summarize_events(e, period):
    counts = e.outcome.value_counts()
    den = int(e.start_evaluable.sum())
    r = {"phase": e.phase.iloc[0], "period": period, "method": e.method.iloc[0],
        "budget": e.budget.iloc[0], "events": len(e), "start_identifiable_events": int((~e.left_unknown).sum()),
        "start_evaluable_events": den, "start_available_rate": ratio(den, int((~e.left_unknown).sum())),
        "events_any_prediction": int(e.predicted_hours.gt(0).sum()),
        "events_any_capture": int(e.any_observed_capture.sum()),
        "any_capture_rate_evaluable": ratio(int(e.any_observed_capture.sum()), int(e.predicted_hours.gt(0).sum())),
        "complete_miss_confirmed": int(e.complete_miss_confirmed.sum()),
        "partial_miss_unresolved": int(e.partial_miss_unresolved.sum()),
        "active_before_start_update": int(e.active_before_start_update.sum()),
        "start_new_notifications": int(e.start_new_notification.sum()),
        "start_continued_warnings": int(e.start_continued_warning.sum()),
        "right_unknown_events": int(e.right_unknown.sum())}
    for outcome in ["pre_start_capture", "late_capture", "missed_observed", "start_unavailable", "start_unknown"]:
        r[outcome] = int(counts.get(outcome, 0))
    r["pre_start_rate_evaluable"] = ratio(r["pre_start_capture"], den)
    r["pre_start_rate_service"] = ratio(r["pre_start_capture"], r["start_identifiable_events"])
    r["late_delay_mean_records"] = e.loc[e.outcome.eq("late_capture"), "delay_target_records"].mean()
    r["late_delay_max_records"] = e.loc[e.outcome.eq("late_capture"), "delay_target_records"].max()
    assert sum(r[x] for x in ["pre_start_capture", "late_capture", "missed_observed", "start_unavailable", "start_unknown"]) == len(e)
    return r


def run():
    self_tests()
    config = json.loads((HERE / "tables" / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
    for rel, h in config["input_sha256"].items():
        assert sha(ROOT / rel) == h, rel
    reproduce_m004()
    coverage = read("predictions/09.29_006_target_coverage.csv", parse_dates=["record_key"])
    coverage["phase"] = np.where(coverage.month.le(6), "development", "review")
    coverage["u"] = coverage.stage.map(config["u_by_stage"])
    assert len(coverage) == 2904 and coverage.record_key.nunique() == 2904
    assert coverage.eligible.sum() == 2808 and (~coverage.eligible).sum() == 96
    catalogue = read("predictions/09.29_005_event_audit.csv", parse_dates=["start", "end"])
    catalogue = catalogue.loc[catalogue.method.eq("hgb_calendar")].copy()
    members = read("predictions/09.29_005_event_members.csv", parse_dates=["record_key"])
    assert len(catalogue) == 131 and not members.duplicated(["stage", "record_key"]).any()
    alerts, bundle_list, event_list, hourly, daily, peak_rows, monthly_peaks, emetrics = [], [], [], [], [], [], [], []
    predictions = {"development": load_predictions("development"), "review": load_predictions("review")}
    for policy in config["policies"]:
        m, budget, c = policy["method"], policy["budget"], policy["c"]
        for phase in ["development", "review"]:
            cov = coverage.loc[coverage.phase.eq(phase)].copy()
            pred = predictions[phase].loc[predictions[phase].method.eq(m),
                ["record_key", "actual", "prediction", "source_data_row", "forecast_after_completed_record", "training_end"]]
            cov = cov.merge(pred, on="record_key", how="left", validate="one_to_one", suffixes=("", "_saved"))
            assert cov.prediction.notna().equals(cov.eligible)
            np.testing.assert_array_equal(cov.loc[cov.eligible, "actual"], cov.loc[cov.eligible, "peak"])
            np.testing.assert_array_equal(cov.loc[cov.eligible, "source_data_row_saved"], cov.loc[cov.eligible, "source_data_row"])
            cov["method"], cov["budget"], cov["c"] = m, budget, c
            cov["target"] = "peak"
            cov["issue_after_completed_record"] = cov.record_key - HOUR
            cov["issue_status"] = np.where(cov.eligible, "issued_assuming_previous_complete", "not_issued")
            f = mark_alerts(cov)
            b = bundles(f)
            assert f.new_notification.sum() == len(b)
            assert int(f.warning.sum()) == int(b.raw_warning_hours.sum())
            alerts.append(f)
            bundle_list.append(b)
            cat = catalogue.loc[catalogue.stage.isin(["May", "Jun"] if phase == "development" else ["Jul-Aug"])]
            es = pd.DataFrame(event_rows(f, cat, members))
            event_list.append(es)
            for period, fg in [("pooled", f), *[(str(month), g) for month, g in f.groupby("month")]]:
                months = fg.month.unique()
                bg = b if period == "pooled" else b.loc[b.start_month.isin(months)]
                hourly.append(hour_summary(fg, bg, period))
                eg = es if period == "pooled" else es.loc[es.month.isin(months)]
                emetrics.append(summarize_events(eg, period))
            for date, g in f.groupby("date"):
                warn = int(g.warning.sum())
                bb = b.loc[b.start_date.eq(date)]
                daily.append({"phase": phase, "method": m, "budget": budget, "date": date,
                    "normal_hours": len(g), "available_hours": int(g.available.sum()),
                    "unavailable_hours": int((~g.available).sum()), "raw_warning_hours": warn,
                    "new_notifications": int(g.new_notification.sum()), "active_record_hours": warn,
                    "false_bundles_observed": int(bb.false_bundle_observed.sum()),
                    "prediction_day_status": "none" if not g.available.any() else "complete" if g.available.all() and len(g) == 24 else "partial"})
                pk = g.loc[g.peak.eq(g.peak.max())]
                flags = [bool(x) if pd.notna(x) else None for x in pk.warning]
                any_cap, all_cap = peak_capture(flags)
                available_pk = pk.loc[pk.available]
                peak_rows.append({"phase": phase, "method": m, "budget": budget, "date": date,
                    "month": int(g.month.iloc[0]), "u": g.u.iloc[0], "day_peak": g.peak.max(),
                    "peak_ge_u": bool(g.peak.max() >= g.u.iloc[0]), "original_peak_hours": len(pk),
                    "available_peak_hours": len(available_pk), "tied": len(pk) > 1,
                    "any_peak_capture": any_cap, "all_peak_capture": all_cap,
                    "date_weighted_shortfall": (available_pk.peak - available_pk.prediction).clip(lower=0).mean()})
            for month, g in f.groupby("month"):
                for _, row in g.loc[g.peak.eq(g.peak.max())].iterrows():
                    monthly_peaks.append({"phase": phase, "method": m, "budget": budget, "month": month,
                        "record_key": row.record_key, "u": row.u, "c": c, "actual": row.peak,
                        "prediction": row.prediction, "available": row.available, "warning": row.warning,
                        "new_notification": row.new_notification, "active_before_update": row.active_before_update,
                        "input_missing_reason": row.input_missing_reason})
    af, bf, ef = pd.concat(alerts, ignore_index=True), pd.concat(bundle_list, ignore_index=True), pd.concat(event_list, ignore_index=True)
    hs, es, peaks = pd.DataFrame(hourly), pd.DataFrame(emetrics), pd.DataFrame(peak_rows)
    save(af, "alerts", rows=True)
    save(bf, "bundles", rows=True)
    save(ef, "events", rows=True)
    save(hs, "hour_metrics")
    save(es, "event_metrics")
    save(pd.DataFrame(daily), "daily_burden")
    save(peaks, "peak_capture")
    save(pd.DataFrame(monthly_peaks), "monthly_peak_cases")
    psummary = []
    for (phase, m, budget), pg in peaks.groupby(["phase", "method", "budget"]):
        for period, g in [("pooled", pg), *[(str(month), sub) for month, sub in pg.groupby("month")]]:
            for category, subset in [("all", g), ("high", g.loc[g.peak_ge_u]), ("below_u", g.loc[~g.peak_ge_u]), ("tied", g.loc[g.tied])]:
                psummary.append({"phase": phase, "period": period, "method": m, "budget": budget,
                    "category": category, "dates": len(subset),
                    "any_captured_dates": int(subset.any_peak_capture.eq("captured").sum()),
                    "all_captured_dates": int(subset.all_peak_capture.eq("captured").sum()),
                    "fully_evaluable_dates": int(subset.available_peak_hours.eq(subset.original_peak_hours).sum()),
                    "unavailable_dates": int(subset.available_peak_hours.eq(0).sum()),
                    "partial_dates": int((subset.available_peak_hours.gt(0) & subset.available_peak_hours.lt(subset.original_peak_hours)).sum()),
                    "date_weighted_shortfall": subset.date_weighted_shortfall.mean()})
    save(pd.DataFrame(psummary), "peak_summary")
    # Fixed May-only c -> June is a separate stability check, never final policy selection.
    internal = []
    dev = predictions["development"]
    for p in config["internal_policies"]:
        g = dev.loc[dev.method.eq(p["method"]) & dev.record_key.dt.month.eq(6)]
        warn = g.prediction.ge(p["c"])
        internal.append({**p, "june_hours": len(g), "june_warning_hours": int(warn.sum()),
                         "june_raw_hours_per_day": warn.sum() / 30,
                         **confusion(g.actual.ge(181), warn)})
    save(pd.DataFrame(internal), "internal_stability")
    decisions = []
    for budget in BUDGETS:
        base_h = hs.loc[hs.phase.eq("review") & hs.budget.eq(budget) & hs.method.eq("hgb_calendar")].set_index("period")
        base_e = es.loc[es.phase.eq("review") & es.budget.eq(budget) & es.method.eq("hgb_calendar")].set_index("period")
        for m in METHODS:
            h = hs.loc[hs.phase.eq("review") & hs.budget.eq(budget) & hs.method.eq(m)].set_index("period")
            e = es.loc[es.phase.eq("review") & es.budget.eq(budget) & es.method.eq(m)].set_index("period")
            improved = e.loc["pooled", "pre_start_capture"] > base_e.loc["pooled", "pre_start_capture"]
            stable_gain = all(e.loc[x, "pre_start_capture"] >= base_e.loc[x, "pre_start_capture"] for x in ["7", "8"])
            burden_nonworse = all(h.loc[x, col] <= base_h.loc[x, col] for x in ["7", "8"]
                for col in ["raw_warning_hours", "new_notifications", "false_bundles_observed"])
            decision = "reference_keep" if m == "hgb_calendar" else (
                "adopt_peak_review_candidate" if improved and stable_gain and burden_nonworse else
                "tradeoff_peak_review_candidate" if improved and stable_gain else
                "case_limited" if improved else "reject_as_start_warning_replacement")
            decisions.append({"method": m, "budget": budget, "decision": decision,
                "delta_pre_start_capture": int(e.loc["pooled", "pre_start_capture"] - base_e.loc["pooled", "pre_start_capture"]),
                "delta_raw_warning_hours": int(h.loc["pooled", "raw_warning_hours"] - base_h.loc["pooled", "raw_warning_hours"]),
                "delta_new_notifications": int(h.loc["pooled", "new_notifications"] - base_h.loc["pooled", "new_notifications"]),
                "delta_false_bundles": int(h.loc["pooled", "false_bundles_observed"] - base_h.loc["pooled", "false_bundles_observed"]),
                "gain_nonworse_both_months": stable_gain, "burden_nonworse_both_months": burden_nonworse})
    save(pd.DataFrame(decisions), "policy_selection")
    # Same-event leave-one-date/week contrast for the primary review policy.
    ev = ef.loc[ef.phase.eq("review") & ef.budget.eq(2) & ef.start_evaluable].copy()
    ev["start_date"] = pd.to_datetime(ev.start).dt.strftime("%Y-%m-%d")
    ev["start_week"] = pd.to_datetime(ev.start).dt.to_period("W").astype(str)
    sens = []
    for m in METHODS[1:]:
        base = ev.loc[ev.method.eq("hgb_calendar")].set_index("event_id")
        candidate = ev.loc[ev.method.eq(m)].set_index("event_id").reindex(base.index)
        delta = candidate.outcome.eq("pre_start_capture").astype(int) - base.outcome.eq("pre_start_capture").astype(int)
        for unit in ["start_date", "start_week"]:
            for value in base[unit].unique():
                mask = base[unit].ne(value)
                sens.append({"method": m, "budget": 2, "excluded_unit": unit, "excluded": value,
                             "remaining_start_events": int(mask.sum()), "delta_pre_start_capture": int(delta.loc[mask].sum())})
    save(pd.DataFrame(sens), "event_sensitivity")
    protected_n = 0
    for rel, h in config["protected_sha256"].items():
        assert sha(ROOT / rel) == h, rel
        protected_n += 1
    write_json({"time": now(), "python": platform.python_version(), "pandas": pd.__version__,
        "numpy": np.__version__, "implementation_sha256": sha(Path(__file__)),
        "frozen_sha256": sha(HERE / "tables" / f"{PREFIX}_frozen.json"),
        "saved_input_files": len(INPUTS), "protected_files_unchanged": protected_n,
        "unique_normal_hours": len(coverage), "unique_available_hours": 2808,
        "alert_rows": len(af), "event_policy_rows": len(ef), "bundle_rows": len(bf),
        "new_model_training": 0, "review_threshold_adjustments": 0,
        "exit_code_on_success": 0}, "facts")
    print(hs.loc[hs.phase.eq("review") & hs.period.eq("pooled") & hs.budget.eq(2),
                 ["method", "c", "raw_warning_hours", "new_notifications", "false_bundles_observed", "precision", "recall"]].to_string(index=False))
    print(es.loc[es.phase.eq("review") & es.period.eq("pooled") & es.budget.eq(2),
                 ["method", "pre_start_capture", "late_capture", "missed_observed", "start_unavailable", "pre_start_rate_evaluable"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["test", "freeze", "run"], required=True)
    args = parser.parse_args()
    assert platform.python_implementation() == "CPython" and platform.python_version_tuple()[:2] == ("3", "13")
    if args.phase == "test":
        print(json.dumps({"tests": self_tests()}, ensure_ascii=False))
    elif args.phase == "freeze":
        freeze()
    else:
        run()
