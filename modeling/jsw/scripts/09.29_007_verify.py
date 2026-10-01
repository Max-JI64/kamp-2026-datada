"""Independent stdlib source/CSV verification of M007; imports no evaluation code."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
PREFIX = "09.29_007"
HOUR = timedelta(hours=1)
CHECKS = Counter()


def read(rel):
    with (HERE / rel).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def time(x):
    return datetime.fromisoformat(x)


def flag(x):
    if x == "":
        return None
    assert x in ("True", "False"), x
    return x == "True"


def number(x):
    return float(x) if str(x) != "" else math.nan


def equal(saved, expected, label):
    if isinstance(expected, bool):
        assert flag(saved) is expected, (label, saved, expected)
    elif isinstance(expected, (int, float)):
        actual = number(saved)
        assert (math.isnan(actual) and math.isnan(expected)) or math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-12), (label, saved, expected)
    else:
        assert saved == str(expected), (label, saved, expected)
    CHECKS[label.split(":")[0]] += 1


def divide(a, b):
    return a / b if b else math.nan


def confusion(items):
    tp = sum(r["high"] and r["warning"] for r in items)
    fp = sum(not r["high"] and r["warning"] for r in items)
    fn = sum(r["high"] and not r["warning"] for r in items)
    tn = sum(not r["high"] and not r["warning"] for r in items)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": divide(tp, tp + fp),
            "recall": divide(tp, tp + fn), "f1": divide(2 * tp, 2 * tp + fp + fn)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert platform.python_version_tuple()[:2] == ("3", "13") and platform.python_implementation() == "CPython"
    cfg = json.loads((HERE / "tables" / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
    for rel, h in cfg["input_sha256"].items():
        assert sha(ROOT / rel) == h, rel
    # Verify normal keys, values, coverage directly from the original CSV.
    source = {}
    with (ROOT / "data/origin/okm_augumented_2021.csv").open(encoding="utf-8-sig", newline="") as f:
        for rownum, row in enumerate(csv.DictReader(f), 1):
            hour = int(row["시간"])
            if not 0 <= hour <= 23:
                continue
            t = datetime.strptime(row["날짜"], "%Y%m%d") + timedelta(hours=hour)
            assert t not in source
            source[t] = {"peak": max(float(row[x]) for x in ("15분", "30분", "45분", "60분")), "source_data_row": rownum}
    coverage = {time(r["record_key"]): r for r in read("predictions/09.29_006_target_coverage.csv")}
    expected_keys = {t for t in source if 5 <= t.month <= 8}
    assert set(coverage) == expected_keys
    for t, r in coverage.items():
        equal(r["peak"], source[t]["peak"], "source_peak")
        equal(r["source_data_row"], source[t]["source_data_row"], "source_row")
        eligible = all(t - HOUR * lag in source for lag in (*range(1, 25), 168))
        equal(r["eligible"], eligible, "source_coverage")
    pred = {}
    for name in ("development", "review"):
        for r in read(f"predictions/09.29_006_{name}_predictions.csv"):
            if r["target"] != "peak" or r["method"] not in cfg["methods"]:
                continue
            key = (r["method"], time(r["record_key"]))
            assert key not in pred
            pred[key] = float(r["prediction"])
            assert time(r["forecast_after_completed_record"]) == key[1] - HOUR
            equal(r["actual"], source[key[1]]["peak"], "prediction_actual")
    for m in cfg["methods"]:
        assert {t for method, t in pred if method == m} == {t for t, r in coverage.items() if flag(r["eligible"])}
    # Exhaustively enumerate all score boundaries independently, including no-warning.
    for pol in cfg["policies"]:
        scores = [v for (m, t), v in pred.items() if m == pol["method"] and t.month in (5, 6)]
        limits = sorted(set(scores)) + [math.nextafter(max(scores), math.inf)]
        candidates = [(sum(v >= c for v in scores), c) for c in limits]
        feasible = [x for x in candidates if x[0] <= 61 * pol["budget"]]
        expected_n, expected_c = max(feasible)
        assert pol["development_warning_hours"] == expected_n
        assert math.isclose(pol["c"], expected_c, rel_tol=0, abs_tol=1e-11)
        assert sum(v >= pol["c"] for v in scores) == expected_n, ("threshold_roundtrip", pol)
        CHECKS["threshold_policies"] += 1
    # Reconstruct the M005 event catalogue independently on all normal targets.
    observed_events = {}
    bounds = {"May": (datetime(2021, 5, 1), datetime(2021, 6, 1)),
              "Jun": (datetime(2021, 6, 1), datetime(2021, 7, 1)),
              "Jul-Aug": (datetime(2021, 7, 1), datetime(2021, 9, 1))}
    for stage, (start, end) in bounds.items():
        u = cfg["u_by_stage"][stage]
        high = sorted(t for t in coverage if start <= t < end and source[t]["peak"] >= u)
        batches = []
        for t in high:
            if batches and t - batches[-1][-1] == HOUR:
                batches[-1].append(t)
            else:
                batches.append([t])
        for i, keys in enumerate(batches, 1):
            s, e = keys[0], keys[-1]
            eid = f"{stage}|ge{u:g}|{i:03d}"
            left = s - HOUR not in source or (s == start and source[s - HOUR]["peak"] >= u)
            right = e + HOUR not in source or (e + HOUR >= end and source[e + HOUR]["peak"] >= u)
            observed_events[eid] = {"keys": keys, "left": left, "right": right, "stage": stage}
    assert len(observed_events) == 131
    member_keys = defaultdict(list)
    for r in read("predictions/09.29_005_event_members.csv"):
        member_keys[r["event_id"]].append(time(r["record_key"]))
    assert set(member_keys) == set(observed_events)
    for eid, e in observed_events.items():
        assert sorted(member_keys[eid]) == e["keys"]
    # Reconstruct flags and exact-hour bundles for each policy, including unavailable rows.
    flags_by_policy = {}
    episodes_by_policy = {}
    alert_lookup = {}
    for r in read(f"predictions/{PREFIX}_alerts.csv"):
        key = (r["phase"], r["method"], int(r["budget"]), time(r["record_key"]))
        assert key not in alert_lookup
        alert_lookup[key] = r
    for pol in cfg["policies"]:
        m, budget, c = pol["method"], pol["budget"], pol["c"]
        for phase in ("development", "review"):
            keys = sorted(t for t in coverage if (t.month <= 6) == (phase == "development"))
            items, episodes, current = [], [], []
            previous = None
            for t in keys:
                score = pred.get((m, t))
                available = score is not None
                warning = score >= c if available else None
                prior_active = bool(previous and previous["warning"] and t - previous["t"] == HOUR)
                new = bool(warning and not prior_active)
                u = cfg["u_by_stage"][coverage[t]["stage"]]
                item = {"t": t, "score": score, "available": available, "warning": warning,
                        "high": source[t]["peak"] >= u, "prior_active": prior_active, "new": new}
                saved = alert_lookup[(phase, m, budget, t)]
                for col, expected in [("available", available), ("warning", warning),
                                      ("actual_high", item["high"]), ("active_before_update", prior_active),
                                      ("new_notification", new), ("c", c), ("u", u)]:
                    if expected is None:
                        assert saved[col] == "", (col, saved[col])
                    else:
                        equal(saved[col], expected, "alerts:" + col)
                if not available:
                    assert saved["prediction"] == "" and saved["active_record_hours"] == ""
                    assert saved["issue_status"] == "not_issued" and saved["input_missing_reason"] != "none"
                else:
                    equal(saved["prediction"], score, "alerts:prediction")
                    equal(saved["active_record_hours"], int(warning), "alerts:active")
                if warning:
                    if new:
                        if current:
                            episodes.append(current)
                        current = []
                    current.append(item)
                elif current:
                    episodes.append(current)
                    current = []
                items.append(item)
                previous = item
            if current:
                episodes.append(current)
            flags_by_policy[(phase, m, budget)] = {r["t"]: r for r in items}
            episodes_by_policy[(phase, m, budget)] = episodes
            CHECKS["flag_policy_groups"] += 1
    assert len(alert_lookup) == 2904 * 12
    bundle_lookup = defaultdict(list)
    for r in read(f"predictions/{PREFIX}_bundles.csv"):
        bundle_lookup[(r["phase"], r["method"], int(r["budget"]))].append(r)
    for key, groups in episodes_by_policy.items():
        saved = sorted(bundle_lookup[key], key=lambda r: r["start"])
        assert len(saved) == len(groups)
        items = flags_by_policy[key]
        for row, g in zip(saved, groups):
            s, e = g[0]["t"], g[-1]["t"]
            nxt = items.get(e + HOUR)
            reason = ("non_warning" if nxt["available"] else "prediction_unavailable") if nxt else (
                "evaluation_boundary" if e == max(items) else "record_gap")
            false = not any(x["high"] for x in g)
            equal(row["start"], str(s), "bundles:start")
            equal(row["end"], str(e), "bundles:end")
            equal(row["raw_warning_hours"], len(g), "bundles:hours")
            equal(row["high_target_hours"], sum(x["high"] for x in g), "bundles:high")
            equal(row["false_bundle_observed"], false, "bundles:false")
            equal(row["false_bundle_complete"], false and reason == "non_warning", "bundles:false_complete")
            equal(row["false_bundle_censored"], false and reason != "non_warning", "bundles:false_censored")
            equal(row["end_reason"], reason, "bundles:reason")
    evlookup = {}
    for r in read(f"predictions/{PREFIX}_events.csv"):
        key = (r["phase"], r["method"], int(r["budget"]))
        e = observed_events[r["event_id"]]
        f = flags_by_policy[key]
        keys = e["keys"]
        items = [f[t] for t in keys]
        first = next((x["t"] for x in items if x["warning"]), None)
        start_available = items[0]["available"]
        outcome = "start_unknown" if e["left"] else "start_unavailable" if not start_available else (
            "pre_start_capture" if items[0]["warning"] else "late_capture" if first else "missed_observed")
        available = sum(x["available"] for x in items)
        for col, expected in [("outcome", outcome), ("left_unknown", e["left"]), ("right_unknown", e["right"]),
            ("start_available", start_available), ("start_evaluable", start_available and not e["left"]),
            ("predicted_hours", available), ("missing_prediction_hours", len(keys) - available),
            ("any_observed_capture", first is not None),
            ("complete_miss_confirmed", outcome == "missed_observed" and available == len(keys)),
            ("partial_miss_unresolved", outcome == "missed_observed" and available < len(keys)),
            ("start_new_notification", items[0]["new"]), ("active_before_start_update", items[0]["prior_active"]),
            ("start_continued_warning", bool(items[0]["warning"] and not items[0]["new"]))]:
            equal(r[col], expected, "events:" + col)
        equal(r["delay_target_records"], (first - keys[0]).total_seconds() / 3600 if first else math.nan, "events:delay")
        if first:
            equal(r["first_warning_target"], str(first), "events:first")
            equal(r["issue_after_completed_record"], str(first - HOUR), "events:issue")
            if outcome == "pre_start_capture":
                assert first == keys[0] and time(r["issue_after_completed_record"]) < first
        else:
            assert r["first_warning_target"] == r["issue_after_completed_record"] == ""
        fullkey = (*key, r["event_id"])
        assert fullkey not in evlookup
        evlookup[fullkey] = r
    assert len(evlookup) == 131 * 12
    # All hour metrics, daily burdens and event summaries from independently reconstructed records.
    for r in read(f"tables/{PREFIX}_hour_metrics.csv"):
        key = (r["phase"], r["method"], int(r["budget"]))
        items = list(flags_by_policy[key].values())
        if r["period"] != "pooled":
            items = [x for x in items if x["t"].month == int(r["period"])]
        a = [x for x in items if x["available"]]
        dates = {x["t"].date() for x in items}
        common = {x["t"].date() for x in a}
        groups = episodes_by_policy[key]
        if r["period"] != "pooled":
            groups = [g for g in groups if g[0]["t"].month == int(r["period"])]
        raw, notifications = sum(bool(x["warning"]) for x in a), sum(x["new"] for x in items)
        expected = {"normal_hours": len(items), "available_hours": len(a), "unavailable_hours": len(items) - len(a),
            "normal_dates": len(dates), "common_prediction_dates": len(common),
            "no_prediction_dates": len(dates - common), "high_unavailable_hours": sum(x["high"] for x in items if not x["available"]),
            "raw_warning_hours": raw, "active_record_hours": raw, "raw_hours_per_common_day": divide(raw, len(common)),
            "raw_hours_per_normal_day": divide(raw, len(dates)), "active_fraction_available": divide(raw, len(a)),
            "new_notifications": notifications, "new_notifications_per_common_day": divide(notifications, len(common)),
            "false_bundles_observed": sum(not any(x["high"] for x in g) for g in groups),
            "bundles_started": len(groups), "bundle_mean_hours": mean([len(g) for g in groups]) if groups else math.nan,
            "bundle_median_hours": median([len(g) for g in groups]) if groups else math.nan,
            "bundle_max_hours": max([len(g) for g in groups]) if groups else math.nan,
            **confusion(a)}
        for col, val in expected.items():
            equal(r[col], val, "hour_metrics:" + col)
    for r in read(f"tables/{PREFIX}_daily_burden.csv"):
        key = (r["phase"], r["method"], int(r["budget"]))
        items = [x for x in flags_by_policy[key].values() if str(x["t"].date()) == r["date"]]
        av = sum(x["available"] for x in items)
        raw = sum(bool(x["warning"]) for x in items)
        groups = [g for g in episodes_by_policy[key] if str(g[0]["t"].date()) == r["date"]]
        for col, val in {"normal_hours": len(items), "available_hours": av, "unavailable_hours": len(items) - av,
            "raw_warning_hours": raw, "active_record_hours": raw, "new_notifications": sum(x["new"] for x in items),
            "false_bundles_observed": sum(not any(x["high"] for x in g) for g in groups),
            "prediction_day_status": "none" if not av else "complete" if av == 24 else "partial"}.items():
            equal(r[col], val, "daily:" + col)
    for r in read(f"tables/{PREFIX}_event_metrics.csv"):
        key = (r["phase"], r["method"], int(r["budget"]))
        events = [x for k, x in evlookup.items() if k[:3] == key and (r["period"] == "pooled" or x["month"] == r["period"])]
        counts = Counter(x["outcome"] for x in events)
        den = sum(flag(x["start_evaluable"]) for x in events)
        identifiable = sum(not flag(x["left_unknown"]) for x in events)
        expected = {"events": len(events), "start_identifiable_events": identifiable, "start_evaluable_events": den,
                    "start_available_rate": divide(den, identifiable),
                    "pre_start_rate_evaluable": divide(counts["pre_start_capture"], den),
                    "pre_start_rate_service": divide(counts["pre_start_capture"], identifiable)}
        for outcome in ("pre_start_capture", "late_capture", "missed_observed", "start_unavailable", "start_unknown"):
            expected[outcome] = counts[outcome]
        for col, val in expected.items():
            equal(r[col], val, "event_metrics:" + col)
    # Daily maxima are determined on ALL source rows, not only model-available rows.
    peak_lookup = {}
    for r in read(f"tables/{PREFIX}_peak_capture.csv"):
        key = (r["phase"], r["method"], int(r["budget"]))
        items = [x for x in flags_by_policy[key].values() if str(x["t"].date()) == r["date"]]
        maximum = max(source[x["t"]]["peak"] for x in items)
        pk = [x for x in items if source[x["t"]]["peak"] == maximum]
        avail = [x for x in pk if x["available"]]
        any_cap = "unavailable" if not avail else "captured" if any(x["warning"] for x in avail) else (
            "partial_unknown" if len(avail) < len(pk) else "not_captured")
        all_cap = "unavailable" if len(avail) < len(pk) else "captured" if all(x["warning"] for x in avail) else "not_captured"
        for col, val in {"day_peak": maximum, "original_peak_hours": len(pk), "available_peak_hours": len(avail),
            "tied": len(pk) > 1, "any_peak_capture": any_cap, "all_peak_capture": all_cap,
            "date_weighted_shortfall": mean(max(source[x["t"]]["peak"] - x["score"], 0) for x in avail) if avail else math.nan}.items():
            equal(r[col], val, "peaks:" + col)
        peak_lookup[(*key, r["date"])] = r
    for r in read(f"tables/{PREFIX}_peak_summary.csv"):
        key = (r["phase"], r["method"], int(r["budget"]))
        rows = [x for k, x in peak_lookup.items() if k[:3] == key and (r["period"] == "pooled" or x["month"] == r["period"])]
        rows = [x for x in rows if r["category"] == "all" or (r["category"] == "high" and flag(x["peak_ge_u"]))
            or (r["category"] == "below_u" and not flag(x["peak_ge_u"])) or (r["category"] == "tied" and flag(x["tied"]))]
        for col, val in {"dates": len(rows), "any_captured_dates": sum(x["any_peak_capture"] == "captured" for x in rows),
            "all_captured_dates": sum(x["all_peak_capture"] == "captured" for x in rows),
            "fully_evaluable_dates": sum(x["original_peak_hours"] == x["available_peak_hours"] for x in rows),
            "unavailable_dates": sum(x["available_peak_hours"] == "0" for x in rows)}.items():
            equal(r[col], val, "peak_summary:" + col)
    for r in read(f"tables/{PREFIX}_monthly_peak_cases.csv"):
        t, month = time(r["record_key"]), int(r["month"])
        maximum = max(v["peak"] for tt, v in source.items() if tt.month == month)
        assert source[t]["peak"] == maximum
        f = flags_by_policy[(r["phase"], r["method"], int(r["budget"]))][t]
        equal(r["available"], f["available"], "monthly_peaks:available")
        equal(r["warning"], f["warning"], "monthly_peaks:warning")
    # Reproduce all M004 confusion and state rows from original stored forecasts.
    q = read("predictions/09.26_002_development_predictions.csv")
    p = [x for x in read("predictions/09.26_003_development_predictions.csv") if x["target"] == "peak" and x["method"] != "hgb_calendar"]
    combined = q + p
    for r in read(f"tables/{PREFIX}_m004_selection_reproduced.csv"):
        g = [x for x in combined if x["method"] == r["method"] and (r["period"] == "pooled" or x["stage"] == r["period"])]
        items = [{"high": float(x["actual"]) >= float(x["high_threshold_from_training"]),
                  "warning": float(x["prediction"]) >= float(x["high_threshold_from_training"])} for x in g]
        c = confusion(items)
        for col, val in {"n": len(items), "high_actual_n": c["tp"] + c["fn"], "not_high_actual_n": c["fp"] + c["tn"],
            "selected_n": c["tp"] + c["fp"], "hit_n": c["tp"], "miss_n": c["fn"],
            "false_selection_n": c["fp"], "recall": c["recall"], "precision": c["precision"]}.items():
            equal(r[col], val, "M004:" + col)
    for r in read(f"tables/{PREFIX}_m004_states_reproduced.csv"):
        g = [x for x in combined if x["method"] == r["method"] and x["peak_state"] == r["peak_state"]
             and (r["period"] == "pooled" or x["stage"] == r["period"])]
        hit = sum(float(x["prediction"]) >= float(x["high_threshold_from_training"]) for x in g)
        equal(r["actual_n"], len(g), "M004_states:actual")
        equal(r["hit_n"], hit, "M004_states:hit")
        equal(r["miss_n"], len(g) - hit, "M004_states:miss")
    for rel, h in cfg["protected_sha256"].items():
        assert sha(ROOT / rel) == h, rel
    facts = json.loads((HERE / "tables" / f"{PREFIX}_facts.json").read_text(encoding="utf-8"))
    assert facts["implementation_sha256"] == sha(HERE / "scripts" / f"{PREFIX}_alert_evaluation.py")
    assert facts["frozen_sha256"] == sha(HERE / "tables" / f"{PREFIX}_frozen.json")
    links = 0
    for rel in ["reports/09.29_007_alert_evaluation_plan.md", "README.md"]:
        path = HERE / rel
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in link or link.startswith("#"):
                continue
            target = link.split("#")[0]
            assert (path.parent / target).resolve().exists(), (rel, link)
            links += 1
    result = {"time": datetime.now().astimezone().isoformat(timespec="seconds"), "python": platform.python_version(),
        "method": "stdlib csv; independent original-source eligibility, event and score-boundary loops; no M007 implementation imports",
        "exit_code_on_success": 0, "checks": dict(CHECKS), "normal_hours": len(coverage), "alert_rows": len(alert_lookup),
        "event_policy_rows": len(evlookup), "protected_files_unchanged": len(cfg["protected_sha256"]),
        "local_links_checked": links, "verifier_sha256": sha(Path(__file__))}
    (HERE / "tables" / f"{PREFIX}_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
