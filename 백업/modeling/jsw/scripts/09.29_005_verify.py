"""Independent source/csv-loop verification of M005. Does not import audit code."""
from collections import defaultdict
import csv
from datetime import datetime, timedelta
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import re
import sys

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent.parent
PREFIX = "09.29_005"
H = timedelta(hours=1)


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def true(value):
    return value == "True"


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-9), (a, b)


def main():
    assert sys.version_info[:2] == (3, 13) and not bool(sys._is_gil_enabled() is False)
    frozen = json.loads((HERE / "tables" / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
    for name, digest in frozen["input_sha256"].items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    src = rows(ROOT / "data/origin/okm_augumented_2021.csv")
    normal, day_vectors = {}, defaultdict(list)
    for number, r in enumerate(src, 1):
        date, hour = int(r["날짜"]), int(r["시간"])
        if date >= 20210901 or not 0 <= hour <= 23:
            continue
        t = datetime.strptime(str(date), "%Y%m%d") + hour * H
        values = tuple(float(r[s]) for s in ["15분", "30분", "45분", "60분"])
        assert t not in normal
        normal[t] = {"peak": max(values), "mean": float(r["평균"]), "source_data_row": number,
                     "date": t.strftime("%Y-%m-%d"), "values": values}
        day_vectors[t.strftime("%Y-%m-%d")].append((hour, values))
    eligible = {t for t in normal if all(t - i * H in normal for i in range(1, 25)) and t - 168 * H in normal}
    assert len(normal) == 5784 and len(eligible) == 5520
    saved = {}
    for name in ["09.26_001_development_predictions.csv", "09.26_001_review_predictions.csv", "09.26_002_development_predictions.csv"]:
        for r in rows(HERE / "predictions" / name):
            t = datetime.fromisoformat(r["record_key"])
            key = (r["stage"], r["target"], r["method"], t)
            close(r["actual"], normal[t][r["target"]])
            assert int(r["source_data_row"]) == normal[t]["source_data_row"] and t in eligible
            if key in saved:
                close(r["prediction"], saved[key]["prediction"])
            saved[key] = r
    audit = rows(HERE / "predictions" / f"{PREFIX}_row_audit.csv")
    assert len({(r["stage"], r["target"], r["method"], r["record_key"]) for r in audit}) == len(audit)
    peak_positions = {t for t in normal if normal[t]["peak"] == max(x[1][i] for x in day_vectors[normal[t]["date"]] for i in range(4))}
    profile_map = {datetime.strptime(r["date"], "%Y%m%d").strftime("%Y-%m-%d"): r["profile_id"]
                   for r in rows(ROOT / "EDA/jsw/tables/09.23_011_daily_profile_summary.csv")}
    vectors = {d: tuple(v for _, values in sorted(hs) for v in values) for d, hs in day_vectors.items() if len(hs) == 24}
    vector_for_id = {}
    for date, vector in vectors.items():
        pid = profile_map[date]
        assert pid not in vector_for_id or vector_for_id[pid] == vector
        vector_for_id[pid] = vector
    stage_status = {}
    thresholds = json.loads((HERE / "tables" / f"{PREFIX}_facts.json").read_text(encoding="utf-8"))["saved_thresholds"]
    targets = rows(HERE / "predictions" / f"{PREFIX}_target_coverage.csv")
    target_map = {datetime.fromisoformat(r["record_key"]): r for r in targets}
    assert len(target_map) == len(targets) == 2904
    for stage, (start, end) in frozen["periods"].items():
        cutoff = datetime.fromisoformat(start)
        train_dates = {normal[t]["date"] for t in eligible if t < cutoff}
        contributed = {vectors[d] for d in train_dates if d in vectors and datetime.fromisoformat(d) + timedelta(days=1) <= cutoff}
        observed = {vector for d, vector in vectors.items() if datetime.fromisoformat(d) + timedelta(days=1) <= cutoff}
        stage_status[stage] = (contributed, observed)
    state_counts = defaultdict(int)
    for r in targets:
        t = datetime.fromisoformat(r["record_key"])
        assert true(r["eligible"]) == (t in eligible)
        assert true(r["is_daily_peak"]) == (t in peak_positions)
        threshold = thresholds[r["stage"]]
        previous = normal.get(t - H)
        state = "boundary_unknown" if previous is None else (
            "high_onset" if normal[t]["peak"] >= threshold and previous["peak"] < threshold else
            "high_continued" if normal[t]["peak"] >= threshold and previous["peak"] >= threshold else
            "descending" if previous["peak"] >= threshold else "ordinary")
        assert r["state"] == state
        contributed, observed = stage_status[r["stage"]]
        vector = vectors.get(r["date"])
        for field, library in [("profile_seen_contributed", contributed), ("profile_seen_observed", observed)]:
            assert r[field] == ("unknown" if vector is None else "seen" if vector in library else "unseen")
        state_counts[(r["stage"], t.month, state)] += int(t in eligible)
    agg = defaultdict(list)
    for r in audit:
        t = datetime.fromisoformat(r["record_key"])
        original = saved[(r["stage"], r["target"], r["method"], t)]
        close(r["actual"], original["actual"])
        close(r["prediction"], original["prediction"])
        assert r["state"] == target_map[t]["state"]
        agg[(r["stage"], str(t.month), r["target"], r["method"], r["state"])].append(float(r["prediction"]) - float(r["actual"]))
    for r in rows(HERE / "tables" / f"{PREFIX}_state_scores.csv"):
        if r["method"] not in ["lag1", "hgb_calendar", "hgb_q75_calendar"]:
            continue
        errors = agg[(r["stage"], r["month"], r["target"], r["method"], r["state"])]
        assert len(errors) == int(r["n"])
        close(r["mae"], sum(abs(e) for e in errors) / len(errors))
        close(r["rmse"], math.sqrt(sum(e * e for e in errors) / len(errors)))
        close(r["bias"], sum(errors) / len(errors))
        close(r["mean_shortfall"], sum(max(-e, 0) for e in errors) / len(errors))
        close(r["mean_over"], sum(max(e, 0) for e in errors) / len(errors))
    members = defaultdict(list)
    for r in rows(HERE / "predictions" / f"{PREFIX}_event_members.csv"):
        members[r["event_id"]].append(datetime.fromisoformat(r["record_key"]))
    all_member_keys = [t for ts in members.values() for t in ts]
    assert len(set(all_member_keys)) == len(all_member_keys)
    assert set(all_member_keys) == {t for t, r in target_map.items() if normal[t]["peak"] >= thresholds[r["stage"]]}
    a42 = {(r["start"], r["end"]): r["event_id"] for r in rows(ROOT / "analysis/jsw/tables/09.28_042_hour_event_comparison.csv")}
    events = rows(HERE / "predictions" / f"{PREFIX}_event_audit.csv")
    for r in events:
        ts = sorted(members[r["event_id"]])
        assert ts[0] == datetime.fromisoformat(r["start"]) and ts[-1] == datetime.fromisoformat(r["end"])
        assert len(ts) == int(r["hours_observed"]) and all(b - a == H for a, b in zip(ts, ts[1:]))
        threshold = float(r["threshold"])
        predictions = [saved.get((r["stage"], "peak", r["method"], t)) for t in ts]
        captured = [t for t, pred in zip(ts, predictions) if pred and float(pred["prediction"]) >= threshold]
        available = predictions[0] is not None
        assert available == true(r["start_input_available"])
        assert int(r["missing_prediction_hours"]) == sum(p is None for p in predictions)
        assert true(r["any_capture_observed"]) == bool(captured)
        expected = "start_unknown" if true(r["left_unknown"]) else "start_unpredictable" if not available else (
            "pre_start_warning" if float(predictions[0]["prediction"]) >= threshold else "late_capture" if captured else "missed")
        assert r["outcome"] == expected
        if not true(r["left_unknown"]):
            assert ts[0] - H in normal and normal[ts[0] - H]["peak"] < threshold
        if captured:
            assert datetime.fromisoformat(r["first_warning_target"]) == captured[0]
            assert datetime.fromisoformat(r["warning_after_completed_record"]) == captured[0] - H
            close(r["delay_hours_from_observed_start"], (captured[0] - ts[0]).total_seconds() / 3600)
        if r["A042_exact_event_id"]:
            assert threshold == 182 and int(float(r["A042_exact_event_id"])) == int(a42[(r["start"], r["end"])])
        elif threshold != 182:
            assert r["A042_link_status"] == "different_threshold_not_linked"
    for r in rows(HERE / "predictions" / f"{PREFIX}_daily_peak_dates.csv"):
        ts = [t for t in peak_positions if normal[t]["date"] == r["date"]]
        preds = [saved.get((r["stage"], "peak", r["method"], t)) for t in ts]
        present = [p for p in preds if p]
        assert len(ts) == int(r["original_peak_hours"]) and len(present) == int(r["available_peak_hours"])
        if present:
            close(r["day_mean_shortfall_available"], sum(max(float(p["actual"]) - float(p["prediction"]), 0) for p in present) / len(present))
        if len(present) != len(ts):
            assert r["all_original_peak_hours_alarm"] == ""
    for r in rows(HERE / "tables" / f"{PREFIX}_coverage.csv"):
        sub = [x for t, x in target_map.items() if x["stage"] == r["stage"] and t.month == int(r["month"])]
        assert int(r["normal_hours"]) == len(sub)
        assert int(r["common_saved_hours"]) == sum(true(x["eligible"]) for x in sub)
        assert int(r["excluded_peak_hours"]) == sum(true(x["is_daily_peak"]) and not true(x["eligible"]) for x in sub)
        assert int(r["excluded_high_hours"]) == sum(float(x["peak"]) >= float(r["threshold"]) and not true(x["eligible"]) for x in sub)
    for r in rows(HERE / "tables" / f"{PREFIX}_event_scores.csv"):
        sub = [x for x in events if x["stage"] == r["stage"] and x["month"] == r["month"] and x["method"] == r["method"]]
        assert int(r["events"]) == len(sub)
        for outcome in ["pre_start_warning", "late_capture", "missed", "start_unknown", "start_unpredictable"]:
            assert int(r[outcome]) == sum(x["outcome"] == outcome for x in sub)
    for r in rows(HERE / "tables" / f"{PREFIX}_monthly_peak_cases.csv"):
        t = datetime.fromisoformat(r["record_key"])
        assert normal[t]["peak"] == max(x["peak"] for s, x in normal.items() if s.month == t.month)
        close(r["actual"], normal[t][r["target"]])
        pred = saved.get((r["stage"], r["target"], r["method"], t))
        assert true(r["available"]) == (pred is not None)
        if pred:
            close(r["prediction"], pred["prediction"])
            close(r["shortfall"], max(float(pred["actual"]) - float(pred["prediction"]), 0))
    for r in rows(HERE / "tables" / f"{PREFIX}_profile_scores.csv"):
        if r["method"] not in ["lag1", "hgb_calendar", "hgb_q75_calendar"]:
            continue
        sub = [x for x in audit if all(x[k] == r[k] for k in ["stage", "month", "target", "method", "profile_seen_contributed"])]
        assert int(r["n"]) == len(sub)
        assert int(r["dates"]) == len(set(x["date"] for x in sub))
        assert int(r["profiles"]) == len(set(x["profile_id"] for x in sub))
        close(r["mae"], sum(abs(float(x["error"])) for x in sub) / len(sub))
        high = [x for x in sub if float(x["peak"]) >= float(x["high_threshold_from_training"])]
        assert int(r["high_n"]) == len(high)
        if high:
            close(r["high_mean_shortfall"], sum(float(x["shortfall"]) for x in high) / len(high))
    handoff = json.loads((HERE / "tables" / f"{PREFIX}_m006_handoff.json").read_text(encoding="utf-8"))
    primary = [r for r in audit if r["stage"] == "Jul-Aug" and r["target"] == "peak" and r["method"] == "hgb_calendar" and r["state"] == "high_continued"]
    evidence = handoff["evidence_Jul_Aug"]
    assert evidence["hours"] == len(primary) and evidence["dates"] == len(set(r["date"] for r in primary))
    assert evidence["events"] == len(set(r["event_id"] for r in primary))
    close(evidence["sum_shortfall_hgb"], sum(float(r["shortfall"]) for r in primary))
    close(evidence["sum_shortfall_lag1"], sum(max(float(r["actual"]) - float(saved[(r["stage"], "peak", "lag1", datetime.fromisoformat(r["record_key"]))]["prediction"]), 0) for r in primary))
    close(evidence["daily_peak_sum_shortfall_hgb"], sum(float(r["shortfall"]) for r in primary if true(r["is_daily_peak"])))
    links_checked = 0
    self_link = HERE / "tables" / f"{PREFIX}_verification.json"
    for path in [HERE / "README.md", HERE / "reports/09.29_005_peak_error_audit_plan.md", HERE / "reports/09.29_006_targeted_model_plan.md"]:
        text = path.read_text(encoding="utf-8")
        assert "\ufffd" not in text
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            target = target.split("#")[0].replace("%20", " ")
            resolved = (path.parent / target).resolve()
            assert resolved == self_link or resolved.exists(), (path, target)
            links_checked += 1
    result = {"time": datetime.now().astimezone().isoformat(timespec="seconds"), "python": platform.python_version(),
              "exit_code_on_success": 0, "checks": "stdlib csv and source loops independent of M005 implementation",
              "input_hashes_preserved": len(frozen["input_sha256"]), "normal_hours": len(normal),
              "eligible_hours": len(eligible), "audited_rows_checked": len(audit),
              "event_method_rows_checked": len(events), "distinct_events": len(members), "local_links_checked": links_checked,
              "validated": ["source actual and row keys", "input coverage", "state partitions and metrics", "original tied daily peaks",
                            "events and pre-start issue chronology", "unpredictable starts", "A042 definition matching",
                            "past-only observed/contributed profile libraries", "input hash preservation", "local document links"]}
    self_link.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    assert self_link.exists()
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
