"""Independent raw maximum, January quantile, errors and alarm-policy checks."""
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from m03_verify import rows, sha, close
from m04_rise_verify import ap

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Modeling/tables/m04_surge"


def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (position-lower) * (ordered[upper]-ordered[lower])


def alarm(y, scores, threshold):
    predicted = [value >= threshold for value in scores]
    tp = sum(a and b for a, b in zip(predicted, y))
    fp = sum(a and not b for a, b in zip(predicted, y))
    positives = sum(y)
    negatives = len(y)-positives
    return {"hours": len(y), "events": positives, "TP": tp, "FP": fp, "FN": positives-tp, "TN": negatives-fp,
        "recall": tp/positives if positives else 0., "precision": tp/(tp+fp) if tp+fp else 0.,
        "false_positive_fraction": fp/negatives if negatives else 0., "threshold": threshold}


def frontier(y, scores, cap):
    candidates = [math.nextafter(max(scores), math.inf)] + sorted(set(scores), reverse=True)
    acceptable = [alarm(y, scores, t) for t in candidates]
    acceptable = [r for r in acceptable if r["false_positive_fraction"] <= cap+1e-12]
    return min(acceptable, key=lambda r: (-r["TP"], r["FP"], -r["threshold"]))


def check_alarm(saved, pool, scores, boundary):
    labels = [float(r["increase"]) >= boundary for r in pool]
    expected = alarm(labels, scores, float(saved["threshold"]))
    for key, value in expected.items():
        close(saved[key], value)
    close(saved["AP"], ap(labels, scores))
    positive_scores = [s for y, s in zip(labels, scores) if y]
    negative_scores = [s for y, s in zip(labels, scores) if not y]
    auc = math.fsum(float(p>n)+.5*(p==n) for p in positive_scores for n in negative_scores)/(len(positive_scores)*len(negative_scores))
    close(saved["AUROC"], auc)
    close(saved["prevalence"], sum(labels)/len(labels))


def main():
    assert sys.version_info[:2] == (3,13) and sys._is_gil_enabled()
    run = json.loads((OUT / "run.json").read_text(encoding="utf-8"))
    cpath = ROOT / "Modeling/config/m04_surge_contract.json"
    c = json.loads(cpath.read_text(encoding="utf-8"))
    assert sha(cpath) == run["contract_sha256"] and sha(ROOT / "Modeling/scripts/m04_surge_diagnose.py") == run["script_sha256"]
    assert run["status"] == "completed" and run["no_fitting"] and run["no_july_august_evaluation"]
    for section in ["inputs_sha256", "outputs_sha256"]:
        for name, expected in run[section].items():
            assert sha(ROOT / name if section=="inputs_sha256" else OUT/name) == expected
    source = ROOT / "data/origin/okm_augumented_2021.csv"
    assert sha(source) == c["source_sha256"]
    raw = {}
    for r in rows(source):
        if not 20210101 <= int(r["날짜"]) <= 20210630 or not 0<=int(r["시간"])<=23:
            continue
        timestamp = datetime.strptime(r["날짜"], "%Y%m%d") + timedelta(hours=int(r["시간"]))
        assert timestamp not in raw
        slots = [float(r[name]) for name in ["15분","30분","45분","60분"]]
        raw[timestamp] = {"maximum":max(slots), "mean":math.fsum(slots)/4, "rounded":float(r["평균"])}
    eligible = {t for t in raw if all(t-timedelta(hours=lag) in raw for lag in [1,2,24,168])}
    frame = {}
    for t in eligible:
        before = raw[t-timedelta(hours=1)]
        frame[t] = {"timestamp":str(t), "date":t.date().isoformat(), "increase":raw[t]["maximum"]-before["maximum"],
            "maximum":raw[t]["maximum"], "previous_maximum":before["maximum"],
            "prior_low":before["maximum"]>0 and 20<=math.floor(before["mean"]+.5)<=26,
            "legacy_rise":before["maximum"]>0 and 20<=math.floor(before["mean"]+.5)<=26 and raw[t]["rounded"]>26}
    january = [r for t,r in frame.items() if t.month==1]
    positive = [r["increase"] for r in january if r["increase"]>0]
    boundaries = {"q95":quantile(positive,.95), "q90":quantile(positive,.90)}
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    close(summary["boundary_primary"], boundaries["q95"])
    close(summary["boundary_sensitivity"], boundaries["q90"])
    assert len(january)==summary["calibration_hours"]==576 and len(positive)==summary["calibration_positive_hours"]==233
    for r in rows(OUT/"event_counts.csv"):
        role = r["role"]
        selected = january if role=="calibration_jan" else [x for t,x in frame.items() if t<datetime(2021,4,1)] if role=="train_jan_mar" else [x for t,x in frame.items() if t.month=={"dev_apr":4,"dev_may":5,"dev_jun":6}[role]]
        boundary = boundaries[r["boundary_name"]]
        events = [x for x in selected if x["increase"]>=boundary]
        expected = {"hours":len(selected), "positive_increases":sum(x["increase"]>0 for x in selected),
            "surge_hours":len(events), "surge_dates":len({x["date"] for x in events}),
            "old_rise_hours":sum(x["legacy_rise"] for x in selected), "overlap_hours":sum(x["legacy_rise"] for x in events),
            "outside_prior_low_surges":sum(not x["prior_low"] for x in events), "previous_zero_hours":sum(x["previous_maximum"]==0 for x in selected)}
        for key,value in expected.items():
            close(r[key],value)
    pred = rows(OUT/"predictions.csv")
    original = {(r["group"],r["timestamp"]):r for r in rows(ROOT/"Modeling/tables/m04_rise_gate/predictions.csv")}
    groups = defaultdict(list)
    for r in pred:
        t = datetime.fromisoformat(r["timestamp"])
        assert t.month in [4,5,6] and t in eligible
        previous = frame[t]
        close(r["actual"],previous["maximum"])
        close(r["lag1_maximum"],previous["previous_maximum"])
        close(r["increase"],previous["increase"])
        close(r["prediction"],original[r["group"],r["timestamp"]]["prediction"])
        close(r["predicted_increase"],float(r["prediction"])-previous["previous_maximum"])
        assert (r["surge"]=="True")== (previous["increase"]>=boundaries["q95"])
        groups[r["group"]].append(r)
    assert len(pred)==run["prediction_rows"]==15288 and all(len(g)==2184 for g in groups.values())
    for r in rows(OUT/"errors.csv"):
        part = groups[r["group"]]
        if r["split"]!="pooled_development":
            part = [x for x in part if x["split"]==r["split"]]
        boundary = boundaries[r["boundary_name"]]
        condition = r["condition"]
        if condition=="surge":
            part = [x for x in part if float(x["increase"])>=boundary]
        elif condition=="non_surge":
            part = [x for x in part if float(x["increase"])<boundary]
        elif condition=="legacy_rise":
            part = [x for x in part if x["rise_event"]=="True"]
        elif condition=="low_stay":
            part = [x for x in part if x["power_transition"]=="low->low"]
        errors = [float(x["prediction"])-float(x["actual"]) for x in part]
        expected = {"hours":len(part), "dates":len({x["date"] for x in part}), "MAE":math.fsum(abs(e) for e in errors)/len(part),
            "RMSE":math.sqrt(math.fsum(e*e for e in errors)/len(part)), "mean_under":math.fsum(max(-e,0) for e in errors)/len(part),
            "mean_over":math.fsum(max(e,0) for e in errors)/len(part), "under_hours":sum(e<0 for e in errors),
            "mean_actual":math.fsum(float(x["actual"]) for x in part)/len(part), "mean_increase":math.fsum(float(x["increase"]) for x in part)/len(part)}
        for key,value in expected.items():
            close(r[key],value)
    cached = rows(ROOT/"Modeling/tables/m04_rise/inner_predictions.csv")
    policies = rows(OUT/"alarm_policies.csv")
    for policy in policies:
        group = "B" if policy["score"]=="B_predicted_increase" else "G_HGB"
        pool = [r for r in cached if r["split"]==policy["split"] and r["group"]==group]
        y = [frame[datetime.fromisoformat(r["timestamp"])]["increase"]>=boundaries["q95"] for r in pool]
        scores = [float(r["prediction"])-frame[datetime.fromisoformat(r["timestamp"])]["previous_maximum"] if group=="B" else float(r["probability"]) if r["prior_low"]=="True" else 0. for r in pool]
        chosen = frontier(y,scores,float(policy["cap"]))
        for key,value in chosen.items():
            close(policy[key],value)
        assert max(r["timestamp"] for r in pool)==policy["inner_last"]<policy["outer_first"]
    alarms = rows(OUT/"alarm_metrics.csv")
    for r in alarms:
        pool = groups[r["group"]]
        if r["split"]!="pooled_development":
            pool = [x for x in pool if x["split"]==r["split"]]
        boundary = boundaries[r["boundary_name"]]
        scores = [float(x["probability"]) if x["prior_low"]=="True" else 0. for x in pool] if r["score"]=="legacy_HGB_probability" else [float(x["predicted_increase"]) for x in pool]
        check_alarm(r,pool,scores,boundary)
        if r["policy"]=="past_inner_selected":
            policy = next(x for x in policies if x["split"]==r["split"] and x["score"]==r["score"] and x["cap"]==r["cap"])
            close(r["threshold"],policy["threshold"])
        elif r["policy"]=="descriptive_outer_frontier":
            chosen = frontier([float(x["increase"])>=boundary for x in pool],scores,float(r["cap"]))
            for key,value in chosen.items():
                close(r[key],value)
    april = [r for r in groups["G_HGB"] if r["split"]=="dev_apr" and r["prior_low"]=="True"]
    indexed = {(r["group"],r["timestamp"]):r for r in pred}
    for case in rows(OUT/"april_legacy_cases.csv"):
        actual = indexed["G_HGB",case["timestamp"]]
        assert actual["rise_event"]=="True" and case["trigger"]==actual["trigger"]
        p = float(case["probability"])
        assert int(case["rank_min"])==1+sum(float(r["probability"])>p for r in april)
        assert int(case["rank_max"])==sum(float(r["probability"])>=p for r in april)
        assert (case["surge"]=="True")== (float(case["increase"])>=boundaries["q95"])
        for name,label in [("B","prediction_B"),("W_rise_weighted","prediction_expert"),("G_HGB","prediction_combined")]:
            close(case[label],indexed[name,case["timestamp"]]["prediction"])
    result = {"status":"passed", "checked_at":datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"),
        "run_sha256":sha(OUT/"run.json"), "script_sha256":sha(Path(__file__)), "boundary":boundaries,
        "prediction_rows":len(pred), "error_rows":len(rows(OUT/"errors.csv")), "alarm_rows":len(alarms),
        "policies_checked":len(policies), "april_cases_checked":8, "raw_common_hours_checked":len(frame),
        "no_fitting":True, "no_july_august_evaluation":True}
    (OUT/"independent_verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
