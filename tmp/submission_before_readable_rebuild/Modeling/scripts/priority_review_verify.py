"""Independent streaming replay, metric and paired-week sensitivity checks."""
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Modeling/tables/priority_review"
PARENT = ROOT / "Modeling/tables/error_warning"

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def records(p):
    with Path(p).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def truth(x):
    assert x in ("True","False")
    return x == "True"

def quantile(sorted_values, q):
    position = (len(sorted_values)-1)*q
    low, high = math.floor(position), math.ceil(position)
    return sorted_values[low] + (sorted_values[high]-sorted_values[low])*(position-low)

def close(a,b):
    assert math.isclose(float(a),float(b),abs_tol=1e-8,rel_tol=1e-10),(a,b)

def summarize(rows):
    event = [truth(x["event"]) for x in rows]
    selected = [truth(x["review"]) for x in rows]
    forced = [truth(x["forced"]) for x in rows]
    under = [float(x["under_amount"]) for x in rows]
    count = sum(event)
    tp = sum(a and b for a,b in zip(event,selected))
    num = sum(selected)
    captured = math.fsum(u for u,s in zip(under,selected) if s)
    total = math.fsum(under)
    return {"hours":len(rows),"days":len({x["date"] for x in rows}),"events":count,
            "reviews":num,"TP":tp,"non_event_reviews":num-tp,"FN":count-tp,
            "precision":tp/num,"recall":tp/count if count else 0.,
            "review_fraction":num/len(rows),"captured_under":captured,
            "missed_under":total-captured,"captured_under_share":captured/total,
            "forced_reviews":sum(forced),
            "forced_TP":sum(a and b for a,b in zip(forced,event))}

def main():
    assert sys.version_info[:2] == (3,13) and sys._is_gil_enabled()
    c = json.loads((OUT/"contract.json").read_text(encoding="utf-8"))
    run = json.loads((OUT/"run.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed" and run["contract_sha256"] == sha(OUT/"contract.json")
    assert c["entrypoint_sha256"] == sha(ROOT/"Modeling/scripts/priority_review.py")
    for path,digest in c["inputs_sha256"].items():
        assert sha(ROOT/path) == digest,path
    for name,digest in run["outputs_sha256"].items():
        assert sha(OUT/name) == digest,name
    parent_verified = json.loads((PARENT/"independent_verification.json").read_text(encoding="utf-8"))
    assert parent_verified["status"] == "passed"
    assert parent_verified["run_sha256"] == sha(PARENT/"run.json")
    assert parent_verified["verifier_sha256"] == sha(ROOT/"Modeling/scripts/error_warning_verify.py")
    parent_run = json.loads((PARENT/"run.json").read_text(encoding="utf-8"))
    for name,digest in parent_run["outputs_sha256"].items():
        assert sha(PARENT/name) == digest
    for m in records(PARENT/"model_manifest.csv"):
        assert sha(ROOT/m["file"]) == m["sha256"]
    parent = {(x["method"],x["timestamp"]):x for x in records(PARENT/"predictions.csv")}
    selection = {x["month"]:x for x in json.loads((PARENT/"selection.json").read_text(encoding="utf-8"))}
    cal = defaultdict(list)
    cal_end = {}
    for x in records(PARENT/"calibration_predictions.csv"):
        key = (x["method"],int(x["outer_month"]))
        assert int(x["month"]) == key[1]-1
        cal[key].append(float(x["score"]))
        cal_end[key] = max(cal_end.get(key,""),x["timestamp"])
    for values in cal.values():
        values.sort()
    result = records(OUT/"predictions.csv")
    groups = defaultdict(list)
    for x in result:
        groups[(x["method"],x["date"])].append(x)
    assert len(result) == 10920 and len(groups) == 455
    for (method,date),rows in groups.items():
        rows.sort(key=lambda x:x["timestamp"])
        assert [datetime.fromisoformat(x["timestamp"]).hour for x in rows] == list(range(24))
        used = 0
        for hour,row in enumerate(rows):
            month = int(row["month"])
            src = parent[(method,row["timestamp"])]
            for field in ["event","under_amount","actual","prediction","score","month","date"]:
                if field in ("under_amount","actual","prediction","score"):
                    close(row[field],src[field])
                else:
                    assert row[field] == src[field],field
            model = selection[month]["ML" if method == "selected_ML" else "simple"] if method.startswith("selected_") else method
            key = (model,month)
            assert row["score_method"] == model and row["calibration_end"] == cal_end[key]
            assert cal_end[key] < row["timestamp"]
            remaining = 2-used
            assert int(row["remaining_before"]) == remaining
            assert int(row["hours_remaining"]) == 24-hour
            forced = remaining > 0 and remaining == 24-hour
            threshold = None
            if remaining == 0:
                chosen = False
            elif forced:
                chosen = True
            else:
                threshold = quantile(cal[key],1-remaining/(24-hour))
                chosen = float(row["score"]) > threshold
            assert truth(row["review"]) == chosen and truth(row["forced"]) == forced
            if threshold is None:
                assert row["boundary"] == ""
            else:
                close(row["boundary"],threshold)
            used += int(chosen)
        assert used == 2
    metric_rows = records(OUT/"metrics.csv")
    for row in metric_rows:
        rs = [x for x in result if x["method"] == row["method"] and
              (row["period"] == "pooled" or x["month"] == row["period"])]
        for field,value in summarize(rs).items():
            close(row[field],value)
    daily = records(OUT/"daily.csv")
    for row in daily:
        for field,value in summarize(groups[(row["method"],row["date"])]).items():
            close(row[field],value)
    for row in records(OUT/"hourly_allocation.csv"):
        selected = [x for x in result if x["method"] == row["method"] and
                    datetime.fromisoformat(x["timestamp"]).hour == int(row["hour"])]
        assert sum(truth(x["review"]) for x in selected) == int(row["reviews"])
        assert sum(truth(x["forced"]) for x in selected) == int(row["forced"])
    by_day = {(x["method"],x["date"]):x for x in daily}
    dates = sorted({x["date"] for x in daily})
    for row in records(OUT/"sensitivity.csv"):
        weeks = defaultdict(lambda:[0.,0.])
        for date in dates:
            target = by_day[("selected_ML",date)]
            base = by_day[(row["reference"],date)]
            t = pd.Timestamp(date)
            monday = (t-pd.Timedelta(days=t.dayofweek)).strftime("%Y-%m-%d")
            weeks[monday][0] += int(target["TP"])-int(base["TP"])
            weeks[monday][1] += float(target["captured_under"])-float(base["captured_under"])
        values = [weeks[w] for w in sorted(weeks)]
        assert int(row["weeks"]) == len(values)
        indices = np.random.default_rng(42).integers(0,len(values),size=(2000,len(values)))
        samples = [[math.fsum(values[i][j] for i in sample) for j in [0,1]] for sample in indices]
        for j,prefix in enumerate(["TP","under"]):
            close(row[prefix+"_difference"],math.fsum(v[j] for v in values))
            sorted_samples = sorted(s[j] for s in samples)
            close(row[prefix+"_low"],quantile(sorted_samples,.025))
            close(row[prefix+"_high"],quantile(sorted_samples,.975))
    index = {(x["method"],x["period"]):x for x in metric_rows}
    a = int(index[("selected_ML","pooled")]["TP"])
    strongest = max(int(index[(m,"pooled")]["TP"]) for m in c["methods"] if m != "selected_ML")
    flags = {"at_least_five_more":a>=strongest+5,"at_least_twenty_percent_more":a>=strongest*1.2,
             "two_month_wins":sum(int(index[("selected_ML",str(m))]["TP"]) >
                                 int(index[("selected_simple",str(m))]["TP"]) for m in [4,5,6])>=2}
    assert flags == run["decision"]["criteria"]
    assert all(flags.values()) == run["decision"]["reportworthy_development_result"]
    # Future score perturbation must leave all preceding decisions unchanged.
    from priority_review import replay
    prefix_cases = 0
    for method in c["methods"]:
        for month in [4,5,6]:
            first_day = sorted((v for (m,d),v in groups.items() if m==method and int(v[0]["month"])==month),
                               key=lambda r:r[0]["timestamp"])[0]
            t = pd.to_datetime([x["timestamp"] for x in first_day])
            scores = np.array([float(x["score"]) for x in first_day])
            model = first_day[0]["score_method"]
            original = replay(scores,np.array(cal[(model,month)]),t)
            for cut in [1,12,23]:
                changed = scores.copy()
                changed[cut:] = 1e9
                altered = replay(changed,np.array(cal[(model,month)]),t)
                pd.testing.assert_frame_equal(original.iloc[:cut],altered.iloc[:cut])
                prefix_cases += 1
    verification = {"status":"passed","verifier_sha256":sha(__file__),
                    "run_sha256":sha(OUT/"run.json"),"parent_models_hash_checked":6,
                    "decision_rows_checked":len(result),"daily_quotas_checked":len(groups),
                    "metrics_checked":len(metric_rows),"future_score_perturbation_cases":prefix_cases,
                    "bootstrap_resamples_per_reference":2000,"new_model_fits":0,
                    "decision":run["decision"],"limitations":"reused development, not plant effectiveness"}
    (OUT/"independent_verification.json").write_text(json.dumps(verification,indent=2),encoding="utf-8")
    print(json.dumps(verification),flush=True)

if __name__ == "__main__":
    main()