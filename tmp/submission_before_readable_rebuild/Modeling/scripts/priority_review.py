"""Replay a fixed daily review budget using current/past scores only.
This is exploratory development evidence, not measured plant intervention.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "Modeling/tables/error_warning"
OUT = ROOT / "Modeling/tables/priority_review"
METHODS = ["high_prediction", "prior_maximum", "neighbor_std", "selected_simple", "selected_ML"]

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(x, p):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding="utf-8")

def read(p):
    return pd.read_csv(p, encoding="utf-8-sig", float_precision="round_trip",
                       parse_dates=["timestamp", "date"])

def freeze():
    assert not (OUT / "contract.json").exists()
    verified = json.loads((PARENT / "independent_verification.json").read_text(encoding="utf-8"))
    assert verified["status"] == "passed"
    inputs = [PARENT / n for n in ["predictions.csv", "calibration_predictions.csv",
              "selection.json", "contract.json", "run.json", "independent_verification.json"]]
    OUT.mkdir(parents=True, exist_ok=True)
    contract = {
        "recorded_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"),
        "entrypoint_sha256": sha(__file__),
        "inputs_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
        "scope": "April-June 2021 reused development, not independent final test; no later-month evaluation",
        "target": "Existing February-derived large-underprediction event >=9.73851536905958",
        "daily_budget": 2,
        "schedule": "24 hourly origins 00..23 on each date; planned availability, not future score/outcome",
        "policy": "At hour h, b reviews remain and r=24-h origins remain. If b=0 skip. If b=r select to use quota. Otherwise select iff current score > previous-calendar-month score quantile 1-b/r (linear interpolation). Reset b=2 each date. No within-evaluation-month outcome or score update.",
        "calibration": "Use original prior-month calibration scores from same frozen model; model selection remains frozen from parent AP, not reselected for new policy",
        "methods": METHODS,
        "primary_reference": "selected_simple frozen in parent study",
        "strong_baseline_check": "compare pooled TP against max of all three fixed simple rules and selected_simple; this is a conservative hurdle, not deployed future selection",
        "reportworthy": "selected_ML pooled TP >= strongest simple TP+5 and >=1.20*strongest simple TP, and TP strictly exceeds selected_simple in at least 2 of 3 months. No all-month-win requirement. Report all monthly and severity results even if mixed.",
        "secondary": "precision, recall, non-event reviews, missed events, captured underprediction amount, forced-end-of-day reviews, hourly allocation",
        "uncertainty": "paired calendar-week cluster bootstrap, 2000 resamples, seed42, descriptive sensitivity only given reused development/repeated arrays; not selection gate",
        "fits": 0,
        "no_policy_search": True,
        "interpretation": "Online replay of choosing which forecast to review, not correction, automatic plant action, cost saving, or forecasting the event after its outcome",
        "report_scope": "If reportworthy preserve original then add bounded detailed manuscript result; otherwise keep research README only. No compressed manuscript edits."
    }
    dump(contract, OUT / "contract.json")
    print("Priority review contract frozen before replay.", flush=True)

def replay(scores, calibration, timestamps, budget=2):
    """Only scalars already available at each origin enter the decision."""
    remaining = budget
    previous_date = None
    rows = []
    for t, score in zip(timestamps, scores):
        date = t.date()
        if date != previous_date:
            if previous_date is not None:
                assert remaining == 0
            remaining = budget
            previous_date = date
        r = 24 - t.hour
        b = remaining
        if not b:
            threshold = np.nan
            chosen, forced = False, False
        elif b == r:
            threshold = np.nan
            chosen, forced = True, True
        else:
            threshold = float(np.quantile(calibration, 1 - b / r, method="linear"))
            chosen, forced = bool(score > threshold), False
        if chosen:
            remaining -= 1
        rows.append({"remaining_before": b, "hours_remaining": r, "boundary": threshold,
                     "review": chosen, "forced": forced})
    assert remaining == 0
    return pd.DataFrame(rows)

def summary(g):
    reviewed = g.review.to_numpy(bool)
    event = g.event.to_numpy(bool)
    under = g.under_amount.to_numpy()
    tp = int((reviewed & event).sum())
    n = int(event.sum())
    reviews = int(reviewed.sum())
    return {"hours": len(g), "days": g.date.nunique(), "events": n,
            "reviews": reviews, "TP": tp, "non_event_reviews": reviews-tp, "FN": n-tp,
            "precision": tp/reviews, "recall": tp/n if n else 0.,
            "review_fraction": reviews/len(g),
            "captured_under": float(under[reviewed].sum()),
            "missed_under": float(under[~reviewed].sum()),
            "captured_under_share": float(under[reviewed].sum()/under.sum()),
            "forced_reviews": int(g.forced.sum()),
            "forced_TP": int((g.forced & g.event).sum())}

def sensitivity(daily):
    """Preserve within-week pairing; do not claim independent-date inference."""
    d = daily.copy()
    d["week"] = pd.to_datetime(d.date).dt.to_period("W-SUN").astype(str)
    rows = []
    for ref in METHODS[:-1]:
        target = d[d.method.eq("selected_ML")].set_index("date")
        base = d[d.method.eq(ref)].set_index("date")
        diff = pd.DataFrame({"week": target.week,
                             "TP_difference": target.TP-base.TP,
                             "under_difference": target.captured_under-base.captured_under})
        clusters = diff.groupby("week")[["TP_difference","under_difference"]].sum().to_numpy()
        rng = np.random.default_rng(42)
        samples = clusters[rng.integers(0,len(clusters),size=(2000,len(clusters)))].sum(axis=1)
        rows.append({"reference": ref, "weeks": len(clusters),
            "TP_difference": int(diff.TP_difference.sum()),
            "TP_low": float(np.quantile(samples[:,0],.025)),
            "TP_high": float(np.quantile(samples[:,0],.975)),
            "under_difference": float(diff.under_difference.sum()),
            "under_low": float(np.quantile(samples[:,1],.025)),
            "under_high": float(np.quantile(samples[:,1],.975))})
    return pd.DataFrame(rows)

def run():
    assert not (OUT / "run.json").exists()
    c = json.loads((OUT / "contract.json").read_text(encoding="utf-8"))
    assert c["entrypoint_sha256"] == sha(__file__)
    for p, h in c["inputs_sha256"].items():
        assert sha(ROOT / p) == h
    source = read(PARENT / "predictions.csv")
    calibration = read(PARENT / "calibration_predictions.csv")
    selections = {x["month"]: x for x in json.loads((PARENT / "selection.json").read_text(encoding="utf-8"))}
    outputs = []
    for method in METHODS:
        for month in [4,5,6]:
            actual_method = selections[month]["ML" if method == "selected_ML" else "simple"] if method.startswith("selected_") else method
            ev = source[source.method.eq(method) & source.month.eq(month)].sort_values("timestamp").reset_index(drop=True)
            cal = calibration[calibration.method.eq(actual_method) & calibration.outer_month.eq(month)].sort_values("timestamp")
            assert cal.timestamp.max() < ev.timestamp.min()
            assert cal.month.eq(month-1).all()
            assert ev.date.nunique() == {4:30,5:31,6:30}[month]
            for _, day in ev.groupby("date"):
                assert day.timestamp.dt.hour.tolist() == list(range(24))
            result = replay(ev.score.to_numpy(), cal.score.to_numpy(), ev.timestamp, c["daily_budget"])
            z = pd.concat([ev[["timestamp","date","month","method","score","event","under_amount","actual","prediction"]],result],axis=1)
            z["calibration_end"] = cal.timestamp.max()
            z["score_method"] = actual_method
            outputs.append(z)
    predictions = pd.concat(outputs,ignore_index=True)
    daily = pd.DataFrame([{"method":m,"date":str(date.date()),"month":int(g.month.iloc[0]),**summary(g)}
                          for (m,date),g in predictions.groupby(["method","date"])])
    assert daily.reviews.eq(c["daily_budget"]).all()
    metrics = []
    for (m,month),g in predictions.groupby(["method","month"]):
        metrics.append({"method":m,"period":str(month),**summary(g)})
    for m,g in predictions.groupby("method"):
        metrics.append({"method":m,"period":"pooled",**summary(g)})
    metrics = pd.DataFrame(metrics)
    index = metrics.set_index(["method","period"])
    a = index.loc[("selected_ML","pooled")]
    strongest = max(int(index.loc[(m,"pooled"),"TP"]) for m in METHODS[:-1])
    flags = {"at_least_five_more": bool(a.TP >= strongest+5),
             "at_least_twenty_percent_more": bool(a.TP >= strongest*1.20),
             "two_month_wins": sum(int(index.loc[("selected_ML",str(m)),"TP"] >
                                       index.loc[("selected_simple",str(m)),"TP"]) for m in [4,5,6]) >= 2}
    decision = {"reportworthy_development_result": all(flags.values()),"criteria":flags,
                "strongest_simple_TP":strongest,
                "not_automatic_warning_acceptance": True,
                "not_independent_test": True}
    hourly = predictions.groupby(["method",predictions.timestamp.dt.hour.rename("hour")]).agg(
        reviews=("review","sum"),forced=("forced","sum")).reset_index()
    for name,table in [("predictions",predictions),("daily",daily),("metrics",metrics),
                       ("sensitivity",sensitivity(daily)),("hourly_allocation",hourly)]:
        table.to_csv(OUT/(name+".csv"),index=False,encoding="utf-8-sig")
    dump(decision,OUT/"decision.json")
    result = {"status":"completed","contract_sha256":sha(OUT/"contract.json"),"new_fits":0,
              "hours_per_method":2184,"dates":91,"reviews_per_method":182,"events":184,
              "decision":decision,"later_evaluated":False,
              "outputs_sha256":{p.name:sha(p) for p in [OUT/(n+".csv") for n in
                 ["predictions","daily","metrics","sensitivity","hourly_allocation"]]+[OUT/"decision.json"]}}
    dump(result, OUT/"run.json")
    print(metrics[metrics.period.eq("pooled")].to_string(index=False),flush=True)
    print(metrics[metrics.method.isin(["selected_ML","selected_simple"])].to_string(index=False),flush=True)
    print(json.dumps(decision),flush=True)

if __name__ == "__main__":
    assert sys.version_info[:2] == (3,13) and sys._is_gil_enabled()
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze","run"])
    globals()[parser.parse_args().stage]()