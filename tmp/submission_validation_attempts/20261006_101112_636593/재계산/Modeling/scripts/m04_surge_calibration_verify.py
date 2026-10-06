"""Independent order-statistic cutoff and frozen-score/selection verification."""
import json
import math
import sys
from collections import defaultdict
from datetime import datetime,timedelta,timezone
from pathlib import Path

from m03_verify import rows,sha,close
from m04_surge_verify import alarm,check_alarm

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"Modeling/tables/m04_surge_calibration"
PARENT=ROOT/"Modeling/tables/m04_surge_risk"


def main():
    assert sys.version_info[:2]==(3,13) and sys._is_gil_enabled()
    run=json.loads((OUT/"run.json").read_text(encoding="utf-8"))
    cpath=ROOT/"Modeling/config/m04_surge_calibration_contract.json"
    c=json.loads(cpath.read_text(encoding="utf-8"))
    risk=json.loads((ROOT/"Modeling/config/m04_surge_risk_contract.json").read_text(encoding="utf-8"))
    verified=json.loads((PARENT/"independent_verification.json").read_text(encoding="utf-8"))
    assert verified["status"]=="passed" and verified["run_sha256"]==sha(PARENT/"run.json")==c["parent_run_sha256"]
    assert run["status"]=="completed" and run["no_new_fitting"] and run["selection_gates_unchanged"] and run["normal_forecast_unchanged"]
    assert sha(cpath)==run["contract_sha256"] and sha(ROOT/"Modeling/scripts/m04_surge_calibration.py")==run["script_sha256"]
    for field in ["inputs_sha256","outputs_sha256"]:
        for name,digest in run[field].items():
            assert sha(ROOT/name if field=="inputs_sha256" else OUT/name)==digest
    parent={(r["split"],r["group"],r["timestamp"]):r for r in rows(PARENT/"predictions.csv")}
    pred=rows(OUT/"predictions.csv")
    inner=rows(PARENT/"inner_predictions.csv")
    parts,inside,groups=defaultdict(list),defaultdict(list),defaultdict(list)
    for r in pred:
        original=parent[r["split"],r["group"],r["timestamp"]]
        for field,value in original.items():
            if field.startswith("alarm_"):
                continue
            assert r[field]==value,(field,r["timestamp"])
        parts[r["split"],r["group"]].append(r)
        groups[r["group"]].append(r)
    assert len(pred)==run["prediction_rows"]==10920
    for r in inner:
        inside[r["split"],r["group"]].append(r)
    policies=rows(OUT/"alarm_policies.csv")
    for r in policies:
        pool=inside[r["split"],r["group"]]
        negatives=sorted([float(x["score"]) for x in pool if float(x["increase"])<93],reverse=True)
        k=math.floor(float(r["cap"])*len(negatives))
        threshold=math.nextafter(negatives[k],math.inf) if k<len(negatives) else math.nextafter(negatives[-1],-math.inf)
        close(r["threshold"],threshold)
        assert len(negatives)==int(r["negative_count"]) and k==int(r["budget_count"])
        assert max(x["timestamp"] for x in pool)==r["inner_last"]<r["outer_first"]
        values=alarm([float(x["increase"])>=93 for x in pool],[float(x["score"]) for x in pool],threshold)
        assert values["FP"]<=k and values["false_positive_fraction"]<=float(r["cap"])+1e-12
        for key,value in values.items():
            close(r[key],value)
        for x in parts[r["split"],r["group"]]:
            assert (x["alarm_"+r["cap"]]=="True")== (float(x["score"])>=threshold)
    alarms=rows(OUT/"alarm_metrics.csv")
    for r in alarms:
        pool=parts[r["split"],r["group"]]
        threshold=next(p["threshold"] for p in policies if p["split"]==r["split"] and p["group"]==r["group"] and p["cap"]==r["cap"])
        close(r["threshold"],threshold)
        check_alarm(r,pool,[float(x["score"]) for x in pool],93 if r["boundary_name"]=="q95" else risk["boundary_sensitivity"])
    table=rows(OUT/"selection_metrics.csv")
    original_table={r["group"]:r for r in rows(PARENT/"selection_metrics.csv")}
    reference=next(r for r in table if r["group"]=="B_score")
    for r in table:
        for field in ["AP","AUROC","events","hours","weighted_monthly_AP"]:
            close(r[field],original_table[r["group"]][field])
        pool=groups[r["group"]]
        tp=sum(x["alarm_0.01"]=="True" and float(x["increase"])>=93 for x in pool)
        fp=sum(x["alarm_0.01"]=="True" and float(x["increase"])<93 for x in pool)
        close(r["TP"],tp);close(r["FP"],fp)
        close(r["recall"],tp/8);close(r["precision"],tp/(tp+fp) if tp+fp else 0)
        close(r["false_positive_fraction"],fp/2176)
        monthly=[x for x in alarms if x["group"]==r["group"] and x["cap"]=="0.01" and x["boundary_name"]=="q95"]
        monthly_ok=all(int(x["TP"])>=int(next(b for b in alarms if b["group"]=="B_score" and b["split"]==x["split"] and b["cap"]=="0.01" and b["boundary_name"]=="q95")["TP"]) for x in monthly)
        flags={"rank_passed":float(r["weighted_monthly_AP"])>=1.05*float(reference["weighted_monthly_AP"]),"recall_passed":tp>=int(reference["TP"]),
            "monthly_recall_passed":monthly_ok,"false_positive_passed":fp/2176<=.01+1e-12,"strict_gain_passed":tp>int(reference["TP"]) or fp<int(reference["FP"])}
        for key,value in flags.items():
            assert (r[key]=="True")==value
        assert (r["eligible"]=="True")== (r["group"] in ["Logistic_all","HGB_all","Q90_upper"] and all(flags.values()))
    eligible=[r for r in table if r["eligible"]=="True"]
    if eligible:
        best=max(float(r["weighted_monthly_AP"]) for r in eligible)
        near=[r for r in eligible if float(r["weighted_monthly_AP"])>=best/1.01]
        chosen=min(near,key=lambda r:(["Logistic_all","HGB_all","Q90_upper"].index(r["group"]),-float(r["weighted_monthly_AP"]))) ["group"]
    else:
        chosen="B_score"
    selection=json.loads((OUT/"selection.json").read_text(encoding="utf-8"))
    assert selection["selected_risk_channel"]==chosen and selection["normal_forecast"]=="B" and selection["selection_gates_unchanged"]
    assert len(policies)==30 and len(alarms)==60
    result={"status":"passed","checked_at":datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"),
        "run_sha256":sha(OUT/"run.json"),"script_sha256":sha(Path(__file__)),"prediction_rows":len(pred),"policy_rows":len(policies),
        "alarm_rows":len(alarms),"scores_and_models_unchanged":True,"normal_forecast_unchanged":True,"selection_gates_unchanged":True,
        "selected_risk_channel":chosen,"no_new_fitting":True,"no_july_august_evaluation":True}
    (OUT/"independent_verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
