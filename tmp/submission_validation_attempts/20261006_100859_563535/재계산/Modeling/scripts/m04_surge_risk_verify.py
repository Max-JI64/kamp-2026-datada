"""Independent trial arrays, chronological gates, saved models and separate forecast checks."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from m03_verify import rows, sha, close
from m04_rise_verify import ap
from m04_surge_verify import frontier, check_alarm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Modeling/tables/m04_surge_risk"


def main():
    assert sys.version_info[:2]==(3,13) and sys._is_gil_enabled()
    run=json.loads((OUT/"run.json").read_text(encoding="utf-8"))
    cpath=ROOT/"Modeling/config/m04_surge_risk_contract.json"
    c=json.loads(cpath.read_text(encoding="utf-8"))
    assert run["status"]=="completed" and run["normal_forecast_unchanged"] and run["no_july_august_evaluation"]
    assert sha(cpath)==run["contract_sha256"] and sha(ROOT/"Modeling/scripts/m04_surge_risk.py")==run["script_sha256"]
    for field in ["inputs_sha256","outputs_sha256","helper_sha256"]:
        for name,digest in run[field].items():
            path=ROOT/name if field=="inputs_sha256" else OUT/name if field=="outputs_sha256" else ROOT/"Modeling/scripts"/name
            assert sha(path)==digest,(field,name)
    diagnosis=json.loads((ROOT/"Modeling/tables/m04_surge/independent_verification.json").read_text(encoding="utf-8"))
    assert diagnosis["status"]=="passed" and diagnosis["run_sha256"]==c["diagnosis_run_sha256"]
    original=rows(ROOT/"Modeling/tables/m04_surge/predictions.csv")
    inherited={(r["group"],r["timestamp"]):r for r in original}
    frame={r["timestamp"]:r for r in rows(ROOT/"Modeling/tables/m01/hourly_frame.csv") if r["eligible_common"]=="True" and r["timestamp"]<"2021-07-01"}
    features=rows(ROOT/"Modeling/tables/m04_rise/feature_frame.csv")
    feature_map={r["timestamp"]:r for r in features}
    pred,inner=rows(OUT/"predictions.csv"),rows(OUT/"inner_predictions.csv")
    groups,parts,inside=defaultdict(list),defaultdict(list),defaultdict(list)
    for r in pred:
        baseline=inherited["B",r["timestamp"]]
        for field in ["actual","normal_prediction","prediction"]:
            close(r[field],baseline["actual"] if field=="actual" else baseline["prediction"])
        close(r["increase"],baseline["increase"])
        close(r["previous_maximum"],baseline["lag1_maximum"])
        assert datetime.fromisoformat(r["timestamp"]).month in [4,5,6]
        if r["group"] in ["B_score","G_HGB_score"]:
            source="B" if r["group"]=="B_score" else "G_HGB"
            close(r["score"],float(inherited[source,r["timestamp"]]["prediction"])-float(baseline["lag1_maximum"]))
        if r["group"]=="Q90_upper":
            close(r["score"],float(r["upper_prediction"])-float(r["previous_maximum"]))
        groups[r["group"]].append(r)
        parts[r["split"],r["group"]].append(r)
    assert len(pred)==run["prediction_rows"]==10920 and len(groups)==5 and all(len(g)==2184 for g in groups.values())
    for r in inner:
        f=frame[r["timestamp"]]
        close(r["increase"],float(f["target_maximum"])-float(f["lag1_maximum"]))
        close(r["target_maximum"],f["target_maximum"])
        close(r["lag1_maximum"],f["lag1_maximum"])
        inside[r["split"],r["group"]].append(r)
    assert len(inner)==32160
    for r in rows(OUT/"fold_audit.csv"):
        month=int(r["month"])
        start=datetime(2021,month,1)
        train=[v for t,v in frame.items() if datetime.fromisoformat(t)<start]
        validation=[v for t,v in frame.items() if datetime.fromisoformat(t).month==month]
        assert len(train)==int(r["train_hours"]) and len(validation)==int(r["validation_hours"])
        assert r["train_last"]<r["validation_first"]<=r["validation_last"]<r["outer_first"]
        for label,pool in [("train",train),("validation",validation)]:
            assert int(r[label+"_events"])==sum(float(v["target_maximum"])-float(v["lag1_maximum"])>=93 for v in pool)
    trial_rows=rows(OUT/"trials.csv")
    configs=json.loads((OUT/"selected_configurations.json").read_text(encoding="utf-8"))
    trial_arrays=np.load(OUT/"trial_oof_predictions.npz",allow_pickle=False)
    checked_trials=0
    for config in configs:
        split,name=config["split"],config["name"]
        part=[r for r in trial_rows if r["split"]==split and r["name"]==name]
        assert len(part)==20 and all(r["state"]=="COMPLETE" for r in part)
        pool=inside[split,name]
        labels=[float(r["increase"])>=93 for r in pool]
        assert max(r["timestamp"] for r in pool)<min(r["timestamp"] for r in parts[split,name])
        array=trial_arrays[split+"_"+name]
        assert array.shape==(20,len(pool))
        for r in part:
            scores=array[int(r["trial"])].tolist()
            values=json.loads(r["metrics"])["metrics"]
            close(r["objective"],ap(labels,scores))
            close(values["AP"],ap(labels,scores))
            close(values["Brier"],math.fsum((s-y)**2 for y,s in zip(labels,scores))/len(pool))
            checked_trials+=1
        selected=min(part,key=lambda r:(-float(r["objective"]),json.loads(r["metrics"])["metrics"]["Brier"],int(r["trial"])))
        assert int(selected["trial"])==config["best_trial"] and json.loads(selected["parameters"])==config["parameters"]
        assert np.allclose(array[config["best_trial"]],[float(r["score"]) for r in pool],atol=1e-12,rtol=0)
    assert checked_trials==run["trials"]==120
    policies=rows(OUT/"alarm_policies.csv")
    for r in policies:
        pool=inside[r["split"],r["group"]]
        expected=frontier([float(x["increase"])>=93 for x in pool],[float(x["score"]) for x in pool],float(r["cap"]))
        for field,value in expected.items():
            close(r[field],value)
        for x in parts[r["split"],r["group"]]:
            assert (x["alarm_"+r["cap"]]=="True")== (float(x["score"])>=float(r["threshold"]))
    alarm_rows=rows(OUT/"alarm_metrics.csv")
    assert len(alarm_rows)==60
    for r in alarm_rows:
        pool=parts[r["split"],r["group"]]
        boundary=93 if r["boundary_name"]=="q95" else c["boundary_sensitivity"]
        check_alarm(r,pool,[float(x["score"]) for x in pool],boundary)
        policy=next(p for p in policies if p["split"]==r["split"] and p["group"]==r["group"] and p["cap"]==r["cap"])
        close(r["threshold"],policy["threshold"])
        if r["Brier"]:
            close(r["Brier"],math.fsum((float(x["score"])-(float(x["increase"])>=93))**2 for x in pool)/len(pool))
    regression_rows=rows(OUT/"regression_metrics.csv")
    for r in regression_rows:
        pool=parts[r["split"],"Q90_upper" if r["group"]=="Q90_upper" else "B_score"]
        condition=r["condition"]
        if condition.startswith("surge_"):
            boundary=93 if condition.endswith("q95") else c["boundary_sensitivity"]
            pool=[x for x in pool if float(x["increase"])>=boundary]
        elif condition=="low_stay":
            pool=[x for x in pool if x["power_transition"]=="low->low"]
        values=[]
        for x in pool:
            prediction=float(x["upper_prediction"]) if r["group"]=="Q90_upper" else float(inherited[r["group"],x["timestamp"]]["prediction"])
            values.append(prediction-float(x["actual"]))
        weights=[float(x["daily_maximum_weight"]) for x in pool]
        expected={"hours":len(pool),"MAE":math.fsum(abs(e) for e in values)/len(pool),"mean_under":math.fsum(max(-e,0) for e in values)/len(pool),
            "mean_over":math.fsum(max(e,0) for e in values)/len(pool),"coverage":sum(e>=0 for e in values)/len(pool),
            "pinball90":math.fsum(.9*max(-e,0)+.1*max(e,0) for e in values)/len(pool)}
        if math.fsum(weights):
            expected["daily_maximum_MAE"]=math.fsum(abs(e)*w for e,w in zip(values,weights))/math.fsum(weights)
        for field,value in expected.items():
            close(r[field],value)
    table=rows(OUT/"selection_metrics.csv")
    baseline=next(r for r in table if r["group"]=="B_score")
    for r in table:
        pool=groups[r["group"]]
        y=[float(x["increase"])>=93 for x in pool]
        scores=[float(x["score"]) for x in pool]
        close(r["AP"],ap(y,scores))
        monthly=[x for x in alarm_rows if x["group"]==r["group"] and x["cap"]=="0.01" and x["boundary_name"]=="q95"]
        close(r["weighted_monthly_AP"],math.fsum(float(x["AP"])*int(x["events"]) for x in monthly)/8)
        tp=sum(x["alarm_0.01"]=="True" and label for x,label in zip(pool,y))
        fp=sum(x["alarm_0.01"]=="True" and not label for x,label in zip(pool,y))
        close(r["TP"],tp);close(r["FP"],fp)
        close(r["recall"],tp/8);close(r["precision"],tp/(tp+fp) if tp+fp else 0)
        close(r["false_positive_fraction"],fp/2176)
        monthly_ok=all(int(x["TP"])>=int(next(b for b in alarm_rows if b["group"]=="B_score" and b["split"]==x["split"] and b["cap"]=="0.01" and b["boundary_name"]=="q95")["TP"]) for x in monthly)
        flags={"rank_passed":float(r["weighted_monthly_AP"])>=1.05*float(baseline["weighted_monthly_AP"]),
            "recall_passed":tp>=int(baseline["TP"]),"monthly_recall_passed":monthly_ok,"false_positive_passed":fp/2176<=.01+1e-12,
            "strict_gain_passed":tp>int(baseline["TP"]) or fp<int(baseline["FP"])}
        for field,value in flags.items():
            assert (r[field]=="True")==value
        assert (r["eligible"]=="True")== (r["group"] in ["Logistic_all","HGB_all","Q90_upper"] and all(flags.values()))
    possible=[r for r in table if r["eligible"]=="True"]
    order=["Logistic_all","HGB_all","Q90_upper"]
    if possible:
        best=max(float(r["weighted_monthly_AP"]) for r in possible)
        chosen=min([r for r in possible if float(r["weighted_monthly_AP"])>=best/1.01],key=lambda r:(order.index(r["group"]),-float(r["weighted_monthly_AP"]))) ["group"]
    else:
        chosen="B_score"
    selection=json.loads((OUT/"selection.json").read_text(encoding="utf-8"))
    assert selection["selected_risk_channel"]==chosen and selection["normal_forecast"]=="B"
    df=pd.read_csv(ROOT/"Modeling/tables/m04_rise/feature_frame.csv",encoding="utf-8-sig",float_precision="round_trip").set_index("timestamp")
    reloads=0
    for r in rows(OUT/"model_manifest.csv"):
        path=ROOT/r["file"]
        assert sha(path)==r["sha256"]
        model=joblib.load(path)
        pool=parts[r["split"],r["name"]]
        columns=run["features"]["quantile" if r["name"]=="Q90_upper" else "classifier"]
        inputs=df.loc[[x["timestamp"] for x in pool],columns]
        if r["name"]=="Q90_upper":
            actual=model.predict(inputs)
            expected=[float(x["upper_prediction"]) for x in pool]
            assert model.loss=="quantile" and model.quantile==.9
        else:
            actual=np.full(len(pool),model["constant_probability"]) if isinstance(model,dict) else model.predict_proba(inputs)[:,1]
            expected=[float(x["score"]) for x in pool]
            if r["name"]=="Logistic_all" and not isinstance(model,dict):
                start=min(x["timestamp"] for x in pool)
                training=df.loc[df.index<start,columns]
                assert np.allclose(model[0].mean_,training.mean().to_numpy(),atol=1e-9,rtol=1e-10)
        assert np.allclose(actual,expected,atol=1e-9,rtol=0)
        reloads+=1
    assert reloads==9 and sum(run["fits"].values())==378
    result={"status":"passed","checked_at":datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"),
        "run_sha256":sha(OUT/"run.json"),"script_sha256":sha(Path(__file__)),"prediction_rows":len(pred),"inner_rows":len(inner),
        "trials_independently_recomputed":checked_trials,"model_reload_checks":reloads,"policies_checked":len(policies),
        "alarm_rows":len(alarm_rows),"regression_rows":len(regression_rows),"normal_forecast_unchanged":True,
        "selected_risk_channel":chosen,"no_july_august_evaluation":True}
    (OUT/"independent_verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
