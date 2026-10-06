"""One cached negative-tail alarm calibration, preserving models and selection requirements."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from m01_prepare import ROOT,sha
from m04_rise_compare import now,read_csv
from m04_surge_diagnose import alarm_metrics,confusion

OUT=ROOT/"Modeling/tables/m04_surge_calibration"
PARENT=ROOT/"Modeling/tables/m04_surge_risk"
CONTRACT=ROOT/"Modeling/config/m04_surge_calibration_contract.json"


def negative_cutoff(labels,scores,cap):
    negative=np.asarray(scores)[~np.asarray(labels,dtype=bool)]
    ordered=np.sort(negative)[::-1]
    k=int(np.floor(cap*len(ordered)))
    threshold=np.nextafter(ordered[k],np.inf) if k<len(ordered) else np.nextafter(ordered[-1],-np.inf)
    return float(threshold),len(ordered),k


def main():
    assert sys.version_info[:2]==(3,13) and sys._is_gil_enabled()
    assert not (OUT/"run.json").exists(),"Preserve completed comparison."
    c=json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent_run=json.loads((PARENT/"run.json").read_text(encoding="utf-8"))
    verified=json.loads((PARENT/"independent_verification.json").read_text(encoding="utf-8"))
    assert verified["status"]=="passed" and verified["run_sha256"]==sha(PARENT/"run.json")==c["parent_run_sha256"]
    for name,digest in parent_run["outputs_sha256"].items():
        assert sha(PARENT/name)==digest,name
    p=read_csv(PARENT/"predictions.csv",["timestamp","date"])
    inner=read_csv(PARENT/"inner_predictions.csv",["timestamp","date"])
    risk_contract=json.loads((ROOT/"Modeling/config/m04_surge_risk_contract.json").read_text(encoding="utf-8"))
    policy_rows,alarm_rows=[] ,[]
    for (split,name),pool in p.groupby(["split","group"]):
        past=inner.loc[inner.split.eq(split)&inner.group.eq(name)]
        assert past.timestamp.max()<pool.timestamp.min()
        for cap in [.01,.05]:
            threshold,negative_count,k=negative_cutoff(past.increase.ge(93),past.score,cap)
            inside=confusion(past.increase.ge(93),past.score,threshold)
            assert inside["FP"]<=k and inside["false_positive_fraction"]<=cap+1e-12
            policy_rows.append({"split":split,"group":name,"cap":cap,"negative_count":negative_count,"budget_count":k,
                "inner_last":str(past.timestamp.max()),"outer_first":str(pool.timestamp.min()),**inside})
            p.loc[pool.index,"alarm_"+str(cap)]=pool.score.ge(threshold).to_numpy()
            for label,boundary in [("q95",93.),("q90",risk_contract["boundary_sensitivity"])]:
                alarm_rows.append({"split":split,"group":name,"cap":cap,"boundary_name":label,
                    **alarm_metrics(pool.increase.ge(boundary),pool.score,threshold)})
    alarm_frame=pd.DataFrame(alarm_rows)
    original=read_csv(PARENT/"selection_metrics.csv")
    table=[]
    for name,g in p.groupby("group"):
        row=original.loc[original.group.eq(name)].iloc[0].to_dict()
        labels=g.increase.ge(93).to_numpy()
        selected=g["alarm_0.01"].to_numpy(dtype=bool)
        tp=int((selected&labels).sum());fp=int((selected&~labels).sum())
        row.update({"TP":tp,"FP":fp,"recall":tp/int(labels.sum()),"precision":tp/(tp+fp) if tp+fp else 0.,"false_positive_fraction":fp/int((~labels).sum())})
        table.append(row)
    table=pd.DataFrame(table)
    reference=table.loc[table.group.eq("B_score")].iloc[0]
    for i,r in table.iterrows():
        monthly=alarm_frame.loc[alarm_frame.group.eq(r.group)&alarm_frame.cap.eq(.01)&alarm_frame.boundary_name.eq("q95")].set_index("split")
        baseline=alarm_frame.loc[alarm_frame.group.eq("B_score")&alarm_frame.cap.eq(.01)&alarm_frame.boundary_name.eq("q95")].set_index("split")
        flags={"rank_passed":bool(r.weighted_monthly_AP>=1.05*reference.weighted_monthly_AP),"recall_passed":bool(r.TP>=reference.TP),
            "monthly_recall_passed":bool((monthly.TP>=baseline.TP).all()),"false_positive_passed":bool(r.false_positive_fraction<=.01+1e-12),
            "strict_gain_passed":bool(r.TP>reference.TP or r.FP<reference.FP)}
        for name,value in flags.items():
            table.loc[i,name]=value
        table.loc[i,"eligible"]=r.group in ["Logistic_all","HGB_all","Q90_upper"] and all(flags.values())
    eligible=table.loc[table.eligible.astype(bool)]
    if len(eligible):
        near=eligible.loc[eligible.weighted_monthly_AP.ge(eligible.weighted_monthly_AP.max()/1.01)].copy()
        near["preference"]=near.group.map({name:i for i,name in enumerate(["Logistic_all","HGB_all","Q90_upper"])})
        chosen=near.sort_values(["preference","weighted_monthly_AP"],ascending=[True,False]).iloc[0].group
    else:
        chosen="B_score"
    selection={"selected_risk_channel":chosen,"normal_forecast":"B","eligible_methods":eligible.group.tolist(),
        "selection_gates_unchanged":True,"no_new_fitting":True,"development_only":True,"no_july_august_evaluation":True}
    OUT.mkdir(parents=True,exist_ok=True)
    for frame,name in [(p,"predictions.csv"),(pd.DataFrame(policy_rows),"alarm_policies.csv"),(alarm_frame,"alarm_metrics.csv"),(table,"selection_metrics.csv")]:
        frame.to_csv(OUT/name,index=False,encoding="utf-8-sig")
    (OUT/"selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
    run={"status":"completed","finished":now(),"script_sha256":sha(Path(__file__)),"contract_sha256":sha(CONTRACT),
        "inputs_sha256":{path.relative_to(ROOT).as_posix():sha(path) for path in [PARENT/"run.json",PARENT/"independent_verification.json",PARENT/"predictions.csv",PARENT/"inner_predictions.csv",PARENT/"selection_metrics.csv",ROOT/"Modeling/config/m04_surge_risk_contract.json"]},
        "outputs_sha256":{path.name:sha(path) for path in OUT.iterdir()},"prediction_rows":len(p),"normal_forecast_unchanged":True,
        "selection_gates_unchanged":True,"no_new_fitting":True,"no_july_august_evaluation":True}
    (OUT/"run.json").write_text(json.dumps(run,indent=2),encoding="utf-8")
    print(table[["group","AP","weighted_monthly_AP","TP","FP","recall","precision","false_positive_fraction","eligible"]].to_string(index=False),flush=True)
    print(json.dumps(selection),flush=True)


if __name__=="__main__":
    main()
