"""Independent peak-weight arithmetic, selection and weighted-fit reproduction."""
import argparse,json,math,sys
from pathlib import Path
from collections import defaultdict
import numpy as np,pandas as pd,joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from m01_prepare import ROOT,sha
import redevelopment_verify as v
from redevelopment_r01 import now
BASE=ROOT/"Modeling/tables/redevelopment"
CF=ROOT/"Modeling/config/redevelopment_peak_contract.json"
def independent_weight(frame,raw,lam):
    dates=defaultdict(list)
    for ts in frame.timestamp:
        t=ts.to_pydatetime();dates[t.date()].append(t)
    weights={}
    for day,times in dates.items():
        hi=max(raw[t]["maximum"] for t in times);ties=sum(raw[t]["maximum"]==hi for t in times)
        for t in times:weights[t]=1+(lam-1)/ties if len(times)==24 and raw[t]["maximum"]==hi else 1.
    return np.array([weights[ts.to_pydatetime()] for ts in frame.timestamp])
def metric(g):
    e=[float(r["prediction"])-float(r["actual"]) for r in g];w=[float(r["daily_maximum_weight"]) for r in g]
    return {"all":math.fsum(abs(x) for x in e)/len(e),"peak":v.avg([abs(x) for x in e],w),
            "under":v.avg([max(-x,0) for x in e],w),"over":v.avg([max(x,0) for x in e],w)}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=["r03b","r04"],required=True);stage=ap.parse_args().stage
    assert sys.version_info[:2]==(3,13) and sys._is_gil_enabled()
    d=BASE/stage;run=v.readj(d/"run.json");c=v.readj(CF)
    assert c["entrypoint_sha256"]==sha(ROOT/"Modeling/scripts/redevelopment_peak.py")
    assert c["helper_sha256"]==sha(ROOT/"Modeling/scripts/redevelopment_compare.py")
    assert c["source_r02"]==sha(BASE/"r02/run.json") and c["source_r03"]==sha(BASE/"r03/run.json")
    if run["status"]=="skipped":
        assert v.readj(BASE/"r03b/selection.json")["selected"] is None
        stats={"scope":"verified no eligible candidate; no later evaluation"}
    else:
        assert run["entrypoint_sha256"]==sha(ROOT/"Modeling/scripts/redevelopment_peak.py")
        if stage=="r03b":assert run["contract_sha256"]==sha(CF)
        raw=v.rawdict();stats=v.stages(stage,raw);f,checks=v.validframe(raw)
        if stage=="r03b":
            conf=v.readj(d/"configurations.json");inner=v.rows(d/"inner_predictions.csv");trials=v.rows(d/"weight_trials.csv")
            assert len(trials)==21
            for split in ["dev_apr","dev_may","dev_jun"]:
                by=defaultdict(list)
                for r in inner:
                    if r["outer"]==split:
                        ts=pd.Timestamp(r["timestamp"]).to_pydatetime()
                        v.close(r["actual"],raw[ts]["maximum"])
                        by[r["method"]].append(r)
                bm=metric(by["fixed_HGB_B"]);eligible=[]
                for tr in [r for r in trials if r["outer"]==split]:
                    lam=float(tr["lambda"]);name="fixed_HGB_B" if lam==1 else f"weight_{lam:g}"
                    cm=metric(by[name]);months={r["month"] for r in by[name]}
                    monthly=all(metric([r for r in by[name] if r["month"]==month])[scope]<=metric([r for r in by["fixed_HGB_B"] if r["month"]==month])[scope]*1.05
                                for month in months for scope in ["all","peak"])
                    feasible=lam==1 or (cm["all"]<=bm["all"]*1.01 and cm["over"]-bm["over"]<=bm["peak"]*.05 and monthly)
                    objective=max(cm["peak"]/bm["peak"],cm["under"]/bm["under"])
                    for col,expected in [("overall_MAE",cm["all"]),("peak_MAE",cm["peak"]),("peak_under",cm["under"]),("peak_over",cm["over"]),
                                         ("base_overall",bm["all"]),("base_peak",bm["peak"]),("base_under",bm["under"]),("base_over",bm["over"]),("objective",objective)]:
                        v.close(tr[col],expected)
                    assert (tr["monthly_feasible"]=="True")==monthly and (tr["feasible"]=="True")==feasible
                    if feasible:eligible.append((objective,lam))
                best=min(eligible)[1];cf=next(x for x in conf if x["split"]==split)
                v.close(cf["training_lambda"],best)
            for a in v.rows(d/"training_weight_audit.csv"):
                train=f.loc[f.timestamp.le(a["train_end"])]
                assert len(train)==int(a["train_hours"]) and pd.Timestamp(a["train_end"])<pd.Timestamp(a["validation_start"])
                w=independent_weight(train,raw,float(a["lambda"]))
                v.close(a["weight_sum"],sum(w))
                if float(a["lambda"])!=1:v.close(a["complete_training_dates_weight"],(sum(w)-len(w))/(float(a["lambda"])-1))
        # Refit each chosen outer model with independently reconstructed weights.
        predictions=v.rows(d/"predictions.csv");c2=v.readj(ROOT/"Modeling/config/redevelopment_r02_contract.json")
        for mr in v.rows(d/"model_manifest.csv"):
            train=f.loc[f.timestamp.le(mr["train_end"])]
            lam=float(mr["training_lambda"]);w=independent_weight(train,raw,lam);v.close(mr["training_weight_sum"],sum(w))
            cols=c2["features"]["B"];params=json.loads(mr["parameters"])
            m=HistGradientBoostingRegressor(**params,early_stopping=False,random_state=42)
            m.fit(train[cols],train.target_maximum,sample_weight=w)
            rec=[r for r in predictions if r["method"]==mr["method"] and r["split"]==mr["split"]]
            ev=f.set_index("timestamp").loc[pd.to_datetime([r["timestamp"] for r in rec])]
            got=m.predict(ev[cols]);assert np.allclose(got,[float(r["prediction"]) for r in rec],rtol=0,atol=1e-8)
        stats["independent_weighted_refits"]=len(v.rows(d/"model_manifest.csv"))
    out={"status":"passed","verified_at":now(),"run_sha256":sha(d/"run.json"),"script_sha256":sha(Path(__file__)),**stats}
    (d/"independent_verification.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out),flush=True)
if __name__=="__main__":main()
