"""Independent redevelopment checks: raw timestamps, weighted errors, trials and saved model predictions."""
import argparse,csv,json,math,sys
from collections import defaultdict
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np,pandas as pd,joblib
from m01_prepare import ROOT,sha
from redevelopment_r01 import now
BASE=ROOT/"Modeling/tables/redevelopment"
def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def close(a,b):
    assert math.isclose(float(a),float(b),abs_tol=1e-8,rel_tol=1e-10),(a,b)
def avg(v,w):
    return math.fsum(x*y for x,y in zip(v,w))/math.fsum(w)
def readj(path):return json.loads(path.read_text(encoding="utf-8"))
def dump(v,path):path.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding="utf-8")
def rawdict():
    c=readj(ROOT/"Modeling/config/m01_contract.json")
    assert sha(ROOT/c["source"])==c["source_sha256"]
    out={}
    for r in rows(ROOT/c["source"]):
        d,h=int(r["날짜"]),int(r["시간"])
        if 20210101<=d<=20210831 and 0<=h<=23:
            t=datetime.strptime(str(d),"%Y%m%d")+timedelta(hours=h)
            vals=[float(r[k]) for k in ["15분","30분","45분","60분"]]
            out[t]={"slots":vals,"mean":sum(vals)/4,"maximum":max(vals),"last":vals[-1],"production":float(r["생산량"])}
    return out
def validframe(raw):
    f=pd.read_csv(ROOT/"Modeling/tables/m01/hourly_frame.csv",encoding="utf-8-sig",float_precision="round_trip",parse_dates=["timestamp"])
    f=f.loc[f.eligible_common]
    checked=0
    for r in f.itertuples():
        ts=r.timestamp.to_pydatetime();own=raw[ts]
        close(r.target_maximum,own["maximum"])
        for lag in [1,2,24,168]:
            prior=raw[ts-timedelta(hours=lag)]
            for col in ["mean","maximum","last","production"]:
                close(getattr(r,f"lag{lag}_{col}"),prior[col]);checked+=1
        close(r.hour_sin,math.sin(2*math.pi*ts.hour/24));close(r.hour_cos,math.cos(2*math.pi*ts.hour/24))
        close(r.weekend,int(ts.weekday()>=5))
        close(r.lag1_production_zero,int(raw[ts-timedelta(hours=1)]["production"]==0))
        for i in range(7):close(getattr(r,f"dow_{i}"),int(ts.weekday()==i))
    return f,checked
def hashes(d):
    run=readj(d/"run.json")
    assert run["status"]=="completed"
    for name,h in run["outputs_sha256"].items():assert sha(d/name)==h,(d,name)
    for name,h in run.get("inputs_sha256",{}).items():assert sha(ROOT/name)==h,name
    return run
def r01(raw):
    d=BASE/"r01";run=hashes(d);c=readj(ROOT/"Modeling/config/redevelopment_r01_contract.json")
    assert sha(ROOT/"Modeling/config/redevelopment_r01_contract.json")==run["contract_sha256"]
    for p,h in c["inputs_sha256"].items():assert sha(ROOT/p)==h
    f,checks=validframe(raw)
    pairs=rows(d/"paired.csv");assert len(pairs)==3528
    old={}
    for path in ["Modeling/tables/m03/ab/predictions.csv","Modeling/tables/m05/predictions.csv"]:
        for r in rows(ROOT/path):
            if r["group"] in ["A","B"] and r.get("target","target_maximum")=="target_maximum" and r.get("pool","common")=="common":
                old[(r["timestamp"],r["group"])]=r
    for r in pairs:
        ts=datetime.fromisoformat(r["timestamp"]);prior=raw[ts-timedelta(hours=1)];slots=prior["slots"]
        close(r["actual"],raw[ts]["maximum"]);close(r["prior_range"],max(slots)-min(slots));close(r["prior_slot_change"],slots[-1]-slots[0])
        close(r["last_gap"],max(slots)-slots[-1])
        state="zero" if max(slots)==0 else "low20_26" if 20<=math.floor(prior["mean"]+.5)<=26 else "positive_below20" if prior["mean"]<20 else "above26"
        assert r["prior_state"]==state
        assert r["prior_trend"]==("up" if slots[-1]>slots[0] else "down" if slots[-1]<slots[0] else "flat")
        for group in ["A","B"]:
            close(r[f"prediction_{group}"],old[(r["timestamp"],group)]["prediction"])
            e=float(r[f"prediction_{group}"])-float(r["actual"])
            for name,value in [("abs",abs(e)),("under",max(-e,0)),("over",max(e,0)),("bias",e)]:close(r[f"{name}_{group}"],value)
        close(r["delta_absolute_error"],float(r["abs_B"])-float(r["abs_A"]))
    thresholds={r["split"]:r for r in rows(d/"thresholds.csv")}
    for split,t in thresholds.items():
        train=f.loc[f.timestamp.le(t["train_end"])]
        assert len(train)==int(t["train_hours"])
        for field,prefix in [("lag1_mean","level"),("prior_range","range")]:
            vals=[]
            for ts in train.timestamp:
                prior=raw[ts.to_pydatetime()-timedelta(hours=1)]
                v=prior["mean"] if field=="lag1_mean" else max(prior["slots"])-min(prior["slots"])
                if v>0:vals.append(v)
            qs=np.quantile(vals,[1/3,2/3]);close(t[prefix+"_q33"],qs[0]);close(t[prefix+"_q67"],qs[1])
        gaps=[raw[ts.to_pydatetime()-timedelta(hours=1)]["maximum"]-raw[ts.to_pydatetime()-timedelta(hours=1)]["last"] for ts in train.timestamp]
        close(t["gap_median"],np.median([x for x in gaps if x>0]))
    for r in pairs:
        t=thresholds[r["split"]]
        for field,prefix in [("lag1_mean","level"),("prior_range","range")]:
            v=float(r[field]);label="zero" if v==0 else "lower" if v<=float(t[prefix+"_q33"]) else "middle" if v<=float(t[prefix+"_q67"]) else "upper"
            assert r[prefix+"_bin"]==label
        gap=float(r["last_gap"]);assert r["last_gap_bin"]==("at_maximum" if gap<=0 else "small_gap" if gap<=float(t["gap_median"]) else "large_gap")
    for m in rows(d/"conditions.csv"):
        selected=[r for r in pairs if (int(r["month"])<=6)==(m["period"]=="development") and (m["condition"]=="all" or r[m["condition"]]==m["value"])]
        w=[float(r["daily_maximum_weight"]) if m["scope"]=="peak" else 1. for r in selected]
        close(m["weight_sum"],sum(w))
        assert int(m["hours"])==sum(x>0 for x in w)
        for group in ["A","B"]:
            for metric in ["abs","under","over","bias"]:close(m[f"{metric}_{group}"],avg([float(r[f"{metric}_{group}"]) for r in selected],w))
        close(m["delta_MAE"],float(m["abs_B"])-float(m["abs_A"]))
    for m in rows(d/"leave_one_date_out.csv"):
        selected=[r for r in pairs if (int(r["month"])<=6)==(m["period"]=="development") and r["date"]!=m["excluded_date"]]
        w=[float(r["daily_maximum_weight"]) if m["scope"]=="peak" else 1. for r in selected]
        close(m["delta_MAE"],avg([float(r["delta_absolute_error"]) for r in selected],w))
    return {"pairs":len(pairs),"raw_lag_values":checks,"conditions":len(rows(d/"conditions.csv"))}
def stages(stage,raw):
    d=BASE/stage;run=hashes(d);c=readj(ROOT/"Modeling/config/redevelopment_r02_contract.json")
    for p,h in c["inputs_sha256"].items():assert sha(ROOT/p)==h,p
    assert run["script_sha256"]==sha(ROOT/"Modeling/scripts/redevelopment_compare.py")
    if stage=="r02":assert run["contract_sha256"]==sha(ROOT/"Modeling/config/redevelopment_r02_contract.json")
    f,checks=validframe(raw);idx=f.set_index("timestamp")
    p=rows(d/"predictions.csv");groups=defaultdict(list)
    for r in p:
        ts=datetime.fromisoformat(r["timestamp"]);close(r["actual"],raw[ts]["maximum"])
        close(r["lag1_last"],raw[ts-timedelta(hours=1)]["last"])
        close(r["lag1_maximum"],raw[ts-timedelta(hours=1)]["maximum"])
        e=float(r["prediction"])-float(r["actual"])
        for name,val in [("absolute_error",abs(e)),("under_amount",max(-e,0)),("over_amount",max(e,0)),("bias",e)]:close(r[name],val)
        groups[r["method"]].append(r)
    for name,g in groups.items():
        assert len({r["timestamp"] for r in g})==len(g)
        dates=defaultdict(list)
        for r in g:dates[r["date"]].append(r)
        for day,part in dates.items():
            hi=max(float(r["actual"]) for r in part);ties=sum(float(r["actual"])==hi for r in part)
            for r in part:close(r["daily_maximum_weight"],1/ties if len(part)==24 and float(r["actual"])==hi else 0)
    for m in rows(d/"metrics.csv"):
        g=groups[m["method"]];period=m["period"];scope=m["scope"]
        if period!="pooled":
            if period.startswith("month:"):g=[r for r in g if r["month"]==period.split(":")[1]]
            elif period.startswith("train_profile_overlap:"):g=[r for r in g if r["train_profile_overlap"]==period.split(":")[1]]
            elif period=="surge93":g=[r for r in g if float(r["actual"])-float(r["lag1_maximum"])>=93]
            elif period=="zero_actual":g=[r for r in g if float(r["actual"])==0]
            elif period=="prior_zero":g=[r for r in g if float(r["lag1_maximum"])==0]
            else:g=[r for r in g if r["split"]==period]
        assert len(g)==int(m["hours"])
        w=[float(r["daily_maximum_weight"]) if scope=="peak" else float(r["profile_weight"]) if scope=="profile_reweighted" else 1. for r in g]
        close(m["weight_sum"],sum(w))
        err=[float(r["prediction"])-float(r["actual"]) for r in g]
        for name,v in [("MAE",[abs(x) for x in err]),("mean_under",[max(-x,0) for x in err]),("mean_over",[max(x,0) for x in err]),("bias",err),
                       ("mean_actual",[float(r["actual"]) for r in g])]:close(m[name],avg(v,w))
    conf=readj(d/"configurations.json")
    for mr in rows(d/"model_manifest.csv"):
        path=ROOT/mr["file"];assert sha(path)==mr["sha256"]
        mm=joblib.load(path);params=json.loads(mr["parameters"])
        for k,v in params.items():assert mm.get_params()[k]==v
        cols=c["features"][mr["input"]];assert list(mm.feature_names_in_)==cols
        records=[r for r in groups[mr["method"]] if r["split"]==mr["split"]]
        ev=idx.loc[pd.to_datetime([r["timestamp"] for r in records])]
        got=mm.predict(ev[cols])+(ev.lag1_last.to_numpy() if mr["variant"]=="delta" else 0)
        assert np.allclose(got,[float(r["prediction"]) for r in records],rtol=0,atol=1e-8)
        eligible=f.loc[f.timestamp.le(mr["train_end"])]
        assert len(eligible)==int(mr["train_hours"]) and eligible.timestamp.max()<ev.index.min()
    if stage=="r02":
        trials=rows(d/"trials.csv");folds=rows(d/"trial_fold_scores.csv");assert len(trials)==360
        studies=defaultdict(list)
        for t in trials:
            key=(t["split"],t["family"],t["input"]);studies[key].append(t)
            selected=[r for r in folds if (r["split"],r["family"],r["input"],r["trial"])==(*key,t["number"])]
            close(t["overall_MAE"],avg([float(r["overall_MAE"]) for r in selected],[int(r["hours"]) for r in selected]))
            close(t["peak_MAE"],avg([float(r["peak_MAE"]) for r in selected],[float(r["peak_weight_sum"]) for r in selected]))
            close(t["balanced_score"],max(float(t["overall_MAE"])/float(t["baseline_overall"]),float(t["peak_MAE"])/float(t["baseline_peak"])))
        for key,triallist in studies.items():
            assert len(triallist)==30 and {int(t["number"]) for t in triallist}==set(range(30))
            best=min(triallist,key=lambda t:(float(t["balanced_score"]),int(t["number"])))
            assert best["selected"]=="True"
        for a in rows(d/"time_audit.csv"):
            assert a["train_end"]<a["validation_start"]<=a["validation_end"]
            parent=readj(ROOT/"Modeling/config/m01_contract.json")
            outer=next(s for s in parent["splits"] if s["name"]==a["split"])
            assert pd.Timestamp(a["validation_end"])<pd.Timestamp(outer["eval_start"])
        inner=rows(d/"selected_inner_predictions.csv")
        by=defaultdict(list)
        for r in inner:by[(r["split"],r["method"])].append(r)
        for cf in conf:
            if cf["variant"]!="tuned":continue
            g=by[(cf["split"],cf["method"])];err=[abs(float(r["prediction"])-float(r["actual"])) for r in g]
            close(cf["inner_objectives"][0],math.fsum(err)/len(err));close(cf["inner_objectives"][1],avg(err,[float(r["daily_maximum_weight"]) for r in g]))
    # Independently rederive declaration gates from verified metric arithmetic and date omissions.
    ms={(r["method"],r["period"],r["scope"]):r for r in rows(d/"metrics.csv")}
    selection=readj(d/"selection.json")
    leave=rows(d/"leave_one_date_out.csv")
    for candidate in selection["candidate_gates"]:
        name=candidate["method"];flags={};a=ms[(name,"pooled","all")];pk=ms[(name,"pooled","peak")]
        for ref in c["gates"]["comparison"]:
            ra=ms[(ref,"pooled","all")];rp=ms[(ref,"pooled","peak")]
            flags[ref+"_overall"]=float(a["MAE"])<=float(ra["MAE"])*1.01
            flags[ref+"_peak"]=float(pk["MAE"])<=float(rp["MAE"])*.95
            flags[ref+"_under"]=float(pk["mean_under"])<=float(rp["mean_under"])*.95
            flags[ref+"_over"]=float(pk["mean_over"])-float(rp["mean_over"])<=float(rp["MAE"])*.05
            lhs={r["timestamp"]:r for r in groups[name]};rhs={r["timestamp"]:r for r in groups[ref]}
            diffs=[]
            for day in {r["date"] for r in groups[name]}:
                g=[r for r in groups[name] if r["date"]!=day];w=[float(r["daily_maximum_weight"]) for r in g]
                value=avg([float(r["absolute_error"])-float(rhs[r["timestamp"]]["absolute_error"]) for r in g],w)
                diffs.append(value)
                stored=next(r for r in leave if r["method"]==name and r["reference"]==ref and r["excluded_date"]==day)
                close(stored["peak_delta_MAE"],value)
            flags[ref+"_leave_one_date"]=max(diffs)<0
        flags["monthly"]=all(float(ms[(name,s,sc)]["MAE"])<=float(ms[("fixed_HGB_B",s,sc)]["MAE"])*1.05 for s in {f"month:{r[chr(109)+chr(111)+chr(110)+chr(116)+chr(104)]}" for r in groups[name]} for sc in ["all","peak"])
        assert flags==candidate["gates"] and candidate["eligible"]==all(flags.values())
        close(candidate["balanced_score"],max(float(a["MAE"])/float(ms[("lag1","pooled","all")]["MAE"]),float(pk["MAE"])/float(ms[("lag1","pooled","peak")]["MAE"])))
    eligible=[r for r in selection["candidate_gates"] if r["eligible"]]
    selected=min(eligible,key=lambda r:(r["balanced_score"],r["method"]))["method"] if eligible else None
    assert selection["selected"]==selected
    return {"prediction_rows":len(p),"raw_lag_values":checks,"models_reloaded":len(rows(d/"model_manifest.csv")),
            "metrics":len(rows(d/"metrics.csv")),"trials":360 if stage=="r02" else 0,"selection":selected}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=["r01","r02","r03","r04"],required=True);stage=ap.parse_args().stage
    assert sys.version_info[:2]==(3,13) and sys._is_gil_enabled()
    raw=rawdict();d=BASE/stage;run=readj(d/"run.json")
    if run["status"]=="skipped":
        dump({"status":"passed","scope":"recorded skip; no prediction result","run_sha256":sha(d/"run.json"),"script_sha256":sha(Path(__file__))},d/"independent_verification.json")
        print("Verified skip",stage);return
    stats=r01(raw) if stage=="r01" else stages(stage,raw)
    dump({"status":"passed","verified_at":now(),"run_sha256":sha(d/"run.json"),"script_sha256":sha(Path(__file__)),**stats},d/"independent_verification.json")
    print(json.dumps({"status":"passed","stage":stage,**stats}),flush=True)
if __name__=="__main__":main()
