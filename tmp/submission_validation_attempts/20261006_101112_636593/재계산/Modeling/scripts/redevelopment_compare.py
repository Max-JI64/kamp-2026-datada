"""R02-R04: bounded model/input comparison and conditional delta target."""
import argparse, json, os, sys, time, importlib.metadata
from pathlib import Path
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ.setdefault(k,"4")
import numpy as np
import pandas as pd
import optuna, joblib
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from m01_prepare import ROOT, feature_columns, read_contract, sha
from redevelopment_r01 import read, dump, now

BASE=ROOT/"Modeling/tables/redevelopment"
CFILE=ROOT/"Modeling/config/redevelopment_r02_contract.json"
MODEL=ROOT/"Modeling/models/redevelopment"
M1=read_contract()
TARGET="target_maximum"
HGB={"max_leaf_nodes":15,"max_iter":150,"learning_rate":.05,"min_samples_leaf":20,"l2_regularization":1.}
FITS=0
def save(df,d,name):
    df.to_csv(d/name,index=False,encoding="utf-8-sig")
def pw(g):
    dates=g.timestamp.dt.normalize()
    full=g.groupby(dates).timestamp.transform("count").eq(24)
    flag=g[TARGET].eq(g.groupby(dates)[TARGET].transform("max")) & full
    ties=flag.groupby(dates).transform("sum").clip(lower=1)
    return np.where(flag,1/ties,0).astype(float)
def model(family,params):
    if family=="HGB":
        return HistGradientBoostingRegressor(**params,early_stopping=False,random_state=42)
    return ExtraTreesRegressor(**params,random_state=42,n_jobs=4,bootstrap=False)
def fit(family,params,tr,cols,delta=False):
    global FITS
    m=model(family,params)
    y=tr[TARGET]-(tr.lag1_last if delta else 0)
    m.fit(tr[cols],y);FITS+=1
    return m
def predict(m,ev,cols,delta=False):
    return m.predict(ev[cols])+(ev.lag1_last.to_numpy() if delta else 0)
def obj(y,p,w):
    e=abs(np.asarray(p)-np.asarray(y))
    return float(e.mean()),float(np.average(e,weights=w))
def score(v,d):
    return max(v[0]/d[0],v[1]/d[1])
def suggest(t,family,spaces):
    params={}
    for name,spec in spaces[family].items():
        if name in ["max_leaf_nodes","max_depth"]:params[name]=t.suggest_categorical(name,spec)
        elif name in ["max_iter","n_estimators","min_samples_leaf"]:
            params[name]=t.suggest_int(name,spec[0],spec[1],step=spec[2] if len(spec)==3 else 1)
        else:params[name]=t.suggest_float(name,spec[0],spec[1],log=len(spec)==3 and spec[2]=="log")
    return params
def current_inputs():
    paths=["Modeling/config/m01_contract.json","Modeling/tables/m01/hourly_frame.csv",
           "Modeling/tables/m01/evaluation_diagnostics.csv","Modeling/config/m02_rerun_contract.json",
           "Modeling/tables/m02_rerun/selected_configurations.json","Modeling/tables/m02_rerun/model_manifest.csv",
           "Modeling/tables/m03/ab/model_manifest.csv","Modeling/tables/redevelopment/r01/run.json",
           "Modeling/tables/redevelopment/r01/conditions.csv","Modeling/tables/redevelopment/r01/independent_verification.json",
           "Modeling/scripts/redevelopment_compare.py"]
    return {p:sha(ROOT/p) for p in paths}
def freeze():
    assert not CFILE.exists()
    ver=json.loads((BASE/"r01/independent_verification.json").read_text(encoding="utf-8"))
    assert ver["status"]=="passed"
    parent=json.loads((ROOT/"Modeling/config/m02_rerun_contract.json").read_text(encoding="utf-8"))
    diag=read(BASE/"r01/paired.csv")
    bias=diag.groupby(["prior_state",diag.month.le(6).map({True:"dev",False:"later"})])[["bias_A","bias_B"]].mean()
    c={"version":"redevelopment-r02-v1","recorded_at":now(),"reason":"R01 high-prior and upward-slot peak effects reversed in later period; compare model/input before any conditional gate.",
       "inputs_sha256":current_inputs(),"features":{g:feature_columns(g,M1) for g in ["A","B"]},
       "families":["HGB","ExtraTrees"],"inputs":["A","B"],"outer":["dev_apr","dev_may","dev_jun"],
       "inner_months":{"dev_apr":[2,3],"dev_may":[2,3,4],"dev_jun":[2,3,4,5]},
       "fixed_HGB":HGB,"fixed_ExtraTrees":"existing per-outer A-selected parameters; same settings applied to A and B",
       "search_spaces":{f:parent["search_spaces"][f] for f in ["HGB","ExtraTrees"]},
       "trials_per_study":30,"max_trials":360,"sampler_seed":42,"startup":10,"extension":False,
       "objective":"two MAEs, choose minimum max(normalized by inner lag1 baseline); tie trial number",
       "postprocessing":"none for direct and delta; negative outputs retained and diagnosed",
       "gates":{"peak_ratio_max":.95,"peak_under_ratio_max":.95,"overall_ratio_max":1.01,
                "peak_over_increase_as_reference_peak_MAE":.05,"month_overall_ratio_max":1.05,"month_peak_ratio_max":1.05,
                "comparison":["fixed_HGB_A","fixed_HGB_B"],"jackknife_peak_improvement":True},
       "R03":"only if no R02 candidate eligible and R01 level-specific bias remains; choose best B-input family/variant by development balanced score; compare delta with identical B direct model and parameters per outer",
       "R03_level_bias_present":bool(bias.loc[("above26","later"),"bias_B"]<0 and bias.loc[("low20_26","later"),"bias_B"]>0),
       "R04":"only an eligible development method; latest chronological outer configuration, no July refit; no test-based reselection",
       "scope":"development redesign informed by already observed July-August; no new independent test"}
    dump(c,CFILE);print("R02 contract recorded; 360-trial cap",flush=True)
def data():
    f=read(ROOT/"Modeling/tables/m01/hourly_frame.csv")
    return f.loc[f.eligible_common].copy()
def splits():
    return {s["name"]:s for s in M1["splits"]}
def evmeta(ev,split):
    diag=read(ROOT/"Modeling/tables/m01/evaluation_diagnostics.csv")
    d=diag.loc[diag.split.eq(split)&diag.pool.eq("common")]
    p=ev[["timestamp","date","month",TARGET,"lag1_maximum","lag1_last","diag_target_profile"]].rename(columns={TARGET:"actual"})
    p=p.merge(d[["timestamp","profile_weight","train_profile_overlap","daily_maximum_weight"]],on="timestamp",validate="one_to_one")
    assert len(p)==len(ev) and np.allclose(p.daily_maximum_weight,pw(ev),rtol=0,atol=1e-12)
    p["split"]=split
    return p
def record(base,name,family,group,variant,p):
    r=base.copy();r["method"]=name;r["family"]=family;r["input"]=group;r["variant"]=variant;r["prediction"]=p
    e=r.prediction-r.actual
    r["absolute_error"]=abs(e);r["under_amount"]=(-e).clip(lower=0);r["over_amount"]=e.clip(lower=0);r["bias"]=e
    return r
def met(g,w=None):
    w=np.ones(len(g)) if w is None else np.asarray(w)
    return {"hours":len(g),"weight_sum":float(w.sum()),"MAE":float(np.average(abs(g.prediction-g.actual),weights=w)),
            "mean_under":float(np.average(np.maximum(g.actual-g.prediction,0),weights=w)),
            "mean_over":float(np.average(np.maximum(g.prediction-g.actual,0),weights=w)),
            "bias":float(np.average(g.prediction-g.actual,weights=w)),
            "mean_actual":float(np.average(g.actual,weights=w)),"negative_predictions":int((g.prediction<0).sum())}
def metrics(p):
    rows=[]
    for method,g in p.groupby("method"):
        for period,part in [("pooled",g)]+[(s,h) for s,h in g.groupby("split")]:
            for scope in ["all","peak","profile_reweighted"]:
                weights=part.daily_maximum_weight if scope=="peak" else part.profile_weight if scope=="profile_reweighted" else None
                if weights is not None and weights.sum()==0:continue
                rows.append({"method":method,"period":period,"scope":scope,**met(part,weights)})
        for field in ["month","train_profile_overlap"]:
            for value,part in g.groupby(field):
                rows.append({"method":method,"period":f"{field}:{value}","scope":"all",**met(part)})
                if field=="month" and part.daily_maximum_weight.sum()>0:
                    rows.append({"method":method,"period":f"{field}:{value}","scope":"peak",**met(part,part.daily_maximum_weight)})
        delta=g.actual-g.lag1_maximum
        for label,mask in [("surge93",delta.ge(93)),("zero_actual",g.actual.eq(0)),("prior_zero",g.lag1_maximum.eq(0))]:
            if mask.any():rows.append({"method":method,"period":label,"scope":"all",**met(g.loc[mask])})
    return pd.DataFrame(rows)
def assess(p,c):
    tables=metrics(p)
    idx=tables.set_index(["method","period","scope"])
    lag=idx.loc[("lag1","pooled","all"),"MAE"],idx.loc[("lag1","pooled","peak"),"MAE"]
    rows=[];leave=[]
    for method,g in p.groupby("method"):
        overall=idx.loc[(method,"pooled","all")];peak=idx.loc[(method,"pooled","peak")]
        flags={}
        for ref in c["gates"]["comparison"]:
            ra=idx.loc[(ref,"pooled","all")];rp=idx.loc[(ref,"pooled","peak")]
            flags[ref+"_overall"]=bool(overall.MAE<=ra.MAE*c["gates"]["overall_ratio_max"])
            flags[ref+"_peak"]=bool(peak.MAE<=rp.MAE*c["gates"]["peak_ratio_max"])
            flags[ref+"_under"]=bool(peak.mean_under<=rp.mean_under*c["gates"]["peak_under_ratio_max"])
            flags[ref+"_over"]=bool(peak.mean_over-rp.mean_over<=rp.MAE*c["gates"]["peak_over_increase_as_reference_peak_MAE"])
            paired=g[["timestamp","date","absolute_error","daily_maximum_weight"]].merge(
                p.loc[p.method.eq(ref),["timestamp","absolute_error"]],on="timestamp",suffixes=("_candidate","_reference"),validate="one_to_one")
            differences=[]
            for day in sorted(paired.date.unique()):
                rem=paired.loc[paired.date.ne(day)]
                value=float(np.average(rem.absolute_error_candidate-rem.absolute_error_reference,weights=rem.daily_maximum_weight))
                leave.append({"method":method,"reference":ref,"excluded_date":day,"peak_delta_MAE":value});differences.append(value)
            flags[ref+"_leave_one_date"]=bool(max(differences)<0)
        mb=[]
        for s in [f"month:{x}" for x in sorted(g.month.unique())]:
            for scope in ["all","peak"]:
                ratio=idx.loc[(method,s,scope),"MAE"]/idx.loc[("fixed_HGB_B",s,scope),"MAE"]
                mb.append(ratio<=c["gates"]["month_overall_ratio_max" if scope=="all" else "month_peak_ratio_max"])
        flags["monthly"]=bool(all(mb))
        rows.append({"method":method,"overall_MAE":float(overall.MAE),"peak_MAE":float(peak.MAE),
                     "peak_under":float(peak.mean_under),"peak_over":float(peak.mean_over),
                     "balanced_score":score((overall.MAE,peak.MAE),lag),"eligible":all(flags.values()),
                     "gates":flags})
    table=pd.DataFrame([{k:v for k,v in r.items() if k!="gates"} for r in rows])
    eligible=table.loc[table.eligible]
    preferred=lambda df:df.sort_values(["balanced_score","method"]).iloc[0].method
    chosen=preferred(eligible) if len(eligible) else None
    return tables,table,{"selected":chosen,"candidate_gates":rows,"no_eligible_candidate":chosen is None},pd.DataFrame(leave)
def manifest_row(m,path,split,name,family,group,variant,params,tr,source):
    return {"split":split,"method":name,"family":family,"input":group,"variant":variant,"file":str(path.relative_to(ROOT)),
            "sha256":sha(path),"parameters":json.dumps(params),"train_start":str(tr.timestamp.min()),"train_end":str(tr.timestamp.max()),
            "train_hours":len(tr),"source":source,"reload_checked":True}
def check_inputs(c):
    for p,h in c["inputs_sha256"].items():assert sha(ROOT/p)==h,p
def compare():
    c=json.loads(CFILE.read_text(encoding="utf-8"));check_inputs(c)
    d=BASE/"r02";d.mkdir(parents=True,exist_ok=True);MODEL.mkdir(parents=True,exist_ok=True)
    assert not (d/"run.json").exists()
    dump({"status":"running","started":now(),"contract_sha256":sha(CFILE),"script_sha256":sha(Path(__file__))},d/"run.json")
    f=data();oldconfigs=json.loads((ROOT/"Modeling/tables/m02_rerun/selected_configurations.json").read_text(encoding="utf-8"))
    oldet=pd.read_csv(ROOT/"Modeling/tables/m02_rerun/model_manifest.csv",encoding="utf-8-sig")
    oldh=pd.read_csv(ROOT/"Modeling/tables/m03/ab/model_manifest.csv",encoding="utf-8-sig")
    predictions=[];man=[];configs=[];trials=[];foldscores=[];innerrecords=[];audit=[]
    start=time.perf_counter()
    for split in c["outer"]:
        s=splits()[split];tr=f.loc[f.timestamp.le(s["train_end"])];ev=f.loc[f.timestamp.between(s["eval_start"],s["eval_end"])]
        base=evmeta(ev,split)
        predictions.append(record(base,"lag1","baseline","none","baseline",ev.lag1_maximum.to_numpy()))
        folds=[]
        for month in c["inner_months"][split]:
            bound=pd.Timestamp(2021,month,1);a=tr.loc[tr.timestamp.lt(bound)];b=tr.loc[tr.timestamp.ge(bound)&tr.timestamp.lt(bound+pd.offsets.MonthBegin(1))]
            assert a.timestamp.max()<b.timestamp.min()<ev.timestamp.min()
            folds.append((month,a,b,pw(b)))
            audit.append({"split":split,"inner_month":month,"train_start":str(a.timestamp.min()),"train_end":str(a.timestamp.max()),
                          "validation_start":str(b.timestamp.min()),"validation_end":str(b.timestamp.max()),"train_hours":len(a),"validation_hours":len(b)})
        ys=np.concatenate([b[TARGET] for _,_,b,_ in folds]);ws=np.concatenate([w for _,_,_,w in folds])
        denomin=obj(ys,np.concatenate([b.lag1_maximum for _,_,b,_ in folds]),ws)
        for family in c["families"]:
            frozen=HGB if family=="HGB" else next(x["parameters"] for x in oldconfigs if x["split"]==split and x["model"]=="ExtraTrees")
            for group in c["inputs"]:
                cols=c["features"][group];name=f"fixed_{family}_{group}"
                if family=="HGB":
                    mr=oldh.loc[oldh.split.eq(split)&oldh.target.eq(TARGET)&oldh.group.eq(group)].iloc[0]
                elif group=="A":
                    mr=oldet.loc[oldet.split.eq(split)&oldet.model.eq("ExtraTrees")].iloc[0]
                else:mr=None
                if mr is not None:
                    path=ROOT/mr.file;assert sha(path)==mr.sha256;m=joblib.load(path);source="parent_reload"
                    for k,v in frozen.items():assert m.get_params()[k]==v
                else:
                    m=fit(family,frozen,tr,cols);path=MODEL/f"{split}_{name}.joblib";joblib.dump(m,path,compress=3);source="new_fit"
                pred=predict(m,ev,cols);assert np.allclose(pred,predict(joblib.load(path),ev,cols),rtol=0,atol=1e-9)
                predictions.append(record(base,name,family,group,"fixed",pred));man.append(manifest_row(m,path,split,name,family,group,"fixed",frozen,tr,source))
                configs.append({"split":split,"method":name,"family":family,"input":group,"variant":"fixed","parameters":frozen})
                cache={}
                def objective(t):
                    params=suggest(t,family,c["search_spaces"]);parts=[]
                    for month,a,b,w in folds:
                        mm=fit(family,params,a,cols);pp=predict(mm,b,cols);o=obj(b[TARGET],pp,w)
                        foldscores.append({"split":split,"family":family,"input":group,"trial":t.number,"month":month,
                                           "hours":len(b),"peak_weight_sum":float(w.sum()),"overall_MAE":o[0],"peak_MAE":o[1]})
                        parts.append(pp)
                    cache[t.number]=np.concatenate(parts)
                    return obj(ys,cache[t.number],ws)
                study=optuna.create_study(directions=["minimize","minimize"],sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=10),
                                         pruner=optuna.pruners.NopPruner())
                if family=="HGB":study.enqueue_trial(HGB)
                study.optimize(objective,n_trials=30,n_jobs=1)
                selected=min(study.best_trials,key=lambda t:(score(t.values,denomin),t.number))
                params=selected.params;tname=f"tuned_{family}_{group}"
                mm=fit(family,params,tr,cols);path=MODEL/f"{split}_{tname}.joblib";joblib.dump(mm,path,compress=3)
                pred=predict(mm,ev,cols);assert np.allclose(pred,predict(joblib.load(path),ev,cols),rtol=0,atol=1e-9)
                predictions.append(record(base,tname,family,group,"tuned",pred));man.append(manifest_row(mm,path,split,tname,family,group,"tuned",params,tr,"new_fit"))
                configs.append({"split":split,"method":tname,"family":family,"input":group,"variant":"tuned","parameters":params,
                                "selected_trial":selected.number,"inner_objectives":list(selected.values),"denominator":list(denomin),"trials":30})
                times=np.concatenate([b.timestamp.to_numpy() for _,_,b,_ in folds])
                innerrecords.append(pd.DataFrame({"split":split,"method":tname,"timestamp":times,"actual":ys,"prediction":cache[selected.number],
                                                  "daily_maximum_weight":ws,"lag1_prediction":np.concatenate([b.lag1_maximum for _,_,b,_ in folds])}))
                for t in study.trials:
                    trials.append({"split":split,"family":family,"input":group,"number":t.number,"state":t.state.name,
                                   "overall_MAE":t.values[0],"peak_MAE":t.values[1],"baseline_overall":denomin[0],"baseline_peak":denomin[1],
                                   "balanced_score":score(t.values,denomin),"parameters":json.dumps(t.params),"selected":t.number==selected.number})
                save(pd.DataFrame(trials),d,"trials.csv");save(pd.DataFrame(foldscores),d,"trial_fold_scores.csv");dump(configs,d/"configurations.json")
                print(f"{split} {tname}:30 trials, trial {selected.number}, MAE/peak {obj(ev[TARGET],pred,pw(ev))}; fits={FITS}",flush=True)
    p=pd.concat(predictions,ignore_index=True)
    save(p,d,"predictions.csv");save(pd.DataFrame(man),d,"model_manifest.csv");save(pd.DataFrame(audit),d,"time_audit.csv");save(pd.concat(innerrecords),d,"selected_inner_predictions.csv")
    tables,table,selection,leave=assess(p,c)
    save(tables,d,"metrics.csv");save(table,d,"comparison.csv");save(leave,d,"leave_one_date_out.csv");dump(selection,d/"selection.json")
    dump({"status":"completed","completed_at":now(),"seconds":time.perf_counter()-start,"new_fits":FITS,"new_trials":len(trials),
          "contract_sha256":sha(CFILE),"script_sha256":sha(Path(__file__)),"inputs_sha256":c["inputs_sha256"],
          "outputs_sha256":{p.name:sha(p) for p in d.iterdir() if p.suffix in [".csv",".json"] and p.name!="run.json"}},d/"run.json")
    print(table.to_string(index=False),flush=True)
def delta_stage():
    c=json.loads(CFILE.read_text(encoding="utf-8"));check_inputs(c);d=BASE/"r03";d.mkdir(parents=True,exist_ok=True)
    assert not (d/"run.json").exists()
    r2=BASE/"r02";sel=json.loads((r2/"selection.json").read_text(encoding="utf-8"))
    assert json.loads((r2/"independent_verification.json").read_text(encoding="utf-8"))["status"]=="passed"
    if sel["selected"] is not None or not c["R03_level_bias_present"]:
        dump({"status":"skipped","reason":"R02 has eligible candidate or level-bias hypothesis unsupported","recorded_at":now()},d/"run.json");print("R03 skipped",flush=True);return
    table=pd.read_csv(r2/"comparison.csv",encoding="utf-8-sig")
    b=table.loc[table.method.str.endswith("_B")].sort_values(["balanced_score","method"]).iloc[0].method
    conf=json.loads((r2/"configurations.json").read_text(encoding="utf-8"))
    dump({"recorded_at":now(),"direct_method":b,"hypothesis":"origin-anchored change may reduce level bias; same direct B parameters, no Optuna",
          "r02_run_sha256":sha(r2/"run.json"),"later_not_used_in_selection":True},d/"contract.json")
    f=data();records=[];man=[]; configs=[]
    p=read(r2/"predictions.csv")
    baseline=p.loc[p.method.isin(["fixed_HGB_A","fixed_HGB_B","lag1",b])].drop_duplicates(["timestamp","method"])
    records.append(baseline)
    for split in c["outer"]:
        s=splits()[split];tr=f.loc[f.timestamp.le(s["train_end"])];ev=f.loc[f.timestamp.between(s["eval_start"],s["eval_end"])]
        cf=next(x for x in conf if x["split"]==split and x["method"]==b)
        family,params=cf["family"],cf["parameters"];cols=c["features"]["B"];name="delta_"+b
        mm=fit(family,params,tr,cols,True);path=MODEL/f"{split}_{name}.joblib";joblib.dump(mm,path,compress=3);pp=predict(mm,ev,cols,True)
        assert np.allclose(pp,predict(joblib.load(path),ev,cols,True),rtol=0,atol=1e-9)
        records.append(record(evmeta(ev,split),name,family,"B","delta",pp))
        man.append(manifest_row(mm,path,split,name,family,"B","delta",params,tr,"new_fit"))
        configs.append({"split":split,"method":name,"family":family,"input":"B","variant":"delta","parameters":params})
    pred=pd.concat(records,ignore_index=True)
    save(pred,d,"predictions.csv");save(pd.DataFrame(man),d,"model_manifest.csv");dump(configs,d/"configurations.json")
    tables,table,selection,leave=assess(pred,c)
    save(tables,d,"metrics.csv");save(table,d,"comparison.csv");save(leave,d,"leave_one_date_out.csv");dump(selection,d/"selection.json")
    dump({"status":"completed","completed_at":now(),"new_fits":FITS,"new_trials":0,"script_sha256":sha(Path(__file__)),"parent_run_sha256":sha(r2/"run.json"),
          "outputs_sha256":{p.name:sha(p) for p in d.iterdir() if p.suffix in [".csv",".json"] and p.name!="run.json"}},d/"run.json")
    print("Direct B comparison selected for R03:",b,flush=True);print(table.to_string(index=False),flush=True)
def later_stage():
    c=json.loads(CFILE.read_text(encoding="utf-8"));check_inputs(c)
    d=BASE/"r04";d.mkdir(parents=True,exist_ok=True);assert not (d/"run.json").exists()
    r2=BASE/"r02";r3=BASE/"r03"
    s2=json.loads((r2/"selection.json").read_text(encoding="utf-8"));s3run=json.loads((r3/"run.json").read_text(encoding="utf-8"))
    selected=s2["selected"];source=r2
    if selected is None and s3run["status"]=="completed":
        selected=json.loads((r3/"selection.json").read_text(encoding="utf-8"))["selected"];source=r3
    if selected is None:
        dump({"status":"skipped","reason":"No candidate met fixed development criteria; do not search later data for a winner",
              "recorded_at":now(),"differentiation_goal":"unresolved"},d/"run.json");print("R04 skipped: no eligible development candidate",flush=True);return
    v=json.loads((source/"independent_verification.json").read_text(encoding="utf-8"));assert v["status"]=="passed"
    configs=json.loads((source/"configurations.json").read_text(encoding="utf-8"))
    cf=next(x for x in configs if x["method"]==selected and x["split"]=="dev_jun")
    # Freeze latest chronological configuration before any new later prediction.
    dump({"recorded_at":now(),"method":selected,"configuration":cf,"training_end":"2021-06-30 23:00:00",
          "refit_july":False,"scope":"observed later-period redevelopment evaluation","parent_run_sha256":sha(source/"run.json")},d/"contract.json")
    f=data();tr=f.loc[f.timestamp.lt("2021-07-01")];ev=f.loc[f.timestamp.ge("2021-07-01")]
    assert len(tr)==4176 and len(ev)==1344
    cols=c["features"][cf["input"]];delta=cf["variant"]=="delta"
    mm=fit(cf["family"],cf["parameters"],tr,cols,delta);path=MODEL/f"later_{selected}.joblib";joblib.dump(mm,path,compress=3)
    pred=predict(mm,ev,cols,delta);assert np.allclose(pred,predict(joblib.load(path),ev,cols,delta),rtol=0,atol=1e-9)
    base=evmeta(ev,"later_jul_aug")
    records=[record(base,selected,cf["family"],cf["input"],cf["variant"],pred)]
    old=read(ROOT/"Modeling/tables/m05/predictions.csv")
    for group,name in [("A","fixed_HGB_A"),("B","fixed_HGB_B"),("lag1","lag1")]:
        part=old.loc[old.pool.eq("common")&old.group.eq(group)]
        assert len(part)==len(ev) and part.timestamp.tolist()==ev.timestamp.tolist()
        records.append(record(base,name,"HGB" if group!="lag1" else "baseline",group,"fixed",part.prediction.to_numpy()))
    p=pd.concat(records,ignore_index=True);save(p,d,"predictions.csv")
    save(pd.DataFrame([manifest_row(mm,path,"later_jul_aug",selected,cf["family"],cf["input"],cf["variant"],cf["parameters"],tr,"new_fit")]),d,"model_manifest.csv")
    tables,table,selection,leave=assess(p,c);save(tables,d,"metrics.csv");save(table,d,"comparison.csv");save(leave,d,"leave_one_date_out.csv");dump(selection,d/"selection.json")
    dump([cf],d/"configurations.json")
    dump({"status":"completed","completed_at":now(),"new_fits":FITS,"new_trials":0,"script_sha256":sha(Path(__file__)),
          "outputs_sha256":{p.name:sha(p) for p in d.iterdir() if p.suffix in [".csv",".json"] and p.name!="run.json"}},d/"run.json")
    print(table.to_string(index=False),flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=["freeze","r02","r03","r04"],required=True);a=ap.parse_args()
    assert sys.version_info[:2]==(3,13) and sys._is_gil_enabled()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    {"freeze":freeze,"r02":compare,"r03":delta_stage,"r04":later_stage}[a.stage]()
if __name__=="__main__":main()
