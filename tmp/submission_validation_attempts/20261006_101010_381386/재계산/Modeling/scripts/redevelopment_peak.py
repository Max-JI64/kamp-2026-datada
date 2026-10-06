"""R03b: historical daily-maximum weighting, same HGB B and no tree retuning."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,joblib
import redevelopment_compare as common
from redevelopment_r01 import now,dump,read
from m01_prepare import ROOT,sha
BASE=common.BASE;CF=ROOT/"Modeling/config/redevelopment_peak_contract.json"
WEIGHTS=[1.,1.25,1.5,2.,3.,4.,8.]
def freeze():
    assert not CF.exists()
    for stage in ["r02","r03"]:
        v=json.loads((BASE/stage/"independent_verification.json").read_text(encoding="utf-8"));assert v["status"]=="passed"
    c={"version":"redevelopment-peak-v1","recorded_at":now(),
       "reason":"R02 has no eligible model/input candidate; same-input delta target has MAE6.181 but peak8.953 versus B7.589. Align training emphasis with daily-max evaluation rather than repeat model tuning.",
       "source_r02":sha(BASE/"r02/run.json"),"source_r03":sha(BASE/"r03/run.json"),
       "helper_sha256":sha(ROOT/"Modeling/scripts/redevelopment_compare.py"),"entrypoint_sha256":sha(Path(__file__)),
       "weight_grid":WEIGHTS,"parameters":common.HGB,"features":"B18; unchanged",
       "training_weight":"1+(lambda-1)*w; w is 1/ties at actual maximum of each complete eligible training date, zero otherwise",
       "selection":"within each outer training interval: overall <=1.01 inner B, over increase <=0.05 inner B peakMAE, monthly overall/peak <=1.05 B; minimize max(peakMAE/B,peakUnder/B), tie smaller lambda. Lambda1 baseline always feasible.",
       "outer_gates":"unchanged redevelopment_r02_contract.json criteria against both original A/B",
       "no_new_optuna":True,"no_later_selection":True,"max_new_fits":75}
    dump(c,CF);print("R03b fixed-weight grid recorded",flush=True)
def fitted(tr,cols,lam):
    m=common.model("HGB",common.HGB)
    w=1+(lam-1)*common.pw(tr)
    m.fit(tr[cols],tr[common.TARGET],sample_weight=w)
    return m,w
def run():
    c=json.loads(CF.read_text(encoding="utf-8"));assert c["entrypoint_sha256"]==sha(Path(__file__))
    assert c["helper_sha256"]==sha(ROOT/"Modeling/scripts/redevelopment_compare.py")
    d=BASE/"r03b";d.mkdir(parents=True,exist_ok=True);assert not (d/"run.json").exists()
    contract=json.loads(common.CFILE.read_text(encoding="utf-8"));common.check_inputs(contract)
    f=common.data();cols=contract["features"]["B"];selected=[];all_inner=[];trials=[];man=[];preds=[];training=[]
    old=read(BASE/"r02/predictions.csv")
    preds.append(old.loc[old.method.isin(["fixed_HGB_A","fixed_HGB_B","lag1"])])
    fits=0;start=time.perf_counter()
    for split in contract["outer"]:
        s=common.splits()[split];tr=f.loc[f.timestamp.le(s["train_end"])];ev=f.loc[f.timestamp.between(s["eval_start"],s["eval_end"])]
        baseline=[];candidates={lam:[] for lam in WEIGHTS}
        for month in contract["inner_months"][split]:
            t=pd.Timestamp(2021,month,1);a=tr.loc[tr.timestamp.lt(t)];b=tr.loc[tr.timestamp.ge(t)&tr.timestamp.lt(t+pd.offsets.MonthBegin(1))]
            base=b[["timestamp","date","month",common.TARGET,"lag1_maximum","lag1_last"]].rename(columns={common.TARGET:"actual"}).copy()
            base["split"]=f"inner_{month}";base["daily_maximum_weight"]=common.pw(b);base["profile_weight"]=1.;base["train_profile_overlap"]=False
            ma=common.fit("HGB",common.HGB,a,contract["features"]["A"]);fits+=1
            baseline.append(common.record(base,"fixed_HGB_A","HGB","A","fixed",ma.predict(b[contract["features"]["A"]])))
            baseline.append(common.record(base,"lag1","baseline","none","baseline",b.lag1_maximum.to_numpy()))
            for lam in WEIGHTS:
                mm,w=fitted(a,cols,lam);fits+=1;name="fixed_HGB_B" if lam==1 else f"weight_{lam:g}"
                rec=common.record(base,name,"HGB","B","weighted",mm.predict(b[cols]))
                candidates[lam].append(rec)
                training.append({"outer":split,"inner_month":month,"lambda":lam,"train_hours":len(a),"weight_sum":float(w.sum()),
                                 "complete_training_dates_weight":float(common.pw(a).sum()),"train_end":str(a.timestamp.max()),"validation_start":str(b.timestamp.min())})
        inn=pd.concat(baseline+[r for parts in candidates.values() for r in parts],ignore_index=True)
        inn["outer"]=split;all_inner.append(inn)
        mtab=common.metrics(inn).set_index(["method","period","scope"]);base_all=mtab.loc[("fixed_HGB_B","pooled","all")];base_peak=mtab.loc[("fixed_HGB_B","pooled","peak")]
        available=[]
        for lam in WEIGHTS:
            name="fixed_HGB_B" if lam==1 else f"weight_{lam:g}"
            allm=mtab.loc[(name,"pooled","all")];pk=mtab.loc[(name,"pooled","peak")]
            monthly=all(mtab.loc[(name,f"month:{month}",scope),"MAE"]<=mtab.loc[("fixed_HGB_B",f"month:{month}",scope),"MAE"]*1.05
                        for month in contract["inner_months"][split] for scope in ["all","peak"])
            feasible=bool(lam==1 or (allm.MAE<=base_all.MAE*1.01 and pk.mean_over-base_peak.mean_over<=base_peak.MAE*.05 and monthly))
            objective=max(pk.MAE/base_peak.MAE,pk.mean_under/base_peak.mean_under)
            trials.append({"outer":split,"lambda":lam,"overall_MAE":float(allm.MAE),"peak_MAE":float(pk.MAE),"peak_under":float(pk.mean_under),
                           "peak_over":float(pk.mean_over),"base_overall":float(base_all.MAE),"base_peak":float(base_peak.MAE),"base_under":float(base_peak.mean_under),
                           "base_over":float(base_peak.mean_over),"monthly_feasible":monthly,"feasible":feasible,"objective":float(objective)})
            if feasible:available.append((objective,lam))
        lam=min(available)[1]
        name="peak_weighted_HGB_B";mm,w=fitted(tr,cols,lam);fits+=1;path=common.MODEL/f"{split}_{name}.joblib";joblib.dump(mm,path,compress=3)
        pp=mm.predict(ev[cols]);assert np.allclose(pp,joblib.load(path).predict(ev[cols]),rtol=0,atol=1e-9)
        preds.append(common.record(common.evmeta(ev,split),name,"HGB","B","peak_weighted",pp))
        mr=common.manifest_row(mm,path,split,name,"HGB","B","peak_weighted",common.HGB,tr,"new_weighted_fit");mr["training_lambda"]=lam;mr["training_weight_sum"]=float(w.sum());man.append(mr)
        selected.append({"split":split,"method":name,"family":"HGB","input":"B","variant":"peak_weighted","parameters":common.HGB,"training_lambda":lam})
        print(f"{split} historical peak lambda={lam}; MAE/peak={common.obj(ev[common.TARGET],pp,common.pw(ev))}",flush=True)
    p=pd.concat(preds,ignore_index=True);tables,tab,sel,loo=common.assess(p,contract)
    for df,name in [(p,"predictions.csv"),(tables,"metrics.csv"),(tab,"comparison.csv"),(loo,"leave_one_date_out.csv"),
                    (pd.DataFrame(man),"model_manifest.csv"),(pd.concat(all_inner),"inner_predictions.csv"),
                    (pd.DataFrame(trials),"weight_trials.csv"),(pd.DataFrame(training),"training_weight_audit.csv")]:common.save(df,d,name)
    dump(selected,d/"configurations.json");dump(sel,d/"selection.json")
    dump({"status":"completed","completed_at":now(),"new_fits":fits,"new_optuna_trials":0,"weight_configurations":21,"seconds":time.perf_counter()-start,
          "script_sha256":sha(ROOT/"Modeling/scripts/redevelopment_compare.py"),"entrypoint_sha256":sha(Path(__file__)),"contract_sha256":sha(CF),
          "outputs_sha256":{p.name:sha(p) for p in d.iterdir() if p.suffix in [".csv",".json"] and p.name!="run.json"}},d/"run.json")
    print(tab.to_string(index=False),flush=True)
def later():
    contract=json.loads(common.CFILE.read_text(encoding="utf-8"));d=BASE/"r04";d.mkdir(parents=True,exist_ok=True);assert not (d/"run.json").exists()
    src=BASE/"r03b";sel=json.loads((src/"selection.json").read_text(encoding="utf-8"));name=sel["selected"]
    if name is None:
        dump({"status":"skipped","recorded_at":now(),"reason":"No eligible candidate in R02, R03 or R03b","differentiation_goal":"unresolved"},d/"run.json")
        print("R04 skipped: no eligible candidate",flush=True);return
    assert json.loads((src/"independent_verification.json").read_text(encoding="utf-8"))["status"]=="passed"
    cf=next(x for x in json.loads((src/"configurations.json").read_text(encoding="utf-8")) if x["split"]=="dev_jun" and x["method"]==name)
    dump({"recorded_at":now(),"method":name,"configuration":cf,"training_end":"2021-06-30 23:00:00","refit_july":False,
          "scope":"observed later-period redevelopment evaluation","parent_run_sha256":sha(src/"run.json")},d/"contract.json")
    f=common.data();tr=f.loc[f.timestamp.lt("2021-07-01")];ev=f.loc[f.timestamp.ge("2021-07-01")];cols=contract["features"]["B"]
    mm,w=fitted(tr,cols,cf["training_lambda"]);path=common.MODEL/f"later_{name}.joblib";joblib.dump(mm,path,compress=3)
    pp=mm.predict(ev[cols]);base=common.evmeta(ev,"later_jul_aug")
    preds=[common.record(base,name,"HGB","B","peak_weighted",pp)]
    old=read(ROOT/"Modeling/tables/m05/predictions.csv")
    for group,method in [("A","fixed_HGB_A"),("B","fixed_HGB_B"),("lag1","lag1")]:
        part=old.loc[old.pool.eq("common")&old.group.eq(group)];assert len(part)==len(ev)
        preds.append(common.record(base,method,"HGB" if group!="lag1" else "baseline",group,"fixed",part.prediction.to_numpy()))
    p=pd.concat(preds,ignore_index=True);tables,tab,sel,loo=common.assess(p,contract)
    mr=common.manifest_row(mm,path,"later_jul_aug",name,"HGB","B","peak_weighted",common.HGB,tr,"new_weighted_fit")
    mr["training_lambda"]=cf["training_lambda"];mr["training_weight_sum"]=float(w.sum())
    for df,n in [(p,"predictions.csv"),(tables,"metrics.csv"),(tab,"comparison.csv"),(loo,"leave_one_date_out.csv"),(pd.DataFrame([mr]),"model_manifest.csv")]:common.save(df,d,n)
    dump([cf],d/"configurations.json");dump(sel,d/"selection.json")
    dump({"status":"completed","completed_at":now(),"new_fits":1,"new_trials":0,"script_sha256":sha(ROOT/"Modeling/scripts/redevelopment_compare.py"),
          "entrypoint_sha256":sha(Path(__file__)),"outputs_sha256":{p.name:sha(p) for p in d.iterdir() if p.suffix in [".csv",".json"] and p.name!="run.json"}},d/"run.json")
    print(tab.to_string(index=False),flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=["freeze","run","later"],required=True);a=ap.parse_args()
    assert sys.version_info[:2]==(3,13) and sys._is_gil_enabled()
    {"freeze":freeze,"run":run,"later":later}[a.stage]()
if __name__=="__main__":main()
