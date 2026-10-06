"""R01: paired A/B errors by forecast-origin information; no model fitting."""
import argparse, json, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from m01_prepare import ROOT, read_contract, sha, load_frame, independent_check

OUT = ROOT / "Modeling/tables/redevelopment/r01"
CFILE = ROOT / "Modeling/config/redevelopment_r01_contract.json"
def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes")
def read(path):
    return pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip", parse_dates=["timestamp"])
def dump(value,path):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
def save(df,name):
    df.to_csv(OUT / name,index=False,encoding="utf-8-sig")
def parents():
    paths = ["Modeling/config/m01_contract.json","Modeling/tables/m01/hourly_frame.csv",
             "Modeling/tables/m03/ab/predictions.csv","Modeling/tables/m03/ab/run.json",
             "Modeling/tables/m03/ab/independent_verification.json",
             "Modeling/tables/m05/predictions.csv","Modeling/tables/m05/run.json",
             "Modeling/tables/m05/independent_verification.json",
             "Modeling/scripts/redevelopment_r01.py"]
    c=read_contract()
    paths.append(c["source"])
    for folder in ["m03/ab","m05"]:
        dr=ROOT/"Modeling/tables"/folder
        run=json.loads((dr/"run.json").read_text(encoding="utf-8"))
        ver=json.loads((dr/"independent_verification.json").read_text(encoding="utf-8"))
        assert run["status"]=="completed" and ver["status"]=="passed"
        assert ver["run_sha256"]==sha(dr/"run.json")
        assert run["outputs_sha256"]["predictions.csv"]==sha(dr/"predictions.csv")
    return {p:sha(ROOT/p) for p in paths}
def freeze():
    assert not CFILE.exists()
    dump({"version":"redevelopment-r01-v1","recorded_at":now(),
          "reason":"M05 A/B peak loss contradicts M03 development gain; diagnose using existing predictions before new fits.",
          "inputs_sha256":parents(),"new_fits":0,
          "splits":["dev_apr","dev_may","dev_jun","later_jul_aug"],
          "bins":"q1/3,q2/3 of positive prior level and range in each outer training interval only; include zero separately",
          "conditions":["prior_state","level_bin","range_bin","prior_trend","last_gap_bin","train_profile_overlap","month"],
          "last_gap":"prior maximum minus prior last; flat <=0, else prior training positive-gap median",
          "weight":"existing complete-date tied-maximum weights; equal dates",
          "interpretation":"exploratory diagnosis, no conditional model selection, later period already observed"},CFILE)
def paired():
    c=read_contract()
    raw,invalid,valid,f=load_frame(c)
    independent_check(c,raw,invalid,valid,f)
    raw_index=valid.set_index("timestamp")
    previous=raw_index.reindex(pd.DatetimeIndex(f.timestamp-pd.Timedelta(hours=1)))
    f["prior_range"]=previous[["15분","30분","45분","60분"]].max(axis=1).to_numpy()-previous[["15분","30분","45분","60분"]].min(axis=1).to_numpy()
    f["prior_slot_change"]=previous["60분"].to_numpy()-previous["15분"].to_numpy()
    f["last_gap"]=f.lag1_maximum-f.lag1_last
    f["prior_state"]=np.select([f.lag1_maximum.eq(0),np.floor(f.lag1_mean+.5).between(20,26),f.lag1_mean.lt(20)],
                                 ["zero","low20_26","positive_below20"],default="above26")
    f["prior_trend"]=np.select([f.prior_slot_change.gt(0),f.prior_slot_change.lt(0)],["up","down"],default="flat")
    dev=read(ROOT/"Modeling/tables/m03/ab/predictions.csv")
    dev=dev.loc[dev.target.eq("target_maximum") & dev.group.isin(["A","B"])].copy()
    later=read(ROOT/"Modeling/tables/m05/predictions.csv")
    later=later.loc[later.pool.eq("common") & later.group.isin(["A","B"])].copy()
    later["split"]="later_jul_aug"
    p=pd.concat([dev,later],ignore_index=True)
    keys=["timestamp","split"]
    metadata=["actual","daily_maximum_weight","profile_weight","train_profile_overlap"]
    a=p.loc[p.group.eq("A"),keys+metadata+["prediction"]].rename(columns={"prediction":"prediction_A"})
    b=p.loc[p.group.eq("B"),keys+["prediction"]].rename(columns={"prediction":"prediction_B"})
    pair=a.merge(b,on=keys,validate="one_to_one").merge(f,on="timestamp",validate="one_to_one")
    assert len(pair)==3528 and np.array_equal(pair.actual,pair.target_maximum)
    thresholds=[]
    splits={s["name"]:s for s in c["splits"]}
    for split,part in pair.groupby("split"):
        train_end = "2021-06-30 23:00:00" if split=="later_jul_aug" else splits[split]["train_end"]
        tr=f.loc[f.eligible_common & f.timestamp.le(train_end)]
        assert tr.timestamp.max()<part.timestamp.min()
        qs={}
        for field,name in [("lag1_mean","level"),("prior_range","range")]:
            values=tr.loc[tr[field].gt(0),field]
            q=values.quantile([1/3,2/3]).tolist()
            qs[name]=q
            pair.loc[part.index,f"{name}_bin"]=np.select([part[field].eq(0),part[field].le(q[0]),part[field].le(q[1])],["zero","lower","middle"],default="upper")
        positive=tr.loc[tr.last_gap.gt(0),"last_gap"]
        median=float(positive.median())
        pair.loc[part.index,"last_gap_bin"]=np.select([part.last_gap.le(0),part.last_gap.le(median)],["at_maximum","small_gap"],default="large_gap")
        thresholds.append({"split":split,"train_end":train_end,"train_hours":len(tr),"level_q33":qs["level"][0],"level_q67":qs["level"][1],
                           "range_q33":qs["range"][0],"range_q67":qs["range"][1],"gap_median":median})
    for group in ["A","B"]:
        err=pair[f"prediction_{group}"]-pair.actual
        pair[f"abs_{group}"]=abs(err)
        pair[f"under_{group}"]=(-err).clip(lower=0)
        pair[f"over_{group}"]=err.clip(lower=0)
        pair[f"bias_{group}"]=err
    pair["delta_absolute_error"]=pair.abs_B-pair.abs_A
    pair["date"]=pair.timestamp.dt.strftime("%Y-%m-%d")
    return pair,pd.DataFrame(thresholds)
def summary(pair):
    rows=[]
    for period, part in [("development",pair.loc[pair.month.le(6)]),("later",pair.loc[pair.month.ge(7)])]:
        for col in ["all","prior_state","level_bin","range_bin","prior_trend","last_gap_bin","train_profile_overlap","month"]:
            sections=[("all",part)] if col=="all" else list(part.groupby(col))
            for value,g in sections:
                for scope in ["all","peak"]:
                    w=np.ones(len(g)) if scope=="all" else g.daily_maximum_weight.to_numpy()
                    if w.sum()==0: continue
                    row={"period":period,"condition":col,"value":str(value),"scope":scope,"hours":len(g) if scope=="all" else int((w>0).sum()),
                         "days":g.loc[w>0,"date"].nunique(),"weight_sum":float(w.sum())}
                    for label in ["A","B"]:
                        for metric in ["abs","under","over","bias"]:
                            row[f"{metric}_{label}"]=float(np.average(g[f"{metric}_{label}"],weights=w))
                    row["delta_MAE"]=row["abs_B"]-row["abs_A"]
                    row["unweighted_delta_sum"]=float(g.delta_absolute_error.sum())
                    rows.append(row)
    return pd.DataFrame(rows)
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["freeze","run"],required=True)
    args=parser.parse_args()
    assert sys.version_info[:2]==(3,13) and sys._is_gil_enabled()
    if args.stage=="freeze": freeze();print("R01 contract recorded",flush=True);return
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/"run.json").exists()
    c=json.loads(CFILE.read_text(encoding="utf-8")); assert c["inputs_sha256"]==parents()
    pair,thresholds=paired()
    save(pair,"paired.csv");save(thresholds,"thresholds.csv");save(summary(pair),"conditions.csv")
    daily=pair.groupby(["split","date","month"]).agg(hours=("timestamp","size"),abs_A=("abs_A","mean"),abs_B=("abs_B","mean"),
                                                   under_A=("under_A","mean"),under_B=("under_B","mean"),delta_sum=("delta_absolute_error","sum")).reset_index()
    save(daily,"daily_errors.csv")
    loo=[]
    for period,g in pair.groupby(pair.month.le(6).map({True:"development",False:"later"})):
        for day in sorted(g.date.unique()):
            rem=g.loc[g.date.ne(day)]
            for scope in ["all","peak"]:
                w=np.ones(len(rem)) if scope=="all" else rem.daily_maximum_weight
                loo.append({"period":period,"excluded_date":day,"scope":scope,"delta_MAE":float(np.average(rem.delta_absolute_error,weights=w))})
    save(pd.DataFrame(loo),"leave_one_date_out.csv")
    run={"status":"completed","completed_at":now(),"new_fits":0,"pairs":len(pair),"contract_sha256":sha(CFILE),"inputs_sha256":c["inputs_sha256"],
         "outputs_sha256":{p.name:sha(p) for p in OUT.glob("*.csv")}}
    dump(run,OUT/"run.json")
    print(summary(pair).loc[lambda d:d.condition.eq("all")].to_string(index=False),flush=True)
if __name__=="__main__":main()
