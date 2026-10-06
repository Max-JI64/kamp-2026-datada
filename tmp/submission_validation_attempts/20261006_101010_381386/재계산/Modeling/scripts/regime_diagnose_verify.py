"""Independent CSV/scalar S01/S02 verification and short/persistent comparisons."""
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Modeling/tables/regime_diagnosis"
SOURCE = ROOT / "data/origin/okm_augumented_2021.csv"
SLOTS = ["15분", "30분", "45분", "60분"]
FEATURES = ["prior_mean","past_median6","past_iqr6","past_range6","past_slope6",
            "prior_delta","prior_last_minus_mean","prior_slot_range","prior_production",
            "prior_production_delta","prior_state_age","prior_state_left_censored"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def truth(x):
    assert x in ("True","False"),x
    return x == "True"


def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-8),(a,b)


def q(values,p):
    a = sorted(values)
    loc = (len(a)-1)*p
    lo,hi = math.floor(loc),math.ceil(loc)
    return a[lo]+(a[hi]-a[lo])*(loc-lo)


def reconstruct(raw,times,thresholds):
    active=None
    gap=None
    results={}
    events=[]
    for i,t in enumerate(times):
        previous=[raw.get(t-timedelta(hours=k)) for k in range(1,7)]
        own=raw.get(t)
        eligible=own is not None and all(x is not None for x in previous)
        state={"age":i-active["index"] if active else 0,
               "direction":active["sign"] if active else 0,
               "censored":int(active is None and (gap is None or i-gap<=6))}
        if own is None:
            active=None
            gap=i
            results[t]={"eligible":False,"onset":0,"state":state,"candidate":0}
            continue
        candidate=0
        features={}
        if eligible:
            levels=[x["mean"] for x in previous]
            med=median(levels)
            iqr=q(levels,.75)-q(levels,.25)
            delta=own["mean"]-levels[0]
            dev=own["mean"]-med
            if delta>=thresholds["up"] and dev>=thresholds["up"] and dev>=1.5*iqr:
                candidate=1
            if -delta>=thresholds["down"] and -dev>=thresholds["down"] and -dev>=1.5*iqr:
                candidate=-1
            features={"prior_mean":levels[0],"past_median6":med,"past_iqr6":iqr,
                "past_range6":max(levels)-min(levels),"past_slope6":(levels[0]-levels[-1])/5,
                "prior_delta":levels[0]-levels[1],
                "prior_last_minus_mean":previous[0]["slots"][-1]-levels[0],
                "prior_slot_range":max(previous[0]["slots"])-min(previous[0]["slots"]),
                "prior_production":previous[0]["production"],
                "prior_production_delta":previous[0]["production"]-previous[1]["production"],
                "prior_state_age":state["age"],"prior_state_left_censored":state["censored"]}
        onset=0
        if active:
            th=thresholds["up" if active["sign"]==1 else "down"]
            if active["sign"]*(own["mean"]-active["baseline"]) < th/2 or (candidate and candidate!=active["sign"]):
                active=None
        if candidate and active is None:
            onset=candidate
            active={"index":i,"sign":candidate,"baseline":features["past_median6"]}
            events.append((t,candidate,features["past_median6"]))
        results[t]={"eligible":eligible,"onset":onset,"candidate":candidate,
                    "state":state,"features":features}
    return results,events


def main():
    assert sys.version_info[:2] == (3,13) and sys._is_gil_enabled()
    c=json.loads((OUT/"contract.json").read_text(encoding="utf-8"))
    run=json.loads((OUT/"run.json").read_text(encoding="utf-8"))
    assert run["status"]=="completed" and run["contract_sha256"]==sha(OUT/"contract.json")
    assert sha(SOURCE)==c["source_sha256"]
    assert sha(ROOT/"Modeling/scripts/regime_diagnose.py")==c["entrypoint_sha256"]
    assert sha(OUT/"probe.json")==c["probe_sha256"]
    for name,digest in run["outputs_sha256"].items():
        assert sha(OUT/name)==digest,name
    raw={}
    for row in read(SOURCE):
        date,hour=int(row["날짜"]),int(row["시간"])
        if not (20210101<=date<=20210630 and 0<=hour<24):
            continue
        t=datetime.strptime(str(date),"%Y%m%d")+timedelta(hours=hour)
        assert t not in raw
        slots=[float(row[s]) for s in SLOTS]
        raw[t]={"slots":slots,"mean":math.fsum(slots)/4,
                "maximum":max(slots),"production":float(row["생산량"])}
    times=[datetime(2021,1,1)+timedelta(hours=i) for i in range(181*24)]
    changes=[]
    for t in times:
        if t.month==1 and t in raw and all(t-timedelta(hours=k) in raw for k in range(1,7)):
            changes.append(raw[t]["mean"]-raw[t-timedelta(hours=1)]["mean"])
    for key,values in [("up",[x for x in changes if x>0]),("down",[-x for x in changes if x<0])]:
        close(c["thresholds"][key],q(values,.9))
    independent,event_starts=reconstruct(raw,times,c["thresholds"])
    hourly=read(OUT/"hourly_labels.csv")
    assert len(hourly)==len(times)
    parent={datetime.fromisoformat(r["timestamp"]):r for r in read(ROOT/"Modeling/tables/m01/hourly_frame.csv")
            if r["timestamp"]<"2021-07"}
    assert set(parent)==set(raw)
    for row,t in zip(hourly,times):
        assert datetime.fromisoformat(row["timestamp"])==t
        expected=independent[t]
        assert truth(row["valid"])==(t in raw)
        assert truth(row["eligible_onset"])==expected["eligible"]
        assert int(row["onset"])==expected["onset"] and int(row["candidate"])==expected["candidate"]
        assert int(row["prior_state_age"])==expected["state"]["age"]
        assert int(row["prior_state_direction"])==expected["state"]["direction"]
        assert int(row["prior_state_left_censored"])==expected["state"]["censored"]
        assert datetime.fromisoformat(row["latest_input_timestamp"])==t-timedelta(hours=1)
        if t in raw:
            for field,value in [("level",raw[t]["mean"]),("maximum",raw[t]["maximum"])]:
                close(row[field],value)
            close(parent[t]["target_mean"],raw[t]["mean"])
            close(parent[t]["target_maximum"],raw[t]["maximum"])
        if expected["eligible"]:
            for feature,value in expected["features"].items():
                close(row[feature],value)
    events=read(OUT/"events.csv")
    assert len(events)==len(event_starts)==run["events"]
    for row,(t,sign,baseline) in zip(events,event_starts):
        assert datetime.fromisoformat(row["timestamp"])==t
        assert row["direction"]==("up" if sign==1 else "down")
        close(row["baseline"],baseline)
        boundary=c["thresholds"][row["direction"]]/2
        n=0
        while t+timedelta(hours=n) in raw and sign*(raw[t+timedelta(hours=n)]["mean"]-baseline)>=boundary:
            n+=1
        assert int(row["reference_departure_hours"])==n
        end=t+timedelta(hours=n)
        reason="scope_end" if end>times[-1] else ("gap" if end not in raw else "return")
        assert row["end_reason"]==reason and truth(row["right_censored"])==(reason!="return")
        for horizon in [1,3,6]:
            ts=[t+timedelta(hours=k) for k in range(horizon)]
            known=all(s in raw for s in ts)
            actual=row[f"sustained{horizon}"]
            if known:
                assert truth(actual)==all(sign*(raw[s]["mean"]-baseline)>=boundary for s in ts)
            else:
                assert actual==""
        if row["sustained3"]:
            assert datetime.fromisoformat(row["label_confirmed_at"])==t+timedelta(hours=2)
            close(row["next3_mean"],math.fsum(raw[t+timedelta(hours=k)]["mean"] for k in range(3))/3)
        hr=hourly[times.index(t)]
        if row["sustained3"]:
            close(hr["sustained_onset"],sign if truth(row["sustained3"]) else 0)
    for row in read(OUT/"event_summary.csv"):
        es=[e for e in events if e["month"]==row["month"] and e["direction"]==row["direction"]]
        assert int(row["events"])==len(es)
        assert int(row["sustained3"])==sum(e["sustained3"]=="True" for e in es)
        assert int(row["short3"])==sum(e["sustained3"]=="False" for e in es)
    early=[r for r in hourly if truth(r["eligible_onset"]) and truth(r["persistence_known"]) and int(r["month"])<=3]
    edges=json.loads((OUT/"precursor_contract.json").read_text(encoding="utf-8"))["prior_level_edges"]
    jan=[float(r["prior_mean"]) for r in early if r["month"]=="1"]
    assert len(edges)==len(sorted(set(q(jan,p) for p in [.2,.4,.6,.8])))
    for a,b in zip(edges,sorted(set(q(jan,p) for p in [.2,.4,.6,.8]))):
        close(a,b)
    targets={"up":lambda r:int(r["onset"])==1,"down":lambda r:int(r["onset"])==-1,
             "sustained_up":lambda r:float(r["sustained_onset"])==1,
             "sustained_down":lambda r:float(r["sustained_onset"])==-1}
    indexed=defaultdict(list)
    for r in early:
        level_bin=sum(float(r["prior_mean"])>=e for e in edges)
        indexed[(int(r["month"]),int(r["hour"]),level_bin)].append(r)
    # Rebuild every matched condition from source scalar features.
    cells=read(OUT/"matched_cells.csv")
    for cell in cells:
        key=(int(cell["month"]),int(cell["hour"]),int(cell["level_bin"]))
        a=[r for r in indexed[key] if targets[cell["target"]](r)]
        b=[r for r in indexed[key] if int(r["onset"])==0]
        assert len(a)==int(cell["event_n"]) and len(b)==int(cell["control_n"])
        close(cell["event_mean"],math.fsum(float(r[cell["feature"]]) for r in a)/len(a))
        close(cell["control_mean"],math.fsum(float(r[cell["feature"]]) for r in b)/len(b))
    for coverage in read(OUT/"matched_coverage.csv"):
        fn=targets[coverage["target"]]
        support=[]
        minimum=int(coverage["minimum_events_per_cell"])
        for key,rows in indexed.items():
            a=[r for r in rows if fn(r)]
            b=[r for r in rows if int(r["onset"])==0]
            if len(a)>=minimum and len(b)>=5:
                support.append((key,a,b))
        assert int(coverage["matched_cells"])==len(support)
        assert int(coverage["events_total"])==sum(fn(r) for r in early)
        assert int(coverage["events_covered"])==sum(len(x[1]) for x in support)
    for r in read(OUT/"precursor_summary.csv"):
        if r["weighting"]=="matched":
            cs=[x for x in cells if x["target"]==r["target"] and x["feature"]==r["feature"]
                and int(x["minimum_events_per_cell"])==int(float(r["minimum_events_per_cell"]))]
            if cs:
                close(r["matched_difference"],sum(float(x["weight"])*(float(x["event_mean"])-float(x["control_mean"])) for x in cs)/sum(float(x["weight"]) for x in cs))
    splits=read(OUT/"split_counts.csv")
    for row in splits:
        boundary=datetime(2021,int(row["month"]),1)
        subset=[r for r in hourly if truth(r["eligible_onset"]) and truth(r["persistence_known"])]
        if row["partition"]=="train":
            subset=[r for r in subset if datetime.fromisoformat(r["timestamp"])<boundary and datetime.fromisoformat(r["label_confirmed_at"])<boundary]
        else:
            subset=[r for r in subset if r["month"]==row["month"]]
        sign=1 if row["direction"]=="up" else -1
        assert int(row["hours"])==len(subset)
        assert int(row["onsets"])==sum(int(r["onset"])==sign for r in subset)
        assert int(row["sustained3"])==sum(float(r["sustained_onset"])==sign for r in subset)
    # Description of pre-onset conditions within events: short vs persistent.
    comparisons=[]
    event_map={r["timestamp"]:r for r in events if int(r["month"])<=3}
    for direction in ["up","down"]:
        groups={flag:[r for r in early if r["timestamp"] in event_map and event_map[r["timestamp"]]["direction"]==direction
                     and event_map[r["timestamp"]]["sustained3"]==flag] for flag in ["True","False"]}
        for feature in FEATURES:
            def avg(flag):
                rs=groups[flag]
                return sum(float(r[feature]) for r in rs)/len(rs) if rs else None
            comparisons.append({"direction":direction,"feature":feature,"persistent_n":len(groups["True"]),
                "short_n":len(groups["False"]),"persistent_mean":avg("True"),"short_mean":avg("False"),
                "scope":"Jan-Mar unmatched event-only description"})
    pd.DataFrame(comparisons).to_csv(OUT/"persistence_precursors.csv",index=False,encoding="utf-8-sig")
    # Prefix invariance: change future records and require all earlier onset/state outputs unchanged.
    prefix_tests=0
    for cut in [datetime(2021,1,16,8),datetime(2021,3,7,8),datetime(2021,6,15)]:
        prefix_times=[t for t in times if t<cut]
        trimmed={t:v for t,v in raw.items() if t<cut}
        rebuilt,_=reconstruct(trimmed,prefix_times,c["thresholds"])
        for t in prefix_times:
            assert rebuilt[t]["onset"]==independent[t]["onset"]
            assert rebuilt[t]["state"]==independent[t]["state"]
        prefix_tests+=1
    # Synthetic cases establish opposite direction, repeated suppression, short persistence and gaps.
    from regime_diagnose import label
    synthetic=[]
    for values,expected in [([100.]*6+[200.]*6,[(6,1)]),
                           ([100.]*6+[200.,100.,100.,100.],[(6,1),(7,-1)]),
                           ([100.]*6+[20.]*6,[(6,-1)]),
                           ([100.]*6+[200.,200.,np.nan,200.,200.],[(6,1)])]:
        ts=pd.date_range("2021-01-01",periods=len(values),freq="h")
        f=pd.DataFrame(index=ts)
        f["level"]=values;f["maximum"]=values;f["valid"]=f.level.notna()
        h=pd.concat([f.level.shift(k) for k in range(1,7)],axis=1)
        f["eligible_onset"]=f.valid&h.notna().all(axis=1)
        f["past_median6"]=h.median(axis=1);f["past_iqr6"]=h.quantile(.75,axis=1)-h.quantile(.25,axis=1)
        f["prior_mean"]=f.level.shift(1);f["delta"]=f.level-f.prior_mean;f["deviation"]=f.level-f.past_median6
        f["date"]=ts.normalize();f["month"]=1;f["hour"]=ts.hour;f["weekend"]=0;f["profile"]="synthetic"
        f["prior_production"]=0.;f["생산량"]=0.
        labeled,ev=label(f,{"up":50.,"down":40.})
        got=[(int(r.position),1 if r.direction=="up" else -1) for _,r in ev.iterrows()]
        # A one-hour spike returning to baseline is not a new fall from a stable high regime:
        # the 6h median is still 100, hence no separate qualifying downward onset.
        if values[6:8]==[200.,100.]:
            expected=[(6,1)]
        assert got==expected,(got,expected)
        if len(values)==12:
            assert bool(ev.sustained3.iloc[0])
        if any(pd.isna(v) for v in values):
            assert pd.isna(ev.sustained3.iloc[0])
        synthetic.append({"events":got,"persistence":str(ev.sustained3.iloc[0])})
    result={"status":"passed","run_sha256":sha(OUT/"run.json"),"verifier_sha256":sha(__file__),
            "raw_hours_checked":len(raw),"hourly_labels_checked":len(hourly),"events_checked":len(events),
            "matched_cells_checked":len(cells),"split_records_checked":len(splits),
            "prefix_tests":prefix_tests,"synthetic_cases":synthetic,"new_fits":0,
            "extra_outputs_sha256":{"persistence_precursors.csv":sha(OUT/"persistence_precursors.csv")}}
    (OUT/"independent_verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
