"""P01 descriptive slot shape audit; no fit, manuscript edits or plots.
Use exact-hour timestamps, reuse verified low/zero episodes, retain every valid row.
"""
import csv, hashlib, json, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'data/origin/okm_augumented_2021.csv'
OUT = ROOT/'EDA/tables/p01_peak_shapes'
SLOTS = ['15분','30분','45분','60분']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(x,p): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def save(d,name): d.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
def main():
    assert sys.version_info[:2]==(3,13) and sys._is_gil_enabled()
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'run.json').exists(), 'Do not overwrite completed analysis'
    parent=json.loads((ROOT/'Analysis/tables/a06_power_state_transitions/summary.json').read_text(encoding='utf-8'))
    assert sha(SOURCE)==parent['source_sha256']
    for name,h in parent['outputs_sha256'].items():
        assert sha(ROOT/'Analysis/tables/a06_power_state_transitions'/name)==h
    c={'version':'p01-v1','recorded_at':datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='minutes'),
       'source_sha256':sha(SOURCE),'script_sha256':sha(Path(__file__)),
       'question':'Does within-hour shape and duration justify history and four-output comparisons?',
       'scope':'All 5784 valid Jan-Aug hours; descriptive only; already observed data',
       'high_definition':'January valid hourly maximum linear q95; fixed before output inspection; slot >= threshold',
       'shape_measures':['maximum-mean','maximum-minimum','maximum-second_largest','tied maxima','high slots count'],
       'episode_definition':'Consecutive recorded slots in column order, connect hours only when exact-hour consecutive; no interpolation',
       'future_rule':'No full-day label or completed episode duration may be a prediction input',
       'reuse':'a06 episodes.csv for low/zero; slot_time_patterns for mean slot shapes',
       'no_manuscript_edit':True}
    dump(c,OUT/'contract.json')
    r=pd.read_csv(SOURCE,encoding='utf-8-sig')
    d=r.loc[r['날짜'].between(20210101,20210831)&r['시간'].between(0,23)].sort_values(['날짜','시간']).copy()
    d['timestamp']=pd.to_datetime(d['날짜'].astype(str),format='%Y%m%d')+pd.to_timedelta(d['시간'],unit='h')
    assert len(d)==5784 and d.timestamp.is_unique
    v=d[SLOTS].to_numpy(float); mx=v.max(1); mn=v.min(1); av=v.mean(1)
    threshold=float(np.quantile(mx[d.timestamp.dt.month.eq(1)],.95))
    o=pd.DataFrame({'timestamp':d.timestamp,'date':d.timestamp.dt.normalize(),'month':d.timestamp.dt.month,'hour':d['시간'],
        'maximum':mx,'mean':av,'range':mx-mn,'max_mean_gap':mx-av,'top_margin':mx-np.sort(v,axis=1)[:,-2],
        'max_ties':(v==mx[:,None]).sum(1),'high_slots':(v>=threshold).sum(1)})
    for i in range(4): o['max_slot_'+str(i+1)]=(v[:,i]==mx).astype(int)
    o['peak_mask']=[''.join('1' if a==max(row) else '0' for a in row) for row in v]
    hashes={day:hashlib.sha256(g[SLOTS].to_numpy(dtype=np.int64).tobytes()).hexdigest() for day,g in d.groupby(d.timestamp.dt.normalize())}
    o['profile']=o.date.map(hashes);freq=pd.Series(hashes).value_counts();o['profile_weight']=1/o.profile.map(freq)
    o['repeated_profile']=o.profile.map(freq).gt(1)
    dm=o.groupby('date').maximum.transform('max'); flag=o.maximum.eq(dm)
    o['daily_maximum_weight']=flag/flag.groupby(o.date).transform('sum')
    records=[]
    def summary(g,label,weight=None):
        w=np.ones(len(g)) if weight is None else np.asarray(weight)
        return {'scope':label,'hours':len(g),'dates':g.date.nunique(),'weight_sum':float(w.sum()),
            **{x:float(np.average(g[x],weights=w)) for x in ['maximum','range','max_mean_gap','top_margin','max_ties','high_slots']},
            'unique_max_fraction':float(np.average(g.max_ties.eq(1),weights=w)),
            **{'slot_'+str(i)+'_share':float(np.average(g['max_slot_'+str(i)]/g.max_ties,weights=w)) for i in range(1,5)}}
    records.extend([summary(o,'all'),summary(o,'profile_weighted',o.profile_weight),summary(o.loc[flag],'daily_peak',o.loc[flag,'daily_maximum_weight']),summary(o.loc[o.high_slots.gt(0)],'high')])
    for field in ['month','hour','repeated_profile','high_slots']:
        for val,g in o.groupby(field):records.append(summary(g,f'{field}:{val}'))
    save(o,'hourly_shapes');save(pd.DataFrame(records),'shape_summary')
    save(o.groupby(['month','peak_mask','high_slots']).size().rename('hours').reset_index(),'shape_counts')
    episodes=[];start=None;count=0;prev=None;last=None
    for ts,row in zip(d.timestamp,v):
        for i,value in enumerate(row):
            ordinal=int(ts.value//3_600_000_000_000)*4+i
            if start is not None and (ordinal!=prev+1 or value<threshold):
                episodes.append({'start_hour':start[0],'start_slot':start[1]+1,'end_hour':last[0],'end_slot':last[1]+1,'slots':count,'left_censored':left,'right_censored':ordinal!=prev+1})
                start=None;count=0
            if value>=threshold:
                if start is None:start=(ts,i);left=prev is None or ordinal!=prev+1
                count+=1;last=(ts,i)
            prev=ordinal
    if start is not None:episodes.append({'start_hour':start[0],'start_slot':start[1]+1,'end_hour':last[0],'end_slot':last[1]+1,'slots':count,'left_censored':left,'right_censored':True})
    ep=pd.DataFrame(episodes);save(ep,'high_episodes')
    assert ep.slots.sum()==int((v>=threshold).sum())
    # Independent stdlib raw-row reconstruction of every derived shape value.
    raw={}
    with SOURCE.open(encoding='utf-8-sig',newline='') as f:
        for q in csv.DictReader(f):
            day=int(q['날짜']);hour=float(q['시간'])
            if 20210101<=day<=20210831 and 0<=hour<=23:
                ts=datetime.strptime(str(day),'%Y%m%d')+timedelta(hours=hour)
                raw[ts]=[float(q[s]) for s in SLOTS]
    for q in o.to_dict('records'):
        a=raw[q['timestamp'].to_pydatetime()];maximum=max(a)
        assert q['maximum']==maximum and q['mean']==sum(a)/4 and q['range']==maximum-min(a)
        assert q['max_mean_gap']==maximum-sum(a)/4 and q['top_margin']==maximum-sorted(a)[-2]
        assert q['max_ties']==a.count(maximum) and q['high_slots']==sum(x>=threshold for x in a)
        assert q['peak_mask']==''.join('1' if x==maximum else '0' for x in a)
    # Check episode records from a distinct list-based run-length reconstruction.
    high_ord=sorted(int(pd.Timestamp(ts).value//3_600_000_000_000)*4+i for ts,a in raw.items() for i,x in enumerate(a) if x>=threshold)
    groups=[]
    for x in high_ord:
        if not groups or x!=groups[-1][-1]+1:groups.append([x])
        else:groups[-1].append(x)
    assert ep.slots.tolist()==list(map(len,groups))
    verify={'status':'passed','raw_rows':len(raw),'high_episodes_checked':len(groups),'outputs_sha256':{p.name:sha(p) for p in OUT.glob('*.csv')}}
    dump(verify,OUT/'verification.json')
    counts=o.loc[o.high_slots.gt(0)].high_slots.value_counts().sort_index().to_dict()
    result={'status':'completed','high_threshold_january_q95':threshold,'rows':len(o),'days':o.date.nunique(),'profiles':len(freq),
        'high_hours':int(o.high_slots.gt(0).sum()),'high_slot_counts':{str(k):int(n) for k,n in counts.items()},
        'high_episodes':len(ep),'high_episode_slots_median':float(ep.slots.median()),'high_episode_slots_max':int(ep.slots.max()),
        'key_summaries':records[:4],'existing_low_zero_episodes':parent['episodes'],'contract_sha256':sha(OUT/'contract.json'),
        'outputs_sha256':verify['outputs_sha256'],'verified':True,'model_fit':False,'manuscript_edited':False}
    dump(result,OUT/'run.json');print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()