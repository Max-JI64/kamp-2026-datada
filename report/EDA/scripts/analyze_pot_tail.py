"""Explore a GPD tail threshold; never force a unique threshold or edit manuscripts.

Stage scan: descriptive Jan-Aug EDA, all integer thresholds 145..205, hourly and
run-declustered excesses; preserve rounding/ties and inspect repeated profiles.
Stage bootstrap: week-block diagnostics under unchanged threshold/event rules.
POT uses STRICT exceedance, unlike earlier >= percentile comparison.
"""
from pathlib import Path
import sys,json,hashlib,warnings
import numpy as np
import pandas as pd
import scipy
from scipy.stats import genpareto
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'data/origin/okm_augumented_2021.csv'
SHA='8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830'
SEED=20261002
BASE=ROOT/'report/EDA/tables/peak_definition_comparison'
OUT=BASE/'pot_tail_eda_full';OUT.mkdir(exist_ok=True)
US=np.arange(145,206);GAPS=[1,3,6,12,24]
BOOT_US=sorted(set(range(145,201,3))|{170,176,179,182,186,187,190,194,200,201})
N_BOOT=60

def save(d,name):d.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
def read(name):return pd.read_csv(OUT/(name+'.csv'),encoding='utf-8-sig')

def inputs():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
    raw=pd.read_csv(SOURCE,encoding='utf-8-sig')
    valid=raw.loc[raw['날짜'].between(20210101,20210831)&raw['시간'].between(0,23)].copy()
    valid['timestamp']=pd.to_datetime(valid['날짜'].astype(str))+pd.to_timedelta(valid['시간'],unit='h')
    valid=valid.sort_values('timestamp').reset_index(drop=True)
    d=valid[['timestamp']].copy()
    d['date']=d.timestamp.dt.normalize();d['month']=d.timestamp.dt.month
    d['P']=valid[['15분','30분','45분','60분']].max(axis=1)
    assert len(d)==5784 and d.date.nunique()==241 and sorted(d.month.unique())==list(range(1,9))
    tr=d.copy();tr['week']=tr.timestamp.dt.to_period('W-SUN').astype(str)
    return d,tr,valid

def events(d,u,gap):
    above=d.loc[d.P.gt(u),['timestamp','P','week']].copy()
    if not len(above):return pd.DataFrame(columns=['start','end','timestamp','P','week','exceed_hours','span_hours'])
    block=(above.timestamp.diff().dt.total_seconds().div(3600).gt(gap)).cumsum()
    rows=[]
    for _,part in above.groupby(block):
        peak=part.loc[part.P.idxmax()]
        rows.append(dict(start=part.timestamp.iloc[0],end=part.timestamp.iloc[-1],
            timestamp=peak.timestamp,P=peak.P,week=peak.week,exceed_hours=len(part),
            span_hours=(part.timestamp.iloc[-1]-part.timestamp.iloc[0]).total_seconds()/3600+1))
    return pd.DataFrame(rows)

def fit(y,u,starts=3):
    y=np.asarray(y,dtype=float)
    result=dict(n=len(y),unique=len(np.unique(y)),mean_excess=float(y.mean()) if len(y) else np.nan,
        shape=np.nan,scale=np.nan,modified_scale=np.nan,endpoint=np.nan,ks_distance=np.nan,
        nll=np.nan,finite_fit=False,nonregular_mle=False,fit_warning='')
    if len(y)<3 or len(np.unique(y))<2:
        result['fit_warning']='fewer than 3 values or fewer than 2 distinct values';return result
    answers=[];messages=[]
    for start in [-.3,0,.3][:starts]:
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                shape,_,scale=genpareto.fit(y,start,floc=0)
                logp=genpareto.logpdf(y,shape,loc=0,scale=scale)
            messages += [str(w.message) for w in ws]
            if np.isfinite(logp).all() and scale>0:answers.append((-logp.sum(),shape,scale))
        except (ValueError,FloatingPointError,RuntimeError) as e:messages.append(str(e))
    if not answers:result['fit_warning']=';'.join(messages) or 'no finite fit';return result
    nll,shape,scale=min(answers)
    y=np.sort(y);cdf=genpareto.cdf(y,shape,scale=scale);n=len(y)
    ks=max(np.max(np.arange(1,n+1)/n-cdf),np.max(cdf-np.arange(n)/n))
    result.update(shape=float(shape),scale=float(scale),modified_scale=float(scale-shape*u),
        endpoint=float(u-scale/shape) if shape<0 else np.nan,ks_distance=float(ks),nll=float(nll),
        finite_fit=True,nonregular_mle=bool(shape<=-.5),fit_warning=';'.join(sorted(set(messages))))
    return result

def scan(finalize_only=False):
    d,tr,raw=inputs();all_rows=[];all_events=[];hour=tr.P.to_numpy()
    plan=dict(previous_evidence='fixed_k/comparison.csv: KMeans3/4 select 30.3/29.2%, GMM4 31.1%; highest group is broad load',
        question='Does the upper tail have a stable GPD modelling threshold independent of an arbitrary target percentage?',
        target='hourly maximum of four power slots; absolute high power, not season-adjusted anomalies',
        analysis_period='Jan-Aug, all 5784 valid hours; descriptive EDA, no training/validation split',
        candidates=[int(u) for u in US],candidate_scope='145 is near upper-load boundary; 205 leaves few distinct excesses; no preferred percentile',
        operator='>',event_gap_hours=GAPS,primary_gap_hours=3,
        gap_reason='1h consecutive hours; 3/6h within-day separation sensitivity; 12/24h checks over-merging across days',
        diagnosis='MRL approximate linearity AND shape/modified-scale stability AND observed-vs-fitted tail agreement AND adequate events; no single score winner',
        caveats=['Do not choose u by largest goodness-of-fit p-value or smallest AIC across different exceedance samples.',
                 'KS is a descriptive distance; no nominal p-value under estimated parameters, rounding and dependence.',
                 'Fitted shape<=-0.5 has nonregular likelihood inference; cannot treat ordinary MLE bands as valid.',
                 'Declustering reduces short-range dependence but does not prove independence or stationarity.',
                 'Threshold u is a modelling threshold, not a physical hazard or bill-increase boundary.'],
        manuscript_edited=False)
    (OUT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    if finalize_only:
        results=read('threshold_scan')
        assert len(results)==len(US)*6
    else:
        for u in US:
            y=hour[hour>u]-u
            all_rows.append(dict(series='hourly',u=int(u),gap_hours=0,exceed_hours=len(y),**fit(y,u)))
            for gap in GAPS:
                e=events(tr,u,gap);y=e.P.to_numpy()-u
                all_rows.append(dict(series='event',u=int(u),gap_hours=gap,exceed_hours=int((hour>u).sum()),**fit(y,u)))
                if gap==3:
                    e['u']=int(u);e['gap_hours']=gap;all_events.append(e)
            if (u-144)%10==0:print(json.dumps(dict(stage='threshold_scan',completed=int(u-144),total=len(US))),flush=True)
        results=pd.DataFrame(all_rows);save(results,'threshold_scan');save(pd.concat(all_events,ignore_index=True),'events_gap3')
    ldc=pd.DataFrame(dict(P=np.sort(hour)[::-1],exceedance_fraction=(np.arange(len(hour))+.5)/len(hour)))
    save(ldc,'load_duration')
    # Match exact timestamps, so missing days never create false lag pairs.
    indexed=tr.set_index('timestamp').P
    dep=[]
    for u in [170,176,182,187,194]:
        flag=indexed.gt(u).astype(float);p=flag.mean()
        for lag in range(1,49):
            paired=pd.concat([flag.rename('now'),flag.shift(lag,freq='h').rename('lagged')],axis=1,sort=True).dropna()
            dep.append(dict(u=u,lag_hours=lag,autocorrelation=float(paired['now'].corr(paired.lagged)),
                conditional_exceedance=float(paired.loc[paired.lagged.gt(0),'now'].mean()),
                marginal_exceedance=p,paired_hours=len(paired)))
    save(pd.DataFrame(dep),'exceedance_dependence')
    # Preserve original records; one vote per exact 96-value daily profile is a diagnostic only.
    seen=set();unique=[]
    for date,part in raw.groupby('날짜'):
        profile=tuple(part[['15분','30분','45분','60분']].to_numpy().ravel())
        if profile not in seen:seen.add(profile);unique.extend(part[['15분','30분','45분','60분']].max(axis=1).to_numpy())
    unique=np.asarray(unique);rep=[]
    for u in BOOT_US:
        y=unique[unique>u]-u
        rep.append(dict(u=u,profile_n=len(seen),hour_n=len(unique),**fit(y,u)))
    save(pd.DataFrame(rep),'unique_profile_sensitivity')
    periods=[]
    for month,p in d.groupby('month'):
        for u in [179,186,187,201]:
            periods.append(dict(month=int(month),u=u,n=len(p),exceed_hours=int(p.P.gt(u).sum()),
                exceed_rate=float(p.P.gt(u).mean()),mean_excess=float((p.loc[p.P.gt(u),'P']-u).mean())))
    save(pd.DataFrame(periods),'monthly_fixed_thresholds')
    # Month differences are EDA composition diagnostics, not validation periods.
    monthrows=[]
    for month,part in tr.groupby('month'):
        for u in [179,186,187,201]:
            for series in ['hourly','event']:
                sample=part.P.to_numpy() if series=='hourly' else events(part,u,3).P.to_numpy()
                monthrows.append(dict(month=int(month),series=series,u=u,**fit(sample[sample>u]-u,u)))
    save(pd.DataFrame(monthrows),'monthly_shape_diagnostics')
    metadata=dict(status='scan_passed',sha256=SHA,analysis_n=len(tr),analysis_period='Jan-Aug',
        role='Descriptive EDA; not a predictive evaluation or training threshold',
        unique_daily_profiles=len(seen),days=tr.date.nunique(),numpy=np.__version__,scipy=scipy.__version__,
        manuscript_edited=False,method_sources=['https://georgebv.github.io/pyextremes/user-guide/5-threshold-selection/',
        'https://georgebv.github.io/pyextremes/user-guide/4-peaks-over-threshold/',
        'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html'])
    (OUT/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(results.loc[results.u.isin([160,170,176,182,187,194,200,205])&
                      (results.series.eq('hourly')|results.gap_hours.eq(3))].to_json(orient='records'),flush=True)

def bootstrap():
    d,tr,_=inputs();e=read('events_gap3');scan=read('threshold_scan')
    weeks=tr.week.unique();rng=np.random.default_rng(SEED)
    samples={}
    for u in BOOT_US:
        samples[('hourly',u)]={w:tr.loc[tr.week.eq(w)&tr.P.gt(u),'P'].to_numpy()-u for w in weeks}
        samples[('event',u)]={w:e.loc[e.u.eq(u)&e.week.eq(w),'P'].to_numpy()-u for w in weeks}
    rows=[]
    for iteration in range(N_BOOT):
        chosen=rng.choice(weeks,size=len(weeks),replace=True)
        for (series,u),parts in samples.items():
            y=np.concatenate([parts[w] for w in chosen])
            rows.append(dict(iteration=iteration,series=series,u=u,**fit(y,u,starts=1)))
        if (iteration+1)%10==0:print(json.dumps(dict(stage='week_block_bootstrap',completed=iteration+1,total=N_BOOT)),flush=True)
    b=pd.DataFrame(rows);save(b,'weekly_resampling')
    summaries=[]
    for (series,u),p in b.groupby(['series','u']):
        row=dict(series=series,u=u,runs=len(p),finite_runs=int(p.finite_fit.sum()),nonregular_runs=int(p.nonregular_mle.sum()))
        for metric in ['shape','modified_scale','mean_excess','ks_distance']:
            vals=p.loc[p.finite_fit,metric].dropna()
            for q,label in [(.025,'low'),(.5,'median'),(.975,'high')]:
                row[f'{metric}_{label}']=float(vals.quantile(q)) if len(vals) else np.nan
        summaries.append(row)
    save(pd.DataFrame(summaries),'bootstrap_summary')
    print(json.dumps(dict(stage='bootstrap_complete',fits=len(b)),ensure_ascii=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):
        stage=sys.argv[1] if len(sys.argv)>1 else 'scan'
        if stage=='bootstrap':bootstrap()
        else:scan(finalize_only=stage=='finalize')
