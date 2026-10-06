"""Approved Analysis 3.1: production changes and observed power variability.
No oracle maximum reconstruction, forecast fitting, backup, or manuscript.
"""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import csv
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'data/origin/okm_augumented_2021.csv'
OUT=ROOT/'Analysis/tables/a01_production_changes'
SHA='8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830'
SLOTS=['15분','30분','45분','60분']
METRICS=['power_delta','abs_power_delta','slot_range']
PAIRS=[('increase','maintain'),('decrease','maintain'),('increase','decrease')]


def save(frame,name):
    frame.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')


def load():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
    raw=pd.read_csv(SOURCE,encoding='utf-8-sig')
    data=raw.loc[raw['날짜'].between(20210101,20210831)&raw['시간'].between(0,23)].copy()
    data['date']=pd.to_datetime(data['날짜'].astype(str),format='%Y%m%d')
    data['timestamp']=data.date+pd.to_timedelta(data['시간'],unit='h')
    data=data.sort_values('timestamp').reset_index(drop=True)
    assert len(data)==5784 and not data.timestamp.duplicated().any()
    data['level']=data[SLOTS].mean(axis=1)
    data['slot_range']=data[SLOTS].max(axis=1)-data[SLOTS].min(axis=1)
    daily={date:hashlib.sha256(cell[SLOTS].to_numpy(dtype=np.int64).tobytes()).hexdigest()
           for date,cell in data.groupby('date')}
    data['profile']=data.date.map(daily)
    data['month']=data.date.dt.month
    data['weekend']=data.date.dt.dayofweek.ge(5).astype(int)
    data['week']=data.date-pd.to_timedelta(data.date.dt.dayofweek,unit='D')
    old=data.set_index('timestamp').reindex(data.timestamp-pd.Timedelta(hours=1))
    data['past_production']=old['생산량'].to_numpy()
    data['past_level']=old.level.to_numpy()
    data['past_profile']=old.profile.to_numpy()
    part=data.dropna(subset=['past_production','past_level']).copy().reset_index(drop=True)
    part['production_delta']=part['생산량']-part.past_production
    part['power_delta']=part.level-part.past_level
    part['abs_power_delta']=part.power_delta.abs()
    part['change']=np.select([part.production_delta.gt(0),part.production_delta.lt(0)],['increase','decrease'],default='maintain')
    current=part['생산량'].gt(0)
    previous=part.past_production.gt(0)
    part['transition']=np.select([~previous&current,previous&~current,previous&current],
                                ['zero_to_positive','positive_to_zero','positive_to_positive'],default='zero_to_zero')
    # Joint current/previous day profile reflects the boundary used by differences.
    part['pair_profile']=part.past_profile+'|'+part.profile
    counts=part[['date','pair_profile']].drop_duplicates().pair_profile.value_counts()
    part['profile_weight']=1/part.pair_profile.map(counts)
    return data,part


def summaries(part):
    rows=[]
    for label,keys in [('change',['change']),('transition',['transition']),
                       ('change_transition',['change','transition']),
                       ('hour_change',['weekend','시간','change']),('month_change',['month','change'])]:
        for key,cell in part.groupby(keys,sort=True):
            key=key if isinstance(key,tuple) else (key,)
            for metric in METRICS:
                rows.append(dict(scope=label,**dict(zip(keys,key)),metric=metric,n=len(cell),days=cell.date.nunique(),
                    mean=cell[metric].mean(),median=cell[metric].median(),
                    q10=cell[metric].quantile(.1),q90=cell[metric].quantile(.9)))
    return pd.DataFrame(rows)


def contrast(part,a,b,metric,mode,boot=False):
    sample=part.loc[part.change.isin([a,b])].copy()
    # Only observed matched conditions; no regression or extrapolation.
    cells=sample.groupby(['month','weekend','시간','change']).date.nunique().unstack('change',fill_value=0)
    if a not in cells or b not in cells:
        return None
    support=cells.index[(cells[a]>=2)&(cells[b]>=2)]
    cell_keys=list(zip(sample.month,sample.weekend,sample['시간']))
    support_map={key:i for i,key in enumerate(support)}
    keep=np.array([key in support_map for key in cell_keys])
    sample=sample.loc[keep].copy()
    if sample.empty:
        return None
    cell_ids=np.array([support_map[key] for key,k in zip(cell_keys,keep) if k])
    side=sample.change.eq(b).to_numpy().astype(int)
    slot=cell_ids*2+side
    n_cells=len(support)
    base_w=np.ones(len(sample)) if mode=='date_hour' else sample.profile_weight.to_numpy()
    value=sample[metric].to_numpy()
    # Common calendar composition: pooled eligible-hour weight within each cell.
    fixed_cell_w=np.bincount(cell_ids,weights=base_w,minlength=n_cells)
    def estimate(extra=None):
        w=base_w if extra is None else base_w*extra
        count=np.bincount(slot,weights=w,minlength=n_cells*2).reshape(-1,2)
        total=np.bincount(slot,weights=w*value,minlength=n_cells*2).reshape(-1,2)
        ok=(count>0).all(axis=1)
        if not ok.any():
            return np.nan,np.nan
        means=total[ok]/count[ok]
        averaged=np.average(means,axis=0,weights=fixed_cell_w[ok])
        return float(averaged[0]),float(averaged[1])
    ma,mb=estimate()
    result=dict(a=a,b=b,metric=metric,weighting=mode,matched_cells=n_cells,
        n_a=int((side==0).sum()),n_b=int((side==1).sum()),
        eligible_a=int(part.change.eq(a).sum()),eligible_b=int(part.change.eq(b).sum()),
        days_a=sample.loc[side==0,'date'].nunique(),days_b=sample.loc[side==1,'date'].nunique(),
        adjusted_a=ma,adjusted_b=mb,difference=ma-mb,
        relative_difference=(ma/mb-1) if metric!='power_delta' and mb>0 else np.nan)
    if boot:
        week_id=pd.factorize(sample.week)[0]
        n_weeks=week_id.max()+1
        rng=np.random.default_rng(3103)
        estimates=[]
        for _ in range(800):
            multiplicity=rng.multinomial(n_weeks,np.full(n_weeks,1/n_weeks))
            va,vb=estimate(multiplicity[week_id])
            estimates.append(va-vb)
        result['week_boot_q025'],result['week_boot_q975']=np.nanquantile(estimates,[.025,.975])
        result['bootstrap_finite']=int(np.isfinite(estimates).sum())
    return result


def verify(data,part):
    independent={}
    with SOURCE.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            date,hour=int(row['날짜']),int(row['시간'])
            if 20210101<=date<=20210831 and 0<=hour<=23:
                stamp=pd.Timestamp(str(date))+pd.Timedelta(hours=hour)
                independent[stamp]=(int(row['생산량']),tuple(int(row[c]) for c in SLOTS))
    checked=0
    for row in part.itertuples():
        prod,q=independent[row.timestamp]
        past,pq=independent[row.timestamp-pd.Timedelta(hours=1)]
        assert row.production_delta==prod-past
        assert row.power_delta==(sum(q)-sum(pq))/4
        assert row.slot_range==max(q)-min(q)
        checked+=1
    expected=sum(stamp-pd.Timedelta(hours=1) in independent for stamp in independent)
    assert checked==expected and len(data)==5784
    return dict(raw_independent_pairs=checked,unpaired_hours=len(data)-len(part),source_preserved=True)


def followup(part):
    """Observed confounding: unchanged-positive has only four observations."""
    transition=part.copy()
    transition['change']=transition.transition
    rows=[]
    comparisons=[('positive_to_positive','zero_to_zero'),('zero_to_positive','positive_to_positive'),
                 ('positive_to_zero','positive_to_positive')]
    for a,b in comparisons:
        for metric in METRICS:
            for mode in ('date_hour','profile_balanced'):
                row=contrast(transition,a,b,metric,mode,boot=True)
                if row:
                    rows.append(row)
    correlations=[]
    positive=part.loc[part.transition.eq('positive_to_positive')].copy()
    for scope,cell in [('positive_both',positive),('same_day_positive_both',positive.loc[positive['시간'].ne(0)])]:
        ids=pd.factorize(pd.MultiIndex.from_frame(cell[['month','weekend','시간']]))[0]
        for mode in ('date_hour','profile_balanced'):
            w=np.ones(len(cell)) if mode=='date_hour' else cell.profile_weight.to_numpy()
            for metric,xcol in [('power_delta','production_delta'),('abs_power_delta','abs_production_delta'),
                                ('slot_range','abs_production_delta')]:
                x=cell.production_delta.to_numpy()
                if xcol=='abs_production_delta':
                    x=abs(x)
                rx,ry=rankdata(x),rankdata(cell[metric])
                for value in (rx,ry):
                    means=np.bincount(ids,weights=value*w)/np.bincount(ids,weights=w)
                    value-=means[ids]
                rho=float(np.sum(w*rx*ry)/np.sqrt(np.sum(w*rx**2)*np.sum(w*ry**2)))
                correlations.append(dict(scope=scope,weighting=mode,metric=metric,n=len(cell),
                    calendar_centered_rank_correlation=rho))
    save(pd.DataFrame(rows),'transition_matched_contrasts')
    save(pd.DataFrame(correlations),'positive_production_correlations')
    stability=[]
    for label,subset in [('same_day',transition.loc[transition['시간'].ne(0)])]+[
            (f'exclude_month_{m}',transition.loc[transition.month.ne(m)]) for m in range(1,9)]:
        for a,b in comparisons:
            for metric in ('abs_power_delta','slot_range'):
                row=contrast(subset,a,b,metric,'date_hour')
                if row:
                    stability.append(dict(sensitivity=label,**row))
    save(pd.DataFrame(stability),'transition_stability')
    # Independent per-cell grouping directly from the observation CSV.
    raw_groups={}
    with (OUT/'observations.csv').open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            key=(int(row['month']),int(row['weekend']),int(row['시간']))
            cell=raw_groups.setdefault(key,{})
            cell.setdefault(row['transition'],[]).append((row['date'],float(row['abs_power_delta'])))
    for result in rows:
        if result['weighting']!='date_hour' or result['metric']!='abs_power_delta':
            continue
        a,b=result['a'],result['b']
        wa=wb=total=0.0
        na=nb=0
        for cell in raw_groups.values():
            ca,cb=cell.get(a,[]),cell.get(b,[])
            if len({x[0] for x in ca})<2 or len({x[0] for x in cb})<2:
                continue
            weight=len(ca)+len(cb)
            wa+=weight*sum(x[1] for x in ca)/len(ca)
            wb+=weight*sum(x[1] for x in cb)/len(cb)
            total+=weight
            na+=len(ca)
            nb+=len(cb)
        assert na==result['n_a'] and nb==result['n_b']
        np.testing.assert_allclose([wa/total,wb/total],[result['adjusted_a'],result['adjusted_b']],rtol=1e-12)
    follow=dict(previous_result='2292 of2296 unchanged-production hours have zero in both records; only4 unchanged-positive. Delta-only effect confounded.',
        questions=['Which zero/nonzero transitions account for observed volatility?',
                   'Within positive production, is the magnitude/direction of production change connected to power change after calendar control?'],
        comparison='Same fixed calendar matching, plus rank associations centered within exact month-hour-weekend cells.',
        not_causal=True,no_threshold=True,no_forecast=True,unchanged_positive_conclusion='Too few records for stable matched contrast; do not infer equivalence.')
    (OUT/'followup_contract.json').write_text(json.dumps(follow,indent=2),encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    contract=dict(question='Within the same month/hour/weekday-weekend, do production increases/decreases/unchanged records have different observed power changes?',
        previous_evidence='Current EDA: weekday noon drop/13h rise; production peak19h versus power8h; recurring within-hour differences.',
        scope='2021 Jan-Aug current EDA valid241 days; exact 1h pairing only, including valid midnight boundaries.',
        metrics=METRICS,definitions={'power_delta':'current unrounded four-slot mean minus exact 1h past mean',
        'abs_power_delta':'absolute value of power_delta','slot_range':'current max-min across four slots'},
        production_groups='sign of exact production difference: >0,<0,=0; even 1-unit production difference retained, actual magnitudes described',
        matching='month x hour x weekend; at least two dates in EACH compared group; pooled common-cell weights, fixed prior to bootstrap',
        repeat_sensitivity='Inverse number of dates sharing joint current/previous day power profile',
        zero_sensitivity='0->positive,positive->0,positive->positive,0->0; not operational closure labels',
        robustness='same-day only; both production records positive; leave each month out; week cluster bootstrap800 seed3103',
        intervals='Exploratory week-resampling sensitivity, not proof of independent weeks or causal/statistical significance; duplicated profiles retained',
        pvalues=False,forecast_training=False,oracle_maximum_reconstruction=False,peak_threshold=None,manuscript=False,
        backup_access=False,stop='Resolve matched relationships and confounding, no desired-result selection.',source_sha256=SHA)
    (OUT/'contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding='utf-8')
    data,part=load()
    save(part[['timestamp','date','month','weekend','시간','생산량','past_production',
        'production_delta','change','transition','level','past_level',*METRICS,'profile_weight']],'observations')
    save(summaries(part),'summaries')
    rows=[]
    for scope,cell in [('all',part),('same_day',part.loc[part['시간'].ne(0)]),
                       ('positive_both',part.loc[part.transition.eq('positive_to_positive')])]:
        for a,b in PAIRS:
            for metric in METRICS:
                for mode in ('date_hour','profile_balanced'):
                    row=contrast(cell,a,b,metric,mode,boot=scope=='all')
                    if row:
                        rows.append(dict(scope=scope,**row))
    results=pd.DataFrame(rows)
    save(results,'matched_contrasts')
    stability=[]
    for month in range(1,9):
        cell=part.loc[part.month.ne(month)]
        for a,b in PAIRS:
            for metric in METRICS:
                row=contrast(cell,a,b,metric,'date_hour')
                if row:
                    stability.append(dict(excluded_month=month,**row))
    save(pd.DataFrame(stability),'leave_month_out')
    followup(part)
    verification=verify(data,part)
    verification.update(status='passed',n=len(part),days=part.date.nunique(),
        production_group_counts=part.change.value_counts().to_dict(),
        transition_counts=part.transition.value_counts().to_dict(),
        transition_matching_independently_verified=True,
        source_sha256=SHA,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outputs_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.csv')})
    (OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(verification,ensure_ascii=False))
    print('Approved production-change analysis and confounding followup complete.')


if __name__=='__main__':
    main()
