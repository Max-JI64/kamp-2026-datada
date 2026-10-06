"""Approved 3.3: contemporaneous weather-power associations by calendar/production.
No forecasting, new peak definition, backup, or manuscript.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr
from a01_production_changes import ROOT,SOURCE,SHA,SLOTS,load
from a02_prior_power_signals import residuals,weighted_corr

OUT=ROOT/'Analysis/tables/a03_conditioned_weather'
WEATHER=['기온','풍속','습도','강수량']


def save(frame,name):
    frame.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')


def prepare():
    data,_=load()
    data['production_positive']=data['생산량'].gt(0).astype(int)
    freq=data[['date','profile']].drop_duplicates().profile.value_counts()
    data['profile_weight']=1/data.profile.map(freq)
    return data


def association(sample,feature,adjustment,mode):
    needed=[feature,'level','생산량']+ (WEATHER if adjustment=='joint' else [])
    sample=sample.dropna(subset=list(dict.fromkeys(needed))).copy()
    keys=['month','weekend','시간']
    if adjustment!='raw':
        counts=sample.groupby(keys).date.transform('nunique')
        sample=sample.loc[counts.ge(3)].copy()
    if len(sample)<10:
        return None
    w=np.ones(len(sample)) if mode=='date_hour' else sample.profile_weight.to_numpy()
    x=rankdata(sample[feature])/len(sample); y=rankdata(sample.level)/len(sample)
    if adjustment=='raw':
        value=weighted_corr(x,y,w)
    else:
        cell=pd.factorize(pd.MultiIndex.from_frame(sample[keys]))[0]
        controls=None
        if adjustment in ('production','joint'):
            rankprod=rankdata(sample['생산량'])/len(sample)
            controls=np.column_stack([sample.production_positive,rankprod,rankprod**2,rankprod**3])
        if adjustment=='joint':
            other=[rankdata(sample[c])/len(sample) for c in WEATHER if c!=feature]
            controls=np.column_stack([controls]+other)
        res=residuals(np.column_stack([x,y]),cell,w,controls)
        value=weighted_corr(res[:,0],res[:,1],w)
    return dict(variable=feature,adjustment=adjustment,weighting=mode,n=len(sample),
        days=sample.date.nunique(),correlation=value)


def scopes(data):
    weekday=data.loc[data.weekend.eq(0)]
    return {'weekday':weekday,'weekday_positive':weekday.loc[weekday.production_positive.eq(1)],
        'weekday_zero':weekday.loc[weekday.production_positive.eq(0)],
        'all':data,'all_positive':data.loc[data.production_positive.eq(1)],
        'august_all':data.loc[data.month.eq(8)],
        'august_positive':data.loc[data.month.eq(8)&data.production_positive.eq(1)]}


def reproduce_eda(data):
    table=pd.read_csv(ROOT/'EDA/tables/restart_relations_august_context.csv',encoding='utf-8-sig').set_index('population')
    subset=data.loc[data.month.eq(8)]
    for label,sample in [('all',subset),('positive',subset.loc[subset.production_positive.eq(1)]),
                         ('zero',subset.loc[subset.production_positive.eq(0)])]:
        r=spearmanr(sample['기온'],sample['평균']).statistic
        assert len(sample)==table.loc[label,'n']
        np.testing.assert_allclose(r,table.loc[label,'spearman'],atol=1e-12)


def raw_check(data):
    reference={}
    with SOURCE.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            date,hour=int(row['날짜']),int(row['시간'])
            if 20210101<=date<=20210831 and 0<=hour<=23:
                stamp=pd.Timestamp(str(date))+pd.Timedelta(hours=hour)
                reference[stamp]=row
    for row in data.to_dict('records'):
        source=reference[row['timestamp']]
        assert row['level']==sum(int(source[c]) for c in SLOTS)/4
        for column in WEATHER+['생산량']:
            expected=float(source[column]) if source[column].strip() else np.nan
            np.testing.assert_allclose(row[column],expected,rtol=0,atol=0,equal_nan=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    contract=dict(previous='Current EDA August temp-power rho-.076 overall vs+.599 production-positive383hours; weekday low-power747hours include92production-positive.',
        question='Does weather-power association remain within calendar and observed production conditions? What explains aggregate/subgroup difference?',
        period='Jan-Aug2021 normal241days5784hours; primary weekday171days4104hours; no zero/low-value filtering.',
        outcome='Unrounded mean of four15min values at same time; descriptive contemporaneous association only.',
        variables=WEATHER,calendar='month x hour x weekday/weekend; adjusted groups require>=3distinct dates per cell.',
        adjustment='Rank correlations raw -> same-calendar residual -> production0/positive plus cubic rank-production -> other three rank-weather variables.',
        sample='Each variable uses its observed pairs except joint analyses use common complete four-weather records; count missing per scope.',
        sensitivity='Daily array inverse-frequency; all/weekdays/positive/zero; per-month and leave-month-out; no pvalues or forecasting.',
        decision='If monthly signs or adjustments differ, diagnose composition/production/weather co-variation before interpreting; report mixed results without seeking stronger correlations.',
        next_scope='Analysis only, do not start power prediction; any Modeling inputs must respect information time.',
        backup=False,forecast=False,manuscript=False,new_peak_threshold=False,source_sha256=SHA)
    (OUT/'contract.json').write_text(json.dumps(contract,indent=2,ensure_ascii=False),encoding='utf-8')
    data=prepare(); raw_check(data); reproduce_eda(data)
    save(data[['timestamp','date','month','weekend','시간','level','생산량','production_positive']+WEATHER+['profile_weight']], 'observations')
    rows=[]
    for scope,sample in scopes(data).items():
        for feature in WEATHER:
            for adjustment in ('raw','calendar','production','joint'):
                for mode in ('date_hour','profile_balanced'):
                    result=association(sample,feature,adjustment,mode)
                    if result:
                        rows.append(dict(scope=scope,**result))
    results=pd.DataFrame(rows); save(results,'associations')
    monthly=[]
    for scope,sample in scopes(data).items():
        if scope not in ('weekday','weekday_positive','weekday_zero','all_positive'):
            continue
        for month,subset in sample.groupby('month'):
            for feature in WEATHER:
                for adjustment in ('raw','production','joint'):
                    result=association(subset,feature,adjustment,'date_hour')
                    if result:
                        monthly.append(dict(scope=scope,month=month,**result))
    save(pd.DataFrame(monthly),'monthly')
    stability=[]
    for scope,sample in scopes(data).items():
        if scope not in ('weekday','weekday_positive','weekday_zero'):
            continue
        for month in range(1,9):
            for feature in WEATHER:
                result=association(sample.loc[sample.month.ne(month)],feature,'joint','date_hour')
                if result:
                    stability.append(dict(scope=scope,exclude_month=month,**result))
    save(pd.DataFrame(stability),'leave_month_out')
    counts=[]
    for scope,sample in scopes(data).items():
        counts.append(dict(scope=scope,n=len(sample),days=sample.date.nunique(),complete=int(sample[WEATHER].notna().all(axis=1).sum()),
            **{f'{column}_missing':int(sample[column].isna().sum()) for column in WEATHER}))
    save(pd.DataFrame(counts),'sample_counts')
    correlations=data[WEATHER].corr(method='spearman'); save(correlations.reset_index(names='variable'),'weather_correlations')
    # Independent explicit-dummy residualization on January weekdays verifies
    # the method without interpreting any model as a future prediction.
    sample=data.loc[data.month.eq(1)&data.weekend.eq(0)].dropna(subset=WEATHER)
    cell=pd.factorize(pd.MultiIndex.from_frame(sample[['month','weekend','시간']]))[0]
    prod=rankdata(sample['생산량'])/len(sample)
    controls=np.column_stack([sample.production_positive,prod,prod**2,prod**3]+
        [rankdata(sample[c])/len(sample) for c in WEATHER if c!='기온'])
    values=np.column_stack([rankdata(sample['기온']),rankdata(sample.level)])/len(sample)
    w=sample.profile_weight.to_numpy()
    res=residuals(values,cell,w,controls)
    design=np.column_stack([np.eye(cell.max()+1)[cell],controls])
    beta=np.linalg.lstsq(design*np.sqrt(w[:,None]),values*np.sqrt(w[:,None]),rcond=None)[0]
    np.testing.assert_allclose(res,values-design@beta,atol=1e-10)
    verification=dict(status='passed',source_sha256=SHA,raw_hours_checked=len(data),eda_august_correlations_reproduced=True,
        joint_calendar_residual_vs_explicit_dummy_verified=True,zero_records_preserved=True,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outputs_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.csv')})
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(results.loc[results.scope.isin(['weekday','weekday_positive','weekday_zero','august_positive'])&results.weighting.eq('date_hour')].to_json(orient='records'))
    print(pd.DataFrame(counts).to_json(orient='records'))
    print(json.dumps(verification))


if __name__=='__main__':
    main()
