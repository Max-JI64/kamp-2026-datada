"""Minimal followup: common rows, monthly differences, and date-level variation."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from a03_conditioned_weather import OUT,WEATHER,association,save
from a02_prior_power_signals import residuals,weighted_corr


def adjusted(sample,date_control=False,prod_form='cubic',other_weather=True):
    sample=sample.dropna(subset=WEATHER).copy()
    counts=sample.groupby(['month','weekend','시간']).date.transform('nunique')
    sample=sample.loc[counts.ge(3)]
    cells=pd.factorize(pd.MultiIndex.from_frame(sample[['month','weekend','시간']]))[0]
    p=rankdata(sample['생산량'])/len(sample)
    controls=np.column_stack([sample.production_positive,p])
    if prod_form=='cubic':
        controls=np.column_stack([controls,p**2,p**3])
    if other_weather:
        controls=np.column_stack([controls]+[rankdata(sample[c])/len(sample) for c in WEATHER if c!='기온'])
    if date_control:
        date_ids=pd.factorize(sample.date)[0]
        # Date-level nuisance controls identify within-date contrasts, not the
        # same estimand as across-date thermal/weather load association.
        controls=np.column_stack([controls,np.eye(date_ids.max()+1)[date_ids]])
    values=np.column_stack([rankdata(sample['기온']),rankdata(sample.level)])/len(sample)
    w=np.ones(len(sample))
    res=residuals(values,cells,w,controls)
    return dict(n=len(sample),days=sample.date.nunique(),correlation=weighted_corr(res[:,0],res[:,1],w))


def main():
    manifest=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    source=OUT/'observations.csv'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['outputs_sha256']['observations.csv']
    data=pd.read_csv(source,encoding='utf-8-sig',parse_dates=['timestamp','date'])
    data['week']=data.date-pd.to_timedelta(data.date.dt.dayofweek,unit='D')
    weekday=data.loc[data.weekend.eq(0)]
    rows=[]
    for scope,sample in [('weekday',weekday),('weekday_positive',weekday.loc[weekday.production_positive.eq(1)])]:
        sample=sample.dropna(subset=WEATHER)
        sample=sample.loc[sample.groupby(['month','weekend','시간']).date.transform('nunique').ge(3)]
        for feature in WEATHER:
            for adjustment in ('raw','calendar','production','joint'):
                result=association(sample,feature,adjustment,'date_hour')
                rows.append(dict(scope=scope,**result))
    save(pd.DataFrame(rows),'common_row_associations')
    monthly=[]
    positive=weekday.loc[weekday.production_positive.eq(1)]
    for month,sample in positive.groupby('month'):
        for label,date_control,prod_form in [('joint',False,'cubic'),('linear_production',False,'linear'),('within_date_joint',True,'cubic')]:
            monthly.append(dict(month=month,check=label,**adjusted(sample,date_control,prod_form)))
    save(pd.DataFrame(monthly),'temperature_monthly_diagnostics')
    weeks=[]
    # Temperature is the predeclared focal variable. All eight months are
    # checked; do not keep only a favorable summer/month result.
    for month,sample in positive.groupby('month'):
        for week in sample.week.unique():
            subset=sample.loc[sample.week.ne(week)]
            weeks.append(dict(month=month,excluded_week=str(pd.Timestamp(week).date()),**adjusted(subset)))
    save(pd.DataFrame(weeks),'temperature_leave_week_out')
    weather=[]
    for month,sample in positive.groupby('month'):
        sample=sample.dropna(subset=WEATHER)
        sample=sample.loc[sample.groupby(['month','weekend','시간']).date.transform('nunique').ge(3)]
        cells=pd.factorize(pd.MultiIndex.from_frame(sample[['month','weekend','시간']]))[0]
        p=rankdata(sample['생산량'])/len(sample)
        c=np.column_stack([p,p**2,p**3])
        values=np.column_stack([rankdata(sample[name]) for name in WEATHER])/len(sample)
        res=residuals(values,cells,np.ones(len(sample)),c)
        for j,name in enumerate(WEATHER[1:],1):
            weather.append(dict(month=month,other_weather=name,n=len(sample),
                correlation=weighted_corr(res[:,0],res[:,j],np.ones(len(sample)))))
    save(pd.DataFrame(weather),'conditional_temperature_weather')
    # Day-level diagnostics retain only daily averages during positive-production
    # observations; summarize selected-hour composition, not daily total power.
    daily=positive.groupby('date').agg(month=('month','first'),hours=('level','size'),
        power=('level','mean'),temperature=('기온','mean'),production=('생산량','mean')).reset_index()
    save(daily,'positive_day_means')
    ds=[]
    from scipy.stats import spearmanr
    for month,sample in daily.groupby('month'):
        ds.append(dict(month=month,days=len(sample),temperature_power_rho=spearmanr(sample.temperature,sample.power).statistic))
    save(pd.DataFrame(ds),'daily_temperature_relations')
    contract=dict(previous='Weekday temperature rho+.227 falls to-.004 after calendar/production/otherweather; August positive+.286 survives; monthly signs differ.',
        checks='Fixed common-row comparisons; all8monthly positive groups with date fixed effects and linear-vs-cubic production; leave each week out; conditional weather co-variation; daily positive-record means.',
        cautions='Date-fixed relation is within-date variation, cannot erase genuine between-day weather effect; daily positive means have different selected hours; no causal or predictive claims.',
        no_forecast=True,no_threshold_search=True,no_backup=True)
    (OUT/'diagnostic_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    verify=dict(status='passed',source_observations_unchanged=True,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outputs_sha256={name:hashlib.sha256((OUT/f'{name}.csv').read_bytes()).hexdigest() for name in [
            'common_row_associations','temperature_monthly_diagnostics','temperature_leave_week_out','conditional_temperature_weather','positive_day_means','daily_temperature_relations']})
    (OUT/'diagnostic_verification.json').write_text(json.dumps(verify,indent=2),encoding='utf-8')
    print(pd.DataFrame(monthly).to_json(orient='records'));print(pd.DataFrame(ds).to_json(orient='records'))
    print(pd.DataFrame(weeks).groupby('month').correlation.agg(['min','max']).to_json())
    print(pd.DataFrame(weather).to_json(orient='records'))


if __name__=='__main__':
    main()
