"""Descriptive EDA: distributions, conditional associations and repeated shapes."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent
SOURCE = BASE / '../../data/origin/okm_augumented_2021.csv'
OUT = BASE / 'tables'
PREFIX = '09.23_004'
def save(frame, name):
    frame.to_csv(OUT / f'{PREFIX}_{name}.csv',index=False,encoding='utf-8-sig')

raw = pd.read_csv(SOURCE,encoding='utf-8-sig')
# Preserve September as reserved; do not extend prior baseline evaluation.
scope = raw.loc[raw['날짜'].lt(20210901)].copy()
valid = scope['시간'].between(0,23)
df = scope.loc[valid].copy()
df.index = pd.to_datetime(df['날짜'].astype(str),format='%Y%m%d') + pd.to_timedelta(df['시간'],unit='h')
df = df.sort_index()
assert df.index.is_unique
physical = ['15분','30분','45분','60분','평균','생산량','기온','풍속','습도','강수량','공장인원']
rows=[]
for col in scope.columns:
    s=scope[col].dropna()
    q1,q3=s.quantile([.25,.75])
    iqr=q3-q1
    rows.append({'column':col,'dtype':str(scope[col].dtype),'rows':len(scope),'count':len(s),
        'missing':int(scope[col].isna().sum()),'unique':s.nunique(),'zero_rows':int(s.eq(0).sum()),
        'min':s.min(),'p01':s.quantile(.01),'p05':s.quantile(.05),'q1':q1,'median':s.median(),
        'q3':q3,'p95':s.quantile(.95),'p99':s.quantile(.99),'max':s.max(),'mean':s.mean(),
        'std':s.std(),'skew':s.skew() if col in physical else np.nan,
        'iqr_flag_rows':int((s.lt(q1-1.5*iqr)|s.gt(q3+1.5*iqr)).sum()) if col in physical else np.nan})
save(pd.DataFrame(rows),'distributions')
freq=[]
for col in scope.columns:
    for value,count in scope[col].value_counts(dropna=False).head(5).items():
        freq.append({'column':col,'value':value,'rows':count,'share_all_scope_rows':count/len(scope)})
save(pd.DataFrame(freq),'frequent_values')
hist=[]
for col in ['평균','생산량','기온','강수량']:
    s=scope[col].dropna()
    counts,edges=np.histogram(s,bins=10)
    for i,n in enumerate(counts):
        hist.append({'column':col,'lower':edges[i],'upper':edges[i+1],
                     'upper_inclusive':i==len(counts)-1,'rows':int(n),'nonmissing_rows':len(s)})
save(pd.DataFrame(hist),'histogram_tables')
positive=scope.loc[scope['생산량'].gt(0),'생산량']
cuts=positive.quantile([.25,.5,.75]).tolist()
df['production_bin']=pd.cut(df['생산량'],[-np.inf,0,*cuts,np.inf],labels=['zero','positive_q1','positive_q2','positive_q3','positive_q4'])
df['weekday']=df.index.dayofweek+1
df['hour']=df.index.hour
df['peak']=df[['15분','30분','45분','60분']].max(axis=1)
df['spread']=df[['15분','30분','45분','60분']].max(axis=1)-df[['15분','30분','45분','60분']].min(axis=1)
def grouped(keys):
    return df.groupby(keys,observed=True).agg(rows=('평균','size'),production_median=('생산량','median'),
        power_mean=('평균','mean'),power_median=('평균','median'),power_p10=('평균',lambda s:s.quantile(.1)),
        power_p90=('평균',lambda s:s.quantile(.9)),peak_max=('peak','max')).reset_index()
save(grouped(['production_bin']),'production_bins')
save(grouped(['hour','weekday','production_bin']),'conditional_production_bins')
cor=[]
for name,block in [('all_valid',df),('positive_production',df.loc[df['생산량'].gt(0)])]:
    for col in ['생산량','기온','풍속','습도','강수량']:
        b=block[[col,'평균']].dropna()
        cor.append({'scope':name,'variable':col,'rows':len(b),'pearson':b[col].corr(b['평균']),
                    'spearman':b[col].corr(b['평균'],method='spearman')})
# Remove group means for a descriptive within-hour/weekday association, not causality.
for name,b in [('all_valid',df),('positive_production',df.loc[df['생산량'].gt(0)])]:
    centered=b[['생산량','평균']]-b.groupby(['hour','weekday'])[['생산량','평균']].transform('mean')
    cor.append({'scope':name+'_within_hour_weekday','variable':'생산량','rows':len(b),
                'pearson':centered['생산량'].corr(centered['평균']),'spearman':np.nan})
save(pd.DataFrame(cor),'associations')
numeric=['평균','생산량','기온','풍속','습도','강수량','전기요금(계절)','공장인원','인건비']
save(df[numeric].corr(method='spearman').reset_index(names='variable'),'spearman_matrix')
zero=df.loc[df['생산량'].eq(0)].copy()
threshold=float(scope[['15분','30분','45분','60분']].max(axis=1).quantile(.95))
zero['high']=zero['peak'].ge(threshold)
save(zero.groupby(zero.index.to_period('M')).agg(rows=('평균','size'),power_median=('평균','median'),
    power_p90=('평균',lambda s:s.quantile(.9)),high_rows=('high','sum')).reset_index(names='month'),'zero_production_months')
exceptions=zero.loc[zero['high']].groupby(zero.loc[zero['high']].index.date).agg(
    rows=('평균','size'),hour_min=('hour','min'),hour_max=('hour','max'),peak_max=('peak','max'),power_mean=('평균','mean'))
save(exceptions.reset_index(names='date'),'zero_production_high_dates')
shape=[]
for col in ['평균','생산량','spread']:
    if col not in df: continue
    s=df[col]
    shape.append({'metric':col,'rows':len(s),'median':s.median(),'p90':s.quantile(.9),
                  'p99':s.quantile(.99),'max':s.max(),'zeros':int(s.eq(0).sum())})
save(pd.DataFrame(shape),'within_hour_shape')
repeats=[]
for col in ['평균','생산량']:
    previous=df[col].copy(); previous.index+=pd.Timedelta(hours=1)
    p=previous.reindex(df.index)
    ok=p.notna()
    repeated=df[col].eq(p)&ok
    ids=(~repeated).cumsum()
    runs=df.groupby(ids)[col].agg(['size','first'])
    repeats.append({'column':col,'adjacent_pairs':int(ok.sum()),'equal_pairs':int(repeated.sum()),
        'equal_pair_rate':float(repeated[ok].mean()),'longest_equal_run_hours':int(runs['size'].max()),
        'longest_nonzero_equal_run_hours':int(runs.loc[runs['first'].ne(0),'size'].max())})
save(pd.DataFrame(repeats),'repetition')
daily=df.assign(date=df.index.date).pivot(index='date',columns='hour',values='평균').dropna()
identical=daily.groupby(list(daily.columns)).size()
facts={'time':datetime.now().astimezone().isoformat(timespec='seconds'),'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'scope_rows':len(scope),'valid_time_rows':len(df),'reserved_september_rows':int(raw['날짜'].ge(20210901).sum()),
    'positive_production_quartiles':cuts,'descriptive_high_threshold':threshold,
    'complete_days':len(daily),'distinct_daily_mean_profiles':len(identical),
    'days_in_repeated_profiles':int(identical.loc[identical.gt(1)].sum()),
    'largest_identical_profile_group':int(identical.max()),
    'zero_production_high_rows':int(zero['high'].sum()),'zero_production_high_dates':len(exceptions),
    'pandas':pd.__version__,'numpy':np.__version__}
assert grouped(['production_bin'])['rows'].sum()==len(df)
assert sum(x['rows'] for x in hist if x['column']=='평균')==len(scope)
(OUT/f'{PREFIX}_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(facts,ensure_ascii=False,indent=2))
print(pd.DataFrame(rows).loc[lambda x:x.column.isin(['평균','생산량','기온','강수량']),['column','median','mean','p99','max','skew','iqr_flag_rows']].to_string(index=False))
print(grouped(['production_bin']).to_string(index=False))
print(pd.DataFrame(cor).to_string(index=False))
