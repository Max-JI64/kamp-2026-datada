"""Iterative descriptive EDA: alternative scales -> strata -> repeated profiles."""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd

BASE=Path(__file__).parent.parent
OUT=BASE/'tables'
P='09.23_006'
cols=['날짜','시간','평균','생산량','강수량','기온','15분','30분','45분','60분']
df=pd.read_csv(BASE/'../../data/origin/okm_augumented_2021.csv',encoding='utf-8-sig',usecols=cols)
df=df.loc[df['날짜'].lt(20210901)&df['시간'].between(0,23)].copy()
df['weekday']=pd.to_datetime(df['날짜'].astype(str),format='%Y%m%d').dt.dayofweek+1
df['month']=df['날짜']//100%100
def save(rows,name):
    pd.DataFrame(rows).to_csv(OUT/f'{P}_{name}.csv',index=False,encoding='utf-8-sig')
scales={'raw':lambda x:x,'sqrt':np.sqrt,'log1p':np.log1p,'asinh':np.arcsinh}
comparison=[]
for col in ['생산량','강수량']:
    for pop,part in [('all',df),('positive',df.loc[df[col].gt(0)])]:
        part=part.dropna(subset=[col,'평균'])
        y=part['평균']
        for label,func in scales.items():
            x=func(part[col].astype(float))
            assert np.isfinite(x).all()
            centered=x-x.groupby([part['시간'],part['weekday']]).transform('mean')
            cy=y-y.groupby([part['시간'],part['weekday']]).transform('mean')
            nbin,_=np.histogram(x,bins=10)
            comparison.append({'variable':col,'population':pop,'scale':label,'n':len(x),
                'skew':x.skew(),'largest_equal_width_bin_share':nbin.max()/len(x),
                'zero_share':x.eq(0).mean(),'pearson':x.corr(y),
                'spearman':x.corr(y,method='spearman'),'within_hour_weekday_pearson':centered.corr(cy)})
save(comparison,'scales')

# Rank bins change the comparison question, rather than forcing a distribution.
relationships=[]
for col in ['생산량','강수량']:
    positive=df.loc[df[col].gt(0)].copy()
    positive['bin']=pd.qcut(positive[col],4,duplicates='drop')
    base=df.loc[df[col].eq(0)]
    blocks=[('zero',base)]+[(str(key),part) for key,part in positive.groupby('bin',observed=True)]
    for label,b in blocks:
        relationships.append({'variable':col,'bin':label,'n':len(b),'x_min':b[col].min(),'x_max':b[col].max(),
            'power_mean':b['평균'].mean(),'power_p10':b['평균'].quantile(.1),'power_median':b['평균'].median(),
            'power_p90':b['평균'].quantile(.9),'positive_production_share':b['생산량'].gt(0).mean()})
save(relationships,'rank_bins')

# Within-group rank bins avoid comparing different mixtures of hour and weekday.
positive=df.loc[df['생산량'].gt(0)].copy()
contrasts=[]
for (hour,weekday),b in positive.groupby(['시간','weekday']):
    if len(b)<12 or b['생산량'].nunique()<4:
        continue
    q1,q3=b['생산량'].quantile([.25,.75])
    low=b.loc[b['생산량'].le(q1)]; high=b.loc[b['생산량'].ge(q3)]
    contrasts.append({'hour':hour,'weekday':weekday,'n':len(b),'low_n':len(low),'high_n':len(high),
        'low_production_median':low['생산량'].median(),'high_production_median':high['생산량'].median(),
        'low_power_median':low['평균'].median(),'high_power_median':high['평균'].median(),
        'power_median_difference':high['평균'].median()-low['평균'].median(),
        'spearman':b['생산량'].corr(b['평균'],method='spearman') if b['평균'].nunique()>1 else np.nan})
save(contrasts,'within_strata')

# Exact daily profiles: compare other variables where the whole power profile is fixed.
powercols=['15분','30분','45분','60분']
wide=df.pivot(index='날짜',columns='시간',values=powercols).dropna()
signatures=wide.apply(lambda row:tuple(row),axis=1)
codes,_=pd.factorize(signatures)
mapping=pd.Series(codes,index=wide.index)
df['profile']=df['날짜'].map(mapping)
profiles=[]
for key,b in df.groupby('profile'):
    days=b['날짜'].nunique()
    if days<2: continue
    totals=b.groupby('날짜')['생산량'].sum()
    temp=b.groupby('날짜')['기온'].mean()
    profiles.append({'profile':key,'days':days,'dates':';'.join(map(str,sorted(b['날짜'].unique()))),
        'production_daily_min':totals.min(),'production_daily_max':totals.max(),
        'distinct_daily_production_totals':totals.nunique(),'zero_production_days':int(totals.eq(0).sum()),
        'daily_temperature_min':temp.min(),'daily_temperature_max':temp.max()})
save(profiles,'profile_context')
fixed=[]
for (profile,hour),b in df.groupby(['profile','시간']):
    if len(b)<2: continue
    assert b[powercols].nunique().max()==1
    fixed.append({'profile':profile,'hour':hour,'n':len(b),'production_distinct':b['생산량'].nunique(),
        'production_min':b['생산량'].min(),'production_max':b['생산량'].max(),
        'power':b['평균'].iloc[0],'temperature_distinct':b['기온'].nunique()})
save(fixed,'fixed_power_context')
# Equal total weight per profile to check whether repeated profiles change descriptive association.
counts=df.groupby('profile')['날짜'].transform('nunique')
def wcorr(x,y,w):
    mx=np.average(x,weights=w); my=np.average(y,weights=w)
    return np.average((x-mx)*(y-my),weights=w)/np.sqrt(np.average((x-mx)**2,weights=w)*np.average((y-my)**2,weights=w))
sensitivity=[]
for name,b in [('all',df),('positive',df.loc[df['생산량'].gt(0)])]:
    for scale,func in scales.items():
        x=func(b['생산량'].astype(float)); y=b['평균']
        sensitivity.append({'population':name,'scale':scale,'n':len(b),'ordinary_pearson':x.corr(y),
            'inverse_profile_day_weighted_pearson':wcorr(x,y,1/counts.loc[b.index])})
save(sensitivity,'profile_weight_sensitivity')
c=pd.DataFrame(contrasts); f=pd.DataFrame(fixed); p=pd.DataFrame(profiles)
facts={'time':datetime.now().astimezone().isoformat(timespec='seconds'),'valid_rows':len(df),
    'strata_compared':len(c),'strata_positive_difference':int(c.power_median_difference.gt(0).sum()),
    'strata_zero_difference':int(c.power_median_difference.eq(0).sum()),
    'strata_negative_difference':int(c.power_median_difference.lt(0).sum()),
    'median_stratum_difference':float(c.power_median_difference.median()),
    'repeated_profiles':len(p),'profiles_with_different_production_totals':int(p.distinct_daily_production_totals.gt(1).sum()),
    'fixed_power_groups':len(f),'fixed_power_groups_different_production':int(f.production_distinct.gt(1).sum()),
    'pandas':pd.__version__,'numpy':np.__version__}
(OUT/f'{P}_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(facts,ensure_ascii=False,indent=2))
