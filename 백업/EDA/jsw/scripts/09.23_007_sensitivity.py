"""Follow up sparse strata and negative weekly centered correlation from 007."""
from pathlib import Path
from datetime import datetime
import json
import sys
import numpy as np
import pandas as pd

BASE=Path(__file__).parent.parent
OUT=BASE/'tables'
P='09.23_007'
cols=['날짜','시간','평균','생산량','기온','풍속','습도','강수량']
df=pd.read_csv(BASE/'../../data/origin/okm_augumented_2021.csv',usecols=cols,encoding='utf-8-sig')
df=df.loc[df['날짜'].lt(20210901)&df['시간'].between(0,23)].copy()
df.index=pd.to_datetime(df['날짜'].astype(str),format='%Y%m%d')+pd.to_timedelta(df['시간'],unit='h')
df=df.sort_index()
df['month']=df.index.month
df['hour']=df.index.hour
df['weekday']=df.index.dayofweek
df['weekend']=df.weekday.ge(5)
df['producing']=df['생산량'].gt(0)
assert df.index.is_unique
def save(rows,name):
    pd.DataFrame(rows).to_csv(OUT/f'{P}_{name}.csv',index=False,encoding='utf-8-sig')
def corr(x,y):
    b=pd.concat([x.rename('x'),y.rename('y')],axis=1).dropna()
    return b.x.corr(b.y) if len(b)>2 and b.x.nunique()>1 and b.y.nunique()>1 else np.nan

if '--resume-august' not in sys.argv:
    rows=[]; contrasts=[]; bins=[]
    coarse=['month','hour','weekend','producing']
    for pop,b0 in [('all',df),('positive',df.loc[df.producing])]:
        for col in ['생산량','기온','풍속','습도','강수량']:
            b=b0.dropna(subset=[col,'평균']).copy()
            for keys,min_n in [(['month','hour','weekday','producing'],2),(coarse,5)]:
                sizes=b.groupby(keys)['평균'].transform('size')
                kept=b.loc[sizes.ge(min_n)]
                residual=kept[[col,'평균']]-kept.groupby(keys)[[col,'평균']].transform('mean')
                rows.append({'population':pop,'variable':col,'controls':'+'.join(keys),'minimum_group_n':min_n,
                    'eligible_n':len(b),'n':len(kept),'groups':kept.groupby(keys).ngroups,
                    'raw_same_sample':corr(kept[col],kept['평균']),
                    'centered_pearson':corr(residual[col],residual['평균'])})
            if col=='생산량':
                continue
            # Direction of high/low weather contrasts within coarse strata, equal group summary.
            for key,part in b.groupby(coarse):
                if len(part)<5 or part[col].nunique()<3: continue
                q1,q3=part[col].quantile([.25,.75])
                if q1>=q3: continue
                low=part.loc[part[col].le(q1)]; high=part.loc[part[col].ge(q3)]
                contrasts.append({'population':pop,'variable':col,**dict(zip(coarse,key)),
                    'n':len(part),'low_n':len(low),'high_n':len(high),
                    'difference':high['평균'].median()-low['평균'].median()})
            # Use the same global quintile edges as relations, but retain coarse controls.
            edges=np.unique(df[col].dropna().quantile(np.linspace(0,1,6)))
            b['bin']=pd.cut(b[col],edges,include_lowest=True)
            sizes=b.groupby(coarse)['평균'].transform('size')
            b['centered']=(b['평균']-b.groupby(coarse)['평균'].transform('mean')).where(sizes.ge(5))
            for label,part in b.groupby('bin',observed=True):
                bins.append({'population':pop,'variable':col,'bin':str(label),'n':len(part),
                    'centered_n':int(part.centered.notna().sum()),'centered_power_mean':part.centered.mean()})
    save(rows,'strata_sensitivity')
    save(contrasts,'weather_within_strata')
    c=pd.DataFrame(contrasts)
    summary=c.groupby(['population','variable']).agg(groups=('difference','size'),
        positive=('difference',lambda x:x.gt(0).sum()),zero=('difference',lambda x:x.eq(0).sum()),
        negative=('difference',lambda x:x.lt(0).sum()),median_difference=('difference','median')).reset_index()
    save(summary,'weather_contrast_summary')
    save(bins,'weather_coarse_bins')
    
    # A weekly lag joins members of the same small month/hour/weekday cell.
    # Mean removal in that small cell can itself create negative dependence.
    lagrows=[]
    for label,keys in [('hour_weekday',['hour','weekday']),('month_hour_weekend',['month','hour','weekend'])]:
        values=df[['평균','생산량','기온','풍속','습도','강수량']]
        residual=values-df.groupby(keys)[list(values.columns)].transform('mean')
        for lag in [1,24,168]:
            shifted=residual.copy(); shifted.index+=pd.Timedelta(hours=lag)
            aligned=shifted.reindex(df.index)
            for col in values:
                mask=aligned[col].notna()&residual['평균'].notna()
                lagrows.append({'controls':label,'variable':col,'lag_hours':lag,'n':int(mask.sum()),
                    'pearson':corr(aligned[col],residual['평균'])})
    save(lagrows,'lag_centering_sensitivity')
    monthly=[]
    for lag in [1,24,168]:
        shifted=df[['평균','생산량','기온']].copy(); shifted.index+=pd.Timedelta(hours=lag)
        aligned=shifted.reindex(df.index)
        for month,part in df.groupby('month'):
            for col in shifted:
                x=aligned.loc[part.index,col]; y=part['평균']
                monthly.append({'month':month,'variable':col,'lag_hours':lag,'n':int((x.notna()&y.notna()).sum()),'pearson':corr(x,y)})
    save(monthly,'monthly_lags')
    
else:
    rows=pd.read_csv(OUT/f'{P}_strata_sensitivity.csv',encoding='utf-8-sig').to_dict('records')
    summary=pd.read_csv(OUT/f'{P}_weather_contrast_summary.csv',encoding='utf-8-sig')
    lagrows=pd.read_csv(OUT/f'{P}_lag_centering_sensitivity.csv',encoding='utf-8-sig').to_dict('records')
# August reversal: summarize existing observations by calendar week and production state.
aug=df.loc[df.month.eq(8)].copy()
aug['week_start']=(aug.index.normalize()-pd.to_timedelta(aug.weekday.to_numpy(),unit='D')).strftime('%Y-%m-%d')
weekly=aug.groupby(['week_start','producing']).agg(n=('평균','size'),days=('날짜','nunique'),
    mean_power=('평균','mean'),temperature_mean=('기온','mean'),production_mean=('생산량','mean')).reset_index()
save(weekly,'august_state_weeks')
assert int(weekly.n.sum())==len(aug)
save([{'population':pop,'variable':col,'n':len(part),
    'pearson':corr(part[col],part['평균'])}
    for pop,part in [('aug_all',aug),('aug_positive',aug.loc[aug.producing]),('aug_zero',aug.loc[~aug.producing])]
    for col in ['생산량','기온']], 'august_state_correlations')
facts={'time':datetime.now().astimezone().isoformat(timespec='seconds'),'rows':len(df),
    'reason':'Sparse exact strata; negative weekly correlation after small-cell centering; August sign reversal',
    'checks':'unique timestamps, August aggregate counts; no row deletion or model tuning'}
(OUT/f'{P}_sensitivity_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
print(pd.DataFrame(rows).round(4).to_string(index=False))
print(summary.to_string(index=False))
print(pd.DataFrame(lagrows).loc[lambda x:x.variable.eq('평균')].round(4).to_string(index=False))
print(weekly.round(3).to_string(index=False))
