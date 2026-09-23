"""EDA scale comparison only; no fitted model, exclusions or row-level output."""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd

BASE=Path(__file__).parent.parent
OUT=BASE/'tables'
PREFIX='09.23_005'
df=pd.read_csv(BASE/'../../data/origin/okm_augumented_2021.csv',encoding='utf-8-sig',
               usecols=['날짜','시간','생산량','강수량','평균'])
df=df.loc[df['날짜'].lt(20210901)].copy()
def save(rows,name):
    pd.DataFrame(rows).to_csv(OUT/f'{PREFIX}_{name}.csv',index=False,encoding='utf-8-sig')
stats=[]; bins=[]; checks=[]
for col in ['생산량','강수량']:
    original=df[col].dropna()
    assert original.ge(0).all()
    for population,x in [('all',original),('positive',original.loc[original.gt(0)])]:
        for scale,s in [('raw',x),('log1p',np.log1p(x))]:
            stats.append({'variable':col,'population':population,'scale':scale,'n':len(s),
                'zero_n':int(s.eq(0).sum()),'mean':s.mean(),'std':s.std(),'skew':s.skew(),
                'min':s.min(),'p25':s.quantile(.25),'median':s.median(),'p75':s.quantile(.75),
                'p95':s.quantile(.95),'p99':s.quantile(.99),'max':s.max()})
            counts,edges=np.histogram(s,bins=10)
            assert int(counts.sum())==len(s)
            for i,n in enumerate(counts):
                bins.append({'variable':col,'population':population,'scale':scale,
                    'lower':edges[i],'upper':edges[i+1],'upper_inclusive':i==9,'n':int(n),
                    'share':n/len(s),'original_lower':np.expm1(edges[i]) if scale=='log1p' else edges[i],
                    'original_upper':np.expm1(edges[i+1]) if scale=='log1p' else edges[i+1]})
        transformed=np.log1p(x)
        err=float(np.max(np.abs(np.expm1(transformed)-x)))
        assert np.allclose(np.expm1(transformed),x)
        assert transformed.eq(0).sum()==x.eq(0).sum()
        checks.append({'variable':col,'population':population,'n':len(x),'inverse_max_abs_error':err})
save(stats,'distribution');save(bins,'bins');save(checks,'checks')
valid=df.loc[df['시간'].between(0,23)].copy()
valid['weekday']=pd.to_datetime(valid['날짜'].astype(str),format='%Y%m%d').dt.dayofweek+1
relations=[]
for col in ['생산량','강수량']:
    for population,b in [('all',valid),('positive',valid.loc[valid[col].gt(0)])]:
        b=b.dropna(subset=[col,'평균'])
        for scale in ['raw','log1p']:
            x=b[col] if scale=='raw' else np.log1p(b[col])
            y=b['평균']
            cx=x-x.groupby([b['시간'],b['weekday']]).transform('mean')
            cy=y-y.groupby([b['시간'],b['weekday']]).transform('mean')
            relations.append({'variable':col,'population':population,'scale':scale,'n':len(b),
                 'pearson':x.corr(y),'spearman':x.corr(y,method='spearman'),
                 'within_hour_weekday_pearson':cx.corr(cy)})
for col in ['생산량','강수량']:
    for pop in ['all','positive']:
        r=[row for row in relations if row['variable']==col and row['population']==pop]
        assert np.isclose(r[0]['spearman'],r[1]['spearman'])
save(relations,'associations')
facts={'time':datetime.now().astimezone().isoformat(timespec='seconds'),
       'scope':'2021-01-01 through 2021-08-31; September excluded',
       'distribution_rows':len(df),'valid_time_rows':len(valid),
       'missing_production':int(df['생산량'].isna().sum()),'missing_rain':int(df['강수량'].isna().sum()),
       'pandas':pd.__version__,'numpy':np.__version__,'transform':'natural log(1+x)',
       'row_level_derivatives_saved':False}
(OUT/f'{PREFIX}_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
print(pd.DataFrame(stats)[['variable','population','scale','n','zero_n','skew']].to_json(orient='records',force_ascii=True))
print(pd.DataFrame(relations).to_json(orient='records',force_ascii=True))
