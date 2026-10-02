"""Independently audit monthly plotted medians against the original CSV."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'data/origin/okm_augumented_2021.csv'
TAB=ROOT/'report/EDA/tables/daily_repetition'
SHA='8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830'
MANUSCRIPT=ROOT/'report/10.02_002_EDA_새원고.md'

def save(name,d):
    d.to_csv(TAB/(name+'.csv'),index=False,encoding='utf-8-sig')

def main():
    before=hashlib.sha256(MANUSCRIPT.read_bytes()).hexdigest()
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
    raw=pd.read_csv(SOURCE,encoding='utf-8-sig')
    v=raw.loc[raw['날짜'].between(20210101,20210831)&raw['시간'].between(0,23)].copy()
    slots=['15분','30분','45분','60분']
    assert len(v)==5784 and not v.duplicated(['날짜','시간']).any()
    assert v.groupby('날짜').size().eq(24).all()
    np.testing.assert_array_equal(np.floor(v[slots].to_numpy().mean(axis=1)+.5),v['평균'])
    date=pd.to_datetime(v['날짜'].astype(str),format='%Y%m%d')
    v['month']=date.dt.month
    v['group']=np.where(date.dt.dayofweek<5,'weekday','weekend')
    v['hour_max']=v[slots].max(axis=1)
    v['date_positive']=v.groupby('날짜')['생산량'].transform(lambda x:x.gt(0).any())
    # Fresh grouping directly from source rows, no matrix/table helper imports.
    cells=[]
    for (m,g,h),p in v.groupby(['month','group','시간']):
        values=p['평균'].to_numpy()
        cells.append({'month':int(m),'group':g,'hour':int(h),'n_raw':len(values),
            'median_raw':float(np.median(values)),'mean_raw':float(np.mean(values)),
            'min_raw':int(values.min()),'max_raw':int(values.max()),
            'max_four_slots':int(p.hour_max.max()),
            'raw_ge_200':int(p['평균'].ge(200).sum())})
    cells=pd.DataFrame(cells)
    stored=pd.read_csv(TAB/'monthly_hourly_distribution.csv',encoding='utf-8-sig')
    check=cells.merge(stored,on=['month','group','hour'],how='outer',validate='one_to_one',indicator=True)
    assert len(check)==384 and check['_merge'].eq('both').all()
    assert check.n_raw.eq(check.n).all()
    np.testing.assert_array_equal(check.median_raw,check['median'])
    save('independent_monthly_cells',cells)
    month_rows=[];hours=[];production=[]
    for m in [7,8]:
        for g in ['all','weekday','weekend']:
            p=v.loc[v.month.eq(m)]
            if g!='all':p=p.loc[p.group.eq(g)]
            month_rows.append({'month':m,'group':g,'days':p['날짜'].nunique(),'hours':len(p),
                'all_hours_average_power':float(p['평균'].mean()),
                'all_hours_median_power':float(p['평균'].median()),
                'largest_hourly_average':int(p['평균'].max()),
                'largest_15minute_value':int(p[slots].to_numpy().max()),
                'hours_with_average_ge_200':int(p['평균'].ge(200).sum())})
        for g in ['weekday','weekend']:
            for h in [8,9,10,11,12,13,14]:
                p=v.loc[v.month.eq(m)&v.group.eq(g)&v['시간'].eq(h)]
                hours.append({'month':m,'group':g,'hour':h,'n':len(p),'arithmetic_mean':float(p['평균'].mean()),
                    'plotted_median':float(np.median(p['평균'])),'maximum_hour_average':int(p['평균'].max()),
                    'maximum_15minute_value':int(p[slots].to_numpy().max())})
        for positive in [False,True]:
            for h in [8,10,13]:
                p=v.loc[v.month.eq(m)&v.group.eq('weekday')&v['시간'].eq(h)&v.date_positive.eq(positive)]
                production.append({'month':m,'hour':h,'any_positive_production_on_date':positive,'days':len(p),
                    'mean_hour_average':float(p['평균'].mean()) if len(p) else None,
                    'median_hour_average':float(p['평균'].median()) if len(p) else None})
    save('july_august_levels',pd.DataFrame(month_rows))
    save('july_august_hour_examples',pd.DataFrame(hours))
    save('july_august_production_composition',pd.DataFrame(production))
    maxima=[]
    for m in [7,8]:
        p=v.loc[v.month.eq(m)]
        row=p.loc[p['평균'].idxmax()]
        maxima.append({'month':m,'date':int(row['날짜']),'hour':int(row['시간']),
            'four_values':[int(row[x]) for x in slots],'hour_average':int(row['평균'])})
    result={'status':'passed','source_sha256':SHA,'raw_valid_hours':len(v),
        'independently_checked_heatmap_cells':len(check),'max_absolute_median_error':float((check.median_raw-check['median']).abs().max()),
        'month_levels':month_rows,'hour_08_examples':[x for x in hours if x['hour']==8],
        'hour_average_maximum_examples':maxima,
        'weekday_production_composition':[x for x in production if x['hour']==8],
        'july_august_weekend_by_date':v.loc[v.month.isin([7,8])&v.group.eq('weekend')].groupby('날짜')['평균'].mean().to_dict(),
        'judgment':'All 384 displayed monthly/hour/day-type medians and counts match the source. Values near or above 200 do occur; a maximum observation, monthly/hourly arithmetic mean, and monthly/hourly median describe different quantities. August all-hour means include low-power dates; do not silently remove them.',
        'manuscript_edited':False}
    assert hashlib.sha256(MANUSCRIPT.read_bytes()).hexdigest()==before
    (TAB/'independent_heatmap_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    result.pop('july_august_weekend_by_date')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
