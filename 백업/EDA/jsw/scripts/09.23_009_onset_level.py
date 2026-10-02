"""Follow 008 level contrast with fixed training bins, no threshold search."""
from pathlib import Path
from datetime import datetime
import json
import hashlib
import numpy as np
import pandas as pd
B=Path(__file__).parent.parent; O=B/'tables'; P='09.23_009'
source=B/'../../data/origin/okm_augumented_2021.csv'
old=json.loads((O/'09.23_008_facts.json').read_text(encoding='utf-8'))
assert hashlib.sha256(source.read_bytes()).hexdigest()==old['source_sha256']
d=pd.read_csv(source,encoding='utf-8-sig',usecols=['날짜','시간','평균','15분','30분','45분','60분'])
d=d.loc[d['날짜'].lt(20210901)&d['시간'].between(0,23)].copy()
d.index=pd.to_datetime(d['날짜'].astype(str),format='%Y%m%d')+pd.to_timedelta(d['시간'],unit='h')
d=d.sort_index(); d['peak']=d[['15분','30분','45분','60분']].max(axis=1)
for h in [1,2,3]:
    s=d[['평균','peak']].copy(); s.index+=pd.Timedelta(hours=h)
    d[f'lag{h}']=s['평균'].reindex(d.index)
    if h==1: d['previous_peak']=s.peak.reindex(d.index)
e=d.loc[d.previous_peak.lt(old['threshold'])].dropna(subset=['lag1','lag2','lag3']).copy()
assert len(e)==old['eligible_rows']
e['onset']=e.peak.ge(old['threshold'])
e['period']=np.where(e.index<'2021-07-01','train','validation')
e['month']=e.index.month; e['hour']=e.index.hour; e['weekend']=e.index.dayofweek>=5
cuts=e.loc[e.period.eq('train'),'lag1'].quantile([.25,.5,.75]).tolist()
e['level']=pd.cut(e.lag1,[-np.inf,*cuts,np.inf],labels=['q1','q2','q3','q4'])
e['rising']=e.lag1.gt(e.lag2)
def save(b,n): b.to_csv(O/f'{P}_{n}.csv',index=False,encoding='utf-8-sig')
for keys,name in [(['period','level'],'level_rates'),(['month','level'],'monthly_rates'),
                  (['period','level','rising'],'rise_within_level'),(['period','hour','level'],'hour_level')]:
    table=e.groupby(keys,observed=True).agg(n=('onset','size'),events=('onset','sum')).reset_index()
    table['onset_rate']=table.events/table.n
    assert table.n.sum()==len(e) and table.events.sum()==old['onsets']
    save(table,name)
# Within identical month/hour/weekend: q4 vs all lower bins, only common-support cells.
rows=[]
for (period,month,hour,weekend),b in e.groupby(['period','month','hour','weekend']):
    high=b.loc[b.level.eq('q4')]; low=b.loc[~b.level.eq('q4')]
    if len(high)<3 or len(low)<3: continue
    rows.append({'period':period,'month':month,'hour':hour,'weekend':weekend,'high_n':len(high),'low_n':len(low),
        'high_events':int(high.onset.sum()),'low_events':int(low.onset.sum()),
        'high_rate':high.onset.mean(),'low_rate':low.onset.mean(),'difference':high.onset.mean()-low.onset.mean()})
c=pd.DataFrame(rows); save(c,'level_matched_cells')
summary=[]
for period,b in c.groupby('period'):
    summary.append({'period':period,'cells':len(b),'high_n':int(b.high_n.sum()),'low_n':int(b.low_n.sum()),
        'high_events':int(b.high_events.sum()),'low_events':int(b.low_events.sum()),
        'equal_cell_high_rate':b.high_rate.mean(),'equal_cell_low_rate':b.low_rate.mean(),
        'equal_cell_difference':b.difference.mean()})
save(pd.DataFrame(summary),'level_matched_summary')
facts={'time':datetime.now().astimezone().isoformat(timespec='seconds'),'training_level_quartiles':cuts,
    'source_sha256':old['source_sha256'],'rows':len(e),'onsets':int(e.onset.sum()),
    'checks':'008 sample counts; aggregation denominators and event sums; 9月 excluded; no tuned cutoffs'}
(O/f'{P}_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(facts,ensure_ascii=False,indent=2))
print(pd.DataFrame(summary).round(4).to_string(index=False))
for n in ['level_rates','rise_within_level','monthly_rates']:
    print(pd.read_csv(O/f'{P}_{n}.csv').round(4).to_string(index=False))
