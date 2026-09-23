"""Pre-onset observations; fixed training threshold, exact contiguous history."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import numpy as np
import pandas as pd

B=Path(__file__).parent.parent
O=B/'tables'
P='09.23_008'
source=B/'../../data/origin/okm_augumented_2021.csv'
old=json.loads((O/'09.23_003_facts.json').read_text(encoding='utf-8'))
assert hashlib.sha256(source.read_bytes()).hexdigest()==old['source_sha256']
d=pd.read_csv(source,encoding='utf-8-sig')
d=d.loc[d['날짜'].lt(20210901)&d['시간'].between(0,23)].copy()
d.index=pd.to_datetime(d['날짜'].astype(str),format='%Y%m%d')+pd.to_timedelta(d['시간'],unit='h')
d=d.sort_index(); assert d.index.is_unique
d['peak']=d[['15분','30분','45분','60분']].max(axis=1)
threshold=old['threshold_train_p95']
assert d.loc[d.index<'2021-07-01','peak'].quantile(.95)==threshold
for lag in [1,2,3]:
    previous=d[['평균','peak','생산량']].copy()
    previous.index+=pd.Timedelta(hours=lag)
    for col in previous:
        d[f'{col}_lag{lag}']=previous[col].reindex(d.index)
d['period']=np.where(d.index<'2021-07-01','train','validation')
d['month']=d.index.month
d['hour']=d.index.hour
d['weekend']=d.index.dayofweek>=5
d['onset']=d.peak.ge(threshold)&d.peak_lag1.lt(threshold)
d['pre_delta1']=d['평균_lag1']-d['평균_lag2']
d['pre_delta2']=d['평균_lag2']-d['평균_lag3']
d['production_pre_delta']=d['생산량_lag1']-d['생산량_lag2']
d['production_pre_start']=(d['생산량_lag2'].eq(0)&d['생산량_lag1'].gt(0)).astype(int)
d['production_at_start']=(d['생산량_lag1'].eq(0)&d['생산량'].gt(0)).astype(int)
d['current_jump']=d['평균']-d['평균_lag1']
history=[f'{c}_lag{h}' for h in [1,2,3] for c in ['평균','peak','생산량']]
eligible=d.loc[d.peak_lag1.lt(threshold)].dropna(subset=history).copy()
features=['평균_lag3','평균_lag2','평균_lag1','peak_lag1','pre_delta2','pre_delta1',
          '생산량_lag1','production_pre_delta','production_pre_start','production_at_start','current_jump']
def save(rows,name):
    pd.DataFrame(rows).to_csv(O/f'{P}_{name}.csv',index=False,encoding='utf-8-sig')
coverage=[]
for period,b in d.groupby('period'):
    e=eligible.loc[eligible.period.eq(period)]
    coverage.append({'period':period,'rows':len(b),'known_previous':int(b.peak_lag1.notna().sum()),
        'onsets_known_previous':int(b.onset.sum()),'eligible_rows':len(e),'onsets_with_3h':int(e.onset.sum()),
        'controls_with_3h':int((~e.onset).sum())})
save(coverage,'coverage')
summ=[]
for (period,onset),b in eligible.groupby(['period','onset']):
    for col in features:
        summ.append({'period':period,'onset':onset,'feature':col,'n':len(b),'mean':b[col].mean(),
            'p10':b[col].quantile(.1),'median':b[col].median(),'p90':b[col].quantile(.9)})
save(summ,'unmatched')
# Controls must also have non-high previous hour; match on target month/hour/weekend.
# Each onset's cell gets equal event weight; no current target features in matching.
matched=[]; strata=[]
for (period,month,hour,weekend),b in eligible.groupby(['period','month','hour','weekend']):
    event=b.loc[b.onset]; control=b.loc[~b.onset]
    if event.empty: continue
    strata.append({'period':period,'month':month,'hour':hour,'weekend':weekend,'events':len(event),'controls':len(control)})
    if control.empty: continue
    for col in features:
        matched.append({'period':period,'month':month,'hour':hour,'weekend':weekend,'feature':col,
            'events':len(event),'controls':len(control),'event_mean':event[col].mean(),
            'control_mean':control[col].mean(),'difference':event[col].mean()-control[col].mean()})
save(strata,'matching_coverage'); save(matched,'matched_strata')
m=pd.DataFrame(matched)
aggregate=[]
for keys in [['period','feature'],['period','month','feature']]:
    for key,b in m.groupby(keys):
        aggregate.append({'scope':'+'.join(keys),**dict(zip(keys,key)),'cells':len(b),'events':int(b.events.sum()),
            'event_mean':np.average(b.event_mean,weights=b.events),
            'matched_control_mean':np.average(b.control_mean,weights=b.events),
            'difference':np.average(b.difference,weights=b.events)})
save(aggregate,'matched_summary')
# Observable direction, not a tuned operational threshold.
eligible['pre_direction']=np.select([eligible.pre_delta1.gt(0),eligible.pre_delta1.lt(0)],['rise','fall'],default='flat')
rates=eligible.groupby(['period','hour','pre_direction']).agg(n=('onset','size'),events=('onset','sum')).reset_index()
rates['onset_rate']=rates.events/rates.n
save(rates,'direction_hour_rates')
event=eligible.loc[eligible.onset]
transition=event.groupby(['period','hour']).agg(events=('onset','size'),pre_rise=('pre_delta1',lambda x:x.gt(0).sum()),
    pre_production_start=('production_pre_start','sum'),current_production_start=('production_at_start','sum'),
    pre_delta_mean=('pre_delta1','mean'),current_jump_mean=('current_jump','mean')).reset_index()
save(transition,'onset_transitions')
assert sum(x['eligible_rows'] for x in coverage)==len(eligible)
assert rates.n.sum()==len(eligible) and rates.events.sum()==eligible.onset.sum()
assert transition.events.sum()==eligible.onset.sum()
# Independent contiguous-history check for retained rows.
for h in [1,2,3]:
    assert (eligible.index-pd.Timedelta(hours=h)).isin(d.index).all()
facts={'time':datetime.now().astimezone().isoformat(timespec='seconds'),'threshold':threshold,
       'source_sha256':old['source_sha256'],'coverage':coverage,'eligible_rows':len(eligible),
       'onsets':int(eligible.onset.sum()),'matched_events':int(pd.DataFrame(strata).query('controls > 0').events.sum()),
       'pandas':pd.__version__,'numpy':np.__version__}
(O/f'{P}_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(facts,ensure_ascii=False,indent=2))
print(pd.DataFrame(aggregate).query("scope == 'period+feature'").round(3).to_string(index=False))
