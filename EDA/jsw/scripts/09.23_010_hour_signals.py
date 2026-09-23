"""Descriptive 08:00/13:00 pre-onset signal audit, with exact daily-profile sensitivity."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent
OUT = BASE / 'tables'
SOURCE = BASE / '../../data/origin/okm_augumented_2021.csv'
P = '09.23_010'
old8 = json.loads((OUT/'09.23_008_facts.json').read_text(encoding='utf-8'))
old9 = json.loads((OUT/'09.23_009_facts.json').read_text(encoding='utf-8'))
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == old8['source_sha256'] == old9['source_sha256']
raw = pd.read_csv(SOURCE, encoding='utf-8-sig', usecols=['날짜','시간','평균','생산량','15분','30분','45분','60분'])
df = raw.loc[raw['날짜'].lt(20210901) & raw['시간'].between(0,23)].copy()
df.index = pd.to_datetime(df['날짜'].astype(str), format='%Y%m%d') + pd.to_timedelta(df['시간'], unit='h')
df = df.sort_index()
assert df.index.is_unique and len(df)==5784
power_cols = ['15분','30분','45분','60분']
df['peak'] = df[power_cols].max(axis=1)
for lag in (1,2,3):
    old = df[['평균','peak','생산량']].copy()
    old.index += pd.Timedelta(hours=lag)
    for col in old:
        df[f'{col}_lag{lag}'] = old[col].reindex(df.index)
required = [f'{col}_lag{lag}' for col in ['평균','peak','생산량'] for lag in (1,2,3)]
eligible = df.loc[df.peak_lag1.lt(old8['threshold'])].dropna(subset=required).copy()
eligible['onset'] = eligible.peak.ge(old8['threshold'])
assert len(eligible)==old8['eligible_rows'] and eligible.onset.sum()==old8['onsets']
e = eligible.loc[eligible.index.hour.isin([8,13])].copy()
e['period'] = np.where(e.index < '2021-07-01','train','validation')
e['month'] = e.index.month
e['hour'] = e.index.hour
e['date'] = e.index.normalize().strftime('%Y-%m-%d')
e['pre_delta_1'] = e['평균_lag1']-e['평균_lag2']
e['pre_delta_2'] = e['평균_lag2']-e['평균_lag3']
e['signal'] = np.where(e.hour.eq(8),e.pre_delta_1.gt(0),e.pre_delta_1.lt(0))
cuts = old9['training_level_quartiles']
e['level'] = pd.cut(e['평균_lag1'],[-np.inf,*cuts,np.inf],labels=['q1','q2','q3','q4'])

# A signature is all 96 raw quarter-hour power values of a complete day.
wide = df.pivot(index='날짜', columns='시간', values=power_cols).dropna()
assert wide.shape==(241,96)
signatures = wide.apply(lambda row:tuple(row),axis=1)
codes,_ = pd.factorize(signatures)
e['profile'] = e['날짜'].map(pd.Series(codes,index=wide.index))
assert e.profile.notna().all()

def save(frame,name):
    pd.DataFrame(frame).to_csv(OUT/f'{P}_{name}.csv',index=False,encoding='utf-8-sig')

def summarize(block, keys):
    result = block.groupby(keys, observed=True).agg(n=('onset','size'),events=('onset','sum'),
        days=('date','nunique'),profiles=('profile','nunique'),
        pre3_mean=('평균_lag3','mean'),pre2_mean=('평균_lag2','mean'),
        pre1_mean=('평균_lag1','mean'),pre_delta_mean=('pre_delta_1','mean'),
        production_pre1_mean=('생산량_lag1','mean')).reset_index()
    result['non_events'] = result.n-result.events
    result['event_rate'] = result.events/result.n
    return result

overview=summarize(e,['period','hour','signal'])
save(overview,'signal_overview')
monthly=summarize(e,['month','hour','signal'])
save(monthly,'signal_monthly')
by_level=summarize(e,['period','hour','level','signal'])
save(by_level,'signal_by_level')
details=summarize(e,['period','hour','signal','onset'])
save(details,'prehistory_cases')

# Exact-profile weighting checks whether repeating identical daily power traces
# makes a row-level signal appear more frequent. Each day has at most one 08/13 row.
unique=e.drop_duplicates(['period','hour','profile'])
assert (e.groupby(['period','hour','profile'])[['onset','signal']].nunique().max(axis=1)<=1).all()
profile=summarize(unique,['period','hour','signal'])
profile=profile.rename(columns={'n':'profiles_equal_weight','events':'event_profiles',
    'non_events':'non_event_profiles','event_rate':'profile_event_rate'})
save(profile,'profile_sensitivity')
profile_counts=e.groupby(['period','hour','signal','profile']).agg(days=('date','nunique'),
    onset=('onset','first')).reset_index()
save(profile_counts,'profile_group_counts')

# Independent arithmetic checks against the prior direction table.
prior=pd.read_csv(OUT/'09.23_008_direction_hour_rates.csv',encoding='utf-8-sig')
for _,r in overview.iterrows():
    direction=('rise' if r.hour==8 else 'fall') if r.signal else None
    b=prior.loc[prior.period.eq(r.period)&prior.hour.eq(r.hour)]
    if direction is not None:
        ref=b.loc[b.pre_direction.eq(direction)]
        assert len(ref)==1 and int(ref.n.iloc[0])==r.n and int(ref.events.iloc[0])==r.events
    else:
        ref=b.loc[~b.pre_direction.eq('rise' if r.hour==8 else 'fall')]
        assert int(ref.n.sum())==r.n and int(ref.events.sum())==r.events
assert overview.n.sum()==len(e) and overview.events.sum()==e.onset.sum()
assert monthly.n.sum()==len(e) and monthly.events.sum()==e.onset.sum()
assert by_level.n.sum()==len(e) and by_level.events.sum()==e.onset.sum()
assert details.n.sum()==len(e) and details.events.sum()==e.onset.sum()
assert profile.profiles_equal_weight.sum()==len(unique)
facts={'time':datetime.now().astimezone().isoformat(timespec='seconds'),
    'source_sha256':old8['source_sha256'],'threshold_train_p95':old8['threshold'],
    'training_level_quartiles':cuts,'eligible_8_13_rows':len(e),'onsets_8_13':int(e.onset.sum()),
    'complete_days':len(wide),'distinct_profiles':len(np.unique(codes)),
    'pandas':pd.__version__,'numpy':np.__version__}
(OUT/f'{P}_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(facts,ensure_ascii=False,indent=2))
print(overview.to_string(index=False))
print(profile[['period','hour','signal','profiles_equal_weight','event_profiles','non_event_profiles','profile_event_rate']].to_string(index=False))
