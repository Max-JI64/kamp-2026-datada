"""Descriptive EDA only. Exact timestamp joins prevent bridging missing hours."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import platform
import pandas as pd
import numpy as np

BASE = Path(__file__).parent.parent
SOURCE = BASE / '../../data/origin/okm_augumented_2021.csv'
OUT = BASE / 'tables'
OUT.mkdir(exist_ok=True)
PREFIX = '09.23_002'

def save(frame, name):
    frame.to_csv(OUT / f'{PREFIX}_{name}.csv', index=False, encoding='utf-8-sig')

raw = pd.read_csv(SOURCE, encoding='utf-8-sig')
valid = raw['시간'].between(0, 23)
df = raw.loc[valid].copy()
df['timestamp'] = pd.to_datetime(df['날짜'].astype(str), format='%Y%m%d') + pd.to_timedelta(df['시간'], unit='h')
assert not df['timestamp'].duplicated().any(), 'Valid timestamps must be unique'
df = df.sort_values('timestamp').set_index('timestamp')
df['weekday'] = df.index.dayofweek + 1
df['production_state'] = np.where(df['생산량'].gt(0), 'positive', 'zero')
df['peak'] = df[['15분', '30분', '45분', '60분']].max(axis=1)
# Keep the first EDA's full-file descriptive threshold for direct comparison.
threshold = float(raw[['15분', '30분', '45분', '60분']].max(axis=1).quantile(.95))
df['high'] = df['peak'].ge(threshold)

def summary(keys):
    return df.groupby(keys).agg(rows=('평균', 'size'), mean_power=('평균', 'mean'),
        median_power=('평균', 'median'), p90_power=('평균', lambda x: x.quantile(.9)),
        p95_quarter_max=('peak', lambda x: x.quantile(.95)),
        high_rows=('high', 'sum'), high_rate=('high', 'mean'),
        positive_production_rate=('생산량', lambda x: x.gt(0).mean())).reset_index()

for keys, name in [(['weekday'], 'weekday'), (['시간'], 'hour'),
                   (['시간', 'weekday'], 'hour_weekday'),
                   (['시간', 'production_state'], 'hour_production'),
                   (['weekday', 'production_state'], 'weekday_production'),
                   (['시간', 'weekday', 'production_state'], 'hour_weekday_production')]:
    save(summary(keys), name)

# Align the actual preceding hour, not the preceding available row.
previous = df[['평균', 'peak', '생산량', 'high']].add_prefix('previous_')
previous.index = previous.index + pd.Timedelta(hours=1)
pairs = df.join(previous, how='inner')
assert ((pairs.index - pd.Timedelta(hours=1)).isin(df.index)).all()
pairs['event'] = np.select([pairs['high'] & ~pairs['previous_high'],
                           pairs['high'] & pairs['previous_high']],
                          ['high_onset', 'high_continuation'], default='not_high')
pairs['power_change'] = pairs['평균'] - pairs['previous_평균']
pairs['production_change'] = pairs['생산량'] - pairs['previous_생산량']
events = pairs.groupby('event').agg(rows=('평균', 'size'),
    current_mean_power=('평균', 'mean'), previous_mean_power=('previous_평균', 'mean'),
    previous_median_power=('previous_평균', 'median'),
    mean_power_change=('power_change', 'mean'),
    current_mean_production=('생산량', 'mean'), previous_mean_production=('previous_생산량', 'mean'),
    mean_production_change=('production_change', 'mean'),
    previous_zero_production_rate=('previous_생산량', lambda x: x.eq(0).mean())).reset_index()
save(events, 'peak_events')
transitions = pairs.groupby(['previous_high', 'high']).size().reset_index(name='rows')
transitions['previous_state_rows'] = transitions.groupby('previous_high')['rows'].transform('sum')
transitions['conditional_rate'] = transitions['rows'] / transitions['previous_state_rows']
save(transitions, 'peak_transitions')
onsets = pairs.loc[pairs['event'].eq('high_onset')].groupby('시간').size().reset_index(name='onset_rows')
eligible = pairs.loc[~pairs['previous_high']].groupby('시간').size().reset_index(name='previous_not_high_rows')
onsets = eligible.merge(onsets, on='시간', how='left').fillna({'onset_rows': 0})
onsets['onset_rate'] = onsets['onset_rows'] / onsets['previous_not_high_rows']
save(onsets, 'onset_hour')

zero = df['평균'].eq(0)
run_break = zero.ne(zero.shift()) | df.index.to_series().diff().ne(pd.Timedelta(hours=1))
runs = df.loc[zero].groupby(run_break.cumsum()[zero]).agg(
    rows=('평균', 'size'), production_min=('생산량', 'min'), production_max=('생산량', 'max'),
    quarter_max=('peak', 'max'))
if not runs.empty:
    groups = df.loc[zero].groupby(run_break.cumsum()[zero])
    runs['start'] = groups.apply(lambda x: x.index.min())
    runs['end'] = groups.apply(lambda x: x.index.max())
save(runs.reset_index(drop=True), 'zero_runs')
bad = raw.loc[~valid].groupby('날짜').agg(rows=('시간', 'size'),
    hour_min=('시간', 'min'), hour_max=('시간', 'max'))
save(bad.reset_index(), 'invalid_days')

eligibility = []
for lag in (1, 24, 168):
    available = (df.index - pd.Timedelta(hours=lag)).isin(df.index)
    eligibility.append({'lag_hours': lag, 'target_rows': len(df),
                        'exact_lag_pairs': int(available.sum()), 'missing_lag': int((~available).sum())})
save(pd.DataFrame(eligibility), 'lag_eligibility')
facts = {'run_time': datetime.now().astimezone().isoformat(timespec='seconds'),
         'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
         'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__,
         'raw_rows': len(raw), 'valid_time_rows': len(df), 'excluded_bad_time': int((~valid).sum()),
         'descriptive_threshold': threshold, 'valid_high_rows': int(df['high'].sum()),
         'exact_previous_hour_pairs': len(pairs), 'unpaired_high_rows': int(df['high'].sum()-pairs['high'].sum()),
         'zero_rows': int(zero.sum()), 'no_zero_endpoint_pairs': int((pairs['평균'].ne(0) & pairs['previous_평균'].ne(0)).sum()),
         'small_three_way_groups_lt10': int(summary(['시간','weekday','production_state'])['rows'].lt(10).sum()),
         'three_way_groups': len(summary(['시간','weekday','production_state']))}
assert len(pairs) == eligibility[0]['exact_lag_pairs']
assert int(events['rows'].sum()) == len(pairs)
assert int(summary(['시간','weekday','production_state'])['rows'].sum()) == len(df)
(OUT / f'{PREFIX}_facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(facts, ensure_ascii=False, indent=2))
print(events.to_string(index=False))
print(transitions.to_string(index=False))
