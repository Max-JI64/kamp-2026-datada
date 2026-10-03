"""Read-only source/table reconciliation; writes only a tmp audit result."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'EDA/scripts'))
from eda_variable_relations import source_frame
from eda_slot_time_patterns import load_data, summarize

source = ROOT / 'data/origin/okm_augumented_2021.csv'
raw = pd.read_csv(source, encoding='utf-8-sig')
slots = ['15분', '30분', '45분', '60분']
scope = raw.loc[raw['날짜'].between(20210101, 20210831)]
v = scope.loc[scope['시간'].between(0, 23)].copy()
dates = pd.to_datetime(v['날짜'].astype(str), format='%Y%m%d')
v['group'] = np.where(dates.dt.dayofweek < 5, 'weekday', 'weekend')
v['month'] = dates.dt.month
def read(name):
    return pd.read_csv(ROOT / 'EDA/tables' / name, encoding='utf-8-sig')

assert len(v) == 5784 and v['날짜'].nunique() == 241
assert len(scope) - len(v) == 48
assert raw.isna().sum().sum() == 21
assert v[slots + ['평균', '생산량']].notna().all().all()
np.testing.assert_array_equal(raw['평균'], np.floor(raw[slots].mean(axis=1) + .5))

for row in read('daily_pattern_restart.csv').itertuples():
    p = v.loc[v.group.eq(row.population) & v['시간'].eq(row.hour)]
    np.testing.assert_allclose([len(p), p['생산량'].mean(), p['평균'].mean()],
                               [row.n, row.production_mean, row.power_mean])
matrix = read('daily_repetition/hourly_matrix.csv').set_index('date')
expected = v.assign(date=dates.dt.strftime('%Y-%m-%d')).pivot(index='date', columns='시간', values='평균')
np.testing.assert_array_equal(matrix.to_numpy(), expected.to_numpy())
assert matrix.index.tolist() == expected.index.tolist()
for row in read('daily_repetition/independent_monthly_cells.csv').itertuples():
    p = v.loc[v.group.eq(row.group) & v.month.eq(row.month) & v['시간'].eq(row.hour), '평균']
    np.testing.assert_allclose([len(p), p.mean()], [row.n_raw, row.mean_raw])

relations = source_frame()
pd.testing.assert_frame_equal(relations[['날짜', '시간']], v[['날짜', '시간']])
for row in read('restart_relations_spearman.csv').itertuples():
    p = relations[list(dict.fromkeys([row.x, row.y]))].dropna()
    assert len(p) == row.n
    coefficient = p[row.x].corr(p[row.y], method='spearman') if row.x != row.y else 1.
    np.testing.assert_allclose(coefficient, row.spearman, atol=1e-12)
for row in read('restart_relations_monthly.csv').itertuples():
    p = relations.loc[relations.month.eq(row.month), [row.variable, '평균 전력']].dropna()
    assert len(p) == row.n
    np.testing.assert_allclose(p[row.variable].corr(p['평균 전력'], method='spearman'), row.spearman, atol=1e-12)
weekday = v.loc[v.group.eq('weekday')]
frequency = read('weekday_power_levels/power_frequency.csv')
np.testing.assert_array_equal(frequency.hours, weekday['평균'].value_counts().reindex(frequency.power, fill_value=0))
for row in read('weekday_power_levels/daily_hour_patterns.csv').itertuples():
    p = weekday.loc[weekday['날짜'].eq(row.date)].sort_values('시간')
    low = p.loc[p['평균'].le(26), '시간'].astype(int).tolist()
    assert (','.join(map(str, low)) if low else 'none') == row.low_hour_pattern
    assert len(low) == row.low_hours
slot_data = load_data()
np.testing.assert_array_equal(slot_data[['날짜','시간']], v.sort_values(['날짜','시간'])[['날짜','시간']])
for table, name in zip(summarize(slot_data), ['hour_slot_summary.csv', 'within_hour_directions.csv', 'within_hour_profiles.csv']):
    pd.testing.assert_frame_equal(table, read('slot_time_patterns/' + name), check_dtype=False, atol=1e-10, rtol=1e-10)
summary = dict(status='passed', raw_shape=list(raw.shape), eda_rows=len(v), eda_days=v['날짜'].nunique(),
               invalid_hours_excluded=len(scope)-len(v), missing_cells=raw.isna().sum()[lambda x:x.gt(0)].to_dict(),
               eda_missing_cells=v.isna().sum()[lambda x:x.gt(0)].to_dict(),
               zero_power_rows_retained=int(v[slots].eq(0).all(axis=1).sum()),
               max_production_retained=int(v['생산량'].max()),
               weekday_rows=len(weekday), weekday_days=weekday['날짜'].nunique(),
               checked_sections=['2.1 daily means','2.2 all hourly values and monthly means','2.3 pairwise overall/monthly correlations','2.4 frequencies and all daily masks','2.5 all three pattern tables'],
               source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
               manuscript_or_existing_results_modified=False)
out=ROOT/'tmp/current_eda_input_audit.json'
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=int), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, default=int))
