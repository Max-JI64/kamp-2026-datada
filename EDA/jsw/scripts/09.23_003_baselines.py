"""Rolling one-step baselines, July-August validation; September not scored."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import platform
import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent
SOURCE = BASE / '../../data/origin/okm_augumented_2021.csv'
TABLE = BASE / 'tables'
PROCESSED = BASE / '../../data/processed/jsw'
PREFIX = '09.23_003'
TABLE.mkdir(exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

def save(frame, name):
    frame.to_csv(TABLE / f'{PREFIX}_{name}.csv', index=False, encoding='utf-8-sig')

raw = pd.read_csv(SOURCE, encoding='utf-8-sig')
df = raw.loc[raw['시간'].between(0, 23)].copy()
df.index = pd.to_datetime(df['날짜'].astype(str), format='%Y%m%d') + pd.to_timedelta(df['시간'], unit='h')
df.index.name = 'timestamp'
df = df.sort_index()
assert df.index.is_unique
df['mean'] = df['평균']
df['peak'] = df[['15분','30분','45분','60분']].max(axis=1)
train = df.loc[df.index < '2021-07-01']
threshold = float(train['peak'].quantile(.95))
production_cut = float(train.loc[train['생산량'].gt(0), '생산량'].median())
for lag in (1, 24, 168):
    for target in ('mean', 'peak'):
        series = df[target].copy()
        series.index = series.index + pd.Timedelta(hours=lag)
        df[f'{target}_lag{lag}'] = series.reindex(df.index)
prior_production = df['생산량'].copy()
prior_production.index += pd.Timedelta(hours=1)
df['production_lag1'] = prior_production.reindex(df.index)
df['production_change'] = df['생산량'] - df['production_lag1']
df['high'] = df['peak'].ge(threshold)
df['peak_state'] = np.select([df['peak_lag1'].isna(), df['high'] & df['peak_lag1'].lt(threshold), df['high']],
                             ['unknown_previous', 'onset', 'continuation'], default='not_high')
df['production_level'] = np.select([df['생산량'].eq(0), df['생산량'].le(production_cut)], ['zero','positive_low'], default='positive_high')
df['production_direction'] = np.select([df['production_change'].isna(), df['production_change'].gt(0), df['production_change'].lt(0)],
                                       ['unknown','increase','decrease'], default='unchanged')
df['hour'] = df.index.hour
df['month'] = df.index.month
validation = df.loc[(df.index >= '2021-07-01') & (df.index < '2021-09-01')].copy()
columns = [f'{t}_lag{lag}' for t in ('mean','peak') for lag in (1,24,168)]
common = validation.dropna(subset=columns).copy()
coverage = []
for lag in (1,24,168):
    n = int(validation[f'mean_lag{lag}'].notna().sum())
    coverage.append({'method': f'lag{lag}', 'validation_rows': len(validation), 'available_rows': n,
                     'coverage': n/len(validation), 'common_rows': len(common)})
save(pd.DataFrame(coverage), 'coverage')

def metrics(frame, target, lag):
    error = frame[f'{target}_lag{lag}'] - frame[target]
    return {'rows': len(frame), 'mae': float(error.abs().mean()),
            'rmse': float(np.sqrt((error**2).mean())), 'bias_pred_minus_actual': float(error.mean()),
            'under_rate': float(error.lt(0).mean()), 'mean_shortfall': float((-error).clip(lower=0).mean())}

results = []
for target in ('mean', 'peak'):
    for lag in (1,24,168):
        results.append({'target': target, 'method': f'lag{lag}', **metrics(common,target,lag)})
save(pd.DataFrame(results), 'scores')
conditions = []
for group in ('month','hour','peak_state','production_level','production_direction'):
    for value, block in common.groupby(group):
        for target in ('mean','peak'):
            for lag in (1,24,168):
                conditions.append({'condition': group, 'value': value, 'target': target,
                                   'method': f'lag{lag}', **metrics(block,target,lag)})
save(pd.DataFrame(conditions), 'conditional_scores')

# Evaluate the same target timestamps for daily maximum magnitude comparisons.
daily_scores = []
daily = common.groupby(common.index.date)['peak'].max().rename('actual_max').to_frame()
daily['hours_evaluated'] = common.groupby(common.index.date).size()
for lag in (1,24,168):
    daily[f'pred_max_lag{lag}'] = common.groupby(common.index.date)[f'peak_lag{lag}'].max()
    err = daily[f'pred_max_lag{lag}'] - daily['actual_max']
    daily_scores.append({'method': f'lag{lag}', 'days': len(daily), 'full_days': int(daily['hours_evaluated'].eq(24).sum()),
                         'mae_daily_max': float(err.abs().mean()), 'bias_daily_max': float(err.mean())})
save(pd.DataFrame(daily_scores), 'daily_max_scores')
save(daily.reset_index(names='date'), 'daily_max_values')

# Descriptive peak analysis restricted to pre-September data; retain ties.
history = df.loc[df.index < '2021-09-01'].copy()
history['date'] = history.index.date
daymax = history.groupby('date')['peak'].transform('max')
maxima = history.loc[history['peak'].eq(daymax)].copy()
maxima['tie_weight'] = 1 / maxima.groupby('date')['peak'].transform('size')
max_hours = maxima.groupby('hour').agg(tied_max_hours=('peak','size'),
                                      weighted_days=('tie_weight','sum')).reset_index()
save(max_hours, 'daily_max_hours')
save(history.groupby('date').agg(rows=('peak','size'), daily_max=('peak','max'),
    daily_mean=('mean','mean'), high_hours=('high','sum')).reset_index().sort_values('daily_max',ascending=False).head(15), 'top_peak_days')
extreme = history.loc[history['peak'].eq(history['peak'].max())]
save(extreme.groupby(['month','hour','production_level']).agg(rows=('peak','size'),
      max_value=('peak','max'), mean_production=('생산량','mean')).reset_index(), 'absolute_max_conditions')
# Consecutive high HOURLY records, not duration of continuous 15-minute exceedance.
gap = history.index.to_series().diff().ne(pd.Timedelta(hours=1))
runid = (history['high'].ne(history['high'].shift()) | gap).cumsum()
lengths = history.loc[history['high']].groupby(runid[history['high']]).size()
save(lengths.value_counts().sort_index().rename_axis('consecutive_high_hour_records').reset_index(name='events'), 'high_run_lengths')

# Sensitivity on an identical subset for all methods, keeping the original main evaluation.
sensitive = common.loc[common[['mean','mean_lag1','mean_lag24','mean_lag168']].ne(0).all(axis=1)]
save(pd.DataFrame([{'target':t, 'method':f'lag{lag}', **metrics(sensitive,t,lag)}
                  for t in ('mean','peak') for lag in (1,24,168)]), 'nonzero_sensitivity')
derived_columns = ['mean','peak','생산량','production_lag1','production_change','production_level',
                   'production_direction','peak_state','high','hour','month'] + columns
common[derived_columns].reset_index().to_csv(PROCESSED / f'{PREFIX}_validation_predictions.csv', index=False, encoding='utf-8-sig')
assert len(common) > 0 and common.index.max() < pd.Timestamp('2021-09-01')
assert len(train) + len(validation) + int((df.index >= '2021-09-01').sum()) == len(df)
assert int(lengths.sum()) == int(history['high'].sum())
assert np.isclose(max_hours['weighted_days'].sum(), history['date'].nunique())
facts = {'time':datetime.now().astimezone().isoformat(timespec='seconds'), 'python':platform.python_version(),
         'pandas':pd.__version__,'numpy':np.__version__,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
         'raw_rows':len(raw),'valid_rows':len(df),'train_rows':len(train),'validation_rows':len(validation),
         'reserved_rows':int((df.index >= '2021-09-01').sum()), 'common_validation_rows':len(common),
         'threshold_train_p95':threshold,'positive_production_train_median':production_cut,
         'nonzero_common_rows':len(sensitive),'history_days':history['date'].nunique(),
         'absolute_max':float(history['peak'].max()),'absolute_max_records':len(extreme),
         'high_events':len(lengths),'max_high_run_records':int(lengths.max()),
         'derived_rows':len(common),'derived_columns_including_timestamp':len(derived_columns)+1}
(TABLE / f'{PREFIX}_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(facts,ensure_ascii=False,indent=2))
print(pd.DataFrame(results).round(3).to_string(index=False))
