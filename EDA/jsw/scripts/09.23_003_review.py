"""Targeted follow-up: read saved validation results, not raw data or September."""
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).parent.parent
PREFIX = '09.23_003'
df = pd.read_csv(BASE / '../../data/processed/jsw/09.23_003_validation_predictions.csv',
                 encoding='utf-8-sig', parse_dates=['timestamp']).set_index('timestamp')
assert df.index.max() < pd.Timestamp('2021-09-01')
out = BASE / 'tables'
def save(rows, name):
    pd.DataFrame(rows).to_csv(out / f'{PREFIX}_{name}.csv',index=False,encoding='utf-8-sig')

dailymax = df.groupby(df.index.date)['peak'].transform('max')
atmax = df.loc[df['peak'].eq(dailymax)].copy()
weights = 1 / atmax.groupby(atmax.index.date)['peak'].transform('size')
rows = []
for lag in (1,24,168):
    err = atmax[f'peak_lag{lag}'] - atmax['peak']
    rows.append({'method':f'lag{lag}', 'days':len(set(atmax.index.date)), 'tied_peak_hours':len(atmax),
                 'daily_equal_weight_mae_at_peak':np.average(err.abs(),weights=weights),
                 'daily_equal_weight_bias_at_peak':np.average(err,weights=weights),
                 'daily_equal_weight_under_rate':np.average(err.lt(0),weights=weights)})
save(rows,'actual_peak_time_errors')
rows=[]
for week, block in df.groupby(pd.Grouper(freq='W-SUN')):
    if block.empty:
        continue
    for lag in (1,24,168):
        rows.append({'week_end':str(week.date()),'rows':len(block),'method':f'lag{lag}',
                     'mae_mean':(block[f'mean_lag{lag}']-block['mean']).abs().mean(),
                     'actual_mean':block['mean'].mean(),
                     'reference_mean':block[f'mean_lag{lag}'].mean(),
                     'zero_production_rows':int(block['생산량'].eq(0).sum())})
save(rows,'weekly_review')
print(pd.read_csv(out/f'{PREFIX}_actual_peak_time_errors.csv').round(3).to_string(index=False))
print(pd.read_csv(out/f'{PREFIX}_weekly_review.csv').query("method == 'lag168'").round(3).to_string(index=False))
