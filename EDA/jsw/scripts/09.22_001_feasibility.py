"""First EDA: descriptive evidence for task selection; no modeling or cleaning."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import platform
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).parent.parent
SOURCE = BASE / '../../data/origin/okm_augumented_2021.csv'
PREFIX = '09.22_001'
for folder in ('tables', 'figures'):
    (BASE / folder).mkdir(parents=True, exist_ok=True)

def save(frame, name):
    frame.to_csv(BASE / 'tables' / f'{PREFIX}_{name}.csv', index=False, encoding='utf-8-sig')

df = pd.read_csv(SOURCE, encoding='utf-8-sig')
power = ['15분', '30분', '45분', '60분']
dates = pd.to_datetime(df['날짜'].astype(str), format='%Y%m%d', errors='raise')
valid = df['시간'].between(0, 23)
total = df[power].sum(axis=1)
ratio = df['생산량'] / total.replace(0, np.nan)
peak = df[power].max(axis=1)
threshold = float(peak.quantile(.95))
facts = {
    'run_time': datetime.now().astimezone().isoformat(timespec='seconds'),
    'python': platform.python_version(), 'pandas': pd.__version__,
    'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
    'source': '../../data/origin/okm_augumented_2021.csv',
    'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'rows': len(df), 'columns': len(df.columns),
    'date_start': str(dates.min().date()), 'date_end': str(dates.max().date()),
    'date_count': dates.nunique(), 'missing_cells': int(df.isna().sum().sum()),
    'duplicate_rows': int(df.duplicated().sum()),
    'duplicate_date_hour': int(df.duplicated(['날짜', '시간']).sum()),
    'invalid_hour_rows': int((~valid).sum()),
    'invalid_hour_dates': sorted(df.loc[~valid, '날짜'].unique().tolist()),
    'rounded_mean_matches': int((np.floor(total / 4 + .5) == df['평균']).sum()),
    'person_ratio_max_error': float((ratio - df['공장인원']).abs().max()),
    'zero_production_rows': int(df['생산량'].eq(0).sum()),
    'zero_production_positive_power_rows': int((df['생산량'].eq(0) & df['평균'].gt(0)).sum()),
    'zero_mean_rows': int(df['평균'].eq(0).sum()),
    'descriptive_peak_p95': threshold,
    'descriptive_peak_ge_p95_rows': int(peak.ge(threshold).sum()),
    'max_quarter_value': int(peak.max()),
    'mean_power': float(df['평균'].mean()),
    'production_mean_pearson': float(df['생산량'].corr(df['평균'])),
    'production_mean_spearman': float(df['생산량'].corr(df['평균'], method='spearman')),
}
save(pd.DataFrame({'column': df.columns, 'dtype': df.dtypes.astype(str).values,
                   'missing': df.isna().sum().values, 'unique': df.nunique().values}), 'columns')
save(df.describe().T.reset_index(names='column'), 'describe')
# Temporary grouping fields are used only in memory; no row-level derived data is saved.
work = pd.DataFrame({'month': dates.dt.month, 'hour': df['시간'],
                     'production_state': np.where(df['생산량'].gt(0), 'positive', 'zero'),
                     'mean_power': df['평균'], 'quarter_max': peak,
                     'high_peak': peak.ge(threshold)})
def group(frame, key):
    return frame.groupby(key).agg(rows=('mean_power', 'size'),
        mean_power=('mean_power', 'mean'), median_power=('mean_power', 'median'),
        max_quarter=('quarter_max', 'max'), high_peak_rows=('high_peak', 'sum'),
        high_peak_rate=('high_peak', 'mean')).reset_index()
monthly = group(work, 'month')
hourly = group(work.loc[valid], 'hour')
production = group(work, 'production_state')
for name, frame in [('monthly', monthly), ('hourly_valid', hourly), ('production_state', production)]:
    save(frame, name)
daily = pd.DataFrame({'date': dates, 'mean_power': df['평균'], 'quarter_max': peak}).groupby('date').agg(
    rows=('mean_power', 'size'), mean_power=('mean_power', 'mean'), max_quarter=('quarter_max', 'max')).reset_index()
save(daily, 'daily')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
axes[0].plot(daily['date'], daily['mean_power'], label='일별 시간 평균값의 평균')
axes[0].plot(daily['date'], daily['max_quarter'], label='일별 15분 값 최댓값', alpha=.7)
axes[0].set(title='전체 6,168행: 일별 전력 수준', ylabel='전력 기록값 (단위 미확인)', xlabel='날짜')
axes[0].legend()
axes[1].plot(hourly['hour'], hourly['mean_power'], marker='o')
axes[1].set(title=f'정상 시간값 {int(valid.sum()):,}행: 시간대별 평균 (시간 오류 48행 제외)',
            xlabel='기록 시간', ylabel='평균 전력 기록값 (단위 미확인)', xticks=range(0, 24, 2))
fig.savefig(BASE / 'figures' / f'{PREFIX}_power_patterns.png', dpi=150)
plt.close(fig)
(BASE / 'tables' / f'{PREFIX}_facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(facts, ensure_ascii=False, indent=2))
print(production.to_string(index=False))
print(hourly.sort_values('high_peak_rate', ascending=False).head(5).to_string(index=False))
