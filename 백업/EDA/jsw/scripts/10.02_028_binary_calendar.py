"""Reuse E027 daily data to compare two exhaustive binary calendar partitions."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent.parent
OUT = BASE / 'tables'
PREFIX = '10.02_028'
INPUT = ROOT / 'data/processed/jsw/10.02_027_daily_calendar.csv'
CONFIG = {'input': '../../../data/processed/jsw/10.02_027_daily_calendar.csv',
          'input_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest(),
          'period': '2021-01-01..2021-08-31; same 241 complete dates as E027',
          'contrasts': ['all weekdays vs all weekends; public holidays retained on both sides',
                        'all non-public-holidays vs all public-holidays; weekends retained on both sides'],
          'metrics': ['daily_mean', 'daily_peak'], 'alpha': .05,
          'test': 'two-sided Welch t, Holm across four tests',
          'sensitivity': 'E027 month-wise circular group-label shifts, absolute mean difference; Holm across four',
          'B': 9999, 'seed': 20261002,
          'boundary': 'descriptive calendar association, not production-adjusted causal effect; exploratory comparison after E027'}

def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    out = np.empty_like(p); out[order] = adjusted
    return out

def save(frame, suffix):
    frame.to_csv(OUT / f'{PREFIX}_{suffix}.csv', index=False, encoding='utf-8-sig')

def main():
    (OUT / f'{PREFIX}_frozen.json').write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2), encoding='utf-8')
    daily = pd.read_csv(INPUT, encoding='utf-8-sig')
    assert len(daily) == 241
    codes = daily.group.to_numpy()
    rng = np.random.default_rng(CONFIG['seed'])
    rotated = np.empty((CONFIG['B'], len(daily)), np.int8)
    for month in range(1, 9):
        inds = np.flatnonzero(daily.month.to_numpy() == month)
        shifts = rng.integers(0, len(inds), CONFIG['B'])
        rotated[:, inds] = codes[inds][(np.arange(len(inds))[None, :] - shifts[:, None]) % len(inds)]
    results, summary = [], []
    for contrast, mask, rmask, labels in [
        ('weekday_vs_weekend', codes >= 2, rotated >= 2, ['평일 전체', '주말 전체']),
        ('nonholiday_vs_holiday', codes % 2 == 1, rotated % 2 == 1, ['비공휴일 전체', '공휴일 전체'])]:
        counts = [np.count_nonzero(~mask), np.count_nonzero(mask)]
        assert np.all(rmask.sum(1) == counts[1])
        for metric in CONFIG['metrics']:
            y = daily[metric].to_numpy()
            a, b = y[~mask], y[mask]
            result = stats.ttest_ind(a, b, equal_var=False)
            diff = b.mean() - a.mean()
            variance_a, variance_b = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
            se = np.sqrt(variance_a + variance_b)
            df = (variance_a + variance_b) ** 2 / (variance_a ** 2 / (len(a) - 1) + variance_b ** 2 / (len(b) - 1))
            assert np.isclose(abs(diff / se), abs(result.statistic))
            rdifference = (rmask * y).sum(1) / counts[1] - ((~rmask) * y).sum(1) / counts[0]
            assert np.isfinite(rdifference).all()
            results.append({'contrast': contrast, 'metric': metric, 'days_a': counts[0], 'days_b': counts[1],
                            'difference_b_minus_a': diff, 't': result.statistic, 'df': df,
                            'p_raw': result.pvalue,
                            'ci_low_independent_days': diff - stats.t.ppf(.975, df) * se,
                            'ci_high_independent_days': diff + stats.t.ppf(.975, df) * se,
                            'rotation_p': (1 + np.count_nonzero(np.abs(rdifference) >= abs(diff))) / (CONFIG['B'] + 1)})
            for flag, label in enumerate(labels):
                part = daily.loc[mask if flag else ~mask]
                summary.append({'contrast': contrast, 'group': label, 'metric': metric, 'days': len(part),
                                'mean': part[metric].mean(), 'median': part[metric].median(),
                                'profile_weighted_mean': np.average(part[metric], weights=part.profile_weight),
                                'producing_hour_share': part.producing_hours.sum() / (len(part) * 24)})
    tests = pd.DataFrame(results)
    tests['p_holm'] = holm(tests.p_raw.to_numpy())
    tests['rotation_p_holm'] = holm(tests.rotation_p.to_numpy())
    tests['both_below_05'] = tests.p_holm.lt(.05) & tests.rotation_p_holm.lt(.05)
    save(tests, 'tests'); save(pd.DataFrame(summary), 'summary')
    assert counts == [231, 10]
    assert tests.loc[tests.contrast.eq('weekday_vs_weekend'), 'days_a'].eq(171).all()
    assert tests.loc[tests.contrast.eq('weekday_vs_weekend'), 'days_b'].eq(70).all()
    assert hashlib.sha256(INPUT.read_bytes()).hexdigest() == CONFIG['input_sha256']
    facts = {'run_at': datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds'),
             'status': 'passed', 'source': 'E027 verified daily dataset, unchanged',
             'days': 241, 'tests': 4, 'input_sha256': CONFIG['input_sha256'],
             'validation': 'exhaustive binary partitions; counts 171/70 and 231/10; analytic t equals SciPy; all rotation statistics finite; frozen input unchanged'}
    (OUT / f'{PREFIX}_validation.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding='utf-8')
    print(tests[['contrast', 'metric', 'days_a', 'days_b', 'difference_b_minus_a', 'p_holm', 'rotation_p_holm', 'both_below_05']].to_string(index=False))

if __name__ == '__main__':
    main()
