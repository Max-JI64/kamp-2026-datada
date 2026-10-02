"""Verify E027 day reconstruction and inference with independent CSV reads."""
import csv
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent.parent
PREFIX = '10.02_027'
OUT = BASE / 'tables'
config = json.loads((OUT / f'{PREFIX}_frozen.json').read_text(encoding='utf-8'))
source = ROOT / 'data/origin/okm_augumented_2021.csv'
assert hashlib.sha256(source.read_bytes()).hexdigest() == config['source_sha256']
by_date = defaultdict(list)
with source.open(encoding='utf-8-sig', newline='') as handle:
    for row in csv.DictReader(handle):
        if int(row['날짜']) < 20210901 and 0 <= int(row['시간']) <= 23:
            by_date[int(row['날짜'])].append(row)
with (ROOT / 'data/processed/jsw' / f'{PREFIX}_daily_calendar.csv').open(encoding='utf-8-sig', newline='') as handle:
    daily = list(csv.DictReader(handle))
holidays = dict(config['holidays'])
counts = defaultdict(int)
weights = defaultdict(float)
ys = {metric: defaultdict(list) for metric in ['daily_mean', 'daily_peak']}
for row in daily:
    date = int(row['date'])
    original = by_date[date]
    assert len(original) == 24 and len({int(x['시간']) for x in original}) == 24
    mean = sum(int(x['평균']) for x in original) / 24
    peak = max(int(x[c]) for x in original for c in ['15분', '30분', '45분', '60분'])
    assert np.isclose(float(row['daily_mean']), mean) and int(row['daily_peak']) == peak
    weekend = datetime.strptime(str(date), '%Y%m%d').weekday() >= 5
    group = 2 * int(weekend) + int(date in holidays)
    assert int(row['group']) == group
    assert row['holiday_name'] == holidays.get(date, '')
    counts[group] += 1
    weights[row['profile_id']] += float(row['profile_weight'])
    ys['daily_mean'][group].append(mean)
    ys['daily_peak'][group].append(peak)
assert dict(counts) == {0: 164, 1: 7, 2: 67, 3: 3}
assert len(daily) == len(by_date) == 241
assert all(np.isclose(v, 1) for v in weights.values())
with (OUT / f'{PREFIX}_primary_tests.csv').open(encoding='utf-8-sig', newline='') as handle:
    tests = list(csv.DictReader(handle))
for row in tests:
    groups = ys[row['metric']]
    if row['question'] == 'Q1':
        result = stats.ttest_ind(groups[0], groups[1] + groups[3], equal_var=False)
    else:
        result = stats.f_oneway(*(groups[g] for g in range(4)), equal_var=False)
    assert np.isclose(result.pvalue, float(row['p_raw']))
    assert np.isclose(abs(result.statistic), float(row['statistic']))
    assert 0 < float(row['rotation_p']) <= 1
    assert float(row['p_holm']) >= float(row['p_raw'])
with (OUT / f'{PREFIX}_pairwise_tests.csv').open(encoding='utf-8-sig', newline='') as handle:
    pairs = list(csv.DictReader(handle))
labels = ['평일 비공휴일', '평일 공휴일', '주말 비공휴일', '주말 공휴일']
for row in pairs:
    a, b = labels.index(row['group_a']), labels.index(row['group_b'])
    groups = ys[row['metric']]
    result = stats.ttest_ind(groups[a], groups[b], equal_var=False)
    assert np.isclose(result.pvalue, float(row['p_raw']))
    assert np.isclose(np.mean(groups[b]) - np.mean(groups[a]), float(row['difference_b_minus_a']))
    assert np.isfinite(float(row['rotation_p_holm']))
assert len(tests) == 4 and len(pairs) == 12
result = {'run_at': datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds'),
          'status': 'passed', 'dates_reconstructed': 241, 'groups': dict(counts),
          'primary_tests_verified': 4, 'pairwise_tests_verified': 12,
          'source_sha256': config['source_sha256'], 'profile_weights_sum_one': True}
(OUT / f'{PREFIX}_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))
