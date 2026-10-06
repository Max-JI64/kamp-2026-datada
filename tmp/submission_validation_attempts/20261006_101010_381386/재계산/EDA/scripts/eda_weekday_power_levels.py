"""Final EDA 2.4: weekday distribution and all-date low-hour patterns.

Run from project root: python EDA/scripts/eda_weekday_power_levels.py
Output tables: EDA/tables/weekday_power_levels; one manuscript histogram:
EDA/figures/weekday_power_levels.png. No discarded preview plots are generated.
"""
from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd

from eda_variable_relations import source_frame, SHA
import eda_figures as style

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'EDA/tables/weekday_power_levels'
FIGURE = ROOT / 'EDA/figures/weekday_power_levels.png'
POWER = '평균 전력'


def save(table, name):
    table.to_csv(OUT / f'{name}.csv', index=False, encoding='utf-8-sig')


def summarize_patterns(data, cutoff):
    """Enumerate exact low-hour sets for every complete weekday, not just counts."""
    daily = []
    for date, cell in data.groupby('날짜', sort=True):
        cell = cell.sort_values('시간')
        assert cell['시간'].tolist() == list(range(24))
        hours = tuple(cell.loc[cell[POWER].le(cutoff), '시간'].astype(int))
        daily.append(dict(date=int(date), low_hours=len(hours),
                          low_hour_pattern=','.join(map(str, hours)) if hours else 'none',
                          daily_any_production=bool(cell['생산량'].gt(0).any())))
    daily = pd.DataFrame(daily)
    counts = daily.groupby('low_hour_pattern').size().rename('days')
    patterns = [tuple(range(7)), tuple(range(8, 24)), tuple(range(24)), ()]
    labels = ['07시부터 49 이상', '08시부터 낮은 구간', '하루 내내 낮은 구간', '하루 내내 49 이상']
    records = []
    expected_keys = set()
    for pattern, label in zip(patterns, labels):
        key = ','.join(map(str, pattern)) if pattern else 'none'
        expected_keys.add(key)
        values = ['20~26' if hour in pattern else '49 이상' for hour in [0, 7, 8]]
        records.append(dict(pattern=label, low_hour_pattern=key,
                            hours_00_06=values[0], hour_07=values[1], hours_08_23=values[2],
                            days=int(counts.get(key, 0)), low_hours_per_day=len(pattern)))
    assert set(counts.index) == expected_keys, 'Manuscript requires revising for additional patterns'
    summary = pd.DataFrame(records)
    assert summary.days.sum() == len(daily) == 171
    assert int((summary.days * summary.low_hours_per_day).sum()) == int(data[POWER].le(cutoff).sum())
    return daily, summary


def plot_distribution(data, cutoff):
    """Edit histogram bin width, colors, axes and legend here."""
    style.setup()
    fig, ax = style.plt.subplots(figsize=(9, 4.6))
    bins = np.arange(19.5, 214.6, 5)
    low = data.loc[data[POWER].le(cutoff), POWER]
    other = data.loc[data[POWER].gt(cutoff), POWER]
    assert bins[0] <= data[POWER].min() and bins[-1] > data[POWER].max()
    counts, _ = np.histogram(data[POWER], bins=bins)
    assert counts.sum() == len(data)
    ax.hist([low, other], bins=bins, stacked=True,
            color=[style.ORANGE, style.BLUE], edgecolor='white', linewidth=.5,
            label=['20~26', '49~208'])
    ax.set(xlabel='시간별 평균 전력', ylabel='기록 수 (시간)', xlim=(15, 215))
    ax.legend(title='평일 전력 구간')
    ax.grid(axis='y', alpha=.16)
    style.finish(fig, 'weekday_power_levels', '평일 전력 분포',
                 '5 간격 히스토그램; 경계는 정수값별 실제 빈 구간에서 확인')
    return counts, bins


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = source_frame().loc[lambda frame: ~frame.weekend].copy()
    assert len(data) == 4104 and data['날짜'].nunique() == 171
    assert data[['날짜', '시간']].duplicated().sum() == 0
    support = np.sort(data[POWER].unique())
    largest = int(np.argmax(np.diff(support)))
    cutoff, next_value = int(support[largest]), int(support[largest + 1])
    assert (int(support.min()), cutoff, next_value, int(support.max())) == (20, 26, 49, 208)
    assert not data[POWER].between(27, 48).any()
    for boundary in range(27, 49):
        assert data[POWER].le(cutoff).equals(data[POWER].lt(boundary))
    frequencies = data[POWER].value_counts().reindex(range(20, 209), fill_value=0)
    save(frequencies.rename_axis('power').reset_index(name='hours'), 'power_frequency')
    daily, patterns = summarize_patterns(data, cutoff)
    save(daily, 'daily_hour_patterns')
    save(patterns, 'pattern_summary')
    data['low'] = data[POWER].le(cutoff)
    production_context = data.groupby('low').agg(
        hours=(POWER, 'size'), positive_production_hours=('생산량', lambda x: int(x.gt(0).sum()))).reset_index()
    save(production_context, 'production_context')
    # Independent cross-check of the exact daily masks using a wide array.
    matrix = data.pivot(index='날짜', columns='시간', values=POWER).sort_index()
    low_mask = matrix.le(cutoff).to_numpy()
    independently_counted = {}
    for row in low_mask:
        key = ','.join(map(str, np.flatnonzero(row))) if row.any() else 'none'
        independently_counted[key] = independently_counted.get(key, 0) + 1
    assert independently_counted == patterns.set_index('low_hour_pattern').days.to_dict()
    np.testing.assert_array_equal(low_mask.sum(axis=1), daily.low_hours.to_numpy())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        histogram_counts, histogram_bins = plot_distribution(data, cutoff)
    important = [str(w.message) for w in caught if 'Glyph' in str(w.message) or 'layout' in str(w.message).lower()]
    assert not important, important
    save(pd.DataFrame(dict(left=histogram_bins[:-1], right=histogram_bins[1:], hours=histogram_counts)), 'histogram_bins')
    result = dict(status='passed', source_sha256=SHA, rows=len(data), days=171,
                  cutoff=cutoff, next_observed_value=next_value, empty_integer_values=[27, 48],
                  low_hours=int(data.low.sum()), low_fraction=float(data.low.mean()),
                  low_production_positive=int(data.loc[data.low, '생산량'].gt(0).sum()),
                  patterns=patterns.to_dict('records'), checked_daily_masks=171,
                  histogram_hours=int(histogram_counts.sum()), warnings=important,
                  image_analysis_performed=False, image=str(FIGURE),
                  image_sha256=hashlib.sha256(FIGURE.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key:value for key,value in result.items() if key not in ['source_sha256', 'script_sha256', 'image_sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
