"""EDA 2.5 candidate: same-hour stored mean versus maximum of four power values.

Use all Jan-Aug valid hours, without a peak cutoff or a calendar split.
Tables: EDA/tables/mean_max_power. Preview: tmp/eda_mean_max_power.
The manuscript and its selected figure inventory are not edited.
"""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import pandas as pd
from PIL import Image

from eda_variable_relations import SOURCE, SHA
import eda_figures as style

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'EDA/tables/mean_max_power'
PREVIEW = ROOT / 'tmp/eda_mean_max_power'
SLOTS = ['15분', '30분', '45분', '60분']


def load_and_verify():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SHA
    raw = pd.read_csv(SOURCE, encoding='utf-8-sig',
                      usecols=['날짜', '시간', '평균', '생산량', *SLOTS])
    data = raw.loc[raw['날짜'].lt(20210901) & raw['시간'].between(0, 23)].copy()
    assert len(data) == 5784 and data['날짜'].nunique() == 241
    assert not data.duplicated(['날짜', '시간']).any()
    slots = data[SLOTS].to_numpy()
    assert np.isfinite(slots).all()
    data['exact_mean'] = slots.mean(axis=1)
    data['maximum'] = slots.max(axis=1)
    data['minimum'] = slots.min(axis=1)
    data['gap'] = data['maximum'] - data['평균']
    data['exact_gap'] = data['maximum'] - data['exact_mean']
    data['slot_range'] = data['maximum'] - data['minimum']
    data['maximum_count'] = (slots == data['maximum'].to_numpy()[:, None]).sum(axis=1)
    assert np.array_equal(np.floor(data['exact_mean'] + .5), data['평균'])
    assert data['gap'].ge(0).all() and data['exact_gap'].ge(0).all()
    dates = pd.to_datetime(data['날짜'].astype(str), format='%Y%m%d')
    data['month'] = dates.dt.month
    data['weekend'] = dates.dt.dayofweek.ge(5)
    # Independently parse every eligible CSV row and recompute with Python scalars.
    independent = []
    with SOURCE.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if int(row['날짜']) >= 20210901 or not 0 <= int(row['시간']) <= 23:
                continue
            values = [int(row[key]) for key in SLOTS]
            mean = sum(values) / 4
            stored = int(row['평균'])
            assert int(mean + .5) == stored
            independent.append((int(row['날짜']), int(row['시간']), mean, max(values),
                                min(values), max(values) - stored, max(values) - mean))
    computed = data[['날짜', '시간', 'exact_mean', 'maximum', 'minimum', 'gap', 'exact_gap']]
    assert np.array_equal(np.array(independent), computed.to_numpy())
    return data


def describe(part, group):
    gap = part['gap']
    return dict(group=group, hours=len(part), gap_mean=float(gap.mean()),
                gap_min=int(gap.min()), gap_median=float(gap.median()),
                gap_p90=float(gap.quantile(.9)), gap_p95=float(gap.quantile(.95)),
                gap_p99=float(gap.quantile(.99)), gap_max=int(gap.max()),
                exact_gap_mean=float(part['exact_gap'].mean()),
                exact_gap_median=float(part['exact_gap'].median()),
                all_four_equal=int(part['slot_range'].eq(0).sum()),
                stored_mean_equal_max=int(gap.eq(0).sum()),
                maximum_unique=int(part['maximum_count'].eq(1).sum()),
                pearson=float(part['평균'].corr(part['maximum'])),
                spearman=float(part['평균'].corr(part['maximum'], method='spearman')))


def summarize(data):
    summaries = [describe(data, 'all'), describe(data.loc[~data.weekend], 'weekday'),
                 describe(data.loc[data.weekend], 'weekend')]
    for month, part in data.groupby('month'):
        summaries.append(describe(part, f'month_{month:02d}'))
    for hour, part in data.groupby('시간'):
        summaries.append(describe(part, f'hour_{hour:02d}'))
    summary = pd.DataFrame(summaries)
    same_mean = data.groupby('평균').agg(hours=('maximum', 'size'),
        maximum_min=('maximum', 'min'), maximum_max=('maximum', 'max'),
        maximum_unique=('maximum', 'nunique')).reset_index()
    same_mean['maximum_spread'] = same_mean['maximum_max'] - same_mean['maximum_min']
    same_mean = same_mean.sort_values(['maximum_spread', 'hours'], ascending=False)
    # Concrete examples are all records attaining the largest gap, plus the two
    # maximum endpoints at the stored mean with the greatest maximum spread.
    widest = same_mean.iloc[0]
    cell = data.loc[data['평균'].eq(widest['평균'])]
    examples = pd.concat([data.loc[data.gap.eq(data.gap.max())],
        cell.loc[cell.maximum.eq(cell.maximum.min())].head(1),
        cell.loc[cell.maximum.eq(cell.maximum.max())].head(1)]).drop_duplicates(['날짜', '시간'])
    return summary, same_mean, examples


def compare_daily_highest_hours(data):
    records = []
    for date, part in data.groupby('날짜', sort=True):
        assert len(part) == 24 and sorted(part['시간']) == list(range(24))
        mean_hours = part.loc[part['평균'].eq(part['평균'].max())]
        max_hours = part.loc[part.maximum.eq(part.maximum.max())]
        overlap = set(mean_hours['시간']) & set(max_hours['시간'])
        # Keep every tie and use the best maximum among tied mean-highest hours.
        shortfall = int(part.maximum.max() - mean_hours.maximum.max())
        assert (shortfall == 0) == bool(overlap)
        records.append(dict(date=int(date), weekday=not bool(part.weekend.iloc[0]),
            mean_highest_hours=','.join(map(str, sorted(mean_hours['시간']))),
            maximum_highest_hours=','.join(map(str, sorted(max_hours['시간']))),
            hour_sets_overlap=bool(overlap), daily_highest_mean=int(part['평균'].max()),
            daily_highest_maximum=int(part.maximum.max()),
            best_maximum_at_mean_highest_hours=int(mean_hours.maximum.max()),
            maximum_shortfall=shortfall))
    return pd.DataFrame(records)


def plot_mean_max(data):
    style.setup()
    fig, axes = style.plt.subplots(1, 2, figsize=(12.6, 5.1))
    ax = axes[0]
    ax.scatter(data['평균'], data.maximum, s=12, alpha=.18, color=style.BLUE,
               linewidths=0, rasterized=True)
    upper = float(max(data.maximum.max(), data['평균'].max()))
    ax.plot([0, upper], [0, upper], ls='--', lw=1.4, color=style.GRAY,
            label='평균 = 최대')
    ax.set(xlabel='시간 평균 전력', ylabel='같은 시간의 최대 전력',
           xlim=(-3, upper + 5), ylim=(-3, upper + 5))
    ax.set_aspect('equal', adjustable='box')
    ax.legend(frameon=False, loc='upper left')
    ax.grid(alpha=.12)
    ax = axes[1]
    edges = np.arange(-.5, data.gap.max() + 1.5, 1)
    counts, _, _ = ax.hist(data.gap, bins=edges, color=style.BLUE,
                           edgecolor='white', linewidth=.45)
    assert int(counts.sum()) == len(data)
    ax.set(xlabel='최대 전력 - 시간 평균 전력', ylabel='기록 수 (시간)',
           xlim=(-1, data.gap.max() + 1))
    ax.grid(axis='y', alpha=.12)
    assert fig._suptitle is None and not fig.texts and all(not ax.texts for ax in axes)
    # Preview stays outside the folder reserved for selected manuscript images.
    style.FIGURES = PREVIEW
    style.finish(fig, 'mean_max_power', '', '')
    target = PREVIEW / 'mean_max_power.png'
    with Image.open(target) as original:
        original.resize((original.width // 2, original.height // 2),
                        Image.Resampling.LANCZOS).save(PREVIEW / 'mean_max_power_50.png')
    return target


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    manuscript = ROOT / 'EDA/02_EDA_원고.md'
    before = hashlib.sha256(manuscript.read_bytes()).hexdigest()
    data = load_and_verify()
    summary, same_mean, examples = summarize(data)
    daily_comparison = compare_daily_highest_hours(data)
    for name, frame in [('hourly_values', data), ('summary_by_context', summary),
                        ('same_mean_maximum_range', same_mean), ('observed_examples', examples),
                        ('daily_highest_hour_comparison', daily_comparison)]:
        frame.to_csv(OUT / f'{name}.csv', index=False, encoding='utf-8-sig')
    gap_counts = data.groupby('gap').size().rename('hours').reset_index()
    gap_counts.to_csv(OUT / 'gap_frequency.csv', index=False, encoding='utf-8-sig')
    target = plot_mean_max(data)
    widest = same_mean.iloc[0].to_dict()
    daily = data.groupby('날짜')['gap'].max()
    result = dict(status='passed', source_sha256=SHA, rows=len(data), days=241,
        period='2021-01 through 2021-08, valid hours only; weekdays and weekends',
        mean_definition='CSV mean: floor(sum(four slots)/4 + 0.5)',
        maximum_definition='maximum of four slots in the same row',
        gap_definition='same-row maximum minus stored rounded hourly mean',
        mean_max_summary=summary.iloc[0].to_dict(), widest_same_mean=widest,
        global_highest_maximum=int(data.maximum.max()),
        mean_at_global_highest_maximum=sorted(data.loc[
            data.maximum.eq(data.maximum.max()), '평균'].unique().tolist()),
        daily_highest_hour_comparison=dict(days=len(daily_comparison),
            overlapping_days=int(daily_comparison.hour_sets_overlap.sum()),
            different_days=int((~daily_comparison.hour_sets_overlap).sum()),
            maximum_shortfall=int(daily_comparison.maximum_shortfall.max()),
            median_shortfall_on_different_days=float(daily_comparison.loc[
                ~daily_comparison.hour_sets_overlap, 'maximum_shortfall'].median())),
        largest_gap_days=int(data.loc[data.gap.eq(data.gap.max()), '날짜'].nunique()),
        daily_max_gap_median=float(daily.median()),
        plot=str(target), plot_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        independent_checked_rows=5784, manuscript_edited=False,
        visual_review=False, peak_threshold=None,
        rationale='2.1-2.4 summarize hourly means; inspect information about four-slot maxima that means may omit.')
    previous_path = OUT / 'summary.json'
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding='utf-8'))
        if previous.get('plot_sha256') == result['plot_sha256']:
            result['visual_review'] = previous.get('visual_review', False)
    assert hashlib.sha256(manuscript.read_bytes()).hexdigest() == before
    (OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    print(examples[['날짜', '시간', *SLOTS, '평균', 'maximum', 'gap']].to_json(
        orient='records', force_ascii=False))


if __name__ == '__main__':
    main()
