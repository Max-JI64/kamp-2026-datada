"""Whole-dataset EDA: recurring four-slot profiles at each hour of day.

Jan-Aug valid dates, all records; weekday/weekend summaries, no peak threshold.
Selected heatmap: EDA/figures/slot_time_patterns.png. The 96-slot line is a preview.
The script never edits the manuscript.
"""
from pathlib import Path
import csv
import argparse
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from PIL import Image

from eda_variable_relations import SOURCE, SHA
import eda_figures as style

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'EDA/tables/slot_time_patterns'
PREVIEW = ROOT / 'tmp/eda_slot_time_patterns'
FIGURE = ROOT / 'EDA/figures/slot_time_patterns.png'
SLOTS = ['15분', '30분', '45분', '60분']


def load_data():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SHA
    raw = pd.read_csv(SOURCE, encoding='utf-8-sig',
                      usecols=['날짜', '시간', '평균', *SLOTS])
    data = raw.loc[raw['날짜'].lt(20210901) & raw['시간'].between(0, 23)].copy()
    data = data.sort_values(['날짜', '시간']).reset_index(drop=True)
    assert len(data) == 5784 and data['날짜'].nunique() == 241
    assert not data.duplicated(['날짜', '시간']).any()
    assert data.groupby('날짜')['시간'].apply(list).map(lambda x: x == list(range(24))).all()
    dates = pd.to_datetime(data['날짜'].astype(str), format='%Y%m%d')
    data['group'] = np.where(dates.dt.dayofweek.ge(5), 'weekend', 'weekday')
    data['exact_mean'] = data[SLOTS].mean(axis=1)
    assert np.array_equal(np.floor(data.exact_mean + .5), data['평균'])
    assert data.groupby('group')['날짜'].nunique().to_dict() == {'weekday':171, 'weekend':70}
    return data


def summarize(data):
    cells, directions, profiles = [], [], []
    for (group, hour), part in data.groupby(['group', '시간'], sort=True):
        values = part[SLOTS].to_numpy()
        centered = values - part.exact_mean.to_numpy()[:, None]
        unique_values = part[SLOTS].drop_duplicates().to_numpy()
        unique_centered = unique_values - unique_values.mean(axis=1)[:, None]
        for i, slot in enumerate(SLOTS):
            delta = centered[:, i]
            cells.append(dict(group=group, hour=int(hour), slot=slot, slot_index=i,
                days=len(part), power_mean=float(values[:, i].mean()),
                hourly_exact_mean=float(part.exact_mean.mean()),
                mean_deviation=float(delta.mean()), median_deviation=float(np.median(delta)),
                above_hours=int((delta > 0).sum()), equal_hours=int((delta == 0).sum()),
                below_hours=int((delta < 0).sum()), above_fraction=float((delta > 0).mean()),
                below_fraction=float((delta < 0).mean()),
                unique_four_value_rows=len(unique_values),
                unique_mean_deviation=float(unique_centered[:, i].mean())))
        comparisons = [('60분-15분', 3, 0), ('30분-15분', 1, 0),
                       ('45분-30분', 2, 1), ('60분-45분', 3, 2)]
        for label, last, first in comparisons:
            delta = values[:, last] - values[:, first]
            unique_delta = unique_values[:, last] - unique_values[:, first]
            directions.append(dict(group=group, hour=int(hour), comparison=label,
                days=len(part), mean_change=float(delta.mean()), median_change=float(np.median(delta)),
                positive_days=int((delta > 0).sum()), zero_days=int((delta == 0).sum()),
                negative_days=int((delta < 0).sum()), positive_fraction=float((delta > 0).mean()),
                negative_fraction=float((delta < 0).mean()),
                unique_rows=len(unique_values), unique_positive_fraction=float((unique_delta > 0).mean()),
                unique_negative_fraction=float((unique_delta < 0).mean())))
        def profile_counts(array):
            return dict(first_strict_lowest=int((array[:, 0] < array[:, 1:].min(axis=1)).sum()),
                middle_below_both_ends=int((array[:, 1:3].max(axis=1) < array[:, [0, 3]].min(axis=1)).sum()),
                late_above_both_early=int((array[:, 2:4].min(axis=1) > array[:, :2].max(axis=1)).sum()),
                strictly_increasing=int(np.all(np.diff(array, axis=1) > 0, axis=1).sum()))
        counts = profile_counts(values)
        unique_counts = profile_counts(unique_values)
        profiles.append(dict(group=group, hour=int(hour), days=len(part), **counts,
            unique_rows=len(unique_values), **{f'unique_{k}':v for k,v in unique_counts.items()}))
    cells, directions = pd.DataFrame(cells), pd.DataFrame(directions)
    assert len(cells) == 192 and len(directions) == 192
    assert np.allclose(cells.groupby(['group', 'hour']).mean_deviation.sum(), 0)
    assert (cells.above_hours + cells.equal_hours + cells.below_hours).eq(cells.days).all()
    assert (directions.positive_days + directions.zero_days + directions.negative_days).eq(directions.days).all()
    return cells, directions, pd.DataFrame(profiles)


def verify_independently(cells, directions, profiles):
    buckets = {}
    with SOURCE.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            date, hour = int(row['날짜']), int(row['시간'])
            if date >= 20210901 or not 0 <= hour <= 23:
                continue
            group = 'weekend' if pd.Timestamp(str(date)).dayofweek >= 5 else 'weekday'
            buckets.setdefault((group, hour), []).append([int(row[x]) for x in SLOTS])
    for row in cells.itertuples(index=False):
        values = buckets[row.group, row.hour]
        mean = sum(x[row.slot_index] for x in values) / len(values)
        deviations = [x[row.slot_index] - sum(x) / 4 for x in values]
        assert len(values) == row.days
        assert abs(mean - row.power_mean) < 1e-10
        assert abs(sum(deviations) / len(values) - row.mean_deviation) < 1e-10
        assert sum(x > 0 for x in deviations) == row.above_hours
        assert sum(x < 0 for x in deviations) == row.below_hours
    for row in directions.itertuples(index=False):
        last, first = [SLOTS.index(x) for x in row.comparison.split('-')]
        delta = [x[last] - x[first] for x in buckets[row.group, row.hour]]
        assert sum(x > 0 for x in delta) == row.positive_days
        assert sum(x == 0 for x in delta) == row.zero_days
        assert sum(x < 0 for x in delta) == row.negative_days
        assert abs(sum(delta) / len(delta) - row.mean_change) < 1e-10
    for row in profiles.itertuples(index=False):
        values = buckets[row.group, row.hour]
        assert sum(x[0] < min(x[1:]) for x in values) == row.first_strict_lowest
        assert sum(max(x[1:3]) < min(x[0], x[3]) for x in values) == row.middle_below_both_ends
        assert sum(min(x[2:]) > max(x[:2]) for x in values) == row.late_above_both_early
        assert sum(x[0] < x[1] < x[2] < x[3] for x in values) == row.strictly_increasing


def save_figure(fig, name):
    assert fig._suptitle is None and not fig.texts
    fig.tight_layout(pad=1.5)
    target = PREVIEW / f'{name}.png'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig.savefig(target, dpi=180)
    assert not [str(x.message) for x in caught if 'Glyph' in str(x.message) or 'layout' in str(x.message)]
    style.plt.close(fig)
    return target


def plot_daily_96(cells):
    style.setup()
    fig, ax = style.plt.subplots(figsize=(12, 7.5))
    for group, label, color in zip(['weekday', 'weekend'],
        ['평일 (171일)', '주말 (70일)'], [style.BLUE, style.ORANGE]):
        part = cells.loc[cells.group.eq(group)].sort_values(['hour', 'slot_index'])
        x = part.hour.to_numpy() + part.slot_index.to_numpy() / 4
        ax.plot(x, part.power_mean, color=color, lw=1.7, marker='o', markersize=2.5,
                label=label)
    ax.set(ylabel='평균 전력', xlim=(-.2, 24), ylim=(0, 210))
    ax.grid(axis='y', alpha=.15)
    ax.legend(loc='upper right', frameon=False, fontsize=11)
    ax.set(xticks=np.arange(24) + .375, xticklabels=[str(h) for h in range(24)],
                 xlabel='시간대 (시)')
    return save_figure(fig, 'daily_96_slot_means')


def plot_centered_heatmap(cells):
    fig, axes = style.plt.subplots(1, 2, figsize=(11.6, 9), sharey=True, layout='constrained')
    bound = float(np.ceil(cells.mean_deviation.abs().max()))
    for ax, group, label in zip(axes, ['weekday', 'weekend'], ['평일 (171일)', '주말 (70일)']):
        matrix = cells.loc[cells.group.eq(group)].pivot(index='hour', columns='slot',
            values='mean_deviation').reindex(columns=SLOTS)
        art = ax.imshow(matrix.to_numpy(), cmap='RdBu_r', vmin=-bound, vmax=bound, aspect='auto')
        ax.set(title=label, xticks=range(4), xticklabels=SLOTS,
               yticks=range(24), yticklabels=[f'{h:02d}시' for h in range(24)])
        ax.tick_params(axis='both', length=0)
        for y in range(24):
            for x in range(4):
                value = matrix.iloc[y, x]
                number = '0.0' if abs(value) < .05 else f'{value:+.1f}'
                ax.text(x, y, number, ha='center', va='center', fontsize=10,
                         color='white' if abs(value) > .60 * bound else '#253746')
    axes[0].set_ylabel('시간대')
    fig.colorbar(art, ax=axes, shrink=.92, label='15분 구간 전력 - 같은 시간의 네 구간 평균')
    assert fig._suptitle is None and not fig.texts
    target = FIGURE
    target.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig.savefig(target, dpi=180)
    assert not [str(x.message) for x in caught if 'Glyph' in str(x.message) or 'layout' in str(x.message)]
    style.plt.close(fig)
    return target


def main(figures_only=False):
    OUT.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    manuscript = ROOT / 'EDA/02_EDA_원고.md'
    before = hashlib.sha256(manuscript.read_bytes()).hexdigest()
    if figures_only:
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SHA
        table_paths = sorted(OUT.glob('*.csv'))
        table_hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in table_paths}
        cells = pd.read_csv(OUT / 'hour_slot_summary.csv', encoding='utf-8-sig')
        assert len(cells) == 192
        assert cells.loc[cells.group.eq('weekday'), 'days'].eq(171).all()
        assert cells.loc[cells.group.eq('weekend'), 'days'].eq(70).all()
        assert np.allclose(cells.groupby(['group','hour']).mean_deviation.sum(), 0)
        style.setup()
        target = plot_centered_heatmap(cells)
        with Image.open(target) as image:
            image.resize((image.width // 2, image.height // 2), Image.Resampling.LANCZOS).save(
                PREVIEW / 'heatmap_50.png')
        result = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
        result.pop('heatmap_figure_group', None)
        result['heatmap_figure_groups'] = ['weekday','weekend']
        for item in result['figures']:
            if Path(item['path']).name in ['hour_slot_mean_deviations.png', target.name]:
                item['path'] = str(target)
                item['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
        result['visual_review'] = False
        result['plot_revision'] = dict(date='2026-10-03', figures_only=True,
            result_tables_unchanged=True, common_color_limits=[-float(np.ceil(cells.mean_deviation.abs().max())),
                float(np.ceil(cells.mean_deviation.abs().max()))])
        assert table_hashes == {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in table_paths}
        assert hashlib.sha256(manuscript.read_bytes()).hexdigest() == before
        (OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(dict(status='passed', figures_only=True, reused_cells=192,
            result_tables_unchanged=True, manuscript_edited=False, figure=str(target)), ensure_ascii=False))
        return
    data = load_data()
    cells, directions, profiles = summarize(data)
    verify_independently(cells, directions, profiles)
    cells.to_csv(OUT / 'hour_slot_summary.csv', index=False, encoding='utf-8-sig')
    directions.to_csv(OUT / 'within_hour_directions.csv', index=False, encoding='utf-8-sig')
    profiles.to_csv(OUT / 'within_hour_profiles.csv', index=False, encoding='utf-8-sig')
    # Rank every hour by the spread of its four mean values; no selected dates.
    ranks = cells.groupby(['group', 'hour']).agg(days=('days', 'first'),
        smallest_slot_mean=('power_mean', 'min'), largest_slot_mean=('power_mean', 'max'))
    ranks['spread_of_slot_means'] = ranks.largest_slot_mean - ranks.smallest_slot_mean
    ranks.reset_index().sort_values(['group', 'spread_of_slot_means'], ascending=[True, False]).to_csv(
        OUT / 'hourly_slot_mean_spread.csv', index=False, encoding='utf-8-sig')
    figures = [plot_daily_96(cells), plot_centered_heatmap(cells)]
    thumbs = []
    for target in figures:
        with Image.open(target) as image:
            thumbs.append(image.convert('RGB').resize((image.width // 2, image.height // 2),
                                                     Image.Resampling.LANCZOS))
    contact = Image.new('RGB', (sum(x.width for x in thumbs) + 16, max(x.height for x in thumbs)), 'white')
    x = 0
    for thumb in thumbs:
        contact.paste(thumb, (x, 0))
        x += thumb.width + 16
    contact.save(PREVIEW / 'contact_50.png')
    result = dict(status='passed', source_sha256=SHA, rows=5784, days=241,
        weekday_days=171, weekend_days=70, checked_mean_deviation_cells=192,
        checked_direction_cells=192, checked_profile_conditions=192,
        line_figure_groups=['weekday','weekend'], heatmap_figure_groups=['weekday','weekend'],
        definition='Each slot minus the exact arithmetic mean of its own four-slot row; average across all dates at the same hour and group.',
        figure_x_definition='Four slots are placed in column order within each hour, not verified exact sensor timestamps.',
        figures=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in figures],
        manuscript_edited=False, visual_review=False, p_values_used=False,
        note='Date frequencies are descriptive; repeated four-value profiles are not independent experiments.')
    assert hashlib.sha256(manuscript.read_bytes()).hexdigest() == before
    (OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    top = ranks.reset_index().sort_values(['group', 'spread_of_slot_means'], ascending=[True, False]).groupby('group').head(4)
    print(top.to_json(orient='records', force_ascii=False))
    hours = sorted(set(top.hour.tolist() + [4]))
    print(directions.loc[directions.comparison.eq('60분-15분') & directions.hour.isin([4,7])].to_json(
        orient='records', force_ascii=False))
    print(profiles.loc[profiles.hour.isin([7,10,12,17])].to_json(orient='records', force_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures-only', action='store_true', help='Reuse verified tables and redraw only the heatmap')
    main(figures_only=parser.parse_args().figures_only)
