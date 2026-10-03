"""EDA 2.5 preview: sequential four-slot patterns and five-hour examples.

Reuse the verified hourly table; preserve source hashes and manuscript contents.
Examples are selected by range/maximum, with earliest date-hour as tie breaker.
"""
from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from PIL import Image

from eda_mean_max_power import SLOTS, SOURCE, SHA
import eda_figures as style

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'EDA/tables/mean_max_power/hourly_values.csv'
OUT = ROOT / 'EDA/tables/quarter_hour_patterns'
PREVIEW = ROOT / 'tmp/eda_quarter_hour_patterns'


def load_verified():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SHA
    data = pd.read_csv(INPUT, encoding='utf-8-sig').sort_values(['날짜', '시간']).reset_index(drop=True)
    raw = pd.read_csv(SOURCE, encoding='utf-8-sig', usecols=['날짜', '시간', '평균', *SLOTS])
    raw = raw.loc[raw['날짜'].lt(20210901) & raw['시간'].between(0, 23)].sort_values(['날짜', '시간'])
    assert len(data) == 5784 and data['날짜'].nunique() == 241
    assert np.array_equal(data[['날짜', '시간', '평균', *SLOTS]], raw[['날짜', '시간', '평균', *SLOTS]])
    assert np.array_equal(data.slot_range, data[SLOTS].max(axis=1) - data[SLOTS].min(axis=1))
    return data


def classify_shapes(data):
    slots = data[SLOTS].to_numpy()
    changes = np.diff(slots, axis=1)
    shape = np.select([np.all(changes == 0, axis=1),
        np.all(changes >= 0, axis=1), np.all(changes <= 0, axis=1)],
        ['all_equal', 'nondecreasing', 'nonincreasing'], default='mixed')
    data['shape'] = shape
    data['interior_above_both_ends'] = slots[:, 1:3].max(axis=1) > slots[:, [0, 3]].max(axis=1)
    data['largest_within_step'] = np.abs(changes).max(axis=1)
    assert sum(pd.Series(shape).value_counts()) == len(data)
    return data


def select_examples(data):
    median = float(data.slot_range.median())
    eligible = data.loc[data['시간'].between(2, 21)].copy()
    eligible['distance_to_median'] = (eligible.slot_range - median).abs()
    typical = eligible.sort_values(['distance_to_median', '날짜', '시간']).iloc[0]
    widest = data.sort_values(['slot_range', '날짜', '시간'], ascending=[False, True, True]).iloc[0]
    highest = data.sort_values(['maximum', '날짜', '시간'], ascending=[False, True, True]).iloc[0]
    selections = []
    windows = []
    for name, row in zip(['median_range', 'largest_range', 'highest_power'], [typical, widest, highest]):
        date, hour = int(row['날짜']), int(row['시간'])
        first = min(max(hour - 2, 0), 19)
        cell = data.loc[data['날짜'].eq(date) & data['시간'].between(first, first + 4)].copy()
        assert len(cell) == 5 and cell['시간'].tolist() == list(range(first, first + 5))
        selected = dict(example=name, date=date, center_hour=hour, first_hour=first,
            last_hour=first + 4, range=int(row.slot_range), maximum=int(row.maximum),
            mean=int(row['평균']), shape=row['shape'], slots=[int(row[x]) for x in SLOTS],
            tied_target_hours=int((data.slot_range.eq(row.slot_range) if name != 'highest_power'
                                   else data.maximum.eq(row.maximum)).sum()))
        selections.append(selected)
        cell['example'] = name
        windows.append(cell)
    return selections, pd.concat(windows, ignore_index=True)


def continuity_statistics(data):
    slots = data[SLOTS].to_numpy()
    within = np.abs(np.diff(slots, axis=1)).ravel()
    dates = pd.to_datetime(data['날짜'].astype(str), format='%Y%m%d')
    stamps = dates + pd.to_timedelta(data['시간'], unit='h')
    adjacent = stamps.diff().eq(pd.Timedelta(hours=1)).to_numpy()[1:]
    boundary = np.abs(slots[1:, 0] - slots[:-1, 3])[adjacent]
    return dict(within_pairs=len(within), within_median=float(np.median(within)),
        within_p90=float(np.quantile(within, .9)), within_max=int(within.max()),
        boundary_pairs=len(boundary), skipped_hour_boundaries=int((~adjacent).sum()),
        boundary_median=float(np.median(boundary)), boundary_p90=float(np.quantile(boundary, .9)),
        boundary_max=int(boundary.max()))


def save_figure(fig, name):
    assert fig._suptitle is None and not fig.texts and all(not ax.texts for ax in fig.axes)
    fig.tight_layout(pad=1.4)
    target = PREVIEW / f'{name}.png'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig.savefig(target, dpi=180)
    assert not [str(x.message) for x in caught if 'Glyph' in str(x.message) or 'layout' in str(x.message)]
    style.plt.close(fig)
    return target


def plot_examples(data, selections, windows):
    style.setup()
    fig, axes = style.plt.subplots(3, 1, figsize=(11.5, 8), sharey=True)
    labels = ['중앙 변동 폭 사례', '최대 변동 폭 사례', '최고 전력 사례']
    for ax, item, label in zip(axes, selections, labels):
        cell = windows.loc[windows.example.eq(item['example'])]
        first, last = item['first_hour'], item['last_hour']
        slots = cell[SLOTS].to_numpy().ravel()
        # Position slots in their recorded order; no claim about exact sensor timestamps.
        x = np.repeat(cell['시간'].to_numpy(), 4) + np.tile([0, .25, .5, .75], 5)
        ax.plot(x, slots, color=style.BLUE, marker='o', markersize=3.5, lw=1.7,
                label='15분 구간 전력')
        ax.stairs(cell['평균'], np.arange(first, last + 2), baseline=None, color=style.ORANGE,
                  lw=1.8, linestyle='--', label='시간 평균 전력')
        for boundary in range(first + 1, last + 1):
            ax.axvline(boundary, color='#D9DEE3', lw=.7, zorder=0)
        date = pd.to_datetime(str(item['date'])).strftime('%Y-%m-%d')
        ax.set_title(f'{label}  |  {date}  {item["center_hour"]:02d}시', fontsize=11, loc='left')
        ax.set(ylim=(0, data.maximum.max() + 10), xlim=(first - .1, last + 1),
            ylabel='전력', xticks=np.arange(first, last + 1) + .375,
            xticklabels=[f'{h:02d}시' for h in range(first, last + 1)])
        ax.grid(axis='y', alpha=.13)
    axes[0].legend(loc='upper left', frameon=False, ncol=2, fontsize=10)
    axes[-1].set_xlabel('시간 (시간 안의 네 점은 15분·30분·45분·60분 순서)')
    return save_figure(fig, 'sequential_power_examples')


def plot_distribution(data):
    fig, ax = style.plt.subplots(figsize=(9.8, 4.4))
    edges = np.arange(-.5, data.slot_range.max() + 1.5, 1)
    counts, _, _ = ax.hist(data.slot_range, bins=edges, color=style.BLUE,
                           edgecolor='white', linewidth=.4)
    assert int(counts.sum()) == 5784
    ax.set(xlabel='한 시간의 최대 전력 - 최소 전력', ylabel='기록 수 (시간)',
           xlim=(-1, data.slot_range.max() + 1))
    ax.grid(axis='y', alpha=.13)
    return save_figure(fig, 'within_hour_range')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    manuscript = ROOT / 'EDA/02_EDA_원고.md'
    before = hashlib.sha256(manuscript.read_bytes()).hexdigest()
    data = classify_shapes(load_verified())
    selections, windows = select_examples(data)
    images = [plot_examples(data, selections, windows), plot_distribution(data)]
    thumbnails = []
    for path in images:
        with Image.open(path) as image:
            thumbnails.append(image.convert('RGB').resize((image.width // 2, image.height // 2),
                                                           Image.Resampling.LANCZOS))
    contact = Image.new('RGB', (max(x.width for x in thumbnails), sum(x.height for x in thumbnails) + 16), 'white')
    y = 0
    for image in thumbnails:
        contact.paste(image, (0, y))
        y += image.height + 16
    contact.save(PREVIEW / 'contact_50.png')
    for name, frame in [('hourly_shapes', data), ('example_windows', windows),
        ('example_selection', pd.DataFrame(selections)),
        ('range_frequency', data.groupby('slot_range').size().rename('hours').reset_index()),
        ('shape_frequency', data.groupby('shape').size().rename('hours').reset_index())]:
        frame.to_csv(OUT / f'{name}.csv', index=False, encoding='utf-8-sig')
    delta = data.slot_range
    result = dict(status='passed', rows=5784, days=241, source_sha256=SHA,
        input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        range_summary=dict(mean=float(delta.mean()), median=float(delta.median()),
            p90=float(delta.quantile(.9)), p95=float(delta.quantile(.95)),
            maximum=int(delta.max()), all_equal=int(delta.eq(0).sum())),
        shape_counts=data['shape'].value_counts().to_dict(),
        interior_above_both_ends=int(data.interior_above_both_ends.sum()),
        continuity=continuity_statistics(data), selections=selections,
        selection_rule='Range nearest global median with 2<=hour<=21, largest range, largest maximum; ties earliest date-hour. Five-hour windows only, full statistics include every hour.',
        figures=[dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in images],
        manuscript_edited=False, visual_review=False)
    assert hashlib.sha256(manuscript.read_bytes()).hexdigest() == before
    (OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'figures'}, ensure_ascii=False))
    print(windows[['example', '날짜', '시간', *SLOTS, '평균']].to_json(orient='records', force_ascii=False))


if __name__ == '__main__':
    main()
