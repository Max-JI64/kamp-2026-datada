"""EDA 2.4 candidate: temperature shape and comparable-hour contrasts.

Run: python EDA/scripts/eda_weather_power.py
Tables: EDA/tables/weather_power; unselected plot previews: tmp/eda_weather_power.
No manuscript edits, fitted prediction model, causal claims or iid p-values.
"""
from pathlib import Path
import argparse
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from PIL import Image

from eda_variable_relations import source_frame, SHA
import eda_figures as style

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / 'EDA/tables/weather_power'
FIGURES = ROOT / 'tmp/eda_weather_power'
POWER = '평균 전력'


def save_table(frame, name):
    frame.to_csv(TABLES / f'{name}.csv', index=False, encoding='utf-8-sig')


def write_json(name, value):
    (TABLES / f'{name}.json').write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def temperature_bins(data, width=2):
    """Fixed-width temperature intervals, separated by recorded production state."""
    frame = data.copy()
    frame['bin_left'] = np.floor(frame['기온'] / width) * width
    rows = []
    for (state, left), cell in frame.groupby(['state', 'bin_left']):
        rows.append(dict(state=state, bin_left=left, bin_right=left + width,
                         n=len(cell), dates=cell['날짜'].nunique(),
                         temperature_mean=cell['기온'].mean(),
                         power_mean=cell[POWER].mean(), power_median=cell[POWER].median(),
                         power_q10=cell[POWER].quantile(.1),
                         power_q90=cell[POWER].quantile(.9),
                         production_mean=cell['생산량'].mean(),
                         displayed=len(cell) >= 30 and cell['날짜'].nunique() >= 5))
    result = pd.DataFrame(rows)
    assert result.n.sum() == len(data)
    return result


def match_hours(data, production_tolerance=.10, min_temperature_gap=2,
                humidity=False, daily_state=False, humidity_caliper=None):
    """Deterministic non-reuse matching within month/day-type/exact clock hour.

    For temperature comparison: >= stated temperature gap, minimum production
    relative gap first, then minimum temperature gap, then record IDs.
    For humidity comparison: temperature gap <=1 C and humidity gap >=10 pp.
    Zero production is matched only to zero production. Relative production gap
    is abs(p1-p2)/max(p1,p2). Each record used at most once per comparison.
    Greedy matching describes a supported subset, not an optimal causal match.
    """
    pairs = []
    group_keys = ['state', 'month', 'weekend', '시간']
    if daily_state:
        group_keys.append('daily_any_production')
    frame = data
    if humidity:
        frame = frame.dropna(subset=['습도'])
    for keys, cell in frame.groupby(group_keys, sort=True):
        values = cell.sort_values('record_id')
        ids = values.record_id.to_numpy()
        temps = values['기온'].to_numpy(float)
        prods = values['생산량'].to_numpy(float)
        i, j = np.triu_indices(len(values), k=1)
        if not len(i):
            continue
        denominator = np.maximum(prods[i], prods[j])
        pgap = np.divide(np.abs(prods[i] - prods[j]), denominator,
                         out=np.zeros(len(i)), where=denominator > 0)
        tgap = np.abs(temps[i] - temps[j])
        allowed = pgap <= production_tolerance + 1e-12
        if humidity:
            hums = values['습도'].to_numpy(float)
            hgap = np.abs(hums[i] - hums[j])
            allowed &= (tgap <= 1 + 1e-12) & (hgap >= 10 - 1e-12)
            order_gap = tgap
        else:
            allowed &= tgap >= min_temperature_gap - 1e-12
            order_gap = tgap
            if humidity_caliper is not None:
                hums = values['습도'].to_numpy(float)
                allowed &= np.abs(hums[i] - hums[j]) <= humidity_caliper + 1e-12
        candidates = np.flatnonzero(allowed)
        order = candidates[np.lexsort((ids[j[candidates]], ids[i[candidates]],
                                      order_gap[candidates], pgap[candidates]))]
        used = set()
        for k in order:
            a, b = int(i[k]), int(j[k])
            if a in used or b in used:
                continue
            used.update([a, b])
            axis_values = hums if humidity else temps
            low, high = (a, b) if axis_values[a] < axis_values[b] else (b, a)
            lo, hi = values.iloc[low], values.iloc[high]
            pairs.append(dict(
                state=keys[0], month=int(keys[1]), weekend=bool(keys[2]), hour=int(keys[3]),
                low_id=int(lo.record_id), high_id=int(hi.record_id),
                low_date=int(lo['날짜']), high_date=int(hi['날짜']),
                low_temperature=float(lo['기온']), high_temperature=float(hi['기온']),
                low_humidity=float(lo['습도']), high_humidity=float(hi['습도']),
                low_production=float(lo['생산량']), high_production=float(hi['생산량']),
                production_relative_gap=float(pgap[k]), temperature_gap=float(tgap[k]),
                low_power=float(lo[POWER]), high_power=float(hi[POWER]),
                power_difference=float(hi[POWER] - lo[POWER])))
    result = pd.DataFrame(pairs)
    if len(result):
        assert not pd.concat([result.low_id, result.high_id]).duplicated().any()
        assert result.production_relative_gap.max() <= production_tolerance + 1e-12
        if humidity:
            assert result.temperature_gap.max() <= 1 + 1e-12
            assert ((result.high_humidity - result.low_humidity) >= 10 - 1e-12).all()
        else:
            assert ((result.high_temperature - result.low_temperature) >= min_temperature_gap - 1e-12).all()
    return result


def summarize_pairs(pairs, by_month=False):
    rows = []
    keys = ['state', 'month'] if by_month else ['state']
    for group, cell in pairs.groupby(keys):
        states = group if isinstance(group, tuple) else (group,)
        dates = pd.concat([cell.low_date, cell.high_date]).unique()
        # Delete all pairs touching each date; dependence-aware descriptive
        # stability check, not a confidence interval or a calibrated hypothesis test.
        omitted = [cell.loc[~(cell.low_date.eq(d) | cell.high_date.eq(d)),
                            'power_difference'].mean() for d in dates]
        finite_omitted = [value for value in omitted if np.isfinite(value)]
        rows.append(dict(
            state=states[0], **({'month': int(states[1])} if by_month else {}),
            pairs=len(cell), hours=2 * len(cell), dates=len(dates),
            power_difference_mean=cell.power_difference.mean(),
            power_difference_median=cell.power_difference.median(),
            difference_q25=cell.power_difference.quantile(.25),
            difference_q75=cell.power_difference.quantile(.75),
            positive_fraction=cell.power_difference.gt(0).mean(),
            temperature_gap_mean=cell.temperature_gap.mean(),
            production_relative_gap_mean=cell.production_relative_gap.mean(),
            production_low_mean=cell.low_production.mean(),
            production_high_mean=cell.high_production.mean(),
            humidity_low_mean=cell.low_humidity.mean(),
            humidity_high_mean=cell.high_humidity.mean(),
            omit_one_date_mean_min=float(min(finite_omitted)) if finite_omitted else None,
            omit_one_date_mean_max=float(max(finite_omitted)) if finite_omitted else None))
    return pd.DataFrame(rows)


def finish(fig, name):
    assert fig._suptitle is None and not fig.texts
    fig.tight_layout(pad=1.4)
    target = FIGURES / f'{name}.png'
    fig.savefig(target, dpi=180)
    style.plt.close(fig)
    return target


def plots(data, bins, pairs, monthly, humidity_pairs, humidity_summary):
    style.setup()
    assert data[POWER].between(0, 220).all(), 'Expand plotted power axes to include all records'
    images = []
    fig, axes = style.plt.subplots(1, 2, figsize=(12, 4.7), sharex=True, sharey=True)
    for ax, state, color, title in zip(
            axes, ['zero', 'positive'], [style.ORANGE, style.BLUE], ['생산량 0', '생산량 양수']):
        cell = data.loc[data.state.eq(state)]
        ax.scatter(cell['기온'], cell[POWER], s=9, alpha=.14, color=color,
                   edgecolors='none', rasterized=True, label='시간별 기록')
        curve = bins.loc[bins.state.eq(state)].sort_values('bin_left').copy()
        curve.loc[~curve.displayed, 'power_mean'] = np.nan
        ax.plot(curve.temperature_mean, curve.power_mean, '-o', color='#243746',
                markersize=4, linewidth=2, label='2°C 구간 평균')
        ax.set(title=title, xlabel='기온 (°C)', ylim=(-5, 225))
        ax.grid(alpha=.16)
        ax.legend(loc='upper left', fontsize=9)
    axes[0].set_ylabel('시간별 평균 전력')
    images.append(finish(fig, '01_temperature_power'))

    fig, axes = style.plt.subplots(1, 2, figsize=(11.4, 5.2), sharex=True, sharey=True)
    for ax, state, color, title in zip(
            axes, ['zero', 'positive'], [style.ORANGE, style.BLUE], ['생산량 0', '생산량 양수']):
        cell = pairs.loc[pairs.state.eq(state)]
        ax.scatter(cell.low_power, cell.high_power, s=16, alpha=.35,
                   color=color, edgecolors='none', label='비교한 기록 쌍')
        ax.plot([0, 220], [0, 220], '--', color=style.GRAY, linewidth=1.3,
                label='두 전력이 같음')
        ax.set(title=title, xlabel='낮은 기온의 전력', xlim=(-5, 225), ylim=(-5, 225))
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.16)
        ax.legend(loc='upper left', fontsize=9)
    axes[0].set_ylabel('높은 기온의 전력')
    images.append(finish(fig, '02_comparable_temperature_pairs'))

    fig, axes = style.plt.subplots(1, 2, figsize=(11.5, 4.5), sharex=True, sharey=True)
    for ax, state, color, title in zip(
            axes, ['zero', 'positive'], [style.ORANGE, style.BLUE], ['생산량 0', '생산량 양수']):
        cell = monthly.loc[monthly.state.eq(state)].set_index('month').reindex(range(1, 9))
        ax.axhline(0, color=style.GRAY, linewidth=1, linestyle='--')
        ax.plot(cell.index, cell.power_difference_mean, '-o', color=color, linewidth=2)
        ax.set(title=title, xlabel='월', xticks=range(1, 9))
        ax.grid(alpha=.16)
    axes[0].set_ylabel('전력 차이 평균 (높은 기온 - 낮은 기온)')
    images.append(finish(fig, '03_monthly_temperature_difference'))

    # Only display humidity groups with >=30 pairs and >=20 represented dates.
    eligible = humidity_summary.loc[
        humidity_summary.pairs.ge(30) & humidity_summary.dates.ge(20), 'state'].tolist()
    eligible = [state for state in ['zero', 'positive'] if state in eligible]
    if eligible:
        fig, axes = style.plt.subplots(1, len(eligible), figsize=(5.7 * len(eligible), 5.1),
                                      sharex=True, sharey=True, squeeze=False)
        for ax, state in zip(axes[0], eligible):
            cell = humidity_pairs.loc[humidity_pairs.state.eq(state)]
            color = style.ORANGE if state == 'zero' else style.BLUE
            ax.scatter(cell.low_power, cell.high_power, s=16, alpha=.35,
                       color=color, edgecolors='none', label='비교한 기록 쌍')
            ax.plot([0, 220], [0, 220], '--', color=style.GRAY, linewidth=1.3,
                    label='두 전력이 같음')
            ax.set(title='생산량 0' if state == 'zero' else '생산량 양수',
                   xlabel='낮은 습도의 전력', xlim=(-5, 225), ylim=(-5, 225))
            ax.set_aspect('equal', adjustable='box')
            ax.grid(alpha=.16)
            ax.legend(loc='upper left', fontsize=9)
        axes[0, 0].set_ylabel('높은 습도의 전력')
        images.append(finish(fig, '04_comparable_humidity_pairs'))
    contacts = []
    for start in range(0, len(images), 2):
        thumbnails = []
        for path in images[start:start + 2]:
            with Image.open(path) as original:
                thumbnails.append(original.resize((original.width // 2, original.height // 2)))
        canvas = Image.new('RGB', (max(i.width for i in thumbnails),
                                  sum(i.height for i in thumbnails)), 'white')
        y = 0
        for thumb in thumbnails:
            canvas.paste(thumb, (0, y))
            y += thumb.height
        contact = FIGURES / f'contact_{start // 2 + 1:02d}_50.png'
        canvas.save(contact)
        contacts.append(str(contact))
    return images, contacts


def humidity_control_followup(data):
    """Follow the July positive contrast whose humidity differed between sides."""
    results = []
    july = data.loc[data.month.eq(7) & data.state.eq('positive')]
    for caliper in [5, 10, 15]:
        pairs = match_hours(july, humidity_caliper=caliper)
        if pairs.empty:
            results.append(dict(humidity_caliper=caliper, pairs=0))
            continue
        summary = summarize_pairs(pairs)
        summary.insert(0, 'humidity_caliper', caliper)
        results.extend(summary.to_dict('records'))
        save_table(pairs, f'july_temperature_pairs_humidity_{caliper}pp')
    table = pd.DataFrame(results)
    save_table(table, 'july_temperature_humidity_control')
    print(json.dumps(dict(july_temperature_humidity_control=results), ensure_ascii=False))
    return results


def main(followup_only=False, plots_only=False):
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    manuscript = ROOT / 'EDA/10.02_002_EDA_새원고.md'
    before = hashlib.sha256(manuscript.read_bytes()).hexdigest()
    if followup_only:
        data = source_frame().reset_index(drop=True)
        data['record_id'] = np.arange(len(data))
        data['state'] = np.where(data['생산량'].eq(0), 'zero', 'positive')
        results = humidity_control_followup(data)
        assert hashlib.sha256(manuscript.read_bytes()).hexdigest() == before
        manifest_path = TABLES / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['july_temperature_humidity_control'] = results
        write_json('manifest', manifest)
        return
    if plots_only:
        data = source_frame().reset_index(drop=True)
        data['record_id'] = np.arange(len(data))
        data['state'] = np.where(data['생산량'].eq(0), 'zero', 'positive')
        def read_table(name):
            return pd.read_csv(TABLES / f'{name}.csv', encoding='utf-8-sig')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            images, contacts = plots(data, read_table('temperature_bins_2c'),
                                     read_table('temperature_pairs_same_daily_state'),
                                     read_table('temperature_same_daily_state_monthly'),
                                     read_table('humidity_pairs'), read_table('humidity_pair_summary'))
        important = [str(w.message) for w in caught if 'Glyph' in str(w.message) or 'layout' in str(w.message).lower()]
        assert not important, important
        assert hashlib.sha256(manuscript.read_bytes()).hexdigest() == before
        manifest = json.loads((TABLES / 'manifest.json').read_text(encoding='utf-8'))
        manifest.update(images=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in images],
                        contacts=contacts, warnings=important,
                        figure_matching='Same daily production presence added to initial matching keys',
                        image_analysis_performed=False)
        write_json('manifest', manifest)
        print(json.dumps(dict(status='passed', warnings=important, plots=len(images), manuscript_unchanged=True)))
        return
    write_json('plan', dict(
        previous_evidence='Current manuscript 2.3: temperature overall rho .058, monthly signs vary; August production-positive rho .599',
        question='Temperature-power shape and whether differences remain among comparable hours',
        scope='Jan-Aug all 5784 valid hours; no prediction split, extreme-value removal or peak threshold',
        outcome='CSV mean (rounded hourly mean of four power values)',
        main_matching=dict(keys=['month', 'weekday/weekend', 'exact clock hour', 'zero/positive production'],
                           production_relative_gap_max=.10, temperature_gap_min=2,
                           reuse=False, algorithm='Greedy minimum production gap, then temperature gap, then row IDs'),
        sensitivities=dict(production_tolerance=[.05, .20], temperature_gap=[1, 3], temperature_bin_width=[3, 4]),
        humidity_followup='Same keys, production gap <=10%, temperature gap <=1 C, humidity gap >=10 percentage points; display only >=30 pairs and >=20 dates',
        interpretation='Descriptive matched-subset contrasts; monitor production balance and month-dependent signs; no causal or iid significance inference',
        output_policy='Preview images in tmp only; current manuscript unchanged'))
    data = source_frame().reset_index(drop=True)
    data['record_id'] = np.arange(len(data))
    assert data[['생산량', POWER, '기온']].notna().all().all()
    assert data['생산량'].ge(0).all()
    data['state'] = np.where(data['생산량'].eq(0), 'zero', 'positive')
    data['daily_any_production'] = data.groupby('날짜')['생산량'].transform('max').gt(0)
    bins = temperature_bins(data)
    save_table(bins, 'temperature_bins_2c')
    for width in [3, 4]:
        save_table(temperature_bins(data, width), f'temperature_bins_{width}c')
    pairs = match_hours(data)
    assert len(pairs) > 0
    summary = summarize_pairs(pairs)
    monthly = summarize_pairs(pairs, by_month=True)
    save_table(pairs, 'temperature_pairs')
    save_table(summary, 'temperature_pair_summary')
    save_table(monthly, 'temperature_pair_monthly')
    sensitivity = []
    for name, tolerance, gap in [('main', .1, 2), ('production_exact', 0, 2), ('production_5pct', .05, 2),
                                  ('production_20pct', .2, 2), ('temperature_1c', .1, 1),
                                  ('temperature_3c', .1, 3)]:
        current = pairs if name == 'main' else match_hours(data, tolerance, gap)
        result = summarize_pairs(current)
        result.insert(0, 'setting', name)
        sensitivity.append(result)
        if name != 'main':
            save_table(summarize_pairs(current, by_month=True), f'sensitivity_{name}_monthly')
    sensitivity = pd.concat(sensitivity, ignore_index=True)
    save_table(sensitivity, 'temperature_sensitivity')
    humidity_pairs = match_hours(data, humidity=True)
    humidity_summary = summarize_pairs(humidity_pairs)
    save_table(humidity_pairs, 'humidity_pairs')
    save_table(humidity_summary, 'humidity_pair_summary')
    save_table(summarize_pairs(humidity_pairs, by_month=True), 'humidity_pair_monthly')
    # Follow up the vertical/horizontal arms in zero-production pair scatter:
    # hourly production zero does not mean the entire date has no production.
    lookup = data.set_index('record_id')
    context = pairs.copy()
    context['low_daily_any_production'] = lookup.loc[context.low_id.to_numpy(), 'daily_any_production'].to_numpy()
    context['high_daily_any_production'] = lookup.loc[context.high_id.to_numpy(), 'daily_any_production'].to_numpy()
    context['same_daily_production_state'] = context.low_daily_any_production.eq(context.high_daily_any_production)
    context_rows = []
    for (state, same), cell in context.groupby(['state', 'same_daily_production_state']):
        context_rows.append(dict(state=state, same_daily_production_state=bool(same),
                                 pairs=len(cell), power_difference_mean=cell.power_difference.mean(),
                                 absolute_power_difference_mean=cell.power_difference.abs().mean(),
                                 absolute_difference_ge50=int(cell.power_difference.abs().ge(50).sum())))
    save_table(pd.DataFrame(context_rows), 'daily_production_context')
    daily_matched = match_hours(data, daily_state=True)
    daily_summary = summarize_pairs(daily_matched)
    daily_monthly = summarize_pairs(daily_matched, by_month=True)
    save_table(daily_matched, 'temperature_pairs_same_daily_state')
    save_table(daily_summary, 'temperature_same_daily_state_summary')
    save_table(daily_monthly, 'temperature_same_daily_state_monthly')
    july_humidity_control = humidity_control_followup(data)
    # Independently validate pair source identity and same comparison keys.
    for frame in [pairs, humidity_pairs, daily_matched]:
        lo = lookup.loc[frame.low_id.to_numpy()].reset_index(drop=True)
        hi = lookup.loc[frame.high_id.to_numpy()].reset_index(drop=True)
        for key in ['month', 'weekend', '시간', 'state']:
            assert lo[key].equals(hi[key])
        np.testing.assert_allclose(frame.power_difference, hi[POWER] - lo[POWER])
        assert (lo['날짜'] != hi['날짜']).all()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        images, contacts = plots(data, bins, daily_matched, daily_monthly, humidity_pairs, humidity_summary)
    important = [str(w.message) for w in caught if 'Glyph' in str(w.message) or 'layout' in str(w.message).lower()]
    assert not important, important
    assert hashlib.sha256(manuscript.read_bytes()).hexdigest() == before
    support = data.groupby('state').agg(hours=('record_id', 'size'), dates=('날짜', 'nunique'))
    save_table(support.reset_index(), 'population')
    manifest = dict(status='passed', source_sha256=SHA, rows=len(data),
                    manuscript_unchanged=True, population=support.reset_index().to_dict('records'),
                    temperature_summary=summary.to_dict('records'),
                    temperature_monthly=monthly.to_dict('records'),
                    humidity_summary=humidity_summary.to_dict('records'),
                    daily_state_summary=daily_summary.to_dict('records'),
                    july_temperature_humidity_control=july_humidity_control,
                    figure_matching='Same daily production presence added to initial matching keys',
                    warnings=important, image_analysis_performed=False,
                    images=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in images],
                    contacts=contacts)
    write_json('manifest', manifest)
    print(json.dumps(dict(status='passed', temperature=summary.to_dict('records'),
                          monthly=monthly[['state', 'month', 'pairs', 'power_difference_mean']].to_dict('records'),
                          sensitivity=sensitivity[['setting', 'state', 'pairs', 'power_difference_mean']].to_dict('records'),
                          humidity=humidity_summary.to_dict('records'), contacts=contacts), ensure_ascii=False))
    print(json.dumps(dict(daily_context=context_rows, daily_summary=daily_summary.to_dict('records'),
                          daily_monthly=daily_monthly[['state', 'month', 'pairs', 'power_difference_mean']].to_dict('records')), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--followup-only', action='store_true',
                        help='Reuse the main results; only compare July temperatures at similar humidity')
    parser.add_argument('--plots-only', action='store_true',
                        help='Reuse saved result tables and regenerate the preview plots')
    args = parser.parse_args()
    assert not (args.followup_only and args.plots_only), 'Choose one targeted option'
    main(followup_only=args.followup_only, plots_only=args.plots_only)
