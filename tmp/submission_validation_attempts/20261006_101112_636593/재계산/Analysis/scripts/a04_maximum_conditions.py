"""3.4: observed next-hour maximum and available antecedents, no forecast fitting."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from a01_production_changes import ROOT, SOURCE, SHA, SLOTS, load, contrast
from a02_prior_power_signals import residuals, weighted_corr

OUT = ROOT / 'Analysis/tables/a04_maximum_conditions'
BASE = ['past_level', 'prior_maximum', 'prior_last']
SHAPE = ['prior_slot_trend', 'prior_last_above_mean', 'prior_maximum_excess',
         'prior_range', 'recent_level_delta']
TARGETS = ['target_maximum', 'level', 'target_excess']
KEYS = ['month', 'weekend', '시간']


def save(frame, name):
    frame.to_csv(OUT / f'{name}.csv', index=False, encoding='utf-8-sig')


def build():
    data, part = load()
    index = data.set_index('timestamp')
    previous = index.reindex(part.timestamp - pd.Timedelta(hours=1))
    older = index.reindex(part.timestamp - pd.Timedelta(hours=2))
    part['input_timestamp'] = part.timestamp - pd.Timedelta(hours=1)
    part['older_timestamp'] = part.timestamp - pd.Timedelta(hours=2)
    part['older_level'] = older.level.to_numpy()
    part['target_maximum'] = part[SLOTS].max(axis=1)
    part['target_excess'] = part.target_maximum - part.level
    part['target_relative_excess'] = part.target_excess / part.level.where(part.level.ne(0))
    part['prior_maximum'] = previous[SLOTS].max(axis=1).to_numpy()
    part['prior_last'] = previous[SLOTS[-1]].to_numpy()
    part['prior_first'] = previous[SLOTS[0]].to_numpy()
    part['prior_slot_trend'] = part.prior_last - part.prior_first
    part['prior_last_above_mean'] = part.prior_last - part.past_level
    part['prior_maximum_excess'] = part.prior_maximum - part.past_level
    part['prior_range'] = previous.slot_range.to_numpy()
    part['recent_level_delta'] = part.past_level - part.older_level
    part['known_production_positive'] = part.past_production.gt(0).astype(int)
    part['prior_profile'] = previous.profile.to_numpy()
    part['older_profile'] = older.profile.to_numpy()
    triples = part.loc[part.older_level.notna()].copy().reset_index(drop=True)
    triples['triple_profile'] = triples.older_profile + '|' + triples.prior_profile + '|' + triples.profile
    counts = triples[['date', 'triple_profile']].drop_duplicates().triple_profile.value_counts()
    triples['profile_weight'] = 1 / triples.triple_profile.map(counts)
    assert len(part) == 5781 and len(triples) == 5778
    return data, part, triples


def control_array(sample, include_level=True, degree=3):
    arrays = [sample.known_production_positive.to_numpy()]
    production = rankdata(sample.past_production) / len(sample)
    arrays.extend(production ** power for power in range(1, degree + 1))
    if include_level:
        level = rankdata(sample.past_level) / len(sample)
        arrays.extend(level ** power for power in range(1, degree + 1))
    return np.column_stack(arrays)


def association(sample, feature, target, adjustment, mode):
    sample = sample.copy()
    if adjustment != 'raw':
        sample = sample.loc[sample.groupby(KEYS).date.transform('nunique').ge(3)].copy()
    if len(sample) < 20:
        return None
    weights = np.ones(len(sample)) if mode == 'date_hour' else sample.profile_weight.to_numpy()
    values = np.column_stack([rankdata(sample[feature]) / len(sample),
                              rankdata(sample[target]) / len(sample)])
    if adjustment == 'raw':
        res = values
    else:
        cells = pd.factorize(pd.MultiIndex.from_frame(sample[KEYS]))[0]
        controls = control_array(sample, include_level=adjustment == 'calendar_level_production')
        res = residuals(values, cells, weights, controls)
    return dict(feature=feature, target=target, adjustment=adjustment, weighting=mode,
                n=len(sample), days=sample.date.nunique(),
                correlation=weighted_corr(res[:, 0], res[:, 1], weights))


def matched(part, sensitivity='main', boot=False):
    clone = part.copy()
    clone['change'] = clone.transition
    metrics = ['target_maximum', 'level', 'target_excess', 'target_relative_excess', 'abs_power_delta']
    pairs = [('zero_to_positive', 'positive_to_positive'),
             ('zero_to_positive', 'zero_to_zero'),
             ('positive_to_zero', 'positive_to_positive')]
    rows = []
    for a, b in pairs:
        for metric in metrics:
            for mode in ['date_hour', 'profile_balanced']:
                eligible = clone.loc[clone[metric].notna()]
                row = contrast(eligible, a, b, metric, mode, boot=boot and mode == 'date_hour')
                if row:
                    row['sensitivity'] = sensitivity
                    row['undefined_metric_hours'] = int(clone[metric].isna().sum())
                    rows.append(row)
    return rows


def independent_check(data, pairs, triples):
    raw = {}
    with SOURCE.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            date, hour = int(row['날짜']), int(row['시간'])
            if 20210101 <= date <= 20210831 and 0 <= hour <= 23:
                timestamp = pd.to_datetime(str(date), format='%Y%m%d') + pd.Timedelta(hours=hour)
                raw[timestamp] = ([float(row[key]) for key in SLOTS], float(row['생산량']))
    for row in pairs.itertuples():
        current, prod = raw[row.timestamp]
        previous, prior_prod = raw[row.input_timestamp]
        mean, past = sum(current) / 4, sum(previous) / 4
        relative = (max(current) - mean) / mean if mean else np.nan
        expected = [max(current), mean, max(current) - mean, relative,
                    past, max(previous), previous[-1], previous[0], previous[-1] - previous[0],
                    previous[-1] - past, max(previous) - past, max(previous) - min(previous)]
        got = [row.target_maximum, row.level, row.target_excess, row.target_relative_excess,
               row.past_level, row.prior_maximum, row.prior_last, row.prior_first,
               row.prior_slot_trend, row.prior_last_above_mean, row.prior_maximum_excess, row.prior_range]
        np.testing.assert_allclose(got, expected, rtol=0, atol=0)
        assert row.input_timestamp < row.timestamp
        assert row.past_production == prior_prod and getattr(row, '생산량') == prod
    for row in triples.itertuples():
        older, _ = raw[row.older_timestamp]
        assert row.recent_level_delta == row.past_level - sum(older) / 4
        assert row.older_timestamp < row.input_timestamp < row.timestamp
    tiny = triples.loc[(triples.month == 1) & (triples.weekend == 0)].copy()
    cells = pd.factorize(pd.MultiIndex.from_frame(tiny[KEYS]))[0]
    controls = control_array(tiny)
    values = np.column_stack([rankdata(tiny.prior_slot_trend), rankdata(tiny.target_maximum)])
    weights = tiny.profile_weight.to_numpy()
    computed = residuals(values, cells, weights, controls)
    dummies = np.eye(cells.max() + 1)[cells]
    design = np.column_stack([dummies, controls])
    fitted = np.linalg.lstsq(design * np.sqrt(weights[:, None]),
                             values * np.sqrt(weights[:, None]), rcond=None)[0]
    np.testing.assert_allclose(computed, values - design @ fitted, atol=1e-9)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SHA
    return dict(raw_pairs_checked=len(pairs), raw_triples_checked=len(triples),
                explicit_dummy_residual_check=True, input_timestamps_precede_target=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    contract = dict(
        previous='3.1 matched absolute mean-change37.92vs30.21;3.2 power-change association+.305 but power forecast improvement unstable;3.3 weather weak.',
        question='Are production transitions high maximum conditions, does mean conceal maximum differences, and which observed power features precede next-hour maximum?',
        outcome='Maximum of the four15min recorded values in next fully observed hour, continuous; mean and maximum-minus-mean for interpretation.',
        period='Normal Jan-Aug241days; exact1h5781pairs and exact2h5778triples; all zeros/extremes retained.',
        comparison='Same target month-hour-weekday/weekend, each production group>=2dates, common pooled-hour weights.',
        signals='Available at completed t-1; raw and calendar+known production, then observed prior mean rank cubic; never control/input target mean or future production.',
        antecedents=BASE + SHAPE, outcomes=TARGETS,
        sensitivity='Daily-array inverse-frequency, same-day pairs, month omission; week-resampling800 exploratory ranges for main matched contrasts.',
        decision='Separate absolute change from maximum level; if conditional shape signal weak or monthly heterogeneous diagnose observed-level and target-gap distinction, no favorable row or threshold search.',
        boundaries='Descriptive rank associations only; no model training, forecast scoring, peak classification, oracle maximum reconstruction, manuscript, backup.',
        zero_mean='Zero means retained for absolute/maximum metrics; only relative excess is undefined for zero denominator and omitted metric-wise with count.',
        source_sha256=SHA)
    (OUT / 'contract.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding='utf-8')
    data, pairs, triples = build()
    columns = ['timestamp', 'input_timestamp', 'older_timestamp', 'date', 'month', 'weekend', '시간',
               '생산량', 'past_production', 'known_production_positive', 'transition', 'level',
               'target_maximum', 'target_excess', 'target_relative_excess', 'past_level', 'older_level',
               *BASE[1:], 'prior_first', *SHAPE, 'abs_power_delta', 'power_delta', 'profile_weight']
    save(pairs[columns], 'observations_pairs')
    save(triples[columns], 'observations_triples')
    rows = []
    for label, keys in [('all', []), ('transition', ['transition']),
                        ('month_transition', ['month', 'transition']), ('hour', ['weekend', '시간'])]:
        groups = [((), pairs)] if not keys else pairs.groupby(keys)
        for key, cell in groups:
            key = key if isinstance(key, tuple) else (key,)
            for metric in ['target_maximum', 'level', 'target_excess', 'target_relative_excess']:
                rows.append(dict(scope=label, **dict(zip(keys, key)), metric=metric, n=len(cell),
                                 days=cell.date.nunique(), mean=cell[metric].mean(),
                                 median=cell[metric].median(), q10=cell[metric].quantile(.1),
                                 q90=cell[metric].quantile(.9), maximum=cell[metric].max()))
    save(pd.DataFrame(rows), 'descriptive')
    contrasts = matched(pairs, boot=True)
    save(pd.DataFrame(contrasts), 'matched_contrasts')
    sensitivities = matched(pairs.loc[pairs.date.eq(pairs.input_timestamp.dt.normalize())], 'same_day')
    for month in range(1, 9):
        sensitivities.extend(matched(pairs.loc[pairs.month.ne(month)], f'leave_month_{month}'))
    save(pd.DataFrame(sensitivities), 'matched_sensitivity')
    scopes = dict(all=triples, weekday=triples.loc[triples.weekend.eq(0)],
                  known_production_zero=triples.loc[triples.past_production.eq(0)],
                  known_production_positive=triples.loc[triples.past_production.gt(0)])
    signals = []
    for scope, sample in scopes.items():
        for target in TARGETS:
            for feature in BASE + SHAPE:
                adjustments = ['raw', 'calendar_production']
                if feature != 'past_level':
                    adjustments.append('calendar_level_production')
                for adjustment in adjustments:
                    for mode in ['date_hour', 'profile_balanced']:
                        row = association(sample, feature, target, adjustment, mode)
                        if row:
                            row['scope'] = scope
                            signals.append(row)
    save(pd.DataFrame(signals), 'antecedent_associations')
    monthly = []
    for month, sample in triples.groupby('month'):
        for target in TARGETS:
            for feature in ['past_level', *SHAPE]:
                adjustment = 'calendar_production' if feature == 'past_level' else 'calendar_level_production'
                row = association(sample, feature, target, adjustment, 'date_hour')
                if row:
                    row['month'] = month
                    monthly.append(row)
    save(pd.DataFrame(monthly), 'monthly_associations')
    audit = independent_check(data, pairs, triples)
    original = pd.read_csv(ROOT / 'Analysis/tables/a01_production_changes/transition_matched_contrasts.csv', encoding='utf-8-sig')
    recreated = pd.DataFrame(contrasts)
    old = original.loc[(original.a == 'zero_to_positive') & (original.b == 'positive_to_positive') &
                       (original.metric == 'abs_power_delta') & (original.weighting == 'date_hour')].iloc[0]
    new = recreated.loc[(recreated.a == old.a) & (recreated.b == old.b) &
                        (recreated.metric == old.metric) & (recreated.weighting == old.weighting)].iloc[0]
    np.testing.assert_allclose([new.adjusted_a, new.adjusted_b, new.matched_cells, new.n_a, new.n_b],
                               [old.adjusted_a, old.adjusted_b, old.matched_cells, old.n_a, old.n_b], rtol=0, atol=1e-12)
    audit.update(status='passed', a01_matched_contrast_reproduced=True,
                 source_sha256=SHA, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 outputs_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*.csv'))})
    (OUT / 'verification.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='passed', pairs=len(pairs), triples=len(triples)), ensure_ascii=False))
    print(recreated.loc[(recreated.weighting == 'date_hour') & recreated.metric.isin(['target_maximum','level','target_excess'])].to_json(orient='records'))
    signal_frame = pd.DataFrame(signals)
    print(signal_frame.loc[(signal_frame.scope == 'all') & (signal_frame.weighting == 'date_hour') &
                           (signal_frame.adjustment != 'raw')].to_json(orient='records'))


if __name__ == '__main__':
    main()
