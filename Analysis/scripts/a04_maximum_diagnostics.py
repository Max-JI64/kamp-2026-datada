"""Follow-up to A04: observed daily priorities and robustness of terminal slot."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from a01_production_changes import SLOTS, load, SOURCE, SHA
from a02_prior_power_signals import residuals, weighted_corr
from a04_maximum_conditions import OUT, KEYS, association, control_array, save


def daily_priorities(data):
    rows = []
    for date, cell in data.groupby('date'):
        maximum = cell[SLOTS].max(axis=1)
        best_maximum = maximum.max()
        top_mean = cell.level.eq(cell.level.max())
        top_maximum = maximum.eq(best_maximum)
        chosen = maximum.loc[top_mean]
        rows.append(dict(date=date, month=int(cell.month.iloc[0]), weekend=int(cell.weekend.iloc[0]),
                         daily_maximum=best_maximum, highest_hourly_mean=cell.level.max(),
                         n_top_mean=int(top_mean.sum()), n_top_maximum=int(top_maximum.sum()),
                         common_top_hours=int((top_mean & top_maximum).sum()),
                         no_common_top=int(not (top_mean & top_maximum).any()),
                         top_mean_hours='|'.join(cell.loc[top_mean, '시간'].astype(str)),
                         top_maximum_hours='|'.join(cell.loc[top_maximum, '시간'].astype(str)),
                         maximum_in_mean_top_best=chosen.max(), maximum_in_mean_top_worst=chosen.min(),
                         minimum_missed_maximum=best_maximum - chosen.max(),
                         maximum_missed_maximum=best_maximum - chosen.min(),
                         equal_tie_average_gap=best_maximum - chosen.mean()))
    frame = pd.DataFrame(rows)
    save(frame, 'daily_priority_diagnostics')
    summaries = []
    scopes = dict(all=frame, weekday=frame.loc[frame.weekend.eq(0)], weekend=frame.loc[frame.weekend.eq(1)],
                  disjoint=frame.loc[frame.no_common_top.eq(1)])
    for scope, cell in scopes.items():
        summaries.append(dict(scope=scope, n_days=len(cell), disjoint_days=int(cell.no_common_top.sum()),
                              mean_minimum_gap=cell.minimum_missed_maximum.mean(),
                              median_minimum_gap=cell.minimum_missed_maximum.median(),
                              q90_minimum_gap=cell.minimum_missed_maximum.quantile(.9),
                              largest_minimum_gap=cell.minimum_missed_maximum.max(),
                              tie_days=int(cell.n_top_mean.gt(1).sum())))
    for month, cell in frame.groupby('month'):
        summaries.append(dict(scope=f'month_{month}', n_days=len(cell), disjoint_days=int(cell.no_common_top.sum()),
                              mean_minimum_gap=cell.minimum_missed_maximum.mean(),
                              median_minimum_gap=cell.minimum_missed_maximum.median(),
                              q90_minimum_gap=cell.minimum_missed_maximum.quantile(.9),
                              largest_minimum_gap=cell.minimum_missed_maximum.max(),
                              tie_days=int(cell.n_top_mean.gt(1).sum())))
    save(pd.DataFrame(summaries), 'daily_priority_summary')
    return frame, summaries


def slot_group_associations(sample, mode='date_hour', degree=3, add_dates=False):
    sample = sample.loc[sample.groupby(KEYS).date.transform('nunique').ge(3)].copy()
    weights = np.ones(len(sample)) if mode == 'date_hour' else sample.profile_weight.to_numpy()
    features = ['prior_first', 'prior_second', 'prior_third', 'prior_last',
                'prior_slot_trend', 'prior_last_above_mean']
    targets = ['target_maximum', 'level', 'target_excess']
    values = np.column_stack([rankdata(sample[column]) / len(sample) for column in features + targets])
    controls = control_array(sample, degree=degree)
    if add_dates:
        dates = pd.factorize(sample.date)[0]
        controls = np.column_stack([controls, np.eye(dates.max() + 1)[dates]])
    cells = pd.factorize(pd.MultiIndex.from_frame(sample[KEYS]))[0]
    result = residuals(values, cells, weights, controls)
    rows = []
    for i, feature in enumerate(features):
        for j, target in enumerate(targets):
            rows.append(dict(feature=feature, target=target, n=len(sample), days=sample.date.nunique(),
                             correlation=weighted_corr(result[:, i], result[:, len(features) + j], weights)))
    return rows


def maximum_context(data, pairs):
    context = data.copy()
    context['target_maximum'] = context[SLOTS].max(axis=1)
    context['target_excess'] = context.target_maximum - context.level
    context = context.merge(pairs[['timestamp', 'transition']], on='timestamp', how='left', validate='one_to_one')
    context['transition'] = context.transition.fillna('prior_unavailable')
    context['daily_top_maximum'] = context.target_maximum.eq(context.groupby('date').target_maximum.transform('max'))
    context['daily_top_mean'] = context.level.eq(context.groupby('date')['level'].transform('max'))
    context['daily_maximum_share'] = context.daily_top_maximum / context.groupby('date').daily_top_maximum.transform('sum')
    summaries = []
    for keys in [['transition'], ['weekend', '시간'], ['weekend', 'transition']]:
        for key, cell in context.groupby(keys):
            key = key if isinstance(key, tuple) else (key,)
            summaries.append(dict(scope='_'.join(keys), **dict(zip(keys, key)), n_hours=len(cell),
                                  n_days=cell.date.nunique(), maximum_mean=cell.target_maximum.mean(),
                                  excess_mean=cell.target_excess.mean(),
                                  daily_maximum_days_equivalent=cell.daily_maximum_share.sum(),
                                  daily_maximum_hours=int(cell.daily_top_maximum.sum())))
    save(pd.DataFrame(summaries), 'daily_maximum_context')
    return context


def main():
    observed_path = OUT / 'observations_triples.csv'
    old_hash = hashlib.sha256(observed_path.read_bytes()).hexdigest()
    contract = dict(
        reason='Main A04 onset maximum135.73vs135.32(nearzero); terminal slot partialrank+.434 versus mean+.431, target-excess relation weaker. Need observed priority differences and stability, not new tuning.',
        daily='All241complete24h dates; compare observed maximum-per-hour top set vs observed hourly-mean top set, all ties retained. Report unavoidable maximum gap if sets disjoint, continuous size; this is not a forecast score.',
        terminal='Compare all four past positions together using same controls; linear vs cubic rank level/production, add date effects, inverse-profile, same-date, monthly/leave-month-out.',
        boundaries='No peak threshold, highest-mean oracle restoration, predictive training or risk classifier. Actual targets used for descriptive context only.',
        stop='If terminal association remains but excess/rank-disagreement is weak, use direct maximum candidate with cautious scope; do not seek a different cutoff or winning month.')
    (OUT / 'diagnostic_contract.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding='utf-8')
    sample = pd.read_csv(observed_path, encoding='utf-8-sig', parse_dates=['timestamp', 'input_timestamp', 'older_timestamp', 'date'])
    pairs = pd.read_csv(OUT / 'observations_pairs.csv', encoding='utf-8-sig', parse_dates=['timestamp', 'date'])
    data, _ = load()
    previous = data.set_index('timestamp').reindex(sample.input_timestamp)
    sample['prior_second'] = previous[SLOTS[1]].to_numpy()
    sample['prior_third'] = previous[SLOTS[2]].to_numpy()
    frame, summaries = daily_priorities(data)
    maximum_context(data, pairs)
    robust = []
    checks = dict(main=sample, same_date=sample.loc[sample.date.eq(sample.input_timestamp.dt.normalize())],
                  weekday=sample.loc[sample.weekend.eq(0)], known_production_zero=sample.loc[sample.past_production.eq(0)],
                  known_production_positive=sample.loc[sample.past_production.gt(0)])
    for name, cell in checks.items():
        for mode in ['date_hour', 'profile_balanced']:
            for row in slot_group_associations(cell, mode):
                robust.append(dict(check=name, weighting=mode, **row))
    for row in slot_group_associations(sample, degree=1):
        robust.append(dict(check='linear_controls', weighting='date_hour', **row))
    for row in slot_group_associations(sample, add_dates=True):
        robust.append(dict(check='date_controls', weighting='date_hour', **row))
    for month in range(1, 9):
        for name, cell in [(f'month_{month}', sample.loc[sample.month.eq(month)]),
                           (f'leave_month_{month}', sample.loc[sample.month.ne(month)])]:
            for row in slot_group_associations(cell):
                robust.append(dict(check=name, weighting='date_hour', **row))
    result = pd.DataFrame(robust)
    save(result, 'terminal_slot_robustness')
    # Independent daily calculation from the four raw columns, retaining ties.
    for row in frame.itertuples():
        cell = data.loc[data.date.eq(row.date)]
        maxima = [max(values) for values in cell[SLOTS].to_numpy()]
        means = [sum(values) / 4 for values in cell[SLOTS].to_numpy()]
        top = [maximum for maximum, mean in zip(maxima, means) if mean == max(means)]
        assert row.minimum_missed_maximum == max(maxima) - max(top)
        assert row.maximum_missed_maximum == max(maxima) - min(top)
        assert row.no_common_top == int(max(top) != max(maxima))
        assert len(cell) == 24
    assert old_hash == hashlib.sha256(observed_path.read_bytes()).hexdigest()
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SHA
    audit = dict(status='passed', daily_tie_aware_diagnostics_checked=241, observations_unchanged=True,
                 source_sha256=SHA, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 outputs_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*.csv'))})
    (OUT / 'diagnostic_verification.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(pd.DataFrame(summaries).to_json(orient='records'))
    print(result.loc[(result.feature == 'prior_last') & (result.weighting == 'date_hour') &
                     (result.target == 'target_maximum')].to_json(orient='records'))
    print(result.loc[(result.check == 'main') & (result.weighting == 'date_hour')].to_json(orient='records'))


if __name__ == '__main__':
    main()
