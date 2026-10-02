"""Reproduce selected A003/A037/A044/A045 results directly from supplied data.

No original analysis imports or saved-result inputs. # %% marks notebook cells.
All power quantities are recorded values; no physical units are inferred.
"""
# %% Fixed scope and raw loading
from pathlib import Path
from hashlib import sha256
import argparse
import json
import platform
from datetime import datetime
import numpy as np
import pandas as pd

REPORT = Path(__file__).resolve().parents[1]
ROOT = REPORT.parent
OUT = REPORT / 'tables/analysis'
SHA = '8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830'
SLOTS = ['15분', '30분', '45분', '60분']
THRESHOLD = 182
BINS = [0, 26, 112.5, 156, 176, 182, 188, 1000]


def save(frame, name):
    frame.to_csv(OUT / f'{name}.csv', index=False, encoding='utf-8-sig')


def load(source):
    assert sha256(source.read_bytes()).hexdigest() == SHA
    raw = pd.read_csv(source, encoding='utf-8-sig')
    assert raw.shape == (6168, 18)
    f = raw.loc[raw['날짜'].lt(20210901) & raw['시간'].between(0, 23)].copy()
    f['date'] = pd.to_datetime(f['날짜'].astype(str), format='%Y%m%d')
    f['timestamp'] = f.date + pd.to_timedelta(f['시간'], unit='h')
    f = f.set_index('timestamp').sort_index()
    assert len(f) == 5784 and f.index.is_unique
    assert f.groupby('date').size().eq(24).all()
    assert f[SLOTS + ['생산량', '평균']].notna().all().all()
    assert f[SLOTS + ['생산량']].ge(0).all().all()
    f['peak'], f['month'], f['hour'] = f[SLOTS].max(axis=1), f.index.month, f.index.hour
    f['weekend'] = f.index.dayofweek >= 5
    f['period'] = np.where(f.month.le(6), 'Jan-Jun', 'Jul-Aug')
    profiles = f.groupby('date')[SLOTS].apply(lambda p: tuple(p.to_numpy().ravel()))
    daily = pd.DataFrame({'profile': pd.factorize(profiles, sort=False)[0]}, index=profiles.index)
    daily['period'] = np.where(daily.index.month <= 6, 'Jan-Jun', 'Jul-Aug')
    daily['frequency'] = daily.groupby(['period', 'profile']).profile.transform('size')
    f['profile'] = f.date.map(daily.profile)
    f['profile_weight'] = 1 / f.date.map(daily.frequency)
    assert len(daily) == 241 and daily.profile.nunique() == 126
    return raw, f


# %% 3.1 Same-hour composition and leave-one-date/week-out sensitivity (A037)
def distribution(f):
    w = f.loc[f.month.isin([6, 7, 8]) & ~f.weekend & f['생산량'].gt(0)].copy()
    w['week'] = w.date - pd.to_timedelta(w.date.dt.dayofweek, unit='D')
    summaries, bins, exclusions, references = [], [], [], []

    def weights(part, ref):
        totals = part.groupby('hour').base.sum()
        return None if set(totals.index) != set(range(24)) else part.base * part.hour.map(ref / totals)

    for weighted in (False, True):
        w['base'] = w.profile_weight if weighted else 1.0
        ref = w.groupby('hour').base.sum()
        ref /= ref.sum()  # pooled June-August distribution, NOT uniform 1/24
        references.extend(dict(profile_weighted=weighted, hour=h, reference_weight=v) for h, v in ref.items())
        for month, p in w.groupby('month'):
            wt = weights(p, ref).to_numpy()
            assert np.isclose(wt.sum(), 1)
            row = dict(profile_weighted=weighted, month=month, hours=len(p), dates=p.date.nunique())
            for metric, col in [('mean', '평균'), ('max', 'peak')]:
                v = p[col].to_numpy()
                row[f'{metric}_average'] = wt @ v
                order = np.argsort(v, kind='stable')
                for q in [.1, .25, .5, .75, .9, .95]:
                    row[f'{metric}_q{int(q*100)}'] = v[order][np.searchsorted(np.cumsum(wt[order]), q)]
                for low, high in zip(BINS[:-1], BINS[1:]):
                    mask = (v >= low) & (v < high)
                    bins.append(dict(profile_weighted=weighted, month=month, metric=metric,
                                     low=low, high=high, mass=wt[mask].sum(),
                                     mean_contribution=wt[mask] @ v[mask]))
            for t in [176, 182, 188]:
                row[f'high_{t}'] = wt @ p.peak.ge(t).to_numpy()
            summaries.append(row)
        for unit in ['date', 'week']:
            for omitted in sorted(w[unit].unique()):
                records = {}
                for month, p in w.loc[w[unit].ne(omitted)].groupby('month'):
                    wt = weights(p, ref)
                    if wt is None:
                        break
                    records[month] = dict(mean=wt @ p['평균'], max=wt @ p.peak, high=wt @ p.peak.ge(182))
                valid = set(records) == {6, 7, 8}
                for month in [7, 8]:
                    exclusions.append(dict(profile_weighted=weighted, unit=unit, omitted=omitted,
                        comparison_month=month, valid=valid,
                        **{f'{m}_delta': records[month][m] - records[6][m] if valid else np.nan
                           for m in ['mean', 'max', 'high']}))
    e = pd.DataFrame(exclusions)
    sensitivity = e.loc[e.valid].groupby(['profile_weighted', 'unit', 'comparison_month'])[
        ['mean_delta', 'max_delta', 'high_delta']].agg(['min', 'max'])
    sensitivity.columns = ['_'.join(c) for c in sensitivity.columns]
    for name, frame in [('distribution', pd.DataFrame(summaries)), ('distribution_bins', pd.DataFrame(bins)),
                        ('distribution_exclusions', e), ('distribution_reference', pd.DataFrame(references)),
                        ('distribution_sensitivity', sensitivity.reset_index())]:
        save(frame, name)


# %% 3.2 Production transition, common month-hour support (A003)
def transitions(f):
    p = f.copy()
    previous = p.index - pd.Timedelta(hours=1)
    prev_peak = p.peak.reindex(previous).to_numpy()
    prev_production = p['생산량'].reindex(previous).to_numpy()
    p['eligible'] = np.isfinite(prev_peak) & (prev_peak < 182)
    p['onset'] = p.eligible & p.peak.ge(182)
    p['start'] = prev_production == 0
    p = p.loc[p.eligible & p['생산량'].gt(0)]
    strata, summaries = [], []
    for (period, month, hour), g in p.groupby(['period', 'month', 'hour']):
        row = dict(period=period, month=month, hour=hour)
        for name, state in [('start', True), ('continuing', False)]:
            a = g.loc[g.start.eq(state)]
            row.update({f'{name}_n': len(a), f'{name}_events': int(a.onset.sum()),
                        f'{name}_weight_n': a.profile_weight.sum(),
                        f'{name}_weight_events': a.loc[a.onset, 'profile_weight'].sum()})
        row['common_support'] = row['start_n'] > 0 and row['continuing_n'] > 0
        strata.append(row)
    strata = pd.DataFrame(strata)
    for period, a in strata.groupby('period'):
        common = a.loc[a.common_support]
        for weighted in (False, True):
            n, ev = ('weight_n', 'weight_events') if weighted else ('n', 'events')
            cw = common[f'start_{n}'] + common[f'continuing_{n}']
            cw /= cw.sum()
            row = dict(period=period, profile_weighted=weighted, common_strata=len(common),
                       total_strata=len(a), strata_with_either_cell_n_le_2=int(
                           common[['start_n', 'continuing_n']].min(axis=1).le(2).sum()))
            for name in ['start', 'continuing']:
                for scope, group in [('all', a), ('overlap', common)]:
                    row[f'{name}_{scope}_n'] = group[f'{name}_n'].sum()
                    row[f'{name}_{scope}_events'] = group[f'{name}_events'].sum()
                row[f'{name}_retention'] = row[f'{name}_overlap_n'] / row[f'{name}_all_n']
                row[f'{name}_crude_rate'] = a[f'{name}_{ev}'].sum() / a[f'{name}_{n}'].sum()
                row[f'{name}_overlap_rate'] = common[f'{name}_{ev}'].sum() / common[f'{name}_{n}'].sum()
                row[f'{name}_standardized_rate'] = cw @ (common[f'{name}_{ev}'] / common[f'{name}_{n}'])
            row['standardized_difference'] = row['start_standardized_rate'] - row['continuing_standardized_rate']
            row['crude_difference'] = row['start_crude_rate'] - row['continuing_crude_rate']
            summaries.append(row)
    save(p[['date', 'period', 'month', 'hour', 'start', 'onset', 'profile_weight']].reset_index(), 'transition_records')
    save(strata, 'transition_strata')
    save(pd.DataFrame(summaries), 'transition_summary')


# %% 3.3-3.4 Direct cap requirements and idealized within-day redistribution (A044)
def lowest_cap(v, budget, receive_zero):
    receivers = np.ones(len(v), dtype=bool) if receive_zero else v > 0
    floor, high = v.sum() / receivers.sum(), v.max()
    required = lambda cap: np.maximum(v - cap, 0).sum()
    if budget == 0 or high <= floor:
        return high, 0.0
    if budget >= required(floor):
        high = floor
    else:
        low = floor
        for _ in range(60):
            mid = (low + high) / 2
            if required(mid) > budget:
                low = mid
            else:
                high = mid
    moved = required(high)
    capacity = np.maximum(high - v[receivers], 0).sum()
    assert moved <= budget + 1e-7 and capacity >= moved - 1e-7
    return high, moved


def redistribution(raw, f):
    days, caps, sims, slots = [], [], [], []
    for date, p in f.groupby('date'):
        v = p[SLOTS].to_numpy(dtype=float).ravel()
        key = dict(date=date.strftime('%Y-%m-%d'), period=p.period.iloc[0], profile=int(p.profile.iloc[0]))
        row = dict(**key, month=date.strftime('%Y-%m'), sum_values=v.sum(), original_max=v.max(),
                   mean_max_ratio=v.mean()/v.max(), zero_cells=int((v == 0).sum()),
                   second_distinct=np.sort(np.unique(v))[-2])
        days.append(row)
        for i, value in enumerate(v):
            slots.append(dict(date=key['date'], slot=i, value=value))
        for cap in [176, 182, 188]:
            caps.append(dict(**key, cap=cap, original_max=v.max(), cells_ge=int((v >= cap).sum()),
                             cells_gt=int((v > cap).sum()), excess_value_sum=np.maximum(v-cap, 0).sum()))
        for budget_fraction in [0.0, .05, .10]:
            for receive_zero in [False, True]:
                budget = v.sum()*budget_fraction
                optimal, moved = lowest_cap(v, budget, receive_zero)
                sims.append(dict(**key, budget_fraction=budget_fraction, receive_zero=receive_zero,
                                 original_max=v.max(), budget_value_sum=budget, optimal_max=optimal,
                                 max_reduction=v.max()-optimal, reduction_fraction=(v.max()-optimal)/v.max(),
                                 moved_value_sum=moved, zero_cells=row['zero_cells']))
    daily, simulations = pd.DataFrame(days), pd.DataFrame(sims)
    assert len(simulations) == 1446 and daily.original_max.ge(182).sum() == 105
    joined = simulations.merge(daily[['date', 'month', 'mean_max_ratio']], on='date', validate='many_to_one')
    groups, monthly = [], []
    for period in ['Jan-Aug', 'Jan-Jun', 'Jul-Aug']:
        part = joined if period == 'Jan-Aug' else joined.loc[joined.period.eq(period)]
        for name in ['all', 'high_ge_182', 'below_182']:
            group = part if name == 'all' else part.loc[part.original_max.ge(182).eq(name == 'high_ge_182')]
            for (budget, policy), g in group.groupby(['budget_fraction', 'receive_zero']):
                for weighting in ['observed_days', 'unique_profiles']:
                    a = g if weighting == 'observed_days' else g.drop_duplicates('profile')
                    row = dict(period=period, group=name, weighting=weighting, days=g.date.nunique(),
                               unique_profiles=g.profile.nunique(), days_or_profiles=len(a),
                               budget_fraction=budget, receive_zero=policy,
                               median_original_max=a.original_max.median(), mean_reduction=a.max_reduction.mean())
                    for metric, col in [('reduction', 'max_reduction'), ('reduction_fraction', 'reduction_fraction')]:
                        for label, q in [('q25', .25), ('median', .5), ('q75', .75)]:
                            row[f'{label}_{metric}'] = a[col].quantile(q)
                    groups.append(row)
    def tied(g, col, target):
        return sorted(g.loc[np.isclose(g[col], target, rtol=0, atol=1e-8), 'date'].tolist())
    for (month, budget, policy), g in joined.groupby(['month', 'budget_fraction', 'receive_zero']):
        orig, adj = g.original_max.max(), g.optimal_max.max()
        before, after = tied(g, 'original_max', orig), tied(g, 'optimal_max', adj)
        tracked = g.loc[g.date.isin(before), 'optimal_max'].max()
        monthly.append(dict(month=month, complete_days=len(g), unique_profiles=g.profile.nunique(),
            budget_fraction=budget, receive_zero=policy, original_month_max=orig, adjusted_month_max=adj,
            month_reduction=orig-adj, month_reduction_fraction=(orig-adj)/orig,
            original_max_dates='|'.join(before), adjusted_max_dates='|'.join(after),
            date_set_relation='unchanged' if before == after else ('partial_overlap' if set(before)&set(after) else 'replaced'),
            original_date_only_adjusted_max=tracked, missed_max_by_original_date_only=adj-tracked))
    monthly = pd.DataFrame(monthly)
    invalid = raw.loc[raw['날짜'].lt(20210901) & ~raw['시간'].between(0, 23)]
    assert len(invalid) == 48
    invalid_max = float(invalid[SLOTS].max().max())
    sensitivity = monthly.loc[monthly.month.eq('2021-07')].copy()
    sensitivity['excluded_days'] = invalid['날짜'].nunique()
    sensitivity['excluded_original_max'] = invalid_max
    sensitivity['mixed_month_max'] = np.maximum(sensitivity.adjusted_month_max, invalid_max)
    sensitivity['mixed_reduction'] = sensitivity.original_month_max - sensitivity.mixed_month_max
    for frame, name in [(daily, 'daily_concentration'), (pd.DataFrame(slots), 'daily_slots'),
                        (pd.DataFrame(caps), 'direct_caps'), (simulations, 'redistribution'),
                        (pd.DataFrame(groups), 'group_summary'), (monthly, 'monthly_max'),
                        (sensitivity, 'july_sensitivity')]:
        save(frame, name)


# %% Execute and preserve provenance
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT/'data/origin/okm_augumented_2021.csv')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    raw, f = load(args.source)
    distribution(f)
    transitions(f)
    redistribution(raw, f)
    assert sha256(args.source.read_bytes()).hexdigest() == SHA
    manifest = dict(executed_at=datetime.now().astimezone().isoformat(timespec='seconds'),
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        source=str(args.source.resolve()), source_sha256=SHA,
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), rows=len(f), days=241,
        scope='2021-01 through 2021-08, complete valid-hour days', threshold=182,
        distribution_months=[6, 7, 8], distribution_bins=BINS,
        distribution_reference='pooled June-August positive-production weekday hours',
        profile_weight='inverse frequency within Jan-Jun / Jul-Aug',
        budgets=[0, .05, .1], receive_zero=[False, True], tie_atol=1e-8,
        sources=['A003', 'A037', 'A044', 'A045'], image_reading=False,
        files={p.name: dict(rows=len(pd.read_csv(p, encoding='utf-8-sig')), sha256=sha256(p.read_bytes()).hexdigest())
               for p in sorted(OUT.glob('*.csv'))})
    (OUT/'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(rows=len(f), days=241, tables=len(manifest['files']), scenarios=1446)))


if __name__ == '__main__':
    main()
