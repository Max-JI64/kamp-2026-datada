"""Approved 3.2: observed past-power signals for the next hourly change.
Exploratory associations only; no forecast fitting, threshold search or manuscript.
"""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import csv
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from a01_production_changes import ROOT, SOURCE, SHA, SLOTS, load, contrast

OUT = ROOT / 'Analysis/tables/a02_prior_power_signals'
FEATURES = ['prior_slot_trend', 'recent_level_delta']


def save(frame, name):
    frame.to_csv(OUT / f'{name}.csv', index=False, encoding='utf-8-sig')


def build():
    data, part = load()
    index = data.set_index('timestamp')
    previous = index.reindex(part.timestamp - pd.Timedelta(hours=1))
    older = index.reindex(part.timestamp - pd.Timedelta(hours=2))
    part['older_level'] = older.level.to_numpy()
    part['prior_slot_trend'] = (previous['60분'] - previous['15분']).to_numpy()
    part['recent_level_delta'] = part.past_level - part.older_level
    part['input_timestamp'] = part.timestamp - pd.Timedelta(hours=1)
    part['older_timestamp'] = part.timestamp - pd.Timedelta(hours=2)
    part['known_production_positive'] = part.past_production.gt(0).astype(int)
    part['prior_profile'] = previous.profile.to_numpy()
    part['older_profile'] = older.profile.to_numpy()
    part = part.dropna(subset=['older_level']).copy().reset_index(drop=True)
    part['triple_profile'] = part.older_profile + '|' + part.prior_profile + '|' + part.profile
    counts = part[['date', 'triple_profile']].drop_duplicates().triple_profile.value_counts()
    part['profile_weight'] = 1 / part.triple_profile.map(counts)
    assert len(part) == 5778
    return data, part


def weighted_corr(x, y, weights):
    x = x - np.average(x, weights=weights)
    y = y - np.average(y, weights=weights)
    denominator = np.sqrt(np.sum(weights*x*x) * np.sum(weights*y*y))
    return float(np.sum(weights*x*y)/denominator) if denominator > 1e-15 else np.nan


def residuals(values, cells, weights, controls=None):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    n = cells.max()+1
    sums = np.zeros((n, values.shape[1]))
    np.add.at(sums, cells, weights[:, None]*values)
    mass = np.bincount(cells, weights=weights, minlength=n)
    centered = values - (sums/mass[:, None])[cells]
    if controls is None or not controls.shape[1]:
        return centered
    centered_controls = residuals(controls, cells, weights)
    weighted_controls = centered_controls*np.sqrt(weights[:, None])
    coefficients = np.linalg.lstsq(weighted_controls,
                                   centered*np.sqrt(weights[:, None]), rcond=None)[0]
    return centered-centered_controls@coefficients


def association(sample, feature, adjustment, mode):
    sample = sample.copy()
    # All available observations remain in raw summaries. Adjusted associations
    # require at least three distinct dates within each calendar cell.
    cell_keys = ['month', 'weekend', '시간']
    if adjustment != 'raw':
        counts = sample.groupby(cell_keys).date.transform('nunique')
        sample = sample.loc[counts.ge(3)]
    if len(sample) < 10:
        return None
    weights = np.ones(len(sample)) if mode == 'date_hour' else sample.profile_weight.to_numpy()
    x = rankdata(sample[feature].to_numpy())/len(sample)
    y = rankdata(sample.power_delta.to_numpy())/len(sample)
    if adjustment == 'raw':
        value = weighted_corr(x, y, weights)
    else:
        cells = pd.factorize(pd.MultiIndex.from_frame(sample[cell_keys]))[0]
        controls = None
        if adjustment in ('calendar_level', 'calendar_level_both'):
            # Rank-level cubic permits a smooth non-linear adjustment without
            # introducing thresholds or treating the target mean as a control.
            level = rankdata(sample.past_level.to_numpy())/len(sample)
            controls = np.column_stack([level, level**2, level**3,
                                        sample.known_production_positive.to_numpy()])
            if adjustment == 'calendar_level_both':
                other = FEATURES[1] if feature == FEATURES[0] else FEATURES[0]
                controls = np.column_stack([controls, rankdata(sample[other].to_numpy())/len(sample)])
        res = residuals(np.column_stack([x, y]), cells, weights, controls)
        value = weighted_corr(res[:, 0], res[:, 1], weights)
    return dict(feature=feature, adjustment=adjustment, weighting=mode,
                n=len(sample), days=sample.date.nunique(), correlation=value)


def scopes(part):
    return {
        'all': part,
        'known_production_zero': part.loc[part.past_production.eq(0)],
        'known_production_positive': part.loc[part.past_production.gt(0)],
        'zero_to_zero_posthoc': part.loc[part.transition.eq('zero_to_zero')],
        'zero_to_positive_posthoc': part.loc[part.transition.eq('zero_to_positive')],
    }


def independent_check(part):
    raw = {}
    with SOURCE.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            date, hour = int(row['날짜']), int(row['시간'])
            if 20210101 <= date <= 20210831 and 0 <= hour <= 23:
                stamp = pd.Timestamp(str(date)) + pd.Timedelta(hours=hour)
                raw[stamp] = (tuple(int(row[c]) for c in SLOTS), int(row['생산량']))
    for row in part.itertuples():
        now, production = raw[row.timestamp]
        prior, prior_production = raw[row.input_timestamp]
        older, _ = raw[row.older_timestamp]
        expected = [sum(now)/4-sum(prior)/4, prior[-1]-prior[0],
                    sum(prior)/4-sum(older)/4, sum(prior)/4]
        np.testing.assert_allclose([row.power_delta, row.prior_slot_trend,
                                    row.recent_level_delta, row.past_level], expected, rtol=0, atol=0)
        assert row.past_production == prior_production
        assert row.input_timestamp < row.timestamp and row.older_timestamp < row.input_timestamp
    a01_path = ROOT/'Analysis/tables/a01_production_changes/observations.csv'
    a01_verification = json.loads((a01_path.parent/'verification.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(a01_path.read_bytes()).hexdigest() == a01_verification['outputs_sha256']['observations.csv']
    a01 = pd.read_csv(a01_path, encoding='utf-8-sig').set_index('timestamp')
    joined = a01.reindex(part.timestamp.astype(str))
    np.testing.assert_allclose(joined.power_delta, part.power_delta, rtol=0, atol=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    contract = dict(
        previous='A01 zero-to-positive absolute change37.92 vs30.21; EDA2.5 shows within-hour patterns.',
        question='Do the last observed hour slot trend and recent mean change precede next-hour mean changes after calendar and observed level adjustment?',
        input_time='Hour t-1 has completely finished and all four slots/production are assumed observed; actual reporting latency is unidentified.',
        target='Unrounded mean(t) minus mean(t-1), signed; never used in controls.',
        features={'prior_slot_trend': 'last minus first slot at t-1',
                  'recent_level_delta': 'mean(t-1) minus mean(t-2)'},
        scope='Same Jan-Aug valid241 days; exact consecutive t-2,t-1,t required; no gaps bridged.',
        primary='Rank associations: raw, calendar, calendar+observed rank-level cubic+known production0/positive, then other-feature adjustment.',
        calendar='Target month x hour x weekday/weekend fixed effects, adjusted scopes need >=3distinct dates per cell.',
        level_adjustment='Cubic of observed rank(mean(t-1)), no target levels, no threshold or parameter search; partial association is descriptive, not a fitted future forecast.',
        sensitivity='Inverse joint3-day profile frequency, same date for all3hours, leave each month out; no pvalue or causal significance.',
        transition='Actual production(t) used only to split posthoc diagnostic groups; cannot be a predictor.',
        next_action='If calendar-only and level-adjusted differ, diagnose scale/coupling; if conditional signal survives use as forecasting candidate, no claim of improvement.',
        forecast_training=False, manuscript=False, backup_access=False,
        source_sha256=SHA)
    (OUT/'contract.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
    _, part = build()
    independent_check(part)
    selected = ['timestamp', 'input_timestamp', 'older_timestamp', 'date', 'month', 'weekend', '시간',
                'past_production', '생산량', 'known_production_positive', 'transition',
                'older_level', 'past_level', 'level', 'prior_slot_trend', 'recent_level_delta',
                'power_delta', 'abs_power_delta', 'profile_weight']
    save(part[selected], 'observations')
    rows=[]
    for scope, sample in scopes(part).items():
        for feature in FEATURES:
            for adjustment in ('raw', 'calendar', 'calendar_level', 'calendar_level_both'):
                for mode in ('date_hour', 'profile_balanced'):
                    result=association(sample, feature, adjustment, mode)
                    if result:
                        rows.append(dict(scope=scope, **result))
    results=pd.DataFrame(rows)
    save(results, 'associations')
    stability=[]
    for scope, sample in scopes(part).items():
        for label, subset in [('same_date', sample.loc[sample['시간'].ge(2)])]+[
            (f'exclude_month_{month}', sample.loc[sample.month.ne(month)]) for month in range(1,9)]:
            for feature in FEATURES:
                for adjustment in ('calendar_level', 'calendar_level_both'):
                    result=association(subset, feature, adjustment, 'date_hour')
                    if result:
                        stability.append(dict(scope=scope, check=label, **result))
    save(pd.DataFrame(stability), 'stability')
    distributions=[]
    for scope, sample in scopes(part).items():
        for metric in FEATURES+['past_level', 'power_delta']:
            distributions.append(dict(scope=scope, metric=metric, n=len(sample),
                mean=sample[metric].mean(), median=sample[metric].median(),
                q10=sample[metric].quantile(.1), q90=sample[metric].quantile(.9)))
    save(pd.DataFrame(distributions), 'distributions')
    # Match calendar composition to compare observable antecedents among hours
    # whose known production is zero. This is a posthoc diagnostic, not detection.
    risk=part.loc[part.past_production.eq(0)].copy()
    risk['change']=risk.transition
    matched=[]
    for metric in FEATURES+['past_level', 'power_delta']:
        for mode in ('date_hour', 'profile_balanced'):
            result=contrast(risk, 'zero_to_positive', 'zero_to_zero', metric, mode)
            if result:
                matched.append(result)
    save(pd.DataFrame(matched), 'zero_production_antecedents')
    # Projection check against explicit weighted dummy least squares on a
    # deterministic small subset, avoiding duplicate full model computation.
    tiny=part.loc[part.month.eq(1)&part['시간'].isin([7,8,12,13])].copy()
    cells=pd.factorize(pd.MultiIndex.from_frame(tiny[['month','weekend','시간']]))[0]
    w=tiny.profile_weight.to_numpy()
    lev=rankdata(tiny.past_level)/len(tiny)
    controls=np.column_stack([lev, lev**2, lev**3, tiny.known_production_positive])
    values=np.column_stack([rankdata(tiny.prior_slot_trend), rankdata(tiny.power_delta)])/len(tiny)
    computed=residuals(values,cells,w,controls)
    design=np.column_stack([np.eye(cells.max()+1)[cells],controls])
    beta=np.linalg.lstsq(design*np.sqrt(w[:,None]),values*np.sqrt(w[:,None]),rcond=None)[0]
    direct=values-design@beta
    np.testing.assert_allclose(computed,direct,atol=1e-10,rtol=1e-10)
    verification=dict(status='passed',source_sha256=SHA,n=len(part),
        exact_raw_triples_checked=len(part),a01_delta_reproduced=True,
        input_timestamps_precede_target=True,zero_values_retained=True,
        calendar_projection_vs_explicit_dummies=True,
        transition_counts=part.transition.value_counts().to_dict(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outputs_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.csv')})
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(results.loc[(results.weighting=='date_hour') & results.scope.isin(['all','known_production_zero','zero_to_positive_posthoc'])].to_json(orient='records'))
    print(pd.DataFrame(matched).to_json(orient='records'))
    print(json.dumps(verification))


if __name__=='__main__':
    main()
