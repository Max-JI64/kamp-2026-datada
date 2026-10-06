"""Freeze development choices, then evaluate fixed January-June models in July-August."""
import argparse
import json
import math
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from m01_prepare import ROOT, SLOTS, feature_columns, read_contract, sha
from m02_compare import metric, state
from m04_rise_compare import now, classifier, gated, read_csv

OUT = ROOT / 'Modeling/tables/m05'
MODELS = ROOT / 'Modeling/models/m05'
CONTRACT = ROOT / 'Modeling/config/m05_contract.json'


def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')


def freeze():
    assert not CONTRACT.exists() and not (OUT / 'run.json').exists(), 'Preserve completed conditions.'
    OUT.mkdir(parents=True, exist_ok=True)
    parent = read_contract()
    assert sha(ROOT / parent['source']) == parent['source_sha256']
    dependencies = [parent['source'], 'Modeling/config/m01_contract.json', 'Modeling/config/m03_ab_contract.json',
        'Modeling/config/m04_rise_contract.json', 'Modeling/tables/m01/hourly_frame.csv',
        'Modeling/tables/m01/evaluation_diagnostics.csv', 'Modeling/tables/m01/split_counts.csv',
        'Modeling/tables/m04_rise/selected_configurations.json', 'Modeling/tables/m04_rise/feature_frame.csv',
        'Modeling/tables/m04_surge_risk/predictions.csv']
    for folder in ['m01', 'm04_rise', 'm04_surge_risk', 'm04_surge_calibration']:
        directory = ROOT / 'Modeling/tables' / folder
        verification = json.loads((directory / 'independent_verification.json').read_text(encoding='utf-8'))
        assert verification['status'] == 'passed'
        dependencies.append(f'Modeling/tables/{folder}/independent_verification.json')
        if folder != 'm01':
            run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
            assert run['status'] == 'completed' and verification['run_sha256'] == sha(directory / 'run.json')
            for name, digest in run['outputs_sha256'].items():
                assert sha(directory / name) == digest, (folder, name)
            dependencies.append(f'Modeling/tables/{folder}/run.json')
    configurations = json.loads((ROOT / 'Modeling/tables/m04_rise/selected_configurations.json').read_text(encoding='utf-8'))
    june = next(r for r in configurations if r['split'] == 'dev_jun')
    development = read_csv(ROOT / 'Modeling/tables/m04_surge_risk/predictions.csv', ['timestamp', 'date'])
    calibration = []
    for group in ['B_score', 'G_HGB_score']:
        part = development.loc[development.group.eq(group)].sort_values('timestamp')
        assert len(part) == 2184 and part.timestamp.min() >= pd.Timestamp('2021-04-01') and part.timestamp.max() < pd.Timestamp('2021-07-01')
        negatives = sorted(part.loc[part.increase.lt(93), 'score'].tolist(), reverse=True)
        for cap in [.01, .05]:
            k = math.floor(cap * len(negatives))
            threshold = math.nextafter(negatives[k], math.inf)
            calibration.append({'group': group, 'cap': cap, 'threshold': threshold,
                'hours': len(part), 'events': int(part.increase.ge(93).sum()), 'negatives': len(negatives),
                'budget': k, 'calibration_FP': sum(value >= threshold for value in negatives),
                'calibration_start': str(part.timestamp.min()), 'calibration_end': str(part.timestamp.max())})
    save(pd.DataFrame(calibration), 'alarm_calibration.csv')
    c = {'version': 'm05-v1', 'recorded_at': now(),
        'reason': 'Confirm M03 last-slot contribution and the post-M04 provisional G risk candidate; preserve fixed later split.',
        'source': parent['source'], 'source_sha256': parent['source_sha256'],
        'inputs_sha256': {name: sha(ROOT / name) for name in dependencies},
        'training': {'end': '2021-06-30 23:00:00', 'pool': 'common', 'hours': 4176, 'fits': 'A, B, weighted expert, prior-low HGB classifier each fitted once'},
        'evaluation': {'start': '2021-07-01 00:00:00', 'end': '2021-08-31 23:00:00', 'common_hours': 1344, 'core_hours': 1436,
            'information_time': 'Observe t-1, predict t; update hourly observations but no model fitting or threshold updates in July-August.',
            'core': 'Inference-only A/B coverage extension with the same common-trained models; separate from main comparison.',
            'history': 'Period already observed in EDA; confirmation on a later internal period, not untouched independent test.'},
        'target': 'target_maximum', 'normal_groups': ['lag1', 'lag24', 'lag168', 'A', 'B'],
        'hgb_parameters': json.loads((ROOT / 'Modeling/config/m03_ab_contract.json').read_text(encoding='utf-8'))['hgb_parameters'],
        'dynamic_features': json.loads((ROOT / 'Modeling/config/m04_rise_contract.json').read_text(encoding='utf-8'))['features']['add'],
        'G': {'source_split': 'dev_jun', 'reason': 'Latest chronological development settings, not best later score.',
            'event_weight': june['event_weight'], 'classifier_parameters': june['classifiers']['HGB']['parameters'],
            'gate': june['classifiers']['HGB']['gate'], 'expert_label': 'legacy low->above26',
            'normal_output': 'B; G is an auxiliary risk score, not a replacement point forecast.'},
        'surge_boundary': 93., 'sensitivity_boundary': 61.60000000000002,
        'alarms': {'calibration': 'Existing rolling outer April-June scores; negative-tail order statistic; no training resubstitution.',
            'primary_cap': .01, 'secondary_cap': .05, 'thresholds': calibration,
            'calibration_sha256': sha(OUT / 'alarm_calibration.csv'), 'same_threshold_for_both_later_months': True},
        'normal_confirmation': {'overall_ratio_max': .99, 'peak_ratio_max': .99,
            'peak_under': 'strictly less than A', 'lag1': 'B overall and peak MAE strictly better than lag1',
            'monthly_ratio_max': 1.01, 'profile_reweighted': 'B overall MAE no worse than A',
            'meaning': 'All gates confirm broad last-slot contribution; 1% is numerical near-tie convention, not economic optimum. Failed gates limit claims and do not trigger automatic retuning.'},
        'risk_confirmation': {'rank_ratio_min': 1.05, 'metric': 'event-count-weighted monthly non-interpolated AP',
            'recall': 'Pooled TP >= B and each nonempty month TP >= B', 'actual_false_positive_max': .01,
            'strict_gain': 'Positive TP and either additional TP or fewer FP',
            'missing_events': 'No events => not assessable; small/repeated samples restrict strength of claims.',
            'secondary': 'q90 and 5% are descriptive only, never select candidate'},
        'postprocessing': 'none; no clipping or deletion of hard cases', 'new_optuna_trials': 0,
        'stop': 'Evaluate once, independently verify, report all outcomes and retain failure conditions; no later-driven optimization.'}
    CONTRACT.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'frozen', 'recorded_at': c['recorded_at'], 'thresholds': calibration, 'no_later_scoring': True}), flush=True)


def construct_frame(c):
    f = read_csv(ROOT / 'Modeling/tables/m01/hourly_frame.csv', ['timestamp', 'date'])
    raw = read_csv(ROOT / c['source'])
    raw = raw.loc[raw['날짜'].between(20210101, 20210831) & raw['시간'].between(0, 23)].copy()
    raw['timestamp'] = pd.to_datetime(raw['날짜'].astype(str), format='%Y%m%d') + pd.to_timedelta(raw['시간'], unit='h')
    raw = raw.set_index('timestamp')
    slots = raw[SLOTS].reindex(pd.DatetimeIndex(f.timestamp-pd.Timedelta(hours=1))).to_numpy()
    f['prev_slot_rise'] = slots[:, 3]-slots[:, 0]
    f['prev_slot_range'] = np.max(slots, axis=1)-np.min(slots, axis=1)
    f['prev_slot_slope'] = slots @ np.array([-1.5, -.5, .5, 1.5])/5
    for stem in ['last', 'mean', 'maximum', 'production']:
        f[stem+'_delta'] = f['lag1_'+stem]-f['lag2_'+stem]
    f['prior_state'] = state(f.lag1_maximum, np.floor(f.lag1_mean+.5))
    f['power_state'] = state(f.target_maximum, f.diag_target_rounded_mean)
    f['power_transition'] = f.prior_state+'->'+f.power_state
    f['prior_low'] = f.prior_state.eq('low')
    f['rise_event'] = f.power_transition.eq('low->above26')
    history = dict(zip(f.timestamp, f.power_state))
    lengths, gaps = [], []
    for ts in f.timestamp:
        length, gap = 0, 0
        for lag in range(1, 7):
            previous = history.get(ts-pd.Timedelta(hours=lag))
            if previous is None:
                gap = 1
                break
            if previous != 'low':
                break
            length += 1
        lengths.append(length)
        gaps.append(gap)
    f['low_run_length6'], f['low_history_gap6'] = lengths, gaps
    f['hour'] = f.timestamp.dt.hour
    f['increase'] = f.target_maximum-f.lag1_maximum
    return f


def alarm_metrics(part, threshold, boundary):
    y = part.increase.ge(boundary).to_numpy()
    score = part.score.to_numpy()
    alarms = score >= threshold
    tp, fp = int((alarms & y).sum()), int((alarms & ~y).sum())
    n, events = len(y), int(y.sum())
    return {'hours': n, 'events': events, 'event_dates': part.loc[y, 'date'].nunique(),
        'TP': tp, 'FP': fp, 'FN': events-tp, 'TN': n-events-fp,
        'recall': tp/events if events else None, 'precision': tp/(tp+fp) if tp+fp else 0.,
        'false_positive_fraction': fp/(n-events) if n>events else None,
        'AP': float(average_precision_score(y, score)) if events else None,
        'AUROC': float(roc_auc_score(y, score)) if 0<events<n else None,
        'threshold': threshold, 'prevalence': events/n}


def subsets(g):
    yield 'all', 'all', g, None
    yield 'profile_reweighted', 'all', g, g.profile_weight
    peak = g.loc[g.daily_maximum_weight.gt(0)]
    if len(peak):
        yield 'daily_maximum', 'all', peak, peak.daily_maximum_weight
    for col in ['month', 'power_transition', 'hour', 'train_profile_overlap']:
        for value, part in g.groupby(col):
            yield col, str(value), part, None
    for label, mask in [('surge_q95', g.increase.ge(93)), ('surge_q90', g.increase.ge(61.60000000000002)),
        ('legacy_rise', g.rise_event), ('actual_zero', g.actual.eq(0)), ('prior_zero', g.lag1_maximum.eq(0)),
        ('restart_after_zero', g.lag1_maximum.eq(0)&g.actual.gt(0)), ('sharp_drop', g.increase.le(-93))]:
        if mask.any():
            yield label, 'all', g.loc[mask], None


def assess(p, metrics, risk, c):
    main = metrics.loc[metrics.pool.eq('common')]
    def value(group, period='pooled', condition='all', key='MAE'):
        return float(main.loc[main.group.eq(group)&main.period.eq(period)&main.condition.eq(condition)&main.value.eq('all'), key].iloc[0])
    normal = {
        'overall': value('B') <= c['normal_confirmation']['overall_ratio_max']*value('A'),
        'daily_maximum': value('B', condition='daily_maximum') <= c['normal_confirmation']['peak_ratio_max']*value('A', condition='daily_maximum'),
        'peak_under': value('B', condition='daily_maximum', key='mean_under') < value('A', condition='daily_maximum', key='mean_under'),
        'lag1_overall': value('B') < value('lag1'),
        'lag1_peak': value('B', condition='daily_maximum') < value('lag1', condition='daily_maximum'),
        'monthly': all(value('B', str(m)) <= 1.01*value('A', str(m)) and value('B', str(m), 'daily_maximum') <= 1.01*value('A', str(m), 'daily_maximum') for m in [7, 8]),
        'profile_reweighted': value('B', condition='profile_reweighted') <= value('A', condition='profile_reweighted')}
    table = []
    for group in ['B_score', 'G_HGB_score']:
        part = risk.loc[risk.group.eq(group)&risk.cap.eq(.01)&risk.boundary.eq(93)]
        pooled = part.loc[part.period.eq('pooled')].iloc[0]
        months = part.loc[part.period.ne('pooled')]
        count = int(months.events.sum())
        weighted_ap = float((months.AP.fillna(0)*months.events).sum()/count) if count else None
        table.append({'group': group, 'weighted_monthly_AP': weighted_ap,
            **{key: int(pooled[key]) for key in ['events', 'TP', 'FP', 'FN', 'TN']},
            **{key: None if pd.isna(pooled[key]) else float(pooled[key]) for key in ['recall', 'precision', 'false_positive_fraction']}})
    b, g = table
    if b['events']:
        base_months = risk.loc[risk.group.eq('B_score')&risk.cap.eq(.01)&risk.boundary.eq(93)&risk.period.ne('pooled')].set_index('period')
        candidate_months = risk.loc[risk.group.eq('G_HGB_score')&risk.cap.eq(.01)&risk.boundary.eq(93)&risk.period.ne('pooled')].set_index('period')
        gates = {'rank': g['weighted_monthly_AP'] >= 1.05*b['weighted_monthly_AP'], 'pooled_recall': g['TP']>=b['TP'],
            'monthly_recall': bool((candidate_months.TP>=base_months.TP).all()),
            'false_positive': g['false_positive_fraction']<=.01,
            'strict_gain': g['TP']>0 and (g['TP']>b['TP'] or g['FP']<b['FP'])}
    else:
        gates = {'assessable': False}
    save(pd.DataFrame(table), 'risk_confirmation.csv')
    return {'normal_forecast_development_candidate': 'B', 'normal_confirmation_gates': normal,
        'broad_last_slot_effect_confirmed': all(normal.values()), 'risk_confirmation_gates': gates,
        'G_risk_effect_confirmed': all(gates.values()), 'risk_summary': table,
        'no_automatic_retuning': True, 'later_period_already_observed': True,
        'normal_forecast_not_replaced_by_G': True}


def run():
    assert not (OUT / 'run.json').exists(), 'Preserve completed run.'
    c = json.loads(CONTRACT.read_text(encoding='utf-8'))
    for name, digest in c['inputs_sha256'].items():
        assert sha(ROOT / name) == digest, name
    assert sha(OUT / 'alarm_calibration.csv') == c['alarms']['calibration_sha256']
    begin = time.perf_counter()
    record = {'status': 'running', 'started': now(), 'contract_sha256': sha(CONTRACT),
        'script_sha256': sha(Path(__file__)), 'inputs_sha256': c['inputs_sha256'],
        'runtime': {'executable': sys.executable, 'version': sys.version}, 'new_optuna_trials': 0}
    (OUT / 'run.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    f = construct_frame(c)
    tr = f.loc[f.eligible_common&f.timestamp.le(c['training']['end'])].copy()
    ev = f.loc[f.eligible_common&f.timestamp.between(c['evaluation']['start'], c['evaluation']['end'])].copy()
    core = f.loc[f.eligible_core&f.timestamp.between(c['evaluation']['start'], c['evaluation']['end'])].copy()
    assert (len(tr), len(ev), len(core)) == (4176, 1344, 1436)
    assert tr.timestamp.max() < ev.timestamp.min()
    features = {name: feature_columns(name) for name in ['A', 'B']}
    features['dynamic'] = features['B']+c['dynamic_features']
    assert all(not col.startswith(('target_', 'diag_')) for cols in features.values() for col in cols)
    previous = read_csv(ROOT / 'Modeling/tables/m04_rise/feature_frame.csv', ['timestamp'])
    assert previous.timestamp.tolist() == tr.timestamp.tolist()
    assert np.allclose(previous[features['dynamic']], tr[features['dynamic']], rtol=0, atol=1e-10)
    save(pd.concat([tr.assign(role='train'), ev.assign(role='evaluation')])[['timestamp', 'role']+features['dynamic']], 'feature_frame.csv')
    estimators, manifest, audits = {}, [], []
    MODELS.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for name in ['A', 'B', 'expert', 'classifier']:
            cols = features[name] if name in ['A', 'B'] else features['dynamic']
            sample = tr.loc[tr.prior_low] if name == 'classifier' else tr
            model = classifier('HGB', c['G']['classifier_parameters']) if name == 'classifier' else HistGradientBoostingRegressor(**c['hgb_parameters'])
            target = sample.rise_event.astype(int) if name == 'classifier' else sample.target_maximum
            weights = np.where(sample.rise_event, c['G']['event_weight'], 1.) if name == 'expert' else None
            start = time.perf_counter()
            if weights is None:
                model.fit(sample[cols], target)
            else:
                model.fit(sample[cols], target, sample_weight=weights)
            path = MODELS / (name+'.joblib')
            joblib.dump(model, path, compress=3)
            loaded = joblib.load(path)
            before = model.predict_proba(ev[cols]) if name == 'classifier' else model.predict(ev[cols])
            after = loaded.predict_proba(ev[cols]) if name == 'classifier' else loaded.predict(ev[cols])
            assert np.allclose(before, after, rtol=0, atol=1e-10)
            estimators[name] = model
            manifest.append({'name': name, 'file': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'reload_checked': True})
            audits.append({'name': name, 'train_hours': len(sample), 'training_start': str(sample.timestamp.min()),
                'training_end': str(sample.timestamp.max()), 'feature_count': len(cols), 'fit_count': 1, 'seconds': time.perf_counter()-start})
        warning_messages = [str(w.message) for w in caught]
    diag = read_csv(ROOT / 'Modeling/tables/m01/evaluation_diagnostics.csv', ['timestamp', 'date'])
    predictions = []
    for pool, part in [('common', ev), ('core', core)]:
        d = diag.loc[diag.split.eq('later_jul_aug')&diag.pool.eq(pool)]
        base = part[['timestamp', 'date', 'month', 'hour', 'power_transition', 'prior_low', 'rise_event', 'lag1_maximum', 'increase', 'diag_target_profile']].merge(
            d[['timestamp', 'train_profile_overlap', 'profile_weight', 'daily_maximum_weight']], on='timestamp', validate='one_to_one')
        assert base.timestamp.tolist() == part.timestamp.tolist()
        outputs = {name: estimators[name].predict(part[features[name]]) for name in ['A', 'B']}
        if pool == 'common':
            outputs.update({lag: part[lag+'_maximum'].to_numpy() for lag in ['lag1', 'lag24', 'lag168']})
            probability = estimators['classifier'].predict_proba(part[features['dynamic']])[:, 1]
            expert = estimators['expert'].predict(part[features['dynamic']])
            outputs['G_aux'], trigger = gated(outputs['B'], expert, probability, part.prior_low, **c['G']['gate'])
            detail = base.copy()
            detail['B'], detail['expert'], detail['probability'], detail['trigger'], detail['G_aux'] = outputs['B'], expert, probability, trigger, outputs['G_aux']
            save(detail, 'G_components.csv')
        for group, values in outputs.items():
            r = base.copy()
            r['pool'], r['group'], r['actual'], r['prediction'] = pool, group, part.target_maximum.to_numpy(), values
            r['signed_error'] = r.prediction-r.actual
            r['absolute_error'], r['under_amount'], r['over_amount'] = r.signed_error.abs(), (-r.signed_error).clip(lower=0), r.signed_error.clip(lower=0)
            predictions.append(r)
    p = pd.concat(predictions, ignore_index=True)
    assert np.isfinite(p.prediction).all()
    save(p, 'predictions.csv')
    save(pd.DataFrame(manifest), 'model_manifest.csv')
    save(pd.DataFrame(audits), 'training_audit.csv')
    metrics, daily, pairs, risk_rows, risk_predictions = [], [], [], [], []
    for (pool, group), g in p.groupby(['pool', 'group']):
        for period, part in [('pooled', g)]+[(str(m), x) for m, x in g.groupby('month')]:
            for condition, value, subset, weights in subsets(part):
                metrics.append({'pool': pool, 'group': group, 'period': period, 'condition': condition, 'value': value, **metric(subset, weights)})
        for date, part in g.groupby('date'):
            daily.append({'pool': pool, 'group': group, 'date': date, **metric(part)})
    for pool in ['common', 'core']:
        a = p.loc[p.pool.eq(pool)&p.group.eq('A')]
        b = p.loc[p.pool.eq(pool)&p.group.eq('B')]
        q = a[['timestamp', 'date', 'prediction', 'absolute_error', 'under_amount', 'over_amount']].merge(
            b[['timestamp', 'prediction', 'absolute_error', 'under_amount', 'over_amount']], on='timestamp', suffixes=('_A', '_B'), validate='one_to_one')
        q['pool'] = pool
        for key in ['absolute_error', 'under_amount', 'over_amount']:
            q['delta_'+key] = q[key+'_B']-q[key+'_A']
        pairs.append(q)
    for name, source_group in [('B_score', 'B'), ('G_HGB_score', 'G_aux')]:
        part = p.loc[p.pool.eq('common')&p.group.eq(source_group)].copy()
        part['group'], part['score'] = name, part.prediction-part.lag1_maximum
        for cap in [.01, .05]:
            threshold = next(r['threshold'] for r in c['alarms']['thresholds'] if r['group'] == name and r['cap'] == cap)
            part['alarm_'+str(cap)] = part.score.ge(threshold)
            for boundary in [93., c['sensitivity_boundary']]:
                for period, subset in [('pooled', part)]+[(str(m), x) for m, x in part.groupby('month')]:
                    risk_rows.append({'group': name, 'cap': cap, 'boundary': boundary, 'period': period, **alarm_metrics(subset, threshold, boundary)})
        part['normal_prediction'] = p.loc[p.pool.eq('common')&p.group.eq('B'), 'prediction'].to_numpy()
        risk_predictions.append(part)
    metric_frame, risk_frame = pd.DataFrame(metrics), pd.DataFrame(risk_rows)
    save(metric_frame, 'metrics.csv')
    save(pd.DataFrame(daily), 'daily_errors.csv')
    save(pd.concat(pairs), 'paired_predictions.csv')
    save(pd.concat(risk_predictions), 'risk_predictions.csv')
    save(risk_frame, 'risk_metrics.csv')
    save(p.loc[p.pool.eq('common')].sort_values('absolute_error', ascending=False).groupby('group').head(20), 'largest_errors.csv')
    save(p.loc[p.pool.eq('common')&p.group.eq('B')&p.increase.ge(93)], 'surge_cases.csv')
    save(p.loc[p.group.isin(['A', 'B'])&(p.actual.eq(0)|p.lag1_maximum.eq(0))], 'zero_cases.csv')
    conclusion = assess(p, metric_frame, risk_frame, c)
    (OUT / 'conclusion.json').write_text(json.dumps(conclusion, ensure_ascii=False, indent=2), encoding='utf-8')
    record.update({'status': 'completed', 'finished': now(), 'seconds': time.perf_counter()-begin, 'fits': 4,
        'prediction_rows': len(p), 'metrics_rows': len(metric_frame), 'features': features,
        'warnings': warning_messages, 'helper_sha256': {name: sha(ROOT / 'Modeling/scripts' / name) for name in ['m01_prepare.py', 'm02_compare.py', 'm04_rise_compare.py']},
        'outputs_sha256': {path.name: sha(path) for path in OUT.iterdir() if path.name != 'run.json'}})
    (OUT / 'run.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(metric_frame.loc[metric_frame.pool.eq('common')&metric_frame.period.eq('pooled')&metric_frame.condition.isin(['all', 'daily_maximum']), ['group','condition','MAE','mean_under','mean_over']].to_string(index=False), flush=True)
    print(json.dumps(conclusion), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['freeze', 'run'], required=True)
    args = parser.parse_args()
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    freeze() if args.stage == 'freeze' else run()


if __name__ == '__main__':
    main()
