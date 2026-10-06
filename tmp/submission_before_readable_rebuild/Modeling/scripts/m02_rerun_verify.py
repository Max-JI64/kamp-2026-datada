"""Independent stdlib verification of chronological folds and both M02 objectives."""
import json
import math
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from m03_verify import ROOT, rows, sha, close, verify_metric, select

OUT = ROOT / 'Modeling/tables/m02_rerun'


def two_metrics(records):
    errors = [abs(float(r['prediction']) - float(r['actual'])) for r in records]
    weights = [float(r['daily_maximum_weight']) for r in records]
    return math.fsum(errors) / len(errors), math.fsum(e*w for e, w in zip(errors, weights)) / math.fsum(weights)


def score(values, denominator):
    return max(a/b for a, b in zip(values, denominator))


def check_peak_weights(g, complete_only=False):
    days = defaultdict(list)
    for r in g:
        days[r['timestamp'][:10]].append(r)
    for day in days.values():
        maximum = max(float(r['actual']) for r in day)
        ties = sum(float(r['actual']) == maximum for r in day)
        for r in day:
            value = 1 / ties if float(r['actual']) == maximum and len(day) == 24 else 0
            close(r['daily_maximum_weight'], value)
        if not complete_only:
            assert len(day) == 24
    return sum(len(day) == 24 for day in days.values())


def main():
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    run = json.loads((OUT / 'run.json').read_text(encoding='utf-8'))
    contract = json.loads((ROOT / 'Modeling/config/m02_rerun_contract.json').read_text(encoding='utf-8-sig'))
    assert run['status'] == 'completed' and not run['july_august_evaluated']
    assert len(run['features']) == 17 and 'lag1_last' not in run['features']
    for rel, digest in run['input_hashes'].items():
        assert sha(ROOT / rel) == digest
    for name, digest in run['outputs_sha256'].items():
        assert sha(OUT / name) == digest
    assert sha(OUT / 'selected_configurations.json') == run['configuration_sha256']
    assert sha(OUT / 'selection.json') == run['selection_sha256']
    m1 = json.loads((ROOT / 'Modeling/config/m01_contract.json').read_text(encoding='utf-8'))
    raw = {}
    for r in rows(ROOT / m1['source']):
        date, hour = int(r['날짜']), float(r['시간'])
        if 20210101 <= date <= 20210630 and hour.is_integer() and 0 <= hour <= 23:
            ts = datetime.strptime(str(date), '%Y%m%d') + timedelta(hours=int(hour))
            slots = [float(r[k]) for k in ['15분', '30분', '45분', '60분']]
            raw[ts] = {'last': slots[-1], 'mean': math.fsum(slots)/4, 'maximum': max(slots)}
    frame = {datetime.fromisoformat(r['timestamp']): r for r in rows(ROOT / 'Modeling/tables/m01/hourly_frame.csv')}
    lag_checks = 0
    for ts, r in frame.items():
        if ts >= datetime(2021, 7, 1) or r['eligible_common'] != 'True':
            continue
        for lag in [1, 2, 24, 168]:
            for name in ['mean', 'maximum']:
                close(r[f'lag{lag}_{name}'], raw[ts - timedelta(hours=lag)][name])
                lag_checks += 1
    p = rows(OUT / 'predictions.csv')
    grouped, daily, keyed = defaultdict(list), defaultdict(list), {}
    for r in p:
        ts = datetime.fromisoformat(r['timestamp'])
        assert datetime(2021, 4, 1) <= ts < datetime(2021, 7, 1)
        close(r['actual'], raw[ts]['maximum'])
        key = (r['model'], r['split'], r['timestamp'])
        assert key not in keyed
        keyed[key] = r
        e = float(r['prediction']) - float(r['actual'])
        assert math.isfinite(e)
        for k, v in {'signed_error': e, 'absolute_error': abs(e), 'under_amount': max(-e, 0), 'over_amount': max(e, 0)}.items():
            close(r[k], v)
        if r['model'].startswith('previous_'):
            lag = {'previous_hour': 1, 'previous_day': 24, 'previous_week': 168}[r['model']]
            close(r['prediction'], raw[ts - timedelta(hours=lag)]['maximum'])
        grouped[(r['target'], r['model'])].append(r)
        daily[(r['target'], r['model'], r['date'])].append(r)
    assert len(grouped) == 14 and len(p) == 14 * 2184
    expected_keys = {(r['split'], r['timestamp']) for r in grouped[('target_maximum', 'previous_hour')]}
    for g in grouped.values():
        assert len(g) == 2184 and {(r['split'], r['timestamp']) for r in g} == expected_keys
        assert check_peak_weights(g) == 91
    metric_checks = 0
    for table in ['comparison', 'condition_errors', 'daily_errors']:
        for r in rows(OUT / f'{table}.csv'):
            g = grouped[(r['target'], r['model'])]
            if table == 'comparison':
                kind = 'all' if r['split'] == 'pooled_development' else 'split'
                g = select(g, kind, r['split'])
            elif table == 'condition_errors':
                kind = r['condition']
                g = select(g, kind, r['value'])
            else:
                kind = 'all'
                g = daily[(r['target'], r['model'], r['date'])]
            verify_metric(r, g, kind)
            metric_checks += 1
    references = {'hgb_01': 'HGB_initial', 'ridge_00': 'Ridge_initial', 'elasticnet_04': 'ElasticNet_initial', 'svr_03': 'SVR_initial'}
    for r in rows(ROOT / 'Modeling/tables/m02/selected_predictions.csv'):
        if r['target'] == 'target_maximum' and r['model'] in references:
            close(keyed[(references[r['model']], r['split'], r['timestamp'])]['prediction'], r['prediction'])
    inner = defaultdict(list)
    for r in rows(OUT / 'inner_selected_predictions.csv'):
        inner[(r['outer_split'], r['model'])].append(r)
        ts = datetime.fromisoformat(r['timestamp'])
        close(r['actual'], raw[ts]['maximum'])
        close(r['previous_hour_prediction'], raw[ts-timedelta(hours=1)]['maximum'])
    for (split, model), g in inner.items():
        check_peak_weights(g, complete_only=True)
        assert len({r['timestamp'] for r in g}) == len(g)
        cutoff = {'dev_apr': '2021-04-01', 'dev_may': '2021-05-01', 'dev_jun': '2021-06-01'}[split]
        assert all('2021-02-01' <= r['timestamp'] < cutoff for r in g)
    trials = defaultdict(list)
    for r in rows(OUT / 'trials.csv'):
        close(r['balanced_score'], score((float(r['overall_MAE']), float(r['daily_maximum_MAE'])),
                                       (float(r['baseline_overall_MAE']), float(r['baseline_daily_maximum_MAE']))))
        trials[(r['split'], r['family'])].append(r)
    configs = json.loads((OUT / 'selected_configurations.json').read_text(encoding='utf-8'))
    for c in configs:
        split, model = c['split'], c['model']
        g = inner[(split, model)]
        values = two_metrics(g)
        close(values[0], c['inner_overall_MAE'])
        close(values[1], c['inner_daily_maximum_MAE'])
        b = [{**r, 'prediction': r['previous_hour_prediction']} for r in g]
        denominator = two_metrics(b)
        close(c['inner_balanced_score'], score(values, denominator))
        if model in contract['families']:
            group = trials[(split, model)]
            assert len(group) == c['trials'] == (50 if c['extended'] else 30)
            first20 = min(float(r['balanced_score']) for r in group[:20])
            first30 = min(float(r['balanced_score']) for r in group[:30])
            assert c['extended'] == (first30 <= first20 * .99)
            best = min(group, key=lambda r: (float(r['balanced_score']), int(r['trial'])))
            assert int(best['trial']) == c['best_trial']
            assert json.loads(best['parameters']) == c['parameters']
            close(best['overall_MAE'], values[0])
            close(best['daily_maximum_MAE'], values[1])
        else:
            close(sum(c['weights']), 1)
            assert all(w >= 0 for w in c['weights'])
            bytime = {m: {r['timestamp']: r for r in inner[(split, m)]} for m in c['members']}
            for r in g:
                close(r['prediction'], math.fsum(w*float(bytime[m][r['timestamp']]['prediction']) for m, w in zip(c['members'], c['weights'])))
            for r in grouped[('target_maximum', model)]:
                if r['split'] == split:
                    close(r['prediction'], math.fsum(w*float(keyed[(m, split, r['timestamp'])]['prediction']) for m, w in zip(c['members'], c['weights'])))
    for r in rows(OUT / 'fold_audit.csv'):
        assert r['train_last'] < r['validation_first'] <= r['validation_last'] < r['outer_evaluation_first']
        assert datetime.fromisoformat(r['validation_first']).month == int(r['month'])
        cutoff = r['validation_first'][:7] + '-01'
        expected_train = sum(ts.isoformat(sep=' ') < cutoff and row['eligible_common'] == 'True' for ts, row in frame.items())
        assert expected_train == int(r['train_hours'])
    trial_folds = defaultdict(list)
    for r in rows(OUT / 'trial_fold_scores.csv'):
        trial_folds[(r['split'], r['family'], r['trial'])].append(r)
    for (split, family), group in trials.items():
        for r in group:
            if family not in contract['families']:
                continue
            parts = trial_folds[(split, family, r['trial'])]
            assert len(parts) == len(contract['inner_validation_months'][split])
            close(r['overall_MAE'], math.fsum(float(x['overall_MAE'])*int(x['validation_hours']) for x in parts)/sum(int(x['validation_hours']) for x in parts))
            close(r['daily_maximum_MAE'], math.fsum(float(x['daily_maximum_MAE'])*float(x['peak_weight_sum']) for x in parts)/math.fsum(float(x['peak_weight_sum']) for x in parts))
    table = rows(OUT / 'selection_metrics.csv')
    denom = two_metrics(grouped[('target_maximum', 'previous_hour')])
    for r in table:
        values = two_metrics(grouped[('target_maximum', r['model'])])
        close(r['overall_MAE'], values[0])
        close(r['daily_maximum_MAE'], values[1])
        close(r['balanced_score'], score(values, denom))
        nondominated = not any(float(x['overall_MAE']) <= values[0] and float(x['daily_maximum_MAE']) <= values[1] and
                              (float(x['overall_MAE']) < values[0] or float(x['daily_maximum_MAE']) < values[1]) for x in table)
        assert r['nondominated'] == str(nondominated)
    selection = json.loads((OUT / 'selection.json').read_text(encoding='utf-8'))
    near = [r for r in table if r['nondominated'] == 'True' and float(r['balanced_score']) <= min(float(x['balanced_score']) for x in table)*1.01]
    order = ['previous_hour', 'previous_day', 'previous_week', 'HGB_initial', 'Ridge_initial', 'ElasticNet_initial', 'SVR_initial', 'HGB', 'XGBoost', 'LightGBM', 'CatBoost', 'ExtraTrees', 'mean5', 'weighted_top3']
    chosen = min(near, key=lambda r: (order.index(r['model']), float(r['balanced_score'])))
    assert selection['selected']['model'] == chosen['model']
    for r in rows(OUT / 'model_manifest.csv'):
        assert sha(ROOT / r['file']) == r['sha256'] and r['reload_predictions_checked'] == 'True'
    result = {'status': 'passed', 'verified_at': datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='minutes'),
              'prediction_rows': len(p), 'metric_rows': metric_checks, 'raw_lag_values_checked': lag_checks,
              'inner_configurations_checked': len(configs), 'trials_checked': sum(map(len, trials.values())),
              'identical_outer_timestamps': 2184, 'daily_maximum_dates': 91,
              'selected_model': chosen['model'], 'july_august_evaluated': False,
              'run_sha256': sha(OUT / 'run.json'), 'verifier_sha256': sha(Path(__file__))}
    (OUT / 'independent_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
