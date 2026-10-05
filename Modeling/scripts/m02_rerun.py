"""Expanded M02: A input, nested chronological two-objective model selection."""
import json
import sys
import time
import importlib.metadata
from pathlib import Path
from m022_compare import (make_model, suggest, summarize, add_errors, optuna,
                         np, pd, joblib, ROOT, feature_columns, read_contract, sha, now)

OUT = ROOT / 'Modeling/tables/m02_rerun'
MODELS = ROOT / 'Modeling/models/m02_rerun'
CONTRACT = ROOT / 'Modeling/config/m02_rerun_contract.json'


def save(df, name):
    df.to_csv(OUT / f'{name}.csv', index=False, encoding='utf-8-sig')


def peak_weights(frame):
    """One total weight per complete date, split evenly across tied maxima."""
    date = frame.timestamp.dt.strftime('%Y-%m-%d')
    counts = frame.groupby(date).timestamp.transform('count')
    highest = frame.groupby(date).target_maximum.transform('max')
    indicator = frame.target_maximum.eq(highest) & counts.eq(24)
    ties = indicator.groupby(date).transform('sum')
    return np.where(indicator, 1 / ties.clip(lower=1), 0).astype(float)


def objectives(actual, predicted, weights):
    error = np.abs(np.asarray(actual) - np.asarray(predicted))
    assert np.isfinite(error).all() and np.sum(weights) > 0
    return float(error.mean()), float(np.average(error, weights=weights))


def balance(values, denominator):
    return max(a / b for a, b in zip(values, denominator))


def choose_trial(study, denominator):
    # Keep all Pareto outcomes; the compromise is specified before the run.
    return min(study.best_trials, key=lambda t: (balance(t.values, denominator), t.number))


def selection_table(predictions):
    rows = []
    for model, g in predictions.groupby('model'):
        overall, peak = objectives(g.actual, g.prediction, g.daily_maximum_weight)
        rows.append({'model': model, 'overall_MAE': overall, 'daily_maximum_MAE': peak})
    result = pd.DataFrame(rows)
    baseline = result.loc[result.model.eq('previous_hour')].iloc[0]
    result['relative_overall_MAE'] = result.overall_MAE / baseline.overall_MAE
    result['relative_daily_maximum_MAE'] = result.daily_maximum_MAE / baseline.daily_maximum_MAE
    result['balanced_score'] = result[['relative_overall_MAE', 'relative_daily_maximum_MAE']].max(axis=1)
    result['nondominated'] = [not (((result.overall_MAE <= r.overall_MAE) &
                                  (result.daily_maximum_MAE <= r.daily_maximum_MAE)) &
                                 ((result.overall_MAE < r.overall_MAE) |
                                  (result.daily_maximum_MAE < r.daily_maximum_MAE))).any()
                               for r in result.itertuples()]
    return result


def select_method(table):
    near = table.loc[table.nondominated & table.balanced_score.le(table.balanced_score.min() * 1.01)].copy()
    order = ['previous_hour', 'previous_day', 'previous_week', 'HGB_initial', 'Ridge_initial',
             'ElasticNet_initial', 'SVR_initial', 'HGB', 'XGBoost', 'LightGBM', 'CatBoost',
             'ExtraTrees', 'mean5', 'weighted_top3']
    near['preference'] = near.model.map({name: i for i, name in enumerate(order)})
    return near.sort_values(['preference', 'balanced_score']).iloc[0].to_dict()


def main():
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'run.json').exists(), 'Use a new version for a repeated executed run.'
    c = json.loads(CONTRACT.read_text(encoding='utf-8-sig'))
    parent = read_contract()
    previous = ROOT / 'Modeling/tables/m02'
    oldrun = json.loads((previous / 'run.json').read_text(encoding='utf-8'))
    verified = json.loads((previous / 'independent_verification.json').read_text(encoding='utf-8'))
    assert verified['status'] == 'passed'
    v1 = json.loads((ROOT / 'Modeling/tables/m01/verification.json').read_text(encoding='utf-8'))
    assert sha(ROOT / parent['source']) == parent['source_sha256']
    paths = {'Modeling/config/m02_rerun_contract.json': sha(CONTRACT),
             parent['source']: sha(ROOT / parent['source'])}
    for rel, expected in [('Modeling/tables/m01/hourly_frame.csv', v1['outputs_sha256']['hourly_frame.csv']),
                          ('Modeling/tables/m01/evaluation_diagnostics.csv', v1['outputs_sha256']['evaluation_diagnostics.csv']),
                          ('Modeling/tables/m02/selected_predictions.csv', oldrun['outputs_sha256']['selected_predictions.csv'])]:
        assert sha(ROOT / rel) == expected
        paths[rel] = expected
    frame = pd.read_csv(ROOT / 'Modeling/tables/m01/hourly_frame.csv', encoding='utf-8-sig', float_precision='round_trip', parse_dates=['timestamp'])
    frame = frame.loc[frame.eligible_common & frame.timestamp.lt('2021-07-01')].copy()
    old = pd.read_csv(previous / 'selected_predictions.csv', encoding='utf-8-sig', float_precision='round_trip', parse_dates=['timestamp'])
    old = old.loc[old.target.eq(c['target'])].copy()
    features = feature_columns('A', parent)
    assert len(features) == 17 and 'lag1_last' not in features
    runtime = {'executable': sys.executable, 'version': sys.version,
               'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'optuna', 'xgboost', 'lightgbm', 'catboost', 'joblib']}}
    run = {'status': 'running', 'started': now(), 'input_hashes': paths, 'features': features,
           'environment': runtime, 'july_august_evaluated': False,
           'selection_scope': 'retrospectively redesigned outer development; not independent final test'}
    (OUT / 'run.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    predictions, trials, fold_scores, audit, configs, manifest, inners, correlations = [], [], [], [], [], [], [], []
    started = time.perf_counter()
    for outer in parent['splits']:
        split = outer['name']
        if split not in c['outer_splits']:
            continue
        train_outer = frame.loc[frame.timestamp.le(outer['train_end'])]
        ev = frame.loc[frame.timestamp.between(outer['eval_start'], outer['eval_end'])]
        base = old.loc[old.split.eq(split) & old.model.eq('previous_hour')].copy()
        assert base.timestamp.tolist() == ev.timestamp.tolist()
        assert np.array_equal(base.actual, ev[c['target']])
        names = {'hgb_01': 'HGB_initial', 'ridge_00': 'Ridge_initial', 'elasticnet_04': 'ElasticNet_initial', 'svr_03': 'SVR_initial'}
        for record in [old.loc[old.split.eq(split) & old.model.eq(name)].copy() for name in ['previous_hour', 'previous_day', 'previous_week', *names]]:
            assert len(record) == len(ev)
            record['model'] = record.model.map(lambda n: names.get(n, n))
            predictions.append(record)
        folds = []
        for month in c['inner_validation_months'][split]:
            start = pd.Timestamp(2021, month, 1)
            train = train_outer.loc[train_outer.timestamp.lt(start)]
            val = train_outer.loc[train_outer.timestamp.ge(start) & train_outer.timestamp.lt(start + pd.offsets.MonthBegin(1))]
            assert len(train) > 300 and len(val) > 300
            assert train.timestamp.max() < val.timestamp.min() and val.timestamp.max() < ev.timestamp.min()
            weight = peak_weights(val)
            folds.append((month, train, val, weight))
            audit.append({'outer_split': split, 'kind': 'inner', 'month': month, 'train_hours': len(train),
                          'validation_hours': len(val), 'peak_weight_sum': float(weight.sum()),
                          'train_last': str(train.timestamp.max()), 'validation_first': str(val.timestamp.min()),
                          'validation_last': str(val.timestamp.max()), 'outer_evaluation_first': str(ev.timestamp.min())})
        actual = np.concatenate([v[c['target']].to_numpy() for _, _, v, _ in folds])
        times = np.concatenate([v.timestamp.to_numpy() for _, _, v, _ in folds])
        weights = np.concatenate([w for _, _, _, w in folds])
        lagpred = np.concatenate([v.lag1_maximum.to_numpy() for _, _, v, _ in folds])
        denominator = objectives(actual, lagpred, weights)
        chosen_inner, chosen_outer, inner_scores = {}, {}, {}
        def log_study(study, family):
            pareto = {t.number for t in study.best_trials}
            for t in study.trials:
                trials.append({'split': split, 'family': family, 'trial': t.number, 'state': t.state.name,
                               'overall_MAE': t.values[0], 'daily_maximum_MAE': t.values[1],
                               'baseline_overall_MAE': denominator[0], 'baseline_daily_maximum_MAE': denominator[1],
                               'balanced_score': balance(t.values, denominator), 'pareto': t.number in pareto,
                               'parameters': json.dumps(t.params), 'seconds': t.duration.total_seconds()})
        for family in c['families']:
            cache = {}
            def objective(trial):
                params = suggest(trial, family, c)
                parts = []
                for month, train, val, w in folds:
                    model = make_model(family, params)
                    model.fit(train[features], train[c['target']])
                    pred = model.predict(val[features])
                    metrics = objectives(val[c['target']], pred, w)
                    fold_scores.append({'split': split, 'family': family, 'trial': trial.number, 'month': month,
                                        'train_hours': len(train), 'validation_hours': len(val),
                                        'peak_weight_sum': float(w.sum()), 'overall_MAE': metrics[0], 'daily_maximum_MAE': metrics[1]})
                    parts.append(pred)
                predicted = np.concatenate(parts)
                cache[trial.number] = predicted
                return objectives(actual, predicted, weights)
            study = optuna.create_study(directions=['minimize', 'minimize'],
                                       sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
                                       pruner=optuna.pruners.NopPruner())
            if family == 'HGB':
                study.enqueue_trial({'max_leaf_nodes': 15, 'max_iter': 150, 'learning_rate': .05, 'min_samples_leaf': 20, 'l2_regularization': 1.0})
            study.optimize(objective, n_trials=30, n_jobs=1)
            first20 = min(balance(t.values, denominator) for t in study.trials[:20])
            extended = balance(choose_trial(study, denominator).values, denominator) <= first20 * .99
            if extended:
                study.optimize(objective, n_trials=20, n_jobs=1)
            selected = choose_trial(study, denominator)
            model = make_model(family, selected.params)
            model.fit(train_outer[features], train_outer[c['target']])
            pred = model.predict(ev[features])
            path = MODELS / f'{split}_{family}.joblib'
            joblib.dump(model, path, compress=3)
            assert np.allclose(joblib.load(path).predict(ev[features]), pred, atol=1e-9, rtol=0)
            manifest.append({'split': split, 'model': family, 'file': str(path.relative_to(ROOT)), 'sha256': sha(path), 'reload_predictions_checked': True})
            chosen_inner[family], chosen_outer[family] = cache[selected.number], pred
            inner_scores[family] = balance(selected.values, denominator)
            inners.append(pd.DataFrame({'outer_split': split, 'model': family, 'timestamp': times, 'actual': actual,
                                       'prediction': chosen_inner[family], 'daily_maximum_weight': weights, 'previous_hour_prediction': lagpred}))
            configs.append({'split': split, 'model': family, 'best_trial': selected.number, 'trials': len(study.trials),
                            'extended': extended, 'inner_overall_MAE': selected.values[0], 'inner_daily_maximum_MAE': selected.values[1],
                            'inner_balanced_score': inner_scores[family], 'parameters': selected.params,
                            'train_hours': len(train_outer), 'eval_hours': len(ev)})
            log_study(study, family)
            record = base.copy()
            record['model'], record['family'], record['prediction'] = family, family, pred
            predictions.append(record)
            save(pd.DataFrame(trials), 'trials')
            save(pd.DataFrame(fold_scores), 'trial_fold_scores')
            (OUT / 'selected_configurations.json').write_text(json.dumps(configs, indent=2), encoding='utf-8')
            print(f'{split} {family}: {len(study.trials)} trials, inner score={inner_scores[family]:.4f}, outer MAEs={objectives(base.actual, pred, base.daily_maximum_weight)}', flush=True)
        for ensemble in ['mean5', 'weighted_top3']:
            members = c['families'] if ensemble == 'mean5' else sorted(inner_scores, key=inner_scores.get)[:3]
            matrix = np.column_stack([chosen_inner[m] for m in members])
            if ensemble == 'mean5':
                w = np.ones(5) / 5
            else:
                def weighted_objective(trial):
                    raw = np.array([trial.suggest_float(f'weight_{i}', 0, 1) for i in range(3)])
                    w = raw / raw.sum() if raw.sum() else np.ones(3) / 3
                    return objectives(actual, matrix @ w, weights)
                ws = optuna.create_study(directions=['minimize', 'minimize'], sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.NopPruner())
                ws.enqueue_trial({f'weight_{i}': 1/3 for i in range(3)})
                for member in range(3):
                    ws.enqueue_trial({f'weight_{i}': float(i == member) for i in range(3)})
                ws.optimize(weighted_objective, n_trials=50, n_jobs=1)
                selected = choose_trial(ws, denominator)
                raw = np.array([selected.params[f'weight_{i}'] for i in range(3)])
                w = raw / raw.sum() if raw.sum() else np.ones(3) / 3
                log_study(ws, ensemble)
            inner_pred = matrix @ w
            pred = np.column_stack([chosen_outer[m] for m in members]) @ w
            record = base.copy()
            record['model'], record['family'], record['prediction'] = ensemble, 'Ensemble', pred
            predictions.append(record)
            configs.append({'split': split, 'model': ensemble, 'members': members, 'weights': w.tolist(),
                            'inner_overall_MAE': objectives(actual, inner_pred, weights)[0],
                            'inner_daily_maximum_MAE': objectives(actual, inner_pred, weights)[1],
                            'inner_balanced_score': balance(objectives(actual, inner_pred, weights), denominator)})
            inners.append(pd.DataFrame({'outer_split': split, 'model': ensemble, 'timestamp': times, 'actual': actual,
                                       'prediction': inner_pred, 'daily_maximum_weight': weights, 'previous_hour_prediction': lagpred}))
        corr = pd.DataFrame({m: p - actual for m, p in chosen_inner.items()}).corr()
        for i, a in enumerate(c['families']):
            for b in c['families'][i+1:]:
                correlations.append({'split': split, 'model_a': a, 'model_b': b, 'inner_residual_correlation': corr.loc[a, b]})
        save(add_errors(pd.concat(predictions, ignore_index=True)), 'predictions')
        save(pd.DataFrame(trials), 'trials')
        save(pd.DataFrame(audit), 'fold_audit')
        save(pd.DataFrame(manifest), 'model_manifest')
        save(pd.concat(inners, ignore_index=True), 'inner_selected_predictions')
        save(pd.DataFrame(correlations), 'inner_residual_correlations')
        (OUT / 'selected_configurations.json').write_text(json.dumps(configs, indent=2), encoding='utf-8')
    predictions = add_errors(pd.concat(predictions, ignore_index=True))
    assert predictions.timestamp.max() < pd.Timestamp('2021-07-01')
    summarize(predictions, OUT)
    table = selection_table(predictions)
    save(table, 'selection_metrics')
    selected = select_method(table)
    selection = {'selected': selected, 'point_best': table.sort_values('balanced_score').iloc[0].to_dict(),
                 'M03_repeat_required': selected['model'] != 'HGB_initial', 'development_only': True,
                 'normalized_against': 'previous_hour', 'selection_rule': c['provisional_selection']}
    (OUT / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    run.update({'status': 'completed', 'finished': now(), 'seconds': time.perf_counter()-started,
                'tuning_fits': len(fold_scores), 'outer_model_fits': len(manifest), 'trials_including_ensemble': len(trials),
                'prediction_rows': len(predictions), 'outer_hours_per_method': 2184,
                'outputs_sha256': {p.name: sha(p) for p in OUT.glob('*.csv')},
                'configuration_sha256': sha(OUT / 'selected_configurations.json'), 'selection_sha256': sha(OUT / 'selection.json'),
                'script_sha256': sha(Path(__file__)), 'helper_sha256': sha(ROOT / 'Modeling/scripts/m022_compare.py')})
    assert paths['Modeling/config/m02_rerun_contract.json'] == sha(CONTRACT)
    (OUT / 'run.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    print(table.sort_values('balanced_score').to_string(index=False), flush=True)
    print(json.dumps(selection), flush=True)


if __name__ == '__main__':
    main()
