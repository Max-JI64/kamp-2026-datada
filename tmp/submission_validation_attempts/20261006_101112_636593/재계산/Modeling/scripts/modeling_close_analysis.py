"""Close modeling: exact Logistic contributions and raw-to-result reproduction.

No new candidate selection or hyperparameter tuning. Keeps frozen results intact.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import json
import io
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from m01_prepare import load_frame
from regime_followup import extended
from regime_forecast import model, threshold, sha, save_json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'Modeling/tables/modeling_close'
AGE = ROOT / 'Modeling/tables/regime_age_ablation'


def readj(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'contract.json').exists()
    names = ['Modeling/tables/regime_age_ablation/contract.json', 'Modeling/tables/regime_age_ablation/run.json',
             'Modeling/tables/regime_age_ablation/independent_verification.json',
             'Modeling/tables/regime_followup/contract.json', 'Modeling/tables/regime_followup/run.json',
             'Modeling/scripts/m01_prepare.py', 'Modeling/scripts/regime_followup.py',
             'Modeling/scripts/regime_diagnose.py', 'Modeling/scripts/regime_forecast.py',
             'data/origin/okm_augumented_2021.csv']
    c = {'created_at': datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='minutes'),
         'script_sha256': sha(__file__), 'inputs_sha256': {n: sha(ROOT / n) for n in names},
         'question': 'Calendar vs prior observed state contribution to fixed no-age Logistic; observed explanation only, no causal claims or new model choice.',
         'method': 'Exact standardized-feature times coefficient additive logit decomposition, grouped calendar/power/production/state; intercept retained. Not SHAP or probability percentage.',
         'case_rule': 'Among true detected onsets with positive absolute error gain choose middle sorted gain (tie timestamp); among missed onsets choose middle sorted actual maximum (tie timestamp). Both require complete t-4..t+5 common forecast windows. State selection counts explicitly.',
         'reproduction': 'Rebuild Jan-Aug exact-time features and causal labels from source; recreate train-only availability flags; refit April/May/June classifiers for OOF threshold, fixed June classifier and B0/B1; match stored predictions and gates. No retuning and no claim of clean environment.',
         'stop': 'Freeze current method as reporting candidate with limited exploratory scope; close model search after explanation, figures, deployment proposal and reproduction checks.',
         'root_readme_sha256': sha(ROOT / 'README.md')}
    save_json(OUT / 'contract.json', c)
    print('Closing analysis contract frozen.', flush=True)


def run():
    c = readj(OUT / 'contract.json'); assert not (OUT / 'run.json').exists()
    assert sha(__file__) == c['script_sha256']
    for p, h in c['inputs_sha256'].items():
        assert sha(ROOT / p) == h
    ac = readj(AGE / 'contract.json'); fc = readj(ROOT / 'Modeling/tables/regime_followup/contract.json')
    sc = readj(ROOT / 'Modeling/tables/regime_forecast/contract.json')
    ar = readj(AGE / 'run.json')
    for n, h in ar['outputs_sha256'].items():
        assert sha(AGE / n) == h
    _, _, _, m = load_frame()
    # Reproduce the original feature artifact's CSV serialization boundary.
    m = pd.read_csv(io.StringIO(m.to_csv(index=False)), parse_dates=['timestamp'])
    z, events = extended(fc)
    ct = z.loc[z.eligible_onset & z.persistence_known & (z.label_confirmed_at < '2021-07-01')].copy()
    assert len(ct) == 4336
    features = ac['features']
    oo = []
    for month in [4, 5, 6]:
        boundary = pd.Timestamp(2021, month, 1)
        tr = ct.loc[ct.label_confirmed_at < boundary]
        te = ct.loc[ct.month == month]
        fit = model('logistic', sc); fit.fit(tr[features], tr.sustained_onset.eq(1).astype(int))
        pp = te[['timestamp','label_confirmed_at','sustained_onset']].copy()
        pp['score'] = fit.predict_proba(te[features])[:,1]; pp['event'] = te.sustained_onset.eq(1).astype(int)
        oo.append(pp)
    oof = pd.concat(oo)
    boundary, _, _, _ = threshold(oof.event.to_numpy(int), oof.score.to_numpy(float))
    close_cal = pd.read_csv(AGE / 'calibration.csv', encoding='utf-8-sig')
    assert np.isclose(boundary, float(close_cal.loc[close_cal.period == 'Jul-Aug', 'threshold'].iloc[0]), atol=1e-10)
    clf = model('logistic', sc); clf.fit(ct[features], ct.sustained_onset.eq(1).astype(int))
    late = z.loc[z.eligible_onset & z.month.ge(7)].copy()
    score = clf.predict_proba(late[features])[:,1]
    saved = pd.read_csv(AGE / 'followup_classification.csv', encoding='utf-8-sig', parse_dates=['timestamp'])
    saved = saved.loc[saved.method == 'noage'].set_index('timestamp').loc[late.timestamp]
    np.testing.assert_allclose(score, saved.score, rtol=1e-9, atol=1e-8)
    np.testing.assert_array_equal(score >= boundary, saved.alarm)
    base = fc['regression_features']['B0']; expanded = fc['regression_features']['B1']
    extras = [x for x in expanded if x not in base and not x.endswith('classifier_available')]
    zh = pd.read_csv(io.StringIO(z.loc[z.month.le(6)].to_csv(index=False)), parse_dates=['timestamp','label_confirmed_at'])
    zz = pd.concat([zh,z.loc[z.month.ge(7)]], ignore_index=True)
    common = zz.loc[zz.eligible_onset, ['timestamp','label_confirmed_at','month','profile','sustained_onset','persistence_known'] + extras]
    common = common.drop(columns='month').merge(m.loc[m.eligible_common, ['timestamp','target_maximum']+base], on='timestamp', validate='one_to_one')
    # Availability at the origin uses mature earlier labels, not target values.
    for direction, sign in [('up',1),('down',-1)]:
        availability = {}
        for month in range(2,9):
            boundary_month = pd.Timestamp(2021, min(month,7), 1)
            tr = ct.loc[ct.label_confirmed_at < boundary_month]
            positive = tr.loc[tr.sustained_onset == sign]
            availability[month] = int(len(positive) >= 8 and positive.date.nunique() >= 5)
        common[f'{direction}_classifier_available'] = common.month.map(availability)
    rt = common.loc[common.month.between(2,6) & common.persistence_known & (common.label_confirmed_at < '2021-07-01')]
    rt = pd.read_csv(io.StringIO(rt.to_csv(index=False)), parse_dates=['timestamp','label_confirmed_at'])
    test = common.loc[common.month.ge(7)].copy()
    assert len(rt) == 3598 and len(test) == 1344
    parent_train = pd.read_csv(ROOT / 'Modeling/tables/regime_integration/meta_features.csv', encoding='utf-8-sig', parse_dates=['timestamp']).set_index('timestamp').loc[rt.timestamp]
    np.testing.assert_allclose(rt[expanded], parent_train[expanded], atol=1e-9)
    differences = {n:float(np.max(np.abs(rt[n].to_numpy()-parent_train[n].to_numpy()))) for n in expanded if not np.array_equal(rt[n].to_numpy(),parent_train[n].to_numpy())}
    print(json.dumps({'remaining_training_feature_differences':differences}),flush=True)
    predictions = {}
    for variant, cols in [('B0',base),('B1_all',expanded)]:
        reg = HistGradientBoostingRegressor(**fc['hgb_parameters']); reg.fit(rt[cols], rt.target_maximum)
        predictions[variant] = reg.predict(test[cols])
    scores_index = pd.Series(score, index=late.timestamp)
    alarm = scores_index.loc[test.timestamp].to_numpy() >= boundary
    predictions['G1_noage'] = np.where(alarm, predictions['B1_all'], predictions['B0'])
    parent = pd.read_csv(AGE / 'followup_predictions.csv', encoding='utf-8-sig', parse_dates=['timestamp'])
    output = []
    for name, pred in predictions.items():
        old = parent.loc[parent.variant == name].set_index('timestamp').loc[test.timestamp]
        np.testing.assert_allclose(old.actual, test.target_maximum, atol=1e-9)
        np.testing.assert_allclose(old.prediction, pred, atol=1e-8, rtol=1e-9)
        part = test[['timestamp','sustained_onset']].copy()
        part['variant'] = name; part['actual'] = test.target_maximum; part['prediction'] = pred
        output.append(part)
    replay = pd.concat(output)
    # Direct linear decomposition in log-odds, checked against both decision and probability.
    scaler, linear = clf[0], clf[1]
    contrib = scaler.transform(late[features]) * linear.coef_[0]
    logit = float(linear.intercept_[0]) + contrib.sum(axis=1)
    np.testing.assert_allclose(logit, clf.decision_function(late[features]), atol=1e-10)
    np.testing.assert_allclose(1/(1+np.exp(-logit)), score, atol=1e-12)
    groups = {'calendar':[n for n in features if n in sc['feature_names'][13:]],
              'production':['prior_production','prior_production_delta'],
              'state':['prior_state_left_censored','prior_state_direction']}
    groups['power'] = [n for n in features if n not in sum(groups.values(), [])]
    assert set(sum(groups.values(), [])) == set(features) and sum(map(len, groups.values())) == len(features)
    explained = late[['timestamp','month','profile','prior_mean','prior_production','sustained_onset','persistence_known']].copy()
    explained['score'] = score; explained['alarm'] = (score >= boundary).astype(int)
    explained['intercept'] = float(linear.intercept_[0]); explained['logit'] = logit
    explained['common'] = explained.timestamp.isin(test.timestamp)
    for group, names in groups.items():
        explained[group+'_contribution'] = contrib[:,[features.index(n) for n in names]].sum(axis=1)
    coefficients = pd.DataFrame({'feature': features, 'standardized_coefficient': linear.coef_[0],
                                 'training_mean':scaler.mean_, 'training_scale':scaler.scale_,
                                 'mean_abs_late_contribution':np.abs(contrib).mean(axis=0)})
    e = explained.loc[explained.common & explained.persistence_known]
    masks = {'detected_up':e.sustained_onset.eq(1)&e.alarm.eq(1),
             'missed_up':e.sustained_onset.eq(1)&e.alarm.eq(0),
             'Monday08_non_event':e.timestamp.dt.dayofweek.eq(0)&e.timestamp.dt.hour.eq(8)&e.sustained_onset.ne(1)}
    summary=[]
    for name, mask in masks.items():
        gg=e.loc[mask]
        summary.append({'condition':name,'hours':len(gg),'mean_score':float(gg.score.mean()),
                        **{n:float(gg[n].mean()) for n in ['intercept','logit']+[k+'_contribution' for k in groups]}})
    events_table = e.loc[e.sustained_onset.eq(1)].copy()
    b0=parent.loc[parent.variant=='B0'].set_index('timestamp'); b1=parent.loc[parent.variant=='G1_noage'].set_index('timestamp')
    events_table['actual_maximum'] = b0.loc[events_table.timestamp,'actual'].to_numpy()
    events_table['base_prediction'] = b0.loc[events_table.timestamp,'prediction'].to_numpy()
    events_table['supplement_prediction'] = b1.loc[events_table.timestamp,'prediction'].to_numpy()
    events_table['absolute_error_gain'] = abs(events_table.actual_maximum-events_table.base_prediction)-abs(events_table.actual_maximum-events_table.supplement_prediction)
    cases=[]; selection=[]
    for category, pool, sortcol in [('detected_improved',events_table.loc[events_table.alarm.eq(1)&events_table.absolute_error_gain.gt(0)],'absolute_error_gain'),
                                    ('missed',events_table.loc[events_table.alarm.eq(0)],'actual_maximum')]:
        eligible=pool.loc[pool.timestamp.apply(lambda t: pd.date_range(t-pd.Timedelta(hours=4), t+pd.Timedelta(hours=5), freq='h').isin(b0.index).all())].sort_values([sortcol,'timestamp'])
        assert len(eligible)
        chosen=eligible.iloc[len(eligible)//2]; t=chosen.timestamp
        times=pd.date_range(t-pd.Timedelta(hours=4),t+pd.Timedelta(hours=5),freq='h')
        window=parent.loc[parent.timestamp.isin(times)&parent.variant.isin(['B0','G1_noage'])].copy()
        window['case']=category;window['onset_time']=t
        window['observed_mean']=z.set_index('timestamp').loc[window.timestamp,'level'].to_numpy()
        window['alarm']=explained.set_index('timestamp').loc[window.timestamp,'alarm'].to_numpy()
        cases.append(window)
        selection.append({'case':category,'timestamp':str(t),'candidate_events':len(pool),'complete_windows':len(eligible),'rank_zero_based':len(eligible)//2,'sort_column':sortcol,'actual':float(chosen.actual_maximum),'gain':float(chosen.absolute_error_gain)})
    outputs={'reproduced_predictions.csv':replay,'logit_contributions.csv':explained,'coefficient_table.csv':coefficients,
             'contribution_summary.csv':pd.DataFrame(summary),'event_explanations.csv':events_table,'case_windows.csv':pd.concat(cases),
             'case_selection.csv':pd.DataFrame(selection)}
    for name, frame in outputs.items():
        frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')
    result={'status':'passed','script_sha256':sha(__file__),'contract_sha256':sha(OUT/'contract.json'),
            'inputs_sha256':c['inputs_sha256'],'outputs_sha256':{n:sha(OUT/n) for n in outputs},
            'raw_valid_hours':len(m),'class_train':len(ct),'regression_train':len(rt),'classification_predictions':len(late),
            'regression_predictions_checked':len(replay),'refits':6,'new_candidates':0,
            'oof_threshold':float(boundary),'exact_logit_reconstruction':True,'clean_environment':False,
            'runtime':sys.version,'scope':'Current fixed-method raw-to-features/training/OOF-threshold/inference replay; historical Optuna search not rerun.'}
    assert sha(ROOT/'README.md')==c['root_readme_sha256']
    save_json(OUT/'run.json',result)
    print(json.dumps({'reproduction':result,'contributions':summary,'cases':selection}),flush=True)


if __name__=='__main__':
    assert sys.version_info[:2]==(3,13) and sys._is_gil_enabled()
    {'freeze':freeze,'run':run}[sys.argv[1]]()
