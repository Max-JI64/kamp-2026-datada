"""Approved extension: chronological onset discrimination and power forecasting.
No future production features or in-sample stacking probabilities.
"""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import hashlib
import json
import csv
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve

ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/'Analysis/tables/a02_prior_power_signals/observations.csv'
OUT=ROOT/'Analysis/tables/a02_transition_forecast'
SETS={'calendar':[], 'level':['past_level'], 'shape':['past_level','prior_slot_trend']}
FAMILIES=['logistic','hgb']


def write_json(name,value):
    (OUT/f'{name}.json').write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def load():
    manifest=json.loads((INPUT.parent/'verification.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(INPUT.read_bytes()).hexdigest()==manifest['outputs_sha256']['observations.csv']
    part=pd.read_csv(INPUT,encoding='utf-8-sig',parse_dates=['timestamp','input_timestamp','older_timestamp','date'])
    assert len(part)==5778
    assert (part.timestamp-part.input_timestamp).eq(pd.Timedelta(hours=1)).all()
    assert (part.input_timestamp-part.older_timestamp).eq(pd.Timedelta(hours=1)).all()
    part['onset']=part.transition.eq('zero_to_positive').astype(int)
    assert (part.loc[part.onset.eq(1),'past_production']==0).all()
    part['known_zero']=part.past_production.eq(0).astype(int)
    part['past_production_log']=np.log1p(part.past_production)
    return part


def matrix(frame,features):
    # Fixed calendar categories, no fitted target-derived transformations.
    hours=np.eye(24)[frame['시간'].to_numpy(dtype=int)]
    return np.column_stack([hours,frame.weekend,frame.month]+[frame[f] for f in features])


def classifier(family):
    if family=='logistic':
        return make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=3203))
    return HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,min_samples_leaf=20,
        l2_regularization=1,learning_rate=.05,early_stopping=False,random_state=3203)


def threshold(y,prob):
    precision,recall,cuts=precision_recall_curve(y,prob)
    f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    best=np.flatnonzero(np.isclose(f1,f1.max()))[-1]
    return float(cuts[best])


def classify_metrics(y,prob,cut):
    y=np.asarray(y,dtype=int)
    pred=np.asarray(prob)>=cut
    tp=int((pred & (y==1)).sum()); fp=int((pred & (y==0)).sum()); fn=int((~pred & (y==1)).sum())
    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp+fn) if tp+fn else 0
    return dict(n=len(y),events=int(y.sum()),prevalence=float(y.mean()),
        ap=float(average_precision_score(y,prob)),roc_auc=float(roc_auc_score(y,prob)) if len(np.unique(y))>1 else None,
        brier=float(brier_score_loss(y,prob)),threshold=cut,tp=tp,fp=fp,fn=fn,
        precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0)


def class_validation(part):
    risk=part.loc[part.known_zero.eq(1)].copy()
    train=risk.loc[risk.month.le(4)]
    valid=risk.loc[risk.month.between(5,6)]
    rows=[]; predictions=valid[['timestamp','date','month','onset','profile_weight']].copy()
    for family in FAMILIES:
        for group,features in SETS.items():
            model=classifier(family)
            model.fit(matrix(train,features),train.onset)
            prob=model.predict_proba(matrix(valid,features))[:,1]
            cut=threshold(valid.onset,prob)
            rows.append(dict(family=family,features=group,**classify_metrics(valid.onset,prob,cut)))
            predictions[f'{family}_{group}']=prob
    results=pd.DataFrame(rows)
    results.to_csv(OUT/'classification_validation.csv',index=False,encoding='utf-8-sig')
    predictions.to_csv(OUT/'classification_validation_predictions.csv',index=False,encoding='utf-8-sig')
    # Select a common family using richer-input validation AP, then keep family
    # constant in the ablation. No July/August outcome consulted here.
    best=results.loc[results.features.eq('shape')].sort_values('ap',ascending=False).iloc[0]
    chosen=results.loc[results.family.eq(best.family)]
    best_ap=chosen.ap.max()
    selected=next(name for name in SETS if float(chosen.loc[chosen.features.eq(name),'ap'].iloc[0])>=best_ap-.01)
    baseline=chosen.loc[chosen.features.eq('calendar')].iloc[0]
    selected_row=chosen.loc[chosen.features.eq(selected)].iloc[0]
    gate=selected!='calendar' and selected_row.ap>baseline.ap and selected_row.brier<=1.05*baseline.brier
    frozen=dict(family=str(best.family),selected_features=selected,power_gate=bool(gate),
        reason='Validation AP within0.01 of best favors fewer inputs; same classifier family for ablations. Power followup requires AP above calendar and Brier no worse than5%.',
        thresholds={name:float(chosen.loc[chosen.features.eq(name),'threshold'].iloc[0]) for name in SETS},
        validated_before_test=True,test_consulted=False,
        classifier_parameters='Fixed logistic C1 or HGB150iterations/7leaves/min20/l2=1/lr.05; no tuning grid.',
        train_n=len(train),train_events=int(train.onset.sum()),validation_n=len(valid),validation_events=int(valid.onset.sum()))
    write_json('frozen',frozen)
    print(results.to_json(orient='records')); print(json.dumps(frozen))


def probability_history(part,frozen):
    risk=part.loc[part.known_zero.eq(1)]
    probabilities=pd.Series(0.,index=part.index)
    audit=[]
    features=SETS[frozen['selected_features']]
    for month in range(3,9):
        # March-June training meta-features see strictly earlier months.
        # July/August both use a final classifier fitted through June.
        cutoff=month-1 if month<=6 else 6
        train=risk.loc[risk.month.le(cutoff)]
        target=risk.loc[risk.month.eq(month)]
        model=classifier(frozen['family'])
        model.fit(matrix(train,features),train.onset)
        probabilities.loc[target.index]=model.predict_proba(matrix(target,features))[:,1]
        assert train.timestamp.max()<target.timestamp.min()
        audit.append(dict(target_month=month,train_last_timestamp=str(train.timestamp.max()),
            target_first_timestamp=str(target.timestamp.min()),train_n=len(train),train_events=int(train.onset.sum()),target_n=len(target)))
    pd.DataFrame(audit).to_csv(OUT/'probability_time_audit.csv',index=False,encoding='utf-8-sig')
    return probabilities


def regression_features():
    base=['past_level','older_level','past_production_log','known_zero']
    return {'baseline':base,'direct_shape':base+['prior_slot_trend'],
            'transition_probability':base+['prior_slot_trend','onset_probability']}


def regressor():
    return HistGradientBoostingRegressor(loss='squared_error',max_iter=200,max_leaf_nodes=15,
        min_samples_leaf=20,l2_regularization=1,learning_rate=.05,early_stopping=False,random_state=3203)


def soft_mixture(train,target):
    # Actual target production labels are used to train experts only; branch
    # selection at inference uses known past production and a past-only score.
    features=regression_features()['direct_shape']
    general=regressor(); general.fit(matrix(train,features),train.level)
    pred=general.predict(matrix(target,features))
    knownzero=target.known_zero.eq(1).to_numpy()
    experts=[]
    for event in (0,1):
        sample=train.loc[train.known_zero.eq(1)&train.onset.eq(event)]
        assert len(sample)>=20
        model=regressor(); model.fit(matrix(sample,features),sample.level)
        experts.append(model.predict(matrix(target.loc[knownzero],features)))
    p=target.loc[knownzero,'onset_probability'].to_numpy()
    pred[knownzero]=(1-p)*experts[0]+p*experts[1]
    return pred


def mixture_validation(part,frozen):
    assert not (OUT/'classification_test.csv').exists()
    assert frozen['power_gate']
    part=part.copy(); part['onset_probability']=probability_history(part,frozen)
    train=part.loc[part.month.between(3,4)]; valid=part.loc[part.month.between(5,6)]
    pred=soft_mixture(train,valid)
    predictions=pd.read_csv(OUT/'power_validation_predictions.csv',encoding='utf-8-sig')
    assert list(pd.to_datetime(predictions.timestamp))==list(valid.timestamp)
    predictions['soft_mixture']=pred
    predictions.to_csv(OUT/'power_validation_predictions.csv',index=False,encoding='utf-8-sig')
    results=pd.read_csv(OUT/'power_validation.csv',encoding='utf-8-sig')
    rows=[]
    for scope,keep in [('all',np.ones(len(valid),dtype=bool)),('onset',valid.onset.eq(1).to_numpy()),('known_zero',valid.known_zero.eq(1).to_numpy())]:
        rows.append(dict(model='soft_mixture',scope=scope,**power_metrics(valid.loc[keep],pred[keep])))
    results=pd.concat([results,pd.DataFrame(rows)],ignore_index=True)
    results.to_csv(OUT/'power_validation.csv',index=False,encoding='utf-8-sig')
    freeze=json.loads((OUT/'power_frozen.json').read_text(encoding='utf-8'))
    freeze['models'].append('soft_mixture')
    baseline=results.loc[results.model.eq('baseline')].set_index('scope')
    mixture=results.loc[results.model.eq('soft_mixture')].set_index('scope')
    freeze['validation_mixture_support']=bool(mixture.loc['all','mae']<baseline.loc['all','mae'] and mixture.loc['onset','mae']<baseline.loc['onset','mae'] and mixture.loc['onset','mean_under_amount']<baseline.loc['onset','mean_under_amount'])
    freeze['mixture_rationale']='Validation score-as-feature worsened onsetMAE9.95vs8.51; one minimum structural change: zero-continuing/onset regression experts mixed by prior-only probabilities. Same fixed regressor; no further search.'
    freeze['report_rule']='Report all four fixed models, including worsening; no changes after July/August evaluation.'
    write_json('power_frozen',freeze)
    print(pd.DataFrame(rows).to_json(orient='records')); print(json.dumps(freeze))


def power_metrics(frame,pred):
    error=pred-frame.level.to_numpy()
    w=frame.profile_weight.to_numpy()
    return dict(n=len(frame),mae=float(np.abs(error).mean()),rmse=float(np.sqrt(np.mean(error**2))),
        mean_error=float(error.mean()),mean_under_amount=float(np.maximum(-error,0).mean()),
        under_rate=float((error<0).mean()),profile_weighted_mae=float(np.average(np.abs(error),weights=w)))


def power_validation(part,frozen):
    if not frozen['power_gate']:
        write_json('power_frozen',dict(run=False,reason='Classifier validation gate not met; no favorable-result tuning.'))
        return
    part=part.copy(); part['onset_probability']=probability_history(part,frozen)
    train=part.loc[part.month.between(3,4)]
    valid=part.loc[part.month.between(5,6)]
    rows=[]; predictions=valid[['timestamp','date','month','level','onset','known_zero','profile_weight','onset_probability']].copy()
    for name,features in regression_features().items():
        model=regressor(); model.fit(matrix(train,features),train.level)
        pred=model.predict(matrix(valid,features)); predictions[name]=pred
        for scope,keep in [('all',np.ones(len(valid),dtype=bool)),('onset',valid.onset.eq(1).to_numpy()),('known_zero',valid.known_zero.eq(1).to_numpy())]:
            rows.append(dict(model=name,scope=scope,**power_metrics(valid.loc[keep],pred[keep])))
    results=pd.DataFrame(rows)
    results.to_csv(OUT/'power_validation.csv',index=False,encoding='utf-8-sig')
    predictions.to_csv(OUT/'power_validation_predictions.csv',index=False,encoding='utf-8-sig')
    # Freeze all three comparisons, including an unhelpful probability addition.
    write_json('power_frozen',dict(run=True,models=list(regression_features()),
        final_training_months=[3,4,5,6],final_test_months=[7,8],test_consulted=False,
        reason='Report all fixed ablations; no model or feature changes after final test. All probability training features were prior-month predictions.'))
    print(results.to_json(orient='records'))


def final_evaluation(part,frozen):
    risk=part.loc[part.known_zero.eq(1)]
    train=risk.loc[risk.month.le(6)]; test=risk.loc[risk.month.ge(7)]
    rows=[]; predictions=test[['timestamp','date','month','onset','profile_weight']].copy()
    for group,features in SETS.items():
        model=classifier(frozen['family']); model.fit(matrix(train,features),train.onset)
        prob=model.predict_proba(matrix(test,features))[:,1]
        predictions[group]=prob
        for scope,keep in [('all',np.ones(len(test),dtype=bool)),('month7',test.month.eq(7).to_numpy()),('month8',test.month.eq(8).to_numpy())]:
            rows.append(dict(features=group,scope=scope,**classify_metrics(test.loc[keep,'onset'],prob[keep],frozen['thresholds'][group])))
    pd.DataFrame(rows).to_csv(OUT/'classification_test.csv',index=False,encoding='utf-8-sig')
    predictions.to_csv(OUT/'classification_test_predictions.csv',index=False,encoding='utf-8-sig')
    print(pd.DataFrame(rows).to_json(orient='records'))
    power_frozen=json.loads((OUT/'power_frozen.json').read_text(encoding='utf-8'))
    if power_frozen['run']:
        part=part.copy(); part['onset_probability']=probability_history(part,frozen)
        train=part.loc[part.month.between(3,6)]; test=part.loc[part.month.ge(7)]
        predictions=test[['timestamp','date','month','level','onset','known_zero','profile_weight','onset_probability']].copy()
        rows=[]
        for name,features in regression_features().items():
            model=regressor(); model.fit(matrix(train,features),train.level)
            pred=model.predict(matrix(test,features)); predictions[name]=pred
            for scope,keep in [('all',np.ones(len(test),dtype=bool)),('onset',test.onset.eq(1).to_numpy()),
                ('known_zero',test.known_zero.eq(1).to_numpy()),('month7',test.month.eq(7).to_numpy()),('month8',test.month.eq(8).to_numpy())]:
                rows.append(dict(model=name,scope=scope,**power_metrics(test.loc[keep],pred[keep])))
        if 'soft_mixture' in power_frozen['models']:
            pred=soft_mixture(train,test); predictions['soft_mixture']=pred
            for scope,keep in [('all',np.ones(len(test),dtype=bool)),('onset',test.onset.eq(1).to_numpy()),
                ('known_zero',test.known_zero.eq(1).to_numpy()),('month7',test.month.eq(7).to_numpy()),('month8',test.month.eq(8).to_numpy())]:
                rows.append(dict(model='soft_mixture',scope=scope,**power_metrics(test.loc[keep],pred[keep])))
        pd.DataFrame(rows).to_csv(OUT/'power_test.csv',index=False,encoding='utf-8-sig')
        predictions.to_csv(OUT/'power_test_predictions.csv',index=False,encoding='utf-8-sig')
        print(pd.DataFrame(rows).to_json(orient='records'))
    verify(part)


def verify(part):
    with (OUT/'classification_test_predictions.csv').open(encoding='utf-8-sig',newline='') as stream:
        records=list(csv.DictReader(stream))
    for group in SETS:
        expected=sum((float(row[group])-int(row['onset']))**2 for row in records)/len(records)
        scores=pd.read_csv(OUT/'classification_test.csv',encoding='utf-8-sig')
        got=scores.loc[scores.features.eq(group)&scores.scope.eq('all'),'brier'].iloc[0]
        np.testing.assert_allclose(expected,got,atol=1e-12)
    if (OUT/'power_test_predictions.csv').exists():
        with (OUT/'power_test_predictions.csv').open(encoding='utf-8-sig',newline='') as stream:
            records=list(csv.DictReader(stream))
        scores=pd.read_csv(OUT/'power_test.csv',encoding='utf-8-sig')
        models=list(regression_features())+(['soft_mixture'] if 'soft_mixture' in records[0] else [])
        for model in models:
            for scope in ('all','onset'):
                rows=[row for row in records if scope=='all' or int(row['onset'])==1]
                mae=sum(abs(float(row[model])-float(row['level'])) for row in rows)/len(rows)
                got=scores.loc[scores.model.eq(model)&scores.scope.eq(scope),'mae'].iloc[0]
                np.testing.assert_allclose(mae,got,atol=1e-12)
    write_json('verification',dict(status='passed',input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        input_timestamps_checked=len(part),n=len(part),target_production_not_a_feature=True,
        independent_csv_brier_and_mae=True,probability_training_strictly_past=True,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outputs_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.csv')}))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['validation','mixture','final']); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    part=load()
    if args.stage=='validation':
        write_json('contract',dict(previous='A02 within-onset correlation.013 does not test onset/no-onset separation; matched prior means81.85 vs39.57.',
            question='Can past information discriminate next-hour production onset among known-zero production hours, then improve next mean-power forecasts?',
            source='Current verified A02 observations only, no backup or previous models.',
            split='Class1-4train/5-6validation/7-8final chronological retrospective evaluation; all months were previously explored in EDA.',
            class_models='Fixed logistic and fixed shallow HGB; calendar -> observed mean -> observed slot trend; select common family on validation shapeAP.',
            class_selection='Highest validation AP within.01 favors simpler features; threshold maximizes validation F1, high threshold breaks ties.',
            probability_gate='Selected noncalendar AP better than calendar and Brier<=1.05calendar on validation.',
            power_models='Fixed HGB baseline past means/calendar/known past production; add direct slot trend; then add soft onset probability.',
            power_probability='March-June probabilities from strictly earlier month data; July/August probabilities from January-June only; positives set0 plus explicit known_zero feature.',
            power_split='3-4train/5-6validation; final3-6train/7-8test; same records across three fixed models.',
            power_metrics='All-hour and actual onset MAE plus average underprediction amount (including0 for overprediction), monthly and profile-weighted sensitivity.',
            no_target_inputs=True,no_peak_threshold=True,no_manuscript=True,stop='After final evaluation no feature/model retuning for favorable scores.'))
        class_validation(part)
        frozen=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'))
        power_validation(part,frozen)
    elif args.stage=='mixture':
        frozen=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'))
        mixture_validation(part,frozen)
    else:
        assert (OUT/'frozen.json').exists() and (OUT/'power_frozen.json').exists()
        frozen=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'))
        final_evaluation(part,frozen)


if __name__=='__main__':
    main()
