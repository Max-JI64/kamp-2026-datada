"""Independent raw input, fixed split, saved model, metrics and decision verification."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from m03_verify import rows, sha, close, verify_metric
from m04_rise_verify import ap

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'Modeling/tables/m05'


def state(maximum, rounded):
    return 'zero' if maximum == 0 else 'below20' if rounded < 20 else 'low' if rounded <= 26 else 'above26'


def subset(records, condition, value):
    if condition in ['all', 'profile_reweighted']:
        return records
    if condition == 'daily_maximum':
        return [r for r in records if float(r['daily_maximum_weight'])>0]
    if condition in ['month', 'power_transition', 'hour', 'train_profile_overlap']:
        return [r for r in records if r[condition] == value]
    rules = {
        'surge_q95': lambda r: float(r['increase'])>=93,
        'surge_q90': lambda r: float(r['increase'])>=61.60000000000002,
        'legacy_rise': lambda r: r['rise_event']=='True',
        'actual_zero': lambda r: float(r['actual'])==0,
        'prior_zero': lambda r: float(r['lag1_maximum'])==0,
        'restart_after_zero': lambda r: float(r['lag1_maximum'])==0 and float(r['actual'])>0,
        'sharp_drop': lambda r: float(r['increase'])<=-93,
    }
    return [r for r in records if rules[condition](r)]


def main():
    assert sys.version_info[:2] == (3,13) and sys._is_gil_enabled()
    run = json.loads((OUT / 'run.json').read_text(encoding='utf-8'))
    cpath = ROOT / 'Modeling/config/m05_contract.json'
    c = json.loads(cpath.read_text(encoding='utf-8'))
    assert run['status']=='completed' and run['fits']==4 and run['new_optuna_trials']==0
    assert sha(cpath)==run['contract_sha256'] and sha(ROOT / 'Modeling/scripts/m05_evaluate.py')==run['script_sha256']
    for name, digest in run['outputs_sha256'].items():
        assert sha(OUT / name)==digest, name
    for name, digest in run['inputs_sha256'].items():
        assert sha(ROOT / name)==digest, name
    for name, digest in run['helper_sha256'].items():
        assert sha(ROOT / 'Modeling/scripts' / name)==digest, name
    raw = {}
    for r in rows(ROOT / c['source']):
        date, hour = int(r['날짜']), float(r['시간'])
        if 20210101<=date<=20210831 and hour.is_integer() and 0<=hour<=23:
            ts = datetime.strptime(str(date),'%Y%m%d')+timedelta(hours=int(hour))
            slots = [float(r[k]) for k in ['15분','30분','45분','60분']]
            assert ts not in raw
            raw[ts] = {'slots': slots, 'maximum': max(slots), 'mean': math.fsum(slots)/4,
                'last': slots[-1], 'production': float(r['생산량']), 'rounded': float(r['평균'])}
    assert len(raw)==5784
    dates = defaultdict(list)
    for ts in sorted(raw):
        dates[ts.date()].extend(raw[ts]['slots'])
    import hashlib
    profiles = {day: hashlib.sha256(','.join(format(x,'.17g') for x in slots).encode()).hexdigest() for day,slots in dates.items()}
    eligible = {pool: [ts for ts in sorted(raw) if datetime(2021,7,1)<=ts<datetime(2021,9,1)
        and all(ts-timedelta(hours=lag) in raw for lag in lags)] for pool,lags in [('common',[1,2,24,168]),('core',[1,2])]}
    training = [ts for ts in sorted(raw) if ts<datetime(2021,7,1) and all(ts-timedelta(hours=lag) in raw for lag in [1,2,24,168])]
    assert (len(training),len(eligible['common']),len(eligible['core']))==(4176,1344,1436)
    expected_inputs = {}
    for r in rows(OUT / 'feature_frame.csv'):
        ts = datetime.fromisoformat(r['timestamp'])
        assert ts in training if r['role']=='train' else ts in eligible['common']
        own, prev, prev2 = raw[ts], raw[ts-timedelta(hours=1)], raw[ts-timedelta(hours=2)]
        vals = {'hour_sin': math.sin(2*math.pi*ts.hour/24), 'hour_cos': math.cos(2*math.pi*ts.hour/24),
            'month': ts.month, 'weekend': int(ts.weekday()>=5), 'lag1_production_zero': int(prev['production']==0),
            'prev_slot_rise': prev['slots'][-1]-prev['slots'][0], 'prev_slot_range': max(prev['slots'])-min(prev['slots']),
            'prev_slot_slope': math.fsum(x*w for x,w in zip(prev['slots'],[-1.5,-.5,.5,1.5]))/5}
        for day in range(7):
            vals[f'dow_{day}']=int(ts.weekday()==day)
        for stem in ['last','mean','maximum','production']:
            vals['lag1_'+stem]=prev[stem]
            vals['lag2_'+stem]=prev2[stem]
            vals[stem+'_delta']=prev[stem]-prev2[stem]
        length,gap=0,0
        for lag in range(1,7):
            old=raw.get(ts-timedelta(hours=lag))
            if old is None:
                gap=1
                break
            if state(old['maximum'],old['rounded'])!='low':
                break
            length+=1
        vals['low_run_length6'],vals['low_history_gap6']=length,gap
        for col in run['features']['dynamic']:
            close(r[col],vals[col])
        expected_inputs[ts]=vals
    assert len(expected_inputs)==5520
    predictions=rows(OUT / 'predictions.csv')
    groups=defaultdict(list)
    keyed={}
    train_profiles={profiles[ts.date()] for ts in training}
    for r in predictions:
        ts=datetime.fromisoformat(r['timestamp'])
        pool,group=r['pool'],r['group']
        assert ts in eligible[pool]
        own,prev=raw[ts],raw[ts-timedelta(hours=1)]
        close(r['actual'],own['maximum'])
        close(r['lag1_maximum'],prev['maximum'])
        close(r['increase'],own['maximum']-prev['maximum'])
        transition=state(prev['maximum'],math.floor(prev['mean']+.5))+'->'+state(own['maximum'],own['rounded'])
        assert r['power_transition']==transition
        assert (r['prior_low']=='True')==transition.startswith('low->')
        assert (r['rise_event']=='True')==(transition=='low->above26')
        assert r['diag_target_profile']==profiles[ts.date()]
        assert (r['train_profile_overlap']=='True')==(profiles[ts.date()] in train_profiles)
        pool_dates={x.date() for x in eligible[pool]}
        count=sum(profiles[day]==profiles[ts.date()] for day in pool_dates)
        close(r['profile_weight'],1/count)
        day_ts=[x for x in eligible[pool] if x.date()==ts.date()]
        maximum=max(raw[x]['maximum'] for x in day_ts)
        ties=[x for x in day_ts if raw[x]['maximum']==maximum]
        close(r['daily_maximum_weight'],1/len(ties) if len(day_ts)==24 and ts in ties else 0)
        error=float(r['prediction'])-own['maximum']
        for col,value in [('signed_error',error),('absolute_error',abs(error)),('under_amount',max(-error,0)),('over_amount',max(error,0))]:
            close(r[col],value)
        if group.startswith('lag'):
            close(r['prediction'],raw[ts-timedelta(hours=int(group[3:]))]['maximum'])
        key=(pool,group,ts)
        assert key not in keyed
        keyed[key]=r
        groups[pool,group].append(r)
    assert len(predictions)==6*1344+2*1436
    for (pool,group),part in groups.items():
        assert [datetime.fromisoformat(r['timestamp']) for r in part]==eligible[pool]
    for r in rows(OUT / 'metrics.csv'):
        part=groups[r['pool'],r['group']]
        if r['period']!='pooled':
            part=[x for x in part if x['month']==r['period']]
        chosen=subset(part,r['condition'],r['value'])
        verify_metric(r,chosen,r['condition'] if r['condition'] in ['daily_maximum','profile_reweighted'] else 'all')
    for r in rows(OUT / 'daily_errors.csv'):
        verify_metric(r,[x for x in groups[r['pool'],r['group']] if x['date']==r['date']])
    for r in rows(OUT / 'paired_predictions.csv'):
        ts=datetime.fromisoformat(r['timestamp'])
        a,b=keyed[r['pool'],'A',ts],keyed[r['pool'],'B',ts]
        for col in ['prediction','absolute_error','under_amount','over_amount']:
            close(r[col+'_A'],a[col]);close(r[col+'_B'],b[col])
            if col!='prediction':
                close(r['delta_'+col],float(b[col])-float(a[col]))
    calibration_source=rows(ROOT / 'Modeling/tables/m04_surge_risk/predictions.csv')
    for r in rows(OUT / 'alarm_calibration.csv'):
        past=[x for x in calibration_source if x['group']==r['group']]
        assert len(past)==2184 and max(x['timestamp'] for x in past)<'2021-07-01'
        negative=sorted([float(x['score']) for x in past if float(x['increase'])<93],reverse=True)
        budget=math.floor(float(r['cap'])*len(negative))
        close(r['threshold'],math.nextafter(negative[budget],math.inf))
        close(r['budget'],budget);close(r['calibration_FP'],sum(x>=float(r['threshold']) for x in negative))
    risk=rows(OUT / 'risk_predictions.csv')
    for r in risk:
        ts=datetime.fromisoformat(r['timestamp'])
        source=keyed['common','B' if r['group']=='B_score' else 'G_aux',ts]
        close(r['score'],float(source['prediction'])-float(source['lag1_maximum']))
        close(r['normal_prediction'],keyed['common','B',ts]['prediction'])
        for cap in [.01,.05]:
            threshold=next(x['threshold'] for x in c['alarms']['thresholds'] if x['group']==r['group'] and x['cap']==cap)
            assert (r['alarm_'+str(cap)]=='True')==(float(r['score'])>=threshold)
    for r in rows(OUT / 'risk_metrics.csv'):
        part=[x for x in risk if x['group']==r['group'] and (r['period']=='pooled' or x['month']==r['period'])]
        boundary,threshold=float(r['boundary']),float(r['threshold'])
        y=[float(x['increase'])>=boundary for x in part]
        scores=[float(x['score']) for x in part]
        alarms=[x>=threshold for x in scores]
        events=sum(y);tp=sum(a and b for a,b in zip(alarms,y));fp=sum(a and not b for a,b in zip(alarms,y))
        expected={'hours':len(part),'events':events,'TP':tp,'FP':fp,'FN':events-tp,'TN':len(part)-events-fp,
            'precision':tp/(tp+fp) if tp+fp else 0,'false_positive_fraction':fp/(len(part)-events),
            'prevalence':events/len(part),'event_dates':len({x['date'] for x in part if float(x['increase'])>=boundary})}
        if events:
            expected.update(recall=tp/events,AP=ap(y,scores))
        else:
            assert r['AP']==r['recall']==''
        if 0<events<len(y):
            positive=[score for label,score in zip(y,scores) if label]
            negative=[score for label,score in zip(y,scores) if not label]
            expected['AUROC']=sum((a>b)+.5*(a==b) for a in positive for b in negative)/(len(positive)*len(negative))
        for col,value in expected.items():
            close(r[col],value)
    models={}
    m1=pd.read_csv(ROOT / 'Modeling/tables/m01/hourly_frame.csv',encoding='utf-8-sig',float_precision='round_trip',parse_dates=['timestamp']).set_index('timestamp')
    features=pd.read_csv(OUT / 'feature_frame.csv',encoding='utf-8-sig',float_precision='round_trip',parse_dates=['timestamp']).set_index('timestamp')
    for r in rows(OUT / 'model_manifest.csv'):
        path=ROOT / r['file']
        assert sha(path)==r['sha256']
        model=joblib.load(path);name=r['name'];models[name]=model
        cols=run['features'][name] if name in ['A','B'] else run['features']['dynamic']
        assert list(model.feature_names_in_)==cols
        parameters=c['hgb_parameters'] if name!='classifier' else {**c['G']['classifier_parameters'],'max_iter':150,'learning_rate':.05,'early_stopping':False,'random_state':42}
        assert all(model.get_params()[key]==value for key,value in parameters.items())
        if name in ['A','B']:
            for pool in ['common','core']:
                actual=model.predict(m1.loc[eligible[pool],cols])
                assert np.allclose(actual,[float(x['prediction']) for x in groups[pool,name]],rtol=0,atol=1e-10)
    components=rows(OUT / 'G_components.csv')
    expert=models['expert'].predict(features.loc[eligible['common'],run['features']['dynamic']])
    probabilities=models['classifier'].predict_proba(features.loc[eligible['common'],run['features']['dynamic']])[:,1]
    for r,w,prob in zip(components,expert,probabilities):
        ts=datetime.fromisoformat(r['timestamp']);b=float(keyed['common','B',ts]['prediction'])
        trigger=r['prior_low']=='True' and prob>=c['G']['gate']['threshold']
        g=b+(c['G']['gate']['blend']*(w-b) if trigger else 0)
        close(r['B'],b);close(r['expert'],w);close(r['probability'],prob);close(r['G_aux'],g)
        assert (r['trigger']=='True')==trigger
        close(keyed['common','G_aux',ts]['prediction'],g)
    audits=rows(OUT / 'training_audit.csv')
    assert len(audits)==4 and sum(int(x['fit_count']) for x in audits)==4
    for r in audits:
        sample=[ts for ts in training if state(raw[ts-timedelta(hours=1)]['maximum'],math.floor(raw[ts-timedelta(hours=1)]['mean']+.5))=='low'] if r['name']=='classifier' else training
        assert r['training_start']==str(min(sample)) and r['training_end']==str(max(sample))
        assert max(sample)<datetime(2021,7,1)
        close(r['train_hours'],len(sample))
    # Rebuild the decision flags without importing the execution evaluator.
    metric_records=rows(OUT / 'metrics.csv')
    def value(group,period='pooled',condition='all',key='MAE'):
        return float(next(r[key] for r in metric_records if r['pool']=='common' and r['group']==group and r['period']==period and r['condition']==condition and r['value']=='all'))
    normal={'overall':value('B')<=.99*value('A'),'daily_maximum':value('B',condition='daily_maximum')<=.99*value('A',condition='daily_maximum'),
        'peak_under':value('B',condition='daily_maximum',key='mean_under')<value('A',condition='daily_maximum',key='mean_under'),
        'lag1_overall':value('B')<value('lag1'),'lag1_peak':value('B',condition='daily_maximum')<value('lag1',condition='daily_maximum'),
        'monthly':all(value('B',str(m))<=1.01*value('A',str(m)) and value('B',str(m),'daily_maximum')<=1.01*value('A',str(m),'daily_maximum') for m in [7,8]),
        'profile_reweighted':value('B',condition='profile_reweighted')<=value('A',condition='profile_reweighted')}
    rmetrics=rows(OUT / 'risk_metrics.csv')
    weighted={}
    for group in ['B_score','G_HGB_score']:
        months=[r for r in rmetrics if r['group']==group and r['cap']=='0.01' and r['boundary']=='93.0' and r['period']!='pooled']
        events=sum(int(r['events']) for r in months)
        weighted[group]=sum(float(r['AP'] or 0)*int(r['events']) for r in months)/events if events else None
    pooled={r['group']:r for r in rmetrics if r['cap']=='0.01' and r['boundary']=='93.0' and r['period']=='pooled'}
    b,g=pooled['B_score'],pooled['G_HGB_score']
    def month_tp(group,m):
        return int(next(r['TP'] for r in rmetrics if r['group']==group and r['cap']=='0.01' and r['boundary']=='93.0' and r['period']==str(m)))
    gates={'rank':weighted['G_HGB_score']>=1.05*weighted['B_score'],'pooled_recall':int(g['TP'])>=int(b['TP']),
        'monthly_recall':all(month_tp('G_HGB_score',m)>=month_tp('B_score',m) for m in [7,8]),
        'false_positive':float(g['false_positive_fraction'])<=.01,
        'strict_gain':int(g['TP'])>0 and (int(g['TP'])>int(b['TP']) or int(g['FP'])<int(b['FP']))} if int(b['events']) else {'assessable':False}
    conclusion=json.loads((OUT / 'conclusion.json').read_text(encoding='utf-8'))
    assert conclusion['normal_confirmation_gates']==normal and conclusion['broad_last_slot_effect_confirmed']==all(normal.values())
    assert conclusion['risk_confirmation_gates']==gates and conclusion['G_risk_effect_confirmed']==all(gates.values())
    for r in rows(OUT / 'risk_confirmation.csv'):
        if weighted[r['group']] is not None:
            close(r['weighted_monthly_AP'],weighted[r['group']])
    events=[]
    for r in risk:
        if r['group']=='B_score' and float(r['increase'])>=93:
            other=next(x for x in risk if x['timestamp']==r['timestamp'] and x['group']=='G_HGB_score')
            events.append({'timestamp':r['timestamp'],'actual':float(r['actual']),'increase':float(r['increase']),
                'B_score':float(r['score']),'G_score':float(other['score']),
                'B_alarm':r['alarm_0.01']=='True','G_alarm':other['alarm_0.01']=='True',
                'prior_low':r['prior_low']=='True'})
    diagnostic={'events':events,'overall_reduction_percent':100*(1-value('B')/value('A')),
        'peak_MAE_change_percent':100*(value('B',condition='daily_maximum')/value('A',condition='daily_maximum')-1),
        'baseline_reduction_percent':100*(1-value('B')/value('lag1')),
        'no_test_based_change':True}
    (OUT / 'confirmed_diagnostics.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf-8')
    result={'status':'passed','checked_at':datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='minutes'),
        'run_sha256':sha(OUT / 'run.json'),'script_sha256':sha(Path(__file__)),
        'raw_feature_cells_checked':5520*27,'prediction_rows_checked':len(predictions),'models_reloaded':4,
        'metrics_rows_checked':len(metric_records),'alarm_rows_checked':len(rmetrics),
        'diagnostics_sha256':sha(OUT / 'confirmed_diagnostics.json'),'confirmation_flags_verified':True,'no_fitting':True}
    (OUT / 'independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
