"""One-time inference with saved models for an explicitly retrospective full-period demo."""
from __future__ import annotations
import io
import hashlib
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Modeling/scripts"))
from regime_diagnose import label


def prepare(q: float):
    fc = json.loads((ROOT / "Modeling/tables/regime_followup/contract.json").read_text(encoding="utf-8"))
    ac = json.loads((ROOT / "Modeling/tables/regime_age_ablation/contract.json").read_text(encoding="utf-8"))
    sc = json.loads((ROOT / "Modeling/tables/regime_forecast/contract.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(ROOT / "data/origin/okm_augumented_2021.csv", encoding="utf-8-sig")
    raw = raw.loc[raw['시간'].between(0,23)].copy()
    raw['timestamp'] = pd.to_datetime(raw['날짜'].astype(str),format='%Y%m%d')+pd.to_timedelta(raw['시간'],unit='h')
    raw = raw.sort_values('timestamp').set_index('timestamp')
    f=raw.reindex(pd.date_range(raw.index.min(),raw.index.max(),freq='h'));f.index.name='timestamp'
    slots=['15분','30분','45분','60분']
    f['valid']=f[slots].notna().all(axis=1);f['level']=f[slots].mean(axis=1).where(f.valid)
    f['maximum']=f[slots].max(axis=1).where(f.valid)
    f['date']=f.index.normalize();f['month']=f.index.month;f['hour']=f.index.hour
    f['weekend']=(f.index.dayofweek>=5).astype(int);f['profile']='not-used-for-prediction'
    h=pd.concat([f.level.shift(k).rename(str(k)) for k in range(1,7)],axis=1)
    f['eligible_onset']=f.valid & h.notna().all(axis=1)
    f['prior_mean']=h['1'];f['past_median6']=h.median(axis=1).where(f.eligible_onset)
    f['past_iqr6']=(h.quantile(.75,axis=1)-h.quantile(.25,axis=1)).where(f.eligible_onset)
    f['past_range6']=(h.max(axis=1)-h.min(axis=1)).where(f.eligible_onset)
    f['past_slope6']=((h['1']-h['6'])/5).where(f.eligible_onset)
    f['prior_delta']=f.level.shift(1)-f.level.shift(2)
    f['prior_last_minus_mean']=f['60분'].shift(1)-f.prior_mean
    f['prior_slot_range']=(f[slots].max(axis=1)-f[slots].min(axis=1)).shift(1).where(f.valid.shift(1,fill_value=False))
    f['prior_production']=f['생산량'].shift(1);f['prior_production_delta']=f['생산량'].shift(1)-f['생산량'].shift(2)
    f['delta']=f.level-f.prior_mean;f['deviation']=f.level-f.past_median6
    f,_=label(f,fc['onset_definition']['thresholds'])
    f['hour_sin']=np.sin(2*np.pi*f.index.hour/24);f['hour_cos']=np.cos(2*np.pi*f.index.hour/24)
    for d in range(7):f[f'dow_{d}']=(f.index.dayofweek==d).astype(int)
    for k in [1,2]:
        f[f'lag{k}_mean']=f.level.shift(k);f[f'lag{k}_maximum']=f.maximum.shift(k)
    f['lag1_last']=f['60분'].shift(1);f['lag1_production']=f['생산량'].shift(1)
    f['lag1_production_zero']=f.lag1_production.eq(0).astype(int)
    f['up_classifier_available']=1;f['down_classifier_available']=1
    # Preserve the established CSV feature boundaries for historical replay consistency.
    historic=pd.read_csv(ROOT/'Modeling/tables/regime_followup/hourly_extended.csv',encoding='utf-8-sig',parse_dates=['timestamp']).set_index('timestamp')
    core=pd.read_csv(ROOT/'Modeling/tables/m01/hourly_frame.csv',encoding='utf-8-sig',parse_dates=['timestamp']).set_index('timestamp')
    all_features=set(fc['regression_features']['B0']+fc['regression_features']['B1']+ac['features']+sc['feature_names'])
    f[list(all_features)]=f[list(all_features)].astype(float)
    common=f.index.intersection(historic.index)
    for c in all_features:
        if c in historic:f.loc[common,c]=historic.loc[common,c]
    common=f.index.intersection(core.index)
    for c in fc['regression_features']['B0']:
        f.loc[common,c]=core.loc[common,c]
    # Extra September rows use the same serialization contract as the original features.
    tail=f.index>=pd.Timestamp('2021-09-01')
    snap=pd.read_csv(io.StringIO(f.loc[tail,list(all_features)].to_csv(index=False)))
    f.loc[tail,list(all_features)]=snap.to_numpy()
    b0model=joblib.load(ROOT/'Modeling/models/regime_followup/B0.joblib')
    b1model=joblib.load(ROOT/'Modeling/models/regime_followup/B1.joblib')
    upmodel=joblib.load(ROOT/'Modeling/models/regime_age_ablation/noage_fixed_june.joblib')
    downmodel=joblib.load(ROOT/'Modeling/models/regime_forecast/down_06_hgb.joblib')
    for name,model,cols in [('b0',b0model,fc['regression_features']['B0']),('b1',b1model,fc['regression_features']['B1'])]:
        mask=f.valid&f[cols].notna().all(axis=1);f[name]=np.nan;f.loc[mask,name]=model.predict(f.loc[mask,cols])
    upmask=f.valid&f[ac['features']].notna().all(axis=1)
    downmask=f.valid&f[sc['feature_names']].notna().all(axis=1)
    f['up_score']=np.nan;f['down_score']=np.nan
    f.loc[upmask,'up_score']=upmodel.predict_proba(f.loc[upmask,ac['features']])[:,1]
    f.loc[downmask,'down_score']=downmodel.predict_proba(f.loc[downmask,sc['feature_names']])[:,1]
    up_threshold=.45546123207077727
    down_threshold=.35224471420611747 # stored prior-OOF threshold for June HGB development model
    f['up_alarm']=f.up_score.ge(up_threshold).astype(int)
    f['down_alarm']=f.down_score.ge(down_threshold).astype(int)
    f['routed']=(f.up_alarm.eq(1)&f.b1.notna()).astype(int)
    f['prediction']=np.where(f.routed,f.b1,f.b0)
    official=pd.read_csv(ROOT/'Modeling/tables/regime_age_ablation/followup_predictions.csv',encoding='utf-8-sig',parse_dates=['timestamp'])
    official=official.loc[official.variant=='G1_noage'].set_index('timestamp')
    np.testing.assert_allclose(f.loc[official.index,'prediction'],official.prediction,rtol=0,atol=1e-10)
    oldscore=pd.read_csv(ROOT/'Modeling/tables/regime_age_ablation/followup_classification.csv',encoding='utf-8-sig',parse_dates=['timestamp'])
    oldscore=oldscore.loc[oldscore.method=='noage'].set_index('timestamp')
    np.testing.assert_allclose(f.loc[oldscore.index,'up_score'],oldscore.score,rtol=0,atol=1e-12)
    scaler,linear=upmodel[0],upmodel[1]
    contributions=scaler.transform(f.loc[upmask,ac['features']])*linear.coef_[0]
    groups={'production':['prior_production','prior_production_delta'],'state':['prior_state_left_censored','prior_state_direction'],
            'calendar':[c for c in ac['features'] if c in ['hour_sin','hour_cos','month','weekend'] or c.startswith('dow_')]}
    groups['power']=[c for c in ac['features'] if c not in sum(groups.values(),[])]
    for name,cols in groups.items():
        f[name+'_contribution']=np.nan
        f.loc[upmask,name+'_contribution']=contributions[:,[ac['features'].index(c) for c in cols]].sum(axis=1)
    reconstructed=linear.intercept_[0]+contributions.sum(axis=1)
    np.testing.assert_allclose(reconstructed,upmodel.decision_function(f.loc[upmask,ac['features']]),atol=1e-10)
    # A clearly marked persistence reference is available after the first complete hour.
    ref=f.valid&f.prediction.isna()&f.lag1_maximum.notna()
    f.loc[ref,'prediction']=f.loc[ref,'lag1_maximum']
    f['reference']=ref
    def clean(v):return None if pd.isna(v) else float(v)
    forecasts=[];contexts=[];events=[]
    for t,r in f.loc[f.valid].iterrows():
        ms=int(t.timestamp()*1000);p=clean(r.prediction)
        mode='reference' if r.reference else 'warmup' if p is None else 'official' if t in official.index else 'backcast' if t<pd.Timestamp('2021-07-01') else 'extension'
        forecasts.append([ms,p,None if p is None or r.reference else p-q,None if p is None or r.reference else p+q,
                          clean(r.up_score),int(r.up_alarm),int(r.routed),clean(r.b0),clean(r.b1),clean(r.down_score),int(r.down_alarm),mode,
                          *[clean(r[g+'_contribution']) for g in ['power','calendar','production','state']]])
        contexts.append([ms,clean(r.prior_mean),clean(r.prior_production),clean(r.prior_delta),clean(r.past_median6),clean(r.past_range6),clean(r.prior_last_minus_mean)])
        if r.onset:events.append([ms,int(r.onset),clean(r.delta)])
    downcases=f.loc[f.valid&f.down_alarm.eq(1)&f.sustained_onset.eq(-1)&(f.index>=pd.Timestamp('2021-06-01'))]
    down_case=downcases.index[0] if len(downcases) else f.loc[f.onset.eq(-1)].index[-1]
    result={'forecasts':forecasts,'contexts':contexts,'events':events,
            'scenes':[{'time':'2021-07-12T07:45:00Z','label':'지속 상승 경고 · 선택 보완'},
                      {'time':(down_case-pd.Timedelta(minutes=15)).isoformat()+'Z','label':'지속 하락 경고 · 실험적'},
                      {'time':'2021-03-21T07:45:00Z','label':'급등 관측 · 큰 전력 전환'},
                      {'time':'2021-01-04T12:00:00Z','label':'평상 구간 · 기본 예측'}],
            'verification':{'official_prediction_matches':1344,'official_score_matches':1428,'fits':0,'saved_models_loaded':4,
                            'model_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                                ROOT/'Modeling/models/regime_followup/B0.joblib',ROOT/'Modeling/models/regime_followup/B1.joblib',
                                ROOT/'Modeling/models/regime_age_ablation/noage_fixed_june.joblib',ROOT/'Modeling/models/regime_forecast/down_06_hgb.joblib']},
                            'feature_contracts':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                                ROOT/'Modeling/tables/regime_followup/contract.json',ROOT/'Modeling/tables/regime_age_ablation/contract.json',
                                ROOT/'Modeling/tables/regime_forecast/contract.json']},
                            'all_valid_hour_rows':len(forecasts),'reference_hours':int(ref.sum()),
                            'model_hours':int((f.valid&f.b0.notna()).sum()),'warmup_hours':int((f.valid&f.prediction.isna()).sum()),
                            'logit_decomposition':'exact; no causal claim',
                            'full_period_scope':'Jan-Jun backcast of model trained on Jan-Jun; NOT out-of-sample validation. Other added rows are unevaluated inference. KPI fixed to original official Jul-Aug comparison.',
                            'down_warning':'Existing May-end HGB plus stored June boundary; experimental illustrative extension, not finalized deployment.'}}
    target=ROOT/'dashboard/full_period_inference.json'
    target.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
    return result
