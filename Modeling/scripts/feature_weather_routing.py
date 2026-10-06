"""No fits: isolate weather in B1 only under the unchanged current gate."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from feature_weather_audit import measure
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Modeling/tables/feature_weather_audit'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not (OUT/'routing_weather_run.json').exists()
    contract={'prior_evidence':'B0/B1 weather development MAE worsened; fixed later B1 weather alarm MAE improved. Isolate the applied 7-hour effect without weather-changing B0.',
              'method':'Reuse all frozen predictions and no-age alarm; select B1_weather only at existing alarms, B0 without weather otherwise. No fitting, new thresholds, or candidate adoption.',
              'predictions_sha256':sha(OUT/'predictions.csv'),'script_sha256':sha(__file__),
              'scope':'Same development and already-observed follow-up; diagnostic arithmetic only.'}
    (OUT/'routing_weather_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding='utf-8')
    p=pd.read_csv(OUT/'predictions.csv',encoding='utf-8-sig',parse_dates=['timestamp','date'],float_precision='round_trip')
    base=p.loc[p.variant.eq('B0')].copy().reset_index(drop=True)
    other=p.loc[p.variant.eq('B1_weather')].set_index('timestamp').loc[base.timestamp]
    base['prediction']=np.where(base.fixed_alarm.to_numpy(bool),other.prediction,base.prediction)
    base['variant']='G1_B1weather_only'
    base.to_csv(OUT/'routing_weather_predictions.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    rows=[]
    for period,mask in [('Apr-Jun',base.fold.lt(7)),('May-Jun',base.fold.between(5,6)),('Jul-Aug',base.fold.eq(7))]:
        f=base.loc[mask].reset_index(drop=True)
        for r in measure(f,f.prediction): rows.append({'period':period,**r})
    q=pd.DataFrame(rows)
    original=pd.read_csv(OUT/'metrics.csv',encoding='utf-8-sig').loc[lambda x:x.variant.eq('G1')]
    q=q.merge(original[['period','condition','mae']].rename(columns={'mae':'original_G1_mae'}),on=['period','condition'],validate='one_to_one')
    q['mae_reduction_pct']=100*(q.original_G1_mae-q.mae)/q.original_G1_mae
    q.to_csv(OUT/'routing_weather_metrics.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    # Recompute the blend directly from baseline + gated delta.
    b=p.loc[p.variant.eq('B0')].set_index('timestamp'); w=p.loc[p.variant.eq('B1_weather')].set_index('timestamp').loc[b.index]
    np.testing.assert_allclose(base.prediction.to_numpy(),b.prediction.to_numpy()+b.fixed_alarm.to_numpy()*(w.prediction.to_numpy()-b.prediction.to_numpy()),atol=1e-12)
    (OUT/'routing_weather_run.json').write_text(json.dumps({'fits':0,'blend_verified':True,'rows':len(base),'script_sha256':sha(__file__),'metrics_sha256':sha(OUT/'routing_weather_metrics.csv')},indent=2),encoding='utf-8')
    print(q.to_string(index=False),flush=True)
if __name__=='__main__': main()
