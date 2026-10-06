"""Scoped relative-change check for approved A01 observation comparisons."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import pandas as pd
from a01_production_changes import ROOT,SOURCE,SHA,OUT,contrast,SLOTS

PREFIX='relative_'
A='zero_to_positive'
B='positive_to_positive'


def save(frame,name):
    frame.to_csv(OUT/f'{PREFIX}{name}.csv',index=False,encoding='utf-8-sig')


def main():
    source_obs=OUT/'observations.csv'
    prior_verification=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(source_obs.read_bytes()).hexdigest()==prior_verification['outputs_sha256']['observations.csv']
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
    part=pd.read_csv(source_obs,encoding='utf-8-sig')
    part['timestamp']=pd.to_datetime(part.timestamp)
    part['date']=pd.to_datetime(part.date)
    part['week']=part.date-pd.to_timedelta(part.date.dt.dayofweek,unit='D')
    part['change']=part.transition
    part['relative_change_pct']=np.where(part.past_level.gt(0),
        part.abs_power_delta/part.past_level.replace(0,np.nan)*100,np.nan)
    pair=part.loc[part.transition.isin([A,B])].copy()
    support=pair.groupby(['month','weekend','시간','transition']).date.nunique().unstack(fill_value=0)
    support=support.index[(support[A]>=2)&(support[B]>=2)]
    matched=pair.set_index(['month','weekend','시간']).loc[support].reset_index()
    assert len(support)==46
    assert matched.transition.eq(A).sum()==150 and matched.transition.eq(B).sum()==567
    contract=dict(previous='Absolute matched mean changes37.9208 versus30.2138; user asked whether prior power levels influence interpretation.',
        scope='Keep same717 matched records and46 month-hour-weekday/weekend conditions for primary comparison.',
        metric='100*abs(current four-slot mean minus exact1h past mean)/past mean',
        zero_denominator='Undefined, no epsilon imputation; count and retain separately in source observations.',
        primary='Absolute change remains main; relative percent only checks conclusion.',
        low_denominator='Use existing currentEDA low-power band20..26 as sensitivity; no new threshold search or peak definition. Prior mean>26 is supplementary.',
        fixed_tests='Basic; profile balanced; same day; exclude each month; prior mean>26.',
        not_causal=True,forecast=False,manuscript=False,backup=False,source_sha256=SHA,
        observations_sha256=prior_verification['outputs_sha256']['observations.csv'])
    (OUT/'relative_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    rows=[]
    cases=[('primary',matched),('same_day',matched.loc[matched['시간'].ne(0)]),
           ('outside_existing_low_band',matched.loc[matched.past_level.gt(26)])]+[
           (f'exclude_month_{m}',matched.loc[matched.month.ne(m)]) for m in range(1,9)]
    for label,sample in cases:
        finite=sample.loc[sample.past_level.gt(0)]
        for metric in ('abs_power_delta','relative_change_pct'):
            for mode in ('date_hour','profile_balanced') if label in ('primary','outside_existing_low_band') else ('date_hour',):
                result=contrast(finite,A,B,metric,mode,boot=label=='primary')
                if result:
                    rows.append(dict(check=label,undefined_excluded=len(sample)-len(finite),**result))
    results=pd.DataFrame(rows)
    save(results,'contrasts')
    detail=[]
    for label,cell in [('all_pair_candidates',pair),('matched_primary',matched)]:
        for group,subset in cell.groupby('transition'):
            valid=subset.loc[subset.past_level.gt(0)]
            low=valid.past_level.between(20,26)
            detail.append(dict(scope=label,transition=group,n=len(subset),zero_denominator=int(subset.past_level.eq(0).sum()),
                past_min=valid.past_level.min(),past_q10=valid.past_level.quantile(.1),past_median=valid.past_level.median(),
                past_q90=valid.past_level.quantile(.9),relative_mean=valid.relative_change_pct.mean(),
                relative_median=valid.relative_change_pct.median(),relative_q90=valid.relative_change_pct.quantile(.9),
                prior_existing_low_band_hours=int(low.sum()),
                low_band_relative_sum_share=float(valid.loc[low,'relative_change_pct'].sum()/valid.relative_change_pct.sum())))
    save(pd.DataFrame(detail),'denominators')
    save(matched[['timestamp','date','month','weekend','시간','transition','past_level','level','abs_power_delta','relative_change_pct','profile_weight']], 'matched_observations')
    # Independent raw integer CSV arithmetic for each relative change.
    raw={}
    with SOURCE.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            date,hour=int(row['날짜']),int(row['시간'])
            if 20210101<=date<=20210831 and 0<=hour<=23:
                stamp=pd.Timestamp(str(date))+pd.Timedelta(hours=hour)
                raw[stamp]=sum(int(row[c]) for c in SLOTS)
    for row in matched.itertuples():
        now=raw[row.timestamp]
        old=raw[row.timestamp-pd.Timedelta(hours=1)]
        expected=abs(now-old)/old*100 if old>0 else np.nan
        np.testing.assert_allclose(row.relative_change_pct,expected,rtol=1e-12,equal_nan=True)
    # independent primary same-calendar percent averaging, using Python groups
    cells={}
    with (OUT/'relative_matched_observations.csv').open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if float(row['past_level'])<=0:
                continue
            key=(row['month'],row['weekend'],row['시간'])
            cells.setdefault(key,{A:[],B:[]})[row['transition']].append(float(row['relative_change_pct']))
    num_a=num_b=den=0.0
    for cell in cells.values():
        assert cell[A] and cell[B]
        w=len(cell[A])+len(cell[B])
        num_a+=w*sum(cell[A])/len(cell[A])
        num_b+=w*sum(cell[B])/len(cell[B])
        den+=w
    result=results.loc[(results.check=='primary')&(results.metric=='relative_change_pct')&(results.weighting=='date_hour')].iloc[0]
    np.testing.assert_allclose([num_a/den,num_b/den],[result.adjusted_a,result.adjusted_b],rtol=1e-12)
    # Preserve exact original absolute scores when denominators did not exclude rows.
    original=pd.read_csv(OUT/'transition_matched_contrasts.csv',encoding='utf-8-sig')
    for mode in ('date_hour','profile_balanced'):
        prior=original.loc[(original.a==A)&(original.b==B)&(original.metric=='abs_power_delta')&(original.weighting==mode)].iloc[0]
        current=results.loc[(results.check=='primary')&(results.metric=='abs_power_delta')&(results.weighting==mode)].iloc[0]
        if not current.undefined_excluded:
            np.testing.assert_allclose([prior.adjusted_a,prior.adjusted_b],[current.adjusted_a,current.adjusted_b],rtol=1e-12)
    verification=dict(status='passed',original_observations_preserved=True,source_sha256=SHA,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        matched_rows=len(matched),matched_cells=len(support),
        all_zero_denominator_hours=int(part.past_level.eq(0).sum()),
        primary_zero_denominator_hours=int(matched.past_level.eq(0).sum()),
        independent_raw_rows_checked=len(matched),primary_percent_grouping_verified=True,
        outputs_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('relative_*.csv')})
    (OUT/'relative_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(results.to_json(orient='records'))
    print(pd.DataFrame(detail).to_json(orient='records'))
    print(json.dumps(verification))


if __name__=='__main__':
    main()
