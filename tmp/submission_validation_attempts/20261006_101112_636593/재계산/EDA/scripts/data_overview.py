"""Data manuscript reproduction and section 2.1 hourly summaries.

Adapted from the moved eda_analysis.py prepare/narrative calculations.
Does not train a model or change the original CSV. No plots in data chapter.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'EDA/tables'
SOURCE=ROOT/'data/origin/okm_augumented_2021.csv'
SHA='8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830'
SLOTS=['15분','30분','45분','60분']

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
    raw=pd.read_csv(SOURCE,encoding='utf-8-sig')
    scope=raw.loc[raw['날짜'].lt(20210901)].copy()
    valid=scope.loc[scope['시간'].between(0,23)].copy()
    assert raw.shape==(6168,18) and len(scope)==5832 and len(valid)==5784
    assert raw['날짜'].nunique()==257 and valid['날짜'].nunique()==241
    assert len(raw.loc[raw['날짜'].ge(20210901)])==336
    assert valid.groupby('날짜').size().eq(24).all()
    np.testing.assert_array_equal(raw['평균'],np.floor(raw[SLOTS].mean(axis=1)+.5))
    ratio=raw['생산량']/raw[SLOTS].sum(axis=1)
    available=raw['공장인원'].notna()
    # The existing variable audit reports max absolute storage error ~4.94e-9
    # (data/origin/데이터_변수표.md, section 6). Do not assume fixed 9 decimals.
    ratio_error=float((raw.loc[available,'공장인원']-ratio.loc[available]).abs().max())
    np.testing.assert_allclose(raw.loc[available,'공장인원'],ratio.loc[available],rtol=0,atol=5e-9)
    missing=raw.isna().sum().rename_axis('variable').reset_index(name='missing_n')
    missing.to_csv(OUT/'data_missing.csv',index=False,encoding='utf-8-sig')
    assert missing.missing_n.sum()==21
    assert raw['풍속'].isna().sum()==3 and raw['강수량'].isna().sum()==1 and (~available).sum()==17
    raw.loc[raw.isna().any(axis=1),['날짜','시간','풍속','강수량','공장인원']].to_csv(
        OUT/'data_missing_records.csv',index=False,encoding='utf-8-sig')
    outliers=[]
    for variable in ['생산량','강수량']:
        v=scope[variable].dropna();q1,q3=v.quantile([.25,.75]);iqr=q3-q1
        outliers.append(dict(variable=variable,n=len(v),zero_n=int(v.eq(0).sum()),
            zero_share=float(v.eq(0).mean()),q1=float(q1),q3=float(q3),iqr=float(iqr),
            outside_iqr_n=int((v.lt(q1-1.5*iqr)|v.gt(q3+1.5*iqr)).sum()),maximum=float(v.max())))
    pd.DataFrame(outliers).to_csv(OUT/'data_distribution.csv',index=False,encoding='utf-8-sig')
    assert outliers[0]['outside_iqr_n']==533 and outliers[0]['maximum']==9830
    assert outliers[1]['outside_iqr_n']==1442 and outliers[1]['iqr']==0
    invalid=scope.loc[~scope['시간'].between(0,23)]
    assert len(invalid)==48 and set(invalid['날짜'])=={20210713,20210715}
    assert invalid['시간'].min()==70 and invalid['시간'].max()==188
    invalid[['날짜','시간']].to_csv(OUT/'data_invalid_times.csv',index=False,encoding='utf-8-sig')
    valid=valid.sort_values(['날짜','시간'])
    profiles=valid.groupby('날짜')[SLOTS].apply(lambda x:tuple(x.to_numpy().ravel()))
    multiplicity=profiles.value_counts()
    assert len(multiplicity)==126 and int(multiplicity.loc[multiplicity.gt(1)].sum())==160
    prod_unique=valid.groupby('날짜')['생산량'].nunique()
    assert prod_unique.gt(1).sum()==181 and not raw.duplicated().any()
    assert raw['생산량'].eq(0).sum()==2657
    assert (raw['생산량'].eq(0)&raw['평균'].gt(0)).sum()==2640
    assert raw[SLOTS].eq(0).all(axis=1).sum()==17
    dates=pd.to_datetime(valid['날짜'].astype(str),format='%Y%m%d')
    valid['weekend']=dates.dt.dayofweek.ge(5)
    summaries=[]
    for group,flag in [('weekday',False),('weekend',True)]:
        for hour,part in valid.loc[valid.weekend.eq(flag)].groupby('시간'):
            summaries.append(dict(population=group,hour=int(hour),n=len(part),
                production_mean=float(part['생산량'].mean()),power_mean=float(part['평균'].mean())))
    pd.DataFrame(summaries).to_csv(OUT/'hourly_overview.csv',index=False,encoding='utf-8-sig')
    # Check availability only; actual model choices remain in archived model evidence.
    series=pd.Series(valid['평균'].to_numpy(),index=dates+pd.to_timedelta(valid['시간'],unit='h'))
    grid=series.reindex(pd.date_range('2021-01-01','2021-08-31 23:00',freq='h'))
    eligible=grid.shift(1).rolling(24,min_periods=24).count().eq(24)
    for lag in (1,2,3,24,168):eligible &= grid.shift(lag).notna()
    assert int(eligible.reindex(series.index).sum())==5520
    facts=dict(status='passed',source_sha256=SHA,raw_rows=6168,columns=18,
        scope_rows=5832,valid_hours=5784,valid_days=241,missing_cells=21,
        headcount_formula_checked_rows=int(available.sum()),headcount_formula_max_abs_error=ratio_error,unique_96slot_profiles=126,
        dates_in_repeated_profiles=160,multiple_production_value_dates=181,
        eligible_history_hours=5520,model_training_performed=False)
    (OUT/'data_overview_manifest.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(facts,ensure_ascii=False))

if __name__=='__main__':main()
