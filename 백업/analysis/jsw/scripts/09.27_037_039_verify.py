"""Independent source arithmetic and saved-result checks for 037-039."""
import json
from hashlib import sha256
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from analysis_common import BASE, TABLES, SOURCE, SOURCE_HASH, SLOTS

assert sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH
raw = pd.read_csv(SOURCE, encoding="utf-8-sig")
raw = raw.loc[raw["시간"].between(0,23) & raw["날짜"].lt(20210901)].copy()
raw["date"] = pd.to_datetime(raw["날짜"].astype(str), format="%Y%m%d")
raw["timestamp"] = raw.date + pd.to_timedelta(raw["시간"],unit="h")
raw["peak"] = raw[SLOTS].max(axis=1)
raw["exact_mean"] = raw[SLOTS].mean(axis=1)
assert len(raw)==5784 and raw.timestamp.is_unique
part = raw.loc[raw.date.dt.month.isin([6,7,8]) & raw.date.dt.dayofweek.lt(5) & raw["생산량"].gt(0)].copy()
assert len(part)==1244
summary = pd.read_csv(TABLES/"09.27_037_summary.csv",encoding="utf-8-sig")
ref = pd.read_csv(TABLES/"09.27_037_reference_0.csv",encoding="utf-8-sig").set_index("hour").reference_weight
for month,p in part.groupby(part.date.dt.month):
    wt = p["시간"].map(ref/p.groupby("시간").size())
    r = summary.loc[summary.month.eq(month)&~summary.profile_weighted].iloc[0]
    assert r.hours==len(p) and r.dates==p.date.nunique()
    assert np.isclose(wt@p.peak,r.max_average) and np.isclose(wt@p["평균"],r.mean_average)
    assert np.isclose(wt@p.peak.ge(182),r.high_182)

hours = pd.read_csv(TABLES/"09.27_038_residual_context.csv",encoding="utf-8-sig")
hours=hours.loc[hours.date.notna()].copy()
hours["date"]=pd.to_datetime(hours.date)
hours["timestamp"]=hours.date+pd.to_timedelta(hours.hour,unit="h")
merged=hours.merge(raw,on="timestamp",validate="one_to_one",suffixes=("_result","_source"))
assert len(merged)==5784
assert np.allclose(merged.actual_mean,merged.exact_mean) and np.allclose(merged.actual_peak,merged.peak)
dates = pd.read_csv(TABLES/"09.27_038_date_context.csv",encoding="utf-8-sig")
dates["date"]=pd.to_datetime(dates.date)
independent = hours.groupby("date").squared_error_sum.sum().div(96).pow(.5)
assert np.allclose(dates.fa_rms,dates.date.map(independent))
assert np.isclose(dates.loc[dates.date.eq("2021-08-09"),"fa_rms"].iloc[0],20.488585590274692)

part["logprod"]=np.log1p(part["생산량"])
cols=["logprod","기온","풍속","습도"]
complete=part.dropna(subset=cols+["강수량"])
assert len(complete)==1242
june=complete.loc[complete.date.dt.month.eq(6)]
mu=june[cols].mean().to_numpy(); sd=june[cols].std(ddof=0).to_numpy()
eligible=complete.set_index("timestamp")
support=pd.read_csv(TABLES/"09.27_039_support.csv",encoding="utf-8-sig")
for later in (7,8):
    stored=pd.read_csv(TABLES/f"09.27_039_support_rows_{later}.csv",encoding="utf-8-sig")
    stored["timestamp"]=pd.to_datetime(stored.timestamp)
    chosen=eligible.loc[stored.loc[stored.included,"timestamp"]]
    assert len(chosen)==support.loc[support.later_month.eq(later),"eligible_rows_after_cell_gate"].iloc[0]
    counts=chosen.groupby([chosen.date.dt.month,"시간"]).size().unstack(0)
    assert len(counts)==18 and counts.min().min()>=5
    for month,p in chosen.groupby(chosen.date.dt.month):
        opposite=chosen.loc[chosen.date.dt.month.ne(month)]
        for hour in p["시간"].unique():
            for wet in (False,True):
                a=p.loc[p["시간"].eq(hour)&p["강수량"].gt(0).eq(wet)]
                b=opposite.loc[opposite["시간"].eq(hour)&opposite["강수량"].gt(0).eq(wet)]
                if a.empty: continue
                assert not b.empty
                distance=cdist((a[cols].to_numpy()-mu)/sd,(b[cols].to_numpy()-mu)/sd,metric="chebyshev")
                assert (distance.min(axis=1)<=1+1e-12).all()
models=pd.read_csv(TABLES/"09.27_039_models.csv",encoding="utf-8-sig")
assert models["rank"].eq(models["columns"]).all()
for prefix in ("09.27_037","09.27_038","09.27_039"):
    facts=json.loads((TABLES/f"{prefix}_facts.json").read_text(encoding="utf-8"))
    assert facts["source_sha256"]==SOURCE_HASH
    assert facts["frozen_sha256"]==sha256((TABLES/f"{prefix}_frozen.json").read_bytes()).hexdigest()
print(json.dumps({"verified":True,"source_rows":5784,"distribution_rows":1244,"complete_case_rows":1242,
                  "fa_source_hours":len(merged),"joint_support_rows":[302,303],"full_rank_models":len(models)},ensure_ascii=True))
