"""Independent checks for fixed cluster-outcome profiles and Ward comparison."""
from hashlib import sha256
import json
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from analysis_common import TABLES, SOURCE, SOURCE_HASH, PROFILES, PROFILE_HASH, SLOTS

assert sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH
assert sha256(PROFILES.read_bytes()).hexdigest() == PROFILE_HASH
raw = pd.read_csv(SOURCE, encoding="utf-8-sig")
src = raw.loc[raw["시간"].between(0, 23) & raw["날짜"].lt(20210901)].copy()
src["date"] = pd.to_datetime(src["날짜"].astype(str), format="%Y%m%d")
assert src.groupby("date").size().eq(24).all()
src["peak"] = src[SLOTS].max(axis=1)
days = src.groupby("date").agg(high_hours=("peak", lambda x: x.ge(182).sum()),
                               mean_peak=("peak", "mean"), low_hours=("peak", lambda x: x.lt(26).sum()))
days["period"] = np.where(days.index < "2021-07-01", "Jan-Jun", "Jul-Aug")
profile = pd.read_csv(PROFILES, encoding="utf-8-sig", dtype={"date": str})
profile["date"] = pd.to_datetime(profile.date)
profile["period"] = np.where(profile.date < "2021-07-01", "Jan-Jun", "Jul-Aug")
profile["weight"] = 1 / profile.groupby(["period", "profile_id"]).date.transform("size")
days["weight"] = profile.set_index("date").weight.reindex(days.index)
assert len(days) == 241 and days.weight.notna().all()
checks = ["source and profile hashes,241 complete days, daily high/low counts"]

prior = pd.read_csv(TABLES / "09.27_008_cluster_days.csv", encoding="utf-8-sig", parse_dates=["date"])
prior = prior.loc[prior.k.eq(3) & ~prior.profile_weighted]
wide = prior.pivot(index="date", columns="representation", values="cluster")
out24 = pd.read_csv(TABLES / "09.27_024_cluster_profiles.csv", encoding="utf-8-sig")
for _, row in out24.iterrows():
    period = days.period.eq(row.period)
    mask = period & wide.loc[days.index, row.representation].eq(row.cluster)
    group = days.loc[mask]
    w = group.weight if row.profile_weighted else np.ones(len(group))
    total = days.loc[period, "weight"].sum() if row.profile_weighted else period.sum()
    assert len(group) == row.days
    assert np.isclose(np.sum(w) / total, row.day_share)
    assert np.isclose(np.average(group.high_hours, weights=w), row.mean_high_hours)
    assert np.isclose(np.average(group.low_hours, weights=w), row.mean_low_hours)
    assert np.isclose(np.average(group.mean_peak, weights=w), row.mean_peak)
out24t = pd.read_csv(TABLES / "09.27_024_totals.csv", encoding="utf-8-sig")
assert np.allclose(out24t.mix_component + out24t.within_component, out24t.high_fraction_difference)
for representation in ("raw", "shape"):
    for period in ("Jan-Jun", "Jul-Aug"):
        subset = days.period.eq(period)
        old = pd.read_csv(TABLES / "09.27_024_representation_agreement.csv", encoding="utf-8-sig").set_index("period")
        assert np.isclose(adjusted_rand_score(wide.loc[days.index[subset], "raw"], wide.loc[days.index[subset], "shape"]), old.loc[period, "raw_shape_ari"])
checks.append("024 prior fixed labels, group profiles and high-hour arithmetic")

matrix = np.vstack([group[SLOTS].to_numpy().ravel() for _, group in src.groupby("date", sort=True)])
train = days.period.eq("Jan-Jun").to_numpy()
X = StandardScaler().fit(matrix[train]).transform(matrix)
with threadpool_limits(limits=1):
    ward_train = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X[train])
centroids = np.array([X[train][ward_train == cluster].mean(axis=0) for cluster in range(3)])
ward_all = np.empty(241, dtype=int)
ward_all[train] = ward_train
ward_all[~train] = np.argmin(((X[~train, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1)
stored = pd.read_csv(TABLES / "09.27_025_assignments.csv", encoding="utf-8-sig", parse_dates=["date"])
assert np.array_equal(stored.date.to_numpy(), days.index.to_numpy())
assert np.array_equal(stored.ward_cluster.to_numpy(), ward_all)
assert np.array_equal(stored.high_hours.to_numpy(), days.high_hours.to_numpy())
assert np.array_equal(stored.kmeans_cluster.to_numpy(), wide.loc[days.index, "raw"].to_numpy())
agreements = pd.read_csv(TABLES / "09.27_025_agreement.csv", encoding="utf-8-sig").set_index("period")
for period in ("Jan-Jun", "Jul-Aug"):
    keep = days.period.eq(period).to_numpy()
    assert np.isclose(adjusted_rand_score(ward_all[keep], stored.kmeans_cluster.to_numpy()[keep]), agreements.loc[period, "ward_kmeans_ari"])
ward_parts = pd.read_csv(TABLES / "09.27_025_contributions.csv", encoding="utf-8-sig")
assert np.allclose(ward_parts.mix_component + ward_parts.within_component, ward_parts.high_fraction_difference)
assert np.allclose(ward_parts.high_fraction_difference.to_numpy(), [220/1440 - 235/4344, out24t.loc[out24t.representation.eq("raw") & out24t.profile_weighted, "high_fraction_difference"].iloc[0]])
checks.append("025 independent Ward fit/projection, stored labels, agreement and decomposition")

for number in (24, 25):
    prefix = f"09.27_{number:03d}"
    facts = json.loads((TABLES / f"{prefix}_facts.json").read_text(encoding="utf-8"))
    assert facts["frozen_sha256"] == sha256((TABLES / f"{prefix}_frozen.json").read_bytes()).hexdigest()
    assert facts["source_sha256"] == SOURCE_HASH and facts["profile_sha256"] == PROFILE_HASH
checks.append("two frozen settings and source/profile hashes")
result = {"status": "passed", "checks": checks, "source_sha256": SOURCE_HASH,
          "profile_sha256": PROFILE_HASH,
          "prior_assignments_sha256": sha256((TABLES / "09.27_008_cluster_days.csv").read_bytes()).hexdigest(),
          "prior_daily_sha256": sha256((TABLES / "09.27_015_daily.csv").read_bytes()).hexdigest()}
(TABLES / "09.27_026_verification.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({"status": "passed", "checks": len(checks), "days": len(days)}))
