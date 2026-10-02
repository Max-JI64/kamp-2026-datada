"""One fixed hierarchical comparison to the prior K-means result."""
import json
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from analysis_common import TABLES, load, SLOTS, save, finish

PREFIX = "09.27_025"
cfg = json.loads((TABLES / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
_, data = load()
matrix = np.vstack([g[SLOTS].to_numpy().ravel() for _, g in data.groupby("date", sort=True)])
dates = pd.Index(sorted(data.date.unique()), name="date")
assert matrix.shape == (241, 96)
daily = pd.read_csv(TABLES / "09.27_015_daily.csv", encoding="utf-8-sig", parse_dates=["date"])
assert np.array_equal(daily.date.to_numpy(), dates.to_numpy())
train = daily.period.eq("Jan-Jun").to_numpy()
xs = StandardScaler().fit(matrix[train]).transform(matrix)
with threadpool_limits(limits=1):
    ward = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(xs[train])
centroids = np.vstack([xs[train][ward == c].mean(axis=0) for c in range(3)])
labels = np.empty(len(daily), dtype=int)
labels[train] = ward
distances = ((xs[~train, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
labels[~train] = distances.argmin(axis=1)
daily["ward_cluster"] = labels
km = pd.read_csv(TABLES / "09.27_008_cluster_days.csv", encoding="utf-8-sig", parse_dates=["date"])
km = km.loc[km.representation.eq("raw") & km.k.eq(3) & ~km.profile_weighted, ["date", "cluster"]]
merged = daily.merge(km.rename(columns={"cluster": "kmeans_cluster"}), on="date", validate="one_to_one")
assert len(merged) == 241
profiles, agreements, parts = [], [], []
for period, group in merged.groupby("period"):
    agreements.append({"period": period, "days": len(group),
                       "ward_kmeans_ari": adjusted_rand_score(group.ward_cluster, group.kmeans_cluster)})
    for weighted in (False, True):
        w = group.profile_weight.to_numpy(copy=True) if weighted else np.ones(len(group))
        w /= w.sum()
        for cluster in range(3):
            mask = group.ward_cluster.eq(cluster).to_numpy()
            profiles.append({"period": period, "profile_weighted": weighted, "ward_cluster": cluster,
                             "days": int(mask.sum()), "day_share": w[mask].sum(),
                             "mean_high_hours": np.average(group.loc[mask, "high_hours"], weights=w[mask]) if mask.any() else np.nan,
                             "mean_peak": np.average(group.loc[mask, "mean_peak"], weights=w[mask]) if mask.any() else np.nan})
for weighted in (False, True):
    tab = pd.DataFrame(profiles).loc[lambda x: x.profile_weighted.eq(weighted)]
    support = tab.pivot(index="ward_cluster", columns="period", values="days")
    if not support.gt(0).all().all():
        parts.append({"profile_weighted": weighted, "common_groups": int(support.gt(0).all(axis=1).sum()),
                      "status": "not_identifiable_on_all_groups", "mix_component": np.nan,
                      "within_component": np.nan, "high_fraction_difference": np.nan})
        continue
    mix_total = within_total = 0.
    for cluster in range(3):
        a = tab.loc[tab.period.eq("Jan-Jun") & tab.ward_cluster.eq(cluster)].iloc[0]
        b = tab.loc[tab.period.eq("Jul-Aug") & tab.ward_cluster.eq(cluster)].iloc[0]
        mix_total += (b.day_share - a.day_share) * (a.mean_high_hours + b.mean_high_hours) / 48
        within_total += (b.mean_high_hours - a.mean_high_hours) * (a.day_share + b.day_share) / 48
    expected = 220 / 1440 - 235 / 4344 if not weighted else pd.read_csv(TABLES / "09.27_021_contributions.csv", encoding="utf-8-sig").loc[lambda x: x.profile_weighted, "high_fraction_difference"].iloc[0]
    assert np.isclose(mix_total + within_total, expected)
    parts.append({"profile_weighted": weighted, "common_groups": 3,
                  "status": "computed", "mix_component": mix_total,
                  "within_component": within_total, "high_fraction_difference": expected})
save(merged[["date", "period", "profile_weight", "high_hours", "ward_cluster", "kmeans_cluster"]], PREFIX, "assignments")
save(pd.DataFrame(profiles), PREFIX, "cluster_profiles")
save(pd.DataFrame(agreements), PREFIX, "agreement")
save(pd.DataFrame(parts), PREFIX, "contributions")
save(merged.groupby(["period", "ward_cluster", "kmeans_cluster"]).size().rename("days").reset_index(), PREFIX, "crosswalk")
finish(PREFIX, {"agreement": agreements, "contributions": parts,
                "training_cluster_sizes": np.bincount(ward, minlength=3).tolist()})
