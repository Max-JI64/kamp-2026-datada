"""Fixed statistical representations and stability of observed power patterns."""
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FactorAnalysis, NMF
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from analysis_common import load, SLOTS, save, finish

PREFIX = "09.27_008"


def main():
    _, data = load()
    days = sorted(data["date"].unique())
    matrix = np.vstack([g[SLOTS].to_numpy().ravel() for _, g in data.groupby("date", sort=True)])
    metadata = data.groupby("date", sort=True).agg(period=("period", "first"), profile=("profile", "first"), weight=("profile_weight", "first"), daily_mean=("exact_mean", "mean"), peak=("peak", "max"), production_record_sum=("생산량", "sum"), temperature_mean=("기온", "mean")).reset_index()
    assert len(metadata) == len(matrix) == 241 and np.all(matrix >= 0)
    train = metadata["period"].eq("Jan-Jun").to_numpy()
    assert train.sum() == 181
    means = matrix.mean(axis=1)
    assert np.all(means > 0)
    reps = {"raw": matrix.astype(float), "shape": matrix / means[:, None]}
    pca_summary, pca_scores, pca_loadings, cluster_summary, cluster_days, stability = [], [], [], [], [], []
    for rep_name, representation in reps.items():
        pca = PCA(n_components=6, svd_solver="full")
        pca.fit(representation[train])
        scores = pca.transform(representation)
        reconstruction = pca.inverse_transform(scores)
        errors = np.sqrt(((representation - reconstruction) ** 2).mean(axis=1))
        for period in metadata["period"].unique():
            keep = metadata["period"].eq(period).to_numpy()
            pca_summary.append({"representation": rep_name, "period": period,
                                "pc1_train_variance": pca.explained_variance_ratio_[0],
                                "pc2_train_variance": pca.explained_variance_ratio_[1],
                                "six_pc_train_variance": pca.explained_variance_ratio_.sum(),
                                "pc1_mean_correlation": np.corrcoef(scores[keep, 0], means[keep])[0, 1],
                                "median_reconstruction_rms": np.median(errors[keep]), "q95_reconstruction_rms": np.quantile(errors[keep], .95)})
        for i, day in enumerate(days):
            pca_scores.append({"date": day, "representation": rep_name, "period": metadata["period"].iloc[i], "reconstruction_rms": errors[i], **{f"pc{k+1}": scores[i, k] for k in range(6)}})
        for slot in range(96):
            pca_loadings.append({"representation": rep_name, "hour": slot//4, "slot": SLOTS[slot%4], **{f"pc{k+1}": pca.components_[k, slot] for k in range(6)}})
        scaler = StandardScaler().fit(representation[train])
        xs = scaler.transform(representation)
        fits = {}
        for k in (2, 3, 4):
            for weighted in (False, True):
                km = KMeans(n_clusters=k, random_state=20260927, n_init=20)
                w = metadata.loc[train, "weight"].to_numpy() if weighted else np.ones(train.sum())
                km.fit(xs[train], sample_weight=w)
                labels = km.predict(xs)
                fits[(k, weighted)] = labels
                for cluster in range(k):
                    for period in metadata["period"].unique():
                        keep = (labels == cluster) & metadata["period"].eq(period).to_numpy()
                        cluster_summary.append({"representation": rep_name, "k": k, "profile_weighted": weighted, "period": period,
                                                "cluster": cluster, "days": int(keep.sum()),
                                                "daily_mean_median": float(np.median(means[keep])) if keep.any() else np.nan,
                                                "daily_peak_median": float(metadata.loc[keep, "peak"].median()) if keep.any() else np.nan,
                                                "train_silhouette": silhouette_score(xs[train], labels[train])})
                for i, day in enumerate(days):
                    cluster_days.append({"date": day, "representation": rep_name, "k": k, "profile_weighted": weighted,
                                         "period": metadata["period"].iloc[i], "cluster": int(labels[i])})
            stability.append({"representation": rep_name, "comparison": "profile_weight", "k": k,
                              "ari_train": adjusted_rand_score(fits[(k, False)][train], fits[(k, True)][train]),
                              "ari_later": adjusted_rand_score(fits[(k, False)][~train], fits[(k, True)][~train])})
        early = (metadata["date"] < pd.Timestamp("2021-04-01")).to_numpy()
        late = train & ~early
        labels_quarters = []
        for subset in (early, late):
            model = KMeans(n_clusters=3, random_state=20260927, n_init=20).fit(xs[subset])
            labels_quarters.append(model.predict(xs))
        stability.append({"representation": rep_name, "comparison": "Jan-Mar_vs_Apr-Jun", "k": 3,
                          "ari_train": adjusted_rand_score(labels_quarters[0][train], labels_quarters[1][train]),
                          "ari_later": adjusted_rand_score(labels_quarters[0][~train], labels_quarters[1][~train])})
    save(pd.DataFrame(pca_summary), PREFIX, "pca_summary")
    save(pd.DataFrame(pca_scores), PREFIX, "pca_scores")
    save(pd.DataFrame(pca_loadings), PREFIX, "pca_loadings")
    save(pd.DataFrame(cluster_summary), PREFIX, "cluster_summary")
    save(pd.DataFrame(cluster_days), PREFIX, "cluster_days")
    save(pd.DataFrame(stability), PREFIX, "stability")
    standardized = StandardScaler().fit(data.loc[data["period"].eq("Jan-Jun"), SLOTS])
    slot_matrix = standardized.transform(data[SLOTS])
    fa_summary, fa_loadings = [], []
    train_hours = data["period"].eq("Jan-Jun").to_numpy()
    for factors in (1, 2):
        fa = FactorAnalysis(n_components=factors, random_state=20260927, max_iter=1000)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fa.fit(slot_matrix[train_hours])
        scores = fa.transform(slot_matrix)
        for period, part in data.groupby("period"):
            keep = data["period"].eq(period).to_numpy()
            fa_summary.append({"factors": factors, "period": period, "rows": int(keep.sum()),
                               "average_gaussian_loglik": fa.score(slot_matrix[keep]), "iterations": fa.n_iter_,
                               "factor1_mean_power_correlation": np.corrcoef(scores[keep, 0], part["exact_mean"])[0, 1],
                               "warnings": "; ".join(str(w.message) for w in caught)})
        for slot, index in zip(SLOTS, range(4)):
            fa_loadings.append({"factors": factors, "slot": slot, "noise_variance": fa.noise_variance_[index],
                                **{f"factor{k+1}_loading": fa.components_[k, index] for k in range(factors)}})
    save(pd.DataFrame(fa_summary), PREFIX, "fa_summary")
    save(pd.DataFrame(fa_loadings), PREFIX, "fa_loadings")
    nmf = NMF(n_components=2, init="nndsvda", max_iter=2000, random_state=20260927)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        nmf.fit(matrix[train])
        weights = nmf.transform(matrix)
    fitted = weights @ nmf.components_
    nmf_summary = []
    for period in metadata["period"].unique():
        keep = metadata["period"].eq(period).to_numpy()
        nmf_summary.append({"period": period, "relative_frobenius_error": np.linalg.norm(matrix[keep] - fitted[keep]) / np.linalg.norm(matrix[keep]),
                            "iterations": nmf.n_iter_, "warnings": "; ".join(str(w.message) for w in caught)})
    save(pd.DataFrame(nmf_summary), PREFIX, "nmf_summary")
    save(pd.DataFrame({"hour": np.arange(96)//4, "slot": [SLOTS[i%4] for i in range(96)], "basis1": nmf.components_[0], "basis2": nmf.components_[1]}), PREFIX, "nmf_basis")
    _, first_indices = np.unique(matrix[train], axis=0, return_index=True)
    train_indices = np.flatnonzero(train)[first_indices]
    nearest = []
    for i, array in enumerate(matrix):
        possible = train_indices[np.any(matrix[train_indices] != array, axis=1)]
        distances = np.sqrt(((matrix[possible].astype(float) - array) ** 2).mean(axis=1))
        j = possible[np.argmin(distances)]
        nearest.append({"date": days[i], "period": metadata["period"].iloc[i], "nearest_other_profile_date": days[j],
                        "rms_distance": float(distances.min()),
                        "production_record_sum_difference": metadata["production_record_sum"].iloc[i] - metadata["production_record_sum"].iloc[j],
                        "temperature_mean_difference": metadata["temperature_mean"].iloc[i] - metadata["temperature_mean"].iloc[j]})
    save(pd.DataFrame(nearest), PREFIX, "nearest_nonidentical")
    finish(PREFIX, {"pca": pca_summary, "fa": fa_summary, "cluster_stability": stability, "nmf": nmf_summary,
                    "near_profile_summary": pd.DataFrame(nearest).groupby("period")["rms_distance"].agg(["min", "median", "max"]).reset_index().to_dict("records"),
                    "near_rms_le_1": int(pd.DataFrame(nearest)["rms_distance"].le(1).sum())})


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
