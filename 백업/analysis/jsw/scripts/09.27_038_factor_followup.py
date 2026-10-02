"""FA suitability, stability and reconstruction context, without time-code inputs."""
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.linalg import subspace_angles
from analysis_common import TABLES, SLOTS, load, save, finish

PREFIX = "09.27_038"
cfg = json.loads((TABLES / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
assert cfg["hourly_columns"] == SLOTS and cfg["hourly_factors"] == 1 and cfg["daily_factors"] == 2
_, data = load()
train = data.period.eq("Jan-Jun")
daily = data.groupby("date")[SLOTS].apply(lambda p: p.to_numpy().reshape(-1))
matrix = np.stack(daily.to_numpy())
assert matrix.shape == (241, 96)
dates = daily.index
dtrain = dates < pd.Timestamp("2021-07-01")
profile = data.groupby("date").profile.first().reindex(dates)
unique = ~profile.loc[dtrain].duplicated()
assert np.array_equal(matrix[dtrain][unique], np.stack([matrix[dtrain][profile.loc[dtrain].to_numpy() == pid][0] for pid in profile.loc[dtrain][unique]]))
fits, loads, residuals, daily_summary, sensitivity, warns = [], [], [], [], [], []

def fit(x, k, label):
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = FactorAnalysis(n_components=k, svd_method="lapack", rotation="varimax" if k > 1 else None,
                               max_iter=cfg["max_iter"], random_state=cfg["seed"]).fit(z)
    warns.extend({"fit": label, "message": str(c.message)} for c in caught)
    model.components_ *= np.where(model.components_.sum(axis=1) >= 0, 1.0, -1.0)[:,None]
    return scaler, model

hs, hf = fit(data.loc[train, SLOTS].to_numpy(), 1, "hourly-main")
hr = hs.inverse_transform(hf.transform(hs.transform(data[SLOTS].to_numpy())) @ hf.components_ + hf.mean_)
sign = 1 if hf.components_.sum() >= 0 else -1
data["fa_score"] = hf.transform(hs.transform(data[SLOTS].to_numpy())).ravel() * sign
data["fa_rms"] = np.sqrt(np.mean((data[SLOTS].to_numpy() - hr) ** 2, axis=1))
data["fa_peak_residual"] = data.peak - hr.max(axis=1)
data["spread"] = data.peak - data.exact_mean
for i, slot in enumerate(SLOTS):
    loads.append({"fit": "hourly-main", "variable": slot, "factor": 1,
                  "loading": float(hf.components_[0,i] * sign), "noise_variance": float(hf.noise_variance_[i]),
                  "communality": float(hf.components_[0,i] ** 2)})
for period, p in data.groupby("period"):
    fits.append({"fit": "hourly-main", "evaluation": period, "n": len(p),
                 "score_mean_correlation": p.fa_score.corr(p.exact_mean),
                 "rms_median": p.fa_rms.median(), "rms_q95": p.fa_rms.quantile(.95),
                 "peak_residual_mean": p.fa_peak_residual.mean()})
for label, mask in (("Jan-Mar", data.month.le(3)), ("Apr-Jun", data.month.between(4,6)),
                    ("unique-profiles", train & data.date.isin(dates[dtrain][unique]))):
    ss, ff = fit(data.loc[mask, SLOTS].to_numpy(), 1, "hourly-" + label)
    angle = float(np.degrees(subspace_angles((hf.components_ * hs.scale_).T, (ff.components_ * ss.scale_).T))[0])
    sensitivity.append({"representation": "hourly", "fit": label, "n": int(mask.sum()), "subspace_angle_max": angle})
    for i, slot in enumerate(SLOTS):
        loads.append({"fit": "hourly-"+label, "variable": slot, "factor": 1,
                      "loading": float(ff.components_[0,i] * np.sign(ff.components_.sum())),
                      "noise_variance": float(ff.noise_variance_[i]), "communality": float(ff.components_[0,i] ** 2)})

ds, df = fit(matrix[dtrain], cfg["daily_factors"], "daily-main")
z = ds.transform(matrix)
scores = df.transform(z)
reconstructed = ds.inverse_transform(scores @ df.components_ + df.mean_)
errors = matrix - reconstructed
for label, mask in (("Jan-Mar", dates.month <= 3), ("Apr-Jun", (dates.month >= 4) & (dates.month <= 6)),
                    ("unique-profiles", np.asarray(dtrain) & np.asarray(~profile.duplicated()))):
    ss, ff = fit(matrix[mask], cfg["daily_factors"], "daily-"+label)
    angles = np.degrees(subspace_angles((df.components_ * ds.scale_).T, (ff.components_ * ss.scale_).T))
    sensitivity.append({"representation": "daily", "fit": label, "n": int(mask.sum()),
                        "subspace_angle_max": float(angles.max()), "subspace_angle_min": float(angles.min())})
    if label != "unique-profiles":
        fitted_profiles = set(profile.loc[mask])
        for evaluation, emask in (("Apr-Jun", (dates.month >= 4) & (dates.month <= 6)), ("Jul-Aug", ~dtrain)):
            if label == "Apr-Jun" and evaluation == "Apr-Jun":
                continue
            ev = matrix[emask]
            pred = ss.inverse_transform(ff.transform(ss.transform(ev)) @ ff.components_ + ff.mean_)
            rms = np.sqrt(np.mean((ev-pred)**2, axis=1))
            fits.append({"fit": "daily-"+label, "evaluation": evaluation, "n": len(ev),
                         "rms_median": float(np.median(rms)), "rms_q95": float(np.quantile(rms,.95)),
                         "evaluation_days_with_training_profile": int(profile.loc[emask].isin(fitted_profiles).sum())})
for i in range(96):
    for f in range(cfg["daily_factors"]):
        loads.append({"fit": "daily-main", "variable": f"hour={i//4:02d};slot={SLOTS[i%4]}", "factor": f+1,
                      "loading": float(df.components_[f,i]), "noise_variance": float(df.noise_variance_[i]),
                      "communality": float(np.sum(df.components_[:,i]**2))})
for j, date in enumerate(dates):
    part = data.loc[data.date.eq(date)]
    daily_summary.append({"date": date, "period": part.period.iloc[0], "profile": part.profile.iloc[0],
                          "weekday": date.dayofweek, "mean_power": matrix[j].mean(), "peak_power": matrix[j].max(),
                          "fa_rms": float(np.sqrt(np.mean(errors[j]**2))), "factor1": scores[j,0], "factor2": scores[j,1],
                          "high_hours": int(part.peak.ge(182).sum()), "zero_hours": int(part.peak.eq(0).sum()),
                          "production_total": part["생산량"].sum(), "temperature_mean": part["기온"].mean(),
                          "worst_hour": int(np.argmax(np.mean(errors[j].reshape(24,4)**2, axis=1)))})
    for h in range(24):
        residuals.append({"date": date, "period": part.period.iloc[0], "hour": h,
                          "squared_error_sum": float(np.sum(errors[j].reshape(24,4)[h]**2)),
                          "actual_mean": float(matrix[j].reshape(24,4)[h].mean()),
                          "reconstructed_mean": float(reconstructed[j].reshape(24,4)[h].mean()),
                          "actual_peak": float(matrix[j].reshape(24,4)[h].max()),
                          "reconstructed_peak": float(reconstructed[j].reshape(24,4)[h].max()),
                          "production": part.iloc[h]["생산량"], "temperature": part.iloc[h]["기온"]})
dd = pd.DataFrame(daily_summary)
for period, p in dd.groupby("period"):
    fits.append({"fit": "daily-main", "evaluation": period, "n": len(p),
                 "rms_median": p.fa_rms.median(), "rms_q95": p.fa_rms.quantile(.95),
                 "factor1_mean_correlation": p.factor1.corr(p.mean_power),
                 "factor2_mean_correlation": p.factor2.corr(p.mean_power)})
for metric in ("fa_rms", "fa_peak_residual", "spread"):
    row = data.loc[data.month.isin([6,7,8]) & data.weekend.eq(0) & data.production_positive.eq(1)].copy()
    ref = row.groupby("hour").size(); ref = ref / ref.sum()
    for month, p in row.groupby("month"):
        residuals.append({"month": month, "metric": metric,
                          "standardized_average": float((p.groupby("hour")[metric].mean() * ref).sum())})
save(pd.DataFrame(loads), PREFIX, "loadings")
save(pd.DataFrame(fits), PREFIX, "summary")
save(pd.DataFrame(sensitivity), PREFIX, "stability")
save(dd, PREFIX, "date_context")
save(pd.DataFrame(residuals), PREFIX, "residual_context")
save(data.reset_index()[["timestamp", "period", "hour", "fa_score", "fa_rms", "fa_peak_residual", "spread", "평균", "peak"]], PREFIX, "hourly_scores")
finish(PREFIX, {"training_days": int(dtrain.sum()), "unique_training_profiles": int(unique.sum()),
               "fits": fits, "stability": sensitivity, "warnings": warns,
               "top_later_dates": dd.loc[dd.period.eq("Jul-Aug")].nlargest(5,"fa_rms").to_dict("records")})
