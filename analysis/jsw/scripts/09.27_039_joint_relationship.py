"""Predefined explanatory spline associations and empirical joint support."""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer, StandardScaler
from scipy.spatial.distance import cdist
from analysis_common import TABLES, load, save, finish

PREFIX = "09.27_039"
cfg = json.loads((TABLES / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
_, data = load()
data["logprod"] = np.log1p(data["생산량"])
data["lograin"] = np.log1p(data["강수량"])
data["wet"] = data["강수량"].gt(0).astype(int)
data["week"] = data.date - pd.to_timedelta(data.date.dt.dayofweek, unit="D")
eligible = data.weekend.eq(0) & data.production_positive.eq(1)
train = data.loc[eligible & data.period.eq("Jan-Jun")]
w = data.loc[eligible & data.month.isin(cfg["months"])].copy()
required = ["logprod", "기온", "풍속", "습도", "lograin"]
missing = w.loc[w[required].isna().any(axis=1), required].reset_index()
save(missing, PREFIX, "missing_covariates")
w = w.dropna(subset=required).copy()
train = train.dropna(subset=required)
spec = cfg["spline"]
prod = SplineTransformer(n_knots=spec["n_knots"], degree=spec["degree"], include_bias=False,
                         extrapolation=spec["extrapolation"]).fit(train[["logprod"]])
temp = SplineTransformer(n_knots=spec["n_knots"], degree=spec["degree"], include_bias=False,
                         extrapolation=spec["extrapolation"]).fit(train[["기온"]])
simple_cols = ["풍속", "습도", "lograin"]
simple = StandardScaler().fit(train[simple_cols])
support_cols = ["logprod", "기온", "풍속", "습도"]
support_scaler = StandardScaler().fit(w.loc[w.month.eq(6), support_cols])
summaries, contrasts, ranges, support_records, sensitivities, group_results = [], [], [], [], [], []

def design(p, kind):
    calendar = pd.get_dummies(p[["month", "hour", "weekday"]].astype(str), drop_first=True, dtype=float)
    calendar.insert(0, "intercept", 1.0)
    mats = [calendar.to_numpy()]
    names = list(calendar.columns)
    if kind in ("production", "joint"):
        z = prod.transform(p[["logprod"]]); mats.append(z)
        names += [f"production_spline_{i}" for i in range(z.shape[1])]
    if kind in ("weather", "joint"):
        z = temp.transform(p[["기온"]]); mats += [z, simple.transform(p[simple_cols]), p[["wet"]].to_numpy()]
        names += [f"temperature_spline_{i}" for i in range(z.shape[1])] + simple_cols + ["wet"]
    return np.column_stack(mats), names

def fitted(p, kind, col, weighted=False, omit=None):
    x, names = design(p, kind)
    weights = p.profile_weight.to_numpy() if weighted else np.ones(len(p))
    y = p[col].to_numpy(float)
    assert np.isfinite(x).all(), f"nonfinite design: {kind}; columns={[names[i] for i in np.where(~np.isfinite(x).all(axis=0))[0]]}"
    assert np.isfinite(y).all() and np.isfinite(weights).all()
    unused = [n for i,n in enumerate(names) if np.all(x[:,i] == 0)]
    group_names = {"production": lambda n:n.startswith("production_spline_"),
                   "temperature": lambda n:n.startswith("temperature_spline_"),
                   "wind": lambda n:n=="풍속", "humidity": lambda n:n=="습도", "rain": lambda n:n in ("wet", "lograin")}
    keep = [i for i,n in enumerate(names) if n not in unused and (omit is None or not group_names[omit](n))]
    x = x[:,keep]; names = [names[i] for i in keep]
    sw = np.sqrt(weights)
    beta, _, rank, singular = np.linalg.lstsq(x * sw[:,None], y * sw, rcond=None)
    error = y - x @ beta
    mean = np.average(y, weights=weights)
    sst = np.sum(weights * (y-mean)**2)
    r2 = 1 - np.sum(weights*error**2)/sst
    condition = float(singular[0]/singular[-1]) if singular[-1] else np.inf
    effects = {int(n.split("_")[1]): float(beta[i]) for i,n in enumerate(names) if n.startswith("month_")}
    return {"r2": float(r2), "rmse": float(np.sqrt(np.average(error**2, weights=weights))),
            "rank": int(rank), "columns": len(names), "condition_number": condition, "unused_basis_columns": ";".join(unused)}, effects

def evaluate(p, label, contrast_allowed):
    for weighted in (False, True):
        for metric, col in (("mean", "평균"), ("max", "peak")):
            for kind in cfg["models"]:
                result, effects = fitted(p, kind, col, weighted)
                summaries.append({"scope": label, "metric": metric, "model": kind, "profile_weighted": weighted,
                                  "n": len(p), "dates": p.date.nunique(), "contrast_allowed": contrast_allowed, **result})
                for month, effect in effects.items():
                    contrasts.append({"scope": label, "metric": metric, "model": kind, "profile_weighted": weighted,
                                      "month": month, "contrast_allowed": contrast_allowed, "conditional_month_contrast": effect})
    if contrast_allowed:
        for weighted in (False, True):
            for metric,col in (("mean","평균"),("max","peak")):
                full,_ = fitted(p,"joint",col,weighted)
                for group in ("production","temperature","wind","humidity","rain"):
                    reduced,_ = fitted(p,"joint",col,weighted,omit=group)
                    group_results.append({"scope":label,"metric":metric,"profile_weighted":weighted,"group":group,
                                          "unique_r2_increment": full["r2"]-reduced["r2"],
                                          "rmse_without_group":reduced["rmse"],"rmse_with_all":full["rmse"]})
        for omitted in sorted(p.week.unique()):
            subset = p.loc[p.week.ne(omitted)]
            if subset.month.nunique() != p.month.nunique():
                continue
            for metric, col in (("mean", "평균"), ("max", "peak")):
                for kind in ("calendar", "joint"):
                    result, effects = fitted(subset, kind, col)
                    for month,effect in effects.items():
                        sensitivities.append({"scope": label, "omitted_week": pd.Timestamp(omitted), "metric": metric,
                                              "model": kind, "month": month, "rank": result["rank"], "columns": result["columns"],
                                              "conditional_month_contrast": effect})

evaluate(w, "all_6_7_8_descriptive", False)
for month in cfg["months"]:
    part = w.loc[w.month.eq(month)]
    for col in support_cols + ["강수량"]:
        ranges.append({"month": month, "variable": col, "min": part[col].min(), "q05": part[col].quantile(.05),
                       "median": part[col].median(), "q95": part[col].quantile(.95), "max": part[col].max()})
for later in (7,8):
    a, b = w.loc[w.month.eq(6)].copy(), w.loc[w.month.eq(later)].copy()
    distances = {6: pd.Series(np.inf, index=a.index), later: pd.Series(np.inf, index=b.index)}
    for hour in range(24):
        for wet in (0,1):
            aa, bb = a.loc[a.hour.eq(hour) & a.wet.eq(wet)], b.loc[b.hour.eq(hour) & b.wet.eq(wet)]
            if aa.empty or bb.empty:
                continue
            d = cdist(support_scaler.transform(aa[support_cols]), support_scaler.transform(bb[support_cols]), metric="chebyshev")
            distances[6].loc[aa.index] = d.min(axis=1)
            distances[later].loc[bb.index] = d.min(axis=0)
    supported = pd.concat([a.loc[distances[6].le(1)], b.loc[distances[later].le(1)]]).sort_index()
    counts = supported.groupby(["hour", "month"]).size().unstack("month").reindex(range(24), fill_value=0).fillna(0)
    good = counts.index[counts.min(axis=1).ge(5)] if {6,later}.issubset(counts.columns) else []
    common = supported.loc[supported.hour.isin(good)]
    date_counts = common.groupby("month").date.nunique()
    gate = cfg["support_gate"]
    permitted = len(good) >= gate["min_hour_cells"] and len(common) >= gate["min_total_rows"] and len(date_counts) == 2 and date_counts.min() >= gate["min_dates_each_month"]
    support_records.append({"later_month": later, "supported_hours_june": int(distances[6].le(1).sum()),
                            "supported_hours_later": int(distances[later].le(1).sum()), "common_clock_cells": len(good),
                            "eligible_rows_after_cell_gate": len(common), "dates_june": int(date_counts.get(6,0)),
                            "dates_later": int(date_counts.get(later,0)), "contrast_allowed": bool(permitted)})
    rows = pd.concat([pd.DataFrame({"timestamp": a.index, "month":6, "distance":distances[6].to_numpy()}),
                      pd.DataFrame({"timestamp":b.index, "month":later, "distance":distances[later].to_numpy()})])
    rows["included"] = rows.timestamp.isin(common.index)
    save(rows, PREFIX, f"support_rows_{later}")
    save(counts.reset_index(), PREFIX, f"support_cells_{later}")
    if permitted:
        evaluate(common, f"supported_6_{later}", True)
        context = common.groupby(["month", "week"], as_index=False).agg(
            hours=("peak","size"), dates=("date","nunique"), peak_mean=("peak","mean"),
            mean_target=("평균","mean"), production_median=("생산량","median"),
            temperature_mean=("기온","mean"), humidity_mean=("습도","mean"),
            wind_mean=("풍속","mean"), wet_hours=("wet","sum"))
        save(context, PREFIX, f"week_context_{later}")

save(pd.DataFrame(summaries), PREFIX, "models")
save(pd.DataFrame(contrasts), PREFIX, "contrasts")
save(pd.DataFrame(ranges), PREFIX, "ranges")
save(pd.DataFrame(support_records), PREFIX, "support")
save(pd.DataFrame(group_results), PREFIX, "partial_groups")
save(pd.DataFrame(sensitivities, columns=["scope","omitted_week","metric","model","month","rank","columns","conditional_month_contrast"]), PREFIX, "week_sensitivity")
finish(PREFIX, {"support": support_records,
               "complete_case_rows": len(w), "excluded_missing_rows": len(missing),
               "ordinary_models": [s for s in summaries if not s["profile_weighted"]],
               "ordinary_contrasts": [s for s in contrasts if not s["profile_weighted"]],
               "week_exclusion_fits": len(sensitivities)})
