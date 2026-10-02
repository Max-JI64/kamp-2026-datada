"""Production size, upper-tail weather contrasts and repetition sensitivity."""
import numpy as np
import pandas as pd
from analysis_common import load, save, finish

PREFIX = "09.27_007"
WEATHER = ["기온", "풍속", "습도", "강수량"]
STRATA = ["month", "hour", "weekend", "production_positive"]


def correlation(x, y, w):
    x, y, w = np.asarray(x), np.asarray(y), np.asarray(w)
    x = x - np.average(x, weights=w)
    y = y - np.average(y, weights=w)
    denominator = np.sqrt(np.sum(w * x*x) * np.sum(w * y*y))
    return float(np.sum(w*x*y) / denominator) if denominator > 0 else np.nan


def main():
    _, data = load()
    cuts = data.loc[data["period"].eq("Jan-Jun") & data["생산량"].gt(0), "생산량"].quantile([.25, .5, .75]).to_numpy()
    data["production_bin"] = np.where(data["생산량"].eq(0), 0, np.searchsorted(cuts, data["생산량"], side="left") + 1)
    data["high"] = data["peak"].ge(182)
    data["eligible"] = data["prev_peak"].notna() & data["prev_peak"].lt(182)
    data["onset"] = data["eligible"] & data["high"]
    production = []
    for period, part in data.groupby("period"):
        candidates = part.loc[part["eligible"] & part["production_bin"].gt(0)]
        cells = []
        for key, group in candidates.groupby(["hour", "weekend"]):
            if set(group["production_bin"].unique()) == {1, 2, 3, 4}:
                cells.append((key, group))
        total_support = sum(len(g) for _, g in cells)
        for level in range(1, 5):
            full = candidates.loc[candidates["production_bin"].eq(level)]
            n_overlap = sum(int(g["production_bin"].eq(level).sum()) for _, g in cells)
            standardized = 0.
            for _, group in cells:
                sub = group.loc[group["production_bin"].eq(level)]
                standardized += len(group) / total_support * sub["onset"].mean()
            production.append({"period": period, "bin": level, "eligible": len(full), "onsets": int(full["onset"].sum()),
                               "raw_rate": full["onset"].mean(), "common_hours": n_overlap,
                               "common_cells": len(cells), "standardized_rate": standardized if total_support else np.nan,
                               "peak_q95": full["peak"].quantile(.95)})
    save(pd.DataFrame(production), PREFIX, "production_bins")
    contrasts = []
    weighted_correlations = []
    for period, part in data.groupby("period"):
        for variable in WEATHER:
            complete = part.dropna(subset=[variable]).copy()
            for key, group in complete.groupby(STRATA):
                if len(group) < 8:
                    continue
                if variable == "강수량":
                    low = group.loc[group[variable].eq(0)]
                    upper = group.loc[group[variable].gt(0)]
                else:
                    qlo, qhi = group[variable].quantile([.25, .75])
                    if qlo >= qhi:
                        continue
                    low = group.loc[group[variable].le(qlo)]
                    upper = group.loc[group[variable].ge(qhi)]
                if min(len(low), len(upper)) < 2:
                    continue
                contrasts.append({"period": period, "variable": variable, **dict(zip(STRATA, key)),
                                  "low_n": len(low), "high_n": len(upper),
                                  "peak_mean_difference": upper["peak"].mean() - low["peak"].mean(),
                                  "peak_q95_difference": upper["peak"].quantile(.95) - low["peak"].quantile(.95),
                                  "high_rate_difference": upper["high"].mean() - low["high"].mean()})
            for weighted in (False, True):
                complete["w"] = complete["profile_weight"] if weighted else 1.
                xmean = complete.groupby(STRATA).apply(lambda g: np.average(g[variable], weights=g["w"]), include_groups=False)
                ymean = complete.groupby(STRATA).apply(lambda g: np.average(g["peak"], weights=g["w"]), include_groups=False)
                ix = pd.MultiIndex.from_frame(complete[STRATA])
                x = complete[variable] - xmean.reindex(ix).to_numpy()
                y = complete["peak"] - ymean.reindex(ix).to_numpy()
                weighted_correlations.append({"period": period, "variable": variable, "profile_weighted": weighted,
                                              "rows": len(complete), "residual_correlation": correlation(x, y, complete["w"])})
    contrast_table = pd.DataFrame(contrasts)
    save(contrast_table, PREFIX, "weather_cells")
    contrast_summary = []
    for (period, variable, state), group in contrast_table.groupby(["period", "variable", "production_positive"]):
        w = group["low_n"] + group["high_n"]
        contrast_summary.append({"period": period, "variable": variable, "production_positive": state, "cells": len(group),
                                 "comparison_rows": int(w.sum()),
                                 "mean_difference": np.average(group["peak_mean_difference"], weights=w),
                                 "q95_difference": np.average(group["peak_q95_difference"], weights=w),
                                 "fraction_q95_positive_cells": group["peak_q95_difference"].gt(0).mean(),
                                 "high_rate_difference": np.average(group["high_rate_difference"], weights=w)})
    save(pd.DataFrame(contrast_summary), PREFIX, "weather_summary")
    save(pd.DataFrame(weighted_correlations), PREFIX, "profile_weather_sensitivity")
    residual_data = data.copy()
    for variable in ["peak", "생산량", *WEATHER]:
        residual_data[f"r_{variable}"] = data[variable] - data.groupby(STRATA)[variable].transform("mean")
    lag_rows = []
    for period, part in residual_data.groupby("period"):
        for variable in ["생산량", *WEATHER]:
            for lag in (0, 1, 3, 24, 168):
                past = residual_data[f"r_{variable}"].reindex(part.index - pd.Timedelta(hours=lag)).to_numpy()
                target = part["r_peak"].to_numpy()
                keep = np.isfinite(past) & np.isfinite(target)
                lag_rows.append({"period": period, "variable": variable, "lag_hours": lag, "pairs": int(keep.sum()),
                                 "residual_correlation": correlation(past[keep], target[keep], np.ones(keep.sum()))})
    save(pd.DataFrame(lag_rows), PREFIX, "residual_lags")
    finish(PREFIX, {"production_cuts": cuts.tolist(), "production": production,
                    "weather_summary": contrast_summary, "profile_correlations": weighted_correlations,
                    "largest_lag_abs": sorted(lag_rows, key=lambda r: abs(r["residual_correlation"]), reverse=True)[:6]})


if __name__ == "__main__":
    main()
