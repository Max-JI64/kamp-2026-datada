"""Exploratory change, periodic residual and counterexample diagnostics."""
import numpy as np
import pandas as pd
from analysis_common import load, save, finish

PREFIX = "09.27_006"


def main():
    _, data = load()
    daily = pd.read_csv("tables/09.27_004_daily.csv", encoding="utf-8-sig", parse_dates=["date"])
    candidates = []
    for metric in ("exact_daily_mean", "high_hours", "excess_sum"):
        x = daily[metric].to_numpy(float)
        total_sse = ((x - x.mean()) ** 2).sum()
        splits = []
        for k in range(28, len(x) - 27):
            before, after = x[:k], x[k:]
            sse = ((before - before.mean()) ** 2).sum() + ((after - after.mean()) ** 2).sum()
            splits.append({"metric": metric, "split_date": daily["date"].iloc[k], "before_days": k,
                           "after_days": len(x) - k, "before_mean": before.mean(), "after_mean": after.mean(),
                           "sse_reduction_fraction": 1 - sse / total_sse})
        best = pd.DataFrame(splits).nlargest(3, "sse_reduction_fraction")
        best["rank"] = np.arange(1, len(best) + 1)
        candidates.extend(best.to_dict("records"))
    save(pd.DataFrame(candidates), PREFIX, "change_candidates")
    periodic = []
    residual_rows = []
    try:
        from statsmodels.tsa.seasonal import STL
    except ModuleNotFoundError as exc:
        periodic.append({"status": "unavailable", "reason": str(exc)})
    else:
        for segment, group in data.groupby("segment"):
            for period in (24, 168):
                if len(group) < 2 * period:
                    periodic.append({"status": "insufficient_cycles", "segment": segment, "period_hours": period, "rows": len(group)})
                    continue
                result = STL(group["exact_mean"].to_numpy(), period=period, robust=True).fit()
                for observed_period, positions in group.groupby("period").indices.items():
                    resid = result.resid[positions]
                    y = group["exact_mean"].to_numpy()[positions]
                    periodic.append({"status": "fit", "segment": segment, "period_hours": period,
                                     "observed_period": observed_period, "rows": len(positions),
                                     "residual_median": np.median(resid), "residual_mad": np.median(np.abs(resid - np.median(resid))),
                                     "residual_abs_q95": np.quantile(np.abs(resid), .95),
                                     "residual_variance_ratio": np.var(resid) / np.var(y)})
                for timestamp, resid in zip(group.index, result.resid):
                    residual_rows.append({"timestamp": timestamp, "segment": segment, "period_hours": period, "residual": resid})
    save(pd.DataFrame(periodic), PREFIX, "periodic_summary")
    if residual_rows:
        save(pd.DataFrame(residual_rows), PREFIX, "periodic_residuals")
    keys = ["hour", "weekend", "production_positive"]
    reference = data.loc[data["period"].eq("Jan-Jun")].groupby(keys)["peak"].median()
    context = pd.MultiIndex.from_frame(data[keys])
    data["reference_peak"] = reference.reindex(context).to_numpy()
    data["residual_peak"] = data["peak"] - data["reference_peak"]
    supported = data.dropna(subset=["reference_peak"])
    anomaly_days = supported.groupby(["period", "date"]).agg(rows=("residual_peak", "size"), signed_mean=("residual_peak", "mean"), abs_mean=("residual_peak", lambda x: x.abs().mean()), q95_residual=("residual_peak", lambda x: x.quantile(.95))).reset_index()
    anomaly_days["abs_rank_within_period"] = anomaly_days.groupby("period")["abs_mean"].rank(method="min", ascending=False)
    save(anomaly_days, PREFIX, "anomaly_days")
    cases = []
    event_table = pd.read_csv("tables/09.27_004_events.csv", encoding="utf-8-sig", parse_dates=["start"])
    data["eligible"] = data["prev_peak"].notna() & data["prev_peak"].lt(182)
    data["onset"] = data["eligible"] & data["peak"].ge(182)
    for period, part in event_table.groupby("period"):
        for _, event in part.nlargest(5, "excess_sum").iterrows():
            row = data.loc[event["start"]]
            controls = data.loc[data["eligible"] & ~data["onset"] & data["month"].eq(row["month"]) & data["hour"].eq(row["hour"]) & data["weekend"].eq(row["weekend"]) & data["production_positive"].eq(row["production_positive"])].copy()
            result = {"period": period, "event_start": event["start"], "event_excess": event["excess_sum"], "event_hours": event["hours"],
                      "event_peak_at_start": row["peak"], "event_previous_peak": row["prev_peak"], "controls_available": len(controls)}
            if len(controls):
                distance = (controls["prev_peak"] - row["prev_peak"]).abs()
                control_key = distance.sort_values(kind="stable").index[0]
                control = controls.loc[control_key]
                result.update({"control_timestamp": control_key, "control_peak": control["peak"], "control_previous_peak": control["prev_peak"],
                               "previous_peak_distance": distance.loc[control_key], "same_profile": row["profile"] == control["profile"]})
            cases.append(result)
    save(pd.DataFrame(cases), PREFIX, "counterexamples")
    finish(PREFIX, {"change_best": [r for r in candidates if r["rank"] == 1], "periodic": periodic,
                    "unsupported_reference_hours": int(data["reference_peak"].isna().sum()),
                    "largest_anomalies": anomaly_days.nlargest(6, "abs_mean").to_dict("records"), "counterexamples": cases})


if __name__ == "__main__":
    main()
