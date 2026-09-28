"""Threshold, severity, concentration and recurrence analysis connected to 002."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from analysis_common import load, events, save, finish

PREFIX = "09.27_004"


def main():
    _, data = load()
    thresholds = {q: float(data.loc[data["period"].eq("Jan-Jun"), "peak"].quantile(q)) for q in (.90, .95, .975)}
    assert thresholds[.95] == 182
    summaries, hour_rates, severity, concentration, survival, recurrence = [], [], [], [], [], []
    for q, threshold in thresholds.items():
        high, eligible, onset, event_table = events(data, threshold)
        for period, part in data.groupby("period"):
            idx = part.index
            es = event_table.loc[event_table["period"].eq(period)]
            summaries.append({"quantile": q, "threshold": threshold, "period": period,
                              "valid_hours": len(part), "high_hours": int(high.loc[idx].sum()),
                              "eligible": int(eligible.loc[idx].sum()), "onsets": int(onset.loc[idx].sum()),
                              "onset_rate": onset.loc[idx].sum() / eligible.loc[idx].sum(),
                              "events": len(es), "multi_hour": int(es["hours"].gt(1).sum()),
                              "multi_hour_share": es["hours"].gt(1).mean(),
                              "left_censored": int(es["left_censored"].sum()),
                              "right_censored": int(es["right_censored"].sum()),
                              "cross_period": int(es["crosses_period"].sum())})
            for hour_name, mask in (("08", part["hour"].eq(8)), ("13", part["hour"].eq(13)), ("other", ~part["hour"].isin([8, 13]))):
                keys = part.index[mask]
                n = int(eligible.loc[keys].sum())
                hour_rates.append({"quantile": q, "threshold": threshold, "period": period, "hour_group": hour_name,
                                   "eligible": n, "onsets": int(onset.loc[keys].sum()),
                                   "rate": onset.loc[keys].sum() / n if n else np.nan})
            severity.append({"quantile": q, "threshold": threshold, "period": period,
                             "events": len(es), "median_hours": es["hours"].median(), "mean_hours": es["hours"].mean(),
                             "median_event_peak": es["peak_max"].median(), "max_event_peak": es["peak_max"].max(),
                             "median_excess_sum": es["excess_sum"].median(), "total_event_excess": es["excess_sum"].sum(),
                             "top5_event_excess_share": es["excess_sum"].nlargest(5).sum() / es["excess_sum"].sum()})
        if q != .95:
            continue
        old = pd.read_csv("tables/09.27_002_period_summary.csv", encoding="utf-8-sig").set_index("period")
        for period in old.index:
            keys = data.index[data["period"].eq(period)]
            assert int(onset.loc[keys].sum()) == int(old.loc[period, "onset_hours"])
            assert int(high.loc[keys].sum()) == int(old.loc[period, "high_hours"])
        save(event_table, PREFIX, "events")
        records = data[["date", "period", "hour", "weekday", "weekend", "peak", "exact_mean", "생산량", "profile", "profile_weight", "prev_peak", "prev_production", "segment"]].copy()
        records["eligible"] = eligible
        records["onset"] = onset
        records["high"] = high
        records["excess"] = (data["peak"] - threshold).clip(lower=0)
        save(records.reset_index(), PREFIX, "records")
        daily = records.groupby(["period", "date"]).agg(high_hours=("high", "sum"), onsets=("onset", "sum"), excess_sum=("excess", "sum"), daily_peak=("peak", "max"), mean_of_record_peaks=("peak", "mean"), exact_daily_mean=("exact_mean", "mean"), production_record_sum=("생산량", "sum")).reset_index()
        daily["week"] = daily["date"].dt.to_period("W-SUN").astype(str)
        save(daily, PREFIX, "daily")
        for period, part in daily.groupby("period"):
            for unit, table in (("date", part), ("week", part.groupby("week")[["high_hours", "onsets", "excess_sum"]].sum())):
                for metric in ("high_hours", "onsets", "excess_sum"):
                    values = table[metric]
                    for top in (1, 5):
                        concentration.append({"period": period, "unit": unit, "metric": metric, "top": top,
                                              "contributors": len(values), "total": values.sum(),
                                              "top_sum": values.nlargest(top).sum(), "share": values.nlargest(top).sum() / values.sum()})
        for (period, hour), part in event_table.groupby(["period", "hour"]):
            for length in range(1, int(part["hours"].max()) + 1):
                survival.append({"period": period, "start_hour": hour, "events": len(part), "duration_ge": length,
                                 "remaining": int(part["hours"].ge(length).sum()), "fraction": part["hours"].ge(length).mean()})
        previous = event_table.shift(1)
        valid_gaps = event_table["segment"].eq(previous["segment"])
        gaps = (event_table["start"] - previous["end"]).dt.total_seconds() / 3600 - 1
        event_table["nonhigh_hours_since_previous"] = gaps.where(valid_gaps)
        save(event_table, PREFIX, "events")
        for period, part in event_table.groupby("period"):
            gap = part["nonhigh_hours_since_previous"].dropna()
            recurrence.append({"period": period, "observable_gaps": len(gap), "median_nonhigh_hours": gap.median(),
                               "gaps_le_6": int(gap.le(6).sum()), "gaps_le_24": int(gap.le(24).sum()),
                               "max_nonhigh_hours": gap.max(), "multi_event_dates": int(part.groupby("date").size().gt(1).sum())})
    save(pd.DataFrame(summaries), PREFIX, "thresholds")
    save(pd.DataFrame(hour_rates), PREFIX, "hour_rates")
    save(pd.DataFrame(severity), PREFIX, "severity")
    save(pd.DataFrame(concentration), PREFIX, "concentration")
    save(pd.DataFrame(survival), PREFIX, "duration_by_start_hour")
    save(pd.DataFrame(recurrence), PREFIX, "recurrence")
    daily = pd.read_csv("tables/09.27_004_daily.csv", encoding="utf-8-sig")
    rank_rows = []
    for period, part in daily.groupby("period"):
        top_peak = set(part.nlargest(10, "daily_peak")["date"])
        top_mean = set(part.nlargest(10, "exact_daily_mean")["date"])
        rank_rows.append({"period": period, "days": len(part), "spearman_peak_vs_daily_mean": spearmanr(part["daily_peak"], part["exact_daily_mean"]).statistic,
                          "top10_common_dates": len(top_peak & top_mean)})
    save(pd.DataFrame(rank_rows), PREFIX, "metric_rank_sensitivity")
    finish(PREFIX, {"threshold_summary": summaries, "severity": severity,
                    "recurrence": recurrence, "metric_rank_sensitivity": rank_rows})


if __name__ == "__main__":
    main()
