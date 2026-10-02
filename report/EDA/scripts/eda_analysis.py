"""Report EDA reproduction from the supplied CSV. Regular CPython 3.13.

Run from any directory: python report/scripts/eda_analysis.py
No network, original-file writes, model training, or image reading.
Sections marked # %% can be converted to notebook cells later.
"""
# %% Imports and fixed definitions
from pathlib import Path
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import scipy
from scipy import stats

REPORT = Path(__file__).resolve().parents[1]
ROOT = REPORT.parent.parent
OUT = REPORT / "tables" / "eda"
SHA = "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"
SLOTS = ["15분", "30분", "45분", "60분"]
# Reuse E027's previously established calendar, without new external lookup.
HOLIDAYS = [20210101, 20210211, 20210212, 20210213, 20210301,
            20210505, 20210519, 20210606, 20210815, 20210816]
SEED, ROTATIONS, THRESHOLD = 20261002, 9999, 182.0


def save(frame, name):
    frame.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.minimum(1, np.maximum.accumulate(
        values[order] * (len(values) - np.arange(len(values)))))
    result = np.empty_like(values)
    result[order] = adjusted
    return result


# %% Raw data and reporting units
def prepare(source):
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SHA, "Source changed"
    raw = pd.read_csv(source, encoding="utf-8-sig")
    assert raw.shape == (6168, 18)
    scope = raw.loc[raw["날짜"].lt(20210901)].copy()
    frame = scope.loc[scope["시간"].between(0, 23)].copy()
    assert len(scope) == 5832 and len(frame) == 5784
    frame["timestamp"] = (pd.to_datetime(frame["날짜"].astype(str), format="%Y%m%d")
                          + pd.to_timedelta(frame["시간"], unit="h"))
    frame = frame.sort_values("timestamp").set_index("timestamp")
    assert frame.index.is_unique and frame.groupby("날짜").size().eq(24).all()
    assert not frame[SLOTS + ["평균", "생산량"]].isna().any().any()
    frame["M"] = frame["평균"]
    frame["A"] = frame[SLOTS].mean(axis=1)
    frame["P"] = frame[SLOTS].max(axis=1)
    np.testing.assert_array_equal(frame.M, np.floor(frame.A + 0.5))
    frame["month"] = frame.index.month
    frame["hour"] = frame.index.hour
    frame["weekday"] = frame.index.dayofweek
    frame["weekend"] = frame.weekday.ge(5)
    frame["producing"] = frame["생산량"].gt(0)
    frame["period"] = np.where(frame.month.le(6), "Jan-Jun", "Jul-Aug")
    # E004 defined its descriptive threshold before excluding invalid clock labels.
    # Those 48 rows have power observations, but are excluded from time-position plots.
    frame.attrs["descriptive_threshold"] = float(scope[SLOTS].max(axis=1).quantile(.95))
    daily = frame.groupby("날짜").agg(
        daily_mean=("M", "mean"), daily_peak=("P", "max"),
        producing_hours=("producing", "sum"), month=("month", "first"),
        weekend=("weekend", "first"))
    # Exact whole-day 96-value equality, not approximate clustering.
    profiles = frame.groupby("날짜")[SLOTS].apply(
        lambda x: tuple(x.to_numpy().ravel()))
    daily["profile"] = pd.factorize(profiles, sort=False)[0]
    daily["profile_days"] = daily.groupby("profile").profile.transform("size")
    daily["profile_weight"] = 1 / daily.profile_days
    daily["q10"] = frame.groupby("날짜")[SLOTS].apply(
        lambda x: np.quantile(x.to_numpy().ravel(), .10, method="linear"))
    daily["date"] = daily.index
    daily["holiday"] = daily.index.isin(HOLIDAYS)
    daily["group"] = 2 * daily.weekend.astype(int) + daily.holiday.astype(int)
    daily["period"] = np.where(daily.month.le(6), "Jan-Jun", "Jul-Aug")
    assert len(daily) == 241 and daily.profile.nunique() == 126
    assert daily.profile_days.gt(1).sum() == 160
    assert daily.holiday.sum() == 10
    save(frame.reset_index(), "hourly_records")
    save(daily.reset_index(drop=True), "daily_records")
    return frame, daily


# %% Topic 1: mean vs maximum; reproduce threshold provenance
def mean_peak(frame):
    grid = frame.M.reindex(pd.date_range("2021-01-01", "2021-08-31 23:00", freq="h"))
    eligible = grid.shift(1).rolling(24, min_periods=24).count().eq(24)
    for lag in (1, 2, 3, 24, 168):
        eligible &= grid.shift(lag).notna()
    eligible = eligible.reindex(frame.index)
    train = frame.loc[eligible & frame.month.le(6)]
    assert len(train) == 4176 and frame.loc[eligible].shape[0] == 5520
    assert train.P.quantile(.95) == THRESHOLD
    rows = []
    for period, part in frame.groupby("period", sort=True):
        for mean in ("M", "A"):
            high = part.P.ge(THRESHOLD)
            hidden = high & part[mean].lt(THRESHOLD)
            rows.append(dict(period=period, mean_definition=mean, normal_hours=len(part),
                             threshold=THRESHOLD, high_hours=int(high.sum()),
                             hidden_hours=int(hidden.sum()), hidden_rate=hidden.sum()/high.sum()))
    save(pd.DataFrame(rows), "mean_peak_visibility")
    examples = frame.loc[pd.to_datetime(["2021-01-30 01:00", "2021-01-11 12:00"]),
                         SLOTS + ["M", "A", "P"]]
    assert examples.A.eq(104).all()
    save(examples.reset_index(), "same_mean_examples")


# %% Topic 2 / appendix: calendar means, Welch-Holm and circular shifts
def calendar(daily):
    codes = daily.group.to_numpy()
    rng = np.random.default_rng(SEED)
    rotated = np.empty((ROTATIONS, len(daily)), dtype=np.int8)
    for month in range(1, 9):
        inds = np.flatnonzero(daily.month.to_numpy() == month)
        shifts = rng.integers(0, len(inds), ROTATIONS)
        rotated[:, inds] = codes[inds][(np.arange(len(inds))[None, :]
                                       - shifts[:, None]) % len(inds)]
    summaries, tests = [], []
    for contrast, mask, rmask, labels in [
        ("weekday_vs_weekend", codes >= 2, rotated >= 2, ["평일 전체", "주말 전체"]),
        ("nonholiday_vs_holiday", codes % 2 == 1, rotated % 2 == 1,
         ["비공휴일 전체", "공휴일 전체"])]:
        counts = [np.count_nonzero(~mask), np.count_nonzero(mask)]
        assert np.all(rmask.sum(axis=1) == counts[1])
        for metric in ("daily_mean", "daily_peak"):
            y = daily[metric].to_numpy()
            a, b = y[~mask], y[mask]
            result = stats.ttest_ind(a, b, equal_var=False)
            diff = b.mean() - a.mean()
            perm = (rmask*y).sum(axis=1)/counts[1] - ((~rmask)*y).sum(axis=1)/counts[0]
            tests.append(dict(contrast=contrast, metric=metric, days_a=counts[0],
                              days_b=counts[1], difference_b_minus_a=diff,
                              t=result.statistic, p_raw=result.pvalue,
                              rotation_p=(1+np.count_nonzero(np.abs(perm) >= abs(diff)))/(ROTATIONS+1)))
            for flag, label in enumerate(labels):
                part = daily.loc[mask if flag else ~mask]
                summaries.append(dict(contrast=contrast, group=label, metric=metric,
                                      days=len(part), mean=part[metric].mean(),
                                      median=part[metric].median(),
                                      profile_weighted_mean=np.average(part[metric], weights=part.profile_weight),
                                      producing_hour_share=part.producing_hours.sum()/(24*len(part))))
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests.p_raw)
    tests["rotation_p_holm"] = holm(tests.rotation_p)
    save(tests, "calendar_tests")
    save(pd.DataFrame(summaries), "calendar_summary")


# %% Topic 3: hourly high-value frequencies vs monthly maximum locations
def hour_patterns(frame):
    long = frame.reset_index().melt(id_vars=["timestamp", "날짜", "hour", "month", "period"],
                                    value_vars=SLOTS, var_name="slot", value_name="power")
    ties = long.loc[long.power.eq(long.groupby("month").power.transform("max"))].copy()
    save(ties, "monthly_maxima")
    rows = []
    for (period, hour), part in frame.groupby(["period", "hour"]):
        n = len(part)
        high = int(part.P.ge(THRESHOLD).sum())
        rows.append(dict(period=period, hour=hour, valid_hours=n,
                         high_hours_ge_182=high, high_rate=high/n,
                         monthly_max_tied_slots=int(((ties.period == period) & (ties.hour == hour)).sum())))
    save(pd.DataFrame(rows), "hour_patterns")


# %% Topic 4: same-sample correlation before/after group centering
def conditional_relations(frame):
    keys = ["month", "hour", "weekend", "producing"]
    summaries, points = [], []
    for population in ("all", "positive"):
        base = frame if population == "all" else frame.loc[frame.producing]
        for variable in ("생산량", "기온", "풍속", "습도", "강수량"):
            part = base[[variable, "M", *keys]].dropna().copy()
            sizes = part.groupby(keys).M.transform("size")
            kept = part.loc[sizes.ge(5)]
            centered = kept[[variable, "M"]] - kept.groupby(keys)[[variable, "M"]].transform("mean")
            summaries.append(dict(population=population, variable=variable, eligible_n=len(part),
                                  n=len(kept), groups=kept.groupby(keys).ngroups,
                                  raw_same_sample=kept[variable].corr(kept.M),
                                  raw_spearman=kept[variable].corr(kept.M, method="spearman"),
                                  centered_pearson=centered[variable].corr(centered.M),
                                  centered_spearman=centered[variable].corr(centered.M, method="spearman")))
            if population == "positive":
                points.append(pd.DataFrame(dict(timestamp=kept.index, variable=variable,
                                                x=kept[variable].to_numpy(), y=kept.M.to_numpy(),
                                                x_centered=centered[variable].to_numpy(),
                                                y_centered=centered.M.to_numpy())))
    save(pd.DataFrame(summaries), "conditional_correlations")
    save(pd.concat(points, ignore_index=True), "conditional_points")


# %% Topic 5: exact-hour lag correlations, including monthly instability
def lags(frame):
    rows = []
    for lag in (1, 24, 168):
        prior = frame.M.reindex(frame.index - pd.Timedelta(hours=lag))
        pairs = pd.DataFrame(dict(current=frame.M.to_numpy(), prior=prior.to_numpy(),
                                  month=frame.month.to_numpy()), index=frame.index).dropna()
        for month in [0, *range(1, 9)]:
            part = pairs if month == 0 else pairs.loc[pairs.month.eq(month)]
            rows.append(dict(month=month, lag_hours=lag, n=len(part),
                             pearson=part.current.corr(part.prior)))
    save(pd.DataFrame(rows), "lag_correlations")


# %% Topic 6: daily lower-tail level, not physical base load
def low_load(daily):
    rows = []
    for month, part in daily.groupby("month"):
        unique = part.drop_duplicates("profile")
        rows.append(dict(month=month, days=len(part), unique_power_profiles=len(unique),
                         q10_min=part.q10.min(), q10_q25=part.q10.quantile(.25),
                         q10_median=part.q10.median(), q10_q75=part.q10.quantile(.75),
                         q10_max=part.q10.max(), q10_unique_profile_median=unique.q10.median()))
    save(pd.DataFrame(rows), "low_load_monthly")


# %% Revised EDA: whole-period patterns, correlations and within-hour shape
def narrative_tables(frame, daily):
    """Fixed 1-8 month scope. Descriptive EDA threshold, not model threshold."""
    threshold = frame.attrs["descriptive_threshold"]
    assert threshold == 187
    overview = []
    for population, base in [("all", frame), ("weekday", frame.loc[~frame.weekend]),
                             ("weekend", frame.loc[frame.weekend])]:
        for hour, part in base.groupby("hour"):
            row = dict(population=population, hour=hour, n=len(part))
            for variable, label in [("M", "power"), ("생산량", "production")]:
                row.update({f"{label}_mean": part[variable].mean(),
                            f"{label}_median": part[variable].median(),
                            f"{label}_q25": part[variable].quantile(.25),
                            f"{label}_q75": part[variable].quantile(.75)})
            overview.append(row)
    save(pd.DataFrame(overview), "hourly_overview")
    monthly, frequency = [], []
    for (month, hour), part in frame.groupby(["month", "hour"]):
        monthly.append(dict(month=month, hour=hour, n=len(part),
                            power_mean=part.M.mean(), production_mean=part['생산량'].mean(),
                            high_hours=int(part.P.ge(threshold).sum()),
                            high_rate=part.P.ge(threshold).mean()))
    for hour, part in frame.groupby("hour"):
        frequency.append(dict(hour=hour, n=len(part), high_hours=int(part.P.ge(threshold).sum()),
                              high_rate=part.P.ge(threshold).mean()))
    save(pd.DataFrame(monthly), "monthly_hour_overview")
    save(pd.DataFrame(frequency), "hour_frequency_187")
    variables = ["M", "P", "생산량", "기온", "풍속", "습도", "강수량"]
    correlations = []
    for x in variables:
        for y in variables:
            part = frame[[x, y]].dropna() if x != y else frame[[x]].dropna()
            correlations.append(dict(x=x, y=y, n=len(part),
                                     pearson=part[x].corr(part[y]) if x != y else 1.0,
                                     spearman=part[x].corr(part[y], method="spearman") if x != y else 1.0))
    save(pd.DataFrame(correlations), "overall_correlations")
    high = frame.loc[frame.P.ge(threshold)].copy()
    hits = high[SLOTS].ge(threshold)
    high["high_slots"] = hits.sum(axis=1)
    high["pattern"] = hits.astype(int).astype(str).agg("".join, axis=1)
    high["profile"] = high['날짜'].map(daily.profile)
    ordered = high.sort_values(["high_slots", "pattern", "P", "M", "날짜", "hour"])
    save(ordered.reset_index()[["timestamp", "날짜", "hour", *SLOTS, "M", "A", "P",
                               "high_slots", "pattern", "profile"]], "high_slot_records")
    summary = high.groupby("high_slots").agg(hours=("P", "size"), mean_power=("M", "mean"),
                                               median_peak=("P", "median")).reset_index()
    summary["share"] = summary.hours / len(high)
    save(summary, "high_slot_summary")
    patterns = high.groupby(["high_slots", "pattern"]).size().rename("hours").reset_index()
    save(patterns, "high_slot_patterns")
    unique = high.drop_duplicates(["profile", "hour"])
    facts = dict(scope="2021-01-01 to 2021-08-31, valid hours only", rows=len(frame), days=len(daily),
                 threshold=threshold, threshold_basis="E004 Jan-Aug 5832 rows including 48 invalid clock rows; row maximum P, linear 95th percentile",
                 threshold_source_rows=5832, valid_hour_p95=float(frame.P.quantile(.95)),
                 high_hours=len(high), multiple_slot_hours=int(high.high_slots.ge(2).sum()),
                 hidden_mean_hours=int(high.M.lt(threshold).sum()),
                 unique_profile_high_hours=len(unique),
                 unique_profile_multiple_slot_hours=int(unique.high_slots.ge(2).sum()),
                 source_sha256=SHA, image_analysis_performed=False,
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 run_at=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"))
    (OUT/"narrative_manifest.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")


# %% Entry point and provenance (all numerical outputs under report)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"data/origin/okm_augumented_2021.csv")
    parser.add_argument("--refresh-visuals", action="store_true",
                        help="Refresh revised narrative tables only; reuse calendar tests, lag and low-load tables")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frame, daily = prepare(args.source)
    if args.refresh_visuals:
        conditional_relations(frame)
        narrative_tables(frame, daily)
        from eda_external_conditions import analyze
        analyze(frame, daily)
        print(json.dumps(dict(status="revised_tables_completed", rows=len(frame), days=len(daily)), ensure_ascii=False))
        return
    mean_peak(frame)
    calendar(daily)
    hour_patterns(frame)
    conditional_relations(frame)
    lags(frame)
    low_load(daily)
    narrative_tables(frame, daily)
    from eda_external_conditions import analyze
    analyze(frame, daily)
    assert hashlib.sha256(args.source.read_bytes()).hexdigest() == SHA
    facts = dict(status="completed", source_sha256=SHA,
                 run_at=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"),
                 python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
                 scipy=scipy.__version__, rows=len(frame), days=len(daily),
                 threshold=THRESHOLD, threshold_basis="M001 Jan-Jun eligible 4176 rows, P 95th percentile",
                 holiday_dates=HOLIDAYS, holiday_source="Existing E027 definitions; no new web data",
                 seed=SEED, rotations=ROTATIONS,
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 image_analysis_performed=False)
    (OUT/"run_manifest.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(status="completed", rows=len(frame), days=len(daily),
                         tables=len(list(OUT.glob('*.csv')))), ensure_ascii=False))


if __name__ == "__main__":
    main()
