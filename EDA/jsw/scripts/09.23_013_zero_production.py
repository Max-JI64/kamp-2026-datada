"""Describe power recorded during zero-production hours without assigning an operational cause."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd


EDA_DIR = Path(__file__).parent.parent
SOURCE = EDA_DIR / "../../data/origin/okm_augumented_2021.csv"
TABLE_DIR = EDA_DIR / "tables"
PREFIX = "09.23_013"
EXPECTED_SHA256 = "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"
POWER_COLS = ["15분", "30분", "45분", "60분"]
QUANTILES = (0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1)


def save(frame: pd.DataFrame, suffix: str) -> None:
    frame.to_csv(TABLE_DIR / f"{PREFIX}_{suffix}.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    digest = sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256
    raw = pd.read_csv(SOURCE, encoding="utf-8-sig", usecols=["날짜", "시간", "생산량", "평균", *POWER_COLS])
    assert len(raw) == 6168
    assert raw["생산량"].eq(0).sum() == 2657
    assert (raw["생산량"].eq(0) & raw["평균"].gt(0)).sum() == 2640
    scope = raw.loc[raw["날짜"].lt(20210901)].copy()
    valid = scope.loc[scope["시간"].between(0, 23)].copy()
    assert (len(scope), len(valid)) == (5832, 5784)
    assert not valid.duplicated(["날짜", "시간"]).any()
    valid["timestamp"] = pd.to_datetime(valid["날짜"].astype(str), format="%Y%m%d") + pd.to_timedelta(valid["시간"], unit="h")
    valid = valid.sort_values("timestamp").reset_index(drop=True)
    valid["quarter_max"] = valid[POWER_COLS].max(axis=1)
    valid["zero_production"] = valid["생산량"].eq(0)
    valid["month"] = valid["timestamp"].dt.month
    valid["weekday_mon1"] = valid["timestamp"].dt.dayofweek + 1
    daily_production = valid.groupby("날짜")["생산량"].transform("sum")
    valid["whole_day_zero_production"] = daily_production.eq(0)
    profiles = pd.read_csv(TABLE_DIR / "09.23_011_daily_profile_summary.csv", encoding="utf-8-sig")
    assert len(profiles) == 241
    profile_map = profiles.set_index("date")["profile_id"]
    valid["profile_id"] = valid["날짜"].map(profile_map)
    assert valid["profile_id"].notna().all()

    # Exact temporal joins: never connect the July invalid-time days as adjacent hours.
    for hours, name in ((-1, "prev"), (1, "next")):
        lookup = valid.set_index("timestamp")[["생산량", "평균", "quarter_max"]]
        target_times = valid["timestamp"] + pd.to_timedelta(hours, unit="h")
        matched = lookup.reindex(pd.DatetimeIndex(target_times))
        for column, output in (("생산량", "production"), ("평균", "mean_power"), ("quarter_max", "quarter_max")):
            valid[f"{name}_{output}"] = matched[column].to_numpy()

    zero = valid.loc[valid["zero_production"]].copy()
    assert len(zero) == 2507
    zero["mean_band"] = pd.cut(
        zero["평균"], bins=[-1, 0, 40, 100, 160, float("inf")],
        labels=["0", "1-40", "41-100", "101-160", ">160"],
    )
    bands = zero.groupby("mean_band", observed=True).agg(
        rows=("평균", "size"), days=("날짜", "nunique"),
        median_mean_power=("평균", "median"),
        quarter_max_at_least_187=("quarter_max", lambda s: int(s.ge(187).sum())),
    ).reset_index()
    bands["share_of_zero_production_rows"] = bands["rows"] / len(zero)
    save(bands, "mean_power_bands")

    quantile_rows = []
    for label, part in (("zero_production", zero), ("positive_production", valid.loc[~valid["zero_production"]])):
        for col in ("평균", "quarter_max"):
            for q, value in part[col].quantile(list(QUANTILES)).items():
                quantile_rows.append({"production_state": label, "variable": col,
                                      "quantile": q, "value": value, "rows": len(part)})
    save(pd.DataFrame(quantile_rows), "quantiles")

    zero["high_quarter_187"] = zero["quarter_max"].ge(187)
    by_day_state = zero.groupby("whole_day_zero_production").agg(
        rows=("평균", "size"), days=("날짜", "nunique"),
        mean_power_median=("평균", "median"), mean_power_p90=("평균", lambda s: s.quantile(.9)),
        high_quarter_187_rows=("high_quarter_187", "sum"),
        mean_power_zero_rows=("평균", lambda s: int(s.eq(0).sum())),
    ).reset_index()
    save(by_day_state, "whole_day_context")

    def temporal_summary(keys: list[str], suffix: str) -> None:
        all_counts = valid.groupby(keys).agg(all_rows=("평균", "size"), all_days=("날짜", "nunique"))
        zero_counts = zero.groupby(keys).agg(
            zero_rows=("평균", "size"), zero_days=("날짜", "nunique"),
            zero_power_median=("평균", "median"), zero_power_p90=("평균", lambda s: s.quantile(.9)),
            high_quarter_187_rows=("high_quarter_187", "sum"),
        )
        out = all_counts.join(zero_counts, how="left").reset_index()
        out[["zero_rows", "zero_days", "high_quarter_187_rows"]] = out[["zero_rows", "zero_days", "high_quarter_187_rows"]].fillna(0).astype(int)
        out["zero_production_share"] = out["zero_rows"] / out["all_rows"]
        save(out, suffix)

    temporal_summary(["month"], "month_context")
    temporal_summary(["시간"], "hour_context")
    temporal_summary(["weekday_mon1"], "weekday_context")

    # Consecutive zero-production episodes, requiring an observed exact one-hour step.
    zero = zero.sort_values("timestamp").copy()
    new_episode = zero["timestamp"].diff().ne(pd.Timedelta(hours=1))
    zero["episode_id"] = new_episode.cumsum()
    zero["position_in_episode"] = zero.groupby("episode_id").cumcount() + 1
    zero["hours_until_episode_end"] = zero.groupby("episode_id")["timestamp"].transform("size") - zero["position_in_episode"] + 1
    zero["distance_to_nearest_positive_record"] = zero[["position_in_episode", "hours_until_episode_end"]].min(axis=1)
    episodes = zero.groupby("episode_id").agg(
        start=("timestamp", "first"), end=("timestamp", "last"), hours=("timestamp", "size"),
        days=("날짜", "nunique"), mean_power_median=("평균", "median"),
        mean_power_min=("평균", "min"), mean_power_max=("평균", "max"),
        quarter_max=("quarter_max", "max"), high_quarter_187_hours=("high_quarter_187", "sum"),
        power_zero_hours=("평균", lambda s: int(s.eq(0).sum())),
        first_date=("날짜", "first"), last_date=("날짜", "last"),
    ).reset_index()
    first = zero.drop_duplicates("episode_id", keep="first").set_index("episode_id")
    last = zero.drop_duplicates("episode_id", keep="last").set_index("episode_id")
    episodes["prev_production"] = episodes["episode_id"].map(first["prev_production"])
    episodes["prev_mean_power"] = episodes["episode_id"].map(first["prev_mean_power"])
    episodes["next_production"] = episodes["episode_id"].map(last["next_production"])
    episodes["next_mean_power"] = episodes["episode_id"].map(last["next_mean_power"])
    episodes["both_edges_positive_production"] = episodes["prev_production"].gt(0) & episodes["next_production"].gt(0)
    episodes["touches_high_quarter_187"] = episodes["high_quarter_187_hours"].gt(0)
    assert episodes["hours"].sum() == len(zero)
    assert episodes["start"].le(episodes["end"]).all()
    save(episodes, "zero_production_episodes")
    episode_summary = pd.DataFrame([{
        "episodes": len(episodes), "zero_production_hours": len(zero),
        "single_hour_episodes": int(episodes["hours"].eq(1).sum()),
        "episodes_at_least_24_hours": int(episodes["hours"].ge(24).sum()),
        "longest_hours": int(episodes["hours"].max()),
        "median_episode_hours": episodes["hours"].median(),
        "episodes_touching_high_quarter_187": int(episodes["touches_high_quarter_187"].sum()),
        "high_quarter_hours_in_zero_episodes": int(episodes["high_quarter_187_hours"].sum()),
        "episodes_with_positive_production_on_both_edges": int(episodes["both_edges_positive_production"].sum()),
    }])
    save(episode_summary, "episode_summary")

    high = zero.loc[zero["high_quarter_187"]].copy()
    high["prev_zero_production"] = high["prev_production"].eq(0).where(high["prev_production"].notna())
    high["next_zero_production"] = high["next_production"].eq(0).where(high["next_production"].notna())
    high["distance_band"] = pd.cut(
        high["distance_to_nearest_positive_record"], bins=[0, 1, 6, float("inf")],
        labels=["adjacent_1_hour", "2_to_6_hours", "at_least_7_hours"],
    )
    distance_summary = high.groupby("distance_band", observed=False).agg(
        high_rows=("날짜", "size"), days=("날짜", "nunique"),
    ).reset_index()
    assert int(distance_summary["high_rows"].sum()) == len(high)
    save(distance_summary, "high_distance_to_production")
    high_dates = high.groupby("날짜").agg(
        high_hours=("시간", "size"), hour_min=("시간", "min"), hour_max=("시간", "max"),
        peak_max=("quarter_max", "max"), mean_power_median=("평균", "median"),
        whole_day_zero_production=("whole_day_zero_production", "first"),
        profile_id=("profile_id", "first"),
        distinct_episodes=("episode_id", "nunique"),
        prev_positive_production_hours=("prev_production", lambda s: int(s.gt(0).sum())),
        next_positive_production_hours=("next_production", lambda s: int(s.gt(0).sum())),
        min_distance_to_positive_record=("distance_to_nearest_positive_record", "min"),
        max_distance_to_positive_record=("distance_to_nearest_positive_record", "max"),
    ).reset_index().rename(columns={"날짜": "date"})
    save(high_dates, "high_quarter_dates")
    high_episodes = episodes.loc[episodes["touches_high_quarter_187"]].copy()
    save(high_episodes, "high_quarter_episodes")

    counterpart_rows = []
    for row in high.itertuples(index=False):
        others = valid.loc[
            valid["profile_id"].eq(row.profile_id)
            & valid["시간"].eq(row.시간)
            & valid["날짜"].ne(row.날짜)
        ]
        assert len(others) == int(profiles.set_index("date").loc[row.날짜, "profile_days"]) - 1
        if not others.empty:
            reference_power = high.loc[
                high["날짜"].eq(row.날짜) & high["시간"].eq(row.시간), POWER_COLS
            ].iloc[0]
            assert others[POWER_COLS].eq(reference_power).all().all()
        counterpart_rows.append({
            "date": row.날짜, "hour": row.시간, "profile_id": row.profile_id,
            "other_dates": len(others),
            "other_dates_positive_production": int(others["생산량"].gt(0).sum()),
            "other_dates_zero_production": int(others["생산량"].eq(0).sum()),
        })
    counterparts = pd.DataFrame(counterpart_rows)
    by_date_counterparts = counterparts.groupby("date").agg(
        high_hours=("hour", "size"),
        high_hours_with_other_date=("other_dates", lambda s: int(s.gt(0).sum())),
        high_hours_with_positive_counterpart=("other_dates_positive_production", lambda s: int(s.gt(0).sum())),
        other_date_hour_pairs=("other_dates", "sum"),
        positive_other_date_hour_pairs=("other_dates_positive_production", "sum"),
    ).reset_index()
    assert by_date_counterparts["high_hours"].sum() == 28
    save(by_date_counterparts, "high_profile_counterparts")

    prior = pd.read_csv(TABLE_DIR / "09.23_004_zero_production_high_dates.csv", encoding="utf-8-sig")
    assert len(high) == 28 and len(high_dates) == 8 and len(prior) == 8
    prior_dates = pd.to_datetime(prior["date"]).dt.strftime("%Y%m%d").astype(int)
    assert set(high_dates["date"]) == set(prior_dates)
    assert high_dates["high_hours"].sum() == prior["rows"].sum() == 28
    prior_by_date = prior.assign(date=prior_dates).set_index("date")
    current_by_date = high_dates.set_index("date")
    for current_column, prior_column in (
        ("high_hours", "rows"), ("hour_min", "hour_min"),
        ("hour_max", "hour_max"), ("peak_max", "peak_max"),
    ):
        assert current_by_date[current_column].sort_index().equals(
            prior_by_date[prior_column].sort_index()
        )
    assert int(bands["rows"].sum()) == len(zero)
    assert int(by_day_state["rows"].sum()) == len(zero)

    facts = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_sha256": digest, "scope_rows": len(scope), "valid_rows": len(valid),
        "zero_production_rows": len(zero), "zero_production_days": zero["날짜"].nunique(),
        "zero_production_positive_mean_rows": int(zero["평균"].gt(0).sum()),
        "zero_production_zero_mean_rows": int(zero["평균"].eq(0).sum()),
        "high_quarter_187_rows": len(high), "high_quarter_187_days": len(high_dates),
        "episode_count": len(episodes), "longest_episode_hours": int(episodes["hours"].max()),
        "pandas": pd.__version__,
    }
    (TABLE_DIR / f"{PREFIX}_facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
