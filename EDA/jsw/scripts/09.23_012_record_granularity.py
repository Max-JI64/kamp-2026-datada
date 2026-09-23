"""Descriptive EDA of recording granularity and repeated-profile calendar groups."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import pandas as pd


EDA_DIR = Path(__file__).parent.parent
SOURCE = EDA_DIR / "../../data/origin/okm_augumented_2021.csv"
TABLE_DIR = EDA_DIR / "tables"
PREFIX = "09.23_012"
EXPECTED_SHA256 = "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"
POWER_COLUMNS = ["15분", "30분", "45분", "60분"]


def save(frame: pd.DataFrame, suffix: str) -> None:
    frame.to_csv(TABLE_DIR / f"{PREFIX}_{suffix}.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    actual_hash = sha256(SOURCE.read_bytes()).hexdigest()
    assert actual_hash == EXPECTED_SHA256, "원본 파일 해시가 이전 EDA와 다릅니다."
    raw = pd.read_csv(
        SOURCE, encoding="utf-8-sig", usecols=["날짜", "시간", "생산량", *POWER_COLUMNS]
    )
    scope = raw.loc[raw["날짜"].lt(20210901)].copy()
    invalid = scope.loc[~scope["시간"].between(0, 23)]
    df = scope.loc[scope["시간"].between(0, 23)].copy()
    assert (len(raw), len(scope), len(invalid), len(df)) == (6168, 5832, 48, 5784)
    assert invalid.groupby("날짜").size().to_dict() == {20210713: 24, 20210715: 24}
    assert not df.duplicated(["날짜", "시간"]).any()
    assert df.groupby("날짜").size().eq(24).all()

    df["month"] = df["날짜"] // 100 % 100
    df["period"] = df["month"].le(6).map({True: "Jan-Jun", False: "Jul-Aug"})
    production_totals = df.groupby("날짜")["생산량"].transform("sum")
    df["production_state"] = production_totals.eq(0).map({True: "zero", False: "positive"})
    df["within_hour_range"] = df[POWER_COLUMNS].max(axis=1) - df[POWER_COLUMNS].min(axis=1)
    df["hour_quarter_mean"] = df[POWER_COLUMNS].mean(axis=1)

    # Compare exact four-value records at the same clock hour, within one month.
    tuple_keys = ["month", "시간", *POWER_COLUMNS]
    state_tuple_keys = ["month", "시간", "production_state", *POWER_COLUMNS]
    df["hour_tuple_repeated_in_month"] = df.groupby(tuple_keys)["날짜"].transform("size").gt(1)
    df["hour_tuple_repeated_in_month_state"] = df.groupby(state_tuple_keys)["날짜"].transform("size").gt(1)

    # A slot is a specific hour and quarter position (e.g. 08:15), not any hour.
    long = df.melt(
        id_vars=["날짜", "시간", "month", "period", "production_state"],
        value_vars=POWER_COLUMNS,
        var_name="quarter_slot", value_name="power_value",
    )
    slot_keys = ["month", "시간", "quarter_slot", "power_value"]
    state_slot_keys = ["month", "시간", "production_state", "quarter_slot", "power_value"]
    long["quarter_value_repeated_in_month_slot"] = long.groupby(slot_keys)["날짜"].transform("size").gt(1)
    long["quarter_value_repeated_in_month_state_slot"] = long.groupby(state_slot_keys)["날짜"].transform("size").gt(1)

    daily = df.groupby("날짜").agg(
        month=("month", "first"),
        period=("period", "first"),
        production_state=("production_state", "first"),
        production_total=("생산량", "sum"),
        mean_quarter_power=("hour_quarter_mean", "mean"),
        median_hour_range=("within_hour_range", "median"),
        mean_hour_range=("within_hour_range", "mean"),
        hour_tuple_repeat_share=("hour_tuple_repeated_in_month", "mean"),
        hour_tuple_repeat_state_share=("hour_tuple_repeated_in_month_state", "mean"),
    )
    daily["distinct_quarter_values"] = long.groupby("날짜")["power_value"].nunique()
    daily["quarter_slot_repeat_share"] = long.groupby("날짜")["quarter_value_repeated_in_month_slot"].mean()
    daily["quarter_slot_repeat_state_share"] = long.groupby("날짜")["quarter_value_repeated_in_month_state_slot"].mean()

    old_daily = pd.read_csv(TABLE_DIR / "09.23_011_daily_profile_summary.csv", encoding="utf-8-sig")
    assert len(old_daily) == 241
    old_daily = old_daily.set_index("date")
    assert daily.index.equals(old_daily.index)
    assert daily["production_total"].equals(old_daily["production_total"])
    daily["full_day_profile_repeated"] = old_daily["is_repeated"].astype(bool)
    save(daily.reset_index().rename(columns={"날짜": "date"}), "daily_granularity")

    monthly = df.groupby("month").agg(
        hours=("시간", "size"),
        hour_tuple_repeat_share=("hour_tuple_repeated_in_month", "mean"),
        median_hour_range=("within_hour_range", "median"),
        p90_hour_range=("within_hour_range", lambda s: s.quantile(0.9)),
    )
    monthly["days"] = daily.groupby("month").size()
    monthly["distinct_quarter_values_pooled"] = long.groupby("month")["power_value"].nunique()
    monthly["quarter_slot_repeat_share"] = long.groupby("month")["quarter_value_repeated_in_month_slot"].mean()
    monthly["median_daily_distinct_quarter_values"] = daily.groupby("month")["distinct_quarter_values"].median()
    monthly["full_day_repeated_days"] = daily.groupby("month")["full_day_profile_repeated"].sum()
    assert monthly["days"].to_dict() == {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 29, 8: 31}
    save(monthly.reset_index(), "monthly_granularity")

    state = daily.groupby(["period", "production_state"]).agg(
        days=("month", "size"),
        full_day_repeated_days=("full_day_profile_repeated", "sum"),
        median_daily_distinct_quarter_values=("distinct_quarter_values", "median"),
        median_hour_range_across_days=("median_hour_range", "median"),
        mean_hour_tuple_repeat_share=("hour_tuple_repeat_share", "mean"),
        mean_hour_tuple_repeat_state_share=("hour_tuple_repeat_state_share", "mean"),
        mean_quarter_slot_repeat_share=("quarter_slot_repeat_share", "mean"),
        mean_quarter_slot_repeat_state_share=("quarter_slot_repeat_state_share", "mean"),
    ).reset_index()
    assert state["days"].sum() == 241
    save(state, "period_production_state")
    monthly_state = daily.groupby(["month", "production_state"]).agg(
        days=("period", "size"),
        median_daily_distinct_quarter_values=("distinct_quarter_values", "median"),
        median_hour_range_across_days=("median_hour_range", "median"),
        mean_hour_tuple_repeat_state_share=("hour_tuple_repeat_state_share", "mean"),
        mean_quarter_slot_repeat_state_share=("quarter_slot_repeat_state_share", "mean"),
    ).reset_index()
    assert monthly_state["days"].sum() == 241
    save(monthly_state, "monthly_production_state")

    boundary = daily.loc[(daily.index >= 20210624) & (daily.index <= 20210707)].reset_index()
    boundary = boundary.rename(columns={"날짜": "date"})
    assert len(boundary) == 14
    save(boundary, "june_july_boundary")

    # Reuse the 11th EDA's 45 repeated groups rather than rescanning the CSV.
    old_profiles = pd.read_csv(TABLE_DIR / "09.23_011_repeated_profiles.csv", encoding="utf-8-sig")
    assert len(old_profiles) == 45 and old_profiles["days"].sum() == 160
    calendar_rows = []
    month_pair_counts: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    for row in old_profiles.itertuples(index=False):
        dates = [pd.Timestamp(item) for item in str(row.dates).split(";")]
        counts = {"all": 0, "same_day_of_month": 0, "exact_seven_day_gap": 0, "adjacent_day": 0}
        for a, b in combinations(dates, 2):
            gap = (b - a).days
            counts["all"] += 1
            counts["same_day_of_month"] += int(a.month != b.month and a.day == b.day)
            counts["exact_seven_day_gap"] += int(gap == 7)
            counts["adjacent_day"] += int(gap == 1)
            month_pair = tuple(sorted((a.month, b.month)))
            month_pair_counts[month_pair][0] += 1
            month_pair_counts[month_pair][1] += int(a.month != b.month and a.day == b.day)
        calendar_rows.append({
            "profile_id": row.profile_id,
            "days": row.days,
            "dates": row.dates,
            "all_matched_pairs": counts["all"],
            "same_day_of_month_pairs": counts["same_day_of_month"],
            "exact_seven_day_gap_pairs": counts["exact_seven_day_gap"],
            "adjacent_day_pairs": counts["adjacent_day"],
        })
    calendar = pd.DataFrame(calendar_rows).sort_values(
        ["same_day_of_month_pairs", "all_matched_pairs"], ascending=[False, False]
    )
    assert calendar["all_matched_pairs"].sum() == 302
    assert calendar["same_day_of_month_pairs"].sum() == 102
    assert calendar["exact_seven_day_gap_pairs"].sum() == 22
    assert calendar["adjacent_day_pairs"].sum() == 15
    save(calendar, "calendar_group_contributions")
    month_pairs = pd.DataFrame([
        {"month_a": first, "month_b": second,
         "equal_power_profile_pairs": counts[0],
         "same_day_of_month_pairs": counts[1]}
        for (first, second), counts in sorted(month_pair_counts.items())
    ])
    assert month_pairs["equal_power_profile_pairs"].sum() == 302
    assert month_pairs["same_day_of_month_pairs"].sum() == 102
    save(month_pairs, "matched_month_pairs")

    largest = calendar.loc[calendar["days"].idxmax()]
    facts = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_sha256": actual_hash,
        "scope_rows": len(scope),
        "valid_rows": len(df),
        "complete_days": len(daily),
        "invalid_time_dates": {str(k): int(v) for k, v in invalid.groupby("날짜").size().items()},
        "calendar_groups": len(calendar),
        "groups_with_same_day_of_month_pair": int(calendar["same_day_of_month_pairs"].gt(0).sum()),
        "groups_with_exact_seven_day_gap_pair": int(calendar["exact_seven_day_gap_pairs"].gt(0).sum()),
        "groups_with_adjacent_day_pair": int(calendar["adjacent_day_pairs"].gt(0).sum()),
        "largest_profile_id": largest["profile_id"],
        "largest_profile_days": int(largest["days"]),
        "largest_group_all_pairs": int(largest["all_matched_pairs"]),
        "largest_group_same_day_of_month_pairs": int(largest["same_day_of_month_pairs"]),
        "pandas": pd.__version__,
    }
    (TABLE_DIR / f"{PREFIX}_facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
