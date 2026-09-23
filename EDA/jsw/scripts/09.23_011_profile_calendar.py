"""Describe exact daily power-profile repetition and its calendar context."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd


EDA_DIR = Path(__file__).parent.parent
SOURCE = EDA_DIR / "../../data/origin/okm_augumented_2021.csv"
TABLE_DIR = EDA_DIR / "tables"
PREFIX = "09.23_011"
EXPECTED_SHA256 = "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"
POWER_COLUMNS = ["15분", "30분", "45분", "60분"]
INPUT_COLUMNS = ["날짜", "시간", *POWER_COLUMNS, "생산량", "기온", "풍속", "습도", "강수량"]


def save(frame: pd.DataFrame, suffix: str) -> None:
    frame.to_csv(TABLE_DIR / f"{PREFIX}_{suffix}.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    actual_hash = sha256(SOURCE.read_bytes()).hexdigest()
    assert actual_hash == EXPECTED_SHA256, "원본 파일 해시가 이전 EDA와 다릅니다."
    raw = pd.read_csv(SOURCE, encoding="utf-8-sig", usecols=INPUT_COLUMNS)
    scope = raw.loc[raw["날짜"].lt(20210901)].copy()
    valid = scope.loc[scope["시간"].between(0, 23)].copy()
    assert (len(raw), len(scope), len(valid)) == (6168, 5832, 5784)
    assert not valid.duplicated(["날짜", "시간"]).any()
    assert valid.groupby("날짜").size().eq(24).all()

    # Each signature contains all four original quarter values at all 24 hours.
    wide = valid.pivot(index="날짜", columns="시간", values=POWER_COLUMNS)
    assert wide.shape == (241, 96) and not wide.isna().any().any()
    signature_to_id: dict[tuple[int, ...], str] = {}
    profile_ids = []
    for row in wide.to_numpy(dtype=np.int64):
        signature = tuple(int(value) for value in row)
        if signature not in signature_to_id:
            signature_to_id[signature] = f"P{len(signature_to_id) + 1:03d}"
        profile_ids.append(signature_to_id[signature])

    daily = valid.groupby("날짜").agg(
        production_total=("생산량", "sum"),
        temperature_mean=("기온", "mean"),
        wind_mean=("풍속", "mean"),
        humidity_mean=("습도", "mean"),
        precipitation_total=("강수량", "sum"),
    ).reset_index().rename(columns={"날짜": "date"})
    daily["profile_id"] = profile_ids
    dates = pd.to_datetime(daily["date"].astype(str), format="%Y%m%d")
    daily["month"] = dates.dt.month
    daily["day_of_month"] = dates.dt.day
    daily["weekday_mon1"] = dates.dt.dayofweek + 1
    sizes = daily.groupby("profile_id")["date"].transform("size")
    daily["profile_days"] = sizes
    daily["is_repeated"] = sizes.gt(1)
    months_per_profile = daily.groupby("profile_id")["month"].transform("nunique")
    daily["cross_month_profile"] = months_per_profile.gt(1)

    # One row per date is an aggregate of its 24 hours, not a transformed source file.
    save(daily, "daily_profile_summary")
    monthly = daily.groupby("month").agg(
        days=("date", "size"),
        distinct_profiles=("profile_id", "nunique"),
        repeated_days=("is_repeated", "sum"),
        cross_month_repeated_days=("cross_month_profile", "sum"),
    ).reset_index()
    monthly["repeated_day_share"] = monthly["repeated_days"] / monthly["days"]
    save(monthly, "monthly_repetition")
    daily["period"] = np.where(daily["month"].le(6), "Jan-Jun", "Jul-Aug")
    daily["production_state"] = np.where(daily["production_total"].eq(0), "zero", "positive")
    state_summary = daily.groupby(["period", "production_state"]).agg(
        days=("date", "size"),
        repeated_days=("is_repeated", "sum"),
        distinct_profiles=("profile_id", "nunique"),
    ).reset_index()
    state_summary["repeated_day_share"] = state_summary["repeated_days"] / state_summary["days"]
    save(state_summary, "production_state_repetition")

    production_signatures = {
        date: tuple(group.sort_values("시간")["생산량"].astype(int))
        for date, group in valid.groupby("날짜")
    }
    profile_rows = []
    for profile_id, group in daily.groupby("profile_id", sort=False):
        if len(group) < 2:
            continue
        date_values = group["date"].tolist()
        datetime_values = pd.to_datetime(group["date"].astype(str), format="%Y%m%d")
        gaps = datetime_values.diff().dropna().dt.days
        power_values = np.asarray(next(
            signature for signature, identifier in signature_to_id.items()
            if identifier == profile_id
        ))
        profile_rows.append({
            "profile_id": profile_id,
            "days": len(group),
            "dates": ";".join(str(date) for date in date_values),
            "months": ";".join(str(month) for month in sorted(group["month"].unique())),
            "distinct_months": group["month"].nunique(),
            "distinct_days_of_month": group["day_of_month"].nunique(),
            "distinct_weekdays": group["weekday_mon1"].nunique(),
            "min_gap_days": int(gaps.min()),
            "max_gap_days": int(gaps.max()),
            "power_min": int(power_values.min()),
            "power_max": int(power_values.max()),
            "production_total_min": int(group["production_total"].min()),
            "production_total_max": int(group["production_total"].max()),
            "distinct_production_totals": group["production_total"].nunique(),
            "distinct_hourly_production_profiles": len({production_signatures[date] for date in date_values}),
            "zero_production_days": int(group["production_total"].eq(0).sum()),
            "temperature_daily_mean_min": group["temperature_mean"].min(),
            "temperature_daily_mean_max": group["temperature_mean"].max(),
            "distinct_temperature_daily_means": group["temperature_mean"].nunique(),
        })
    profiles = pd.DataFrame(profile_rows).sort_values(["days", "profile_id"], ascending=[False, True])
    save(profiles, "repeated_profiles")

    # Report matched pairs with the number of *all eligible date pairs* beside them.
    pair_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    pair_rules = {
        "all_pairs": lambda a, b, gap: True,
        "same_month": lambda a, b, gap: a.month == b.month,
        "different_month": lambda a, b, gap: a.month != b.month,
        "same_day_of_month": lambda a, b, gap: a.day == b.day,
        "different_day_of_month": lambda a, b, gap: a.day != b.day,
        "different_month_same_day_of_month": lambda a, b, gap: a.month != b.month and a.day == b.day,
        "same_weekday": lambda a, b, gap: a.weekday() == b.weekday(),
        "different_weekday": lambda a, b, gap: a.weekday() != b.weekday(),
        "adjacent_days": lambda a, b, gap: gap == 1,
        "exactly_seven_days_apart": lambda a, b, gap: gap == 7,
        "gap_more_than_30_days": lambda a, b, gap: gap > 30,
    }
    rows = list(daily.itertuples(index=False))
    for left, right in combinations(rows, 2):
        a = pd.Timestamp(str(left.date))
        b = pd.Timestamp(str(right.date))
        gap = (b - a).days
        equal_power = left.profile_id == right.profile_id
        for name, rule in pair_rules.items():
            if rule(a, b, gap):
                pair_counts[name][0] += 1
                pair_counts[name][1] += int(equal_power)
    pairs = pd.DataFrame([
        {"relation": name, "all_date_pairs": pair_counts[name][0],
         "equal_power_profile_pairs": pair_counts[name][1],
         "equal_share_within_relation": pair_counts[name][1] / pair_counts[name][0]}
        for name in pair_rules
    ])
    save(pairs, "calendar_pair_summary")

    repeated_days = int(daily["is_repeated"].sum())
    assert len(signature_to_id) == 126
    assert repeated_days == 160 and len(profiles) == 45
    assert int(profiles["distinct_production_totals"].gt(1).sum()) == 40
    assert int(monthly["days"].sum()) == 241
    assert int(state_summary["days"].sum()) == 241
    assert pairs.loc[pairs["relation"].eq("all_pairs"), "all_date_pairs"].item() == 241 * 240 // 2
    facts = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_sha256": actual_hash,
        "scope_rows": len(scope),
        "valid_rows": len(valid),
        "complete_days": len(daily),
        "distinct_power_profiles": len(signature_to_id),
        "repeated_profiles": len(profiles),
        "repeated_days": repeated_days,
        "repeated_profiles_with_different_production_totals": int(profiles["distinct_production_totals"].gt(1).sum()),
        "repeated_profiles_with_different_hourly_production_profiles": int(profiles["distinct_hourly_production_profiles"].gt(1).sum()),
        "repeated_profiles_with_different_temperature_daily_means": int(profiles["distinct_temperature_daily_means"].gt(1).sum()),
        "same_month_equal_pairs": int(pairs.loc[pairs["relation"].eq("same_month"), "equal_power_profile_pairs"].item()),
        "different_month_equal_pairs": int(pairs.loc[pairs["relation"].eq("different_month"), "equal_power_profile_pairs"].item()),
        "different_month_same_day_of_month_equal_pairs": int(pairs.loc[pairs["relation"].eq("different_month_same_day_of_month"), "equal_power_profile_pairs"].item()),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    (TABLE_DIR / f"{PREFIX}_facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
