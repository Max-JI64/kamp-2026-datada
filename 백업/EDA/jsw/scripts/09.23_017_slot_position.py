"""Describe power values within a recorded hour and across exact hour boundaries."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


EDA_DIR = Path(__file__).parent.parent
SOURCE = EDA_DIR / "../../data/origin/okm_augumented_2021.csv"
TABLE_DIR = EDA_DIR / "tables"
PREFIX = "09.23_017"
EXPECTED_SHA256 = "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"
POWER_COLS = ["15분", "30분", "45분", "60분"]
HIGH_VALUE = 187  # EDA 004/015 descriptive cutoff, not an operating threshold.


def save(frame: pd.DataFrame, suffix: str) -> None:
    frame.to_csv(TABLE_DIR / f"{PREFIX}_{suffix}.csv", index=False, encoding="utf-8-sig")


def difference_stats(values: pd.Series) -> dict:
    present = values.dropna()
    n = len(present)
    assert n > 0
    return {
        "n": n,
        "median": present.median(),
        "p10": present.quantile(0.10),
        "p90": present.quantile(0.90),
        "positive": int(present.gt(0).sum()),
        "negative": int(present.lt(0).sum()),
        "zero": int(present.eq(0).sum()),
        "positive_share": present.gt(0).mean(),
        "negative_share": present.lt(0).mean(),
    }


def context_summary(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    for label, group in frame.groupby(key, sort=True):
        net = difference_stats(group["net_60_minus_15"])
        boundary = difference_stats(group["boundary_next15_minus_60"])
        rows.append({
            key: label,
            "hours": len(group),
            "15분_median": group["15분"].median(),
            "30분_median": group["30분"].median(),
            "45분_median": group["45분"].median(),
            "60분_median": group["60분"].median(),
            "any_high_hours": int(group["any_high"].sum()),
            "net_60_minus_15_median": net["median"],
            "net_positive_hours": net["positive"],
            "net_negative_hours": net["negative"],
            "net_zero_hours": net["zero"],
            "net_positive_share": net["positive_share"],
            "exact_next_hour_pairs": boundary["n"],
            "boundary_next15_minus_60_median": boundary["median"],
            "boundary_positive_pairs": boundary["positive"],
            "boundary_negative_pairs": boundary["negative"],
            "boundary_zero_pairs": boundary["zero"],
            "boundary_negative_share": boundary["negative_share"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    digest = sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256
    raw = pd.read_csv(SOURCE, encoding="utf-8-sig", usecols=["날짜", "시간", *POWER_COLS])
    assert len(raw) == 6168
    scope = raw.loc[raw["날짜"].lt(20210901)].copy()
    valid = scope.loc[scope["시간"].between(0, 23)].copy()
    assert (len(scope), len(valid)) == (5832, 5784)
    assert not valid.duplicated(["날짜", "시간"]).any()
    assert not valid[POWER_COLS].isna().any().any()
    valid["timestamp"] = (
        pd.to_datetime(valid["날짜"].astype(str), format="%Y%m%d")
        + pd.to_timedelta(valid["시간"], unit="h")
    )
    valid = valid.sort_values("timestamp").reset_index(drop=True)
    valid["month"] = valid["날짜"] // 100 % 100
    valid["period"] = np.where(valid["month"].le(6), "1_6", "7_8")
    valid["any_high"] = valid[POWER_COLS].ge(HIGH_VALUE).any(axis=1)
    assert int(valid["any_high"].sum()) == 287
    assert valid[POWER_COLS].ge(HIGH_VALUE).sum().tolist() == [64, 153, 176, 194]

    valid["step_15_to_30"] = valid["30분"] - valid["15분"]
    valid["step_30_to_45"] = valid["45분"] - valid["30분"]
    valid["step_45_to_60"] = valid["60분"] - valid["45분"]
    valid["net_60_minus_15"] = valid["60분"] - valid["15분"]
    next_lookup = valid.set_index("timestamp")["15분"]
    next_first = next_lookup.reindex(
        pd.DatetimeIndex(valid["timestamp"] + pd.Timedelta(hours=1))
    )
    valid["next_first_value"] = next_first.to_numpy()
    valid["boundary_next15_minus_60"] = valid["next_first_value"] - valid["60분"]
    assert int(valid["next_first_value"].notna().sum()) == 5781  # EDA 016

    slot_rows = []
    for context, group in [
        ("all", valid), ("1_6", valid.loc[valid["period"].eq("1_6")]),
        ("7_8", valid.loc[valid["period"].eq("7_8")]),
        ("any_high_hour", valid.loc[valid["any_high"]]),
        ("no_high_hour", valid.loc[~valid["any_high"]]),
    ]:
        for slot in POWER_COLS:
            values = group[slot]
            slot_rows.append({
                "context": context, "slot": slot, "hours": len(group),
                "median": values.median(), "p10": values.quantile(0.10),
                "p90": values.quantile(0.90), "p95": values.quantile(0.95),
                "p99": values.quantile(0.99),
                "high_187_count": int(values.ge(HIGH_VALUE).sum()),
            })
    slots = pd.DataFrame(slot_rows)
    save(slots, "slot_distribution")

    steps = [
        ("15_to_30", "step_15_to_30"),
        ("30_to_45", "step_30_to_45"),
        ("45_to_60", "step_45_to_60"),
        ("15_to_60", "net_60_minus_15"),
        ("60_to_next_15", "boundary_next15_minus_60"),
    ]
    step_rows = []
    for context, group in [
        ("all", valid), ("1_6", valid.loc[valid["period"].eq("1_6")]),
        ("7_8", valid.loc[valid["period"].eq("7_8")]),
        ("any_high_hour", valid.loc[valid["any_high"]]),
        ("no_high_hour", valid.loc[~valid["any_high"]]),
    ]:
        for label, column in steps:
            step_rows.append({"context": context, "step": label,
                              **difference_stats(group[column])})
    step_summary = pd.DataFrame(step_rows)
    save(step_summary, "step_summary")
    assert int(step_summary.loc[
        step_summary["context"].eq("all") & step_summary["step"].eq("60_to_next_15"), "n"
    ].iloc[0]) == 5781

    by_hour = context_summary(valid, "시간")
    by_month = context_summary(valid, "month")
    save(by_hour, "by_hour")
    save(by_month, "by_month")
    by_hour_high_status = pd.concat([
        context_summary(valid.loc[~valid["any_high"]], "시간").assign(
            high_status="no_high_hour"
        ),
        context_summary(valid.loc[valid["any_high"]], "시간").assign(
            high_status="any_high_hour"
        ),
    ], ignore_index=True).sort_values(["시간", "high_status"])
    save(by_hour_high_status, "by_hour_high_status")
    assert len(by_hour) == 24 and by_hour["hours"].eq(241).all()
    assert len(by_month) == 8
    assert int(by_hour["any_high_hours"].sum()) == 287
    assert int(by_hour["exact_next_hour_pairs"].sum()) == 5781
    assert int(by_hour_high_status["hours"].sum()) == 5784

    profiles = pd.read_csv(TABLE_DIR / "09.23_011_daily_profile_summary.csv", encoding="utf-8-sig")
    assert len(profiles) == 241
    valid["profile_id"] = valid["날짜"].map(profiles.set_index("date")["profile_id"])
    assert valid["profile_id"].notna().all()
    assert valid.groupby(["profile_id", "시간"])[POWER_COLS].nunique().le(1).all().all()
    weighting_rows = []
    for context, group in [
        ("all", valid), ("1_6", valid.loc[valid["period"].eq("1_6")]),
        ("7_8", valid.loc[valid["period"].eq("7_8")]),
    ]:
        for weighting, weighted in [
            ("observed_hours", group),
            ("unique_profile_hour", group.drop_duplicates(["profile_id", "시간"])),
        ]:
            stats = difference_stats(weighted["net_60_minus_15"])
            weighting_rows.append({
                "context": context, "weighting": weighting,
                "hours": len(weighted), "high_15_count": int(weighted["15분"].ge(HIGH_VALUE).sum()),
                "high_60_count": int(weighted["60분"].ge(HIGH_VALUE).sum()),
                "net_60_minus_15_median": stats["median"],
                "net_positive_hours": stats["positive"],
                "net_negative_hours": stats["negative"],
                "net_zero_hours": stats["zero"],
                "net_positive_share": stats["positive_share"],
            })
    weighting = pd.DataFrame(weighting_rows)
    save(weighting, "profile_weighting")
    assert int(weighting.loc[
        weighting["context"].eq("all") & weighting["weighting"].eq("observed_hours"), "hours"
    ].iloc[0]) == 5784

    facts = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_sha256": digest, "pandas": pd.__version__, "numpy": np.__version__,
        "scope_rows": len(scope), "valid_rows": len(valid),
        "exact_next_hour_pairs": int(valid["next_first_value"].notna().sum()),
        "high_value": HIGH_VALUE, "any_high_hours": int(valid["any_high"].sum()),
        "slot_high_counts": valid[POWER_COLS].ge(HIGH_VALUE).sum().to_dict(),
        "overall_net": difference_stats(valid["net_60_minus_15"]),
        "overall_boundary": difference_stats(valid["boundary_next15_minus_60"]),
        "unique_profile_hours": int(weighting.loc[
            weighting["context"].eq("all") & weighting["weighting"].eq("unique_profile_hour"),
            "hours"
        ].iloc[0]),
    }
    (TABLE_DIR / f"{PREFIX}_facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
