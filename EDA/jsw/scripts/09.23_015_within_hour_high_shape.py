"""Explore the four recorded power values within each hour, without operational labels."""

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
PREFIX = "09.23_015"
EXPECTED_SHA256 = "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"
POWER_COLS = ["15분", "30분", "45분", "60분"]
HIGH_VALUE = 187  # Reuse the exploratory 1-8-month hourly-maximum threshold from EDA 004.


def save(frame: pd.DataFrame, suffix: str) -> None:
    frame.to_csv(TABLE_DIR / f"{PREFIX}_{suffix}.csv", index=False, encoding="utf-8-sig")


def longest_true_run(flags: np.ndarray) -> int:
    run = longest = 0
    for flag in flags:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return longest


def main() -> None:
    digest = sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256
    raw = pd.read_csv(
        SOURCE, encoding="utf-8-sig",
        usecols=["날짜", "시간", "생산량", "평균", *POWER_COLS],
    )
    assert len(raw) == 6168
    scope = raw.loc[raw["날짜"].lt(20210901)].copy()
    valid = scope.loc[scope["시간"].between(0, 23)].copy()
    assert (len(scope), len(valid)) == (5832, 5784)
    assert not valid.duplicated(["날짜", "시간"]).any()
    assert valid.groupby("날짜").size().eq(24).all()

    values = valid[POWER_COLS].to_numpy(dtype=np.int64)
    flags = values >= HIGH_VALUE
    valid["quarter_max"] = values.max(axis=1)
    valid["quarter_min"] = values.min(axis=1)
    valid["spread"] = valid["quarter_max"] - valid["quarter_min"]
    valid["max_mean_gap"] = valid["quarter_max"] - valid["평균"]
    valid["high_quarters"] = flags.sum(axis=1)
    valid["high_hour"] = valid["high_quarters"].gt(0)
    valid["high_pattern"] = ["".join(map(str, row.astype(int))) for row in flags]
    valid["longest_high_run"] = [longest_true_run(row) for row in flags]
    valid["production_state"] = np.where(valid["생산량"].eq(0), "zero", "positive")
    valid["month"] = valid["날짜"] // 100 % 100
    max_ties = (values == values.max(axis=1, keepdims=True)).sum(axis=1)
    first_max = values.argmax(axis=1)
    valid["max_position"] = np.where(
        max_ties == 1, np.array(POWER_COLS, dtype=object)[first_max], "tie"
    )
    assert (valid["평균"] == np.floor(values.mean(axis=1) + 0.5)).all()
    assert valid["max_mean_gap"].ge(0).all()
    assert valid["high_hour"].equals(valid["quarter_max"].ge(HIGH_VALUE))
    assert int(valid["high_quarters"].sum()) == int(flags.sum())

    profiles = pd.read_csv(TABLE_DIR / "09.23_011_daily_profile_summary.csv", encoding="utf-8-sig")
    assert len(profiles) == valid["날짜"].nunique() == 241
    valid["profile_id"] = valid["날짜"].map(profiles.set_index("date")["profile_id"])
    assert valid["profile_id"].notna().all()
    assert valid.groupby(["profile_id", "시간"])[POWER_COLS].nunique().le(1).all().all()

    prior_spread = pd.read_csv(TABLE_DIR / "09.23_004_within_hour_shape.csv", encoding="utf-8-sig")
    spread_row = prior_spread.set_index("metric").loc["spread"]
    for column, value in (("median", valid["spread"].median()),
                          ("p90", valid["spread"].quantile(.9)),
                          ("p99", valid["spread"].quantile(.99)),
                          ("max", valid["spread"].max()),
                          ("zeros", valid["spread"].eq(0).sum())):
        assert float(spread_row[column]) == float(value)

    high = valid.loc[valid["high_hour"]].copy()
    zero_high = high.loc[high["production_state"].eq("zero")]
    assert len(zero_high) == 28 and zero_high["날짜"].nunique() == 8  # EDA 013

    counts = high.groupby("high_quarters").agg(
        hours=("날짜", "size"), days=("날짜", "nunique"),
        zero_production_hours=("production_state", lambda s: int(s.eq("zero").sum())),
        median_max_mean_gap=("max_mean_gap", "median"),
        median_hour_mean=("평균", "median"),
        median_hour_max=("quarter_max", "median"),
    ).reset_index()
    counts["share_of_high_hours"] = counts["hours"] / len(high)
    save(counts, "high_quarter_counts")

    patterns = high.groupby(["high_pattern", "high_quarters", "longest_high_run"]).agg(
        hours=("날짜", "size"), days=("날짜", "nunique"),
        zero_production_hours=("production_state", lambda s: int(s.eq("zero").sum())),
    ).reset_index()
    pattern_profiles = high.groupby(["high_pattern", "high_quarters", "longest_high_run"]).apply(
        lambda part: part[["profile_id", "시간"]].drop_duplicates().shape[0],
        include_groups=False,
    ).rename("unique_profile_hours").reset_index()
    patterns = patterns.merge(
        pattern_profiles, on=["high_pattern", "high_quarters", "longest_high_run"], validate="one_to_one"
    )
    patterns["share_of_high_hours"] = patterns["hours"] / len(high)
    save(patterns.sort_values(["hours", "high_pattern"], ascending=[False, True]), "high_segment_patterns")
    run_summary = high.assign(
        slot_arrangement=np.where(
            high["longest_high_run"].eq(high["high_quarters"]), "consecutive", "separated"
        )
    ).groupby(["high_quarters", "slot_arrangement"]).agg(
        hours=("날짜", "size"), days=("날짜", "nunique")
    ).reset_index()
    save(run_summary, "high_run_summary")

    visibility = high.assign(mean_below_same_number=high["평균"].lt(HIGH_VALUE)).groupby(
        ["high_quarters", "mean_below_same_number"]
    ).agg(hours=("날짜", "size"), days=("날짜", "nunique"),
          median_max_mean_gap=("max_mean_gap", "median")).reset_index()
    save(visibility, "mean_visibility")

    max_positions = high.groupby("max_position").agg(
        hours=("날짜", "size"), days=("날짜", "nunique")
    ).reindex([*POWER_COLS, "tie"], fill_value=0).reset_index()
    max_positions["share_of_high_hours"] = max_positions["hours"] / len(high)
    save(max_positions, "max_positions")
    slot_hits = pd.DataFrame({
        "slot": POWER_COLS,
        "high_cells": flags.sum(axis=0),
        "all_hours": len(valid),
    })
    save(slot_hits, "slot_hits")
    slot_by_hour = pd.DataFrame({
        "hour": valid["시간"].to_numpy(),
        **{f"{slot}_high": flags[:, i].astype(int) for i, slot in enumerate(POWER_COLS)},
    }).groupby("hour").sum().reset_index()
    slot_by_hour["all_hours"] = valid.groupby("시간").size().reindex(slot_by_hour["hour"]).to_numpy()
    save(slot_by_hour, "slot_hits_by_hour")
    max_position_by_hour = high.pivot_table(
        index="시간", columns="max_position", values="날짜", aggfunc="size", fill_value=0
    ).reindex(columns=[*POWER_COLS, "tie"], fill_value=0).reset_index()
    max_position_by_hour["high_hours"] = max_position_by_hour[POWER_COLS + ["tie"]].sum(axis=1)
    save(max_position_by_hour, "max_position_by_hour")

    states = valid.groupby("production_state").agg(
        all_hours=("날짜", "size"), high_hours=("high_hour", "sum"),
        median_spread_all=("spread", "median"),
        median_max_mean_gap_all=("max_mean_gap", "median"),
    ).reset_index()
    state_high = high.groupby("production_state").agg(
        high_one_slot_hours=("high_quarters", lambda s: int(s.eq(1).sum())),
        high_multiple_slot_hours=("high_quarters", lambda s: int(s.ge(2).sum())),
        median_max_mean_gap_high=("max_mean_gap", "median"),
        mean_below_same_number_hours=("평균", lambda s: int(s.lt(HIGH_VALUE).sum())),
    ).reset_index()
    states = states.merge(state_high, on="production_state", validate="one_to_one")
    states["high_share_of_state_hours"] = states["high_hours"] / states["all_hours"]
    save(states, "production_context")

    def context(key: str, suffix: str) -> None:
        all_rows = valid.groupby(key).agg(all_hours=("날짜", "size"))
        high_rows = high.groupby(key).agg(
            high_hours=("날짜", "size"),
            one_slot_hours=("high_quarters", lambda s: int(s.eq(1).sum())),
            multiple_slot_hours=("high_quarters", lambda s: int(s.ge(2).sum())),
        )
        out = all_rows.join(high_rows, how="left").fillna(0).reset_index()
        for col in ("high_hours", "one_slot_hours", "multiple_slot_hours"):
            out[col] = out[col].astype(int)
        out["high_share_of_all_hours"] = out["high_hours"] / out["all_hours"]
        save(out, suffix)

    context("month", "month_context")
    context("시간", "hour_context")

    unique_profile_hours = valid.drop_duplicates(["profile_id", "시간"])
    unique_high = unique_profile_hours.loc[unique_profile_hours["high_hour"]]
    sensitivity = pd.DataFrame([
        {"weighting": "observed_hour_rows", "all_hours": len(valid), "high_hours": len(high),
         "one_slot_high_hours": int(high["high_quarters"].eq(1).sum()),
         "multiple_slot_high_hours": int(high["high_quarters"].ge(2).sum()),
         **{f"{slot}_high_cells": int(high[slot].ge(HIGH_VALUE).sum()) for slot in POWER_COLS}},
        {"weighting": "unique_daily_profile_hour", "all_hours": len(unique_profile_hours),
         "high_hours": len(unique_high),
         "one_slot_high_hours": int(unique_high["high_quarters"].eq(1).sum()),
         "multiple_slot_high_hours": int(unique_high["high_quarters"].ge(2).sum()),
         **{f"{slot}_high_cells": int(unique_high[slot].ge(HIGH_VALUE).sum()) for slot in POWER_COLS}},
    ])
    sensitivity["one_slot_share_of_high_hours"] = sensitivity["one_slot_high_hours"] / sensitivity["high_hours"]
    save(sensitivity, "profile_weighting")

    assert int(counts["hours"].sum()) == len(high)
    assert int(patterns["hours"].sum()) == len(high)
    assert int(run_summary["hours"].sum()) == len(high)
    assert int(visibility["hours"].sum()) == len(high)
    assert int(max_positions["hours"].sum()) == len(high)
    assert int(slot_hits["high_cells"].sum()) == int(high["high_quarters"].sum())
    assert int(slot_by_hour[[f"{slot}_high" for slot in POWER_COLS]].to_numpy().sum()) == int(flags.sum())
    assert int(max_position_by_hour["high_hours"].sum()) == len(high)
    assert int(states["all_hours"].sum()) == len(valid)
    assert int(states["high_hours"].sum()) == len(high)
    assert len(unique_profile_hours) == profiles["profile_id"].nunique() * 24

    facts = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_sha256": digest, "pandas": pd.__version__, "numpy": np.__version__,
        "scope_rows": len(scope), "valid_rows": len(valid), "days": valid["날짜"].nunique(),
        "high_value": HIGH_VALUE, "high_hours": len(high),
        "high_quarter_cells": int(flags.sum()),
        "one_slot_high_hours": int(high["high_quarters"].eq(1).sum()),
        "multiple_slot_high_hours": int(high["high_quarters"].ge(2).sum()),
        "multiple_slot_consecutive_high_hours": int((
            high["high_quarters"].ge(2) & high["longest_high_run"].eq(high["high_quarters"])
        ).sum()),
        "mean_below_same_number_high_hours": int(high["평균"].lt(HIGH_VALUE).sum()),
        "zero_production_high_hours": len(zero_high),
        "unique_daily_profiles": profiles["profile_id"].nunique(),
        "unique_profile_high_hours": len(unique_high),
    }
    (TABLE_DIR / f"{PREFIX}_facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
