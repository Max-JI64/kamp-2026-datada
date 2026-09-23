"""Compare the last power slot of a recorded hour with the next exact hour."""

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
PREFIX = "09.23_016"
EXPECTED_SHA256 = "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"
POWER_COLS = ["15분", "30분", "45분", "60분"]
HIGH_VALUE = 187  # EDA 004 and 015 descriptive cutoff, not an operating threshold.


def save(frame: pd.DataFrame, suffix: str) -> None:
    frame.to_csv(TABLE_DIR / f"{PREFIX}_{suffix}.csv", index=False, encoding="utf-8-sig")


def summarize_next(frame: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    chosen = frame.loc[mask]
    linked = chosen.loc[chosen["next_exists"]]
    return {
        "category": label,
        "candidate_hours": len(chosen),
        "next_exact_hour_pairs": len(linked),
        "without_next_exact_hour": len(chosen) - len(linked),
        "next_first_slot_high_pairs": int(linked["next_first_high"].sum()),
        "next_hour_any_high_pairs": int(linked["next_any_high"].sum()),
        "median_current_last_value": linked["60분"].median(),
        "median_next_first_value": linked["next_first_value"].median(),
        "median_boundary_difference": (linked["next_first_value"] - linked["60분"]).median(),
    }


def main() -> None:
    digest = sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256
    raw = pd.read_csv(
        SOURCE, encoding="utf-8-sig",
        usecols=["날짜", "시간", "생산량", *POWER_COLS],
    )
    assert len(raw) == 6168
    scope = raw.loc[raw["날짜"].lt(20210901)].copy()
    valid = scope.loc[scope["시간"].between(0, 23)].copy()
    assert (len(scope), len(valid)) == (5832, 5784)
    assert not valid.duplicated(["날짜", "시간"]).any()
    valid["timestamp"] = (
        pd.to_datetime(valid["날짜"].astype(str), format="%Y%m%d")
        + pd.to_timedelta(valid["시간"], unit="h")
    )
    valid = valid.sort_values("timestamp").reset_index(drop=True)
    flags = valid[POWER_COLS].to_numpy(dtype=np.int64) >= HIGH_VALUE
    valid["high_pattern"] = ["".join(map(str, row.astype(int))) for row in flags]
    valid["month"] = valid["날짜"] // 100 % 100
    valid["production_state"] = np.where(valid["생산량"].eq(0), "zero", "positive")

    lookup = valid.set_index("timestamp")[[*POWER_COLS, "high_pattern"]]
    next_rows = lookup.reindex(pd.DatetimeIndex(valid["timestamp"] + pd.Timedelta(hours=1)))
    previous_rows = lookup.reindex(pd.DatetimeIndex(valid["timestamp"] - pd.Timedelta(hours=1)))
    valid["next_first_value"] = next_rows["15분"].to_numpy()
    valid["next_pattern"] = next_rows["high_pattern"].to_numpy()
    valid["previous_last_value"] = previous_rows["60분"].to_numpy()
    valid["previous_pattern"] = previous_rows["high_pattern"].to_numpy()
    valid["next_exists"] = valid["next_first_value"].notna()
    valid["previous_exists"] = valid["previous_last_value"].notna()
    valid["next_first_high"] = valid["next_exists"] & valid["next_first_value"].ge(HIGH_VALUE)
    valid["next_any_high"] = valid["next_exists"] & valid["next_pattern"].ne("0000")
    valid["previous_last_high"] = valid["previous_exists"] & valid["previous_last_value"].ge(HIGH_VALUE)
    valid["previous_any_high"] = valid["previous_exists"] & valid["previous_pattern"].ne("0000")
    assert int(valid["next_exists"].sum()) == int(valid["previous_exists"].sum()) == 5781

    last_only = valid["high_pattern"].eq("0001")
    first_only = valid["high_pattern"].eq("1000")
    last_high = valid["60분"].ge(HIGH_VALUE)
    assert int(last_only.sum()) == 43  # EDA 015
    assert int(first_only.sum()) == 19
    assert int(last_high.sum()) == 194
    assert valid.loc[last_only, "시간"].lt(23).all()

    summary = pd.DataFrame([
        summarize_next(valid, pd.Series(True, index=valid.index), "all_valid_hours"),
        summarize_next(valid, valid["high_pattern"].ne("0000"), "any_high_current_hour"),
        summarize_next(valid, last_high, "last_slot_high_any"),
        summarize_next(valid, last_only, "last_slot_only_0001"),
        summarize_next(valid, last_high & ~last_only, "last_slot_high_with_other_slots"),
        summarize_next(valid, ~last_high, "last_slot_below_187"),
    ])
    summary["next_first_high_share_of_pairs"] = (
        summary["next_first_slot_high_pairs"] / summary["next_exact_hour_pairs"]
    )
    summary["next_any_high_share_of_pairs"] = (
        summary["next_hour_any_high_pairs"] / summary["next_exact_hour_pairs"]
    )
    save(summary, "next_boundary_summary")

    last_only_rows = valid.loc[last_only].copy()
    linked_last_only = last_only_rows.loc[last_only_rows["next_exists"]].copy()
    next_patterns = linked_last_only.groupby("next_pattern").agg(
        pairs=("날짜", "size"), days=("날짜", "nunique"),
        next_first_high_pairs=("next_first_high", "sum"),
    ).reset_index().sort_values(["pairs", "next_pattern"], ascending=[False, True])
    save(next_patterns, "last_only_next_patterns")

    def group_last_only(key: str, suffix: str) -> None:
        out = last_only_rows.groupby(key).agg(
            candidate_hours=("날짜", "size"), days=("날짜", "nunique"),
            next_exact_hour_pairs=("next_exists", "sum"),
            next_first_high_pairs=("next_first_high", "sum"),
            next_any_high_pairs=("next_any_high", "sum"),
            median_current_last_value=("60분", "median"),
            median_next_first_value=("next_first_value", "median"),
        ).reset_index()
        save(out, suffix)

    group_last_only("시간", "last_only_by_hour")
    group_last_only("month", "last_only_by_month")
    group_last_only("production_state", "last_only_by_production")

    first_only_rows = valid.loc[first_only].copy()
    first_only_summary = pd.DataFrame([{
        "candidate_hours": len(first_only_rows),
        "previous_exact_hour_pairs": int(first_only_rows["previous_exists"].sum()),
        "previous_last_high_pairs": int(first_only_rows["previous_last_high"].sum()),
        "previous_any_high_pairs": int(first_only_rows["previous_any_high"].sum()),
        "median_previous_last_value": first_only_rows.loc[
            first_only_rows["previous_exists"], "previous_last_value"
        ].median(),
        "median_current_first_value": first_only_rows.loc[
            first_only_rows["previous_exists"], "15분"
        ].median(),
    }])
    save(first_only_summary, "first_only_previous")

    profiles = pd.read_csv(TABLE_DIR / "09.23_011_daily_profile_summary.csv", encoding="utf-8-sig")
    assert len(profiles) == 241
    last_only_rows["profile_id"] = last_only_rows["날짜"].map(
        profiles.set_index("date")["profile_id"]
    )
    assert last_only_rows["profile_id"].notna().all()
    assert last_only_rows.groupby(["profile_id", "시간"])["next_first_value"].nunique().le(1).all()
    unique_pairs = last_only_rows.drop_duplicates(["profile_id", "시간"])
    profile_weighting = pd.DataFrame([
        {"weighting": "observed_hour_rows", "candidate_hours": len(last_only_rows),
         "next_exact_hour_pairs": int(last_only_rows["next_exists"].sum()),
         "next_first_high_pairs": int(last_only_rows["next_first_high"].sum()),
         "next_any_high_pairs": int(last_only_rows["next_any_high"].sum())},
        {"weighting": "unique_daily_profile_hour", "candidate_hours": len(unique_pairs),
         "next_exact_hour_pairs": int(unique_pairs["next_exists"].sum()),
         "next_first_high_pairs": int(unique_pairs["next_first_high"].sum()),
         "next_any_high_pairs": int(unique_pairs["next_any_high"].sum())},
    ])
    profile_weighting["next_first_high_share_of_pairs"] = (
        profile_weighting["next_first_high_pairs"] / profile_weighting["next_exact_hour_pairs"]
    )
    save(profile_weighting, "profile_weighting")

    assert int(next_patterns["pairs"].sum()) == len(linked_last_only)
    assert int(summary.loc[summary["category"].eq("last_slot_only_0001"), "candidate_hours"].iloc[0]) == 43
    assert int(profile_weighting.loc[0, "next_exact_hour_pairs"]) == len(linked_last_only)
    assert int(first_only_summary.loc[0, "candidate_hours"]) == 19

    facts = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_sha256": digest, "pandas": pd.__version__, "numpy": np.__version__,
        "scope_rows": len(scope), "valid_rows": len(valid),
        "exact_next_hour_pairs": int(valid["next_exists"].sum()),
        "high_value": HIGH_VALUE, "last_only_hours": len(last_only_rows),
        "last_only_exact_pairs": len(linked_last_only),
        "last_only_next_first_high_pairs": int(linked_last_only["next_first_high"].sum()),
        "last_only_next_any_high_pairs": int(linked_last_only["next_any_high"].sum()),
        "first_only_hours": len(first_only_rows),
        "first_only_previous_last_high_pairs": int(first_only_rows["previous_last_high"].sum()),
        "unique_profile_last_only_hours": len(unique_pairs),
    }
    (TABLE_DIR / f"{PREFIX}_facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
