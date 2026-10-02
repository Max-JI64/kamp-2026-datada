"""Check whether opposite noon boundary changes occur on the same dates."""

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
PREFIX = "09.23_018"
EXPECTED_SHA256 = "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"
POSITIONS = ["11h_60m", "12h_15m", "12h_60m", "13h_15m"]


def save(frame: pd.DataFrame, suffix: str) -> None:
    frame.to_csv(TABLE_DIR / f"{PREFIX}_{suffix}.csv", index=False, encoding="utf-8-sig")


def describe(group: pd.DataFrame) -> dict:
    both = group.loc[group["down_then_up"]]
    return {
        "dates_or_profiles": len(group),
        "11_to_12_decrease": int(group["d_11_to_12"].lt(0).sum()),
        "11_to_12_equal": int(group["d_11_to_12"].eq(0).sum()),
        "11_to_12_increase": int(group["d_11_to_12"].gt(0).sum()),
        "12_to_13_decrease": int(group["d_12_to_13"].lt(0).sum()),
        "12_to_13_equal": int(group["d_12_to_13"].eq(0).sum()),
        "12_to_13_increase": int(group["d_12_to_13"].gt(0).sum()),
        "down_then_up": int(group["down_then_up"].sum()),
        "down_then_up_share": group["down_then_up"].mean(),
        "both_12_values_below_both_edges": int(group["both_midday_values_low"].sum()),
        "13_first_at_least_11_last_among_down_up": int(both["recovered_to_11_last"].sum()),
        "d_11_to_12_median": group["d_11_to_12"].median(),
        "d_12_inside_median": group["d_12_inside"].median(),
        "d_12_to_13_median": group["d_12_to_13"].median(),
        "d_11last_to_13first_median": group["d_11last_to_13first"].median(),
    }


def main() -> None:
    digest = sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256
    raw = pd.read_csv(SOURCE, encoding="utf-8-sig", usecols=["날짜", "시간", "15분", "60분"])
    assert len(raw) == 6168
    scope = raw.loc[raw["날짜"].lt(20210901)].copy()
    valid = scope.loc[scope["시간"].between(0, 23)].copy()
    assert (len(scope), len(valid)) == (5832, 5784)
    assert not valid.duplicated(["날짜", "시간"]).any()
    assert not valid[["15분", "60분"]].isna().any().any()
    assert valid.groupby("날짜")["시간"].nunique().eq(24).all()

    wide = valid.loc[valid["시간"].isin([11, 12, 13])].pivot(
        index="날짜", columns="시간", values=["15분", "60분"]
    )
    assert len(wide) == 241 and not wide.isna().any().any()
    days = pd.DataFrame({
        "date": wide.index,
        "11h_60m": wide[("60분", 11)].to_numpy(),
        "12h_15m": wide[("15분", 12)].to_numpy(),
        "12h_60m": wide[("60분", 12)].to_numpy(),
        "13h_15m": wide[("15분", 13)].to_numpy(),
    })
    days["month"] = days["date"] // 100 % 100
    days["d_11_to_12"] = days["12h_15m"] - days["11h_60m"]
    days["d_12_inside"] = days["12h_60m"] - days["12h_15m"]
    days["d_12_to_13"] = days["13h_15m"] - days["12h_60m"]
    days["d_11last_to_13first"] = days["13h_15m"] - days["11h_60m"]
    days["down_then_up"] = days["d_11_to_12"].lt(0) & days["d_12_to_13"].gt(0)
    days["both_midday_values_low"] = (
        days[["12h_15m", "12h_60m"]].max(axis=1)
        < days[["11h_60m", "13h_15m"]].min(axis=1)
    )
    assert (days.loc[days["both_midday_values_low"], "down_then_up"]).all()
    days["recovered_to_11_last"] = days["13h_15m"].ge(days["11h_60m"])
    days["first_boundary_sign"] = np.select(
        [days["d_11_to_12"].lt(0), days["d_11_to_12"].gt(0)],
        ["decrease", "increase"], default="equal",
    )
    days["second_boundary_sign"] = np.select(
        [days["d_12_to_13"].lt(0), days["d_12_to_13"].gt(0)],
        ["decrease", "increase"], default="equal",
    )

    prior = pd.read_csv(TABLE_DIR / "09.23_017_by_hour.csv", encoding="utf-8-sig")
    assert days["d_11_to_12"].median() == prior.loc[
        prior["시간"].eq(11), "boundary_next15_minus_60_median"
    ].iloc[0] == -50
    assert days["d_12_to_13"].median() == prior.loc[
        prior["시간"].eq(12), "boundary_next15_minus_60_median"
    ].iloc[0] == 35

    overview = pd.DataFrame([describe(days)])
    save(overview, "overview")
    grid = pd.crosstab(days["first_boundary_sign"], days["second_boundary_sign"])
    grid = grid.reindex(index=["decrease", "equal", "increase"],
                        columns=["decrease", "equal", "increase"], fill_value=0)
    grid = grid.rename_axis(index="11_to_12", columns="12_to_13").reset_index()
    save(grid, "sign_grid")
    assert int(grid[["decrease", "equal", "increase"]].to_numpy().sum()) == 241
    assert int(grid.loc[grid["11_to_12"].eq("decrease"), "increase"].iloc[0]) == int(
        days["down_then_up"].sum()
    )

    position_rows = []
    for position in POSITIONS:
        values = days[position]
        position_rows.append({
            "position": position, "days": len(values),
            "median": values.median(), "p10": values.quantile(0.10),
            "p90": values.quantile(0.90),
        })
    save(pd.DataFrame(position_rows), "position_distribution")

    by_month = pd.DataFrame([
        {"month": month, **describe(group)}
        for month, group in days.groupby("month", sort=True)
    ])
    save(by_month, "by_month")
    month_positions = pd.DataFrame([
        {"month": month, "position": position, "days": len(group),
         "median": group[position].median(),
         "p10": group[position].quantile(0.10),
         "p90": group[position].quantile(0.90)}
        for month, group in days.groupby("month", sort=True)
        for position in POSITIONS
    ])
    save(month_positions, "by_month_position")
    assert len(by_month) == 8 and int(by_month["dates_or_profiles"].sum()) == 241
    assert len(month_positions) == 32
    assert int(by_month["down_then_up"].sum()) == int(days["down_then_up"].sum())

    profiles = pd.read_csv(TABLE_DIR / "09.23_011_daily_profile_summary.csv", encoding="utf-8-sig")
    assert len(profiles) == 241
    days = days.merge(profiles[["date", "profile_id", "is_repeated"]], on="date", validate="one_to_one")
    assert days.groupby("profile_id")[POSITIONS].nunique().le(1).all().all()
    unique = days.drop_duplicates("profile_id")
    assert len(unique) == 126  # EDA 011
    weighting = pd.DataFrame([
        {"weighting": "observed_dates", **describe(days)},
        {"weighting": "unique_daily_power_profile", **describe(unique)},
        {"weighting": "dates_in_repeated_profiles", **describe(days.loc[days["is_repeated"]])},
        {"weighting": "dates_in_singleton_profiles", **describe(days.loc[~days["is_repeated"]])},
    ])
    save(weighting, "profile_weighting")
    assert int(weighting.loc[2, "dates_or_profiles"] + weighting.loc[3, "dates_or_profiles"]) == 241

    facts = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_sha256": digest, "pandas": pd.__version__, "numpy": np.__version__,
        "scope_rows": len(scope), "valid_rows": len(valid), "valid_days": len(days),
        "unique_profiles": len(unique),
        "overall": describe(days),
    }
    (TABLE_DIR / f"{PREFIX}_facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
