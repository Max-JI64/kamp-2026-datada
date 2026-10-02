"""Compare same-date down-then-up boundary patterns across all three-hour windows."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd


EDA_DIR = Path(__file__).parent.parent
SOURCE = EDA_DIR / "../../data/origin/okm_augumented_2021.csv"
TABLE_DIR = EDA_DIR / "tables"
PREFIX = "09.23_019"
EXPECTED_SHA256 = "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"


def save(frame: pd.DataFrame, suffix: str) -> None:
    frame.to_csv(TABLE_DIR / f"{PREFIX}_{suffix}.csv", index=False, encoding="utf-8-sig")


def summarize(frame: pd.DataFrame) -> dict:
    d1 = frame["middle_first"] - frame["start_last"]
    d2 = frame["end_first"] - frame["middle_last"]
    down_up = d1.lt(0) & d2.gt(0)
    low_middle = (
        frame[["middle_first", "middle_last"]].max(axis=1)
        < frame[["start_last", "end_first"]].min(axis=1)
    )
    assert low_middle.le(down_up).all()
    return {
        "n": len(frame),
        "first_boundary_down": int(d1.lt(0).sum()),
        "second_boundary_up": int(d2.gt(0).sum()),
        "down_then_up": int(down_up.sum()),
        "down_then_up_share": down_up.mean(),
        "both_middle_values_below_both_edges": int(low_middle.sum()),
        "end_reaches_start_among_down_up": int(
            (down_up & frame["end_first"].ge(frame["start_last"])).sum()
        ),
        "first_boundary_difference_median": d1.median(),
        "middle_within_hour_difference_median": (
            frame["middle_last"] - frame["middle_first"]
        ).median(),
        "second_boundary_difference_median": d2.median(),
        "end_minus_start_median": (
            frame["end_first"] - frame["start_last"]
        ).median(),
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
    assert valid.groupby("날짜")["시간"].nunique().eq(24).all()
    assert not valid[["15분", "60분"]].isna().any().any()
    wide = valid.pivot(index="날짜", columns="시간", values=["15분", "60분"])
    assert len(wide) == 241 and not wide.isna().any().any()

    profiles = pd.read_csv(TABLE_DIR / "09.23_011_daily_profile_summary.csv", encoding="utf-8-sig")
    assert len(profiles) == 241 and profiles["date"].is_unique
    profile_map = profiles.set_index("date")["profile_id"]

    window_rows = []
    month_rows = []
    for start in range(22):
        frame = pd.DataFrame({
            "date": wide.index,
            "month": wide.index // 100 % 100,
            "profile_id": wide.index.map(profile_map),
            "start_last": wide[("60분", start)].to_numpy(),
            "middle_first": wide[("15분", start + 1)].to_numpy(),
            "middle_last": wide[("60분", start + 1)].to_numpy(),
            "end_first": wide[("15분", start + 2)].to_numpy(),
        })
        assert frame["profile_id"].notna().all()
        assert frame.groupby("profile_id")[
            ["start_last", "middle_first", "middle_last", "end_first"]
        ].nunique().le(1).all().all()
        for weighting, chosen in [
            ("observed_dates", frame),
            ("unique_daily_power_profile", frame.drop_duplicates("profile_id")),
        ]:
            window_rows.append({
                "start_hour": start, "window": f"{start:02d}-{start+1:02d}-{start+2:02d}",
                "weighting": weighting, **summarize(chosen),
            })
        for month, group in frame.groupby("month", sort=True):
            month_rows.append({
                "start_hour": start, "month": month,
                **summarize(group),
            })

    windows = pd.DataFrame(window_rows)
    windows["down_up_rank_within_weighting"] = (
        windows.groupby("weighting")["down_then_up"].rank(method="min", ascending=False).astype(int)
    )
    save(windows, "window_summary")
    months = pd.DataFrame(month_rows)
    save(months, "by_month_window")
    assert len(windows) == 44 and len(months) == 176
    assert windows.loc[windows["weighting"].eq("observed_dates"), "n"].eq(241).all()
    assert windows.loc[windows["weighting"].eq("unique_daily_power_profile"), "n"].eq(126).all()
    assert months.groupby("start_hour")["n"].sum().eq(241).all()
    assert months.groupby("start_hour")["down_then_up"].sum().eq(
        windows.loc[windows["weighting"].eq("observed_dates")].set_index("start_hour")["down_then_up"]
    ).all()

    prior = pd.read_csv(TABLE_DIR / "09.23_018_overview.csv", encoding="utf-8-sig").iloc[0]
    noon = windows.loc[
        windows["start_hour"].eq(11) & windows["weighting"].eq("observed_dates")
    ].iloc[0]
    assert int(noon["down_then_up"]) == int(prior["down_then_up"]) == 164
    assert int(noon["both_middle_values_below_both_edges"]) == int(
        prior["both_12_values_below_both_edges"]
    ) == 163
    assert int(noon["end_reaches_start_among_down_up"]) == int(
        prior["13_first_at_least_11_last_among_down_up"]
    ) == 59
    assert noon["first_boundary_difference_median"] == -50
    assert noon["second_boundary_difference_median"] == 35
    noon_unique = windows.loc[
        windows["start_hour"].eq(11) & windows["weighting"].eq("unique_daily_power_profile")
    ].iloc[0]
    assert int(noon_unique["down_then_up"]) == 84

    top = windows.loc[windows["weighting"].eq("observed_dates")].sort_values(
        ["down_then_up", "start_hour"], ascending=[False, True]
    ).head(5)
    facts = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_sha256": digest, "pandas": pd.__version__,
        "scope_rows": len(scope), "valid_rows": len(valid),
        "valid_days": len(wide), "windows_per_day": 22,
        "unique_profiles": profiles["profile_id"].nunique(),
        "noon_observed": noon.to_dict(), "noon_unique": noon_unique.to_dict(),
        "top_five_observed": top[["start_hour", "down_then_up", "both_middle_values_below_both_edges"]].to_dict(orient="records"),
    }
    (TABLE_DIR / f"{PREFIX}_facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
