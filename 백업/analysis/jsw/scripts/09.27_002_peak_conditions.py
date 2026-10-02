"""Describe high-power episodes and conditional onset rates without model fitting."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd


BASE = Path(__file__).parent.parent
SOURCE = BASE / "../../data/origin/okm_augumented_2021.csv"
PROFILES = BASE / "../../EDA/jsw/tables/09.23_011_daily_profile_summary.csv"
FROZEN = BASE / "tables/09.27_002_frozen.json"
TABLES = BASE / "tables"
PREFIX = "09.27_002"


def save(frame: pd.DataFrame, label: str) -> None:
    frame.to_csv(TABLES / f"{PREFIX}_{label}.csv", index=False, encoding="utf-8-sig")


def rate_row(frame: pd.DataFrame, columns: dict, weighted: bool = False) -> dict:
    eligible = frame.loc[frame["eligible"]]
    onsets = int(eligible["onset"].sum())
    row = {**columns, "eligible_hours": len(eligible), "onset_hours": onsets,
           "onset_rate": onsets / len(eligible) if len(eligible) else np.nan}
    if weighted:
        exposure = float(eligible["profile_weight"].sum())
        events = float(eligible.loc[eligible["onset"], "profile_weight"].sum())
        row.update({"profile_weighted_exposure": exposure,
                    "profile_weighted_onsets": events,
                    "profile_weighted_rate": events / exposure if exposure else np.nan})
    return row


def main() -> None:
    frozen_bytes = FROZEN.read_bytes()
    frozen = json.loads(frozen_bytes)
    threshold = 182
    assert "182" in frozen["high_definition"]
    digest = sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == frozen["source_sha256"]

    slots = ["15분", "30분", "45분", "60분"]
    raw = pd.read_csv(SOURCE, encoding="utf-8-sig", usecols=["날짜", "시간", *slots, "생산량"])
    assert len(raw) == 6168
    valid = raw.loc[raw["날짜"].lt(20210901) & raw["시간"].between(0, 23)].copy()
    assert len(valid) == 5784
    valid["date"] = pd.to_datetime(valid["날짜"].astype(str), format="%Y%m%d")
    valid["record_key"] = valid["date"] + pd.to_timedelta(valid["시간"], unit="h")
    valid = valid.set_index("record_key").sort_index()
    assert valid.index.is_unique and valid.groupby("date").size().eq(24).all()
    assert not valid[[*slots, "생산량"]].isna().any().any()
    valid["peak"] = valid[slots].max(axis=1)
    valid["high"] = valid["peak"].ge(threshold)
    valid["period"] = np.where(valid.index < pd.Timestamp("2021-07-01"), "Jan-Jun", "Jul-Aug")
    valid["hour_group"] = np.select([valid.index.hour == 8, valid.index.hour == 13], ["08", "13"], default="other")

    # Reindex by exact timestamps; neither July invalid-hour date can bridge a gap.
    prev = valid["peak"].reindex(valid.index - pd.Timedelta(hours=1)).to_numpy()
    prev_prod = valid["생산량"].reindex(valid.index - pd.Timedelta(hours=1)).to_numpy()
    valid["previous_known"] = np.isfinite(prev)
    valid["eligible"] = valid["previous_known"] & (prev < threshold)
    valid["onset"] = valid["eligible"] & valid["high"]
    valid["previous_production"] = prev_prod
    valid["target_production_state"] = np.where(valid["생산량"].eq(0), "zero", "positive")
    valid["previous_production_state"] = np.where(valid["previous_production"].eq(0), "zero", "positive")
    assert valid.loc[valid["eligible"], "previous_production"].notna().all()

    # Independent reconciliation to EDA 008's exact three-hour eligibility.
    past_three = np.column_stack([
        valid["peak"].reindex(valid.index - pd.Timedelta(hours=lag)).to_numpy()
        for lag in (1, 2, 3)
    ])
    eligible_three = np.isfinite(past_three).all(axis=1) & (past_three[:, 0] < threshold)
    checks = {}
    for period, expected_n, expected_onsets in (("Jan-Jun", 4106, 139), ("Jul-Aug", 1214, 85)):
        current = valid["period"].eq(period).to_numpy()
        n = int((current & eligible_three).sum())
        events = int((current & eligible_three & valid["high"].to_numpy()).sum())
        assert (n, events) == (expected_n, expected_onsets)
        checks[period] = {"eda008_eligible": n, "eda008_onsets": events}

    profiles = pd.read_csv(PROFILES, encoding="utf-8-sig")
    profiles["date"] = pd.to_datetime(profiles["date"].astype(str), format="%Y%m%d")
    assert len(profiles) == 241 and profiles["date"].is_unique
    profiles["period"] = np.where(profiles["date"].lt(pd.Timestamp("2021-07-01")), "Jan-Jun", "Jul-Aug")
    profiles["profile_days_in_period"] = profiles.groupby(["period", "profile_id"])["date"].transform("size")
    profile_map = profiles.set_index("date")[["profile_id", "profile_days_in_period"]]
    valid["profile_id"] = valid["date"].map(profile_map["profile_id"])
    valid["profile_days_in_period"] = valid["date"].map(profile_map["profile_days_in_period"])
    assert valid["profile_id"].notna().all()
    valid["profile_weight"] = 1 / valid["profile_days_in_period"]
    for period, part in valid.groupby("period"):
        unique_profiles = profiles.loc[profiles["period"].eq(period), "profile_id"].nunique()
        assert np.isclose(part["profile_weight"].sum(), 24 * unique_profiles)

    summary = []
    by_hour = []
    by_month = []
    by_production = []
    by_hour_production = []
    by_transition = []
    for period, group in valid.groupby("period", sort=True):
        summary.append({"period": period, "valid_hours": len(group), "high_hours": int(group["high"].sum()),
                        **rate_row(group, {})})
        for hour, part in group.groupby("hour_group", sort=True):
            by_hour.append(rate_row(part, {"period": period, "hour_group": hour}, weighted=True))
        for month, part in group.groupby(group.index.month, sort=True):
            by_month.append(rate_row(part, {"period": period, "month": month}, weighted=True))
        candidates = group.loc[group["eligible"]]
        transitions = candidates["previous_production_state"] + "->" + candidates["target_production_state"]
        for transition, part in candidates.groupby(transitions, sort=True):
            by_transition.append(rate_row(part, {"period": period, "production_transition": transition}))
        for timing, column in (("target_posthoc", "target_production_state"),
                               ("previous_availability_unconfirmed", "previous_production_state")):
            for state, part in candidates.groupby(column, sort=True):
                by_production.append(rate_row(part, {"period": period, "timing": timing,
                                                     "production_state": state}))
        for (hour, state), part in candidates.groupby(["hour_group", "previous_production_state"], sort=True):
            by_hour_production.append(rate_row(part, {"period": period, "hour_group": hour,
                                                   "previous_production_state": state}))

    # Start a fresh episode on a high record with a nonhigh or missing predecessor.
    starts = valid["high"] & ~pd.Series(prev >= threshold, index=valid.index)
    high = valid.loc[valid["high"]].copy()
    high["episode_id"] = starts.cumsum().loc[high.index].to_numpy()
    episodes = high.groupby("episode_id", sort=True).agg(
        start=("date", "first"), first_record=("peak", "first"),
        hours=("peak", "size"), peak_max=("peak", "max"),
        any_positive_production=("생산량", lambda x: bool(x.gt(0).any())),
    ).reset_index()
    first = high.groupby("episode_id", sort=True).head(1).set_index("episode_id")
    first_keys = high.reset_index().groupby("episode_id", sort=True)["record_key"].first()
    episodes["start_record_key"] = episodes["episode_id"].map(first_keys)
    episodes["period"] = np.where(episodes["start_record_key"].lt(pd.Timestamp("2021-07-01")), "Jan-Jun", "Jul-Aug")
    episodes["start_hour"] = episodes["start_record_key"].dt.hour
    episodes["target_production_at_start"] = episodes["episode_id"].map(first["target_production_state"])
    episodes["left_censored_start"] = ~episodes["episode_id"].map(first["previous_known"])
    assert int(episodes["hours"].sum()) == int(valid["high"].sum())
    assert len(episodes) == int(starts.sum())
    assert len(episodes) == int(valid["onset"].sum() + episodes["left_censored_start"].sum())
    for entry in summary:
        period = entry["period"]
        assert sum(row["eligible_hours"] for row in by_hour if row["period"] == period) == entry["eligible_hours"]
        assert sum(row["onset_hours"] for row in by_hour if row["period"] == period) == entry["onset_hours"]
        assert sum(row["eligible_hours"] for row in by_transition if row["period"] == period) == entry["eligible_hours"]
        assert sum(row["onset_hours"] for row in by_transition if row["period"] == period) == entry["onset_hours"]
    episode_summary = []
    for period, part in episodes.groupby("period", sort=True):
        episode_summary.append({"period": period, "episodes": len(part),
                                "left_censored_starts": int(part["left_censored_start"].sum()),
                                "single_hour_episodes": int(part["hours"].eq(1).sum()),
                                "multi_hour_episodes": int(part["hours"].gt(1).sum()),
                                "longest_hours": int(part["hours"].max()),
                                "episodes_with_any_positive_production": int(part["any_positive_production"].sum())})

    save(pd.DataFrame(summary), "period_summary")
    save(pd.DataFrame(by_hour), "onset_by_hour_group")
    save(pd.DataFrame(by_month), "onset_by_month")
    save(pd.DataFrame(by_production), "onset_by_production")
    save(pd.DataFrame(by_hour_production), "onset_by_hour_and_previous_production")
    save(pd.DataFrame(by_transition), "onset_by_production_transition")
    save(episodes, "episodes")
    save(pd.DataFrame(episode_summary), "episode_summary")
    facts = {"source_sha256": digest, "frozen_sha256": sha256(frozen_bytes).hexdigest(),
             "python": platform.python_version(), "eda008_checks": checks,
             "period_summary": summary, "episode_summary": episode_summary,
             "scope_rows": len(valid), "unique_dates": valid["date"].nunique()}
    (TABLES / f"{PREFIX}_facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    assert sha256(FROZEN.read_bytes()).hexdigest() == sha256(frozen_bytes).hexdigest()
    assert sha256(SOURCE.read_bytes()).hexdigest() == digest
    print(json.dumps({"python": platform.python_version(), "source_sha256": digest,
                      "frozen_sha256": sha256(frozen_bytes).hexdigest(), "eda008_checks": checks,
                      "period_summary": summary, "episode_summary": episode_summary,
                      "by_hour": by_hour, "by_production": by_production,
                      "by_transition": by_transition}, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
