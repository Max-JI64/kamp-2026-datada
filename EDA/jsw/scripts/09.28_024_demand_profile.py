"""Measure daily concentration and locate observed 15-minute demand maxima."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).parent.parent
SOURCE = BASE / "../../data/origin/okm_augumented_2021.csv"
TABLES = BASE / "tables"
PREFIX = "09.28_024"
HASH = "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"
SLOTS = ["15분", "30분", "45분", "60분"]
THRESHOLD = 182  # Existing exploratory threshold from A004, not an operating limit.


def save(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(TABLES / f"{PREFIX}_{name}.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    assert sha256(SOURCE.read_bytes()).hexdigest() == HASH
    raw = pd.read_csv(SOURCE, encoding="utf-8-sig", usecols=["날짜", "시간", *SLOTS])
    scoped = raw.loc[raw["날짜"] < 20210901].copy()
    valid = scoped.loc[scoped["시간"].between(0, 23)].copy()
    invalid = scoped.loc[~scoped["시간"].between(0, 23)].copy()
    assert (len(raw), len(scoped), len(valid), len(invalid)) == (6168, 5832, 5784, 48)
    assert not valid.duplicated(["날짜", "시간"]).any()
    assert valid.groupby("날짜").size().eq(24).all()
    assert valid[SLOTS].notna().all().all()

    long = valid.melt(id_vars=["날짜", "시간"], value_vars=SLOTS,
                      var_name="slot", value_name="power")
    long["month"] = long["날짜"] // 100
    long["period"] = np.where(long["날짜"] < 20210701, "Jan-Jun", "Jul-Aug")
    assert len(long) == 5784 * 4

    days = []
    for date, group in long.groupby("날짜", sort=True):
        x = group["power"].to_numpy(dtype=float)
        maximum = float(x.max())
        days.append({
            "date": int(date), "month": int(date // 100),
            "period": "Jan-Jun" if date < 20210701 else "Jul-Aug",
            "recorded_slots": len(x), "exact_mean": float(x.mean()),
            "peak": maximum, "flatness_mean_over_peak": float(x.mean() / maximum) if maximum else np.nan,
            "q10": float(np.quantile(x, .10)), "q25": float(np.quantile(x, .25)),
            "p90": float(np.quantile(x, .90)), "p95": float(np.quantile(x, .95)),
            "top4_mean": float(np.sort(x)[-4:].mean()),
            "slots_at_peak": int(np.count_nonzero(x == maximum)),
            "slots_at_least_90pct_peak": int(np.count_nonzero(x >= .9 * maximum)) if maximum else 96,
            "slots_at_least_182": int(np.count_nonzero(x >= THRESHOLD)),
            "zero_slots": int(np.count_nonzero(x == 0)),
        })
    daily = pd.DataFrame(days)
    assert len(daily) == 241 and daily["recorded_slots"].eq(96).all()
    old = pd.read_csv(BASE / "../../analysis/jsw/tables/09.27_004_daily.csv", encoding="utf-8-sig")
    old["date"] = old["date"].str.replace("-", "").astype(int)
    check = daily.merge(old[["date", "daily_peak", "exact_daily_mean", "high_hours"]],
                        on="date", validate="one_to_one")
    assert len(check) == 241
    assert np.array_equal(check["peak"].to_numpy(), check["daily_peak"].to_numpy())
    assert np.allclose(check["exact_mean"], check["exact_daily_mean"])
    assert ((check["slots_at_least_182"] > 0) == (check["high_hours"] > 0)).all()

    # Low quantiles describe the recorded lower tail, not an operational base load.
    production = pd.read_csv(BASE / "tables/09.28_023_daily.csv", encoding="utf-8-sig")
    profiles = pd.read_csv(BASE / "tables/09.23_011_daily_profile_summary.csv", encoding="utf-8-sig")
    daily = daily.merge(production[["date", "zero_hours", "positive_hours"]], on="date", validate="one_to_one")
    daily = daily.merge(profiles[["date", "profile_id"]], on="date", validate="one_to_one")
    assert len(daily) == 241 and daily["zero_hours"].add(daily["positive_hours"]).eq(24).all()
    daily["production_record_shape"] = np.select(
        [daily["zero_hours"].eq(24), daily["zero_hours"].eq(0)],
        ["all_zero", "all_positive"], default="mixed_zero_positive")
    assert daily["production_record_shape"].value_counts().to_dict() == {
        "mixed_zero_positive": 153, "all_zero": 60, "all_positive": 28}
    zero_hours = valid[SLOTS].eq(0).all(axis=1)
    assert int(zero_hours.sum()) == 17
    zero_dates = set(valid.loc[zero_hours, "날짜"])
    daily["all_four_zero_hours"] = daily["date"].map(
        valid.assign(all_four_zero=zero_hours).groupby("날짜")["all_four_zero"].sum()).astype(int)
    assert daily["all_four_zero_hours"].sum() == 17 and zero_dates == {20210828, 20210829}

    def low_summary(g: pd.DataFrame) -> dict:
        one_per_profile = g.drop_duplicates("profile_id")
        return {"days": len(g), "unique_power_profiles": len(one_per_profile),
                "q10_min": g["q10"].min(), "q10_q25": g["q10"].quantile(.25),
                "q10_median": g["q10"].median(), "q10_q75": g["q10"].quantile(.75),
                "q10_max": g["q10"].max(),
                "q10_unique_profile_median": one_per_profile["q10"].median(),
                "q25_median": g["q25"].median(),
                "days_with_zero_q10": int(g["q10"].eq(0).sum()),
                "days_with_all_four_zero_hour": int(g["all_four_zero_hours"].gt(0).sum())}

    low_monthly = pd.DataFrame([{"month": int(month), **low_summary(g)}
                                for month, g in daily.groupby("month", sort=True)])
    low_by_production = pd.DataFrame([
        {"period": period, "production_record_shape": shape, **low_summary(g)}
        for (period, shape), g in daily.groupby(["period", "production_record_shape"], sort=True)])
    assert low_monthly["days"].sum() == 241 and low_by_production["days"].sum() == 241

    # Compare high-peak days with lower-peak days. Quartiles remain descriptive.
    period_rows = []
    for period, g in daily.groupby("period", sort=False):
        for label, s in [("all", g), ("peak_ge_182", g[g["peak"] >= THRESHOLD]),
                         ("peak_lt_182", g[g["peak"] < THRESHOLD])]:
            ratio = s["flatness_mean_over_peak"].dropna()
            period_rows.append({"period": period, "group": label, "days": len(s),
                                "zero_peak_days": int(s["peak"].eq(0).sum()),
                                "median_peak": s["peak"].median(),
                                "median_exact_mean": s["exact_mean"].median(),
                                "flatness_q25": ratio.quantile(.25),
                                "flatness_median": ratio.median(),
                                "flatness_q75": ratio.quantile(.75),
                                "median_p95_over_peak": (s["p95"] / s["peak"].replace(0, np.nan)).median(),
                                "median_slots_ge_90pct_peak": s["slots_at_least_90pct_peak"].median(),
                                "median_slots_ge_182": s["slots_at_least_182"].median()})
    concentration = pd.DataFrame(period_rows)

    # All tied maximum slots are retained, so a peak is not assigned to one arbitrary hour.
    daily_ties = long.merge(daily[["date", "peak"]], left_on="날짜", right_on="date", validate="many_to_one")
    daily_ties = daily_ties.loc[daily_ties["power"] == daily_ties["peak"],
                                ["date", "period", "시간", "slot", "power"]]
    daily_ties = daily_ties.rename(columns={"시간": "hour"}).sort_values(["date", "hour", "slot"])
    assert daily_ties.groupby("date").size().reindex(daily["date"]).to_numpy().tolist() == daily["slots_at_peak"].tolist()
    month_rows = []
    month_peak_slots = []
    for month, g in long.groupby("month", sort=True):
        maximum = g["power"].max()
        tied = g[g["power"] == maximum].sort_values(["날짜", "시간", "slot"])
        daily_month = daily[daily["month"] == month]
        month_rows.append({"month": int(month), "complete_days": len(daily_month),
                           "valid_hours": len(g) // 4, "valid_slots": len(g),
                           "monthly_peak_valid_time": maximum,
                           "peak_tied_slots": len(tied), "peak_tied_dates": tied["날짜"].nunique(),
                           "high_days_ge_182": int(daily_month["slots_at_least_182"].gt(0).sum()),
                           "high_slots_ge_182": int((g["power"] >= THRESHOLD).sum())})
        month_peak_slots.append(tied[["month", "날짜", "시간", "slot", "power"]])
    monthly = pd.DataFrame(month_rows)
    peaks = pd.concat(month_peak_slots, ignore_index=True).rename(columns={"날짜": "date", "시간": "hour"})
    # Existing frequent high-hour condition uses an hourly maximum, not slot count.
    hours = valid.assign(hour_peak=valid[SLOTS].max(axis=1),
                         period=np.where(valid["날짜"] < 20210701, "Jan-Jun", "Jul-Aug"))
    freq = hours.groupby(["period", "시간"], sort=True).agg(valid_hours=("hour_peak", "size"),
              high_hours_ge_182=("hour_peak", lambda x: int((x >= THRESHOLD).sum()))).reset_index()
    freq = freq.rename(columns={"시간": "hour"})
    day_peak_hours = daily_ties.groupby(["period", "hour"]).agg(
        daily_max_tied_slots=("power", "size"), daily_max_distinct_dates=("date", "nunique")).reset_index()
    freq = freq.merge(day_peak_hours, on=["period", "hour"], how="left")
    high_day_peak_hours = daily_ties.loc[daily_ties["power"] >= THRESHOLD].groupby(["period", "hour"]).agg(
        high_day_max_tied_slots=("power", "size"), high_day_max_distinct_dates=("date", "nunique")).reset_index()
    freq = freq.merge(high_day_peak_hours, on=["period", "hour"], how="left")
    peak_hours = peaks.assign(period=np.where(peaks["date"] < 20210701, "Jan-Jun", "Jul-Aug"))
    peak_hours = peak_hours.groupby(["period", "hour"]).size().rename("monthly_max_tied_slots").reset_index()
    freq = freq.merge(peak_hours, on=["period", "hour"], how="left")
    for col in ["daily_max_tied_slots", "daily_max_distinct_dates", "high_day_max_tied_slots",
                "high_day_max_distinct_dates", "monthly_max_tied_slots"]:
        freq[col] = freq[col].fillna(0).astype(int)

    # Invalid hour labels have a known date but cannot be placed on the clock.
    invalid_rows = []
    for date, g in invalid.groupby("날짜", sort=True):
        vals = g[SLOTS].to_numpy(dtype=float)
        month = date // 100
        valid_peak = monthly.loc[monthly["month"] == month, "monthly_peak_valid_time"].iloc[0]
        invalid_rows.append({"date": int(date), "month": int(month), "invalid_hours": len(g),
                             "invalid_slots": vals.size, "invalid_peak": float(vals.max()),
                             "valid_month_peak": float(valid_peak),
                             "invalid_exceeds_valid_month_peak": bool(vals.max() > valid_peak),
                             "invalid_ties_valid_month_peak": bool(vals.max() == valid_peak),
                             "invalid_slots_ge_182": int((vals >= THRESHOLD).sum())})
    invalid_candidates = pd.DataFrame(invalid_rows)
    assert len(invalid_candidates) == 2
    assert monthly["complete_days"].sum() == 241
    assert monthly["valid_slots"].sum() == 23136
    assert freq["high_hours_ge_182"].sum() == int((hours["hour_peak"] >= THRESHOLD).sum())
    assert peaks.groupby("month").size().to_numpy().tolist() == monthly["peak_tied_slots"].tolist()

    save(daily, "daily_concentration")
    save(low_monthly, "low_load_monthly")
    save(low_by_production, "low_load_by_production")
    save(concentration, "concentration_summary")
    save(monthly, "monthly_maxima")
    save(daily_ties, "daily_max_ties")
    save(peaks, "monthly_max_ties")
    save(freq, "hour_comparison")
    save(invalid_candidates, "invalid_time_candidates")
    facts = {"source_sha256": HASH, "source_rows": len(raw), "scoped_rows": len(scoped),
             "complete_days": len(daily), "valid_hours": len(valid), "valid_slots": len(long),
             "invalid_hours": len(invalid), "daily_peak_matches_A004": True,
             "daily_mean_matches_A004": True, "zero_peak_days": int(daily["peak"].eq(0).sum()),
             "high_peak_days": int(daily["peak"].ge(THRESHOLD).sum()),
             "all_four_zero_hours": int(zero_hours.sum()),
             "daily_q10_zero_days": int(daily["q10"].eq(0).sum()),
             "monthly_maxima": monthly.to_dict("records")}
    (TABLES / f"{PREFIX}_facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(facts, ensure_ascii=True))


if __name__ == "__main__":
    main()
