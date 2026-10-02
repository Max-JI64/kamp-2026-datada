"""Reaggregate saved A044 caps; no optimization, training, or alert evaluation."""
from __future__ import annotations

import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
TABLES = BASE / "tables"
P = "09.28_045"
ATOL = 1e-8


def read(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative, encoding="utf-8-sig")


def dates(values: pd.Series, fmt: str) -> pd.Series:
    return pd.to_datetime(values.astype(str), format=fmt).dt.strftime("%Y-%m-%d")


def ties(part: pd.DataFrame, column: str, maximum: float) -> list[str]:
    return sorted(part.loc[np.isclose(part[column], maximum, rtol=0, atol=ATOL), "date"])


def save(frame: pd.DataFrame, suffix: str) -> None:
    frame.to_csv(TABLES / f"{P}_{suffix}.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    config_path = TABLES / f"{P}_frozen.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["tie_atol"] == ATOL and config["tie_rtol"] == 0
    assert config["budgets"] == [0, .05, .1] and config["high_cutoff"] == 182
    assert config["expected_days"] == 241 and config["expected_scenarios"] == 1446
    for relative, expected in config["input_sha256"].items():
        assert sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative

    daily = read("analysis/jsw/tables/09.28_044_daily_concentration.csv")
    sims = read("analysis/jsw/tables/09.28_044_redistribution.csv")
    legacy = read("analysis/jsw/tables/09.28_044_scenario_summary.csv")
    old_months = read("analysis/jsw/tables/09.28_044_monthly_max.csv")
    eda = read("EDA/jsw/tables/09.28_024_daily_concentration.csv")
    eda_ties = read("EDA/jsw/tables/09.28_024_monthly_max_ties.csv")
    invalid = read("EDA/jsw/tables/09.28_024_invalid_time_candidates.csv")
    for frame in (daily, sims):
        frame["date"] = dates(frame["date"], "%Y-%m-%d")
        dt = pd.to_datetime(frame["date"], format="%Y-%m-%d")
        assert dt.between("2021-01-01", "2021-08-31").all()
        assert frame["period"].eq(np.where(dt.dt.month <= 6, "Jan-Jun", "Jul-Aug")).all()
    assert len(daily) == 241 and daily["date"].is_unique
    assert daily["month"].eq(pd.to_datetime(daily["date"]).dt.month).all()
    assert len(sims) == 1446 and not sims.duplicated(["date", "budget_fraction", "receive_zero"]).any()
    assert set(sims["receive_zero"]) == {True, False}
    assert set(sims["budget_fraction"]) == {0, .05, .1}
    assert sims.groupby("date").size().eq(6).all()
    assert set(sims["date"]) == set(daily["date"])
    assert np.isfinite(sims[["original_max", "optimal_max", "max_reduction"]]).all().all()
    eda["date"] = dates(eda["date"], "%Y%m%d")
    aligned = daily.merge(eda, on="date", validate="one_to_one")
    assert len(aligned) == 241
    assert np.allclose(aligned["max_value"], aligned["peak"], rtol=0, atol=ATOL)
    assert aligned["profile"].eq(aligned["profile_id"]).all()
    assert int(daily["max_value"].ge(182).sum()) == config["expected_high_days"] == 105
    assert daily.groupby("period")["max_value"].apply(lambda x: x.ge(182).sum()).to_dict() == {
        "Jan-Jun": 69, "Jul-Aug": 36}

    joined = sims.merge(daily[["date", "max_value", "sum_values", "month", "profile", "period",
                              "mean_max_ratio", "zero_cells"]], on="date", validate="many_to_one",
                        suffixes=("", "_daily"))
    for column in ("profile", "period", "zero_cells"):
        assert joined[column].eq(joined[f"{column}_daily"]).all()
    assert np.allclose(joined["original_max"], joined["max_value"], rtol=0, atol=ATOL)
    assert np.allclose(joined["budget_value_sum"], joined["budget_fraction"] * joined["sum_values"], rtol=0, atol=ATOL)
    assert np.allclose(joined["max_reduction"], joined["original_max"] - joined["optimal_max"], rtol=0, atol=ATOL)
    expected_ratio = np.where(joined["original_max"].gt(0), joined["max_reduction"] / joined["original_max"], np.nan)
    assert np.allclose(joined["reduction_fraction"], expected_ratio, rtol=0, atol=ATOL, equal_nan=True)
    assert joined["optimal_max"].ge(-ATOL).all() and joined["max_reduction"].ge(-ATOL).all()
    assert joined["moved_value_sum"].le(joined["budget_value_sum"] + 1e-7).all()
    zero = joined.loc[joined["budget_fraction"].eq(0)]
    assert np.allclose(zero["optimal_max"], zero["original_max"], rtol=0, atol=ATOL)
    # A profile is a complete 96-value vector. Once-per-profile statistics are valid
    # only if all dates with that ID have identical caps under the same scenario.
    for _, group in joined.groupby(["profile", "budget_fraction", "receive_zero"]):
        for column in ("original_max", "optimal_max", "max_reduction", "reduction_fraction", "mean_max_ratio"):
            assert group[column].max() - group[column].min() <= ATOL

    summaries = []
    for period in ("Jan-Aug", "Jan-Jun", "Jul-Aug"):
        period_days = daily if period == "Jan-Aug" else daily.loc[daily["period"].eq(period)]
        for name, subset in (("all", period_days), ("high_ge_182", period_days.loc[period_days["max_value"].ge(182)]),
                             ("below_182", period_days.loc[period_days["max_value"].lt(182)])):
            for weighting in ("observed_days", "unique_profiles"):
                chosen = subset if weighting == "observed_days" else subset.drop_duplicates("profile")
                selected = joined.loc[joined["date"].isin(chosen["date"])]
                for (fraction, receive_zero), group in selected.groupby(["budget_fraction", "receive_zero"]):
                    result = {"period": period, "group": name, "weighting": weighting,
                              "days": len(subset), "unique_profiles": subset["profile"].nunique(),
                              "days_or_profiles": len(chosen), "budget_fraction": fraction, "receive_zero": bool(receive_zero),
                              "median_original_max": float(chosen["max_value"].median()),
                              "median_mean_max_ratio": float(chosen["mean_max_ratio"].median()),
                              "mean_reduction": float(group["max_reduction"].mean()),
                              "days_or_profiles_with_reduction": int(group["max_reduction"].gt(1e-7).sum()),
                              "days_or_profiles_with_zero": int(chosen["zero_cells"].gt(0).sum()),
                              "undefined_reduction_fraction": int(group["reduction_fraction"].isna().sum())}
                    for column, label in (("max_reduction", "reduction"), ("reduction_fraction", "reduction_fraction")):
                        for q, prefix in ((.25, "q25"), (.5, "median"), (.75, "q75")):
                            result[f"{prefix}_{label}"] = float(group[column].quantile(q, interpolation="linear"))
                    summaries.append(result)
    summary = pd.DataFrame(summaries)
    assert len(summary) == 108
    # Reproduce every stored A044 all-days summary field, not just one headline median.
    for _, previous in legacy.iterrows():
        current = summary.loc[summary["period"].eq(previous["period"]) & summary["group"].eq("all") &
                              summary["weighting"].eq(previous["weighting"]) &
                              summary["budget_fraction"].eq(previous["budget_fraction"]) &
                              summary["receive_zero"].eq(previous["receive_zero"])]
        assert len(current) == 1
        for column in legacy.columns.difference(["period", "weighting", "budget_fraction", "receive_zero"]):
            assert np.isclose(current.iloc[0][column], previous[column], rtol=0, atol=ATOL), column

    eda_ties["date"] = dates(eda_ties["date"], "%Y%m%d")
    month_rows, tie_rows = [], []
    for (month, fraction, receive_zero), part in joined.groupby(["month", "budget_fraction", "receive_zero"]):
        original, adjusted = float(part["original_max"].max()), float(part["optimal_max"].max())
        before, after = ties(part, "original_max", original), ties(part, "optimal_max", adjusted)
        expected_dates = sorted(eda_ties.loc[eda_ties["month"].eq(202100 + month), "date"].unique())
        assert before == expected_dates
        previous = old_months.loc[old_months["month"].eq(f"2021-{month:02d}")].iloc[0]
        assert original == previous["normal_max"] and len(part) == previous["complete_days"]
        assert adjusted + ATOL >= part["optimal_max"].max()
        assert np.isclose(part.drop_duplicates("profile")["optimal_max"].max(), adjusted, rtol=0, atol=ATOL)
        assert not fraction or adjusted <= original + ATOL
        if fraction == 0:
            assert original == adjusted and before == after
        b, a = set(before), set(after)
        relation = "unchanged" if b == a else "partial_overlap" if b & a else "replaced"
        tracked = float(part.loc[part["date"].isin(before), "optimal_max"].max())
        month_rows.append({"month": f"2021-{month:02d}", "complete_days": len(part), "unique_profiles": part["profile"].nunique(),
                           "budget_fraction": fraction, "receive_zero": bool(receive_zero), "original_month_max": original,
                           "adjusted_month_max": adjusted, "month_reduction": original - adjusted,
                           "month_reduction_fraction": (original - adjusted) / original,
                           "original_max_dates": "|".join(before), "adjusted_max_dates": "|".join(after),
                           "original_tied_dates": len(before), "adjusted_tied_dates": len(after),
                           "date_set_relation": relation, "retained_dates": "|".join(sorted(b & a)),
                           "removed_dates": "|".join(sorted(b - a)), "added_dates": "|".join(sorted(a - b)),
                           "original_date_only_adjusted_max": tracked, "missed_max_by_original_date_only": adjusted - tracked})
        for phase, chosen_dates, value in (("original", before, original), ("adjusted", after, adjusted)):
            for date in chosen_dates:
                row = part.loc[part["date"].eq(date)].iloc[0]
                tie_rows.append({"month": f"2021-{month:02d}", "budget_fraction": fraction, "receive_zero": bool(receive_zero),
                                 "phase": phase, "date": date, "profile": row["profile"], "month_max": value,
                                 "daily_max": row["original_max"] if phase == "original" else row["optimal_max"]})
    months, month_ties = pd.DataFrame(month_rows), pd.DataFrame(tie_rows)
    assert len(months) == 48 and not months.duplicated(["month", "budget_fraction", "receive_zero"]).any()
    for _, part in joined.groupby(["date", "receive_zero"]):
        assert (np.diff(part.sort_values("budget_fraction")["optimal_max"]) <= ATOL).all()

    invalid["date"] = dates(invalid["date"], "%Y%m%d")
    assert set(invalid["date"]) == {"2021-07-13", "2021-07-15"}
    assert invalid["invalid_hours"].sum() == 48 and invalid["invalid_slots"].sum() == 192
    sensitivity = []
    for _, row in months.loc[months["month"].eq("2021-07")].iterrows():
        candidate = float(invalid["invalid_peak"].max())
        peak = max(row["adjusted_month_max"], candidate)
        kept = row["adjusted_max_dates"].split("|") if abs(peak - row["adjusted_month_max"]) <= ATOL else []
        added = sorted(invalid.loc[np.isclose(invalid["invalid_peak"], peak, rtol=0, atol=ATOL), "date"])
        sensitivity.append({"month": "2021-07", "budget_fraction": row["budget_fraction"], "receive_zero": row["receive_zero"],
                            "adjusted_normal_days": 29, "unchanged_error_dates": "|".join(invalid["date"]),
                            "error_original_maxima": "190|202", "error_candidate_max": candidate,
                            "adjusted_normal_month_max": row["adjusted_month_max"],
                            "error_candidate_exceeds_adjusted_normal": candidate > row["adjusted_month_max"] + ATOL,
                            "auxiliary_max_errors_unchanged": peak,
                            "auxiliary_max_dates": "|".join(sorted(kept + added)), "auxiliary_reduction_from_222": 222 - peak})
    sensitivity = pd.DataFrame(sensitivity)
    policy = joined.pivot(index=["date", "period", "profile", "original_max", "budget_fraction"],
                          columns="receive_zero", values="optimal_max").reset_index().rename(
                              columns={True: "allow_zero_max", False: "forbid_zero_max"})
    policy["forbid_minus_allow"] = policy["forbid_zero_max"] - policy["allow_zero_max"]
    assert len(policy) == 723 and policy["forbid_minus_allow"].ge(-ATOL).all()
    save(summary, "group_summary")
    save(months, "monthly_max")
    save(month_ties, "monthly_max_ties")
    save(sensitivity, "excluded_date_sensitivity")
    save(policy, "zero_policy_difference")
    outputs = {path.name: sha256(path.read_bytes()).hexdigest() for path in TABLES.glob(f"{P}_*.csv")}
    facts = {"completed_at": datetime.now().astimezone().isoformat(timespec="minutes"),
             "normal_days": len(daily), "scenario_rows": len(sims), "high_days": 105,
             "period_counts": {p: {"days": len(g), "high_days": int(g["max_value"].ge(182).sum()),
                                   "profiles": int(g["profile"].nunique())} for p, g in daily.groupby("period")},
             "summary_rows": len(summary), "legacy_summary_rows_reproduced": len(legacy),
             "monthly_scenario_rows": len(months), "calendar_tie_rows": len(month_ties),
             "zero_percent_daily_rows_reproduced": len(zero), "zero_percent_months_reproduced": 16,
             "positive_budget_date_sets": months.loc[months["budget_fraction"].gt(0), "date_set_relation"].value_counts().to_dict(),
             "max_original_date_only_miss": float(months["missed_max_by_original_date_only"].max()),
             "excluded_dates": invalid[["date", "invalid_hours", "invalid_slots", "invalid_peak"]].to_dict("records"),
             "zero_policy_affected_rows": int(policy["forbid_minus_allow"].gt(ATOL).sum()),
             "zero_policy_max_gap": float(policy["forbid_minus_allow"].max()),
             "optimization_executed": False, "model_training_executed": False, "alert_evaluation_executed": False,
             "input_sha256": config["input_sha256"], "output_sha256": outputs,
             "frozen_sha256": sha256(config_path.read_bytes()).hexdigest()}
    # Check source and saved inputs once more after writing our own outputs.
    for relative, expected in config["input_sha256"].items():
        assert sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    (TABLES / f"{P}_facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: facts[k] for k in ("normal_days", "scenario_rows", "high_days", "summary_rows",
                     "positive_budget_date_sets", "max_original_date_only_miss", "zero_policy_max_gap")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
