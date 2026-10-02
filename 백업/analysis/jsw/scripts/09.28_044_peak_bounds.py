"""Recorded quarter-value concentration and idealized daily peak bounds."""
from __future__ import annotations

import json
from hashlib import sha256

import numpy as np
import pandas as pd

from analysis_common import SLOTS, SOURCE, TABLES, finish, load, save


PREFIX = "09.28_044"


def lowest_cap(values: np.ndarray, budget: float, receive_zero: bool) -> tuple[float, float]:
    """Water-fill upper bound: remove peaks into eligible capacity with sum fixed."""
    eligible = (values >= 0) if receive_zero else (values > 0)
    if not eligible.any():
        return 0.0, 0.0
    floor = float(values.sum() / eligible.sum())
    high = float(values.max())
    if budget == 0 or high <= floor:
        return high, 0.0
    required_at_floor = float(np.maximum(values - floor, 0).sum())
    if budget >= required_at_floor:
        return floor, required_at_floor
    low = floor
    for _ in range(60):
        mid = (low + high) / 2
        if np.maximum(values - mid, 0).sum() > budget:
            low = mid
        else:
            high = mid
    cap = high
    moved = float(np.maximum(values - cap, 0).sum())
    capacity = float(np.maximum(cap - values[eligible], 0).sum())
    assert moved <= budget + 1e-7 and moved <= capacity + 1e-7
    return cap, moved


def main() -> None:
    config_path = TABLES / f"{PREFIX}_frozen.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["slots"] == SLOTS and config["cutoffs"] == [176, 182, 188]
    assert sha256(SOURCE.read_bytes()).hexdigest() == config["source_sha256"]
    raw, data = load()
    assert len(data) == 5784 and data["date"].nunique() == 241
    assert data.groupby("date").size().eq(24).all()

    daily, caps, scenarios = [], [], []
    for date, part in data.groupby("date", sort=True):
        values = part[SLOTS].to_numpy(dtype=float).ravel()
        assert len(values) == 96 and np.isfinite(values).all() and (values >= 0).all()
        ordered = np.sort(values)[::-1]
        distinct = np.unique(values)[::-1]
        maximum = float(ordered[0])
        mean = float(values.mean())
        period = str(part["period"].iat[0])
        profile = str(part["profile"].iat[0])
        base = {"date": date.strftime("%Y-%m-%d"), "period": period, "profile": profile,
                "month": int(date.month), "sum_values": float(values.sum()),
                "mean_value": mean, "max_value": maximum,
                "mean_max_ratio": mean / maximum if maximum else np.nan,
                "zero_cells": int((values == 0).sum()), "top_ties": int((values == maximum).sum()),
                "second_distinct": float(distinct[1]) if len(distinct) > 1 else np.nan,
                "third_distinct": float(distinct[2]) if len(distinct) > 2 else np.nan,
                "cells_within_5_of_max": int((values >= maximum - 5).sum()),
                "cells_within_10_of_max": int((values >= maximum - 10).sum())}
        daily.append(base)
        for cutoff in config["cutoffs"]:
            excess = np.maximum(values - cutoff, 0)
            caps.append({"date": base["date"], "period": period, "profile": profile,
                         "cutoff": cutoff, "original_max": maximum,
                         "cells_ge": int((values >= cutoff).sum()),
                         "cells_gt": int((values > cutoff).sum()),
                         "max_excess": float(excess.max()),
                         "excess_value_sum": float(excess.sum()),
                         "direct_cap_max": float(min(maximum, cutoff)),
                         "direct_cap_reduction": float(excess.max()),
                         "direct_cap_changed_cells": int((values > cutoff).sum())})
        for fraction in (0.0, 0.05, 0.10):
            budget = fraction * float(values.sum())
            for receive_zero in (True, False):
                cap, moved = lowest_cap(values, budget, receive_zero)
                scenarios.append({"date": base["date"], "period": period, "profile": profile,
                                  "budget_fraction": fraction, "receive_zero": receive_zero,
                                  "original_max": maximum, "budget_value_sum": budget,
                                  "optimal_max": cap, "max_reduction": maximum - cap,
                                  "reduction_fraction": (maximum - cap) / maximum if maximum else np.nan,
                                  "moved_value_sum": moved, "zero_cells": base["zero_cells"]})

    daily = pd.DataFrame(daily)
    caps = pd.DataFrame(caps)
    scenarios = pd.DataFrame(scenarios)
    assert len(daily) == 241 and len(caps) == 723 and len(scenarios) == 1446
    assert scenarios.loc[scenarios["budget_fraction"].eq(0), "max_reduction"].abs().lt(1e-9).all()
    assert scenarios["moved_value_sum"].le(scenarios["budget_value_sum"] + 1e-7).all()
    assert scenarios["optimal_max"].ge(0).all()
    for (_, fraction), part in scenarios.groupby(["date", "budget_fraction"]):
        allow = part.loc[part["receive_zero"], "optimal_max"].iat[0]
        forbid = part.loc[~part["receive_zero"], "optimal_max"].iat[0]
        assert allow <= forbid + 1e-8

    summary = []
    for period, part in daily.groupby("period", sort=False):
        ids = part["profile"].drop_duplicates()
        for weighting, subset in (("observed_days", part),
                                  ("unique_profiles", part.drop_duplicates("profile"))):
            assert len(subset) == (len(part) if weighting == "observed_days" else len(ids))
            chosen = scenarios.merge(subset[["date"]], on="date", validate="many_to_one")
            for (fraction, receive_zero), group in chosen.groupby(["budget_fraction", "receive_zero"]):
                summary.append({"period": period, "weighting": weighting, "days_or_profiles": len(subset),
                                "budget_fraction": fraction, "receive_zero": receive_zero,
                                "median_original_max": float(subset["max_value"].median()),
                                "median_mean_max_ratio": float(subset["mean_max_ratio"].median()),
                                "median_reduction": float(group["max_reduction"].median()),
                                "mean_reduction": float(group["max_reduction"].mean()),
                                "median_reduction_fraction": float(group["reduction_fraction"].median()),
                                "days_or_profiles_with_reduction": int(group["max_reduction"].gt(1e-7).sum()),
                                "days_or_profiles_with_zero": int(subset["zero_cells"].gt(0).sum())})
    summary = pd.DataFrame(summary)

    # Keep invalid-hour rows out of schedules; expose their values as unassigned monthly candidates.
    invalid = raw.loc[raw["날짜"].lt(20210901) & ~raw["시간"].between(0, 23)].copy()
    invalid["month"] = invalid["날짜"] // 100
    invalid["row_max"] = invalid[SLOTS].max(axis=1)
    month_rows = []
    for month, part in data.groupby(data.index.to_period("M")):
        vals = part[SLOTS].to_numpy(dtype=float)
        max_value = float(vals.max())
        im = invalid.loc[invalid["month"].eq(int(month.strftime("%Y%m")))]
        invalid_max = float(im["row_max"].max()) if len(im) else np.nan
        month_rows.append({"month": str(month), "complete_days": int(part["date"].nunique()),
                           "valid_hours": len(part), "normal_max": max_value,
                           "normal_max_cells": int((vals == max_value).sum()),
                           "normal_max_dates": int(part.loc[part[SLOTS].eq(max_value).any(axis=1), "date"].nunique()),
                           "invalid_hour_rows": len(im), "invalid_hour_candidate_max": invalid_max,
                           "candidate_exceeds_normal_max": bool(invalid_max > max_value) if len(im) else False})
    months = pd.DataFrame(month_rows)
    save(daily, PREFIX, "daily_concentration")
    save(caps, PREFIX, "direct_caps")
    save(scenarios, PREFIX, "redistribution")
    save(summary, PREFIX, "scenario_summary")
    save(months, PREFIX, "monthly_max")
    finish(PREFIX, {"normal_hours": len(data), "complete_days": len(daily),
                    "invalid_hour_rows_jan_aug": len(invalid),
                    "zero_days": int(daily["zero_cells"].gt(0).sum()),
                    "maximum": float(daily["max_value"].max()),
                    "monthly_candidate_exceeds": int(months["candidate_exceeds_normal_max"].sum()),
                    "periods": summary.to_dict(orient="records")})


if __name__ == "__main__":
    main()
