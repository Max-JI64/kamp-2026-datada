"""Distinguish within-day tail coexistence from between-day dispersion."""
import json
import numpy as np
import pandas as pd
from analysis_common import TABLES, save, finish

PREFIX = "09.27_015"
cfg = json.loads((TABLES / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
data = pd.read_csv(TABLES / "09.27_004_records.csv", encoding="utf-8-sig")
assert data.groupby("date").size().eq(24).all()
data["low"] = data.peak.lt(cfg["low_threshold"])
data["high"] = data.peak.ge(cfg["high_threshold"])
daily = data.groupby(["period", "date"]).agg(mean_peak=("peak", "mean"),
    within_variance=("peak", lambda x: np.var(x, ddof=0)),
    low_hours=("low", "sum"), high_hours=("high", "sum"),
    profile=("profile", "first"), profile_weight=("profile_weight", "first")).reset_index()
daily["category"] = np.select([
    daily.low_hours.gt(0) & daily.high_hours.gt(0),
    daily.low_hours.gt(0), daily.high_hours.gt(0)],
    ["both_low_and_high", "low_only", "high_only"], default="neither")
summary, categories = [], []
for period, part in daily.groupby("period"):
    original = data.loc[data.period.eq(period)]
    for weighted in (False, True):
        w = part.profile_weight.to_numpy() if weighted else np.ones(len(part))
        w = w / w.sum()
        mean = np.sum(w * part.mean_peak)
        within = np.sum(w * part.within_variance)
        between = np.sum(w * (part.mean_peak - mean) ** 2)
        hour_w = original.profile_weight.to_numpy() if weighted else np.ones(len(original))
        direct_mean = np.average(original.peak, weights=hour_w)
        total = np.average((original.peak - direct_mean) ** 2, weights=hour_w)
        assert np.isclose(mean, direct_mean) and np.isclose(within + between, total)
        summary.append({"period": period, "profile_weighted": weighted, "days": len(part),
                        "mean_peak": mean, "total_variance": total,
                        "within_day_variance": within, "between_day_variance": between,
                        "within_share": within / total, "between_share": between / total})
        for category in ("both_low_and_high", "low_only", "high_only", "neither"):
            mask = part.category.eq(category)
            mass = w[mask].sum()
            categories.append({"period": period, "profile_weighted": weighted, "category": category,
                               "days": int(mask.sum()), "fraction": mass,
                               "mean_peak": np.average(part.loc[mask, "mean_peak"], weights=w[mask]) if mask.any() else np.nan,
                               "mean_low_hours": np.average(part.loc[mask, "low_hours"], weights=w[mask]) if mask.any() else np.nan,
                               "mean_high_hours": np.average(part.loc[mask, "high_hours"], weights=w[mask]) if mask.any() else np.nan})
save(daily, PREFIX, "daily")
save(pd.DataFrame(summary), PREFIX, "variance_partition")
save(pd.DataFrame(categories), PREFIX, "coexistence")
finish(PREFIX, {"variance_partition": summary,
                "coexistence": [r for r in categories if r["category"] == "both_low_and_high"],
                "input_reuse": "004_records", "days": len(daily)})
