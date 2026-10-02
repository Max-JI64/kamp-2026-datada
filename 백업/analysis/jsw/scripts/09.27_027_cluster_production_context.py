"""Check whether fixed power clusters map uniquely to daily production records."""
import json
import numpy as np
import pandas as pd
from analysis_common import TABLES, PROFILES, SOURCE_HASH, PROFILE_HASH, save, finish

PREFIX = "09.27_027"
cfg = json.loads((TABLES / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
profile = pd.read_csv(PROFILES, encoding="utf-8-sig", dtype={"date": str})
profile["date"] = pd.to_datetime(profile["date"])
assign = pd.read_csv(TABLES / "09.27_008_cluster_days.csv", encoding="utf-8-sig", parse_dates=["date"])
assign = assign.loc[assign.representation.eq("raw") & assign.k.eq(3) & ~assign.profile_weighted, ["date", "period", "cluster"]]
daily = profile.merge(assign, on="date", validate="one_to_one")
assert len(daily) == 241 and daily.date.is_unique
daily["production_zero"] = daily.production_total.eq(0)
cluster_rows = []
for (period, cluster), group in daily.groupby(["period", "cluster"]):
    positive = group.loc[~group.production_zero, "production_total"]
    cluster_rows.append({"period": period, "cluster": cluster, "days": len(group),
                         "distinct_power_profiles": group.profile_id.nunique(),
                         "zero_production_days": int(group.production_zero.sum()),
                         "positive_production_days": int((~group.production_zero).sum()),
                         "production_total_min": group.production_total.min(),
                         "production_total_median": group.production_total.median(),
                         "production_total_max": group.production_total.max(),
                         "positive_production_median": positive.median() if len(positive) else np.nan,
                         "temperature_mean_min": group.temperature_mean.min(),
                         "temperature_mean_max": group.temperature_mean.max()})
exact_rows = []
for profile_id, group in daily.groupby("profile_id"):
    repeated = len(group) > 1
    production_varies = group.production_total.nunique() > 1
    zero_positive = group.production_zero.any() and (~group.production_zero).any()
    clusters = group.cluster.nunique()
    exact_rows.append({"profile_id": profile_id, "days": len(group), "periods": group.period.nunique(),
                       "clusters": clusters, "repeated": repeated, "production_varies": production_varies,
                       "zero_and_positive_production": zero_positive,
                       "production_min": group.production_total.min(),
                       "production_max": group.production_total.max(),
                       "temperature_min": group.temperature_mean.min(),
                       "temperature_max": group.temperature_mean.max(),
                       "first_date": group.date.min(), "last_date": group.date.max()})
exact = pd.DataFrame(exact_rows)
assert exact.clusters.eq(1).all()
repeated = exact.loc[exact.repeated]
summary = {"days": len(daily), "clusters": 3, "exact_profiles": len(exact),
           "repeated_profiles": len(repeated), "days_in_repeated_profiles": int(repeated.days.sum()),
           "repeated_profiles_with_different_production": int(repeated.production_varies.sum()),
           "days_in_different_production_profiles": int(repeated.loc[repeated.production_varies, "days"].sum()),
           "repeated_profiles_with_zero_and_positive": int(repeated.zero_and_positive_production.sum()),
           "days_in_zero_and_positive_profiles": int(repeated.loc[repeated.zero_and_positive_production, "days"].sum())}
save(pd.DataFrame(cluster_rows), PREFIX, "cluster_context")
save(exact, PREFIX, "exact_profile_variation")
save(daily[["date", "period", "cluster", "profile_id", "production_total", "temperature_mean"]], PREFIX, "daily_crosswalk")
finish(PREFIX, {"summary": summary, "cluster_context": cluster_rows})
