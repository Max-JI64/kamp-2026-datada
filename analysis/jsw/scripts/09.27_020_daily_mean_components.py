"""Separate daily category composition from within-category mean changes."""
import json
import numpy as np
import pandas as pd
from analysis_common import TABLES, save, finish

PREFIX = "09.27_020"
cfg = json.loads((TABLES / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
daily = pd.read_csv(TABLES / cfg["inputs"][0], encoding="utf-8-sig")
old = pd.read_csv(TABLES / cfg["inputs"][2], encoding="utf-8-sig")
cells, contributions, totals = [], [], []
for weighted in (False, True):
    for period, group in daily.groupby("period"):
        weights = group.profile_weight.to_numpy(copy=True) if weighted else np.ones(len(group))
        weights /= weights.sum()
        for category in cfg["categories"]:
            mask = group.category.eq(category).to_numpy()
            assert mask.any()
            cells.append({"profile_weighted": weighted, "period": period, "category": category,
                          "days": int(mask.sum()), "probability": weights[mask].sum(),
                          "mean_peak": np.average(group.loc[mask, "mean_peak"], weights=weights[mask])})
    table = pd.DataFrame(cells).loc[lambda x: x.profile_weighted.eq(weighted)]
    for category in cfg["categories"]:
        a = table.loc[table.period.eq("Jan-Jun") & table.category.eq(category)].iloc[0]
        b = table.loc[table.period.eq("Jul-Aug") & table.category.eq(category)].iloc[0]
        composition = (b.probability - a.probability) * (a.mean_peak + b.mean_peak) / 2
        level = (b.mean_peak - a.mean_peak) * (a.probability + b.probability) / 2
        delta = b.probability * b.mean_peak - a.probability * a.mean_peak
        assert np.isclose(composition + level, delta)
        contributions.append({"profile_weighted": weighted, "category": category,
                              "earlier_days": a.days, "later_days": b.days,
                              "composition_component": composition, "level_component": level,
                              "total_mean_component": delta})
    means = table.assign(contribution=table.probability * table.mean_peak).groupby("period").contribution.sum()
    for period in cfg["periods"]:
        expected = old.loc[old.period.eq(period) & old.profile_weighted.eq(weighted), "mean_peak"].iloc[0]
        assert np.isclose(means[period], expected)
    c = pd.DataFrame(contributions).loc[lambda x: x.profile_weighted.eq(weighted)]
    delta = means["Jul-Aug"] - means["Jan-Jun"]
    assert np.isclose(c.total_mean_component.sum(), delta)
    totals.append({"profile_weighted": weighted, "earlier_mean": means["Jan-Jun"], "later_mean": means["Jul-Aug"],
                   "mean_difference": delta, "composition_component": c.composition_component.sum(),
                   "level_component": c.level_component.sum(),
                   "low_only_level_component": c.loc[c.category.eq("low_only"), "level_component"].iloc[0]})
save(pd.DataFrame(cells), PREFIX, "categories")
save(pd.DataFrame(contributions), PREFIX, "contributions")
save(pd.DataFrame(totals), PREFIX, "totals")
finish(PREFIX, {"totals": totals, "minimum_category_days": int(min(r["days"] for r in cells)), "days": len(daily)})
