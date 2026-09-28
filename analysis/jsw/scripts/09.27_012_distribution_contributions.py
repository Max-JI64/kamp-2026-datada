"""Explain opposite mean/tail directions using fixed training bins."""
import json
import numpy as np
import pandas as pd
from analysis_common import TABLES, save, finish

PREFIX = "09.27_012"
cfg = json.loads((TABLES / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
data = pd.read_csv(TABLES / "09.27_004_records.csv", encoding="utf-8-sig")
data["production_positive"] = data["생산량"].gt(0).astype(int)
train = data.loc[data["period"].eq("Jan-Jun"), "peak"]
edges = sorted(set([0., 1., *train.quantile([.25, .5, .75]).tolist(), float(cfg["threshold"]), np.inf]))
assert min(data["peak"]) >= 0 and all(np.diff(edges) > 0)
data["bin"] = pd.cut(data["peak"], edges, labels=False, right=False)
assert data["bin"].notna().all()
cells = pd.read_csv(TABLES / "09.27_005_strata.csv", encoding="utf-8-sig")
old = pd.read_csv(TABLES / "09.27_005_standardized_distributions.csv", encoding="utf-8-sig")
KEYS = ["hour", "weekend", "production_positive"]
out = []
for weighted in (False, True):
    support = cells.loc[cells["metric"].eq("high") & cells["profile_weighted"].eq(weighted) & cells["common"]].copy()
    pooled = support["earlier_exposure_weight"] + support["later_exposure_weight"]
    refs = pooled / pooled.sum()
    refmap = {tuple(row[k] for k in KEYS): refs.loc[i] for i, row in support.iterrows()}
    for period in ("Jan-Jun", "Jul-Aug"):
        parts = []
        for key, group in data.loc[data["period"].eq(period)].groupby(KEYS):
            if key not in refmap:
                continue
            group = group.copy()
            w = group["profile_weight"].to_numpy() if weighted else np.ones(len(group))
            group["standard_weight"] = w / w.sum() * refmap[key]
            parts.append(group)
        sample = pd.concat(parts)
        assert np.isclose(sample["standard_weight"].sum(), 1)
        for i, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            b = sample.loc[sample["bin"].eq(i)]
            mass = b["standard_weight"].sum()
            contribution = (b["peak"] * b["standard_weight"]).sum()
            out.append({"profile_weighted": weighted, "period": period, "bin": i,
                        "lower_inclusive": lower, "upper_exclusive": upper,
                        "support_n": len(sample), "bin_n": len(b), "dates": b["date"].nunique(),
                        "standardized_probability": mass, "mean_contribution": contribution,
                        "within_bin_mean": contribution / mass if mass else np.nan})
        mean = np.sum(sample["peak"] * sample["standard_weight"])
        expected = old.loc[old["period"].eq(period) & old["profile_weighted"].eq(weighted), "mean"].iloc[0]
        assert np.isclose(mean, expected)
table = pd.DataFrame(out)
differences, totals = [], []
for weighted in (False, True):
    part = table.loc[table["profile_weighted"].eq(weighted)]
    for i in range(len(edges) - 1):
        a = part.loc[part["period"].eq("Jan-Jun") & part["bin"].eq(i)].iloc[0]
        b = part.loc[part["period"].eq("Jul-Aug") & part["bin"].eq(i)].iloc[0]
        differences.append({"profile_weighted": weighted, "bin": i, "lower_inclusive": edges[i],
                            "upper_exclusive": edges[i + 1],
                            "probability_difference": b.standardized_probability - a.standardized_probability,
                            "mean_contribution_difference": b.mean_contribution - a.mean_contribution})
    for period in ("Jan-Jun", "Jul-Aug"):
        p = part.loc[part["period"].eq(period)]
        assert np.isclose(p["standardized_probability"].sum(), 1)
    d = pd.DataFrame(differences).loc[lambda x: x.profile_weighted.eq(weighted)]
    total = d["mean_contribution_difference"].sum()
    means = part.groupby("period")["mean_contribution"].sum()
    assert np.isclose(total, means["Jul-Aug"] - means["Jan-Jun"])
    totals.append({"profile_weighted": weighted, "earlier_mean": means["Jan-Jun"], "later_mean": means["Jul-Aug"],
                   "mean_difference": total,
                   "below182_contribution": d.loc[d.lower_inclusive.lt(182), "mean_contribution_difference"].sum(),
                   "atleast182_contribution": d.loc[d.lower_inclusive.ge(182), "mean_contribution_difference"].sum()})
save(table, PREFIX, "bins")
save(pd.DataFrame(differences), PREFIX, "differences")
save(pd.DataFrame(totals), PREFIX, "totals")
finish(PREFIX, {"training_edges": edges, "totals": totals, "common_cells": len(support),
                "input_reuse": ["004_records", "005_strata", "005_standardized_distributions"]})
