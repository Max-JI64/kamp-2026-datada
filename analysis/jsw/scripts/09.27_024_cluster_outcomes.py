"""Connect fixed prior cluster assignments to high-hour burden without refitting."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from analysis_common import TABLES, save, finish

PREFIX = "09.27_024"
cfg = json.loads((TABLES / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
daily = pd.read_csv(TABLES / cfg["inputs"][1], encoding="utf-8-sig")
assign = pd.read_csv(TABLES / cfg["inputs"][0], encoding="utf-8-sig")
assign = assign.loc[assign.k.eq(3) & ~assign.profile_weighted]
assert len(assign) == 241 * 2
wide = assign.pivot(index="date", columns="representation", values="cluster").reset_index()
data = daily.merge(wide, on="date", validate="one_to_one")
assert len(data) == 241 and data[["raw", "shape"]].notna().all().all()
old = pd.read_csv(TABLES / cfg["inputs"][2], encoding="utf-8-sig")
cells, components, totals, concordance = [], [], [], []
for period, group in data.groupby("period"):
    concordance.append({"period": period, "days": len(group),
                        "raw_shape_ari": adjusted_rand_score(group.raw, group["shape"])})
for representation in ("raw", "shape"):
    for weighted in (False, True):
        for period, group in data.groupby("period"):
            w = group.profile_weight.to_numpy(copy=True) if weighted else np.ones(len(group))
            w /= w.sum()
            for cluster in range(3):
                mask = group[representation].eq(cluster).to_numpy()
                assert mask.any()
                high_hours_mean = np.average(group.loc[mask, "high_hours"], weights=w[mask])
                cells.append({"representation": representation, "profile_weighted": weighted,
                              "period": period, "cluster": cluster, "days": int(mask.sum()),
                              "day_share": float(w[mask].sum()), "mean_high_hours": high_hours_mean,
                              "high_day_fraction": float(np.average(group.loc[mask, "high_hours"].gt(0), weights=w[mask])),
                              "mean_peak": float(np.average(group.loc[mask, "mean_peak"], weights=w[mask])),
                              "mean_low_hours": float(np.average(group.loc[mask, "low_hours"], weights=w[mask]))})
        part = pd.DataFrame(cells).loc[lambda x: x.representation.eq(representation) & x.profile_weighted.eq(weighted)]
        for cluster in range(3):
            a = part.loc[part.period.eq("Jan-Jun") & part.cluster.eq(cluster)].iloc[0]
            b = part.loc[part.period.eq("Jul-Aug") & part.cluster.eq(cluster)].iloc[0]
            mix = (b.day_share - a.day_share) * (a.mean_high_hours + b.mean_high_hours) / 48
            within = (b.mean_high_hours - a.mean_high_hours) * (a.day_share + b.day_share) / 48
            delta = b.day_share * b.mean_high_hours / 24 - a.day_share * a.mean_high_hours / 24
            assert np.isclose(mix + within, delta)
            components.append({"representation": representation, "profile_weighted": weighted,
                               "cluster": cluster, "mix_component": mix, "within_component": within,
                               "total_component": delta})
        comp = pd.DataFrame(components).loc[lambda x: x.representation.eq(representation) & x.profile_weighted.eq(weighted)]
        delta = comp.total_component.sum()
        oldpart = old.loc[old.profile_weighted.eq(weighted)].set_index("period")
        expected = oldpart.loc["Jul-Aug", "high_fraction"] - oldpart.loc["Jan-Jun", "high_fraction"]
        assert np.isclose(delta, expected)
        totals.append({"representation": representation, "profile_weighted": weighted,
                       "high_fraction_difference": delta, "mix_component": comp.mix_component.sum(),
                       "within_component": comp.within_component.sum()})
save(pd.DataFrame(cells), PREFIX, "cluster_profiles")
save(pd.DataFrame(components), PREFIX, "components")
save(pd.DataFrame(totals), PREFIX, "totals")
save(pd.DataFrame(concordance), PREFIX, "representation_agreement")
save(data.groupby(["period", "raw", "shape"]).size().rename("days").reset_index(), PREFIX, "raw_shape_crosswalk")
finish(PREFIX, {"totals": totals, "representation_agreement": concordance,
                "source_days": len(data), "cluster_fit_reused": True})
