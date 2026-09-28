"""Connected distribution and date/week sensitivity following 035."""
import json
import numpy as np
import pandas as pd
from analysis_common import TABLES, load, save, finish

PREFIX = "09.27_037"
cfg = json.loads((TABLES / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
_, data = load()
w = data.loc[data.month.isin(cfg["months"]) & data.weekend.eq(0) & data.production_positive.eq(1)].copy()
w["week"] = w.date - pd.to_timedelta(w.date.dt.dayofweek, unit="D")
old = pd.read_csv(TABLES / "09.27_035_standardized_months.csv", encoding="utf-8-sig")
summaries, bins, exclusions, cases = [], [], [], []

def weights(part, ref):
    total = part.groupby("hour").base.sum()
    if set(total.index) != set(range(24)):
        return None
    return part.base * part.hour.map(ref / total)

for weighted in (False, True):
    w["base"] = w.profile_weight if weighted else 1.0
    ref = w.groupby("hour").base.sum()
    ref = ref / ref.sum()
    save(ref.rename("reference_weight").reset_index(), PREFIX, f"reference_{int(weighted)}")
    base_results = {}
    for month, p in w.groupby("month"):
        p = p.copy()
        p["standard_weight"] = weights(p, ref)
        assert np.isclose(p.standard_weight.sum(), 1)
        row = {"profile_weighted": weighted, "month": month, "hours": len(p), "dates": p.date.nunique()}
        for metric, col in (("mean", "평균"), ("max", "peak")):
            v, wt = p[col].to_numpy(), p.standard_weight.to_numpy()
            row[f"{metric}_average"] = float(wt @ v)
            order = np.argsort(v, kind="stable")
            for q in cfg["quantiles"]:
                row[f"{metric}_q{int(q*100)}"] = float(v[order][np.searchsorted(np.cumsum(wt[order]), q)])
            for low, high in zip(cfg["bins"][:-1], cfg["bins"][1:]):
                mask = (v >= low) & (v < high)
                bins.append({"profile_weighted": weighted, "month": month, "metric": metric,
                             "low": low, "high": high, "mass": float(wt[mask].sum()),
                             "mean_contribution": float(wt[mask] @ v[mask])})
        for threshold in cfg["thresholds"]:
            row[f"high_{threshold}"] = float(p.standard_weight @ p.peak.ge(threshold))
        summaries.append(row)
        base_results[month] = row
        for metric, old_metric in (("mean", "target_mean"), ("max", "target_max")):
            expected = old.loc[old.metric.eq(old_metric) & old.profile_weighted.eq(weighted)].iloc[0]
            assert np.isclose(row[f"{metric}_average"], expected[{6:"june",7:"july",8:"august"}[month]])
        expected = old.loc[old.metric.eq("high") & old.profile_weighted.eq(weighted)].iloc[0]
        assert np.isclose(row["high_182"], expected[{6:"june",7:"july",8:"august"}[month]])
        for date, day in p.groupby("date"):
            cases.append({"profile_weighted": weighted, "date": date, "month": month,
                          "hours": len(day), "weight": day.standard_weight.sum(),
                          "peak_mean": day.peak.mean(), "high_hours": day.peak.ge(182).sum(),
                          "high_contribution": float(day.standard_weight @ day.peak.ge(182)),
                          "peak_contribution": float(day.standard_weight @ day.peak),
                          "production_total": day["생산량"].sum(), "temperature_mean": day["기온"].mean(),
                          "profile": day.profile.iloc[0]})
    for unit in ("date", "week"):
        for omitted in sorted(w[unit].unique()):
            sample = w.loc[w[unit].ne(omitted)]
            records = {}
            for month, part in sample.groupby("month"):
                wt = weights(part, ref)
                if wt is None:
                    break
                records[month] = {"mean": float(wt @ part["평균"]), "max": float(wt @ part.peak),
                                  "high": float(wt @ part.peak.ge(182))}
            valid = set(records) == set(cfg["months"])
            for month in (7, 8):
                exclusions.append({"profile_weighted": weighted, "unit": unit, "omitted": pd.Timestamp(omitted),
                                   "comparison_month": month, "valid": valid,
                                   **{f"{metric}_delta": records[month][metric] - records[6][metric] if valid else np.nan
                                      for metric in ("mean", "max", "high")}})

s = pd.DataFrame(summaries)
b = pd.DataFrame(bins)
assert np.allclose(b.groupby(["profile_weighted", "month", "metric"]).mass.sum(), 1)
for metric in ("mean", "max"):
    check = b.loc[b.metric.eq(metric)].groupby(["profile_weighted", "month"]).mean_contribution.sum()
    assert np.allclose(check.to_numpy(), s.set_index(["profile_weighted", "month"])[f"{metric}_average"].reindex(check.index))
save(s, PREFIX, "summary")
save(b, PREFIX, "bins")
save(pd.DataFrame(exclusions), PREFIX, "exclusions")
save(pd.DataFrame(cases), PREFIX, "date_context")
e = pd.DataFrame(exclusions)
sensitivity = e.loc[e.valid].groupby(["profile_weighted", "unit", "comparison_month"])[["mean_delta", "max_delta", "high_delta"]].agg(["min", "max"])
sensitivity.columns = ["_".join(c) for c in sensitivity.columns]
save(sensitivity.reset_index(), PREFIX, "sensitivity")
finish(PREFIX, {"summary": summaries, "excluded_cases": len(e), "invalid_exclusions": int((~e.valid).sum()),
               "sensitivity": sensitivity.reset_index().to_dict("records")})
