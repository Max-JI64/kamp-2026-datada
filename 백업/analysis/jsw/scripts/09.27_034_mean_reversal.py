"""Account for the overall/within-context mean reversal without causal claims."""
from hashlib import sha256
import json
import numpy as np
import pandas as pd
from analysis_common import TABLES, SOURCE, SOURCE_HASH, PROFILES, PROFILE_HASH, load, save, finish

PREFIX = "09.27_034"
assert sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH
assert sha256(PROFILES.read_bytes()).hexdigest() == PROFILE_HASH
config = json.loads((TABLES / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
assert "four exhaustive" in config["input"]
_, data = load()
data["target_mean"] = data["평균"].astype(float)
data["target_max"] = data.peak.astype(float)
rows, totals = [], []
for target in ("target_mean", "target_max"):
    period_total = data.groupby("period").size().to_dict()
    for weekend in (0, 1):
        for positive in (0, 1):
            entry = {"target": target, "weekend": weekend, "production_positive": positive}
            for period, tag in (("Jan-Jun", "earlier"), ("Jul-Aug", "later")):
                subset = data.loc[data.period.eq(period) & data.weekend.eq(weekend)
                                  & data.production_positive.eq(positive)]
                assert len(subset) > 0
                entry[f"{tag}_n"] = len(subset)
                entry[f"{tag}_days"] = int(subset.date.nunique())
                entry[f"{tag}_share"] = len(subset) / period_total[period]
                entry[f"{tag}_mean"] = float(subset[target].mean())
            a, b = entry["earlier_share"], entry["later_share"]
            x, y = entry["earlier_mean"], entry["later_mean"]
            entry["mean_difference"] = y - x
            entry["mix_contribution"] = (b - a) * (x + y) / 2
            entry["within_contribution"] = (y - x) * (a + b) / 2
            entry["total_contribution"] = b * y - a * x
            assert np.isclose(entry["mix_contribution"] + entry["within_contribution"],
                              entry["total_contribution"], atol=1e-12)
            rows.append(entry)
    part = pd.DataFrame(rows[-4:])
    first = float(data.loc[data.period.eq("Jan-Jun"), target].mean())
    second = float(data.loc[data.period.eq("Jul-Aug"), target].mean())
    assert np.isclose(part.earlier_share.sum(), 1, atol=1e-12)
    assert np.isclose(part.later_share.sum(), 1, atol=1e-12)
    assert np.isclose((part.earlier_share * part.earlier_mean).sum(), first, atol=1e-12)
    assert np.isclose((part.later_share * part.later_mean).sum(), second, atol=1e-12)
    assert np.isclose(part.total_contribution.sum(), second - first, atol=1e-12)
    totals.append({"target": target, "earlier_mean": first, "later_mean": second,
                   "difference": second - first, "mix_contribution": float(part.mix_contribution.sum()),
                   "within_contribution": float(part.within_contribution.sum())})
save(pd.DataFrame(rows), PREFIX, "stratum_components")
save(pd.DataFrame(totals), PREFIX, "totals")
finish(PREFIX, {"stratum_components": rows, "totals": totals})
