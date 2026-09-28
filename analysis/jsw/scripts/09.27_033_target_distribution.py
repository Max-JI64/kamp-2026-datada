"""Compare the two continuous prediction targets in the 032 context."""
from hashlib import sha256
import json
import numpy as np
import pandas as pd
from analysis_common import TABLES, SOURCE, SOURCE_HASH, PROFILES, PROFILE_HASH, load, save, finish

PREFIX = "09.27_033"
assert sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH
assert sha256(PROFILES.read_bytes()).hexdigest() == PROFILE_HASH
config = json.loads((TABLES / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
assert len(config["targets"]) == 2
_, data = load()
data["target_mean"] = data["평균"].astype(float)
data["target_max"] = data.peak.astype(float)
selected = data.loc[data.weekend.eq(0) & data.production_positive.eq(1)]
frames = {"all_valid": data, "positive_weekday": selected}
distribution, standardization = [], []
for scope, frame in frames.items():
    for period, part in frame.groupby("period"):
        for target in ("target_mean", "target_max"):
            x = part[target]
            distribution.append({"scope": scope, "period": period, "target": target,
                                 "n": len(part), "days": int(part.date.nunique()),
                                 "mean": float(x.mean()), "median": float(x.median()),
                                 "q90": float(x.quantile(.9)), "q95": float(x.quantile(.95))})
for target in ("target_mean", "target_max"):
    for weighted in (False, True):
        cell_rows = []
        for hour in range(24):
            row = {"target": target, "hour": hour, "profile_weighted": weighted}
            for period, tag in (("Jan-Jun", "earlier"), ("Jul-Aug", "later")):
                part = selected.loc[(selected.period == period) & (selected.hour == hour)]
                w = part.profile_weight if weighted else pd.Series(1.0, index=part.index)
                row[f"{tag}_n"] = len(part)
                row[f"{tag}_weight"] = float(w.sum())
                row[f"{tag}_weighted_sum"] = float((w * part[target]).sum())
                row[f"{tag}_mean"] = row[f"{tag}_weighted_sum"] / row[f"{tag}_weight"]
            cell_rows.append(row)
        cells = pd.DataFrame(cell_rows)
        assert cells.earlier_n.gt(0).all() and cells.later_n.gt(0).all()
        pooled = cells.earlier_weight + cells.later_weight
        ref = pooled / pooled.sum()
        early = float((ref * cells.earlier_mean).sum())
        late = float((ref * cells.later_mean).sum())
        standardization.append({"target": target, "profile_weighted": weighted,
                                "common_hours": 24, "earlier_n": int(cells.earlier_n.sum()),
                                "later_n": int(cells.later_n.sum()),
                                "earlier_raw_mean": float(cells.earlier_weighted_sum.sum() / cells.earlier_weight.sum()),
                                "later_raw_mean": float(cells.later_weighted_sum.sum() / cells.later_weight.sum()),
                                "earlier_standardized_mean": early, "later_standardized_mean": late,
                                "standardized_difference": late - early})
        if not weighted:
            save(cells, PREFIX, target + "_hour_cells")
save(pd.DataFrame(distribution), PREFIX, "distribution")
save(pd.DataFrame(standardization), PREFIX, "standardized_means")
finish(PREFIX, {"distribution": distribution, "standardization": standardization})
