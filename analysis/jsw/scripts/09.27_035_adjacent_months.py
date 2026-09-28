"""Test whether positive-weekday contrasts survive a June reference month."""
from hashlib import sha256
import json
import numpy as np
import pandas as pd
from analysis_common import TABLES, SOURCE, SOURCE_HASH, PROFILES, PROFILE_HASH, load, save, finish

PREFIX = "09.27_035"
assert sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH
assert sha256(PROFILES.read_bytes()).hexdigest() == PROFILE_HASH
setting = json.loads((TABLES / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
assert "adjacent June" in setting["question"]
_, data = load()
work = data.loc[data.month.isin([6, 7, 8]) & data.weekend.eq(0) & data.production_positive.eq(1)].copy()
work["high"] = work.peak.ge(182).astype(float)
work["target_mean"] = work["평균"].astype(float)
work["target_max"] = work.peak.astype(float)
raw, standardized, cells_all = [], [], []
for month, part in work.groupby("month"):
    raw.append({"month": int(month), "hours": len(part), "dates": int(part.date.nunique()),
                "high_hours": int(part.high.sum()), "high_rate": float(part.high.mean()),
                "mean_target_average": float(part.target_mean.mean()),
                "max_target_average": float(part.target_max.mean()),
                "temperature_median": float(part["기온"].median())})
assert {r["month"] for r in raw} == {6, 7, 8}
for metric in ("high", "target_mean", "target_max"):
    for weighted in (False, True):
        frame = work.copy()
        frame["weight"] = frame.profile_weight if weighted else 1.0
        frame["weighted_target"] = frame.weight * frame[metric]
        cells = frame.groupby(["month", "hour"], as_index=False).agg(
            n=(metric, "size"), weight=("weight", "sum"), weighted_target=("weighted_target", "sum"))
        assert len(cells) == 72 and cells.n.min() > 0
        ref = cells.groupby("hour").weight.sum()
        ref = ref / ref.sum()
        means = {}
        for month in (6, 7, 8):
            part = cells.loc[cells.month.eq(month)].set_index("hour").reindex(range(24))
            means[month] = float((ref * part.weighted_target / part.weight).sum())
            assert np.isclose(float(part.weighted_target.sum() / part.weight.sum()),
                              float(np.average(frame.loc[frame.month.eq(month), metric],
                                               weights=frame.loc[frame.month.eq(month), "weight"])))
        standardized.append({"metric": metric, "profile_weighted": weighted,
                             "june": means[6], "july": means[7], "august": means[8],
                             "july_minus_june": means[7] - means[6],
                             "august_minus_june": means[8] - means[6],
                             "minimum_hour_n_june": int(cells.loc[cells.month.eq(6), "n"].min()),
                             "minimum_hour_n_july": int(cells.loc[cells.month.eq(7), "n"].min()),
                             "minimum_hour_n_august": int(cells.loc[cells.month.eq(8), "n"].min())})
        if not weighted:
            cells["metric"] = metric
            cells_all.append(cells)
save(pd.DataFrame(raw), PREFIX, "monthly_context")
save(pd.DataFrame(standardized), PREFIX, "standardized_months")
save(pd.concat(cells_all, ignore_index=True), PREFIX, "hour_cells")
finish(PREFIX, {"monthly_context": raw, "standardized_months": standardized})
