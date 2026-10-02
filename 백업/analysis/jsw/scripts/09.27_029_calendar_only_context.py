"""Follow up 005 by matching calendar cells without same-time production."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analysis_common import BASE, TABLES, SOURCE, SOURCE_HASH, PROFILES, PROFILE_HASH, save, finish
from hashlib import sha256

PREFIX = "09.27_029"
assert sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH
assert sha256(PROFILES.read_bytes()).hexdigest() == PROFILE_HASH
config = json.loads((TABLES / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
assert config["strata"] == ["exact hour 0-23", "weekend 0-1"]
records = pd.read_csv(TABLES / "09.27_004_records.csv", encoding="utf-8-sig")
assert len(records) == 5784 and set(records.period) == {"Jan-Jun", "Jul-Aug"}
assert records.groupby("period").size().to_dict() == {"Jan-Jun": 4344, "Jul-Aug": 1440}
assert records.high.sum() == 455 and records.onset.sum() == 224
assert records.high.eq(records.peak.ge(182)).all()
assert records.hour.between(0, 23).all() and records.weekend.isin([0, 1]).all()
assert records.profile_weight.gt(0).all()

rows, summaries = [], []
for metric in ("high", "onset"):
    subset = records if metric == "high" else records.loc[records.eligible].copy()
    for profile_weighted in (False, True):
        frame = subset.copy()
        frame["weight"] = frame.profile_weight if profile_weighted else 1.0
        frame["event_weight"] = frame.weight * frame[metric].astype(float)
        aggregate = frame.groupby(["hour", "weekend", "period"], as_index=False).agg(
            n=(metric, "size"), events=(metric, "sum"), exposure_weight=("weight", "sum"),
            event_weight=("event_weight", "sum"), dates=("date", "nunique"), profiles=("profile", "nunique"))
        for hour in range(24):
            for weekend in (0, 1):
                row = {"metric": metric, "profile_weighted": profile_weighted,
                       "hour": hour, "weekend": weekend}
                for period, tag in (("Jan-Jun", "earlier"), ("Jul-Aug", "later")):
                    part = aggregate.loc[(aggregate.hour == hour) & (aggregate.weekend == weekend) & (aggregate.period == period)]
                    assert len(part) == 1
                    for name in ("n", "events", "exposure_weight", "event_weight", "dates", "profiles"):
                        row[f"{tag}_{name}"] = part.iloc[0][name]
                row["common"] = row["earlier_n"] > 0 and row["later_n"] > 0
                rows.append(row)
        table = pd.DataFrame(rows[-48:])
        support = table.loc[table.common].copy()
        pooled = support.earlier_exposure_weight + support.later_exposure_weight
        reference = pooled / pooled.sum()
        result = {"metric": metric, "profile_weighted": profile_weighted,
                  "common_cells": len(support), "total_cells": len(table),
                  "cells_either_n_le_5": int(((support.earlier_n <= 5) | (support.later_n <= 5)).sum())}
        for tag in ("earlier", "later"):
            result[f"{tag}_n"] = int(table[f"{tag}_n"].sum())
            result[f"{tag}_events"] = int(table[f"{tag}_events"].sum())
            result[f"{tag}_min_cell_n"] = int(support[f"{tag}_n"].min())
            result[f"{tag}_raw_rate"] = float(table[f"{tag}_event_weight"].sum() / table[f"{tag}_exposure_weight"].sum())
            result[f"{tag}_standardized_rate"] = float((reference * support[f"{tag}_event_weight"] / support[f"{tag}_exposure_weight"]).sum())
        result["raw_difference"] = result["later_raw_rate"] - result["earlier_raw_rate"]
        result["standardized_difference"] = result["later_standardized_rate"] - result["earlier_standardized_rate"]
        summaries.append(result)

strata = pd.DataFrame(rows)
save(strata, PREFIX, "strata")
save(pd.DataFrame(summaries), PREFIX, "standardization")
prior = json.loads((TABLES / "09.27_005_facts.json").read_text(encoding="utf-8"))
for row in summaries:
    match = next(x for x in prior["standardization"] if x["metric"] == row["metric"] and x["profile_weighted"] == row["profile_weighted"])
    assert np.isclose(row["earlier_raw_rate"], match["earlier_raw_rate"])
    assert np.isclose(row["later_raw_rate"], match["later_raw_rate"])
assert all(row["common_cells"] == 48 for row in summaries)
finish(PREFIX, {"standardization": summaries, "input_reuse": "09.27_004_records.csv", "prior_context": "09.27_005_facts.json"})
