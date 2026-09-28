"""Check day-to-day association and weekday common support."""
import json
import numpy as np
import pandas as pd
from analysis_common import TABLES, save, finish

PREFIX = "09.27_022"
cfg = json.loads((TABLES / f"{PREFIX}_frozen.json").read_text(encoding="utf-8"))
daily = pd.read_csv(TABLES / cfg["input"], encoding="utf-8-sig", parse_dates=["date"]).set_index("date").sort_index()
daily["high_day"] = daily.high_hours.gt(0).astype(int)
before = daily.index - pd.Timedelta(days=1)
daily["previous_high_day"] = daily.high_day.reindex(before).to_numpy()
daily["previous_period"] = daily.period.reindex(before).to_numpy()
pair = daily.loc[daily.previous_high_day.notna() & daily.period.eq(daily.previous_period)].copy()
pair["previous_high_day"] = pair.previous_high_day.astype(int)
pair["weekday"] = pair.index.dayofweek
assert len(pair) == 237
raw, cells, summary = [], [], []
for period, group in pair.groupby("period"):
    for weighted in (False, True):
        for state, part in group.groupby("previous_high_day"):
            w = part.profile_weight.to_numpy() if weighted else np.ones(len(part))
            raw.append({"period": period, "profile_weighted": weighted, "previous_high_day": state,
                        "pairs": len(part), "current_high_days": int(part.high_day.sum()),
                        "weighted_exposure": w.sum(), "weighted_current_high": np.sum(w * part.high_day),
                        "current_high_fraction": np.average(part.high_day, weights=w)})
        rows = []
        for weekday, part in group.groupby("weekday"):
            row = {"period": period, "profile_weighted": weighted, "weekday": weekday}
            for state in (0, 1):
                sample = part.loc[part.previous_high_day.eq(state)]
                w = sample.profile_weight.to_numpy() if weighted else np.ones(len(sample))
                row[f"state{state}_n"] = len(sample)
                row[f"state{state}_high"] = int(sample.high_day.sum())
                row[f"state{state}_exposure_weight"] = float(w.sum())
                row[f"state{state}_high_weight"] = float(np.sum(w * sample.high_day))
            row["common"] = row["state0_n"] > 0 and row["state1_n"] > 0
            rows.append(row)
        table = pd.DataFrame(rows)
        common = table.loc[table.common]
        result = {"period": period, "profile_weighted": weighted, "all_pairs": len(group),
                  "common_weekdays": len(common),
                  "small_cells_either_n_le2": int(common[["state0_n", "state1_n"]].min(axis=1).le(2).sum())}
        refs = common.state0_exposure_weight + common.state1_exposure_weight
        refs = refs / refs.sum()
        for state in (0, 1):
            result[f"state{state}_all_pairs"] = int(group.previous_high_day.eq(state).sum())
            result[f"state{state}_common_pairs"] = int(common[f"state{state}_n"].sum())
            result[f"state{state}_standardized_fraction"] = float(np.sum(refs * common[f"state{state}_high_weight"] / common[f"state{state}_exposure_weight"])) if len(common) else np.nan
        result["standardized_difference"] = result["state1_standardized_fraction"] - result["state0_standardized_fraction"]
        if len(common):
            assert np.isclose(refs.sum(), 1)
        cells.extend(rows)
        summary.append(result)
save(pd.DataFrame(raw), PREFIX, "raw_transitions")
save(pd.DataFrame(cells), PREFIX, "weekday_cells")
save(pd.DataFrame(summary), PREFIX, "standardization")
finish(PREFIX, {"raw_transitions": raw, "standardization": summary,
                "excluded_current_dates": [str(d.date()) for d in daily.index.difference(pair.index)]})
