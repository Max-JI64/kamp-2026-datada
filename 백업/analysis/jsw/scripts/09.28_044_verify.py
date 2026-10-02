"""Independently check A044 source counts, cap arithmetic, and feasible transfers."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_common import SOURCE, TABLES

P = "09.28_044"
SLOTS = ("15분", "30분", "45분", "60분")


def main() -> None:
    by_day = defaultdict(list)
    invalid = 0
    with SOURCE.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            date, hour = int(row["날짜"]), int(row["시간"])
            if date >= 20210901:
                continue
            if not 0 <= hour <= 23:
                invalid += 1
                continue
            by_day[date].append((hour, [float(row[s]) for s in SLOTS]))
    assert len(by_day) == 241 and invalid == 48
    daily = pd.read_csv(TABLES / f"{P}_daily_concentration.csv", encoding="utf-8-sig").set_index("date")
    caps = pd.read_csv(TABLES / f"{P}_direct_caps.csv", encoding="utf-8-sig")
    sims = pd.read_csv(TABLES / f"{P}_redistribution.csv", encoding="utf-8-sig")
    assert (len(daily), len(caps), len(sims)) == (241, 723, 1446)
    max_zero_gap = 0.0
    zero_gap_dates = []
    for date_int, rows in by_day.items():
        assert sorted(h for h, _ in rows) == list(range(24))
        values = np.array([v for _, cells in sorted(rows) for v in cells], dtype=float)
        key = f"{date_int:08d}"
        date = f"{key[:4]}-{key[4:6]}-{key[6:]}"
        recorded = daily.loc[date]
        assert len(values) == 96 and values.sum() == recorded["sum_values"]
        assert values.max() == recorded["max_value"]
        assert np.isclose(values.mean() / values.max(), recorded["mean_max_ratio"])
        assert int((values == 0).sum()) == recorded["zero_cells"]
        for _, row in caps.loc[caps["date"].eq(date)].iterrows():
            t = row["cutoff"]
            excess = np.maximum(values - t, 0)
            assert int((values >= t).sum()) == row["cells_ge"]
            assert int((values > t).sum()) == row["cells_gt"]
            assert excess.sum() == row["excess_value_sum"]
            assert excess.max() == row["max_excess"]
        for _, row in sims.loc[sims["date"].eq(date)].iterrows():
            t = float(row["optimal_max"])
            eligible = np.ones(96, dtype=bool) if row["receive_zero"] else values > 0
            removed = np.maximum(values - t, 0)
            capacity = np.maximum(t - values[eligible], 0)
            assert np.isclose(removed.sum(), row["moved_value_sum"], atol=1e-7)
            assert removed.sum() <= row["budget_value_sum"] + 1e-7
            assert removed.sum() <= capacity.sum() + 1e-7
            adjusted = np.minimum(values, t)
            if removed.sum() > 0:
                assert capacity.sum() > 0
                adjusted[eligible] += removed.sum() * capacity / capacity.sum()
            assert np.isclose(adjusted.sum(), values.sum(), atol=1e-6)
            assert adjusted.max() <= t + 1e-6
            assert (adjusted >= -1e-9).all()
            if not row["receive_zero"]:
                assert (adjusted[values == 0] == 0).all()
            floor = values.sum() / eligible.sum()
            if row["budget_fraction"] == 0:
                assert t == values.max()
            elif t > floor + 1e-5:
                assert np.isclose(removed.sum(), row["budget_value_sum"], atol=1e-5)
        if (values == 0).any():
            a = sims.loc[sims["date"].eq(date) & sims["budget_fraction"].eq(.10)]
            gap = float(a.loc[~a["receive_zero"], "optimal_max"].iat[0] -
                        a.loc[a["receive_zero"], "optimal_max"].iat[0])
            max_zero_gap = max(max_zero_gap, gap)
            zero_gap_dates.append({"date": date, "gap_10pct": round(gap, 4)})
    eda = pd.read_csv(TABLES.parent.parent.parent / "EDA/jsw/tables/09.28_024_daily_concentration.csv",
                      encoding="utf-8-sig")
    eda["date"] = pd.to_datetime(eda["date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
    eda = eda.set_index("date").sort_index()
    ours = daily.sort_index()
    assert eda.index.equals(ours.index)
    assert np.allclose(eda["exact_mean"], ours["mean_value"])
    assert np.allclose(eda["peak"], ours["max_value"])
    assert np.allclose(eda["flatness_mean_over_peak"], ours["mean_max_ratio"])
    assert (eda["zero_slots"] == ours["zero_cells"]).all()
    high182 = caps.loc[caps["cutoff"].eq(182)].set_index("date").sort_index()
    assert (eda["slots_at_least_182"] == high182["cells_ge"]).all()
    eda_month = pd.read_csv(TABLES.parent.parent.parent / "EDA/jsw/tables/09.28_024_monthly_maxima.csv",
                            encoding="utf-8-sig")
    ours_month = pd.read_csv(TABLES / f"{P}_monthly_max.csv", encoding="utf-8-sig")
    assert np.allclose(eda_month["monthly_peak_valid_time"], ours_month["normal_max"])
    assert (eda_month["complete_days"] == ours_month["complete_days"]).all()
    print(json.dumps({"verified_days": len(by_day), "invalid_rows": invalid,
                      "eda024_daily_matches": len(eda), "eda024_monthly_matches": len(eda_month),
                      "max_zero_receiving_gap_10pct": round(max_zero_gap, 4),
                      "zero_dates": zero_gap_dates}, ensure_ascii=False))


if __name__ == "__main__":
    main()
