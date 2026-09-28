"""Describe within-date production record shapes without inferring ERP allocation."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).parent.parent
SOURCE = BASE / "../../data/origin/okm_augumented_2021.csv"
TABLES = BASE / "tables"
PREFIX = "09.28_023"
EXPECTED_HASH = "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"


def save(frame: pd.DataFrame, label: str) -> None:
    frame.to_csv(TABLES / f"{PREFIX}_{label}.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    assert sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED_HASH
    raw = pd.read_csv(SOURCE, encoding="utf-8-sig", usecols=["날짜", "시간", "생산량"])
    scoped = raw.loc[raw["날짜"].lt(20210901)].copy()
    invalid = scoped.loc[~scoped["시간"].between(0, 23)].copy()
    valid = scoped.loc[scoped["시간"].between(0, 23)].copy()
    assert (len(raw), len(scoped), len(invalid), len(valid)) == (6168, 5832, 48, 5784)
    assert not valid["생산량"].isna().any()
    assert valid.groupby("날짜")["시간"].nunique().eq(24).all()
    daily = []
    for date, group in valid.groupby("날짜", sort=True):
        group = group.sort_values("시간")
        values = group["생산량"].to_numpy()
        changes = values[1:] != values[:-1]
        signs = values > 0
        runs = np.diff(np.r_[0, np.flatnonzero(changes) + 1, 24])
        unique = int(np.unique(values).size)
        nondecreasing = bool(np.all(np.diff(values) >= 0))
        nonincreasing = bool(np.all(np.diff(values) <= 0))
        if unique == 1:
            shape = "one_value_all_day"
        elif nondecreasing or nonincreasing:
            shape = "monotone_recorded_values"
        else:
            shape = "mixed_recorded_values"
        daily.append({"date": date, "period": "Jan-Jun" if date < 20210701 else "Jul-Aug",
                      "unique_values": unique, "shape": shape, "zero_hours": int((values == 0).sum()),
                      "positive_hours": int(signs.sum()), "zero_positive_transitions": int(np.count_nonzero(signs[1:] != signs[:-1])),
                      "exact_value_changes": int(changes.sum()), "longest_equal_run": int(runs.max()),
                      "first_value": values[0], "last_value": values[-1], "min_value": values.min(), "max_value": values.max(),
                      "sum_of_hour_records": values.sum()})
    days = pd.DataFrame(daily)
    summary = days.groupby(["period", "shape"], sort=True).agg(days=("date", "size"),
        median_unique_values=("unique_values", "median"), median_equal_run=("longest_equal_run", "median"),
        days_with_zero_positive_transition=("zero_positive_transitions", lambda s: int(s.gt(0).sum()))).reset_index()
    distributions = days.groupby(["period", "unique_values"], sort=True).size().rename("days").reset_index()
    examples = days.sort_values(["period", "shape", "date"]).groupby(["period", "shape"], sort=True).head(2)
    invalid_summary = invalid.groupby("날짜", sort=True).agg(invalid_rows=("시간", "size"),
                                                              distinct_invalid_hours=("시간", "nunique")).reset_index()
    assert len(days) == 241 and days["positive_hours"].sum() + days["zero_hours"].sum() == 5784
    assert summary["days"].sum() == 241 and distributions["days"].sum() == 241
    assert set(invalid_summary["날짜"]) == {20210713, 20210715} and invalid["생산량"].eq(0).all()
    prior = pd.read_csv(BASE / "tables/09.23_011_daily_profile_summary.csv", encoding="utf-8-sig")
    joined = days.merge(prior[["date", "production_total"]], on="date", validate="one_to_one")
    assert len(joined) == 241 and np.allclose(joined["sum_of_hour_records"], joined["production_total"])
    save(days, "daily")
    save(summary, "shape_summary")
    save(distributions, "unique_distribution")
    save(examples, "examples")
    save(invalid_summary, "invalid_time_dates")
    facts = {"source_sha256": EXPECTED_HASH, "source_rows": len(raw), "scoped_rows": len(scoped),
             "invalid_time_rows": len(invalid), "valid_hours": len(valid), "valid_days": len(days),
             "shape_summary": summary.to_dict("records"), "days_with_zero_positive_transition": int(days["zero_positive_transitions"].gt(0).sum()),
             "days_with_multiple_exact_values": int(days["unique_values"].gt(1).sum())}
    (TABLES / f"{PREFIX}_facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(facts, ensure_ascii=True))


if __name__ == "__main__":
    main()
