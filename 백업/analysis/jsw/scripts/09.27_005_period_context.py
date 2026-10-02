"""Common-context period differences, contributor influence and block variability."""
import numpy as np
import pandas as pd
from analysis_common import load, save, finish

PREFIX = "09.27_005"
KEYS = ["hour", "weekend", "production_positive"]
PERIODS = ["Jan-Jun", "Jul-Aug"]


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    return float(values[np.searchsorted(np.cumsum(weights) / weights.sum(), q, side="left")])


def main():
    _, data = load()
    records = pd.read_csv("tables/09.27_004_records.csv", encoding="utf-8-sig", parse_dates=["timestamp", "date"]).set_index("timestamp")
    assert records.index.equals(data.index)
    for col in ("high", "onset", "eligible"):
        records[col] = records[col].astype(bool)
    data["high"] = records["high"]
    data["eligible"] = records["eligible"]
    data["onset"] = records["onset"]
    strata_rows, summaries, distributions = [], [], []
    for metric, mask, label in (("onset", data["eligible"], "onset"), ("high", pd.Series(True, index=data.index), "high")):
        sample = data.loc[mask]
        for weighted in (False, True):
            cells = []
            for key, group in sample.groupby(KEYS):
                cell = dict(zip(KEYS, key))
                for period, tag in zip(PERIODS, ("earlier", "later")):
                    part = group.loc[group["period"].eq(period)]
                    w = part["profile_weight"] if weighted else pd.Series(1., index=part.index)
                    cell[f"{tag}_n"] = len(part)
                    cell[f"{tag}_events"] = int(part[label].sum())
                    cell[f"{tag}_exposure_weight"] = float(w.sum())
                    cell[f"{tag}_event_weight"] = float(w.loc[part[label]].sum())
                    cell[f"{tag}_dates"] = part["date"].nunique()
                    cell[f"{tag}_profiles"] = part["profile"].nunique()
                cell["common"] = cell["earlier_n"] > 0 and cell["later_n"] > 0
                cell["metric"] = metric
                cell["profile_weighted"] = weighted
                cells.append(cell)
            table = pd.DataFrame(cells)
            strata_rows.extend(cells)
            support = table.loc[table["common"]]
            pooled = support["earlier_exposure_weight"] + support["later_exposure_weight"]
            ref = pooled / pooled.sum()
            out = {"metric": metric, "profile_weighted": weighted,
                   "common_cells": len(support), "total_cells": len(table),
                   "cells_either_n_le_5": int(support[["earlier_n", "later_n"]].min(axis=1).le(5).sum())}
            for tag in ("earlier", "later"):
                out[f"{tag}_raw_rate"] = table[f"{tag}_event_weight"].sum() / table[f"{tag}_exposure_weight"].sum()
                out[f"{tag}_overlap_rate"] = support[f"{tag}_event_weight"].sum() / support[f"{tag}_exposure_weight"].sum()
                out[f"{tag}_standardized_rate"] = float((ref * support[f"{tag}_event_weight"] / support[f"{tag}_exposure_weight"]).sum())
                out[f"{tag}_all_n"] = int(table[f"{tag}_n"].sum())
                out[f"{tag}_overlap_n"] = int(support[f"{tag}_n"].sum())
                out[f"{tag}_excluded_events"] = int(table[f"{tag}_events"].sum() - support[f"{tag}_events"].sum())
                out[f"{tag}_min_cell_n"] = int(support[f"{tag}_n"].min())
            out["raw_difference"] = out["later_raw_rate"] - out["earlier_raw_rate"]
            out["standardized_difference"] = out["later_standardized_rate"] - out["earlier_standardized_rate"]
            assert np.isclose(ref.sum(), 1)
            summaries.append(out)
            if metric == "high":
                weight_lookup = {tuple(row[k] for k in KEYS): ref.loc[i] for i, row in support.iterrows()}
                for period, tag in zip(PERIODS, ("earlier", "later")):
                    xs, ws = [], []
                    for key, group in sample.loc[sample["period"].eq(period)].groupby(KEYS):
                        if key not in weight_lookup:
                            continue
                        base = group["profile_weight"].to_numpy() if weighted else np.ones(len(group))
                        ws.extend(base / base.sum() * weight_lookup[key])
                        xs.extend(group["peak"].to_numpy())
                    result = {"period": period, "profile_weighted": weighted, "mean": float(np.average(xs, weights=ws)),
                              "zero_fraction": float(np.average(np.asarray(xs) == 0, weights=ws))}
                    for q in (.1, .5, .9, .95, .99):
                        result[f"q{q}"] = weighted_quantile(xs, ws, q)
                    distributions.append(result)
    save(pd.DataFrame(strata_rows), PREFIX, "strata")
    save(pd.DataFrame(summaries), PREFIX, "standardization")
    save(pd.DataFrame(distributions), PREFIX, "standardized_distributions")
    raw_distributions = []
    for period, part in data.groupby("period"):
        raw_distributions.append({"period": period, "n": len(part), "mean": part["peak"].mean(),
                                  "std": part["peak"].std(), "zero_fraction": part["peak"].eq(0).mean(),
                                  **{f"q{q}": part["peak"].quantile(q) for q in (.1, .5, .9, .95, .99)}})
    save(pd.DataFrame(raw_distributions), PREFIX, "raw_distributions")
    data["week"] = data["date"].dt.to_period("W-SUN").astype(str)
    influence = []
    for grouping in ("week", "month", "profile"):
        for value in data[grouping].unique():
            retained = data.loc[data[grouping].ne(value)]
            row = {"grouping": grouping, "removed_group": value}
            for period, tag in zip(PERIODS, ("earlier", "later")):
                part = retained.loc[retained["period"].eq(period)]
                row[f"{tag}_n"] = len(part)
                row[f"{tag}_onset_rate"] = part["onset"].sum() / part["eligible"].sum() if part["eligible"].sum() else np.nan
                row[f"{tag}_high_rate"] = part["high"].mean()
            row["onset_difference"] = row["later_onset_rate"] - row["earlier_onset_rate"]
            row["high_difference"] = row["later_high_rate"] - row["earlier_high_rate"]
            influence.append(row)
    influence_table = pd.DataFrame(influence)
    save(influence_table, PREFIX, "leave_group_out")
    influence_ranges = influence_table.groupby("grouping")[["onset_difference", "high_difference"]].agg(["min", "max"])
    influence_ranges.columns = ["_".join(c) for c in influence_ranges.columns]
    save(influence_ranges.reset_index(), PREFIX, "influence_ranges")
    rng = np.random.default_rng(20260927)
    bootstrap = []
    for block in ("date", "week"):
        period_draws = {}
        for period, part in data.groupby("period"):
            agg = part.groupby(block).agg(onsets=("onset", "sum"), eligible=("eligible", "sum"), highs=("high", "sum"), n=("high", "size"))
            draws = rng.integers(0, len(agg), size=(2000, len(agg)))
            totals = agg.to_numpy()[draws].sum(axis=1)
            rates = totals[:, 0] / totals[:, 1]
            high_rates = totals[:, 2] / totals[:, 3]
            period_draws[period] = (rates, high_rates)
            for metric, values in (("onset", rates), ("high", high_rates)):
                bootstrap.append({"block": block, "period": period, "metric": metric, "blocks": len(agg),
                                  "p025": np.quantile(values, .025), "p50": np.quantile(values, .5), "p975": np.quantile(values, .975)})
        for i, metric in enumerate(("onset", "high")):
            values = period_draws["Jul-Aug"][i] - period_draws["Jan-Jun"][i]
            bootstrap.append({"block": block, "period": "later-minus-earlier", "metric": metric, "blocks": np.nan,
                              "p025": np.quantile(values, .025), "p50": np.quantile(values, .5), "p975": np.quantile(values, .975)})
    save(pd.DataFrame(bootstrap), PREFIX, "block_variability")
    finish(PREFIX, {"standardization": summaries, "distributions": distributions,
                    "influence_ranges": influence_ranges.reset_index().to_dict("records"),
                    "bootstrap_differences": [r for r in bootstrap if r["period"] == "later-minus-earlier"]})


if __name__ == "__main__":
    main()
