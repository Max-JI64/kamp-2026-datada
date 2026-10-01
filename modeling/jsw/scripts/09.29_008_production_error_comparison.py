"""M008: posthoc production-condition errors from saved predictions; no fitting."""
from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent.parent
PREFIX = "09.29_008"
METHODS = ["hgb_calendar", "lag1", "lag24", "lag168"]
TARGETS = ["mean", "peak"]
FROZEN = HERE / "tables" / f"{PREFIX}_frozen.json"
SCORES = HERE / "tables" / f"{PREFIX}_condition_scores.csv"
DIFFS = HERE / "tables" / f"{PREFIX}_method_differences.csv"
ZERO_DATES = HERE / "tables" / f"{PREFIX}_zero_date_contributions.csv"
VERIFICATION = HERE / "tables" / f"{PREFIX}_verification.json"
SOURCE = ROOT / "data/origin/okm_augumented_2021.csv"
PRED = HERE / "predictions/09.29_006_review_predictions.csv"
M005 = HERE / "predictions/09.29_005_row_audit.csv"
M006_SCORES = HERE / "tables/09.29_006_scores.csv"
M005_PRODUCTION = HERE / "tables/09.29_005_production_scores.csv"
E003_CONDITIONS = ROOT / "EDA/jsw/tables/09.23_003_conditional_scores.csv"
COVERAGE = HERE / "predictions/09.29_006_target_coverage.csv"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def freeze() -> None:
    if FROZEN.exists():
        raise FileExistsError(f"Frozen conditions already exist: {FROZEN}")
    contract = {
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evaluation_period": "2021-07-01 00:00:00 <= record_key < 2021-09-01 00:00:00",
        "source_prediction": str(PRED.relative_to(ROOT)).replace("\\", "/"),
        "prediction_sha256": digest(PRED),
        "source_sha256": digest(SOURCE),
        "targets": TARGETS,
        "methods": METHODS,
        "common_rows": "intersection of all four methods for each target; both targets must share record keys",
        "production_level": {"zero": "production == 0", "positive_low": "0 < production <= 487", "positive_high": "production > 487", "unknown": "missing or invalid production"},
        "positive_boundary": 487,
        "production_change": {"increase": "production(t) - production(t-1h) > 0", "decrease": "< 0", "same": "== 0", "unknown": "missing or invalid exact-hour pair"},
        "bias": "mean(prediction - actual); positive means overprediction",
        "mae_difference": "HGB MAE - baseline MAE; negative means HGB improvement",
        "periods": ["all", "7", "8"],
        "no_fit": True,
    }
    write_json(FROZEN, contract)


def scored(group: pd.DataFrame) -> dict:
    e = group.prediction.to_numpy(dtype=float) - group.actual.to_numpy(dtype=float)
    return {"n": len(group), "dates": group.record_key.dt.date.nunique(),
            "mae": float(np.abs(e).mean()), "bias": float(e.mean()),
            "mean_shortfall": float(np.maximum(-e, 0).mean())}


def run() -> None:
    cfg = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert cfg["methods"] == METHODS and cfg["targets"] == TARGETS and cfg["positive_boundary"] == 487
    assert cfg["source_sha256"] == digest(SOURCE) and cfg["prediction_sha256"] == digest(PRED)
    pred = pd.read_csv(PRED, encoding="utf-8-sig", parse_dates=["record_key"])
    assert not pred.duplicated(["record_key", "target", "method"]).any()
    assert pred.stage.eq("Jul-Aug").all() and pred.input_available.eq(True).all()
    assert pred.record_key.between("2021-07-01", "2021-08-31 23:00:00").all()
    assert pred.loc[pred.method.isin(METHODS), ["actual", "prediction"]].notna().all().all()
    common = None
    for target in TARGETS:
        sets = [set(pred.loc[pred.target.eq(target) & pred.method.eq(method), "record_key"]) for method in METHODS]
        assert all(sets), (target, [len(s) for s in sets])
        target_common = set.intersection(*sets)
        common = target_common if common is None else common
        assert target_common == common, "Targets have different common record keys"
    assert len(common) == 1344
    p = pred[pred.target.isin(TARGETS) & pred.method.isin(METHODS) & pred.record_key.isin(common)].copy()
    assert len(p) == len(common) * len(TARGETS) * len(METHODS)
    assert p.groupby("record_key").source_data_row.nunique().eq(1).all()
    assert p.groupby(["record_key", "target"]).actual.nunique().eq(1).all()
    assert p.groupby("record_key").production_posthoc.nunique().eq(1).all()
    assert p.groupby("record_key").month.nunique().eq(1).all()
    assert p.groupby("record_key").size().eq(8).all()

    raw = pd.read_csv(SOURCE, encoding="utf-8-sig", usecols=["날짜", "시간", "생산량", "평균", "15분", "30분", "45분", "60분"])
    raw["source_data_row"] = np.arange(1, len(raw) + 1)
    valid = raw[raw["시간"].between(0, 23)].copy()
    valid["record_key"] = pd.to_datetime(valid["날짜"].astype(str), format="%Y%m%d") + pd.to_timedelta(valid["시간"], unit="h")
    assert valid.record_key.is_unique
    valid["peak"] = valid[["15분", "30분", "45분", "60분"]].max(axis=1)
    valid["mean"] = valid["평균"]
    base = valid.set_index("record_key").sort_index()
    base["previous_production"] = base["생산량"].reindex(base.index - pd.Timedelta(hours=1)).to_numpy()
    representative = p[p.target.eq("mean") & p.method.eq("hgb_calendar")].copy().set_index("record_key").sort_index()
    assert representative.index.equals(pd.Index(sorted(common), name="record_key"))
    np.testing.assert_array_equal(representative.source_data_row, base.loc[representative.index, "source_data_row"])
    np.testing.assert_array_equal(representative.production_posthoc, base.loc[representative.index, "생산량"])
    for target in TARGETS:
        for method in METHODS:
            sub = p[p.target.eq(target) & p.method.eq(method)].set_index("record_key").sort_index()
            np.testing.assert_array_equal(sub.source_data_row, base.loc[sub.index, "source_data_row"])
            np.testing.assert_array_equal(sub.actual, base.loc[sub.index, target])
            np.testing.assert_array_equal(sub.production_posthoc, base.loc[sub.index, "생산량"])
            np.testing.assert_allclose(sub.error, sub.prediction - sub.actual, rtol=0, atol=1e-12)
    prod = base.loc[representative.index, "생산량"]
    prior = base.loc[representative.index, "previous_production"]
    assert prod.notna().all() and prod.ge(0).all()
    level = pd.Series(np.select([prod.eq(0), prod.between(0, 487, inclusive="right"), prod.gt(487)],
                                 ["zero", "positive_low", "positive_high"], default="unknown"), index=prod.index)
    delta = prod - prior
    change = pd.Series(np.select([delta.gt(0), delta.lt(0), delta.eq(0)],
                                  ["increase", "decrease", "same"], default="unknown"), index=prod.index)
    assert (level != "unknown").all()
    assert (change.eq("unknown") == prior.isna()).all()
    assert len(level) == sum(level.value_counts()) == len(change) == sum(change.value_counts())
    assert (representative.index.to_series().dt.month == representative.month).all()
    assert (representative.index.to_series().diff().dropna() >= pd.Timedelta(hours=1)).all()
    p["level"] = p.record_key.map(level)
    p["change"] = p.record_key.map(change)
    p["period"] = p.record_key.dt.month.astype(str)
    all_p = p.assign(period="all")
    p = pd.concat([all_p, p], ignore_index=True)
    rows = []
    for dimension in ["level", "change"]:
        for (period, target, condition, method), g in p.groupby(["period", "target", dimension, "method"], sort=True):
            rows.append({"period": period, "target": target, "dimension": dimension,
                         "condition": condition, "method": method, **scored(g)})
    scores = pd.DataFrame(rows).sort_values(["period", "target", "dimension", "condition", "method"]).reset_index(drop=True)
    diffs = []
    keys = ["period", "target", "dimension", "condition"]
    for group_key, group in scores.groupby(keys, sort=True):
        by_method = group.set_index("method")
        assert set(by_method.index) == set(METHODS)
        for baseline in METHODS[1:]:
            hgb, other = by_method.loc["hgb_calendar"], by_method.loc[baseline]
            assert hgb.n == other.n and hgb.dates == other.dates
            diffs.append(dict(zip(keys, group_key)) | {"baseline": baseline, "n": int(hgb.n), "dates": int(hgb.dates),
                "hgb_mae": float(hgb.mae), "baseline_mae": float(other.mae),
                "hgb_minus_baseline_mae": float(hgb.mae - other.mae),
                "hgb_bias": float(hgb.bias), "baseline_bias": float(other.bias)})
    differences = pd.DataFrame(diffs).sort_values(keys + ["baseline"]).reset_index(drop=True)

    zero = p[p.period.ne("all") & p.level.eq("zero") & p.method.isin(["hgb_calendar", "lag1"])].copy()
    zero["absolute_error"] = (zero.prediction - zero.actual).abs()
    zero_pair = zero.pivot(index=["period", "target", "record_key"], columns="method", values="absolute_error")
    assert zero_pair.notna().all().all()
    zero_pair["difference"] = zero_pair.hgb_calendar - zero_pair.lag1
    zero_pair = zero_pair.reset_index()
    zero_pair["date"] = zero_pair.record_key.dt.date.astype(str)
    zero_dates = zero_pair.groupby(["period", "target", "date"], as_index=False).agg(
        n=("difference", "size"), sum_hgb_minus_lag1_absolute_error=("difference", "sum"),
        mean_hgb_minus_lag1_absolute_error=("difference", "mean"))

    checks = []
    old = pd.read_csv(M006_SCORES, encoding="utf-8-sig")
    for period in ["7", "8", "all"]:
        subset = all_p if period == "all" else p[p.period.eq(period)]
        for target in TARGETS:
            for method in METHODS:
                g = subset[subset.target.eq(target) & subset.method.eq(method)]
                metric = scored(g)
                if period != "all":
                    existing = old[(old.period.astype(str).eq(period)) & old.target.eq(target) & old.method.eq(method)]
                    assert len(existing) == 1
                    assert metric["n"] == int(existing.iloc[0].n)
                    np.testing.assert_allclose([metric["mae"], metric["bias"]], existing[["mae", "bias"]].iloc[0].to_numpy(dtype=float), rtol=0, atol=1e-9)
                for dimension in ["level", "change"]:
                    s = scores[(scores.period.eq(period)) & scores.target.eq(target) & scores.method.eq(method) & scores.dimension.eq(dimension)]
                    assert s.n.sum() == metric["n"]
                    np.testing.assert_allclose([(s.n*s.mae).sum()/s.n.sum(), (s.n*s.bias).sum()/s.n.sum()],
                                               [metric["mae"], metric["bias"]], rtol=0, atol=1e-10)
                checks.append({"period": period, "target": target, "method": method, **metric})
    coverage = pd.read_csv(COVERAGE, encoding="utf-8-sig", parse_dates=["record_key"])
    coverage = coverage[coverage.stage.eq("Jul-Aug")]
    assert len(coverage) == 1440 and coverage.eligible.sum() == 1344
    assert set(coverage.loc[coverage.eligible, "record_key"]) == common
    assert coverage.loc[~coverage.eligible].month.value_counts().to_dict() == {7: 96}
    old_prod = pd.read_csv(M005_PRODUCTION, encoding="utf-8-sig")
    for period in ["7", "8"]:
        for target in TARGETS:
            for method in ["hgb_calendar", "lag1"]:
                condition = scores[(scores.period.eq(period)) & scores.target.eq(target) &
                                   scores.dimension.eq("level") & scores.method.eq(method)]
                for name, sub in [("zero", condition[condition.condition.eq("zero")]),
                                  ("positive", condition[condition.condition.ne("zero")])]:
                    old_row = old_prod[(old_prod.stage.eq("Jul-Aug")) & old_prod.month.eq(int(period)) &
                                       old_prod.target.eq(target) & old_prod.method.eq(method) &
                                       old_prod.production_condition.eq(name)]
                    assert len(old_row) == 1
                    assert sub.n.sum() == int(old_row.iloc[0].n)
                    np.testing.assert_allclose([(sub.n*sub.mae).sum()/sub.n.sum(), (sub.n*sub.bias).sum()/sub.n.sum()],
                                               old_row[["mae", "bias"]].iloc[0].to_numpy(dtype=float), rtol=0, atol=1e-9)
    e003 = pd.read_csv(E003_CONDITIONS, encoding="utf-8-sig")
    e003_count = 0
    for dimension, old_dimension in [("level", "production_level"), ("change", "production_direction")]:
        for _, row in scores[(scores.period.eq("all")) & scores.dimension.eq(dimension) &
                             scores.method.ne("hgb_calendar")].iterrows():
            old_value = "unchanged" if row.condition == "same" else row.condition
            old_row = e003[(e003.condition.eq(old_dimension)) & e003.value.eq(old_value) &
                           e003.target.eq(row.target) & e003.method.eq(row.method)]
            assert len(old_row) == 1
            assert row.n == int(old_row.iloc[0].rows)
            np.testing.assert_allclose([row.mae, row.bias],
                                       old_row[["mae", "bias_pred_minus_actual"]].iloc[0].to_numpy(dtype=float),
                                       rtol=0, atol=1e-9)
            e003_count += 1
    assert e003_count == 36
    audit = pd.read_csv(M005, encoding="utf-8-sig", usecols=["record_key", "target", "method", "actual", "prediction"], parse_dates=["record_key"])
    audit = audit[audit.record_key.isin(common) & audit.target.isin(TARGETS) & audit.method.isin(METHODS)]
    matched = p[p.period.eq("all")][["record_key", "target", "method", "actual", "prediction"]].merge(
        audit, on=["record_key", "target", "method"], how="left", validate="one_to_one", suffixes=("", "_m005"))
    # M005 audits HGB and lag1; lag24/lag168 were added in M006 from saved M001 predictions.
    assert matched.loc[matched.method.isin(["hgb_calendar", "lag1"]), "prediction_m005"].notna().all()
    for field in ["actual", "prediction"]:
        sub = matched[matched[f"{field}_m005"].notna()]
        np.testing.assert_allclose(sub[field], sub[f"{field}_m005"], rtol=0, atol=1e-12)
    scores.to_csv(SCORES, index=False, encoding="utf-8-sig")
    differences.to_csv(DIFFS, index=False, encoding="utf-8-sig")
    zero_dates.to_csv(ZERO_DATES, index=False, encoding="utf-8-sig")
    write_json(VERIFICATION, {"completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in [SOURCE, PRED, M005, M006_SCORES, M005_PRODUCTION, E003_CONDITIONS, COVERAGE]},
        "frozen_sha256": digest(FROZEN), "common_hours": len(common), "normal_hours": len(coverage),
        "unavailable_hours": len(coverage) - len(common), "months": {str(k): int(v) for k, v in representative.month.value_counts().sort_index().items()},
        "level_counts": {str(k): int(v) for k, v in level.value_counts().items()},
        "change_counts": {str(k): int(v) for k, v in change.value_counts().items()},
        "exact_hour_unknown": int(change.eq("unknown").sum()),
        "scores_rows": len(scores), "difference_rows": len(differences),
        "zero_date_rows": len(zero_dates), "m005_zero_positive_reproduced": True,
        "e003_production_condition_rows_reproduced": e003_count,
        "checks": checks, "no_fit": True})
    print(json.dumps({"common_hours": len(common), "level_counts": level.value_counts().to_dict(),
                      "change_counts": change.value_counts().to_dict(), "scores_rows": len(scores),
                      "difference_rows": len(differences)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["freeze", "run"], required=True)
    args = parser.parse_args()
    freeze() if args.phase == "freeze" else run()
