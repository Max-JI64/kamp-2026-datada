"""Check incremental value of lag-one production and weather on 001 folds."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


HERE = Path(__file__).parent.parent
PREFIX = "09.26_003"
FROZEN = HERE / "tables" / f"{PREFIX}_frozen.json"
BASE_SCRIPT = HERE / "scripts" / "09.26_001_power_models.py"
BASE_PRED = HERE / "predictions" / "09.26_001_development_predictions.csv"
OUT_PRED = HERE / "predictions" / f"{PREFIX}_development_predictions.csv"
OUT_SCORE = HERE / "tables" / f"{PREFIX}_scores.csv"
OUT_COVERAGE = HERE / "tables" / f"{PREFIX}_coverage.csv"


def score(group: pd.DataFrame) -> dict:
    error = group["prediction"].to_numpy() - group["actual"].to_numpy()
    return {"n": len(error), "mae": float(np.abs(error).mean()),
            "bias": float(error.mean()), "under_n": int((error < 0).sum()),
            "shortfall_per_hour": float(np.maximum(-error, 0).mean())}


def main() -> None:
    frozen_bytes = FROZEN.read_bytes()
    frozen = json.loads(frozen_bytes)
    spec = importlib.util.spec_from_file_location("power_001", BASE_SCRIPT)
    base = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(base)
    assert frozen["model_settings"] == base.SETTINGS["hgb"]
    assert frozen["offset_hours"] == -1
    X, labels, _, full, digest = base.read_data(save_catalog=False)
    raw_columns = ["날짜", "시간", *frozen["candidate_groups"]["both"]]
    raw = pd.read_csv(base.SOURCE, encoding="utf-8-sig", usecols=raw_columns)
    valid = raw.loc[raw["날짜"].lt(20210901) & raw["시간"].between(0, 23)].copy()
    valid["record_key"] = pd.to_datetime(valid["날짜"].astype(str), format="%Y%m%d") + pd.to_timedelta(valid["시간"], unit="h")
    assert valid["record_key"].is_unique and len(valid) == 5784
    valid = valid.set_index("record_key").sort_index()
    # Exact time lookup prevents crossing the two invalid-hour dates.
    lagged = valid[frozen["candidate_groups"]["both"]].reindex(X.index - pd.Timedelta(hours=1))
    lagged.index = X.index
    lagged.columns = [f"{name}_lag1" for name in lagged.columns]
    present = lagged.notna().all(axis=1)
    print(json.dumps({"missing_by_feature": lagged.isna().sum().to_dict(),
                      "affected_timestamps": [str(x) for x in lagged.index[~present][:12]]}, ensure_ascii=False), flush=True)
    # Original weather gaps remain missing; HGB learns a missing branch from training.
    assert not np.isinf(lagged.to_numpy()).any()
    assert lagged["생산량_lag1"].notna().all()
    for source in frozen["candidate_groups"]["both"]:
        direct = valid[source].reindex(X.index - pd.Timedelta(hours=1)).to_numpy()
        np.testing.assert_array_equal(lagged[f"{source}_lag1"], direct)
    augmented = pd.concat([X, lagged], axis=1)
    prior = pd.read_csv(BASE_PRED, encoding="utf-8-sig", parse_dates=["record_key"])
    prior = prior.loc[prior["method"].eq("hgb_calendar")].copy()
    assert len(prior) == 2928 and not prior.duplicated(["target", "record_key"]).any()
    parts = []
    coverage = []
    with threadpool_limits(limits=2):
        for stage, cutoff, end in (
            ("May", pd.Timestamp("2021-05-01"), pd.Timestamp("2021-06-01")),
            ("Jun", pd.Timestamp("2021-06-01"), pd.Timestamp("2021-07-01")),
        ):
            train = X.index < cutoff
            test = (X.index >= cutoff) & (X.index < end)
            assert X.index[train].max() < X.index[test].min()
            coverage.append({"stage": stage, "training_n": int(train.sum()), "evaluation_n": int(test.sum()),
                             "training_missing_weather_hours": int((~present[train]).sum()),
                             "evaluation_missing_weather_hours": int((~present[test]).sum())})
            for target in base.TARGETS:
                old = prior.loc[prior["stage"].eq(stage) & prior["target"].eq(target)].sort_values("record_key").reset_index(drop=True)
                assert old["record_key"].tolist() == list(X.index[test])
                np.testing.assert_array_equal(old["actual"], labels.loc[test, target])
                parts.append(old)
                for group in ("production", "weather", "both"):
                    extra = [f"{name}_lag1" for name in frozen["candidate_groups"][group]]
                    columns = full + extra
                    model = HistGradientBoostingRegressor(**frozen["model_settings"])
                    model.fit(augmented.loc[train, columns], labels.loc[train, target])
                    predicted = np.maximum(model.predict(augmented.loc[test, columns]), 0)
                    assert np.isfinite(predicted).all()
                    new = old.copy()
                    new["method"] = f"hgb_{group}_lag1"
                    new["prediction"] = predicted
                    parts.append(new)
    compared = pd.concat(parts, ignore_index=True)
    counts = compared.groupby(["stage", "target", "method"]).size()
    assert counts.loc["May"].eq(744).all() and counts.loc["Jun"].eq(720).all()
    rows = []
    for period, section in [("pooled", compared), *[(s, compared.loc[compared["stage"].eq(s)]) for s in ("May", "Jun")]]:
        for (target, method, state), group in section.groupby(["target", "method", "peak_state"], sort=True):
            rows.append({"period": period, "target": target, "method": method, "peak_state": state, **score(group)})
        for (target, method), group in section.groupby(["target", "method"], sort=True):
            rows.append({"period": period, "target": target, "method": method, "peak_state": "all", **score(group)})
    scores = pd.DataFrame(rows)
    assert scores.loc[scores["period"].eq("pooled") & scores["peak_state"].eq("all"), "n"].eq(1464).all()
    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)
    compared.to_csv(OUT_PRED, index=False, encoding="utf-8-sig")
    scores.to_csv(OUT_SCORE, index=False, encoding="utf-8-sig")
    pd.DataFrame(coverage).to_csv(OUT_COVERAGE, index=False, encoding="utf-8-sig")
    assert sha256(FROZEN.read_bytes()).hexdigest() == sha256(frozen_bytes).hexdigest()
    assert sha256(base.SOURCE.read_bytes()).hexdigest() == digest
    print(json.dumps({"runtime": platform.python_version(), "sklearn": sklearn.__version__, "source_sha256": digest,
                      "frozen_sha256": sha256(frozen_bytes).hexdigest(), "coverage": coverage,
                      "overall": scores.loc[scores["peak_state"].eq("all"), ["period", "target", "method", "n", "mae"]].to_dict(orient="records")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
