"""Compare one prespecified upper-quantile peak model to 001 on development hours."""

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
PREFIX = "09.26_002"
FROZEN = HERE / "tables" / f"{PREFIX}_frozen.json"
BASE_SCRIPT = HERE / "scripts" / "09.26_001_power_models.py"
BASE_PRED = HERE / "predictions" / "09.26_001_development_predictions.csv"
OUT_PRED = HERE / "predictions" / f"{PREFIX}_development_predictions.csv"
OUT_TABLE = HERE / "tables" / f"{PREFIX}_scores.csv"


def score(group: pd.DataFrame) -> dict:
    error = group["prediction"].to_numpy() - group["actual"].to_numpy()
    under = error < 0
    over = error > 0
    return {
        "n": len(error), "mae": float(np.abs(error).mean()),
        "bias": float(error.mean()),
        "under_n": int(under.sum()), "under_rate": float(under.mean()),
        "shortfall_per_hour": float(np.maximum(-error, 0).mean()),
        "shortfall_when_under": float((-error[under]).mean()) if under.any() else 0.0,
        "over_n": int(over.sum()), "over_rate": float(over.mean()),
        "over_per_hour": float(np.maximum(error, 0).mean()),
        "over_when_over": float(error[over].mean()) if over.any() else 0.0,
    }


def main() -> None:
    frozen_bytes = FROZEN.read_bytes()
    frozen = json.loads(frozen_bytes)
    assert frozen["quantile"] == frozen["candidate_settings"]["quantile"] == 0.75
    spec = importlib.util.spec_from_file_location("power_001", BASE_SCRIPT)
    base = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(base)
    assert frozen["candidate_settings"] == {**base.SETTINGS["hgb"], "loss": "quantile", "quantile": 0.75}
    X, labels, _, full, digest = base.read_data(save_catalog=False)
    prior = pd.read_csv(BASE_PRED, encoding="utf-8-sig", parse_dates=["record_key"])
    prior = prior.loc[prior["target"].eq("peak") & prior["method"].eq("hgb_calendar")].copy()
    assert len(prior) == 1464 and prior["record_key"].is_unique
    parts = []
    with threadpool_limits(limits=2):
        for stage, cutoff, end in (
            ("May", pd.Timestamp("2021-05-01"), pd.Timestamp("2021-06-01")),
            ("Jun", pd.Timestamp("2021-06-01"), pd.Timestamp("2021-07-01")),
        ):
            train = X.index < cutoff
            test = (X.index >= cutoff) & (X.index < end)
            assert X.index[train].max() < X.index[test].min()
            model = HistGradientBoostingRegressor(**frozen["candidate_settings"])
            model.fit(X.loc[train, full], labels.loc[train, "peak"])
            predicted = np.maximum(model.predict(X.loc[test, full]), 0)
            assert np.isfinite(predicted).all()
            old = prior.loc[prior["stage"].eq(stage)].sort_values("record_key").reset_index(drop=True)
            assert old["record_key"].tolist() == list(X.index[test])
            np.testing.assert_array_equal(old["actual"], labels.loc[test, "peak"])
            assert old["high_threshold_from_training"].nunique() == 1
            assert old["high_threshold_from_training"].iloc[0] == base.threshold_before(labels, cutoff)
            new = old.copy()
            new["method"] = "hgb_q75_calendar"
            new["prediction"] = predicted
            parts.extend([old, new])
    compared = pd.concat(parts, ignore_index=True)
    assert compared.groupby(["stage", "method"]).size().to_dict() == {
        ("Jun", "hgb_calendar"): 720, ("Jun", "hgb_q75_calendar"): 720,
        ("May", "hgb_calendar"): 744, ("May", "hgb_q75_calendar"): 744,
    }
    rows = []
    for period, section in [("pooled", compared), *[(s, compared.loc[compared["stage"].eq(s)]) for s in ("May", "Jun")]]:
        for (method, state), group in section.groupby(["method", "peak_state"], sort=True):
            rows.append({"period": period, "method": method, "peak_state": state, **score(group)})
        for method, group in section.groupby("method", sort=True):
            rows.append({"period": period, "method": method, "peak_state": "all", **score(group)})
    scores = pd.DataFrame(rows)
    assert scores.loc[scores["period"].eq("pooled") & scores["peak_state"].eq("all"), "n"].eq(1464).all()
    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)
    compared.to_csv(OUT_PRED, index=False, encoding="utf-8-sig")
    scores.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")
    assert sha256(FROZEN.read_bytes()).hexdigest() == sha256(frozen_bytes).hexdigest()
    assert sha256(base.SOURCE.read_bytes()).hexdigest() == digest
    print(json.dumps({"runtime": platform.python_version(), "sklearn": sklearn.__version__, "source_sha256": digest,
                      "frozen_sha256": sha256(frozen_bytes).hexdigest(),
                      "pooled": scores.loc[scores["period"].eq("pooled")].to_dict(orient="records")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
