"""Provisional sequential next-record power modeling with chronological selection."""

from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ANALYSIS_DIR = Path(__file__).parent.parent
SOURCE = ANALYSIS_DIR / "../../data/origin/okm_augumented_2021.csv"
TABLE_DIR = ANALYSIS_DIR / "tables"
PREDICTION_DIR = ANALYSIS_DIR / "predictions"
EDA_PROCESSED_DIR = ANALYSIS_DIR / "../../data/processed/jsw"
PREFIX = "09.26_001"
EXPECTED_SHA256 = "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"
TARGETS = ("mean", "peak")
METHODS = (
    "lag1", "lag24", "lag168", "ridge_power", "ridge_calendar",
    "hgb_power", "hgb_calendar",
)
SETTINGS = {
    "ridge": {"alpha": 1.0},
    "hgb": {
        "loss": "squared_error", "learning_rate": 0.05, "max_iter": 200,
        "max_leaf_nodes": 15, "min_samples_leaf": 30, "l2_regularization": 1.0,
        "early_stopping": False, "random_state": 20260926,
    },
    "clip_predictions_below_zero": True,
    "selection": "pooled May-Jun MAE on common timestamps, by target",
    "past_observation_assumption": "All preceding record's values are available before target record",
}


def save(frame: pd.DataFrame, name: str, row_level: bool = False) -> None:
    folder = PREDICTION_DIR if row_level else TABLE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / f"{PREFIX}_{name}.csv", index=False, encoding="utf-8-sig")


def read_data(save_catalog: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], str]:
    digest = sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256
    slots = ["15분", "30분", "45분", "60분"]
    raw = pd.read_csv(SOURCE, encoding="utf-8-sig", usecols=["날짜", "시간", *slots, "평균", "생산량"])
    assert len(raw) == 6168
    raw["source_data_row"] = np.arange(1, len(raw) + 1)
    scope = raw.loc[raw["날짜"].lt(20210901)].copy()
    valid = scope.loc[scope["시간"].between(0, 23)].copy()
    assert len(scope) == 5832 and len(valid) == 5784
    valid["record_key"] = pd.to_datetime(valid["날짜"].astype(str), format="%Y%m%d") + pd.to_timedelta(valid["시간"], unit="h")
    valid = valid.set_index("record_key").sort_index()
    assert valid.index.is_unique and not valid[[*slots, "평균", "생산량"]].isna().any().any()
    series = pd.DataFrame({
        "mean": valid["평균"], "peak": valid[slots].max(axis=1),
        "last_slot": valid["60분"], "range": valid[slots].max(axis=1) - valid[slots].min(axis=1),
    })
    grid = series.reindex(pd.date_range("2021-01-01", "2021-08-31 23:00", freq="h"))
    features = pd.DataFrame(index=grid.index)
    descriptions = []
    for variable in TARGETS:
        for lag in (1, 2, 3, 24, 168):
            name = f"{variable}_lag{lag}"
            features[name] = grid[variable].shift(lag)
            descriptions.append({"feature": name, "group": "past_power", "source": variable, "oldest_offset_hours": -lag, "latest_offset_hours": -lag})
    for variable in ("last_slot", "range"):
        name = f"{variable}_lag1"
        features[name] = grid[variable].shift(1)
        descriptions.append({"feature": name, "group": "past_power", "source": variable, "oldest_offset_hours": -1, "latest_offset_hours": -1})
    for variable, functions in (("mean", ("mean", "std", "min", "max")), ("peak", ("mean", "max"))):
        past = grid[variable].shift(1).rolling(24, min_periods=24)
        for function in functions:
            name = f"{variable}_past24_{function}"
            features[name] = past.std(ddof=0) if function == "std" else getattr(past, function)()
            descriptions.append({"feature": name, "group": "past_power", "source": variable, "oldest_offset_hours": -24, "latest_offset_hours": -1})
    power_columns = list(features.columns)
    calendar = {
        "hour": features.index.hour, "weekday": features.index.dayofweek,
        "month": features.index.month,
        "hour_sin": np.sin(2 * np.pi * features.index.hour / 24),
        "hour_cos": np.cos(2 * np.pi * features.index.hour / 24),
        "weekday_sin": np.sin(2 * np.pi * features.index.dayofweek / 7),
        "weekday_cos": np.cos(2 * np.pi * features.index.dayofweek / 7),
        "weekend": (features.index.dayofweek >= 5).astype(int),
    }
    for name, values in calendar.items():
        features[name] = values
        descriptions.append({"feature": name, "group": "known_calendar", "source": "record_key", "oldest_offset_hours": 0, "latest_offset_hours": 0})
    features = features.reindex(valid.index)
    eligible = features.notna().all(axis=1)
    X = features.loc[eligible].copy()
    labels = pd.DataFrame({
        "source_data_row": valid.loc[eligible, "source_data_row"],
        "mean": series.loc[eligible, "mean"], "peak": series.loc[eligible, "peak"],
        "peak_lag1": features.loc[eligible, "peak_lag1"],
        "production_posthoc": valid.loc[eligible, "생산량"],
    })
    manifest = pd.read_csv(EDA_PROCESSED_DIR / "09.26_021_coverage_manifest.csv", encoding="utf-8-sig")
    manifest["record_key"] = pd.to_datetime(manifest["record_key"])
    manifest = manifest.set_index("record_key").reindex(X.index)
    labels["zero_window_posthoc"] = manifest["window_touches_zero_24h"].astype(bool)
    assert manifest["complete_past_24h"].all() and manifest["exact_lag_168h"].all()
    assert labels.index.equals(X.index) and np.isfinite(X.to_numpy()).all()
    desc = pd.DataFrame(descriptions)
    assert desc.loc[desc["group"].eq("past_power"), "latest_offset_hours"].lt(0).all()
    if save_catalog:
        save(desc, "feature_catalog")
    # Direct source-key reconstructions independently verify every lag used.
    for variable in TARGETS:
        for lag in (1, 2, 3, 24, 168):
            direct = series[variable].reindex(X.index - pd.Timedelta(hours=lag)).to_numpy()
            np.testing.assert_array_equal(X[f"{variable}_lag{lag}"], direct)
    return X, labels, power_columns, list(X.columns), digest


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = np.asarray(predicted) - np.asarray(actual)
    return {
        "n": len(error), "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(error.mean()),
        "under_rate": float((error < 0).mean()), "mean_shortfall": float(np.maximum(-error, 0).mean()),
    }


def predict_all(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, power: list[str], full: list[str]) -> dict:
    result = {}
    for target in TARGETS:
        for method in METHODS:
            if method.startswith("lag"):
                prediction = X_test[f"{target}_{method}"].to_numpy()
            else:
                columns = power if method.endswith("_power") else full
                if method.startswith("ridge"):
                    model = make_pipeline(StandardScaler(), Ridge(**SETTINGS["ridge"]))
                else:
                    model = HistGradientBoostingRegressor(**SETTINGS["hgb"])
                model.fit(X_train[columns], y_train[target])
                prediction = model.predict(X_test[columns])
            result[(target, method)] = np.maximum(prediction, 0)
            assert np.isfinite(result[(target, method)]).all()
    return result


def make_predictions(X: pd.DataFrame, y: pd.DataFrame, predictions: dict, stage: str, threshold: float) -> pd.DataFrame:
    rows = []
    for (target, method), prediction in predictions.items():
        rows.append(pd.DataFrame({
            "record_key": X.index, "source_data_row": y["source_data_row"].to_numpy(),
            "stage": stage, "target": target, "method": method,
            "actual": y[target].to_numpy(), "prediction": prediction,
            "peak_state": np.where(y["peak"].ge(threshold), np.where(y["peak_lag1"].lt(threshold), "high_onset", "high_continued"), "not_high"),
            "production_posthoc": y["production_posthoc"].to_numpy(),
            "zero_window_posthoc": y["zero_window_posthoc"].to_numpy(),
            "high_threshold_from_training": threshold,
        }))
    return pd.concat(rows, ignore_index=True)


def summarize(predictions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([
        {**dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,))), **metrics(frame["actual"].to_numpy(), frame["prediction"].to_numpy())}
        for keys, frame in predictions.groupby(group_columns, sort=True)
    ])


def threshold_before(labels: pd.DataFrame, cutoff: pd.Timestamp) -> float:
    # The cutoff is always before the scored period; all eligible training labels only.
    return float(labels.loc[labels.index < cutoff, "peak"].quantile(0.95))


def development(X: pd.DataFrame, y: pd.DataFrame, power: list[str], full: list[str], digest: str) -> None:
    parts = []
    coverage = []
    for name, cutoff, end in (
        ("May", pd.Timestamp("2021-05-01"), pd.Timestamp("2021-06-01")),
        ("Jun", pd.Timestamp("2021-06-01"), pd.Timestamp("2021-07-01")),
    ):
        train = X.index < cutoff
        test = (X.index >= cutoff) & (X.index < end)
        assert X.index[train].max() < X.index[test].min()
        prediction = predict_all(X.loc[train], y.loc[train], X.loc[test], power, full)
        parts.append(make_predictions(X.loc[test], y.loc[test], prediction, name, threshold_before(y, cutoff)))
        coverage.append({"fold": name, "training_rows": int(train.sum()), "evaluation_rows": int(test.sum()), "training_end": X.index[train].max(), "evaluation_start": X.index[test].min(), "evaluation_end": X.index[test].max()})
        print(json.dumps({"completed_fold": name, "train": int(train.sum()), "scored": int(test.sum())}), flush=True)
    predictions = pd.concat(parts, ignore_index=True)
    pooled = summarize(predictions, ["target", "method"])
    monthly = summarize(predictions, ["stage", "target", "method"])
    assert pooled["n"].eq(1464).all()
    winners = {}
    for target in TARGETS:
        chosen = pooled.loc[pooled["target"].eq(target)].sort_values(["mae", "method"]).iloc[0]
        winners[target] = str(chosen["method"])
    freeze = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_sha256": digest, "settings": SETTINGS, "power_features": power,
        "all_features": full, "selected_by_development": winners,
        "development_rows_by_fold": coverage, "development_scores": pooled.to_dict(orient="records"),
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
    }
    save(predictions, "development_predictions", row_level=True)
    save(pooled, "development_scores")
    save(monthly, "development_by_month")
    save(pd.DataFrame(coverage), "development_coverage")
    (TABLE_DIR / f"{PREFIX}_frozen.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    assert sha256(SOURCE.read_bytes()).hexdigest() == digest
    print(json.dumps({"selected_before_Jul_Aug": winners, "development_scores": freeze["development_scores"]}, indent=2), flush=True)


def evaluation(X: pd.DataFrame, y: pd.DataFrame, power: list[str], full: list[str], digest: str) -> None:
    frozen_bytes = (TABLE_DIR / f"{PREFIX}_frozen.json").read_bytes()
    frozen = json.loads(frozen_bytes)
    assert frozen["settings"] == SETTINGS and frozen["source_sha256"] == digest
    assert frozen["power_features"] == power and frozen["all_features"] == full
    cutoff = pd.Timestamp("2021-07-01")
    train = X.index < cutoff
    test = X.index >= cutoff
    assert int(train.sum()) == 4176 and int(test.sum()) == 1344
    assert X.index[train].max() < X.index[test].min()
    prediction = predict_all(X.loc[train], y.loc[train], X.loc[test], power, full)
    predictions = make_predictions(X.loc[test], y.loc[test], prediction, "Jul-Aug", threshold_before(y, cutoff))
    # Compare predictions to the independently saved 003 timestamps and baseline values.
    prior = pd.read_csv(EDA_PROCESSED_DIR / "09.23_003_validation_predictions.csv", encoding="utf-8-sig")
    prior["timestamp"] = pd.to_datetime(prior["timestamp"])
    prior = prior.set_index("timestamp").sort_index()
    assert prior.index.equals(X.index[test])
    for target in TARGETS:
        np.testing.assert_array_equal(prior[target], y.loc[test, target])
        for lag in (1, 24, 168):
            np.testing.assert_array_equal(prior[f"{target}_lag{lag}"], prediction[(target, f"lag{lag}")])
    scores = summarize(predictions, ["target", "method"])
    predictions["month"] = predictions["record_key"].dt.month
    predictions["hour"] = predictions["record_key"].dt.hour
    monthly = summarize(predictions, ["month", "target", "method"])
    state = summarize(predictions, ["peak_state", "target", "method"])
    hours = summarize(predictions, ["hour", "target", "method"])
    zero = summarize(predictions, ["zero_window_posthoc", "target", "method"])
    comparisons = []
    for target in TARGETS:
        selected = frozen["selected_by_development"][target]
        baseline = scores.loc[(scores["target"].eq(target)) & (scores["method"].eq("lag1"))].iloc[0]
        chosen = scores.loc[(scores["target"].eq(target)) & (scores["method"].eq(selected))].iloc[0]
        comparisons.append({"target": target, "selected_before_review": selected, "review_n": int(chosen["n"]), "lag1_mae": float(baseline["mae"]), "selected_mae": float(chosen["mae"]), "mae_reduction_percent": float(100 * (1 - chosen["mae"] / baseline["mae"]))})
    save(predictions, "review_predictions", row_level=True)
    save(scores, "review_scores")
    save(monthly, "review_by_month")
    save(state, "review_by_peak_state")
    save(hours, "review_by_hour")
    save(zero, "review_zero_windows")
    save(pd.DataFrame(comparisons), "selected_comparison")
    # Every model is evaluated on the same source rows; conditional partitions reconcile.
    assert predictions.groupby(["target", "method"])["source_data_row"].nunique().eq(1344).all()
    assert monthly.groupby(["target", "method"])["n"].sum().eq(1344).all()
    assert state.groupby(["target", "method"])["n"].sum().eq(1344).all()
    assert zero.groupby(["target", "method"])["n"].sum().eq(1344).all()
    assert not predictions["record_key"].dt.month.eq(9).any()
    assert sha256(SOURCE.read_bytes()).hexdigest() == digest
    assert (TABLE_DIR / f"{PREFIX}_frozen.json").read_bytes() == frozen_bytes
    facts = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_sha256": digest, "frozen_configuration_sha256": sha256(frozen_bytes).hexdigest(),
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
        "common_training_rows": int(train.sum()), "common_review_rows": int(test.sum()),
        "power_features": len(power), "calendar_features": len(full) - len(power),
        "high_threshold": threshold_before(y, cutoff), "selected_comparison": comparisons,
        "review_scores": scores.to_dict(orient="records"),
        "development_prediction_shape": list(pd.read_csv(PREDICTION_DIR / f"{PREFIX}_development_predictions.csv", encoding="utf-8-sig").shape),
        "review_prediction_shape": list(predictions.shape),
        "leakage_controls": "Power offsets <= -1; scalers fit inside chronological training only; no production/weather/target power features",
    }
    (TABLE_DIR / f"{PREFIX}_facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(facts, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "evaluation"), required=True)
    args = parser.parse_args()
    print(json.dumps({"runtime_started": platform.python_version(), "sklearn": sklearn.__version__, "stage": args.stage}), flush=True)
    X, _labels, power, full, digest = read_data()
    with threadpool_limits(limits=2):
        if args.stage == "development":
            development(X, _labels, power, full, digest)
        else:
            evaluation(X, _labels, power, full, digest)
