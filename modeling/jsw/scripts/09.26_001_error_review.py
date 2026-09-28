"""Diagnose frozen model predictions without refitting or changing candidates."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).parent.parent
TABLE_DIR = ANALYSIS_DIR / "tables"
PREDICTION_DIR = ANALYSIS_DIR / "predictions"
EDA_PROCESSED_DIR = ANALYSIS_DIR / "../../data/processed/jsw"
PREFIX = "09.26_001"


def summarize(frame, keys):
    rows = []
    for group_keys, group in frame.groupby(keys):
        errors = group["prediction"] - group["actual"]
        rows.append({
            **dict(zip(keys, group_keys if isinstance(group_keys, tuple) else (group_keys,))),
            "n": len(group), "mae": float(errors.abs().mean()),
            "bias": float(errors.mean()), "under_rate": float(errors.lt(0).mean()),
            "actual_mean": float(group["actual"].mean()), "prediction_mean": float(group["prediction"].mean()),
        })
    return pd.DataFrame(rows)


def main():
    frozen_path = TABLE_DIR / f"{PREFIX}_frozen.json"
    frozen_hash = sha256(frozen_path.read_bytes()).hexdigest()
    development = pd.read_csv(PREDICTION_DIR / f"{PREFIX}_development_predictions.csv", encoding="utf-8-sig")
    review = pd.read_csv(PREDICTION_DIR / f"{PREFIX}_review_predictions.csv", encoding="utf-8-sig")
    chosen = ["lag1", "hgb_calendar"]
    dev_states = summarize(development.loc[development["method"].isin(chosen)], ["stage", "target", "method", "peak_state"])
    dev_states.to_csv(TABLE_DIR / f"{PREFIX}_development_peak_states.csv", index=False, encoding="utf-8-sig")
    # Zero-target and whole-window diagnostics are distinct; do not remove either.
    zero = review.loc[review["actual"].eq(0) & review["method"].isin(chosen)].copy()
    assert zero.groupby(["target", "method"]).size().eq(17).all()
    zero_scores = summarize(zero, ["target", "method"])
    zero_scores.to_csv(TABLE_DIR / f"{PREFIX}_zero_target_scores.csv", index=False, encoding="utf-8-sig")
    worst = review.loc[review["method"].eq("hgb_calendar")].copy()
    worst["absolute_error"] = (worst["prediction"] - worst["actual"]).abs()
    worst = worst.sort_values(["target", "absolute_error"], ascending=[True, False]).groupby("target", sort=False).head(10)
    worst.to_csv(PREDICTION_DIR / f"{PREFIX}_largest_errors.csv", index=False, encoding="utf-8-sig")
    source = ANALYSIS_DIR / "../../data/origin/okm_augumented_2021.csv"
    assert sha256(source.read_bytes()).hexdigest() == json.loads(frozen_path.read_bytes())["source_sha256"]
    raw = pd.read_csv(source, encoding="utf-8-sig", usecols=["날짜", "시간", "15분", "30분", "45분", "60분", "평균"])
    raw["source_data_row"] = np.arange(1, len(raw) + 1)
    manifest = pd.read_csv(EDA_PROCESSED_DIR / "09.26_021_coverage_manifest.csv", encoding="utf-8-sig")
    # Reproduce training eligibility from the earlier audit, independent of model feature code.
    eligible = manifest.loc[manifest["complete_past_24h"] & manifest["exact_lag_168h"], "source_data_row"]
    train = raw.loc[raw["source_data_row"].isin(eligible) & raw["날짜"].lt(20210701)].copy()
    assert len(train) == 4176
    train["peak"] = train[["15분", "30분", "45분", "60분"]].max(axis=1)
    training_context = pd.DataFrame([
        {"target": target, "n": len(train), "minimum": float(train[col].min()), "maximum": float(train[col].max()), "zero_rows": int(train[col].eq(0).sum())}
        for target, col in [("mean", "평균"), ("peak", "peak")]
    ])
    training_context.to_csv(TABLE_DIR / f"{PREFIX}_training_context.csv", index=False, encoding="utf-8-sig")
    assert sha256(frozen_path.read_bytes()).hexdigest() == frozen_hash
    print(json.dumps({
        "development_high_continued": dev_states.loc[dev_states["peak_state"].eq("high_continued")].to_dict(orient="records"),
        "zero_target_scores": zero_scores.to_dict(orient="records"),
        "training_context": training_context.to_dict(orient="records"),
        "worst_errors": worst[["record_key", "target", "actual", "prediction", "absolute_error", "peak_state", "zero_window_posthoc"]].groupby("target").head(3).to_dict(orient="records"),
        "frozen_configuration_unchanged": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
