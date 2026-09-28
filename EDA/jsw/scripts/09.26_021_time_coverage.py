"""Audit recorded-time coverage for subsequent analysis; no model or prediction."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import platform

import pandas as pd


EDA_DIR = Path(__file__).parent.parent
SOURCE = EDA_DIR / "../../data/origin/okm_augumented_2021.csv"
TABLE_DIR = EDA_DIR / "tables"
PROCESSED_DIR = EDA_DIR / "../../data/processed/jsw"
PREFIX = "09.26_021"
EXPECTED_SHA256 = "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"
HISTORY_HOURS = (1, 24, 168)
POWER_COLUMNS = ["15분", "30분", "45분", "60분"]


def save_table(frame: pd.DataFrame, suffix: str) -> None:
    frame.to_csv(TABLE_DIR / f"{PREFIX}_{suffix}.csv", index=False, encoding="utf-8-sig")


def coverage_summary(frame: pd.DataFrame, history: int) -> dict:
    complete = frame[f"complete_past_{history}h"]
    touches = frame.loc[complete, f"window_touches_zero_{history}h"]
    return {
        "history_hours": history,
        "normal_target_rows": len(frame),
        "exact_lag_available": int(frame[f"exact_lag_{history}h"].sum()),
        "complete_past_available": int(complete.sum()),
        "incomplete_past": int((~complete).sum()),
        "complete_share": float(complete.mean()),
        "target_and_history_touches_all_power_zero": int(touches.sum()),
        "complete_without_zero_window": int((~touches).sum()),
    }


def main() -> None:
    source_bytes = SOURCE.read_bytes()
    digest = sha256(source_bytes).hexdigest()
    assert digest == EXPECTED_SHA256, "Input changed; revise scope before reusing prior evidence"
    raw = pd.read_csv(
        SOURCE, encoding="utf-8-sig",
        usecols=["날짜", "시간", *POWER_COLUMNS, "평균", "생산량"],
    )
    assert len(raw) == 6168
    raw["source_data_row"] = range(1, len(raw) + 1)  # Header excluded, one-based.
    scope = raw.loc[raw["날짜"].lt(20210901)].copy()
    valid = scope.loc[scope["시간"].between(0, 23)].copy()
    assert (len(scope), len(valid)) == (5832, 5784)
    assert not valid.duplicated(["날짜", "시간"]).any()
    assert valid.groupby("날짜")["시간"].nunique().eq(24).all()
    assert not valid[[*POWER_COLUMNS, "평균", "생산량"]].isna().any().any()
    valid["record_key"] = (
        pd.to_datetime(valid["날짜"].astype(str), format="%Y%m%d")
        + pd.to_timedelta(valid["시간"], unit="h")
    )
    valid = valid.sort_values("record_key").reset_index(drop=True)
    assert valid["record_key"].is_unique
    step = pd.Timedelta(hours=1)
    valid["segment_id"] = valid["record_key"].diff().ne(step).cumsum()
    valid["available_contiguous_past_hours"] = valid.groupby("segment_id").cumcount()
    valid["target_all_power_zero"] = valid[POWER_COLUMNS].eq(0).all(axis=1)
    assert int(valid["target_all_power_zero"].sum()) == 17
    assert valid.loc[valid["target_all_power_zero"], "생산량"].eq(0).all()

    segments = valid.groupby("segment_id").agg(
        first_record_key=("record_key", "min"),
        last_record_key=("record_key", "max"),
        normal_rows=("record_key", "size"),
        all_power_zero_rows=("target_all_power_zero", "sum"),
    ).reset_index()
    assert int(segments["normal_rows"].sum()) == len(valid)
    gaps = []
    for left, right in zip(segments.itertuples(index=False), segments.iloc[1:].itertuples(index=False)):
        missing = pd.date_range(left.last_record_key + step, right.first_record_key - step, freq="h")
        gaps.append({
            "last_key_before_gap": left.last_record_key,
            "first_key_after_gap": right.first_record_key,
            "unusable_from": missing.min(), "unusable_to": missing.max(),
            "unusable_record_keys": len(missing),
            "reason": "invalid source hour; not repaired or interpolated",
        })
    gaps_frame = pd.DataFrame(gaps)
    assert int(gaps_frame["unusable_record_keys"].sum()) == 48

    manifest = valid[[
        "source_data_row", "record_key", "segment_id", "available_contiguous_past_hours",
        "target_all_power_zero",
    ]].copy()
    manifest["month"] = manifest["record_key"].dt.month
    manifest["period_candidate"] = manifest["month"].map(
        lambda m: "Jan-Jun_development_candidate" if m <= 6 else "Jul-Aug_review_candidate"
    )
    existing_keys = pd.Index(valid["record_key"])
    zero = valid["target_all_power_zero"].astype(int)
    for history in HISTORY_HOURS:
        manifest[f"exact_lag_{history}h"] = (
            manifest["record_key"] - pd.Timedelta(hours=history)
        ).isin(existing_keys)
        complete = manifest["available_contiguous_past_hours"].ge(history)
        manifest[f"complete_past_{history}h"] = complete
        # Include target and immediately preceding h records only within a segment.
        touch_count = zero.groupby(valid["segment_id"]).transform(
            lambda s: s.rolling(history + 1, min_periods=history + 1).sum()
        )
        touches = pd.Series(pd.NA, index=manifest.index, dtype="boolean")
        touches.loc[complete] = touch_count.loc[complete].gt(0)
        manifest[f"window_touches_zero_{history}h"] = touches
        assert manifest.loc[complete, f"exact_lag_{history}h"].all()
        # Independent key-set checks prevent a positional shift crossing unusable dates.
        for t in manifest.loc[complete, "record_key"]:
            assert all(t - k * step in existing_keys for k in range(1, history + 1))
        expected = int((segments["normal_rows"] - history).clip(lower=0).sum())
        assert int(complete.sum()) == expected
    assert manifest["complete_past_168h"].le(manifest["complete_past_24h"]).all()
    assert manifest["complete_past_24h"].le(manifest["complete_past_1h"]).all()

    overview = pd.DataFrame([coverage_summary(manifest, h) for h in HISTORY_HOURS])
    months = pd.DataFrame([
        {"month": int(month), **coverage_summary(frame, h)}
        for month, frame in manifest.groupby("month") for h in HISTORY_HOURS
    ])
    for h in HISTORY_HOURS:
        by_month = months.loc[months["history_hours"].eq(h)]
        total = overview.loc[overview["history_hours"].eq(h)].iloc[0]
        for col in (
            "normal_target_rows", "exact_lag_available", "complete_past_available",
            "incomplete_past", "target_and_history_touches_all_power_zero", "complete_without_zero_window",
        ):
            assert int(by_month[col].sum()) == int(total[col])
    # Reuse the prior normal-day scope as an independent reconciliation.
    prior = pd.read_csv(TABLE_DIR / "09.23_011_daily_profile_summary.csv", encoding="utf-8-sig")
    assert set(prior["date"]) == set(valid["날짜"])
    assert len(prior) == 241 and prior["profile_id"].nunique() == 126
    assert sha256(SOURCE.read_bytes()).hexdigest() == digest

    TABLE_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_table(segments, "segments")
    save_table(gaps_frame, "unusable_time_gaps")
    save_table(overview, "coverage_overview")
    save_table(months, "coverage_by_month")
    manifest_path = PROCESSED_DIR / f"{PREFIX}_coverage_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    loaded = pd.read_csv(manifest_path, encoding="utf-8-sig")
    assert loaded.shape == manifest.shape and loaded["source_data_row"].is_unique
    facts = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(), "pandas": pd.__version__,
        "source_sha256": digest,
        "scope_rows": len(scope), "valid_rows": len(valid), "valid_days": len(prior),
        "segments": len(segments), "unusable_hours": len(scope) - len(valid),
        "all_power_zero_rows": int(valid["target_all_power_zero"].sum()),
        "coverage": overview.to_dict(orient="records"),
        "manifest_shape": list(manifest.shape),
        "manifest_relative_path": "../../data/processed/jsw/" + manifest_path.name,
        "scope_note": "Recorded key offsets only; physical time anchor and collection latency unconfirmed",
        "model_fitted": False, "prediction_generated": False,
    }
    (TABLE_DIR / f"{PREFIX}_facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
