"""Count exploratory high-hour misses and false selections from saved predictions."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


HERE = Path(__file__).parent.parent
PREFIX = "09.26_004"


def summarize(frame: pd.DataFrame, period: str, method: str) -> dict:
    actual_high = frame["actual"].ge(frame["high_threshold_from_training"])
    predicted_high = frame["prediction"].ge(frame["high_threshold_from_training"])
    hit = int((actual_high & predicted_high).sum())
    miss = int((actual_high & ~predicted_high).sum())
    false = int((~actual_high & predicted_high).sum())
    return {"period": period, "method": method, "n": len(frame),
            "high_actual_n": int(actual_high.sum()), "not_high_actual_n": int((~actual_high).sum()),
            "selected_n": int(predicted_high.sum()), "hit_n": hit, "miss_n": miss,
            "false_selection_n": false,
            "recall": hit / (hit + miss) if hit + miss else np.nan,
            "precision": hit / (hit + false) if hit + false else np.nan}


def main() -> None:
    q = pd.read_csv(HERE / "predictions" / "09.26_002_development_predictions.csv", encoding="utf-8-sig")
    p = pd.read_csv(HERE / "predictions" / "09.26_003_development_predictions.csv", encoding="utf-8-sig")
    p = p.loc[p["target"].eq("peak") & ~p["method"].eq("hgb_calendar")]
    combined = pd.concat([q, p], ignore_index=True)
    assert combined.groupby(["stage", "method"]).size().loc["May"].eq(744).all()
    assert combined.groupby(["stage", "method"]).size().loc["Jun"].eq(720).all()
    reference = combined.loc[combined["method"].eq("hgb_calendar")].sort_values(["stage", "record_key"])
    for method, group in combined.groupby("method"):
        check = group.sort_values(["stage", "record_key"])
        assert check["record_key"].tolist() == reference["record_key"].tolist()
        np.testing.assert_array_equal(check["actual"], reference["actual"])
        np.testing.assert_array_equal(check["high_threshold_from_training"], reference["high_threshold_from_training"])
    rows = []
    states = []
    for period, section in [("pooled", combined), *[(s, combined.loc[combined["stage"].eq(s)]) for s in ("May", "Jun")]]:
        for method, group in section.groupby("method"):
            rows.append(summarize(group, period, method))
            for state in ("high_onset", "high_continued"):
                subset = group.loc[group["peak_state"].eq(state)]
                threshold = subset["high_threshold_from_training"]
                picked = subset["prediction"].ge(threshold)
                assert subset["actual"].ge(threshold).all()
                states.append({"period": period, "method": method, "peak_state": state,
                               "actual_n": len(subset), "hit_n": int(picked.sum()), "miss_n": int((~picked).sum())})
    scores = pd.DataFrame(rows)
    state_scores = pd.DataFrame(states)
    assert scores.loc[scores["period"].eq("pooled"), "high_actual_n"].eq(79).all()
    assert scores.loc[scores["period"].eq("pooled"), "not_high_actual_n"].eq(1385).all()
    assert state_scores.loc[state_scores["period"].eq("pooled")].groupby("method")["actual_n"].sum().eq(79).all()
    scores.to_csv(HERE / "tables" / f"{PREFIX}_selection_scores.csv", index=False, encoding="utf-8-sig")
    state_scores.to_csv(HERE / "tables" / f"{PREFIX}_by_high_state.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"pooled": scores.loc[scores["period"].eq("pooled")].to_dict(orient="records"),
                      "states": state_scores.loc[state_scores["period"].eq("pooled")].to_dict(orient="records")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
