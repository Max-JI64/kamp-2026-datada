"""Frozen January surge boundary, existing errors, alarm policies and April gate diagnosis."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from m01_prepare import ROOT, sha
from m04_rise_compare import now, read_csv

OUT = ROOT / "Modeling/tables/m04_surge"
CONTRACT = ROOT / "Modeling/config/m04_surge_contract.json"


def confusion(y, score, threshold):
    y, score = np.asarray(y, dtype=bool), np.asarray(score, dtype=float)
    alarm = score >= threshold
    tp, fp, fn, tn = [int(v.sum()) for v in [alarm & y, alarm & ~y, ~alarm & y, ~alarm & ~y]]
    return {"hours": len(y), "events": int(y.sum()), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "recall": tp / (tp + fn) if tp + fn else 0., "precision": tp / (tp + fp) if tp + fp else 0.,
        "false_positive_fraction": fp / (fp + tn) if fp + tn else 0., "threshold": float(threshold)}


def threshold_at_cap(y, score, cap):
    y, score = np.asarray(y, dtype=bool), np.asarray(score, dtype=float)
    choices = [np.nextafter(score.max(), np.inf)] + sorted(set(score), reverse=True)
    values = [confusion(y, score, threshold) for threshold in choices]
    eligible = [r for r in values if r["false_positive_fraction"] <= cap + 1e-12]
    return min(eligible, key=lambda r: (-r["TP"], r["FP"], -r["threshold"]))


def alarm_metrics(y, score, threshold):
    return {**confusion(y, score, threshold), "AP": float(average_precision_score(y, score)),
            "AUROC": float(roc_auc_score(y, score)), "prevalence": float(np.mean(y))}


def error_metrics(part):
    error = part.prediction.to_numpy() - part.actual.to_numpy()
    return {"hours": len(part), "dates": part.date.nunique(), "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())), "mean_under": float(np.maximum(-error, 0).mean()),
        "mean_over": float(np.maximum(error, 0).mean()), "under_hours": int((error < 0).sum()),
        "mean_actual": float(part.actual.mean()), "mean_increase": float(part.increase.mean())}


def main():
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    assert not (OUT / "run.json").exists(), "Preserve completed diagnosis."
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    first = ROOT / "Modeling/tables/m04_rise"
    parent = ROOT / "Modeling/tables/m04_rise_gate"
    for directory, expected in [(first, c["first_run_sha256"]), (parent, c["parent_run_sha256"])]:
        verified = json.loads((directory / "independent_verification.json").read_text(encoding="utf-8"))
        assert verified["status"] == "passed" and verified["run_sha256"] == sha(directory / "run.json") == expected
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        for name, digest in run["outputs_sha256"].items():
            assert sha(directory / name) == digest, name
    frame_path = ROOT / "Modeling/tables/m01/hourly_frame.csv"
    assert sha(frame_path) == c["frame_sha256"] and sha(ROOT / "data/origin/okm_augumented_2021.csv") == c["source_sha256"]
    f = read_csv(frame_path, ["timestamp", "date"])
    f = f.loc[f.eligible_common & f.timestamp.lt("2021-07-01")].copy()
    f["increase"] = f.target_maximum - f.lag1_maximum
    calibration = f.loc[f.timestamp.lt("2021-02-01")]
    positive = calibration.loc[calibration.increase.gt(0), "increase"]
    boundaries = {"q95": float(positive.quantile(.95, interpolation="linear")), "q90": float(positive.quantile(.90, interpolation="linear"))}
    assert boundaries["q95"] >= boundaries["q90"] > 0
    p = read_csv(parent / "predictions.csv", ["timestamp", "date"])
    p = p.merge(f[["timestamp", "lag1_maximum", "increase"]], on="timestamp", validate="many_to_one")
    p["surge"] = p.increase.ge(boundaries["q95"])
    p["predicted_increase"] = p.prediction - p.lag1_maximum
    counts, errors, alarms, policies = [], [], [], []
    for boundary_name, boundary in boundaries.items():
        for role, part in [("calibration_jan", calibration), ("train_jan_mar", f.loc[f.timestamp.lt("2021-04-01")]),
            ("dev_apr", f.loc[f.timestamp.dt.month.eq(4)]), ("dev_may", f.loc[f.timestamp.dt.month.eq(5)]),
            ("dev_jun", f.loc[f.timestamp.dt.month.eq(6)])]:
            event = part.increase.ge(boundary)
            old = (np.floor(part.lag1_mean + .5).between(20,26) & part.lag1_maximum.gt(0) & part.diag_target_rounded_mean.gt(26))
            counts.append({"boundary_name": boundary_name, "boundary": boundary, "role": role, "hours": len(part),
                "positive_increases": int(part.increase.gt(0).sum()), "surge_hours": int(event.sum()),
                "surge_dates": part.loc[event, "date"].nunique(), "old_rise_hours": int(old.sum()),
                "overlap_hours": int((old & event).sum()), "outside_prior_low_surges": int((event & ~np.floor(part.lag1_mean + .5).between(20,26)).sum()),
                "previous_zero_hours": int(part.lag1_maximum.eq(0).sum())})
        for name, g in p.groupby("group"):
            for split, part in [("pooled_development", g)] + list(g.groupby("split")):
                for condition, mask in [("all", np.ones(len(part), dtype=bool)), ("surge", part.increase.ge(boundary)),
                    ("non_surge", part.increase.lt(boundary)), ("legacy_rise", part.rise_event),
                    ("low_stay", part.power_transition.eq("low->low"))]:
                    selected = part.loc[mask]
                    if len(selected):
                        errors.append({"boundary_name": boundary_name, "boundary": boundary, "group": name,
                            "split": split, "condition": condition, **error_metrics(selected)})
                alarms.append({"boundary_name": boundary_name, "group": name, "split": split,
                    "score": "predicted_increase", "policy": "frozen_magnitude_boundary", "cap": np.nan,
                    **alarm_metrics(part.increase.ge(boundary), part.predicted_increase, boundary)})
    old_inner = read_csv(first / "inner_predictions.csv", ["timestamp", "date"])
    old_inner = old_inner.merge(f[["timestamp", "lag1_maximum", "increase"]], on="timestamp", validate="many_to_one")
    primary = boundaries["q95"]
    for score_name, group in [("B_predicted_increase", "B"), ("legacy_HGB_probability", "G_HGB")]:
        for split in ["dev_apr", "dev_may", "dev_jun"]:
            iv = old_inner.loc[old_inner.group.eq(group) & old_inner.split.eq(split)].copy()
            ev = p.loc[p.group.eq(group) & p.split.eq(split)].copy()
            inner_score = iv.prediction - iv.lag1_maximum if group == "B" else np.where(iv.prior_low, iv.probability, 0.)
            outer_score = ev.predicted_increase if group == "B" else np.where(ev.prior_low, ev.probability, 0.)
            assert iv.timestamp.max() < ev.timestamp.min()
            for cap in c["alarm"]["fixed_false_positive_caps"]:
                policy = threshold_at_cap(iv.increase.ge(primary), inner_score, cap)
                policies.append({"split": split, "score": score_name, "cap": cap, "inner_first": str(iv.timestamp.min()),
                    "inner_last": str(iv.timestamp.max()), "outer_first": str(ev.timestamp.min()), **policy})
                alarms.append({"boundary_name": "q95", "group": group, "split": split, "score": score_name,
                    "policy": "past_inner_selected", "cap": cap, **alarm_metrics(ev.increase.ge(primary), outer_score, policy["threshold"])})
                frontier = threshold_at_cap(ev.increase.ge(primary), outer_score, cap)
                alarms.append({"boundary_name": "q95", "group": group, "split": split, "score": score_name,
                    "policy": "descriptive_outer_frontier", "cap": cap, **alarm_metrics(ev.increase.ge(primary), outer_score, frontier["threshold"])})
    configs = json.loads((first / "selected_configurations.json").read_text(encoding="utf-8"))
    gate = next(r for r in configs if r["split"] == "dev_apr")["classifiers"]["HGB"]["gate"]
    april = p.loc[p.group.eq("G_HGB") & p.split.eq("dev_apr") & p.prior_low].copy()
    pool_scores = april.probability.to_numpy()
    indexed = p.set_index(["group", "timestamp"])
    diagnostic = []
    for _, r in april.loc[april.rise_event].iterrows():
        b = float(indexed.loc[("B", r.timestamp), "prediction"])
        w = float(indexed.loc[("W_rise_weighted", r.timestamp), "prediction"])
        diagnostic.append({"timestamp": r.timestamp, "actual": r.actual, "previous_maximum": r.lag1_maximum,
            "increase": r.increase, "surge": bool(r.surge), "probability": r.probability,
            "rank_min": 1 + int((pool_scores > r.probability).sum()), "rank_max": int((pool_scores >= r.probability).sum()),
            "ranking_pool_hours": len(april), "threshold": gate["threshold"], "blend": gate["blend"], "trigger": bool(r.trigger),
            "prediction_B": b, "prediction_expert": w, "prediction_combined": r.prediction,
            "expert_minus_B": w - b, "counterfactual_blend_one": w if r.trigger else b,
            "B_under": max(r.actual-b,0), "expert_under": max(r.actual-w,0)})
    assert len(diagnostic) == 8
    OUT.mkdir(parents=True, exist_ok=True)
    for data, filename in [(counts,"event_counts.csv"), (errors,"errors.csv"), (alarms,"alarm_metrics.csv"),
        (policies,"alarm_policies.csv"), (diagnostic,"april_legacy_cases.csv")]:
        pd.DataFrame(data).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    p.to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")
    summary = {"boundary_primary": primary, "boundary_sensitivity": boundaries["q90"],
        "calibration_hours": len(calibration), "calibration_positive_hours": len(positive),
        "calibration_first": str(calibration.timestamp.min()), "calibration_last": str(calibration.timestamp.max()),
        "development_surges": int(p.loc[p.group.eq("B"), "surge"].sum()),
        "april_old_cases": len(diagnostic), "april_old_cases_above_gate": sum(r["trigger"] for r in diagnostic),
        "april_old_rank_min": min(r["rank_min"] for r in diagnostic), "april_old_rank_max": max(r["rank_max"] for r in diagnostic),
        "no_fitting": True, "no_july_august_evaluation": True}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    run = {"status": "completed", "finished": now(), "contract_sha256": sha(CONTRACT), "script_sha256": sha(Path(__file__)),
        "inputs_sha256": {str(path.relative_to(ROOT).as_posix()): sha(path) for path in [frame_path, first/"run.json", first/"inner_predictions.csv", first/"selected_configurations.json", parent/"run.json", parent/"predictions.csv"]},
        "outputs_sha256": {path.name: sha(path) for path in OUT.iterdir()}, "prediction_rows": len(p),
        "no_fitting": True, "no_july_august_evaluation": True}
    (OUT / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    print(pd.DataFrame(errors).query("boundary_name=='q95' and split=='pooled_development' and condition=='surge'")[["group","hours","MAE","mean_under","mean_over"]].to_string(index=False), flush=True)
    print(pd.DataFrame(diagnostic)[["timestamp","increase","probability","rank_min","rank_max","trigger","prediction_B","prediction_expert"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
