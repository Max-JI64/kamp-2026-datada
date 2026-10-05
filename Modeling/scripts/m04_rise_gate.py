"""One objective change using verified past-only cached predictions; no new fitting."""
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from m01_prepare import ROOT, sha
from m02_compare import metric
from m04_compare import subsets
from m04_rise_compare import gated, stats, score, alarm_stats, now, read_csv

OUT = ROOT / "Modeling/tables/m04_rise_gate"
CONTRACT = ROOT / "Modeling/config/m04_rise_gate_contract.json"


def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def main():
    assert sys.version_info[:2] == (3,13) and sys._is_gil_enabled()
    assert not (OUT / "run.json").exists(), "Preserve completed run."
    follow = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent_dir = ROOT / "Modeling/tables/m04_rise"
    parent_run = json.loads((parent_dir / "run.json").read_text(encoding="utf-8"))
    verified = json.loads((parent_dir / "independent_verification.json").read_text(encoding="utf-8"))
    assert verified["status"] == "passed" and verified["run_sha256"] == sha(parent_dir / "run.json") == follow["parent_run_sha256"]
    for name in ["predictions.csv", "inner_predictions.csv"]:
        assert sha(parent_dir / name) == parent_run["outputs_sha256"][name]
    c = json.loads((ROOT / "Modeling/config/m04_rise_contract.json").read_text(encoding="utf-8"))
    p = read_csv(parent_dir / "predictions.csv", ["timestamp","date"])
    inner = read_csv(parent_dir / "inner_predictions.csv", ["timestamp","date"])
    OUT.mkdir(parents=True, exist_ok=True)
    records, configs, trials, classifications, selected_inner = [p], [], [], [], []
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for split in c["outer_splits"]:
        iv = inner.loc[inner.split.eq(split) & inner.group.eq("B")].copy()
        iv["target_maximum"] = iv.actual
        ib = iv.prediction.to_numpy()
        iw = inner.loc[inner.split.eq(split) & inner.group.eq("W_rise_weighted")].prediction.to_numpy()
        ev = p.loc[p.split.eq(split) & p.group.eq("B")].copy()
        ev["target_maximum"] = ev.actual
        expert = p.loc[p.split.eq(split) & p.group.eq("W_rise_weighted")].prediction.to_numpy()
        reference = stats(iv, ib)
        for family in follow["families"]:
            ip = inner.loc[inner.split.eq(split) & inner.group.eq("G_"+family)].probability.to_numpy()
            op = p.loc[p.split.eq(split) & p.group.eq("G_"+family)].probability.to_numpy()
            def objective(trial):
                threshold = trial.suggest_float("threshold", *c["gate"]["threshold"])
                blend = trial.suggest_float("blend", *c["gate"]["blend"])
                prediction, trigger = gated(ib, iw, ip, iv.prior_low, threshold, blend)
                values = stats(iv, prediction)
                alarm = alarm_stats(iv, ip, threshold)
                maximum_term = max(values[k] / reference[k] for k in ["overall_MAE","daily_maximum_MAE","rise_MAE"])
                new_score = score(values, reference, c["development_selection"], alarm["false_positive_fraction"]) - maximum_term + values["rise_MAE"] / reference["rise_MAE"]
                trial.set_user_attr("metrics", values)
                trial.set_user_attr("classification", alarm)
                return new_score
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=10), pruner=optuna.pruners.NopPruner())
            for params in c["gate"]["enqueue"]:
                study.enqueue_trial(params)
            study.optimize(objective,n_trials=50)
            best = min(study.trials,key=lambda trial:(trial.value,trial.number))
            name = "G_" + family + "_rise_focus"
            rec = ev.copy()
            rec["group"] = name
            rec["prediction"], rec["trigger"] = gated(ev.prediction,expert,op,ev.prior_low,**best.params)
            rec["probability"] = op
            rec = rec.drop(columns="target_maximum")
            records.append(rec)
            ir = iv.copy()
            ir["group"] = name
            ir["prediction"], ir["trigger"] = gated(ib,iw,ip,iv.prior_low,**best.params)
            ir["probability"] = ip
            selected_inner.append(ir.drop(columns="target_maximum"))
            configs.append({"split":split,"group":name,"best_trial":best.number,"parameters":best.params,"inner_metrics":best.user_attrs})
            for trial in study.trials:
                trials.append({"split":split,"group":name,"trial":trial.number,"objective":trial.value,"parameters":json.dumps(trial.params),"metrics":json.dumps(trial.user_attrs)})
            for role,part,prob in [("inner",iv,ip),("outer",ev,op)]:
                classifications.append({"split":split,"role":role,"group":name,**alarm_stats(part,prob,best.params["threshold"])})
            print(f"{split} {name}: {best.params}",flush=True)
    p = pd.concat(records,ignore_index=True)
    p["signed_error"] = p.prediction-p.actual
    p["absolute_error"] = p.signed_error.abs()
    p["under_amount"] = (-p.signed_error).clip(lower=0)
    p["over_amount"] = p.signed_error.clip(lower=0)
    save(p,"predictions.csv")
    save(pd.concat(selected_inner,ignore_index=True),"inner_predictions.csv")
    save(pd.DataFrame(trials),"trials.csv")
    save(pd.DataFrame(classifications),"classification.csv")
    (OUT/"selected_configurations.json").write_text(json.dumps(configs,indent=2),encoding="utf-8")
    metrics,table = [],[]
    baseline = p.loc[p.group.eq("B")].sort_values("timestamp").copy()
    baseline["target_maximum"] = baseline.actual
    reference = stats(baseline,baseline.prediction)
    for name,part in p.groupby("group"):
        for kind,value,subset,weights in subsets(part):
            metrics.append({"group":name,"kind":kind,"value":value,**metric(subset,weights)})
        values = stats(baseline,part.sort_values("timestamp").prediction)
        row = {"group":name,**values,"balanced_score":max(values[k]/reference[k] for k in ["overall_MAE","daily_maximum_MAE","rise_MAE"])}
        conditions = []
        for key,setting in [("overall_MAE","overall_MAE_ratio_max"),("daily_maximum_MAE","daily_maximum_MAE_ratio_max"),
            ("low_stay_MAE","low_stay_MAE_ratio_max"),("low_stay_over","low_stay_mean_over_ratio_max"),
            ("rise_MAE","rise_MAE_ratio_max"),("rise_under","rise_mean_under_ratio_max")]:
            row[key+"_ratio"] = values[key]/reference[key]
            row[key+"_passed"] = bool(row[key+"_ratio"] <= c["development_selection"][setting]+1e-12)
            conditions.append(row[key+"_passed"])
        row["rise_monthly_passed"] = bool(all(part.loc[part.split.eq(split)&part.rise_event,"absolute_error"].mean() <=
            baseline.loc[baseline.split.eq(split)&baseline.rise_event,"absolute_error"].mean()+1e-9 for split in c["outer_splits"]))
        conditions.append(row["rise_monthly_passed"])
        if name.startswith("G_"):
            low = part.loc[part.prior_low]
            fp = int((low.trigger&~low.rise_event).sum())
            negatives = int((~low.rise_event).sum())
            row["gate_false_positive_fraction"] = fp/negatives
            row["gate_false_positive_passed"] = bool(fp/negatives <= .05+1e-12)
            conditions.append(row["gate_false_positive_passed"])
        row["eligible"] = name!="B" and all(conditions)
        table.append(row)
    table = pd.DataFrame(table)
    eligible = table.loc[table.eligible].copy()
    order = c["methods"]+["G_Logistic_rise_focus","G_HGB_rise_focus"]
    if len(eligible):
        near = eligible.loc[eligible.balanced_score.le(eligible.balanced_score.min()*1.01)].copy()
        near["preference"] = near.group.map({name:i for i,name in enumerate(order)})
        chosen = near.sort_values(["preference","balanced_score"]).iloc[0].group
    else:
        chosen = "B"
    selection = {"selected":chosen,"eligible_methods":eligible.group.tolist(),"acceptance_criteria_unchanged":True,"development_only":True,"no_july_august_evaluation":True}
    save(pd.DataFrame(metrics),"metrics.csv")
    save(table,"selection_metrics.csv")
    (OUT/"selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
    run = {"status":"completed","finished":now(),"contract_sha256":sha(CONTRACT),"parent_run_sha256":sha(parent_dir/"run.json"),
        "script_sha256":sha(Path(__file__)),"no_new_fitting":True,"no_july_august_evaluation":True,"trials":len(trials),
        "outputs_sha256":{path.name:sha(path) for path in OUT.iterdir() if path.name!="run.json"},"prediction_rows":len(p),"metric_rows":len(metrics)}
    (OUT/"run.json").write_text(json.dumps(run,indent=2),encoding="utf-8")
    print(table[["group","overall_MAE","daily_maximum_MAE","rise_MAE","rise_under","low_stay_MAE","low_stay_over","eligible"]].to_string(index=False),flush=True)
    print(json.dumps(selection),flush=True)


if __name__ == "__main__":
    main()
