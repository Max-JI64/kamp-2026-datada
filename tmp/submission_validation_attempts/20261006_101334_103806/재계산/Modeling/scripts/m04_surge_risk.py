"""All-state surge classification and a separate upper quantile; chronological development."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, mean_pinball_loss

from m01_prepare import ROOT, sha, read_contract
from m04_rise_compare import classifier, now, read_csv
from m04_surge_diagnose import alarm_metrics, threshold_at_cap

OUT = ROOT / "Modeling/tables/m04_surge_risk"
MODELS = ROOT / "Modeling/models/m04_surge_risk"
CONTRACT = ROOT / "Modeling/config/m04_surge_risk_contract.json"


def probability(model, inputs):
    return np.full(len(inputs),model["constant_probability"]) if isinstance(model,dict) else model.predict_proba(inputs)[:,1]


def fit_classifier(name, parameters, tr, ev, features):
    labels = tr.increase.ge(93).astype(int)
    if labels.nunique()<2:
        model = {"constant_probability":float(labels.mean())}
        return model, probability(model,ev[features]), True
    model = classifier(name.removesuffix("_all"),parameters)
    model.fit(tr[features],labels)
    return model, probability(model,ev[features]), False


def rank(y, score):
    return {"AP":float(average_precision_score(y,score)), "AUROC":float(roc_auc_score(y,score)), "events":int(np.sum(y)), "hours":len(y)}


def regression(frame, prediction):
    error = np.asarray(prediction)-frame.actual.to_numpy()
    weight = frame.daily_maximum_weight.to_numpy()
    return {"hours":len(frame), "MAE":float(np.abs(error).mean()), "mean_under":float(np.maximum(-error,0).mean()),
        "mean_over":float(np.maximum(error,0).mean()), "coverage":float(np.mean(error>=0)),
        "pinball90":float(mean_pinball_loss(frame.actual,prediction,alpha=.9)),
        "daily_maximum_MAE":float(np.average(np.abs(error),weights=weight)) if weight.sum() else np.nan}


def main():
    assert sys.version_info[:2]==(3,13) and sys._is_gil_enabled()
    assert not (OUT/"run.json").exists(), "Preserve completed comparison."
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    diagnosis = ROOT/"Modeling/tables/m04_surge"
    verified = json.loads((diagnosis/"independent_verification.json").read_text(encoding="utf-8"))
    assert verified["status"]=="passed" and verified["run_sha256"]==sha(diagnosis/"run.json")==c["diagnosis_run_sha256"]
    assert verified["boundary"]["q95"]==c["boundary_primary"]==93
    parent = ROOT/"Modeling/tables/m04_rise"
    parent_run = json.loads((parent/"run.json").read_text(encoding="utf-8"))
    for filename in ["feature_frame.csv","inner_predictions.csv","predictions.csv"]:
        assert sha(parent/filename)==parent_run["outputs_sha256"][filename]
    original_contract = json.loads((ROOT/"Modeling/config/m04_rise_contract.json").read_text(encoding="utf-8"))
    m1 = read_contract()
    f = read_csv(ROOT/"Modeling/tables/m01/hourly_frame.csv",["timestamp","date"])
    f = f.loc[f.eligible_common & f.timestamp.lt("2021-07-01")].copy()
    f = f.merge(read_csv(parent/"feature_frame.csv",["timestamp"])[["timestamp"]+original_contract["features"]["add"]],on="timestamp",validate="one_to_one")
    f["increase"] = f.target_maximum-f.lag1_maximum
    features = parent_run["features"]["dynamic"]
    base_features = parent_run["features"]["B"]
    cached = read_csv(parent/"inner_predictions.csv",["timestamp","date"])
    old = read_csv(parent/"predictions.csv",["timestamp","date"])
    OUT.mkdir(parents=True,exist_ok=True)
    MODELS.mkdir(parents=True,exist_ok=True)
    predictions,inners,trials,configs,manifests,fold_audit,policies,alarms,ranks,regressions = [],[],[],[],[],[],[],[],[],[]
    trial_arrays = {}
    fit_counts = {"inner_classifier":0,"inner_quantile":0,"outer_classifier":0,"outer_quantile":0}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for outer in m1["splits"]:
            split = outer["name"]
            if split not in c["time"]["outer"]:
                continue
            tr = f.loc[f.timestamp.le(outer["train_end"])]
            ev = f.loc[f.timestamp.between(outer["eval_start"],outer["eval_end"])].copy()
            template = old.loc[old.split.eq(split)&old.group.eq("B")].sort_values("timestamp").copy()
            assert template.timestamp.tolist()==ev.timestamp.tolist()
            template["increase"] = ev.increase.to_numpy()
            template["previous_maximum"] = ev.lag1_maximum.to_numpy()
            template["normal_prediction"] = template.prediction
            folds = []
            for month in c["time"]["inner_months"][split]:
                start = pd.Timestamp(2021,month,1)
                t = tr.loc[tr.timestamp.lt(start)]
                v = tr.loc[tr.timestamp.dt.month.eq(month)]
                assert t.timestamp.max()<v.timestamp.min()<=v.timestamp.max()<ev.timestamp.min()
                folds.append((t,v))
                fold_audit.append({"split":split,"month":month,"train_hours":len(t),"validation_hours":len(v),
                    "train_events":int(t.increase.ge(93).sum()),"validation_events":int(v.increase.ge(93).sum()),
                    "train_last":str(t.timestamp.max()),"validation_first":str(v.timestamp.min()),"validation_last":str(v.timestamp.max()),"outer_first":str(ev.timestamp.min())})
            iv = pd.concat([v for t,v in folds],ignore_index=True)
            inner_score,outer_score = {},{}
            for name,source_group in [("B_score","B"),("G_HGB_score","G_HGB")]:
                record = cached.loc[cached.split.eq(split)&cached.group.eq(source_group)].sort_values("timestamp")
                assert record.timestamp.tolist()==iv.timestamp.tolist()
                inner_score[name] = record.prediction.to_numpy()-iv.lag1_maximum.to_numpy()
                outer_record = old.loc[old.split.eq(split)&old.group.eq(source_group)].sort_values("timestamp")
                outer_score[name] = outer_record.prediction.to_numpy()-ev.lag1_maximum.to_numpy()
            for name in c["classifiers"]["methods"]:
                predictions_by_trial = {}
                def objective(trial):
                    if name=="Logistic_all":
                        parameters = {"C":trial.suggest_float("C",.001,100,log=True)}
                    else:
                        parameters = {"max_leaf_nodes":trial.suggest_categorical("max_leaf_nodes",[3,7,15]),
                            "min_samples_leaf":trial.suggest_categorical("min_samples_leaf",[10,20,40]),
                            "l2_regularization":trial.suggest_float("l2_regularization",.1,10,log=True)}
                    pieces,constants = [],0
                    for t,v in folds:
                        model,values,constant = fit_classifier(name,parameters,t,v,features)
                        fit_counts["inner_classifier"]+=int(not constant)
                        constants+=int(constant)
                        pieces.append(values)
                    values = np.concatenate(pieces)
                    result = rank(iv.increase.ge(93),values)
                    result["Brier"] = float(brier_score_loss(iv.increase.ge(93),values))
                    trial.set_user_attr("metrics",result)
                    trial.set_user_attr("constant_folds",constants)
                    predictions_by_trial[trial.number]=values
                    return result["AP"]
                study = optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=10),pruner=optuna.pruners.NopPruner())
                study.optimize(objective,n_trials=20)
                selected = min(study.trials,key=lambda t:(-t.value,t.user_attrs["metrics"]["Brier"],t.number))
                inner_score[name]=predictions_by_trial[selected.number]
                trial_arrays[split+"_"+name]=np.stack([predictions_by_trial[t.number] for t in study.trials])
                model,values,constant = fit_classifier(name,selected.params,tr,ev,features)
                outer_score[name]=values
                fit_counts["outer_classifier"]+=int(not constant)
                path = MODELS/f"{split}_{name}.joblib"
                joblib.dump(model,path,compress=3)
                assert np.allclose(probability(joblib.load(path),ev[features]),values,atol=1e-9,rtol=0)
                manifests.append({"split":split,"name":name,"file":path.relative_to(ROOT).as_posix(),"sha256":sha(path),"features":len(features)})
                configs.append({"split":split,"name":name,"best_trial":selected.number,"parameters":selected.params,"inner_metrics":selected.user_attrs["metrics"],"constant_outer":constant})
                for trial in study.trials:
                    trials.append({"split":split,"name":name,"trial":trial.number,"state":trial.state.name,"objective":trial.value,
                        "parameters":json.dumps(trial.params),"metrics":json.dumps(trial.user_attrs)})
                print(f"{split} {name} inner AP={selected.value:.4f}",flush=True)
            quantile_pieces=[]
            for t,v in folds:
                model=HistGradientBoostingRegressor(**original_contract["hgb_parameters"],loss="quantile",quantile=.9)
                model.fit(t[base_features],t.target_maximum)
                quantile_pieces.append(model.predict(v[base_features]))
                fit_counts["inner_quantile"]+=1
            inner_upper=np.concatenate(quantile_pieces)
            model=HistGradientBoostingRegressor(**original_contract["hgb_parameters"],loss="quantile",quantile=.9)
            model.fit(tr[base_features],tr.target_maximum)
            outer_upper=model.predict(ev[base_features])
            fit_counts["outer_quantile"]+=1
            path=MODELS/f"{split}_Q90_upper.joblib"
            joblib.dump(model,path,compress=3)
            assert np.allclose(joblib.load(path).predict(ev[base_features]),outer_upper,atol=1e-9,rtol=0)
            manifests.append({"split":split,"name":"Q90_upper","file":path.relative_to(ROOT).as_posix(),"sha256":sha(path),"features":18})
            inner_score["Q90_upper"]=inner_upper-iv.lag1_maximum.to_numpy()
            outer_score["Q90_upper"]=outer_upper-ev.lag1_maximum.to_numpy()
            for name in c["comparators"]:
                record=template.copy()
                record["group"]=name
                record["score"]=outer_score[name]
                record["score_type"]="probability" if name.endswith("_all") else "predicted_increase"
                record["upper_prediction"]=outer_upper if name=="Q90_upper" else np.nan
                inner_record=iv[["timestamp","date","increase","lag1_maximum","target_maximum"]].copy()
                inner_record["split"],inner_record["group"],inner_record["score"]=split,name,inner_score[name]
                inner_record["upper_prediction"]=inner_upper if name=="Q90_upper" else np.nan
                inners.append(inner_record)
                for cap in [c["alarms"]["primary_false_positive_cap"],c["alarms"]["secondary_false_positive_cap"]]:
                    chosen=threshold_at_cap(iv.increase.ge(93),inner_score[name],cap)
                    policies.append({"split":split,"group":name,"cap":cap,**chosen})
                    alarm=outer_score[name]>=chosen["threshold"]
                    record["alarm_"+str(cap)]=alarm
                    for key,boundary in [("q95",93.),("q90",c["boundary_sensitivity"])]:
                        values=alarm_metrics(ev.increase.ge(boundary),outer_score[name],chosen["threshold"])
                        if name.endswith("_all") and key=="q95":
                            values["Brier"]=float(brier_score_loss(ev.increase.ge(boundary),outer_score[name]))
                        alarms.append({"split":split,"group":name,"cap":cap,"boundary_name":key,**values})
                predictions.append(record)
            for name,prediction in [("B",template.normal_prediction.to_numpy()),("G_HGB",old.loc[old.split.eq(split)&old.group.eq("G_HGB")].sort_values("timestamp").prediction.to_numpy()),("Q90_upper",outer_upper)]:
                for condition,mask in [("all",np.ones(len(ev),dtype=bool)),("surge_q95",ev.increase.ge(93)),("surge_q90",ev.increase.ge(c["boundary_sensitivity"])),("low_stay",template.power_transition.eq("low->low").to_numpy())]:
                    part=template.loc[np.asarray(mask)]
                    if len(part):
                        regressions.append({"split":split,"group":name,"condition":condition,**regression(part,np.asarray(prediction)[np.asarray(mask)])})
        p=pd.concat(predictions,ignore_index=True)
        # All new methods retain exactly B as their normal point forecast.
        assert np.array_equal(p.prediction.to_numpy(),p.normal_prediction.to_numpy())
        alarm_frame=pd.DataFrame(alarms)
        selection=[]
        for name,g in p.groupby("group"):
            y=g.increase.ge(93).to_numpy()
            values=rank(y,g.score)
            monthly=alarm_frame.loc[alarm_frame.group.eq(name)&alarm_frame.cap.eq(.01)&alarm_frame.boundary_name.eq("q95")]
            values["weighted_monthly_AP"]=float(np.average(monthly.AP,weights=monthly.events))
            chosen=g["alarm_0.01"].to_numpy()
            tp=int((chosen&y).sum());fp=int((chosen&~y).sum())
            values.update({"group":name,"TP":tp,"FP":fp,"recall":tp/int(y.sum()),"precision":tp/(tp+fp) if tp+fp else 0.,"false_positive_fraction":fp/int((~y).sum())})
            selection.append(values)
        table=pd.DataFrame(selection)
        ref=table.loc[table.group.eq("B_score")].iloc[0]
        for i,r in table.iterrows():
            monthly=alarm_frame.loc[alarm_frame.group.eq(r.group)&alarm_frame.cap.eq(.01)&alarm_frame.boundary_name.eq("q95")].set_index("split")
            baseline=alarm_frame.loc[alarm_frame.group.eq("B_score")&alarm_frame.cap.eq(.01)&alarm_frame.boundary_name.eq("q95")].set_index("split")
            flags={"rank_passed":bool(r.weighted_monthly_AP>=1.05*ref.weighted_monthly_AP),"recall_passed":bool(r.TP>=ref.TP),
                "monthly_recall_passed":bool((monthly.TP>=baseline.TP).all()),"false_positive_passed":bool(r.false_positive_fraction<=.01+1e-12),
                "strict_gain_passed":bool(r.TP>ref.TP or r.FP<ref.FP)}
            for key,value in flags.items():
                table.loc[i,key]=value
            table.loc[i,"eligible"]=r.group in ["Logistic_all","HGB_all","Q90_upper"] and all(flags.values())
        eligible=table.loc[table.eligible.astype(bool)]
        if len(eligible):
            near=eligible.loc[eligible.weighted_monthly_AP.ge(eligible.weighted_monthly_AP.max()/1.01)].copy()
            preference={name:i for i,name in enumerate(["Logistic_all","HGB_all","Q90_upper"])}
            near["preference"]=near.group.map(preference)
            chosen=near.sort_values(["preference","weighted_monthly_AP"],ascending=[True,False]).iloc[0].group
        else:
            chosen="B_score"
        selection_result={"selected_risk_channel":chosen,"normal_forecast":"B","eligible_methods":eligible.group.tolist(),"development_only":True,"no_july_august_evaluation":True}
        for data,filename in [(p,"predictions.csv"),(pd.concat(inners,ignore_index=True),"inner_predictions.csv"),(pd.DataFrame(trials),"trials.csv"),
            (pd.DataFrame(manifests),"model_manifest.csv"),(pd.DataFrame(fold_audit),"fold_audit.csv"),(pd.DataFrame(policies),"alarm_policies.csv"),
            (alarm_frame,"alarm_metrics.csv"),(pd.DataFrame(regressions),"regression_metrics.csv"),(table,"selection_metrics.csv")]:
            data.to_csv(OUT/filename,index=False,encoding="utf-8-sig")
        np.savez_compressed(OUT/"trial_oof_predictions.npz",**trial_arrays)
        (OUT/"selected_configurations.json").write_text(json.dumps(configs,indent=2),encoding="utf-8")
        (OUT/"selection.json").write_text(json.dumps(selection_result,indent=2),encoding="utf-8")
        warning_messages=[str(w.message) for w in caught]
    run={"status":"completed","finished":now(),"seconds":time.perf_counter()-started,"script_sha256":sha(Path(__file__)),"contract_sha256":sha(CONTRACT),
        "inputs_sha256":{path.relative_to(ROOT).as_posix():sha(path) for path in [diagnosis/"run.json",diagnosis/"independent_verification.json",parent/"run.json",parent/"feature_frame.csv",parent/"inner_predictions.csv",parent/"predictions.csv",ROOT/"Modeling/tables/m01/hourly_frame.csv",ROOT/"Modeling/config/m04_rise_contract.json"]},
        "helper_sha256":{name:sha(ROOT/"Modeling/scripts"/name) for name in ["m01_prepare.py","m04_rise_compare.py","m04_surge_diagnose.py"]},
        "features":{"classifier":features,"quantile":base_features},"fits":fit_counts,"trials":len(trials),"warnings":warning_messages,
        "normal_forecast_unchanged":True,"no_july_august_evaluation":True,"runtime":{"executable":sys.executable,"version":sys.version},
        "outputs_sha256":{path.name:sha(path) for path in OUT.iterdir()},"prediction_rows":len(p)}
    (OUT/"run.json").write_text(json.dumps(run,indent=2),encoding="utf-8")
    print(table.to_string(index=False),flush=True)
    print(json.dumps(selection_result),flush=True)
    print(json.dumps({"fits":fit_counts,"trials":len(trials),"warnings":warning_messages}),flush=True)


if __name__=="__main__":
    main()
