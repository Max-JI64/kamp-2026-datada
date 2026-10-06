import os
os.environ['OMP_NUM_THREADS']='4'
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
root=Path.cwd(); out=root/'Modeling/tables/feature_weather_audit'
tr=pd.read_csv(out/'train_04.csv',float_precision='round_trip')
ev=pd.read_csv(out/'eval_04.csv',float_precision='round_trip')
m=joblib.load(root/'Modeling/models/regime_integration/04_B0.joblib'); cols=list(m.feature_names_in_)
X=ev[cols]; bg=tr[cols].iloc[np.sort(np.random.default_rng(42).choice(len(tr),100,replace=False))]
actual=m.predict(X)
for method in ['tree_path_dependent','interventional']:
    ex=shap.TreeExplainer(m,data=bg if method=='interventional' else None,feature_perturbation=method,model_output='raw')
    phi=ex.shap_values(X,check_additivity=False)
    rec=phi.sum(axis=1)+float(ex.expected_value)
    native=ex.model.predict(X.to_numpy())
    idx=int(np.argmax(abs(rec-actual)))
    print(method,'max reconstruction error',float(max(abs(rec-actual))),'converted model diff',float(max(abs(native-actual))), 'worst',idx,actual[idx],rec[idx],native[idx],flush=True)
    print('base',ex.expected_value,'background mean',m.predict(bg).mean(),'finite',np.isfinite(X).all().all(),'input dtype',ex.model.input_dtype,flush=True)
