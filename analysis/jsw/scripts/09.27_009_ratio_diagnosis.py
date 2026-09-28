"""Diagnose a failed exact-formula assertion before choosing a tolerance."""
import json
import numpy as np
import pandas as pd
from analysis_common import load, SLOTS

raw, _ = load(include_september=True)
denominator = raw[SLOTS].sum(axis=1).to_numpy(float)
ratio = np.divide(raw["생산량"].to_numpy(float), denominator, out=np.full(len(raw), np.nan), where=denominator > 0)
error = raw["공장인원"].to_numpy(float) - ratio
valid = np.isfinite(error)
failed = valid & ~np.isclose(raw["공장인원"], ratio, rtol=1e-12, atol=1e-10)
rows = raw.loc[failed, ["날짜", "시간", "생산량", "공장인원", *SLOTS]].copy()
rows["ratio"] = ratio[failed]
rows["difference"] = error[failed]
result = {"failed": int(failed.sum()), "max_absolute_error": float(np.abs(error[valid]).max()),
          "rounded_9_decimal_matches": int(np.isclose(raw.loc[valid, "공장인원"], np.round(ratio[valid], 9), rtol=0, atol=1e-12).sum()),
          "examples": rows.iloc[:5].to_dict("records")}
print(json.dumps(result, ensure_ascii=True), flush=True)
