"""Read-only regular CPython package startup check for authorized M02-2."""
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
result = {"executable": sys.executable, "version": sys.version, "packages": {}}
for name in ["numpy", "pandas", "sklearn", "joblib", "optuna", "xgboost", "lightgbm", "catboost"]:
    try:
        module = importlib.import_module(name)
        result["packages"][name] = {"status": "ok", "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:
        result["packages"][name] = {"status": "error", "type": type(exc).__name__, "message": str(exc)}
out = Path(__file__).resolve().parents[2] / "Modeling/tables/m022"
out.mkdir(parents=True, exist_ok=True)
(out / "environment.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result), flush=True)
