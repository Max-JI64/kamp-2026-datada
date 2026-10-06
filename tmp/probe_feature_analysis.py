import json
import importlib
import sys
result = {'executable': sys.executable, 'version': sys.version}
for name in ['numpy', 'pandas', 'sklearn', 'shap', 'scipy', 'joblib']:
    try:
        module = importlib.import_module(name)
        result[name] = getattr(module, '__version__', 'available')
    except Exception as exc:
        result[name] = {'error_type': type(exc).__name__, 'message': str(exc)}
print(json.dumps(result, ensure_ascii=False), flush=True)
