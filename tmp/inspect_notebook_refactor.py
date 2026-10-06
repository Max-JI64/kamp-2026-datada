from pathlib import Path
import ast,json,importlib.metadata
root=Path(__file__).resolve().parents[1]
names=['data_overview','eda_daily_pattern','eda_daily_repetition','eda_variable_relations','eda_weekday_power_levels','eda_slot_time_patterns','plot_daily_repetition_simple','plot_monthly_power_heatmap','plot_report_daily_patterns','eda_figures','p01_peak_shapes','m01_prepare','regime_diagnose','regime_forecast','regime_followup','modeling_close_analysis']
for name in names:
    p=next(root.glob('*/scripts/'+name+'.py'));t=ast.parse(p.read_text(encoding='utf-8-sig'))
    print(name,'imports:',[ast.unparse(n) for n in t.body if isinstance(n,(ast.Import,ast.ImportFrom))])
    for f in [n for n in t.body if isinstance(n,ast.FunctionDef) and n.name in ['main','run','load_frame','extended','model','threshold']]:
        print(' function',f.name,'arguments',ast.unparse(f.args),'body lines',f.end_lineno-f.lineno)
    print(' top vars',[ast.unparse(n) for n in t.body if isinstance(n,(ast.Assign,ast.AnnAssign))][:15])
for p in ['nbclient','nbformat','jupyter_client','ipykernel','jupyterlab']:
    try:print(p,importlib.metadata.version(p))
    except importlib.metadata.PackageNotFoundError:print(p,'missing')
