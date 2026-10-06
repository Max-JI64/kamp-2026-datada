"""Verify current links and every displayed expanded-M02 numeric table row."""
import ast
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from m03_verify import ROOT, rows, sha
from verify_m03_documents import main as verify_previous_documents


def main():
    out = ROOT / 'Modeling/tables/m02_rerun'
    v = json.loads((out / 'independent_verification.json').read_text(encoding='utf-8'))
    assert v['status'] == 'passed' and v['run_sha256'] == sha(out / 'run.json')
    docs = [ROOT / rel for rel in ['README.md', 'Modeling/README.md', 'Modeling/04_Modeling_원고.md',
                                  'Modeling/10.04_M02_확장모델비교_실행기록.md',
                                  'Modeling/10.04_M02_기본모형비교_실행기록.md']]
    links = 0
    for path in docs:
        text = path.read_text(encoding='utf-8-sig')
        if path == ROOT / 'README.md':
            text = text.split('## 이전 jsw 작업 단계 기록')[0]
        assert text.count('```') % 2 == 0
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
            if target.startswith(('https:', 'http:', '#', 'app:')):
                continue
            assert (path.parent / target.split('#')[0].strip('<>')).exists(), (path, target)
            links += 1
    manuscript = docs[2].read_text(encoding='utf-8')
    section = manuscript.split('## 4.4 M02 확장 재실행:')[1]
    assert manuscript.count('## 4.4 M02 확장 재실행:') == 1
    labels = {'previous_hour': '직전 값', 'previous_day': '전날 같은 시각', 'previous_week': '전주 같은 시각',
              'Ridge_initial': '기존 Ridge', 'ElasticNet_initial': '기존 Elastic Net', 'SVR_initial': '기존 RBF SVR',
              'HGB_initial': '기존 HGB', 'HGB': 'Optuna HGB', 'mean5': '5모델 동일 평균', 'weighted_top3': '내부 상위 3모델 가중 평균'}
    checks = 0
    for r in rows(out / 'selection_metrics.csv'):
        line = '| '+labels.get(r['model'], r['model'])+' | '+' | '.join(f'{float(r[k]):.3f}' for k in ['overall_MAE', 'daily_maximum_MAE'])+f" | {float(r['balanced_score']):.4f} |"
        assert line in section and line in docs[3].read_text(encoding='utf-8'), line
        checks += 1
    comparisons = {(r['model'], r['split']): r for r in rows(out / 'comparison.csv')}
    conditions = {(r['model'], r['condition'], r['value']): r for r in rows(out / 'condition_errors.csv')}
    for model in ['HGB_initial', 'ExtraTrees', 'weighted_top3']:
        numbers = [float(comparisons[(model, split)]['MAE']) for split in ['dev_apr', 'dev_may', 'dev_jun']]
        numbers.append(float(conditions[(model, 'train_profile_overlap', 'False')]['MAE']))
        line = '| '+labels.get(model, model)+' | '+' | '.join(f'{x:.3f}' for x in numbers)+' |'
        assert line in section, line
        checks += 1
    for value, label in [('low->above26', '낮은 구간→26 초과'), ('above26->low', '26 초과→낮은 구간'),
                         ('low->low', '낮은 구간 유지'), ('above26->above26', '26 초과 유지')]:
        a, b = (conditions[(m, 'power_transition', value)] for m in ['HGB_initial', 'ExtraTrees'])
        line = f"| {label} | {a['hours']} | {float(a['MAE']):.3f} | {float(b['MAE']):.3f} |"
        assert line in section, line
        checks += 1
    hgb = json.loads((ROOT / 'Modeling/config/m02_contract.json').read_text(encoding='utf-8'))['grids']['HGB']
    assert hgb['max_leaf_nodes'] == [7, 15]
    fixed = {'max_leaf_nodes': 15, 'max_iter': 150, 'learning_rate': .05, 'min_samples_leaf': 20, 'l2_regularization': 1.0}
    names = {'max_leaf_nodes': '최대 잎 수', 'max_iter': '반복 횟수', 'learning_rate': '학습률',
             'min_samples_leaf': '잎의 최소 표본 수', 'l2_regularization': 'L2 규제'}
    for key, value in fixed.items():
        assert f'| {names[key]} (`{key}`) | {value} |' in section
        if key != 'max_leaf_nodes':
            assert hgb[key] == value
        checks += 1
    trials = [r for r in rows(out / 'trials.csv') if r['family'] == 'HGB']
    assert len(trials) == 90
    assert {split: sum(r['split'] == split for r in trials) for split in ['dev_apr', 'dev_may', 'dev_jun']} == {'dev_apr': 30, 'dev_may': 30, 'dev_jun': 30}
    for model in ['HGB_initial', 'HGB']:
        r = next(r for r in rows(out / 'selection_metrics.csv') if r['model'] == model)
        assert f"| {float(r['overall_MAE']):.3f} | {float(r['daily_maximum_MAE']):.3f} |" in section
        checks += 1
    memory = (ROOT / 'AGENT_MEMORY.md').read_text(encoding='utf-8')
    assert memory.count('- `session:20261004-1056`') == 1
    assert memory.count('- `decision:modeling-m02-rerun-metric-first`') == 1
    for name in ['m02_rerun.py', 'm02_rerun_verify.py', 'm02_rerun_documents.py', 'm02_rerun_document_verify.py']:
        ast.parse((ROOT / 'Modeling/scripts' / name).read_text(encoding='utf-8'))
    verify_previous_documents()
    result = {'status': 'passed', 'verified_at': datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='minutes'),
              'links_checked': links, 'expanded_M02_table_rows_checked': checks,
              'HGB_tuning_trials_checked': len(trials),
              'documents_sha256': {str(p.relative_to(ROOT)): sha(p) for p in docs},
              'no_new_fitting': True, 'script_sha256': sha(Path(__file__))}
    (out / 'document_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
