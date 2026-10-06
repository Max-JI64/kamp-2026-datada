"""Record or verify the complete preservation inventory for the current Modeling manuscript.

No training, prediction, deletion, package installation, or original-data modification.
Run --record after an intentional documented update, then use --verify before cleanup.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'Modeling/scripts'
MANIFEST = ROOT / 'Modeling/submission_manifest.json'
PACKAGES = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'optuna',
            'xgboost', 'lightgbm', 'catboost', 'matplotlib', 'Pillow']
MANUSCRIPT_SCRIPT_NAMES = [
    'm01_prepare.py', 'm01_verify.py', 'm02_compare.py', 'm02_verify.py',
    'm03_ablation.py', 'm03_verify.py', 'm03_error_followup.py',
    'plot_observed_power_timeline.py', 'plot_m02_manuscript.py', 'plot_m03_manuscript.py',
    'm022_environment.py', 'm022_compare.py', 'm02_rerun.py', 'm02_rerun_verify.py',
    'm02_rerun_documents.py', 'verify_m03_documents.py', 'm02_rerun_document_verify.py',
    'm04_compare.py', 'm04_verify.py', 'modeling_submission_audit.py',
    'm04_rise_compare.py', 'm04_rise_verify.py', 'm04_rise_gate.py',
    'm04_rise_gate_verify.py', 'm04_rise_documents.py',
    'm04_surge_diagnose.py', 'm04_surge_verify.py', 'm04_surge_risk.py',
    'm04_surge_risk_verify.py', 'm04_surge_calibration.py',
    'm04_surge_calibration_verify.py', 'm04_surge_documents.py',
    'm05_evaluate.py', 'm05_verify.py', 'm05_documents.py',
    'redevelopment_r01.py', 'redevelopment_compare.py', 'redevelopment_verify.py', 'redevelopment_documents.py',
    'redevelopment_peak.py', 'redevelopment_peak_verify.py',
]
RESEARCH_SCRIPT_NAMES = ['profile_history.py', 'profile_history_verify.py', 'training_design_diagnose.py', 'training_design_compare.py', 'training_design_recent.py', 'training_design_verify.py', 'error_warning.py', 'error_warning_fit.py', 'error_warning_verify.py', 'priority_review.py', 'priority_review_verify.py', 'priority_review_documents.py', 'regime_diagnose.py', 'regime_diagnose_verify.py', 'regime_forecast.py', 'regime_forecast_verify.py', 'regime_integrate.py', 'regime_integrate_verify.py', 'regime_route.py', 'regime_route_verify.py', 'regime_followup.py', 'regime_followup_verify.py']
RESEARCH_SCRIPT_NAMES += ['regime_age_ablation.py', 'regime_age_ablation_verify.py', 'regime_age_calendar_control.py']
MANUSCRIPT_SCRIPT_NAMES += ['regime_manuscript_documents.py']
MANUSCRIPT_SCRIPT_NAMES += ['modeling_close_analysis.py', 'modeling_close_figures.py', 'modeling_close_documents.py', 'modeling_close_verify.py', 'modeling_reproduce_selected.py']
SCRIPT_NAMES = MANUSCRIPT_SCRIPT_NAMES + RESEARCH_SCRIPT_NAMES
TABLE_DIRS = ['m01', 'm02', 'm03/ab', 'm03/c', 'm03/diagnostics', 'm02_rerun', 'm04', 'm04_rise', 'm04_rise_gate', 'm04_surge', 'm04_surge_risk', 'm04_surge_calibration', 'm05', 'observed_timeline', 'redevelopment', 'profile_history', 'training_design', 'error_warning', 'priority_review', 'regime_diagnosis', 'regime_forecast', 'regime_integration', 'regime_routing', 'regime_followup']
MODEL_DIRS = ['m02', 'm03/ab', 'm03/c', 'm02_rerun', 'm04_rise', 'm04_surge_risk', 'm05', 'redevelopment', 'profile_history', 'training_design', 'error_warning', 'regime_forecast', 'regime_integration', 'regime_followup']
TABLE_DIRS += ['regime_age_ablation']
TABLE_DIRS += ['regime_manuscript']
TABLE_DIRS += ['modeling_close', 'modeling_replay']
MODEL_DIRS += ['regime_age_ablation']
CONFIG_NAMES = ['m01_contract.json', 'm02_contract.json', 'm03_ab_contract.json',
                'm03_c_contract.json', 'm02_rerun_contract.json', 'm022_contract.json', 'm04_contract.json',
                'm04_rise_contract.json', 'm04_rise_gate_contract.json', 'm04_surge_contract.json',
                'm04_surge_risk_contract.json', 'm04_surge_calibration_contract.json', 'm05_contract.json', 'redevelopment_r01_contract.json', 'redevelopment_r02_contract.json', 'redevelopment_peak_contract.json', 'profile_history_contract.json']
DOC_NAMES = ['EDA/README.md', 'Analysis/README.md', 'README.md', 'AGENT_MEMORY.md', 'AGENTS.md', 'Modeling/README.md',
             'Modeling/04_Modeling_원고.md', 'Modeling/10.04_M01_자료시차평가조건_검증기록.md',
             'Modeling/10.04_M02_기본모형비교_실행기록.md',
             'Modeling/10.04_M03_마지막값추가와시차입력_실행기록.md',
             'Modeling/10.04_M02_확장모델비교_실행기록.md',
             'Modeling/10.05_M04_조건별오류와낮은전력규칙_실행기록.md',
             'Modeling/10.05_M04_상승예측결합_실행기록.md',
             'Modeling/10.05_M04_증가량급등과위험출력_실행기록.md',
             'Modeling/10.05_M05_고정후반기평가_실행기록.md',
             'Modeling/10.05_R01_R04_차별성재개발_실행기록.md']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_imports(path, local_modules):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            name = node.module.split('.')[0]
            if name in local_modules:
                found.add(name)
        elif isinstance(node, ast.Import):
            found.update(n.name.split('.')[0] for n in node.names if n.name.split('.')[0] in local_modules)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == '__import__':
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                name = node.args[0].value.split('.')[0]
                if name in local_modules:
                    found.add(name)
    return sorted(found)


def source_and_result_checks():
    contract = json.loads((ROOT / 'Modeling/config/m01_contract.json').read_text(encoding='utf-8'))
    assert digest(ROOT / contract['source']) == contract['source_sha256']
    m1 = ROOT / 'Modeling/tables/m01'
    verify = json.loads((m1 / 'verification.json').read_text(encoding='utf-8'))
    assert verify['status'] == 'passed'
    for name, expected in verify['outputs_sha256'].items():
        assert digest(m1 / name) == expected, name
    for folder in ['m02', 'm03/ab', 'm03/c', 'm02_rerun']:
        directory = ROOT / 'Modeling/tables' / folder
        run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
        independent = json.loads((directory / 'independent_verification.json').read_text(encoding='utf-8'))
        assert run['status'] == 'completed' and independent['status'] == 'passed'
        if 'run_sha256' in independent:
            assert independent['run_sha256'] == digest(directory / 'run.json')
        for name, expected in run['outputs_sha256'].items():
            assert digest(directory / name) == expected, (folder, name)
        if 'configuration_sha256' in run:
            assert digest(directory / 'selected_configurations.json') == run['configuration_sha256']
        if 'selection_sha256' in run:
            assert digest(directory / 'selection.json') == run['selection_sha256']
        if 'script_sha256' in run:
            script = {'m02': 'm02_compare.py', 'm03/ab': 'm03_ablation.py',
                      'm03/c': 'm03_ablation.py', 'm02_rerun': 'm02_rerun.py'}[folder]
            assert digest(SCRIPTS / script) == run['script_sha256']
        import csv
        with (directory / 'model_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
            for record in csv.DictReader(stream):
                assert digest(ROOT / record['file']) == record['sha256'], record['file']
    m4 = ROOT / 'Modeling/tables/m04'
    run = json.loads((m4 / 'run.json').read_text(encoding='utf-8'))
    independent = json.loads((m4 / 'independent_verification.json').read_text(encoding='utf-8'))
    assert run['status'] == 'completed' and independent['status'] == 'passed'
    assert independent['run_sha256'] == digest(m4 / 'run.json')
    assert run['script_sha256'] == digest(SCRIPTS / 'm04_compare.py')
    assert independent['script_sha256'] == digest(SCRIPTS / 'm04_verify.py')
    assert run['contract_sha256'] == digest(ROOT / 'Modeling/config/m04_contract.json')
    for name, expected in run['outputs_sha256'].items():
        assert digest(m4 / name) == expected, ('m04', name)
    for name, expected in run['inputs_sha256'].items():
        assert digest(ROOT / name) == expected, ('m04', name)
    for folder, script, verifier, configuration in [
        ('m04_rise', 'm04_rise_compare.py', 'm04_rise_verify.py', 'm04_rise_contract.json'),
        ('m04_rise_gate', 'm04_rise_gate.py', 'm04_rise_gate_verify.py', 'm04_rise_gate_contract.json'),
        ('m04_surge', 'm04_surge_diagnose.py', 'm04_surge_verify.py', 'm04_surge_contract.json'),
        ('m04_surge_risk', 'm04_surge_risk.py', 'm04_surge_risk_verify.py', 'm04_surge_risk_contract.json'),
        ('m04_surge_calibration', 'm04_surge_calibration.py', 'm04_surge_calibration_verify.py', 'm04_surge_calibration_contract.json')]:
        directory = ROOT / 'Modeling/tables' / folder
        run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
        independent = json.loads((directory / 'independent_verification.json').read_text(encoding='utf-8'))
        assert run['status'] == 'completed' and independent['status'] == 'passed'
        assert independent['run_sha256'] == digest(directory / 'run.json')
        assert run['script_sha256'] == digest(SCRIPTS / script)
        assert independent['script_sha256'] == digest(SCRIPTS / verifier)
        assert run['contract_sha256'] == digest(ROOT / 'Modeling/config' / configuration)
        for name, expected in run['outputs_sha256'].items():
            assert digest(directory / name) == expected, (folder, name)
        for name, expected in run.get('inputs_sha256', {}).items():
            assert digest(ROOT / name) == expected, (folder, name)
        for name, expected in run.get('helper_sha256', {}).items():
            assert digest(SCRIPTS / name) == expected, (folder, name)
        if folder in ['m04_rise', 'm04_surge_risk']:
            import csv
            with (directory / 'model_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
                for record in csv.DictReader(stream):
                    assert digest(ROOT / record['file']) == record['sha256'], record['file']
        elif folder == 'm04_rise_gate':
            assert run['parent_run_sha256'] == digest(ROOT / 'Modeling/tables/m04_rise/run.json')
    m5 = ROOT / 'Modeling/tables/m05'
    run = json.loads((m5 / 'run.json').read_text(encoding='utf-8'))
    independent = json.loads((m5 / 'independent_verification.json').read_text(encoding='utf-8'))
    assert run['status'] == 'completed' and independent['status'] == 'passed'
    assert independent['run_sha256'] == digest(m5 / 'run.json')
    assert run['script_sha256'] == digest(SCRIPTS / 'm05_evaluate.py')
    assert independent['script_sha256'] == digest(SCRIPTS / 'm05_verify.py')
    assert run['contract_sha256'] == digest(ROOT / 'Modeling/config/m05_contract.json')
    assert independent['diagnostics_sha256'] == digest(m5 / 'confirmed_diagnostics.json')
    for name, expected in run['outputs_sha256'].items():
        assert digest(m5 / name) == expected, ('m05', name)
    for name, expected in run['inputs_sha256'].items():
        assert digest(ROOT / name) == expected, ('m05', name)
    for name, expected in run['helper_sha256'].items():
        assert digest(SCRIPTS / name) == expected, ('m05', name)
    import csv
    with (m5 / 'model_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        for record in csv.DictReader(stream):
            assert digest(ROOT / record['file']) == record['sha256'], record['file']
    for stage in ['r01','r02','r03','r03b','r04']:
        directory = ROOT / 'Modeling/tables/redevelopment' / stage
        run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
        verified = json.loads((directory / 'independent_verification.json').read_text(encoding='utf-8'))
        assert run['status'] in ['completed','skipped'] and verified['status'] == 'passed'
        assert verified['run_sha256'] == digest(directory / 'run.json')
        assert verified['script_sha256'] == digest(SCRIPTS / ('redevelopment_peak_verify.py' if stage in ['r03b','r04'] else 'redevelopment_verify.py'))
        for name, expected in run.get('outputs_sha256', {}).items():
            assert digest(directory / name) == expected, (stage,name)
        for name, expected in run.get('inputs_sha256', {}).items():
            assert digest(ROOT / name) == expected, (stage,name)
        if 'entrypoint_sha256' in run:
            assert run['entrypoint_sha256'] == digest(SCRIPTS / 'redevelopment_peak.py')
        if run['status'] == 'completed' and stage != 'r01':
            assert run['script_sha256'] == digest(SCRIPTS / 'redevelopment_compare.py')
            with (directory / 'model_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
                for record in csv.DictReader(stream):
                    assert digest(ROOT / record['file']) == record['sha256'], record['file']
    directory = ROOT / 'Modeling/tables/profile_history'
    run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    verified = json.loads((directory / 'independent_verification.json').read_text(encoding='utf-8'))
    assert run['status'] == 'completed' and verified['status'] == 'passed'
    assert verified['run_sha256'] == digest(directory / 'run.json')
    assert verified['verifier_sha256'] == digest(SCRIPTS / 'profile_history_verify.py')
    cpath = ROOT / 'Modeling/config/profile_history_contract.json'
    assert run['contract_sha256'] == digest(cpath)
    c = json.loads(cpath.read_text(encoding='utf-8'))
    for path, expected in {**c['inputs_sha256'], **run['outputs_sha256'], **verified['analysis_outputs_sha256']}.items():
        assert digest(ROOT / path) == expected, path
    with (directory / 'model_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        for record in csv.DictReader(stream):
            assert digest(ROOT / record['file']) == record['sha256'], record['file']
    p1 = ROOT / 'EDA/tables/p01_peak_shapes'
    c1 = json.loads((p1 / 'contract.json').read_text(encoding='utf-8'))
    r1 = json.loads((p1 / 'run.json').read_text(encoding='utf-8'))
    assert r1['contract_sha256'] == digest(p1 / 'contract.json')
    assert c1['script_sha256'] == digest(ROOT / 'EDA/scripts/p01_peak_shapes.py')
    assert c1['source_sha256'] == digest(ROOT / contract['source'])
    for name, expected in r1['outputs_sha256'].items():
        assert digest(p1 / name) == expected, name
    td = ROOT / 'Modeling/tables/training_design'
    for stage in ['diagnosis', 'weights', 'recent']:
        directory = td / stage
        run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
        verified = json.loads((directory / 'independent_verification.json').read_text(encoding='utf-8'))
        c = json.loads((directory / 'contract.json').read_text(encoding='utf-8'))
        assert run['status'] == 'completed' and verified['status'] == 'passed'
        assert verified['run_sha256'] == digest(directory / 'run.json')
        assert verified['verifier_sha256'] == digest(SCRIPTS / 'training_design_verify.py')
        assert run['contract_sha256'] == digest(directory / 'contract.json')
        for path, expected in c['inputs_sha256'].items():
            assert digest(ROOT / path) == expected, path
        for path, expected in run['outputs_sha256'].items():
            assert digest(directory / path) == expected, path
        if stage != 'diagnosis':
            with (directory / 'model_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
                for record in csv.DictReader(stream):
                    assert digest(ROOT / record['file']) == record['sha256'], record['file']
    directory = ROOT / 'Modeling/tables/error_warning'
    run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    verified = json.loads((directory / 'independent_verification.json').read_text(encoding='utf-8'))
    c = json.loads((directory / 'contract.json').read_text(encoding='utf-8'))
    diagnosis = json.loads((directory / 'diagnosis.json').read_text(encoding='utf-8'))
    dc = json.loads((directory / 'diagnosis_contract.json').read_text(encoding='utf-8'))
    assert run['status'] == diagnosis['status'] == 'completed' and verified['status'] == 'passed'
    assert verified['run_sha256'] == digest(directory / 'run.json')
    assert verified['diagnosis_sha256'] == c['diagnosis_sha256'] == digest(directory / 'diagnosis.json')
    assert verified['verifier_sha256'] == digest(SCRIPTS / 'error_warning_verify.py')
    assert run['contract_sha256'] == digest(directory / 'contract.json')
    assert diagnosis['contract_sha256'] == digest(directory / 'diagnosis_contract.json')
    assert c['entrypoint_sha256'] == digest(SCRIPTS / 'error_warning_fit.py')
    assert c['frame_sha256'] == digest(directory / 'risk_frame.csv')
    for path, expected in dc['inputs_sha256'].items():
        assert digest(ROOT / path) == expected, path
    for record in [diagnosis, run]:
        for path, expected in record['outputs_sha256'].items():
            assert digest(directory / path) == expected, path
    with (directory / 'model_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        for record in csv.DictReader(stream):
            assert digest(ROOT / record['file']) == record['sha256'], record['file']
    directory = ROOT / 'Modeling/tables/priority_review'
    run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    verified = json.loads((directory / 'independent_verification.json').read_text(encoding='utf-8'))
    c = json.loads((directory / 'contract.json').read_text(encoding='utf-8'))
    docs = json.loads((directory / 'document_verification.json').read_text(encoding='utf-8'))
    assert run['status'] == 'completed' and verified['status'] == docs['status'] == 'passed'
    assert run['contract_sha256'] == digest(directory / 'contract.json')
    assert verified['run_sha256'] == docs['run_sha256'] == digest(directory / 'run.json')
    assert verified['verifier_sha256'] == digest(SCRIPTS / 'priority_review_verify.py')
    assert c['entrypoint_sha256'] == digest(SCRIPTS / 'priority_review.py')
    assert docs['script_sha256'] == digest(SCRIPTS / 'priority_review_documents.py')
    assert docs['verifier_sha256'] == digest(directory / 'independent_verification.json')
    assert docs['original_manuscript_sha256'] == digest(directory / 'manuscript_before.md')
    for path, expected in {**c['inputs_sha256'], **docs['documents_sha256']}.items():
        assert digest(ROOT / path) == expected, path
    for path, expected in run['outputs_sha256'].items():
        assert digest(directory / path) == expected, path
    directory = ROOT / 'Modeling/tables/regime_diagnosis'
    run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    c = json.loads((directory / 'contract.json').read_text(encoding='utf-8'))
    verified = json.loads((directory / 'independent_verification.json').read_text(encoding='utf-8'))
    assert run['status'] == 'completed' and verified['status'] == 'passed'
    assert run['contract_sha256'] == digest(directory / 'contract.json')
    assert verified['run_sha256'] == digest(directory / 'run.json')
    assert verified['verifier_sha256'] == digest(SCRIPTS / 'regime_diagnose_verify.py')
    assert c['entrypoint_sha256'] == digest(SCRIPTS / 'regime_diagnose.py')
    assert c['source_sha256'] == digest(ROOT / contract['source'])
    assert c['probe_sha256'] == digest(directory / 'probe.json')
    for path, expected in {**run['outputs_sha256'], **verified['extra_outputs_sha256']}.items():
        assert digest(directory / path) == expected, path
    directory = ROOT / 'Modeling/tables/regime_forecast'
    run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    c = json.loads((directory / 'contract.json').read_text(encoding='utf-8'))
    verified = json.loads((directory / 'independent_verification.json').read_text(encoding='utf-8'))
    assert run['status'] == 'completed' and verified['status'] == 'passed'
    assert run['contract_sha256'] == digest(directory / 'contract.json')
    assert verified['run_sha256'] == digest(directory / 'run.json')
    assert verified['verifier_sha256'] == digest(SCRIPTS / 'regime_forecast_verify.py')
    assert c['entrypoint_sha256'] == run['script_sha256'] == digest(SCRIPTS / 'regime_forecast.py')
    for path, expected in {**c['inputs_sha256'], **run['inputs_sha256']}.items():
        assert digest(ROOT / path) == expected, path
    for path, expected in {**run['outputs_sha256'], **verified['extra_outputs_sha256']}.items():
        assert digest(directory / path) == expected, path
    import csv
    with (directory / 'model_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == run['new_fits'] == verified['models_reloaded'] == 16
    for record in records:
        assert digest(ROOT / record['file']) == record['sha256'], record['file']
    for folder, script, verifier in [('regime_integration', 'regime_integrate.py', 'regime_integrate_verify.py'),
                                      ('regime_routing', 'regime_route.py', 'regime_route_verify.py')]:
        directory = ROOT / 'Modeling/tables' / folder
        run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
        c = json.loads((directory / 'contract.json').read_text(encoding='utf-8'))
        verified = json.loads((directory / 'independent_verification.json').read_text(encoding='utf-8'))
        assert run['status'] == 'completed' and verified['status'] == 'passed'
        assert run['contract_sha256'] == digest(directory / 'contract.json')
        assert verified['run_sha256'] == digest(directory / 'run.json')
        assert verified['verifier_sha256'] == digest(SCRIPTS / verifier)
        assert c['entrypoint_sha256'] == digest(SCRIPTS / script)
        for path, expected in c['inputs_sha256'].items():
            assert digest(ROOT / path) == expected, path
        for path, expected in {**run['outputs_sha256'], **verified.get('extra_outputs_sha256', {})}.items():
            assert digest(directory / path) == expected, path
        if folder == 'regime_integration':
            with (directory / 'model_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
                records = list(csv.DictReader(stream))
            assert len(records) == run['new_regression_fits'] == verified['models_reloaded'] == 15
            for record in records:
                assert digest(ROOT / record['file']) == record['sha256'], record['file']
        else:
            assert run['new_fits'] == verified['new_fits'] == 0
    directory = ROOT / 'Modeling/tables/regime_followup'
    run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    c = json.loads((directory / 'contract.json').read_text(encoding='utf-8'))
    verified = json.loads((directory / 'independent_verification.json').read_text(encoding='utf-8'))
    assert run['status'] == 'completed' and verified['status'] == 'passed'
    assert run['contract_sha256'] == digest(directory / 'contract.json')
    assert c['entrypoint_sha256'] == run['script_sha256'] == digest(SCRIPTS / 'regime_followup.py')
    assert verified['run_sha256'] == digest(directory / 'run.json')
    assert verified['verifier_sha256'] == digest(SCRIPTS / 'regime_followup_verify.py')
    for path, expected in c['inputs_sha256'].items():
        assert digest(ROOT / path) == expected, path
    for path, expected in {**run['outputs_sha256'], **verified['extra_outputs_sha256']}.items():
        assert digest(directory / path) == expected, path
    with (directory / 'model_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == run['new_fits'] == verified['models_reloaded'] == 3
    for record in records:
        assert digest(ROOT / record['file']) == record['sha256'], record['file']
    directory = ROOT / 'Modeling/tables/regime_age_ablation'
    c = json.loads((directory / 'contract.json').read_text(encoding='utf-8'))
    run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    verified = json.loads((directory / 'independent_verification.json').read_text(encoding='utf-8'))
    assert run['status'] == 'completed' and verified['status'] == 'passed'
    assert c['entrypoint_sha256'] == run['script_sha256'] == digest(SCRIPTS / 'regime_age_ablation.py')
    assert run['contract_sha256'] == digest(directory / 'contract.json')
    assert verified['run_sha256'] == digest(directory / 'run.json')
    assert verified['verifier_sha256'] == digest(SCRIPTS / 'regime_age_ablation_verify.py')
    for path, expected in c['inputs_sha256'].items():
        assert digest(ROOT / path) == expected, path
    for path, expected in run['outputs_sha256'].items():
        assert digest(directory / path) == expected, path
    with (directory / 'model_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == run['new_fits'] == verified['models_reloaded'] == verified['independent_classifier_refits']
    assert run['new_regression_fits'] == verified['new_regression_fits'] == 0
    for record in records:
        assert digest(ROOT / record['file']) == record['sha256'], record['file']
    if (directory / 'calendar_control_run.json').exists():
        diagnostic = json.loads((directory / 'calendar_control_run.json').read_text(encoding='utf-8'))
        dc = json.loads((directory / 'calendar_control_contract.json').read_text(encoding='utf-8'))
        assert diagnostic['status'] == 'completed' and diagnostic['scalar_routing_check'] == 'passed'
        assert dc['entrypoint_sha256'] == diagnostic['script_sha256'] == digest(SCRIPTS / 'regime_age_calendar_control.py')
        assert diagnostic['contract_sha256'] == digest(directory / 'calendar_control_contract.json')
        for path, expected in {**dc['inputs_sha256'], **diagnostic['outputs_sha256']}.items():
            assert digest(directory / path) == expected, path
    directory = ROOT / 'Modeling/tables/regime_manuscript'
    docs = json.loads((directory / 'document_verification.json').read_text(encoding='utf-8'))
    assert docs['status'] == 'passed' and docs['historical_tables_preserved']
    assert docs['script_sha256'] == digest(SCRIPTS / 'regime_manuscript_documents.py')
    assert docs['original_manuscript_sha256'] == digest(directory / 'manuscript_before.md')
    for path, expected in {**docs['source_sha256'], **docs['documents_sha256']}.items():
        assert digest(ROOT / path) == expected, path
    directory = ROOT / 'Modeling/tables/modeling_close'
    closed = json.loads((directory / 'verification.json').read_text(encoding='utf-8'))
    assert closed['status'] == 'passed' and closed['visual_review'] == 'passed'
    assert closed['script_sha256'] == digest(SCRIPTS / 'modeling_close_verify.py')
    assert closed['run_sha256'] == digest(directory / 'run.json')
    assert closed['documents_sha256'] == digest(directory / 'documents.json')
    assert closed['figures_manifest_sha256'] == digest(directory / 'figures.json')
    for folder in ['modeling_close', 'modeling_replay']:
        d = ROOT / 'Modeling/tables' / folder
        r = json.loads((d / 'run.json').read_text(encoding='utf-8'))
        assert r['status'] == 'passed' and r['script_sha256'] == digest(SCRIPTS / 'modeling_close_analysis.py')
        assert r['contract_sha256'] == digest(d / 'contract.json')
        for path, expected in r['inputs_sha256'].items():
            assert digest(ROOT / path) == expected, path
        for path, expected in r['outputs_sha256'].items():
            assert digest(d / path) == expected, path
    for file, field in [('documents.json', 'outputs_sha256'), ('figures.json', 'figures_sha256')]:
        record = json.loads((directory / file).read_text(encoding='utf-8'))
        for path, expected in record[field].items():
            assert digest(ROOT / path) == expected, path
    return contract['source']


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--record', action='store_true')
    group.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    assert sys.version_info[:2] == (3, 13) and sys._is_gil_enabled()
    modules = {Path(name).stem for name in SCRIPT_NAMES}
    manuscript_path = ROOT / 'Modeling/04_Modeling_원고.md'
    manuscript = manuscript_path.read_text(encoding='utf-8')
    assert manuscript.count('## 4.5 제출 코드와 재현 파일의 보존') == 1
    for name in SCRIPT_NAMES:
        path = SCRIPTS / name
        assert path.is_file(), path
        reference_text = manuscript if name in MANUSCRIPT_SCRIPT_NAMES else (ROOT / 'Modeling/README.md').read_text(encoding='utf-8')
        assert f'(scripts/{name})' in reference_text, f'Code missing from manuscript or research README: {name}'
    all_local = {p.stem for p in SCRIPTS.glob('*.py')}
    dependencies = {name: local_imports(SCRIPTS / name, all_local) for name in SCRIPT_NAMES}
    for name, imports in dependencies.items():
        assert set(imports) <= modules, (name, imports)
    linked_files = set()
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', manuscript):
        if target.startswith(('http:', 'https:', '#', 'app:')):
            continue
        path = (manuscript_path.parent / target.split('#')[0].strip('<>')).resolve()
        if args.record and path in {MANIFEST.resolve(), (ROOT / 'Modeling/requirements.txt').resolve()}:
            continue
        assert path.exists(), target
        if path.is_file():
            linked_files.add(path)
    if args.verify:
        current = json.loads(MANIFEST.read_text(encoding='utf-8'))
        assert current['script_names'] == SCRIPT_NAMES
        assert current['local_import_dependencies'] == dependencies
        for record in current['files']:
            path = ROOT / record['path']
            assert path.is_file(), f'Missing preserved file: {record["path"]}'
            assert path.stat().st_size == record['bytes'] and digest(path) == record['sha256'], record['path']
        source_and_result_checks()
        print(json.dumps({'status': 'passed', 'mode': 'verify', 'scripts': len(SCRIPT_NAMES),
                          'preserved_files': len(current['files']), 'no_training_or_deletion': True}), flush=True)
        return
    source = source_and_result_checks()
    packages = {name: importlib.metadata.version(name) for name in PACKAGES}
    requirements = ROOT / 'Modeling/requirements.txt'
    requirements.write_text('# Direct libraries for regular CPython 3.13; verified installed versions.\n' +
                            '\n'.join(f'{name}=={version}' for name, version in packages.items()) + '\n', encoding='utf-8')
    keep = set(linked_files)
    keep.add(ROOT / 'EDA/scripts/p01_peak_shapes.py')
    for extra in ['EDA/tables/p01_peak_shapes', 'Analysis/tables/p02_profile_history']:
        keep.update(p for p in (ROOT / extra).rglob('*') if p.is_file())
    keep.update(SCRIPTS / name for name in SCRIPT_NAMES)
    keep.update(ROOT / 'Modeling/config' / name for name in CONFIG_NAMES)
    keep.update(ROOT / name for name in DOC_NAMES)
    keep.update([ROOT / source, requirements, ROOT / 'Modeling/tables/m022/environment.json'])
    for folder in TABLE_DIRS:
        directory = ROOT / 'Modeling/tables' / folder
        assert directory.is_dir(), folder
        keep.update(p for p in directory.rglob('*') if p.is_file())
    for folder in MODEL_DIRS:
        directory = ROOT / 'Modeling/models' / folder
        assert directory.is_dir(), folder
        keep.update(p for p in directory.rglob('*') if p.is_file())
    keep.update(p for p in (ROOT / 'Modeling/figures').rglob('*') if p.is_file())
    keep.discard(MANIFEST)
    records = []
    for path in sorted(keep):
        assert path.is_file(), path
        path = path.resolve()
        relative = path.relative_to(ROOT.resolve()).as_posix()
        records.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': digest(path)})
    manifest = {'scope': 'Current Modeling manuscript M01/M02/M03/expanded M02/M04 and its direct references; not a whole-project deletion authorization',
                'recorded_at': datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='minutes'),
                'self': 'Modeling/submission_manifest.json', 'script_names': SCRIPT_NAMES,
                'manuscript_script_names': MANUSCRIPT_SCRIPT_NAMES, 'research_script_names': RESEARCH_SCRIPT_NAMES,
                'external_research_scripts': ['EDA/scripts/p01_peak_shapes.py'],
                'local_import_dependencies': dependencies, 'runtime': {'executable': sys.executable, 'version': sys.version, 'packages': packages},
                'requirements': 'Modeling/requirements.txt', 'files': records,
                'not_final_submission': True, 'clean_environment_reproduction_executed': False,
                'no_training_or_deletion': True}
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'recorded', 'scripts': len(SCRIPT_NAMES), 'preserved_files': len(records),
                      'direct_packages': len(packages), 'no_training_or_deletion': True}), flush=True)


if __name__ == '__main__':
    main()
