from pathlib import Path
import ast,json,hashlib,datetime,re
import nbformat,pandas as pd,numpy as np
from PIL import Image,ImageDraw
root=Path(__file__).resolve().parents[1];package=root/'제출'
nb=nbformat.read(package/'통합_분석.ipynb',as_version=4)
code=[c for c in nb.cells if c.cell_type=='code']
assert len(code)==26 and len(nb.cells)==76
for i,c in enumerate(code):
    tree=ast.parse(c.source)
    assert c.execution_count==i+1 and c.outputs
    assert not any(o.output_type=='error' for o in c.outputs)
    if i:assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['exec','eval'] for n in ast.walk(tree))
    assert '__name__' not in c.source and 'submission_source' not in c.source
assert len(list(package.glob('*.ipynb')))==1
assert not list(package.rglob('*.py'))
raw=package/'data/raw/okm_augumented_2021.csv'
source_sha=hashlib.sha256(raw.read_bytes()).hexdigest()
assert source_sha==hashlib.sha256((root/'data/origin/okm_augumented_2021.csv').read_bytes()).hexdigest()
original=pd.read_csv(raw,encoding='utf-8-sig',float_precision='round_trip')
processed=pd.read_csv(package/'data/processed/정상시간_전처리자료.csv',encoding='utf-8-sig',float_precision='round_trip')
expected=original.loc[original['날짜'].between(20210101,20210831)&original['시간'].between(0,23)].sort_values(['날짜','시간']).reset_index(drop=True)
np.testing.assert_allclose(expected.to_numpy(float),processed[expected.columns].to_numpy(float),rtol=0,atol=0,equal_nan=True)
# Keep only columns and rows actually read by the final validation cell.
ref_class=package/'reference/고정_분류예측.csv';cls=pd.read_csv(ref_class,encoding='utf-8-sig',float_precision='round_trip')
cls=cls.loc[cls.method.eq('noage'),['timestamp','method','score','alarm']]
cls.to_csv(ref_class,index=False,encoding='utf-8-sig',float_format='%.17g')
ref_reg=package/'reference/고정_회귀예측.csv';reg=pd.read_csv(ref_reg,encoding='utf-8-sig',float_precision='round_trip')
reg=reg.loc[reg.variant.isin(['B0','B1_all','G1_noage']),['timestamp','variant','actual','prediction']]
reg.to_csv(ref_reg,index=False,encoding='utf-8-sig',float_format='%.17g')
pred=pd.read_csv(package/'테스트데이터_예측결과.csv',encoding='utf-8-sig')
assert len(pred)==1344
ref=reg.loc[reg.variant.eq('G1_noage')].set_index('timestamp').loc[pred.timestamp]
np.testing.assert_allclose(pred.actual_maximum,ref.actual,rtol=0,atol=0)
np.testing.assert_allclose(pred.final_prediction,ref.prediction,rtol=1e-9,atol=1e-8)
summary=json.loads((root/'tmp/readable_notebook_execution.json').read_text(encoding='utf-8'))
summary.update({'raw_sha256':source_sha,'notebook_sha256':hashlib.sha256((package/'통합_분석.ipynb').read_bytes()).hexdigest(),
 'raw_rows':len(original),'processed_rows':len(processed),'classification_predictions':len(cls),'regression_predictions':len(reg),'final_prediction_rows':len(pred),
 'separate_python_source_files':0,'imports_only_first_code_cell':True,'raw_values_preserved':True,
 'historical_optuna_search_rerun':False,'scope':'기존 CPython 3.13 환경의 전체 실행·원자료·예측 대조. 신규 가상환경 설치 검증은 미수행.'})
(package/'output/실행검증.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
# Read newest memory and update only this conversation's keyed entries.
memory=root/'AGENT_MEMORY.md';m=memory.read_text(encoding='utf-8');now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec='minutes')
def replace_entry(text,key,new):
    return re.sub(r'(?ms)^- `'+re.escape(key)+r'`\n.*?(?=^- `|^## |\Z)',new+'\n\n',text,count=1)
m=re.sub(r'(?m)^- Last updated: .*', '- Last updated: '+now,m,count=1)
m=replace_entry(m,'decision:submission-notebook-from-raw',f'''- `decision:submission-notebook-from-raw`
  - Created: 2026-10-06T10:02+09:00
  - Updated: {now}
  - Status: active
  - Content: 사용자 예시 기준으로 제출 노트북을 보고서 목차·계산·실제 표/그림 출력·결과 해석으로 재구성했다. 76셀 중 계산26셀·마크다운50셀, 셀 수 상한은 없음. import는 첫 계산 셀에만 두고 main/exec/모듈 등록·별도 제출 py를 제거했다. 원본 raw6168행을 제출 data/raw에서 읽어 정상5784행 전처리 후 processed_data를 EDA/Analysis/Modeling이 공유한다. 개발 고정 설정의 모델 비교와 A/B 입력 효과·현재 Logistic/HGB를 원자료 재학습한다. 과거 CSV 숫자 읽기 경계를 보존해 원고 모델 수치 재현, Optuna 전체탐색은 미수행. 정규Python3.13 전체26셀 종료0·그림15개·분류1428/회귀4032 예측 대조 통과. 원본과 예시·기존 제출 사본은 tmp/submission_before_readable_rebuild에 보존, 제출에는 노트북1개·raw·requirements·README·대조CSV2개·예측CSV·생성 결과만 둔다. 새 가상환경 설치는 미검증, 커널 등록·선택 및 nbconvert 전체실행 명령은 README에 기록.
  - Evidence: 제출/통합_분석.ipynb; 제출/data/raw/okm_augumented_2021.csv; 제출/README.md; 제출/output/실행검증.json; tmp/readable_notebook_execution.json.''')
m=replace_entry(m,'session:20261006-1002',f'''- `session:20261006-1002`
  - Started: 2026-10-06T10:02+09:00
  - Last activity: {now}
  - Focus: 사용자 예시에 따른 제출용 노트북 재구성·raw 전처리·실제 셀 출력·원고 결과 재현.
  - Updated keys: decision:submission-notebook-from-raw
  - Summary: 기존97스크립트 붙여넣기 방식은 사용자 요구에 맞지 않아 폐기하고 원고 순서의 직접 계산26셀·해석50셀로 재작성했다. 모든 단계가 공통 전처리 자료를 쓰고 py 없이 실행하며 전체실행 종료0·15그림·분류1428/회귀4032 대조 통과. 개발 CSV 정밀도 차이를 수정해 HGB6.738 재현, A/B 입력 효과와 38→0 오경보 비교 추가. 예시·이전 제출본 보존, 신규 환경 설치·전체Optuna·외부 제출은 미수행. 원고·동시 PPT 작업 미변경.''')
m=m.replace('Jupyter Notebook은 제출 폴더에서 raw→전처리→분석·모델 재현 전체 실행 완료.','제출 노트북은 사용자 예시에 맞춰 직접 계산26셀·실제 출력·해석으로 재작성했고 raw→전처리→분석·모델 전체 실행을 검증했다.')
memory.write_text(m,encoding='utf-8')
# Scientific-image QA: 50 percent in each dimension, at most four per sheet.
qa=root/'tmp/readable_visual_review';files=sorted((package/'output').glob('*.png'))
for start in range(0,len(files),4):
    thumbs=[]
    for p in files[start:start+4]:
        im=Image.open(p).convert('RGB');thumbs.append((p.stem,im.resize((im.width//2,im.height//2))))
    w=max(im.width for _,im in thumbs);h=max(im.height for _,im in thumbs)+25
    sheet=Image.new('RGB',(2*w,2*h),'white');draw=ImageDraw.Draw(sheet)
    for i,(label,im) in enumerate(thumbs):
        x=(i%2)*w;y=(i//2)*h;draw.text((x+4,y+4),label,fill='black');sheet.paste(im,(x,y+25))
    sheet.save(qa/f'sheet_{start//4+1}.png')
print(json.dumps(summary,ensure_ascii=False))