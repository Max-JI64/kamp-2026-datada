/* Build the review deck from existing manuscripts. No model/data recalculation. */
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const ROOT = path.resolve(__dirname, '..');
const OUT = __dirname;
const BASE = fs.readFileSync('C:/Users/swoo6/.codex/skills/frontend-slides/viewport-base.css', 'utf8');
const sources = {
  data:'EDA/01_데이터_0.5페이지용_압축본.md', eda:'EDA/02_EDA_3페이지용_압축본.md',
  analysis:'Analysis/03_Analysis_3페이지용_압축본.md', model:'Modeling/04_Modeling_3페이지용_압축본.md',
  report:'Report/03_05_보고서_원고.md', task:'대회개요/과제공개.md',
  form:'경진대회 결과보고서 양식_일반국민,대학(원)생 부문.hwpx'
};
const assets = new Map();
const esc = s => String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
function figure(file, caption, extra='') {
  if (!fs.existsSync(path.join(ROOT,file))) throw new Error('Missing '+file);
  const name = file.replaceAll('/','__');
  fs.mkdirSync(path.join(OUT,'assets'),{recursive:true});
  fs.copyFileSync(path.join(ROOT,file),path.join(OUT,'assets',name));
  assets.set(file,name);
  return `<figure class="figure ${extra}"><img src="data:image/png;base64,${fs.readFileSync(path.join(ROOT,file)).toString('base64')}" alt="${esc(caption)}"><figcaption>${caption}</figcaption></figure>`;
}
const table = (headers,rows) => `<table><thead><tr>${headers.map(x=>`<th>${x}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(x=>`<td>${x}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
const point = (title,body) => `<div class="point"><h3 data-edit>${title}</h3><p data-edit>${body}</p></div>`;
const stat = (value,label) => `<div class="stat"><strong data-edit>${value}</strong><span data-edit>${label}</span></div>`;
const call = text => `<div class="callout" data-edit>${text}</div>`;
const cols = (a,b,cls='') => `<div class="columns ${cls}"><div>${a}</div><div>${b}</div></div>`;
const steps = items => `<div class="steps">${items.map((x,i)=>`<div><span class="step-no">${String(i+1).padStart(2,'0')}</span><h3 data-edit>${x[0]}</h3><p data-edit>${x[1]}</p></div>`).join('')}</div>`;
const slides=[];
function add(section,title,body,source,note,cls='',foot='') { slides.push({section,title,body,source,note,cls,foot}); }

/* === OPENING: DATASET CHOICE, BACKGROUND, GOALS === */
add('분석 배경 및 목표','제조 전력의 다음 한 시간,<br>최대값과 지속 상승을 예측하다',
 `<p class="cover-sub" data-edit>생산·전력 기록으로 부하를 예측하고<br>오류가 집중되는 조건을 현장 판단에 연결</p><div class="cover-bottom"><span>선택 과제 ⑤</span><span>자원 최적화 AI 데이터셋</span></div><div class="cover-mark" aria-hidden="true"><span>01</span><div></div><span>NEXT<br>HOUR</span></div>`,
 'task,data,model','표지는 과제명보다 우리가 구현한 예측 대상을 앞에 둔다. 팀명은 사용자가 실제 PPT에 넣는다. 오른쪽 도형은 장식이며 관측 전력 그래프가 아니다.','cover');
add('분석 배경 및 목표','5개 제조 과제 중<br>전력 수요를 다루는 과제를 선택했다',
 cols(`<div class="dataset-list">${[['①','사출성형기','품질불량 사전예측'],['②','용접기','용접불량 예측'],['③','소성가공 예지보전','유압펌프 이상 조기탐지'],['④','X-ray 검사장비','이물질 탐지'],['⑤','자원 최적화','전력사용량 예측·최대피크 위험조건']].map((r,i)=>`<div class="dataset-row ${i===4?'selected':''}"><span>${r[0]}</span><b data-edit>${r[1]}</b><small data-edit>${r[2]}</small></div>`).join('')}</div>`,
 `<div class="selection"><span class="eyebrow">선택 과제 ⑤</span><h3 class="large" data-edit>부하의 크기와<br>변화 시점을<br>함께 예측</h3><p data-edit>시간별 전력에 생산량·기상·달력이 연결돼 있어, 전력 변화의 조건과 예측 실패를 함께 분석할 수 있다.</p></div>`),
 'task,data','다른 데이터가 더 나쁘다고 주장하지 않는다. 이 선택 이유는 현재 분석 목적과 제공 변수에 근거한 발표용 설명이며 과거의 팀 의사결정 경위를 확인한 기록은 아니다.');
add('분석 배경 및 목표','생산량만으로는<br>전력 부하를 설명하기 어렵다',
 cols(`<p class="lead" data-edit>최대수요전력이 예상보다 높아지면<br>전력비용과 운영 부담이 커질 수 있다.</p>${point('과제에서 제시한 문제','동시 가동, 생산량 변화, 제품 전환, 교대·계절 조건에 따라 전력 수요가 달라진다.')}${point('이번 자료에서 확인한 문제','생산량이 0이어도 전력이 관측되고, 시간 평균 안에서도 15분 구간값이 다르다.')}`,
 `<div class="big-question"><span>분석 질문</span><h3 data-edit>다음 시간의<br>최대 전력과<br>상승 시작을<br>미리 구분할 수 있는가?</h3></div>`),
 'task,eda,model','앞 문단은 대회 출제 배경, 뒤 문단은 EDA 관측이다. 특정 설비의 동시 가동이나 교대가 실제 기록의 원인이라고 주장하지 않는다.');
add('분석 배경 및 목표','예측값·오류 조건·활용 방안을<br>하나의 분석으로 연결한다',
 steps([['최대 전력 예측','과거 관측으로 다음 한 시간의 네 15분 값 중 최대값을 예측한다.'],['위험·오류 조건 구분','지속 상승 시작, 실제 일별 최대 시간, 미탐지·오경보를 별도로 확인한다.'],['현장 판단에 연결','예측과 경고를 보고, 담당자가 가동 시작·동시 사용 조건을 점검한다.']])+
 call('확인할 성과는 예측오차와 경고의 개선이다. 가동 조정과 비용 절감은 활용 제안으로 구분한다.'),
 'model,report','발표의 세 목표. 최대값이라는 연속값과 지속 상승이라는 사건을 구분한다. 경고가 요금 기준이나 설비 허용 전력 초과를 뜻하지 않는다는 설명은 후반 활용 슬라이드로 이어진다.','', '발표 흐름: 데이터 진단 → EDA → 입력 근거 → 모델 비교 → 오류·활용');

/* === DATA AND DIAGNOSIS === */
add('데이터 이해 및 진단','한 행은 한 시간,<br>한 시간 안에는 네 전력값이 있다',
 `<div class="stats">${stat('6,168 × 18','원본 행 × 열')}${stat('5,784시간','1~8월 정상 시간')}${stat('241일','정상 하루 기록')}</div>`+
 cols(table(['변수 묶음','관측 정보'],[['15·30·45·60분','시간 안의 네 전력값'],['평균·생산량','시간별 전력 수준·생산 규모'],['기상·달력','외부 환경·월·요일·시간대']]),
 `${point('자료의 범위','2021.01.01~09.14 원본 중 이번 분석은 1~8월을 사용한다.')}${point('해석의 범위','공장 변압기 전력 기록이다. 개별 설비 상태·생산량 배정 규칙·전력 원단위는 현 자료에서 미식별이다.')}`),
 'data','정상 5,784시간은 EDA 표본이다. 모델별 시차와 평가 기간 때문에 모델링의 표본 수는 달라진다. 단위는 기록값이며 kW/kWh로 쓰지 않는다.');
add('데이터 이해 및 진단','시간 오류는 제외하고,<br>큰 값과 반복 기록은 보존했다',
 table(['점검','처리','판단 근거'],[['시간 오류 48행','제외','7/13·7/15의 시간값 70~188. 시각 임의 복원 없음'],['전력·생산량 결측','없음','기상 결측은 해당 변수 계산에서만 제외'],['공장인원','독립 입력에서 제외','생산량 ÷ 네 전력값의 합과 일치'],['전력 0·큰 값','유지','크기만으로 오류라고 판단하지 않음'],['하루 전력 배열 반복','유지','241일에 126종. 날짜가 다른 관측 기록']])+
 call('예측 입력은 직전까지의 관측과 미리 알 수 있는 달력으로 제한한다.'),
 'data','기상 결측은 풍속 3셀·강수량 1셀. 공장인원 결측 17셀 때문에 행을 삭제하지 않았다. 목표 시간의 실제 생산량·날씨·전력을 입력으로 넣지 않는다.');

/* === EDA: OBSERVATION TO QUESTIONS === */
add('EDA · 하루 패턴','생산량의 정점과<br>전력의 정점은 달랐다',
 cols(figure('EDA/figures/report_compact/daily_pattern_distribution.png','전력 음영: 날짜별 10~90백분위 범위. 신뢰구간이 아님.'),
 `${stat('19시 / 08시','평일 생산량 / 평균 전력의 정점')}${point('12시 하락·13시 상승','평일 147/171일, 86.0%에서 관측됐다.')}${point('주말 생산량 0도 전력 0은 아님','주말 19시 생산량은 모두 0이지만 평균 전력은 41.11이었다.')}`,'figure-split'),
 'eda','같은 시각이라도 날짜별 전력값이 달랐다. 평균 곡선은 개별 날짜 예측을 대신하지 못한다. 원본 그래프의 평일·주말 축과 음영 정의를 유지한다.');
add('EDA · 월별 구성','월별 전력 차이에는<br>낮은 전력 날짜의 구성도 섞인다',
 cols(figure('EDA/figures/daily_repetition/07_monthly_power_mean_heatmap.png','월별·시간대별 평균 전력. 두 패널은 같은 0~210 색 척도.'),
 `${stat('185.05 → 142.68','평일 08시: 7월 → 8월')}${point('8월 평일 22일 중 5일','하루 생산량이 모두 0이었다.')}${point('양수 생산량이 있던 17일','08시 평균 전력은 178.47이었다.')}`,'figure-split'),
 'eda','월 차이를 계절이나 생산량의 인과효과로 단정하지 않는다. 전체 8월 평균과 생산량 양수 날짜의 8월 평균은 표본이 다르다.');
add('EDA · 낮은 전력','평일에도 낮은 전력이<br>별도의 구간으로 모였다',
 cols(figure('EDA/figures/weekday_power_levels.png','평일 4,104시간의 CSV 평균 전력 분포. 막대 폭 5.'),
 `${stat('747시간','20~26 구간 · 평일의 18.20%')}${point('27~48 기록은 없음','분포의 빈 구간을 전력값별 빈도표로 확인했다.')}${point('낮은 전력 중 생산량 양수 92시간','낮은 전력·생산량 0·휴무를 같은 상태로 묶지 않았다.')}`,'figure-split')+
 call('전체 기간의 20~26 연속 구간은 71개, 지속시간 중앙값은 24시간이었다.'),
 'eda','71개는 전수 기록상의 연속 구간이며 실제 정지 횟수가 아니다. 최장 223시간과 전력 0 단일 17시간은 별도 상태다. 네 날짜 유형의 전체 표는 부록 25에 있다.');
add('EDA · 변수 관계','전체 상관만으로<br>조건별 관계를 결론 내리지 않았다',
 cols(figure('EDA/figures/restart_02_variable_relations.png','정상 5,784시간의 Spearman 순위상관. 결측은 변수 쌍별 제외.'),
 `${stat('0.752','생산량·평균 전력 상관')}${point('날씨·전력 전체 상관은 작음','기온 0.058 · 풍속 0.137<br>습도 −0.119 · 강수량 −0.014')}${point('8월 기온 관계는 생산 기록별로 다름','8월 전체 −0.076<br>생산량 양수 383시간 +0.599')}`,'figure-split'),
 'eda','전체 상관은 인과·예측 성능이 아니다. 표본 구성이 관계를 바꿨으므로 Analysis에서 같은 조건을 고려한 관계를 확인한다.');
add('EDA · 시간 내부 구조','시간 평균은<br>네 15분 값의 차이를 모두 담지 못한다',
 cols(figure('EDA/figures/slot_time_patterns.png','각 칸은 같은 시간의 반올림 전 평균 대비 차이. 전력 수준·백분율이 아님.'),
 `${point('평일 07시','60분 값 > 15분 값: 160/171일, 93.6%')}${point('평일 10시','첫 구간이 나머지 세 값보다 낮음: 145/171일, 84.8%')}${point('평일 12시·17시','가운데 두 구간이 낮음: 75.4%<br>뒤 두 구간이 높음: 84.8%')}`,'figure-split'),
 'eda','부등호는 엄격한 조건이다. 07시가 계속 단조 증가한다는 뜻은 아니다. 주말은 07시 37.1%, 10시 18.6%, 12시 14.3%, 17시 15.7%. 반복 기록을 독립 실험으로 해석하지 않는다.');

/* === ANALYSIS: WHY THESE INPUTS AND CONDITIONS === */
add('Analysis · 생산량 전환','생산량 전환의 큰 변화와<br>전력 예측 성과는 구분했다',
 figure('Analysis/figures/a01_a02_transition_summary.png','왼쪽: 공통 46조건·717시간의 전력 변화. 오른쪽: 7~8월 생산량 0인 617시간의 전환 분류.','wide')+
 `<div class="two-findings"><p data-edit><b>관측 비교</b> 0→양수 전환의 변화 크기 37.92<br>양수 지속 30.21 대비 25.5% 큼</p><p data-edit><b>입력 탐색</b> 달력에 직전 평균 전력 추가<br>적중 35→50건 · 오탐 51→25건</p></div>`,
 'analysis','왼쪽은 월·시간대·평일/주말 공통 조건을 같은 가중치로 비교한다. 표본은 전환150·양수지속567시간이며 전체 전환215시간 중69.8% 포함. 오른쪽 실제 전환52건, 재현율67.3→96.2%, 정밀도40.7→66.7%, AP0.377→0.902. 생산량 전환 분류는 입력 가치의 탐색이며 전력 회귀 성능이 아니다.');
add('Analysis · 조건을 구분','전력 상승은 생산 전환과 다르고,<br>날씨 관계도 조건에 따라 약해졌다',
 cols(`${stat('5 / 64시간','낮은 전력에서 상승할 때 생산량도 0→양수')}${point('생산량 전환만으로 상승을 설명할 수 없음','전력 상승·하락과 구간 안의 변동을 함께 평가한다.')}${point('큰 변화가 높은 최대값을 뜻하지 않음','생산 전환 / 양수 지속의 최대값: 135.73 / 135.32')}`,
 `<h3 data-edit>평일 기온·전력의 관계</h3>${table(['비교','순위 관계'],[['단순 Spearman','+0.227'],['월·시간·생산·다른 날씨 고려','−0.004']])}${point('예측 입력의 우선순위','과거 전력을 우선 비교한다. 날씨의 추가 가치는 과거 기상정보를 더했을 때의 예측오차로 판단한다.')}`)+
 call('26 초과는 낮은 구간을 벗어난 상태다. 피크·이상 기준으로 쓰지 않는다.'),
 'analysis','상승64시간은 CSV 평균20~26에서26초과 전환. 최대전력 변화 중앙값 상승+72·하락−68. 높은상태유지3,757시간은 전체 최대전력 절대변화 합계81.1%. 기온표는 평일4,102시간이며 슬라이드10 전체표본과 다르다. 비선형 날씨 예측 가치를 배제한 결론이 아니다.');
add('Analysis · 입력 선택','마지막 15분 값에<br>다음 최대값의 정보가 남았다',
 cols(figure('Analysis/figures/a04_terminal_slots.png','정확히 연속된 세 시간 기록 5,778시간. 조건을 고려한 부분 순위 관계.'),
 `${stat('+0.434','달력·직전 생산량·평균 고려 후')}${stat('+0.409','직전 최대값도 고려 후')}${point('모델에서 직접 비교할 질문','평균·최대에 마지막 15분 값을 더하면 실제 예측오차도 줄어드는가?')}`,'figure-split')+
 call('관계 수치는 정확도나 오차 개선률이 아니다. 엄격한 동일 조건 비교에서는 초기 기간 관계가 거의 없었다.'),
 'analysis','직전최대 고려후 기간별+0.394/+0.430/+0.390, 반복배열 비중을 낮추면+0.374. 엄격한 과거평균·달력·생산량0 맞춤 표본은1,039시간,71.5%낮은전력;1~4월−0.009. 모든조건 일반화 금지.');

/* === MODELING: TARGET, EVALUATION AND RESULTS === */
add('모델 개발 · 시간순 검증','관측을 마친 뒤,<br>다음 한 시간의 최대값을 예측한다',
 `<div class="timeline"><div><small>관측 완료</small><strong>직전 시간</strong><p data-edit>전력 네 값·생산량<br>과거 시차·상태 정보</p></div><span class="arrow">→</span><div class="target"><small>예측 대상</small><strong>다음 한 시간</strong><p data-edit>15·30·45·60분 값<br>중 최대값</p></div></div>`+
 table(['개발 평가','학습','평가 월'],[['1차','1~3월','4월'],['2차','1~4월','5월'],['3차','1~5월','6월']])+
 call('Optuna 튜닝도 평가 월보다 앞선 내부 시간순 검증에서 수행했다. 목표 시간의 실측 생산량·날씨는 입력에서 제외했다.'),
 'model','무작위 분할 대신 시간순 확장. 시차는 정확히 연속된 시각만 연결하며 공백을 건너 연결하지 않는다. 미래3시간은 지속상승 정답확인에만 사용한다. 개발평가와 후반기 탐색적재평가를 뒤 슬라이드에서 구분한다.');
add('모델 개발 · 기본 모델','전체·최대 시간 오차를<br>함께 고려해 HGB를 유지했다',
 table(['대표 모델','전체 MAE ↓','일별 최대 시간 MAE ↓'],[['직전 값 기준','13.793','36.589'],['<b>HGB</b>','<b>6.738</b>','11.751'],['ExtraTrees','6.825','<b>11.540</b>'],['XGBoost','7.131','12.864']])+
 cols(point('같은 입력·같은 2,184시간','Ridge·Elastic Net·SVR, HGB·XGBoost·LightGBM·CatBoost·ExtraTrees와 앙상블을 비교했다.'),point('선택의 근거','ExtraTrees는 최대 시간 오차가 작지만, 전체·최대 오차를 함께 보는 사전 규칙에서는 HGB를 유지했다.'))+
 call('마지막 15분 값의 개발 이점은 후반기 최대 시간까지 유지되지 않았다. 상승 시작을 별도 문제로 다룬다.'),
 'model','표는 기본입력의개발비교다. 뒤의B0/G1 후반기수치와 표본·입력이 달라 같은표로 순위를 비교하지 않는다. HGB는Histogram Gradient Boosting. MAE는평균절대오차.');
add('모델 개발 · 지속 상승','한 번의 급증과<br>높은 수준의 지속을 구분했다',
 steps([['시작 후보','시간 산술평균이 직전값과 과거 6시간 중앙값보다 각각 59.5 이상 높다.'],['변동 규모 확인','중앙값 대비 상승폭이 과거 IQR의 1.5배 이상이다.'],['지속 정답 확인','이후 3시간 동안 기준선보다 29.75 이상 높은 상태가 이어진다.']])+
 call('기준은 1월 자료에서 정했다. 미래 3시간은 정답 확인용이며, 경고 모델은 과거 정보만 사용한다.')+
 `<p class="lead lower" data-edit>최대 전력은 ‘얼마나 높은가’,<br>지속 상승 경고는 ‘높은 상태로 바뀌기 시작하는가’를 다룬다.</p>`,
 'model','59.5·29.75는전력기록값단위. IQR은사분위범위. 업무/요금/설비허용경계가 아니라 분석적 사건정의다. 모든급등이나 양방향전환 전체의 정답으로 확대하지 않는다.');
add('모델 개발 · 선택적 보완','상승을 경고한 시간에만<br>기본 예측을 보완했다',
 `<div class="routing"><div class="routing-input"><small>과거 전력·달력·상태</small><h3 data-edit>Logistic<br>지속 상승 판정</h3></div><div class="routing-branches"><div><span>경고 있음</span><h3 data-edit>보완 HGB · B1</h3><p data-edit>상태 입력을 보완한 최대값 예측</p></div><div><span>경고 없음</span><h3 data-edit>기본 HGB · B0</h3><p data-edit>기존 최대값 예측 유지</p></div></div></div>`+
 cols(point('초기 오경보의 조건','38건 모두 상태 경과시간 180~227시간.<br>학습에서 본 최대값은 64시간이었다.'),point('한 입력만 제거','상승 분류기에서 경과시간을 제거했다.<br>과거 OOF 경계 0.4555 · 개발 적중 12건/오경보 0건 유지.')),
 'model','경과시간은분류기에서만제거했고회귀B1의age는유지한다. OOF는해당평가월이전학습으로생성된예측. 이전OOF로경계를정했다. 후반기를 이미 보고 수정했으므로 탐색적이라는 경계는19~21장에 유지한다.');
add('성능 · 후반기 탐색적 재평가','보완 효과는<br>상승 시작과 최대 시간에 집중됐다',
 `<div class="period-label">7~8월 탐색적 재평가 · 같은 1,344시간 · 완전한 56일</div>`+
 table(['평가 조건','기본 B0','선택 보완 G1','오차 감소'],[['전체 MAE','6.335','6.317','0.28%'],['<b>상승 시작 MAE</b>','20.037','18.205','<b>9.1%</b>'],['<b>상승 시작 평균 과소예측</b>','19.419','16.342','<b>15.8%</b>'],['일별 최대 시간 MAE','13.179','12.574','4.6%']])+
 `<div class="stats compact-stats">${stat('7시간','1,344시간 중 보완 적용')}${stat('38 → 0건','오경보 감소 · 적중 7건 유지')}${stat('1,337시간','기본 예측 유지')}</div>`,
 'model,report','회귀1,344시간과경고정답확인1,340시간을구분한다. 분류1~6월4,336시간·회귀2~6월3,598시간학습후고정;7월정답재학습없음. 다만7~8월기존결과를보고입력수정했으므로 독립최종시험 아님. 감소율은원표정밀값 기준이라 표시반올림값직접계산과차이날수있다.','', '후반기 결과를 보고 수정했으므로, 독립 최종시험이 아닌 탐색적 결과');
add('오류 분석 · 미탐지와 오경보','정밀도 100%에도<br>지속 상승 6건을 놓쳤다',
 `<div class="stats">${stat('7 / 13건','지속 상승 탐지 · 재현율 53.8%')}${stat('7 / 7건','경고 적중 · 정밀도 100%')}${stat('4 / 7건','경고 시간 중 값의 오차 감소')}</div>`+
 cols(`${point('경고는 모두 월요일 08시','과거 전력의 판단 기여도 확인했지만 특정 주간 조건에 집중됐다.')}${point('미탐지 6건은 남음','경고가 없다는 이유로 높은 전력 가능성을 배제할 수 없다.')}`,
 `${point('사후 단순 주간 규칙과 비교','단순 규칙은 적중 8건·오경보 1건.<br>일별 최대 오차는 같아 추가 이득은 작았다.')}${point('변수 기여는 모델 판단의 설명','적중/미탐지의 과거 전력 평균 로짓 기여: 13.86 / 4.16. 공정 원인을 규명한 결과는 아니다.')}`),
 'model,report','1,340시간에서공통상승13건. 오경보0건은이기간·정의에서의관측이며일반무오경보보장이아님. 표준화입력×계수로로짓분해. 순수독립시험이나주간규칙일반우위주장없음.','', '7~8월 탐색적 재평가 · 분류 정답 확인 1,340시간');
add('오류 분석 · 사례','개선 사례와 미탐지 사례를<br>함께 확인했다',
 figure('Modeling/figures/modeling_close/transition_cases.png','음영: 지속 상승 정답 3시간. 점선: 직전 관측 완료 시점. 각 집단의 중앙 순위 사례.','wide')+
 `<div class="two-findings"><p data-edit><b>7/12 개선</b> 실제 최대 198<br>기본 181.13 → 보완 192.52</p><p data-edit><b>8/13 미탐지</b> 실제 최대 207<br>경고 없이 182.40으로 과소예측</p></div>`,
 'model,report','개선4건은개선폭,미탐지6건은실제최대값을정렬한중앙순위사례. 가장좋은사례만뽑지않았다. 그래프선은기존검증된결과이며새분석없음.','', '7~8월 탐색적 재평가 · 사례와 전체 집계의 역할을 구분');

/* === USE, REPRODUCTION AND CONCLUSION === */
add('현장 활용방안','예측과 경고를<br>가동 시점 점검에 연결한다',
 steps([['직전 관측 완료','최근 전력·생산량·달력을 갱신하고 다음 시간 최대값과 상승 경고를 산출한다.'],['담당자 확인','예측 대상 시간·최근 전력·예측 최대값·경고·기본/보완 적용 여부를 확인한다.'],['조정 가능성 검토','예정된 가동이 겹치는지 확인하고 생산 일정·공정 제약 안에서 시작 시점 분산을 검토한다.']])+
 table(['상황','판단·조치'],[['경고 있음','보완 예측과 예정 가동을 함께 확인'],['경고 없음','기본 예측 계속 확인. 무경고를 안전 판정으로 쓰지 않음'],['필수 시차 누락','예측을 판단 근거로 사용하지 않고 자료 상태 표시']]),
 'model,report','활용절차는제안이다. 예정가동은담당자정보이며모델입력아님. 특정설비정지·생산량축소자동지시없음. 매시간갱신예측을하루사전일정선택으로쓰지않는다. 실제현장적용·자동제어·절감량미검증.','', '전력·비용 절감은 실측 성과가 아닌 활용 제안');
add('코드 및 재현성','현재 고정 방법을<br>원자료부터 다시 실행해 대조했다',
 steps([['원자료·시차 구성','기간·시간 오류 필터를 적용하고 관측 완료 시점에 맞춰 입력을 만든다.'],['모델 학습·추론','현재 선택한 Logistic과 B0/B1 HGB를 학습하고 고정 후반기 예측을 생성한다.'],['결과 일치 확인','분류 1,428개·회귀 4,032개 예측을 기존 결과와 대조했다.']])+
 call('정규 CPython 3.13에서 현재 재현 스크립트 실행을 확인했다.')+
 `<p class="lead lower" data-edit>새 환경 설치와 과거 Optuna 전체 재탐색은 검증 범위에 포함하지 않았다.</p>`,
 'model','회귀4,032개는한모델의독립시간수가아닌복수예측결과합계. 제출보고서제6장은노트북정리이후작성하도록보류한기존결정을유지하고,이슬라이드는Modeling압축본의이미확인된재현범위만요약한다. 발표자노트에실행파일과새출력폴더명령을연결한다.');
add('결론 및 차별성','최대값 예측에 지속 상승 판정을 더해,<br>일부 상승의 과소예측을 완화했다',
 `<div class="closing-thesis"><span>이번 분석의 기여</span><h3 data-edit>기본 예측 유지<br><em>필요한 시간만 보완</em><br>실패 조건을 입력 설계에 반영</h3></div>`+
 `<div class="closing-metrics">${stat('15.8%','상승 시작 평균 과소예측 감소')}${stat('38 → 0건','탐색적 재평가의 오경보')}${stat('6건','남은 지속 상승 미탐지')}</div>`+
 `<p class="closing-limit" data-edit>확인된 성과: 일부 지속 상승의 경고·예측오차 개선<br>활용 제안: 담당자의 가동 시점·동시 사용 점검</p>`,
 'model,report','새알고리즘이란말보다어떤오류를어떻게줄였는지를마지막으로남긴다. 특정주간조건편중·단순규칙대비추가이득작음·독립시험아님을발표설명에서유지한다. 소속이나로고는넣지않았다.','closing');

/* === APPENDIX: DETAILS THAT SHOULD NOT CROWD MAIN STORY === */
add('부록 · 상태 구분','낮은 전력·전력 0·생산량 0은<br>서로 다른 기록 조건이다',
 table(['평일의 20~26 관측 시각','날짜 수'],[['00~06시만','29일'],['08~23시만','7일'],['하루 내내','18일'],['없음','117일'],['합계','171일']])+
 cols(point('낮은 구간','전체 20~26 구간은 1,936시간. 최장 223시간은 7/31 00시~8/9 06시이며 생산량은 모두 0.'),point('전력 0 구간','8/28 18시~8/29 10시의 단일 17시간. 직후 최대값은 41. 원인은 현 자료에서 미식별.')),
 'eda','양수20미만7시간은별도구분. 표는평일만대상. 낮은기록을정지·휴무·오류로단정하지않는다.','appendix');
add('부록 · 조건부 날씨 관계','같은 표본에서 조건을 고려하자<br>날씨 관계는 대체로 약해졌다',
 `<div class="period-label">평일 171일·4,102시간 · 네 기상변수 모두 관측</div>`+
 table(['기상변수','단순 순위상관','조건 고려 후 부분 순위 관계'],[['기온','+0.227','−0.004'],['풍속','+0.195','+0.046'],['습도','−0.102','+0.100'],['강수량','+0.057','−0.060']])+
 cols(point('8월 양수 생산량인 평일','360시간·17일에서 기온 관계 +0.302.<br>날짜별 차이를 고려하면 +0.014.'),point('남겨둔 해석','조건부 관계가 약해도 비선형 예측 가치를 배제하지 않는다. 과거 기상정보의 입력 가치는 예측오차로 판단한다.')),
 'analysis','월·시간·생산기록·다른기상변수로설명되는순위차이를제거한다. EDA의8월양수383시간은주말포함이며이표360시간은평일만이어서다르다.','appendix');

/* === SHORT DECK: TWELVE SLIDES, SECONDARY DETAIL IN NOTES === */
const original = slides.slice();
slides.length = 0;
slides.push({...original[0], note:original[0].note+' 압축 원고의 핵심만 묶은 12장 발표 구성이다.'});
add('분석 배경 및 목표','생산량만으로 설명되지 않는 전력,<br>다음 한 시간의 부하를 예측한다',
 cols(`<div class="selection compact-selection"><span class="eyebrow">5개 과제 중 선택한 과제 ⑤</span><h3 class="large" data-edit>자원 최적화<br>AI 데이터셋</h3><p data-edit>전력에 생산량·기상·달력이 연결돼 있어, 부하 예측과 오류 조건 분석을 함께 할 수 있다.</p>${point('분석 배경','예상보다 높은 최대 전력은 비용과 운영 부담으로 이어질 수 있다. 생산량이 0이어도 전력이 관측돼 최근 전력 정보가 필요했다.')}</div>`,
 `${point('01 · 최대 전력 예측','직전 관측으로 다음 한 시간의 최대값을 예측한다.')}${point('02 · 오류 조건 구분','지속 상승 시작과 최대 시간의 과소예측·미탐지·오경보를 확인한다.')}${point('03 · 현장 판단에 연결','담당자의 예정 가동·동시 사용 점검에 예측과 경고를 활용한다.')}`),
 'task,data,eda,model,report','대회 과제는 사출성형·용접·소성가공 예지보전·X-ray·자원 최적화 5종이다. 선택 이유는 현재 분석 목표와 제공 변수에 근거한 발표용 설명이며, 다른 과제와의 우열이나 과거 팀의 의사결정 경위는 주장하지 않는다. 비용·운영 부담은 출제 배경이고, 생산량 0의 전력 관측은 EDA 결과다. 활용 효과는 제안이며 실측 절감 성과가 아니다.');
add('데이터 이해 및 진단','한 시간의 네 전력값을 사용하고,<br>시간 오류와 입력 누수를 통제했다',
 `<div class="stats">${stat('6,168 × 18','원본 행 × 열')}${stat('5,784시간 · 241일','1~8월 정상 EDA 기록')}</div>`+
 table(['데이터 구조','진단·사용 기준'],[['한 행 = 한 시간','15·30·45·60분 전력값 + 생산량·기상·달력'],['시간 오류 48행','제외하고 정확한 시차만 연결'],['공장인원','전력·생산량에서 유도된 값이므로 독립 입력에서 제외'],['전력 0·큰 값·반복 배열','크기나 반복만으로 삭제하지 않고 유지']])+
 call('목표 시간의 실측 생산량·날씨·전력은 입력에 넣지 않는다. 전력 수치는 기록값 단위다.'),
 'data,model','원본은2021.01.01~09.14이며1~8월분석범위. EDA 5,832행중7/13·7/15의시간70~188기록48행제외. 전력·생산량결측없음;풍속3·강수량1셀은해당변수계산에서만제외. 공장인원결측17셀로행삭제하지않음. 비결측공장인원은생산량÷전력네값합과일치. 정상241일배열126종. 전력원단위·개별설비상태·생산량배정규칙은현자료에서미식별. 모델별적격표본은시차·평가기간에따라다르다.');
add('EDA · 핵심 관찰','생산량·시간 평균만으로는<br>전력의 수준과 변화를 설명하기 어렵다',
 cols(figure('EDA/figures/report_compact/daily_pattern_distribution.png','전력 음영: 날짜별 10~90백분위 범위. 신뢰구간이 아님.'),
 `${point('생산 정점과 전력 정점이 다름','평일 생산량은 19시, 평균 전력은 08시에 가장 높았다.')}${point('낮은 전력은 별도 구간으로 모임','평일 20~26은 747시간, 18.20%.<br>27~48에는 기록이 없었다.')}${point('시간 안에서도 구간별 차이가 남음','평일 07시의 마지막 값 > 첫 값:<br>160/171일, 93.6%')}`,'figure-split'),
 'eda','12시하락·13시상승은147/171일86.0%. 주말19시생산량모두0이나전력41.11. 8월평일08시전체142.68과양수생산날짜178.47차이로월별구성영향확인. 전체20~26연속구간71개·중앙지속24시간,최장223시간;전력0단일17시간은별도. 전체생산량·전력Spearman0.752,날씨전체관계는작지만8월기온은생산량양수조건에서0.599로달라짐. 시간내07·10·12·17시배열은각93.6·84.8·75.4·84.8%. 보조그림은assets에보존. 이숫자들은운영원인·예측성과가아님.');
add('Analysis · 모델 입력의 근거','마지막 15분 값에 남은 정보를<br>다음 시간 최대값 예측에 반영했다',
 cols(figure('Analysis/figures/a04_terminal_slots.png','연속 세 시간 기록 5,778시간. 달력·과거 관측을 고려한 부분 순위 관계.'),
 `${stat('+0.409','직전 평균·최대도 고려한 마지막 값의 관계')}${point('생산 전환과 전력 상승은 다름','낮은 전력에서 상승한 64시간 중<br>생산량도 0→양수인 경우는 5시간.')}${point('모델에서 확인할 질문','마지막 값과 상승 상태를 더하면 최대값·전환 시간의 예측오차가 줄어드는가?')}`,'figure-split'),
 'analysis','마지막값부분관계는직전최대추가전+0.434,추가후+0.409;예측정확도·오차개선율아님. 엄격동일조건표본1,039시간·낮은구간71.5%,1~4월−0.009예외. 생산0→양수의전력변화37.92는양수지속30.21보다25.5%큼;조건별최대값135.73/135.32차이는작음. 생산전환분류는달력+직전평균에서적중35→50/오탐51→25로입력가치확인했으나전력회귀성과와구분. 날씨는평일기온단순+0.227→조건고려−0.004;과거전력우선비교,비선형날씨예측가치배제아님.');
add('모델 개발 · 시간순 비교','같은 시간순 조건에서 비교하고,<br>전체·최대 오차를 함께 봤다',
 `<div class="validation-strip"><span>1~3월 → 4월</span><span>1~4월 → 5월</span><span>1~5월 → 6월</span></div>`+
 table(['대표 모델 · 같은 2,184시간','전체 MAE ↓','일별 최대 시간 MAE ↓'],[['직전 값 기준','13.793','36.589'],['<b>HGB</b>','<b>6.738</b>','11.751'],['ExtraTrees','6.825','<b>11.540</b>'],['XGBoost','7.131','12.864']])+
 call('평가 월 이전 자료로 튜닝했다. 사전 선택 규칙에 따라 HGB를 유지하고, 상승 시작의 오류를 별도로 보완했다.'),
 'model','Ridge·ElasticNet·SVR, HGB·XGBoost·LightGBM·CatBoost·ExtraTrees와앙상블비교. Optuna도내부시간순검증. ExtraTrees최대시간MAE는더작지만전체와최대오차함께본규칙에서HGB유지. 마지막15분값추가는개발에서유리했으나후반기최대시간개선이유지되지않아선택적상승보완으로연결. 이개발표와뒤B0/G1후반기표는표본·입력이달라직접성능순위비교하지않음. 예측대상은관측완료후다음1시간의네구간중최대값.');
slides.push({...original[17], note:original[17].note+' 지속 상승은직전값과과거6시간중앙값보다각각59.5이상,중앙값대비과거IQR1.5배이상상승하고이후3시간기준선+29.75이상유지한시작이다. 1월기준이며미래3시간은정답확인에만쓴다.'});
slides.push({...original[18]});
add('오류 분석 · 성과의 범위','7건을 탐지했지만 6건을 놓쳤고,<br>경고 시간의 예측도 항상 좋아지지는 않았다',
 `<div class="error-strip"><span><b>7/13건</b> 탐지 · 재현율 53.8%</span><span><b>7/7건</b> 경고 적중 · 정밀도 100%</span><span><b>4/7건</b> 값의 오차 감소</span></div>`+
 figure('Modeling/figures/modeling_close/transition_cases.png','중앙 순위 사례. 왼쪽 실제198: 181.13→192.52. 오른쪽 실제207: 경고 없이182.40.','wide')+
 call('경고는 모두 월요일 08시였다. 사후 주간 규칙 대비 추가 이득도 작았다.'),
 'model,report','7~8월탐색적재평가·분류정답1,340시간의공통상승13건. 개선4건은개선폭,미탐지6건은실제최대값중앙순위선정. 음영은지속3시간정답,점선은직전관측완료. 사후주간규칙8TP/1FP이고일별최대오차는동일. 적중/미탐지의과거전력평균로짓기여13.86/4.16으로과거전력도기여했지만공정인과해석은아님. 무경고를안전으로쓰지않으며범위밖일반무오경보를보장하지않음.','', '7~8월 탐색적 재평가 · 독립 최종시험이 아님');
slides.push({...original[21]});
add('코드 및 재현성','현재 고정 방법의 재현을 확인하고,<br>검증 범위를 함께 남겼다',
 steps([['원자료에서 입력 구성','시간 오류·공백을 처리하고 예측 시점에 맞는 시차·상태 정보를 만든다.'],['현재 선택 방법 재실행','Logistic과 기본·보완 HGB를 다시 학습해 고정 후반기 예측을 생성한다.'],['기존 결과와 대조','분류 1,428개·회귀 4,032개 예측의 일치를 확인했다.']])+
 call('정규 CPython 3.13에서 현재 재현 스크립트 실행을 확인했다.')+
 `<p class="lead lower" data-edit>새 환경 설치와 과거 Optuna 전체 재탐색은 검증하지 않았다.</p>`,
 'model',original[22].note);
add('결론 및 차별성','기본 예측을 유지하면서,<br>일부 지속 상승의 과소예측을 줄였다',
 cols(`<div class="closing-thesis"><span>분석의 기여</span><h3 data-edit>최대값과 상승 시작을<br>함께 예측<br><em>경고 시간만 보완</em></h3></div>`,
 `${point('상승 시작의 평균 과소예측','15.8% 감소. 전체 MAE 감소는 0.28%로 작았다.')}${point('오류 조건을 입력에 반영','분류기의 경과시간을 제거해 적중 7건을 유지하고 오경보 38→0건.')}${point('현장에 연결할 판단','예측·경고를 보고 가동 시작·동시 사용을 점검한다. 전력·비용 절감은 활용 제안이다.')}`)+
 call('남은 한계: 지속 상승 6건 미탐지 · 월요일 08시 편중 · 후반기 수정 결과는 탐색적 재평가'),
 'model,report','핵심기여는일부지속상승의선택적보완과실패조건반영이다. 사후주간규칙대비추가이득제한을유지한다. 가동조정·현장효과는미검증제안. 모든상승·양방향전환탐지나독립최종시험으로확대하지않는다.');

/* === STYLES: FIXED STAGE WITH QUIET INDUSTRIAL TYPOGRAPHY === */
const css=`
${BASE}
*{box-sizing:border-box} :root{--slide-bg:#f8f9f8;--stage-bg:#172a35;--ink:#162d3a;--muted:#516572;--accent:#087d8b;--rule:#cdd7db;--wash:#e8f1f2;--orange:#bb5932}
body{font-family:'Noto Sans KR','Malgun Gothic',sans-serif;color:var(--ink)}h1,h2,h3,p,figure{margin:0}button,select{font:inherit}button{cursor:pointer}
/* === SLIDE FRAME === */
.slide{padding:74px 98px 78px;background:var(--slide-bg)}.kicker{font-size:24px;font-weight:600;color:var(--accent);letter-spacing:.02em;margin-bottom:28px}.kicker span{float:right;color:var(--muted);font-size:21px;font-weight:400}.slide h2{font-size:62px;line-height:1.3;letter-spacing:-.045em;font-weight:750;margin-bottom:40px;max-width:1660px}.body{height:660px;position:relative}.slide footer{position:absolute;left:98px;right:98px;bottom:36px;border-top:1px solid var(--rule);padding-top:16px;display:flex;justify-content:space-between;color:var(--muted);font-size:21px;line-height:1.5;gap:25px}.slide footer span:first-child{max-width:1480px}.page-no{font-family:'DM Mono',monospace;color:var(--accent);white-space:nowrap}
/* === TYPOGRAPHY AND GRIDS === */
.columns{display:grid;grid-template-columns:1fr 1fr;gap:70px;align-items:start}.columns.figure-split{grid-template-columns:1120px 1fr;gap:48px}.point{padding:22px 0;border-top:1px solid var(--rule)}.point h3{font-size:31px;line-height:1.35;margin-bottom:15px;letter-spacing:-.02em}.point p,.selection p{font-size:28px;line-height:1.7;color:var(--muted);word-break:keep-all}.lead{font-size:34px;line-height:1.7;letter-spacing:-.02em;margin-bottom:38px}.lower{margin-top:42px}.callout{font-size:27px;line-height:1.6;padding:22px 28px;border-left:5px solid var(--accent);background:var(--wash);margin-top:28px;word-break:keep-all}.stats{display:flex;gap:60px;padding:4px 0 37px;border-bottom:1px solid var(--rule);margin-bottom:30px}.stat{flex:1}.stat strong{display:block;font-family:'Noto Sans KR',sans-serif;font-size:60px;line-height:1.25;font-weight:750;letter-spacing:-.04em;color:var(--accent)}.stat span{display:block;font-size:26px;line-height:1.5;margin-top:12px;color:var(--muted)}.figure-split .stat{margin:0 0 25px}.figure-split .stat strong{font-size:56px}.figure-split .point{padding:19px 0}.figure-split .point h3{font-size:29px}.figure-split .point p{font-size:26px;line-height:1.6}.period-label{color:var(--accent);font-size:25px;margin:0 0 20px}.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:52px;margin-bottom:28px}.steps>div{border-top:2px solid var(--accent);padding-top:27px}.step-no{display:block;font-family:'DM Mono',monospace;font-size:52px;color:var(--accent);margin-bottom:29px}.steps h3{font-size:34px;margin-bottom:20px;line-height:1.4}.steps p{font-size:28px;line-height:1.7;color:var(--muted);word-break:keep-all}
/* === TABLES AND FIGURES === */
table{border-collapse:collapse;width:100%;font-size:28px;line-height:1.55;table-layout:auto}th{text-align:left;font-size:25px;font-weight:600;background:var(--wash);padding:19px 22px;border-bottom:2px solid var(--accent);color:var(--ink)}td{padding:20px 22px;border-bottom:1px solid var(--rule);vertical-align:middle}td b{color:var(--accent)}.columns table{font-size:27px}.figure{height:555px;display:flex;flex-direction:column;justify-content:center;gap:15px;min-width:0}.figure img{width:100%;height:490px;object-fit:contain;background:white}.figure figcaption{font-size:22px;line-height:1.5;color:var(--muted);min-height:40px}.figure.wide{height:455px}.figure.wide img{height:400px}.two-findings{display:grid;grid-template-columns:1fr 1fr;gap:52px;border-top:1px solid var(--rule);padding-top:22px;margin-top:18px}.two-findings p{font-size:29px;line-height:1.6}.two-findings b{color:var(--accent);font-size:25px;margin-right:15px}.compact-stats{margin-top:32px;border:0;padding:0}.compact-stats .stat strong{font-size:48px}.compact-stats .stat span{font-size:24px}
/* === OPENING AND DECISION DIAGRAMS === */
.cover{background:#142e3b;color:#f8fbfa;padding-top:80px}.cover .kicker{color:#7dc7cf}.cover h2{font-size:88px;line-height:1.32;width:1350px;margin-top:95px;margin-bottom:47px;letter-spacing:-.055em}.cover .body{height:410px}.cover-sub{font-size:37px;line-height:1.6;color:#bdced6}.cover-bottom{display:flex;gap:24px;font-size:26px;margin-top:85px;color:#bdced6}.cover-bottom span:first-child{color:#80d1d5;border-right:1px solid #547481;padding-right:24px}.cover footer{border-color:#426071;color:#9bb0bc}.cover footer .page-no{color:#83cbd0}.cover-mark{position:absolute;right:18px;top:-275px;width:320px;height:545px;border-left:1px solid #5b8493;padding-left:35px;color:#79c7cb;display:flex;flex-direction:column;justify-content:space-between}.cover-mark>span:first-child{font-family:'DM Mono',monospace;font-size:150px;line-height:1}.cover-mark>span:last-child{font-family:'DM Mono',monospace;font-size:47px;line-height:1.2}.cover-mark>div{width:230px;height:95px;border-left:16px solid #78c8cb;border-bottom:16px solid #78c8cb;transform:skewY(-25deg)}
.dataset-list{border-top:1px solid var(--rule)}.dataset-row{display:grid;grid-template-columns:45px 245px 1fr;align-items:center;gap:17px;padding:28px 18px;border-bottom:1px solid var(--rule);font-size:28px}.dataset-row small{font-size:25px;color:var(--muted)}.dataset-row.selected{background:var(--wash);border-left:6px solid var(--accent);padding-left:12px}.dataset-row.selected b{color:var(--accent)}.selection{padding:28px 0 0 35px}.eyebrow{font-size:24px;color:var(--accent)}.selection h3.large{font-size:65px;line-height:1.35;letter-spacing:-.04em;margin:23px 0}.selection p{max-width:650px}.big-question{border-left:3px solid var(--accent);padding:30px 0 15px 52px}.big-question>span{font-size:25px;color:var(--accent)}.big-question h3{font-size:65px;line-height:1.45;letter-spacing:-.04em;margin-top:25px}.timeline{display:flex;justify-content:center;align-items:center;gap:65px;margin-bottom:35px}.timeline>div{flex:1;padding:28px 35px;border-top:3px solid var(--accent);background:var(--wash)}.timeline small{font-size:24px;color:var(--accent)}.timeline strong{display:block;font-size:42px;margin:15px 0}.timeline p{font-size:27px;line-height:1.6;color:var(--muted)}.timeline .target{background:#163b49;color:white}.timeline .target small,.timeline .target p{color:#badbdc}.arrow{font-size:54px;color:var(--accent)}.routing{display:grid;grid-template-columns:650px 1fr;gap:70px;align-items:center;margin-bottom:28px}.routing-input{padding:50px;background:var(--ink);color:white;border-right:8px solid var(--accent)}.routing-input small{font-size:26px;color:#b5d8db}.routing-input h3{font-size:47px;line-height:1.5;margin-top:22px}.routing-branches>div{padding:23px 30px;margin-bottom:20px;border-left:3px solid var(--accent);background:var(--wash)}.routing-branches span{font-size:24px;color:var(--accent)}.routing-branches h3{font-size:35px;margin:9px 0}.routing-branches p{font-size:26px;color:var(--muted)}
/* === COMPACT DECK: SMALL DATA TABLE AND SUMMARY STRIPS === */
.compact-selection{padding:0 25px 0 0}
.compact-selection h3.large{font-size:52px;margin:20px 0}
.compact-selection p{font-size:28px}
.compact-selection .point{margin-top:25px}
.validation-strip,.error-strip{display:flex;justify-content:space-between;gap:35px;background:var(--wash);padding:24px 28px;font-size:27px;line-height:1.5;margin-bottom:27px;color:var(--accent)}
.error-strip{font-size:25px;padding:19px 22px;margin-bottom:15px;color:var(--muted)}
.error-strip b{color:var(--accent)}
.slide[data-slide="3"] .stats{padding-bottom:20px;margin-bottom:22px}
.slide[data-slide="3"] .stat strong{font-size:54px}
.slide[data-slide="3"] th{padding:14px 22px}
.slide[data-slide="3"] td{padding:16px 22px}
.slide[data-slide="3"] .callout{padding:17px 28px;font-size:26px;line-height:1.5}
.slide[data-slide="6"] td{padding:17px 22px}
.slide[data-slide="9"] .figure.wide{height:408px}
.slide[data-slide="9"] .figure.wide img{height:355px}
.slide[data-slide="9"] .callout{margin-top:18px}
.slide[data-slide="12"] .closing-thesis h3{font-size:57px}
/* === TARGETED FIT: PRESERVE CONTENT, REDUCE UNUSED SPACING === */
.cover-sub,.cover-bottom{max-width:1320px}
.slide[data-slide="9"] .figure,.slide[data-slide="14"] .figure{height:510px}
.slide[data-slide="9"] .figure img,.slide[data-slide="14"] .figure img{height:445px}
.slide[data-slide="15"] .timeline{margin-bottom:22px}
.slide[data-slide="15"] .timeline>div{padding:18px 35px}
.slide[data-slide="15"] th{padding:13px 22px}
.slide[data-slide="15"] td{padding:12px 22px}
.slide[data-slide="15"] .callout{padding:16px 28px;margin-top:20px;font-size:26px;line-height:1.5}
.slide[data-slide="16"] th{padding:16px 22px}
.slide[data-slide="16"] td{padding:14px 22px}
.slide[data-slide="16"] .point{padding:14px 0}
.slide[data-slide="16"] .point h3{font-size:29px;margin-bottom:12px}
.slide[data-slide="16"] .point p{font-size:26px;line-height:1.55}
.slide[data-slide="16"] .callout{padding:16px 28px;margin-top:20px;font-size:26px;line-height:1.5}
.appendix td{padding:12px 22px}
.appendix .point p{font-size:27px;line-height:1.55}
/* === CLOSING === */
.closing{background:#e9f2f2}.closing h2{font-size:57px}.closing-thesis{padding-left:32px;border-left:5px solid var(--accent)}.closing-thesis>span{font-size:24px;color:var(--accent)}.closing-thesis h3{font-size:59px;line-height:1.45;letter-spacing:-.03em;margin-top:18px}.closing-thesis em{font-style:normal;color:var(--accent)}.closing-metrics{display:flex;gap:70px;margin-top:42px;border-top:1px solid #b6ccd0;padding-top:28px}.closing-metrics .stat strong{font-size:55px}.closing-limit{font-size:27px;line-height:1.7;color:var(--muted);margin-top:34px}
/* === MOTION AND EXTERNAL REVIEW UI === */
.slide.visible .body{animation:enter .25s ease-out}@keyframes enter{from{opacity:.55;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}.deck-controls{bottom:14px;background:#0d202bdc;color:#edf5f6;display:flex;align-items:center;justify-content:center;flex-wrap:wrap;width:max-content;gap:12px;padding:9px 16px;border:1px solid #4b6570;border-radius:8px;font-size:13px;white-space:nowrap;max-width:96vw}.deck-controls button,.deck-controls select{background:transparent;color:inherit;border:1px solid #607784;border-radius:4px;padding:6px 10px}.deck-controls select{max-width:270px;background:#19323e}.deck-controls button:hover{background:#345462}.deck-controls .counter{font-family:'DM Mono',monospace;min-width:50px;text-align:center}.notes-panel{position:fixed;right:18px;top:18px;width:min(520px,90vw);max-height:80vh;overflow:auto;background:#fff;box-shadow:0 6px 28px #0004;padding:25px;z-index:1100;border-top:5px solid var(--accent);font-size:16px;line-height:1.8}.notes-panel[hidden]{display:none}.notes-panel h3{font-size:20px;line-height:1.6;margin-bottom:16px}.notes-panel h4{margin:20px 0 6px}.notes-panel a{color:#086676}.notes-panel button{float:right}.edit-hotzone{position:fixed;left:0;top:0;width:70px;height:60px;z-index:1150}.edit-toggle{position:fixed;top:14px;left:14px;z-index:1200;opacity:0;pointer-events:none;padding:8px 12px;background:#163846;color:#fff;border:1px solid #80a1ac;border-radius:4px}.edit-toggle.show,.edit-toggle.editing{opacity:1;pointer-events:auto}.editing [data-edit]{outline:2px dashed #5999a7;outline-offset:4px}[contenteditable=true]{cursor:text}.progress{position:fixed;top:0;left:0;height:3px;background:#70c6ce;z-index:1000}.theme-paper{--slide-bg:#fbfaf6;--accent:#284b67;--wash:#eeeee7;--stage-bg:#272f35}.theme-signal{--slide-bg:#f0ece3;--ink:#1c2644;--muted:#566174;--accent:#8b662e;--wash:#e7e1d5;--rule:#cec5b5;--stage-bg:#1c2644}.theme-signal .cover{background:#1c2644}.theme-signal .closing{background:#e7e1d5}.theme-signal .cover-mark{color:#c8a870;border-color:#c8a870}.theme-signal .cover-mark>div{border-color:#c8a870}
@media(max-width:650px){.deck-controls{gap:5px;padding:7px 8px;font-size:11px;bottom:8px}.deck-controls select{max-width:115px}.deck-controls button{padding:5px}.deck-controls .theme-select,.deck-controls #printBtn,.deck-controls #saveBtn{display:none}}
@page{size:16in 9in;margin:0}@media print{.notes-panel,.edit-hotzone,.edit-toggle,.progress{display:none!important}.slide{page-break-inside:avoid} [data-edit]{outline:none!important}}
`;
const labels={data:'数据',eda:'EDA',analysis:'Analysis',model:'Modeling',report:'활용·차별성 원고',task:'대회 과제공개',form:'보고서 양식'};
labels.data='데이터';
const clean = s => s.replace(/<br\s*\/?\s*>/g,' ').replace(/<[^>]+>/g,'').replace(/&amp;/g,'&');
const shortSource = keys => keys.split(',').map(k=>labels[k]).join(' · ');
const rendered=slides.map((s,i)=>`<!-- Slide ${i+1}: ${clean(s.title)} -->\n<section class="slide ${s.cls} ${i===0?'active visible':''}" aria-label="${i+1}. ${esc(clean(s.title))}" data-slide="${i+1}"><div class="kicker" data-edit>${s.section}<span>제6회 K-인공지능 제조데이터 분석 경진대회</span></div><h2 data-edit>${s.title}</h2><div class="body">${s.body}</div><footer><span data-edit>${s.foot||'근거: '+shortSource(s.source)+' · 전력 수치는 기록값 단위'}</span><span class="page-no">${String(i+1).padStart(2,'0')} / ${slides.length}</span></footer></section>`).join('\n');
const metadata=slides.map((s,i)=>({number:i+1,title:clean(s.title),section:s.section,note:s.note,sources:s.source.split(',').map(k=>sources[k])}));
const script=`
/* === NAVIGATION: FIXED 1920 × 1080 STAGE === */
const metadata=${JSON.stringify(metadata)};
class SlidePresentation {
 constructor(){this.slides=[...document.querySelectorAll('.slide')];this.currentSlide=0;this.stage=document.querySelector('.deck-stage');this.notes=document.querySelector('.notes-panel');this.select=document.querySelector('#jump');this.setup();this.showSlide(Number(location.hash.slice(1)||1)-1);this.scale();}
 setup(){this.scale();addEventListener('resize',()=>this.scale());metadata.forEach((m,i)=>{const o=document.createElement('option');o.value=i;o.textContent=String(i+1).padStart(2,'0')+' '+m.title;this.select.append(o)});this.select.addEventListener('change',()=>this.showSlide(+this.select.value));document.querySelector('#prev').onclick=()=>this.showSlide(this.currentSlide-1);document.querySelector('#next').onclick=()=>this.showSlide(this.currentSlide+1);document.querySelector('#noteBtn').onclick=()=>{this.notes.hidden=!this.notes.hidden;};document.querySelector('#closeNotes').onclick=()=>this.notes.hidden=true;document.querySelector('#printBtn').onclick=()=>print();addEventListener('keydown',e=>{if(e.target.isContentEditable||e.target.tagName==='SELECT'||e.target.tagName==='INPUT')return;let n=this.currentSlide;if(['ArrowRight','ArrowDown','PageDown',' '].includes(e.key))n++;if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key))n--;if(e.key==='Home')n=0;if(e.key==='End')n=this.slides.length-1;if(n!==this.currentSlide){e.preventDefault();this.showSlide(n)}if(e.key.toLowerCase()==='n')this.notes.hidden=!this.notes.hidden;if(e.key.toLowerCase()==='e')editor.toggle();});let last=0;addEventListener('wheel',e=>{if(this.notes.contains(e.target)||editing)return;if(Date.now()-last<500||Math.abs(e.deltaY)<20)return;last=Date.now();this.showSlide(this.currentSlide+Math.sign(e.deltaY));},{passive:true});let start=null;this.stage.addEventListener('touchstart',e=>{start=e.touches[0].clientX;},{passive:true});this.stage.addEventListener('touchend',e=>{if(start!==null){const d=e.changedTouches[0].clientX-start;if(Math.abs(d)>40&&!editing)this.showSlide(this.currentSlide+(d<0?1:-1));start=null;}},{passive:true});addEventListener('hashchange',()=>this.showSlide(Number(location.hash.slice(1)||1)-1));document.querySelector('#theme').onchange=e=>{document.body.classList.remove('theme-paper','theme-signal');if(e.target.value)document.body.classList.add(e.target.value);};}
 scale(){const controlHeight=document.querySelector('.deck-controls').offsetHeight;const viewHeight=Math.max(1,innerHeight-controlHeight-28);const f=Math.min(innerWidth/1920,viewHeight/1080);this.stage.style.transform='translate('+((innerWidth-1920*f)/2)+'px,'+((viewHeight-1080*f)/2)+'px) scale('+f+')';}
 showSlide(n){n=Math.max(0,Math.min(this.slides.length-1,Number.isFinite(n)?n:0));this.currentSlide=n;this.slides.forEach((s,i)=>{s.classList.toggle('active',i===n);s.classList.toggle('visible',i===n);s.setAttribute('aria-hidden',String(i!==n));s.inert=i!==n;});this.select.value=n;document.querySelector('#counter').textContent=(n+1)+' / '+this.slides.length;document.querySelector('#prev').disabled=n===0;document.querySelector('#next').disabled=n===this.slides.length-1;document.querySelector('.progress').style.width=((n+1)/this.slides.length*100)+'%';history.replaceState(null,'','#'+(n+1));const m=metadata[n];document.querySelector('#noteTitle').textContent=String(n+1).padStart(2,'0')+' '+m.title;document.querySelector('#noteText').textContent=m.note;const list=document.querySelector('#sourceList');list.replaceChildren();m.sources.forEach(p=>{const li=document.createElement('li');const a=document.createElement('a');a.href='../'+p;a.textContent=p;a.target='_blank';li.append(a);list.append(li);});}
}
/* === LOCAL TEXT EDITING AND EXPORT === */
let editing=false;const key='manufacturing-power-deck-v1';const editable=[...document.querySelectorAll('[data-edit]')];editable.forEach((el,i)=>el.dataset.editId=i);try{const saved=JSON.parse(localStorage.getItem(key)||'{}');editable.forEach((el,i)=>{if(saved[i]!==undefined)el.innerHTML=saved[i];});}catch{}
const editor={toggle(){editing=!editing;document.body.classList.toggle('editing',editing);editable.forEach(el=>el.contentEditable=editing?'true':'false');const b=document.querySelector('#editToggle');b.classList.toggle('editing',editing);b.textContent=editing?'편집 종료 (E)':'문구 편집 (E)';}};
document.querySelector('#editToggle').onclick=()=>editor.toggle();let timer;const hot=document.querySelector('.edit-hotzone'),toggle=document.querySelector('#editToggle');[hot,toggle].forEach(el=>{el.onmouseenter=()=>{clearTimeout(timer);toggle.classList.add('show')};el.onmouseleave=()=>{timer=setTimeout(()=>{if(!editing)toggle.classList.remove('show')},400)}});hot.onclick=()=>editor.toggle();document.addEventListener('input',e=>{if(!e.target.closest('[data-edit]'))return;const saved={};editable.forEach((el,i)=>saved[i]=el.innerHTML);try{localStorage.setItem(key,JSON.stringify(saved))}catch{};});document.querySelector('#saveBtn').onclick=()=>{if(editing)editor.toggle();const cloned=document.documentElement.cloneNode(true);cloned.querySelector('body').classList.remove('editing');cloned.querySelectorAll('[contenteditable]').forEach(el=>el.removeAttribute('contenteditable'));const blob=new Blob(['<!DOCTYPE html>\\n'+cloned.outerHTML],{type:'text/html;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='발표구성_수정본.html';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),2000);};
window.deck=new SlidePresentation();
/* === REVIEW DIAGNOSTICS: READ-ONLY GEOMETRY, NO DATA/MODEL CLAIM === */
window.auditLayout=()=>{const scale=document.querySelector('.deck-stage').getBoundingClientRect().width/1920;const issues=[];for(const slide of document.querySelectorAll('.slide')){const sr=slide.getBoundingClientRect(),foot=slide.querySelector('footer').getBoundingClientRect();for(const el of slide.querySelectorAll('h2,.point,.stat,.callout,table,figure,.steps,.dataset-row,.selection,.big-question,.timeline,.routing,.two-findings,.closing-thesis,.closing-metrics,.closing-limit')){const r=el.getBoundingClientRect();if(r.left<sr.left-2||r.right>sr.right+2||r.top<sr.top-2||r.bottom>foot.top-8*scale)issues.push({slide:+slide.dataset.slide,type:'bounds',element:el.className||el.tagName});if(el.scrollHeight>el.clientHeight+3&&getComputedStyle(el).overflow!=='visible')issues.push({slide:+slide.dataset.slide,type:'overflow',element:el.className||el.tagName});}const children=[...slide.querySelector('.body').children];for(let i=0;i<children.length;i++)for(let j=i+1;j<children.length;j++){const a=children[i].getBoundingClientRect(),b=children[j].getBoundingClientRect();if(Math.min(a.right,b.right)-Math.max(a.left,b.left)>3&&Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>3)issues.push({slide:+slide.dataset.slide,type:'overlap',elements:[children[i].className,children[j].className]});}}return {slides:metadata.length,stage:[1920,1080],viewport:[innerWidth,innerHeight],images:[...document.images].every(i=>i.complete&&i.naturalWidth>0),issues};};
`;
const html=`<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>제조 전력 예측 · 발표 구성 시안</title><link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap" rel="stylesheet"><style>${css}</style></head><body><div class="progress"></div><div class="deck-viewport"><main class="deck-stage">${rendered}</main></div><nav class="deck-controls" aria-label="발표자료 탐색"><button id="prev" aria-label="이전 슬라이드">←</button><span class="counter" id="counter"></span><button id="next" aria-label="다음 슬라이드">→</button><select id="jump" aria-label="슬라이드 선택"></select><button id="noteBtn">발표·제작 메모 (N)</button><select id="theme" class="theme-select" aria-label="색상 비교"><option value="">청록·네이비</option><option value="theme-paper">흰색·잉크</option><option value="theme-signal">네이비·골드</option></select><button id="printBtn">인쇄</button><button id="saveBtn">HTML 저장</button></nav><aside class="notes-panel" hidden aria-label="발표 및 PPT 제작 메모"><button id="closeNotes">닫기</button><h3 id="noteTitle"></h3><h4>발표·제작 메모</h4><p id="noteText"></p><h4>근거 원고</h4><ul id="sourceList"></ul><p>실제 PPT는 와이드 16:9로 제작합니다. 문구 편집: E. 방향키·휠·좌우 스와이프로 이동합니다. 핵심 문구와 표는 PPT 텍스트·표로 옮기고, 기존 그림은 ppt/assets에서 가져오면 됩니다.</p></aside><div class="edit-hotzone" aria-hidden="true"></div><button id="editToggle" class="edit-toggle">문구 편집 (E)</button><script>${script}</script></body></html>`;
fs.writeFileSync(path.join(OUT,'발표구성_시안.html'),html,'utf8');
const outline=`# 발표 구성 시안\n\n압축 원고 전체를 발표 흐름으로 재구성했다. 본편 24장과 부록 2장, 와이드 16:9다. 사용자가 별도로 PPT를 만드는 참고 자료이며, 발표시간이 확정된 최종 제출본은 아니다.\n\n## 구성 원칙\n\n- 표지 다음에 5개 중 선택한 과제와 이유, 배경, 목표를 배치한다. 선택 이유는 현재 자료와 분석 목적에 따른 발표용 설명이다. 다른 과제와의 성능 우열이나 과거 팀의 선택 경위는 만들지 않았다.\n- EDA 관측에서 생긴 질문이 입력 선택과 모델링으로 이어지도록 구성한다.\n- 개발 모델 비교와 후반기 탐색적 재평가를 분리한다. 원고의 수치·정의·표본·한계를 유지한다.\n- 문서 양식은 구성 참고자료로 읽었다. 서명·제출·설문 등의 문구는 이번 요청의 실행 지시로 취급하지 않았다.\n- 현장 활용과 차별성은 Report의 제4·5장 설명을 참고하되, 성과 수치는 Modeling 압축본에 맞춘다. 제6장 원고 작성·노트북 생성은 이번 범위에 포함하지 않는다.\n\n## 슬라이드별 제작·발표 메모\n\n${metadata.map(m=>`### ${String(m.number).padStart(2,'0')}. ${m.title}\n\n- 구분: ${m.section}\n- 메모: ${m.note}\n- 근거: ${m.sources.map(p=>`[${p}](<../${p}>)`).join(' · ')}\n`).join('\n')}\n## PPT로 옮길 때\n\n16:9 와이드 슬라이드에서 제목 28~34pt, 본문 18~22pt, 핵심 수치 32~42pt를 출발점으로 잡는다. HTML의 1920×1080 비율과 배치를 참고하되 실제 PPT 화면에서 가독성을 다시 확인한다. 원고의 여러 문단을 그대로 붙이지 않고, 한 장의 제목과 핵심 근거를 중심으로 옮긴다. 그래프 원본 복사본은 assets에 있다.\n\n발표시간을 줄일 경우 8·10·11·13·17·23장을 부록으로 이동하거나 발표 설명을 짧게 하여, 배경·목표 / 진단 / 입력 근거 / 기본 모델 / 선택 보완 / 성능 / 실패 사례 / 활용 / 결론을 중심으로 남긴다. 25·26장은 원래 부록이다.\n`;
const compactOutline=outline.replace('본편 24장과 부록 2장','총 12장, 부록 없이').replace('압축 원고 전체를 발표 흐름으로 재구성했다.','압축 원고 전체에서 핵심 메시지를 묶어 발표 흐름으로 재구성했다.').replace(/발표시간을 줄일 경우[\s\S]*$/,'첫 26장 시안은 archive/26장_시안에 보존했다. 현재는 배경·목표, 진단, 핵심 관찰, 입력 근거, 모델 비교, 선택 보완, 성능, 실패 사례, 활용, 재현성, 결론을 12장으로 묶었다. 보조 수치와 조건은 발표·제작 메모에 남겼으며 별도 부록 슬라이드는 두지 않았다.\n');
fs.writeFileSync(path.join(OUT,'슬라이드_구성안.md'),compactOutline,'utf8');
const README=`# PPT 제작 참고 자료\n\n[발표 구성 HTML](<발표구성_시안.html>)을 브라우저에서 열면 본편 24장·부록 2장을 볼 수 있습니다. 실제 PPT 파일은 사용자가 별도로 제작합니다.\n\n- 이동: 좌우 방향키, Page Up/Down, 휠, 스와이프, 하단 슬라이드 선택.\n- 발표·제작 메모: 하단 버튼 또는 N. 원고 연결과 표본·해석 한계를 확인할 수 있습니다.\n- 문구 편집: E 또는 왼쪽 위에 마우스를 놓은 뒤 편집 버튼. 같은 브라우저에 문구를 저장합니다. HTML 저장으로 수정 파일을 내려받습니다.\n- 색상: 청록·네이비 기본, 흰색·잉크, 네이비·골드 비교.\n- 인쇄: 브라우저 인쇄. PDF 내보내기용 페이지 규칙을 포함했지만 실제 PDF 생성·페이지 수는 이번에 검증하지 않았습니다.\n- 글꼴: Google Fonts의 Noto Sans KR·DM Mono. 인터넷 없이 열면 로컬 맑은 고딕 등으로 대체됩니다.\n\n[슬라이드 구성안](<슬라이드_구성안.md>)에 장별 의도와 발표 설명을 정리했습니다. 기존 그림을 PPT에 넣을 때는 assets의 PNG 복사본을 사용합니다. HTML에도 그림을 내장하여 원고 폴더 밖으로 복사해도 화면을 볼 수 있습니다. 단, 메모의 원고 링크는 원래 프로젝트 폴더에서 사용할 때 유효합니다.\n\n원자료·원고·모델 결과를 변경하거나 모델을 재학습하지 않았습니다. build_presentation.cjs는 HTML·구성안·그림 복사본을 재생성하는 Node.js 스크립트입니다. 원고 내용이 바뀌면 이 스크립트의 슬라이드 내용도 대조해서 수정해야 합니다.\n`;
fs.writeFileSync(path.join(OUT,'README.md'),README.replace('본편 24장·부록 2장','총 12장').replace('원자료·원고·모델 결과를 변경하거나 모델을 재학습하지 않았습니다.','26장 첫 시안은 archive/26장_시안에 보존했습니다. 현재 시안은 부록 없이 12장으로 줄이고 보조 설명은 제작 메모에 남겼습니다.\n\n원자료·원고·모델 결과를 변경하거나 모델을 재학습하지 않았습니다.'),'utf8');
const sha=file=>crypto.createHash('sha256').update(fs.readFileSync(path.join(ROOT,file))).digest('hex');
fs.writeFileSync(path.join(OUT,'source_manifest.json'),JSON.stringify({created:new Date().toISOString(),slides:slides.length,main_slides:12,appendix_slides:0,embedded_figures:3,sources:Object.fromEntries(Object.values(sources).map(p=>[p,sha(p)])),figures:Object.fromEntries([...assets].map(([p,n])=>[p,{copy:'assets/'+n,sha256:sha(p)}])),scope:'Twelve-slide HTML draft; 26-slide original archived; existing results reused; no model execution'},null,2),'utf8');
console.log(JSON.stringify({slides:slides.length,figures:assets.size,htmlBytes:Buffer.byteLength(html),output:path.join(OUT,'발표구성_시안.html')}));
