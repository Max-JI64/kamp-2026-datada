from pathlib import Path
import ast, json, re, shutil, datetime, textwrap, importlib.metadata
import nbformat

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / '제출'
cells = []
def md(s): cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s): cells.append(nbformat.v4.new_code_cell(textwrap.dedent(s).strip()))
def section(title, question, source, interpretation):
    md(title + '\n\n' + question)
    code(source)
    md('**결과 해석**\n\n' + interpretation)
def function(path, name, new=None, replacements=()):
    source = (ROOT/path).read_text(encoding='utf-8-sig')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    out = ast.get_source_segment(source,node)
    if new: out = out.replace('def '+name+'(', 'def '+new+'(',1)
    for a,b in replacements: out=out.replace(a,b)
    return out
def readj(path): return json.loads((ROOT/path).read_text(encoding='utf-8'))
M01=readj('Modeling/config/m01_contract.json')
FC=readj('Modeling/tables/regime_followup/contract.json')
SC=readj('Modeling/tables/regime_forecast/contract.json')
AC=readj('Modeling/tables/regime_age_ablation/contract.json')
config={'period':M01['period'],'valid_hour':M01['valid_hour'],'lags_hours':M01['lags_hours'],'pools':M01['pools']}
settings={'onset_definition': {'thresholds':FC['onset_definition']['thresholds']}, 'logistic':AC['logistic'],
          'regression_features':FC['regression_features'],'hgb_parameters':FC['hgb_parameters'],
          'classification_features':AC['features'],'calendar_features':M01['calendar_features']}

md('''# 제조데이터 분석 제출 노트북

보고서의 데이터 확인 → EDA → Analysis → Modeling 순서로 핵심 결과를 재현한다.
각 소목차는 질문과 계산 방법, 실행 코드, 실제 표·그림 출력, 결과 해석으로 구성한다.
원본 CSV를 이 폴더에서 읽어 전처리하며, 이후 분석과 현재 모델 학습은 이 자료를 이어서 사용한다.

`requirements.txt` 설치 후 해당 Python 커널을 선택하고 **커널 재시작 및 전체 실행**을 수행한다.
모든 계산 코드는 이 노트북 안에 있으며 별도 Python 소스 파일을 사용하지 않는다.
과거 튜닝은 당시 선택한 설정을 고정해 재학습한다. Optuna 전체 탐색을 다시 수행하지 않는다.
7~8월은 이미 관찰한 자료의 탐색적 재평가이며 새로운 독립 시험이 아니다.
''')
md('## 실행 환경과 공통 설정\n\n라이브러리는 한 번만 불러온다. 원자료와 결과 경로는 제출 폴더 내부의 상대 경로를 사용한다.')
code('''from pathlib import Path
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import hashlib
import io
import math
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from IPython.display import display

PACKAGE = Path.cwd()
if not (PACKAGE / "data/raw/okm_augumented_2021.csv").exists():
    PACKAGE = PACKAGE / "제출"
SOURCE = PACKAGE / "data/raw/okm_augumented_2021.csv"
assert SOURCE.exists(), "제출 폴더에서 노트북을 실행하세요."
OUTPUT = PACKAGE / "output"
OUTPUT.mkdir(exist_ok=True)
SLOTS = ["15분", "30분", "45분", "60분"]
WEATHER = ["기온", "풍속", "습도", "강수량"]
WEATHER_MAP = dict(temperature="기온", wind="풍속", humidity="습도", rain="강수량")
KEYS = ["month", "weekend", "시간"]
COLORS = ["#2878B5", "#D97A27", "#439B80"]
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False,
                     "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.max_rows", 30)
pd.set_option("display.max_columns", 12)
def show_figure(name):
    plt.gcf().tight_layout()
    plt.gcf().savefig(OUTPUT / (name + ".png"), dpi=160, bbox_inches="tight")
    plt.show()
def show_table(frame, name):
    frame.to_csv(OUTPUT / (name + ".csv"), index=False, encoding="utf-8-sig")
    display(frame.round(4))
def heatmap(ax, values, xlabels, ylabels, vmin, vmax, cmap="coolwarm", annotate=True):
    im=ax.imshow(values, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=0)
    ax.set_yticks(range(len(ylabels)), ylabels)
    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i,j]):
                    ax.text(j,i,f"{values[i,j]:.1f}",ha="center",va="center",fontsize=8)
    return im
display(pd.DataFrame({"항목":["Python", "원자료", "결과 저장"],
                      "값":[sys.version.split()[0], str(SOURCE.relative_to(PACKAGE)), "output/"]}))
''')
section('# 1. 데이터 확인과 전처리\n\n## 1.1 원본 자료와 EDA 범위',
        '원자료의 규모를 확인하고 1~8월 정상 시간 기록을 공통 분석 자료로 만든다. 시간 오류는 별도로 보존하며 결측·0·극단값은 임의로 바꾸지 않는다.', '''
raw_data = pd.read_csv(SOURCE, encoding="utf-8-sig")
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"
raw_data["날짜"] = pd.to_numeric(raw_data["날짜"], errors="raise").astype(int)
raw_data["시간"] = pd.to_numeric(raw_data["시간"], errors="raise").astype(int)
scoped = raw_data.loc[raw_data["날짜"].between(20210101,20210831)].copy()
invalid = scoped.loc[~scoped["시간"].between(0,23)].copy()
data = scoped.loc[scoped["시간"].between(0,23)].copy()
data["date"] = pd.to_datetime(data["날짜"].astype(str), format="%Y%m%d")
data["timestamp"] = data.date + pd.to_timedelta(data["시간"],unit="h")
data = data.sort_values("timestamp").reset_index(drop=True)
assert data.timestamp.is_unique
assert data.groupby("date")["시간"].apply(lambda x:set(x)==set(range(24))).all()
data["month"] = data.date.dt.month
data["weekend"] = data.date.dt.dayofweek.ge(5).astype(int)
data["level"] = data[SLOTS].mean(axis=1)
data["maximum"] = data[SLOTS].max(axis=1)
data["slot_range"] = data[SLOTS].max(axis=1)-data[SLOTS].min(axis=1)
data["week"] = data.date-pd.to_timedelta(data.date.dt.dayofweek,unit="D")
(PACKAGE/"data/processed").mkdir(exist_ok=True)
data.to_csv(PACKAGE/"data/processed/정상시간_전처리자료.csv", index=False, encoding="utf-8-sig",float_format="%.17g")
invalid.to_csv(OUTPUT/"제외한_시간오류.csv",index=False,encoding="utf-8-sig")
scope_table=pd.DataFrame({"범위":["원본 전체","1~8월","시간 오류","정상 분석 자료","9월 범위 밖"],
    "행 수":[len(raw_data),len(scoped),len(invalid),len(data),len(raw_data)-len(scoped)]})
show_table(scope_table,"01_분석범위")
''', '원본은 6,168행·18열이다. 1~8월 5,832행에서 시간 오류 48행을 제외한 5,784시간·241일을 사용한다. 9월 336행은 기간 범위 밖이며 오류로 제거한 행이 아니다. 이후 분석은 위에서 만든 `data`를 사용한다.')
section('## 1.2 변수의 의미와 계산 관계', 'CSV 평균의 반올림 관계와 공장인원의 계산 관계를 확인한다. 관찰 변수의 의미와 예측 시점에 사용할 수 있는 정보를 구분한다.', '''
slot_mean=raw_data[SLOTS].mean(axis=1)
rounded=np.floor(slot_mean+.5)
ratio=raw_data["생산량"]/raw_data[SLOTS].sum(axis=1).replace(0,np.nan)
known=raw_data["공장인원"].notna()
variable_checks=pd.DataFrame({"점검":["평균 = floor(네 값 평균 + 0.5)","공장인원 = 생산량 / 네 전력값 합"],
    "비교 행 수":[len(raw_data),int(known.sum())],
    "최대 절대 차이":[float(abs(rounded-raw_data["평균"]).max()),float(abs(ratio[known]-raw_data.loc[known,"공장인원"]).max())]})
show_table(variable_checks,"02_변수계산관계")
display(pd.DataFrame({"변수":["네 15분 전력", "CSV 평균", "생산량", "기상", "날짜·시간", "공장인원"],
    "사용":["목표 최대값·과거 전력 입력", "EDA 수준 비교", "관계 탐색·직전 생산 입력", "관계 탐색·과거 기상 후보", "정확한 시차·달력", "별도 외부 조건에서 제외"]}))
''','평균은 네 값의 산술평균을 정수로 반올림한 값이다. 공장인원은 생산량/전력합과 저장 오차 수준으로 일치하므로 실제 근무 인원으로 해석하지 않는다. 구간 차이·시차 분석에는 반올림 전 평균을 사용한다. 목표 시간의 실측 전력·생산량·날씨는 예측 입력에 넣지 않는다. 원단위와 설비 운전 원인은 현 자료에서 미식별이다.')
section('## 1.3 결측 확인과 처리', '어느 변수의 어느 시간에 결측이 있는지 확인한다. 관계 분석에서 필요한 변수의 결측만 제외한다.', '''
missing=raw_data.isna().sum().rename("결측 셀 수").reset_index(names="변수")
show_table(missing.loc[missing["결측 셀 수"].gt(0)],"03_결측수")
missing_rows=data.loc[data[WEATHER+["공장인원"]].isna().any(axis=1),["timestamp"]+WEATHER+["공장인원"]]
show_table(missing_rows,"03_결측위치")
''','결측은 풍속 3셀, 강수량 1셀, 공장인원 17셀이다. 전력·생산량에는 결측이 없다. 보간하거나 0으로 채우지 않는다. 공장인원 17셀은 전력과 생산량이 모두 0인 0/0 기록이며, 해당 시간의 전력 기록은 유지한다.')
section('## 1.4 시간 오류·극단값·0값의 처리', '시간 오류와 분포상 큰 값의 차이를 점검한다. IQR 밖이라는 이유만으로 관측을 삭제하지 않는다.', '''
rows=[]
for col in ["생산량","강수량"]:
    v=scoped[col].dropna(); q1,q3=v.quantile([.25,.75]); width=q3-q1
    rows.append({"변수":col,"관측 수":len(v),"0 비율(%)":100*v.eq(0).mean(),"IQR":width,
                 "IQR 기준 밖":int((v.lt(q1-1.5*width)|v.gt(q3+1.5*width)).sum()),"최댓값":v.max()})
show_table(pd.DataFrame(rows),"04_분포점검")
show_table(invalid.groupby("날짜")["시간"].agg(["size","min","max"]).reset_index(),"04_시간오류")
print("네 전력값 모두 0인 정상 시간:",int(data[SLOTS].eq(0).all(axis=1).sum()))
''','7월 13일·15일 48행의 시간은 70~188로 정상 시각을 식별할 수 없어 제외한다. 생산량 극단값과 양수 강수는 유지한다. 강수 IQR이 0이므로 IQR 기준을 삭제 규칙으로 쓰면 모든 양수 강수를 제거하게 된다. 전력 네 값이 모두 0인 17시간도 유지하며 그 원인을 추정하지 않는다.')
section('## 1.5 반복 기록과 최종 분석 자료', '날짜가 다른 동일한 96개 전력 배열을 확인하고 요일별 분석 분모를 확정한다.', '''
daily_profiles={date:hashlib.sha256(g[SLOTS].to_numpy(dtype=np.int64).tobytes()).hexdigest()
                for date,g in data.groupby("date")}
data["profile"]=data.date.map(daily_profiles)
profile_counts=pd.Series(daily_profiles).value_counts()
data["profile_weight"]=1/data.profile.map(profile_counts)
summary=pd.DataFrame({"항목":["완전 중복 원본 행","정상 날짜","서로 다른 하루 배열","반복 그룹 날짜","하루 생산량 여러 값", "원본 생산량 0", "생산량 0이면서 양수 전력"],
    "수":[raw_data.duplicated().sum(),data.date.nunique(),len(profile_counts),profile_counts[profile_counts.gt(1)].sum(),
           data.groupby("date")["생산량"].nunique().gt(1).sum(),raw_data["생산량"].eq(0).sum(),
           (raw_data["생산량"].eq(0)&raw_data["평균"].gt(0)).sum()]})
show_table(summary,"05_반복기록")
show_table(data.groupby("weekend").agg(날짜수=("date","nunique"),시간수=("timestamp","size")).reset_index(),"05_요일범위")
''','241일의 서로 다른 전력 배열은 126개이고 160일이 반복 그룹에 속한다. 날짜별 생산량·날씨가 동일한 완전 중복 행은 아니므로 날짜를 삭제하지 않는다. 평일은 171일·4,104시간, 주말은 70일·1,680시간이다. 반복 원인은 현 자료에서 미식별이며 민감도 비교에서는 역빈도 가중으로 반복 배열의 비중을 점검한다.')
section('# 2. EDA\n\n## 2.1 평일·주말의 생산량과 전력 하루 패턴', '같은 시간대의 생산량과 CSV 평균 전력을 요일 집단별로 평균한다. 서로 다른 단위는 축을 분리한다.', '''
hourly=data.groupby(["weekend","시간"])[["생산량","평균"]].mean().reset_index()
show_table(hourly.loc[hourly["시간"].isin([8,11,12,13,19])],"06_시간대평균")
fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
for group,label,color in [(0,"평일",COLORS[0]),(1,"주말",COLORS[1])]:
    g=hourly.loc[hourly.weekend.eq(group)]
    for ax,col in zip(axes,["생산량","평균"]):
        ax.plot(g["시간"],g[col],label=label,color=color,linestyle="-" if group==0 else "--")
        ax.set_ylabel(col+" (기록값)");ax.grid(alpha=.2);ax.legend()
axes[-1].set_xticks(range(24));axes[-1].set_xlabel("시간")
show_figure("06_하루패턴")
''','평일 두 값은 12시에 낮아지고 13시에 다시 높아진다. 생산량 최고 시간은 19시, 전력 최고 시간은 08시로 서로 다르다. 주말 19시 생산량은 모두 0이지만 전력 평균은 41.11이다. 생산량만으로 전력 패턴을 설명하기 어렵고 시간대·요일을 함께 고려해야 한다.')
section('## 2.2 날짜별 차이와 월별 전력 수준', '평균선이 모든 날짜를 대표하는지 분포와 월별 평균을 함께 확인한다. 생산량 0인 날짜도 포함한다.', '''
day_power=data.pivot(index="date",columns="시간",values="평균")
day_production=data.groupby("date")["생산량"].max()
lunch=(day_power[12]<day_power[11])&(day_power[13]>day_power[12])
rows=[]
for group,label in [(0,"평일"),(1,"주말")]:
    dates=data.loc[data.weekend.eq(group),"date"].unique();p=day_power.loc[dates]
    rows.append({"집단":label,"전체 날짜":len(p),"12시 하락·13시 상승":int(lunch.loc[dates].sum()),
                 "11→12 중앙값":(p[12]-p[11]).median(),"12→13 중앙값":(p[13]-p[12]).median()})
show_table(pd.DataFrame(rows),"07_날짜별패턴")
fig,axes=plt.subplots(2,2,figsize=(13,7))
for group,label in [(0,"평일"),(1,"주말")]:
    p=data.loc[data.weekend.eq(group)].groupby("시간")["평균"]
    q=p.quantile([.1,.9]).unstack(); ax=axes[0,group]
    ax.fill_between(q.index,q[.1],q[.9],alpha=.2,color=COLORS[group],label="10~90백분위")
    ax.plot(p.mean(),ls="--",color=COLORS[group],label="평균")
    ax.plot(p.median(),color=COLORS[group],label="중앙값");ax.set_title(label);ax.legend(fontsize=8);ax.set_ylabel("전력")
    matrix=data.loc[data.weekend.eq(group)].pivot_table(index="month",columns="시간",values="평균",aggfunc="mean")
    im=heatmap(axes[1,group],matrix.to_numpy(),list(range(24)),matrix.index,0,210,"YlOrRd",False)
    axes[1,group].set_xlabel("시간");axes[1,group].set_ylabel("월");fig.colorbar(im,ax=axes[1,group],label="평균 전력")
show_figure("07_날짜와월별차이")
''','평일 147/171일, 주말 13/70일에 12시 하락·13시 상승이 나타난다. 음영은 날짜별 분포이며 신뢰구간이 아니다. 평균과 중앙값의 차이는 같은 시간대에 낮고 높은 날짜가 섞였음을 보여준다. 8월 평일의 낮은 평균에는 생산량이 하루 내내 0인 5일의 구성도 포함돼 있다.')
section('### 실제 시간순 전력과 낮은 구간의 지속', '관측 공백에서 선을 끊고 월별 실제 평균·최대값을 표시한다. 낮은 구간의 연속 기록은 공백을 넘어 연결하지 않는다.', '''
grid=data.set_index("timestamp").reindex(pd.date_range("2021-01-01","2021-08-31 23:00",freq="h"))
fig,axes=plt.subplots(4,2,figsize=(14,10),sharey=True)
for month,ax in zip(range(1,9),axes.flat):
    g=grid.loc[grid.index.month==month]
    ax.plot(g.index,g.level,label="산술평균",lw=.6,color=COLORS[0]);ax.plot(g.index,g.maximum,label="최대값",lw=.6,color=COLORS[1])
    ax.set_title(f"{month}월");ax.set_ylim(0,225);ax.tick_params(axis="x",labelsize=8)
axes[0,0].legend();show_figure("08_월별실제전력")
low=grid["평균"].between(20,26)
segments=(low.ne(low.shift())).cumsum()
episodes=grid.loc[low].groupby(segments[low]).apply(lambda g:pd.Series({"시작":g.index.min(),"종료":g.index.max(),"지속시간":len(g)}),include_groups=False)
show_table(episodes.sort_values("지속시간",ascending=False).head(5).reset_index(drop=True),"08_낮은구간지속")
''','낮은 수준이 이어지는 기간과 높은 값이 반복되는 기간이 함께 있다. 공백 시각은 보간하지 않는다. 지속시간은 정상 시각의 연속 기록 수이며 실제 설비 정지 시간을 뜻하지 않는다. 이후 분석은 낮은 구간의 출입뿐 아니라 높은 구간 안의 추가 상승·하락도 포함한다.')
section('## 2.3 생산량·날씨와 전력의 관계', '전체와 월별 Spearman 순위상관을 비교한다. 변수별 결측은 해당 쌍에서만 제외한다.', '''
variables=["생산량"]+WEATHER+["평균"]
correlations=data[variables].corr(method="spearman")
monthly=pd.DataFrame({month:g[["생산량"]+WEATHER].corrwith(g["평균"],method="spearman") for month,g in data.groupby("month")}).T
show_table(correlations.reset_index(names="변수"),"09_전체상관")
fig,axes=plt.subplots(1,2,figsize=(13,5))
im=heatmap(axes[0],correlations.to_numpy(),variables,variables,-1,1)
heatmap(axes[1],monthly.to_numpy(),monthly.columns,monthly.index,-1,1)
axes[0].set_title("전체 순위상관");axes[1].set_title("월별 전력과의 순위상관");fig.colorbar(im,ax=axes[1])
show_figure("09_전체월별관계")
aug=data.loc[data.month.eq(8)]
show_table(pd.DataFrame([{"8월 집단":name,"시간 수":len(g),"기온·전력 상관":g["기온"].corr(g["평균"],method="spearman")}
                        for name,g in [("전체",aug),("생산 양수",aug.loc[aug["생산량"].gt(0)]),("생산 0",aug.loc[aug["생산량"].eq(0)])]]),"09_8월구성")
''','전체 생산량·전력 상관은 0.752이다. 전체 날씨 상관은 작지만 기온은 2월 −0.356, 4월 +0.383으로 방향이 다르다. 8월 전체 −0.076과 생산량 양수 383시간 +0.599의 차이는 기간·생산 구성의 영향을 점검할 근거다. 상관은 인과효과나 예측 성능을 뜻하지 않는다.')
section('## 2.4 평일 전력 분포와 낮은 시간 조합', '평일 4,104시간을 모두 포함해 실제 빈 구간과 날짜별 낮은 시각의 정확한 조합을 계산한다.', '''
weekday=data.loc[data.weekend.eq(0)]
unique=np.sort(weekday["평균"].unique());gap=int(np.argmax(np.diff(unique)));cutoff=unique[gap]
assert cutoff==26 and unique[gap+1]==49
patterns=weekday.groupby("date").apply(lambda g:tuple(g.loc[g["평균"].le(cutoff),"시간"]),include_groups=False)
labels={tuple(range(7)):"07시부터 낮은 구간 벗어남",tuple(range(8,24)):"08시부터 낮은 구간 진입",tuple(range(24)):"하루 내내 낮음",():"하루 내내 49 이상"}
pattern_table=pd.DataFrame([{"형태":label,"날짜 수":int(patterns.map(lambda x:x==key).sum())} for key,label in labels.items()])
show_table(pattern_table,"10_평일시간조합")
fig,ax=plt.subplots(figsize=(10,4))
ax.hist([weekday.loc[weekday["평균"].le(cutoff),"평균"],weekday.loc[weekday["평균"].gt(cutoff),"평균"]],
        bins=np.arange(19.5,214.6,5),stacked=True,color=[COLORS[1],COLORS[0]],label=["20~26","49 이상"])
ax.set_xlim(15,215);ax.set_xlabel("CSV 평균 전력");ax.set_ylabel("시간 수");ax.legend();show_figure("10_평일분포")
print("낮은 시간:",int(weekday["평균"].le(cutoff).sum()),"그중 양수 생산:",int((weekday["평균"].le(cutoff)&weekday["생산량"].gt(0)).sum()))
''','평일 747시간(18.20%)은 20~26에 있고 27~48은 비어 있다. 네 시간 조합은 각각 29·7·18·117일이다. 낮은 747시간 중 92시간에는 생산량이 양수다. 이 구간은 관측 분포를 요약한 기술적 기준이며 휴무·설비 정지 판정이 아니다.')
section('## 2.5 시간대별 15분 구간의 전력 패턴', '각 구간에서 같은 시간의 반올림 전 평균을 빼고 평일·주말의 시간대별 차이를 평균한다. 높은 값이 나타난 구간 수도 확인한다.', '''
centered=data[SLOTS].sub(data.level,axis=0)
centered["weekend"]=data.weekend;centered["시간"]=data["시간"]
fig,axes=plt.subplots(1,2,figsize=(9,10))
for group,label in [(0,"평일"),(1,"주말")]:
    matrix=centered.loc[centered.weekend.eq(group)].groupby("시간")[SLOTS].mean()
    im=heatmap(axes[group],matrix.to_numpy(),SLOTS,[f"{h:02}시" for h in matrix.index],-23,23)
    axes[group].set_title(label)
fig.colorbar(im,ax=axes[1],label="시간 평균 대비 차이");show_figure("11_구간별차이")
rows=[]
for group,label in [(0,"평일"),(1,"주말")]:
    for hour in [7,10,12,17]:
        g=data.loc[data.weekend.eq(group)&data["시간"].eq(hour)]
        mask={7:g["60분"]>g["15분"],10:g["15분"]<g[SLOTS[1:]].min(axis=1),
              12:g[["30분","45분"]].max(axis=1)<g[["15분","60분"]].min(axis=1),
              17:g[["45분","60분"]].min(axis=1)>g[["15분","30분"]].max(axis=1)}[hour]
        rows.append({"집단":label,"시간":hour,"해당 날짜":int(mask.sum()),"전체 날짜":len(g)})
show_table(pd.DataFrame(rows),"11_배열조건")
peak_cut=data.loc[data.month.eq(1),"maximum"].quantile(.95)
counts=data[SLOTS].ge(peak_cut).sum(axis=1)
peak_counts=counts[counts.gt(0)].value_counts().sort_index().rename("시간 수").reset_index(names="기준 이상 구간 수")
peak_counts["비율(%)"]=100*peak_counts["시간 수"]/peak_counts["시간 수"].sum()
print("1월 고정 기준:",peak_cut);show_table(peak_counts,"11_높은구간수")
''','평일 07·10·12·17시에는 구간 위치에 따른 차이가 여러 날짜에서 관측된다. 평일 10시 첫 구간의 평균 대비 차이는 −22.44다. 1월 최대값 95분위수 185를 고정하면 기준 이상 360시간 중 한 구간만 높은 시간은 136개, 네 구간 모두 높은 시간은 28개다. 최대값 하나로 높은 기록의 분포 폭을 모두 설명할 수 없으며 미래 예측 가치는 시간순 검증이 필요하다.')

md('# 3. Analysis\n\n## 연속 시간 비교를 위한 준비\n\n정확히 1·2시간 전의 실제 기록을 연결한다. 순위 관계를 조정할 때 같은 달력 조건의 평균 차이를 먼저 제거하고 과거 생산량·전력 수준을 고려한다. 아래 함수는 이 절에서 반복 사용하는 계산만 정의한다.')
analysis_load=function('Analysis/scripts/a01_production_changes.py','load','build_pairs',[("    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA\n",''),("    raw=pd.read_csv(SOURCE,encoding='utf-8-sig')","    raw=raw_data.copy()")])
analysis_build=function('Analysis/scripts/a04_maximum_conditions.py','build','build_triples',[("data, part = load()","data, part = build_pairs()")])
stats=function('Analysis/scripts/a02_prior_power_signals.py','weighted_corr')+'\n\n'+function('Analysis/scripts/a02_prior_power_signals.py','residuals')
code(analysis_load+'\n\n'+analysis_build+'\n\n'+stats+'\n\n'+function('Analysis/scripts/a01_production_changes.py','contrast')+'\n\n'+function('Analysis/scripts/a04_maximum_conditions.py','control_array')+'\n\n'+'''analysis_data, pairs, triples = build_triples()
display(pd.DataFrame({"자료":["정상 기록","정확한 1시간 연결","정확한 1·2시간 연결"],"시간 수":[len(analysis_data),len(pairs),len(triples)]}))''')
section('## 3.1 생산량 전환과 전력 변화', '생산 증감·유지의 구성부터 확인한 뒤 같은 월·시간대·평일/주말에서 생산량 0→양수와 양수 지속을 비교한다. 각 집단에 최소 두 날짜가 있는 조건만 맞춘다.', '''
show_table(pairs.groupby(["change","transition"]).size().rename("시간 수").reset_index(),"12_생산전환구성")
transition_pairs=pairs.assign(change=pairs.transition)
comparisons=pd.DataFrame([contrast(transition_pairs,"zero_to_positive","positive_to_positive",metric,mode)
    for metric in ["abs_power_delta","target_maximum","slot_range"] for mode in ["date_hour","profile_balanced"]])
show_table(comparisons[["metric","weighting","matched_cells","n_a","n_b","adjusted_a","adjusted_b","difference"]],"12_달력조건비교")
main=comparisons.loc[comparisons.weighting.eq("date_hour")]
fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(main))
ax.bar(x-.18,main.adjusted_a,.36,label="생산 0→양수",color=COLORS[0]);ax.bar(x+.18,main.adjusted_b,.36,label="양수 지속",color=COLORS[1])
ax.set_xticks(x,["시간 사이 변화 크기","현재 최대전력","현재 구간 범위"]);ax.set_ylabel("조건을 맞춘 평균");ax.legend();show_figure("12_생산전환비교")
''','생산량 유지 2,296시간 중 2,292시간은 0→0이다. 유지와 증감의 비교를 증감 자체의 영향으로 해석하기 어렵다. 같은 달력 조건에서 0→양수의 전력 변화 크기는 더 컸지만 현재 최대전력은 135.73 대 135.32로 비슷했다. 큰 변화와 높은 최대값은 구분해야 하며 기록상 생산 전환을 실제 설비 재가동으로 해석하지 않는다.')
section('### 전력 구간 전환과 생산량 전환의 차이', '전력 20~26·26 초과·20 미만·0을 구분해 전환과 생산량 전환이 얼마나 겹치는지 확인한다. 분류는 관측된 목표값의 사후 분석에만 사용한다.', '''
def power_state(v):
    return np.select([v.eq(0),v.between(20,26),v.gt(26)],["0","20~26","26 초과"],default="양수 20 미만")
pairs["previous_state"]=power_state(np.floor(pairs.past_level+.5))
pairs["current_state"]=power_state(pairs["평균"])
prior_max=analysis_data.set_index("timestamp").maximum.reindex(pairs.timestamp-pd.Timedelta(hours=1)).to_numpy()
pairs["maximum_delta"]=pairs.target_maximum-prior_max
state_summary=pairs.groupby(["previous_state","current_state"]).agg(시간수=("timestamp","size"),평균변화중앙값=("power_delta","median"),최대변화중앙값=("maximum_delta","median")).reset_index()
show_table(state_summary,"13_전력전환")
overlap=pd.crosstab(pairs.previous_state+"→"+pairs.current_state,pairs.transition)
show_table(overlap.reset_index(names="전력 전환"),"13_생산전환겹침")
fig,ax=plt.subplots(figsize=(11,4))
overlap.plot.bar(stacked=True,ax=ax,color=[COLORS[0],COLORS[1],COLORS[2],"#999999"])
ax.set_ylabel("시간 수");ax.set_xlabel("전력 구간 전환");ax.tick_params(axis="x",rotation=30);ax.legend(fontsize=8);show_figure("13_전환겹침")
''','20~26→26 초과 상승 64시간 중 생산량 0→양수는 5시간뿐이다. 반대 하락 64시간 중 생산량 양수→0은 35시간이다. 두 전환은 일치하지 않는다. 26 초과가 이어진 시간에도 큰 상승·하락이 있어 생산 전환이나 낮은 구간 출입만으로 전체 전력 변동을 대신할 수 없다.')
section('## 3.2 과거 전력으로 생산량 전환을 사전 구분', '직전 생산량 0인 기록에서 다음 양수 전환을 분류한다. 1~4월 학습·5~6월 검증에서 정한 HGB 설정과 경계를 고정하고 7~8월을 평가한다.',
function('Analysis/scripts/a02_transition_forecast.py','matrix','transition_matrix')+'\n\n'+'''
feature_sets={"달력":[],"달력+직전 평균":["past_level"],"달력+평균+구간 변화":["past_level","prior_slot_trend"]}
fixed_cuts=[.32418130623928276,.23541496682336432,.2580651743247188]
risk=triples.loc[triples.past_production.eq(0)].copy()
risk["event"]=risk.transition.eq("zero_to_positive").astype(int)
train=risk.loc[risk.month.le(6)];test_transition=risk.loc[risk.month.ge(7)]
rows=[]
for (name,features_transition),cut in zip(feature_sets.items(),fixed_cuts):
    transition_clf=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=1,learning_rate=.05,early_stopping=False,random_state=3203)
    transition_clf.fit(transition_matrix(train,features_transition),train.event)
    prob=transition_clf.predict_proba(transition_matrix(test_transition,features_transition))[:,1]
    y=test_transition.event.to_numpy();alarm_transition=prob>=cut
    tp=int(((y==1)&alarm_transition).sum());fp=int(((y==0)&alarm_transition).sum())
    rows.append({"입력":name,"시간 수":len(y),"실제 전환":int(y.sum()),"TP":tp,"FP":fp,"FN":int(y.sum())-tp,
                 "정밀도":tp/(tp+fp),"재현율":tp/y.sum(),"AP":average_precision_score(y,prob),"고정 경계":cut})
transition_results=pd.DataFrame(rows);show_table(transition_results,"14_생산전환사전구분")
fig,ax=plt.subplots(figsize=(9,4));x=np.arange(3)
ax.bar(x-.18,transition_results.TP,.36,label="적중",color=COLORS[0]);ax.bar(x+.18,transition_results.FP,.36,label="오탐",color=COLORS[1])
ax.axhline(test_transition.event.sum(),ls="--",color="gray",label="실제 전환 전체")
ax.set_xticks(x,transition_results["입력"]);ax.set_ylabel("시간 수");ax.legend();show_figure("14_생산전환탐지")
''','617시간 중 실제 전환은 52시간이다. 달력만 사용할 때 TP 35·FP 51에서 직전 평균 추가 시 TP 50·FP 25로 개선된다. 구간 변화 추가는 적중·오탐 수가 같아 큰 추가 이득은 확인되지 않는다. 이는 생산 전환의 정보 가치 검토이며 다음 시간 전력값의 예측 성능과는 구분한다.')
weather_function=function('Analysis/scripts/a03_conditioned_weather.py','association','weather_association')
weather_adjusted=function('Analysis/scripts/a03_weather_diagnostics.py','adjusted','temperature_adjusted')
section('## 3.3 날씨 관계를 조건 안에서 검토', '네 기상변수가 모두 있는 평일의 동일 기록에서 단순 순위상관과 달력·생산·다른 기상정보를 고려한 잔여 관계를 비교한다.',weather_function+'\n\n'+weather_adjusted+'\n\n'+'''
weather_data=data.assign(production_positive=data["생산량"].gt(0).astype(int))
common_weather=weather_data.loc[weather_data.weekend.eq(0)].dropna(subset=WEATHER)
common_weather=common_weather.loc[common_weather.groupby(KEYS).date.transform("nunique").ge(3)]
weather_results=pd.DataFrame([weather_association(common_weather,f,adjustment,"date_hour") for f in WEATHER for adjustment in ["raw","joint"]])
show_table(weather_results,"15_동일행날씨관계")
chart=weather_results.pivot(index="variable",columns="adjustment",values="correlation").reindex(WEATHER)
fig,ax=plt.subplots(figsize=(8,4));chart.rename(columns={"raw":"단순 관계","joint":"조건 고려 후"}).plot.bar(ax=ax,color=COLORS[:2])
ax.axhline(0,color="gray",lw=.8);ax.set_ylabel("순위 관계");ax.tick_params(axis="x",rotation=0);show_figure("15_날씨관계조정")
aug_positive=weather_data.loc[weather_data.weekend.eq(0)&weather_data.month.eq(8)&weather_data.production_positive.eq(1)]
show_table(pd.DataFrame([{"비교":label,**temperature_adjusted(aug_positive,date_control=control)} for label,control in [("8월 양수 생산 평일",False),("동일 날짜 차이까지 고려",True)]]),"15_8월날짜조정")
''','동일한 평일 4,102시간에서 조건을 고려한 관계는 기온 −0.004·풍속 +0.046·습도 +0.100·강수량 −0.060으로 약하다. 8월 양수 생산 평일의 기온 관계도 날짜 차이를 고려하면 +0.302에서 +0.014로 약해진다. 날씨의 예측 가치를 단정하지 않고 관측을 마친 과거 기상 입력의 실제 오차 효과로 판단한다.')
terminal_function=function('Analysis/scripts/a04_terminal_unique.py','compare','terminal_compare',[("FEATURES + ['target_maximum']","TERMINAL_FEATURES + ['target_maximum']"),("enumerate(FEATURES)","enumerate(TERMINAL_FEATURES)")])
slot_function=function('Analysis/scripts/a04_maximum_diagnostics.py','slot_group_associations')
section('## 3.4 다음 최대전력과 마지막 15분 값', '직전 평균·생산·달력을 고려한 네 구간의 부분 순위 관계를 비교하고, 직전 최대값까지 조정한 관계와 엄격한 동일 평균 비교의 예외를 확인한다.',terminal_function+'\n\n'+slot_function+'\n\n'+'''
TERMINAL_FEATURES=["prior_last","prior_slot_trend","prior_last_above_mean","prior_maximum_excess"]
previous=data.set_index("timestamp").reindex(triples.timestamp-pd.Timedelta(hours=1))
triples["prior_second"]=previous["30분"].to_numpy();triples["prior_third"]=previous["45분"].to_numpy()
slot_results=pd.DataFrame(slot_group_associations(triples))
slot_max=slot_results.loc[slot_results.target.eq("target_maximum")&slot_results.feature.isin(["prior_first","prior_second","prior_third","prior_last"])]
show_table(slot_max,"16_직전구간관계")
unique_rows=[]
for period,g in [("전체",triples),("1~4월",triples.loc[triples.month.le(4)]),("5~6월",triples.loc[triples.month.between(5,6)]),("7~8월",triples.loc[triples.month.ge(7)])]:
    for adjustment in ["mean_maximum","exact_mean"]:
        unique_rows += [{"기간":period,**r} for r in terminal_compare(g,adjustment=adjustment) if r["feature"]=="prior_last"]
terminal_results=pd.DataFrame(unique_rows);show_table(terminal_results,"16_마지막값민감도")
fig,ax=plt.subplots(figsize=(8,4));ax.bar(["15분","30분","45분","60분"],slot_max.correlation,color=COLORS[0])
ax.axhline(0,color="gray",lw=.8);ax.set_ylabel("다음 최대값과 부분 순위 관계");show_figure("16_직전네구간")
''','마지막 값의 관계는 +0.434로 가장 크고 직전 최대값까지 고려해도 +0.409다. 기간별로 양의 방향이 유지된다. 그러나 정확히 같은 과거 평균을 맞춘 비교는 1,039시간만 남으며 초기 관계는 −0.009다. 모든 조건의 동일한 효과로 일반화하지 않는다. 실제 예측 이득은 다음 Modeling의 시간순 오차 비교로 판단한다.')

md('# 4. Modeling\n\n## 4.1 시간순 평가와 기본 모델 선택\n\n다음 한 시간의 네 구간 중 최대값을 목표로 사용한다. 달력은 목표 시간 정보, 생산·전력은 직전 관측까지의 정보다. 정확한 1·2·24·168시간 시차가 있는 공통 행으로 비교한다. 당시 과거 내부 검증에서 선택한 설정을 고정해 4·5·6월을 재학습·예측하며, Optuna 전체 탐색은 재실행하지 않는다.')
model_frame=function('Modeling/scripts/m01_prepare.py','load_frame','build_model_frame',[("c = contract or read_contract()","c = contract or FEATURE_CONTRACT"),("    source = ROOT / c[\"source\"]\n    assert sha(source) == c[\"source_sha256\"], \"Source changed: review contract before execution\"\n    raw = pd.read_csv(source, encoding=\"utf-8-sig\")","    raw = raw_data.copy()"),("WEATHER.items()","WEATHER_MAP.items()")])
code('FEATURE_CONTRACT = '+repr(config)+'\n\n'+function('Modeling/scripts/m01_prepare.py','profile')+'\n\n'+model_frame+'\n\n'+function('Modeling/scripts/m022_compare.py','make_model')+'\n\n'+'''_, _, _, model_frame = build_model_frame()
# 당시 중간 CSV의 숫자 읽기 경계를 재현해 트리의 경계 비교를 일치시킨다.
model_frame=pd.read_csv(io.StringIO(model_frame.to_csv(index=False)),parse_dates=["timestamp"])
model_frame["date"]=pd.to_datetime(model_frame["date"])
display(pd.DataFrame({"자료":["시차 1·2", "시차 1·2·24·168"],"시간 수":[int(model_frame.eligible_core.sum()),int(model_frame.eligible_common.sum())]}))''')
selected=readj('Modeling/tables/m02_rerun/selected_configurations.json')
md('### 모델별 재학습과 개발 평가\n\n각 평가 월 이전 자료에서만 학습한다. HGB·XGBoost·LightGBM·CatBoost·ExtraTrees와 고정 앙상블을 동일한 행에서 비교한다. 아래 표·그림은 이번 실행에서 생성한 예측의 지표다. 과거 초기 선형·SVR 비교는 저장 예측을 대조 자료로 함께 제시한다.')
code('SELECTED_CONFIGS = '+repr(selected)+'\n\n'+function('Modeling/scripts/regime_followup.py','metrics')+'\n\n'+'''
basic_features=["hour_sin","hour_cos","month","weekend"]+[f"dow_{i}" for i in range(7)]+["lag1_production","lag1_production_zero","lag1_mean","lag1_maximum","lag2_mean","lag2_maximum"]
common_frame=model_frame.loc[model_frame.eligible_common].copy()
development_parts=[]
for month,split in [(4,"dev_apr"),(5,"dev_may"),(6,"dev_jun")]:
    start=pd.Timestamp(2021,month,1)
    train_dev=common_frame.loc[common_frame.timestamp.lt(start)]
    valid_dev=common_frame.loc[common_frame.month.eq(month)]
    dev_predictions={"previous_hour":valid_dev.lag1_maximum.to_numpy()}
    for item in [c for c in SELECTED_CONFIGS if c["split"]==split and "parameters" in c]:
        fitted=make_model(item["model"],item["parameters"])
        fitted.fit(train_dev[basic_features],train_dev.target_maximum)
        dev_predictions[item["model"]]=fitted.predict(valid_dev[basic_features])
    for item in [c for c in SELECTED_CONFIGS if c["split"]==split and "members" in c]:
        dev_predictions[item["model"]]=np.column_stack([dev_predictions[n] for n in item["members"]])@np.asarray(item["weights"])
    for name,values in dev_predictions.items():
        p=valid_dev[["timestamp","date","month"]].copy();p["model"]=name;p["actual"]=valid_dev.target_maximum;p["prediction"]=values
        development_parts.append(p)
development_predictions=pd.concat(development_parts,ignore_index=True)
development_summary=[]
for name,g in development_predictions.groupby("model"):
    daily_errors=[]
    for date,day in g.groupby("date"):
        if len(day)==24:
            peak=day.loc[day.actual.eq(day.actual.max())];daily_errors.append(metrics(peak)["mae"])
    development_summary.append({"模型":name,"전체 MAE":metrics(g)["mae"],"일별 최대 시간 MAE":np.mean(daily_errors),"평가 시간":len(g)})
development_summary=pd.DataFrame(development_summary).sort_values("전체 MAE")
show_table(development_summary,"17_시간순모델비교")
fig,ax=plt.subplots(figsize=(10,4));development_summary.set_index("模型")[["전체 MAE","일별 최대 시간 MAE"]].plot.bar(ax=ax,color=COLORS[:2])
ax.set_xlabel("모델");ax.set_ylabel("MAE (기록값)");ax.tick_params(axis="x",rotation=30);show_figure("17_모델개발비교")
development_predictions.to_csv(OUTPUT/"개발_모델별_예측.csv",index=False,encoding="utf-8-sig")
initial_reference=pd.read_csv(PACKAGE/"reference/초기모델_개발예측.csv",encoding="utf-8-sig")
show_table(pd.DataFrame([{"초기 비교 모델":name,**metrics(g)} for name,g in initial_reference.groupby("model")]),"17_초기선형모델대조")
''')
md('**결과 해석**\n\n동일 2,184시간에서 HGB 전체 MAE는 6.738, 일별 최대 시간 MAE는 11.751이다. ExtraTrees는 전체 6.825·최대 시간 11.540이다. 전체와 최대 시간 오차를 함께 보는 원고의 선택 규칙에 따라 HGB를 유지한다. 단순 직전값보다 큰 개선을 확인했으며, 마지막 구간과 지속 상승 정보는 다음 절에서 별도로 활용한다. 초기 선형·SVR 대조 표는 당시 저장 예측의 재집계로 재학습 결과와 구분한다.')
md('## 4.2 지속 상승 정의와 선택적 HGB 보완\n\n1월에서 고정한 큰 변화 기준과 과거 6시간의 중앙값·IQR로 상승 시작을 정의한다. 미래 3시간은 정답 확인에만 사용하고 입력에는 넣지 않는다. 상태 경과시간은 분류기에서만 제거하고 회귀 B1에서는 유지한다. 아래 함수는 사건 정의와 과거 상태 구성에 필요한 계산이다.')
code('MODEL_SETTINGS = '+repr(settings)+'\n\n'+function('Modeling/scripts/regime_diagnose.py','label')+'\n\n'+function('Modeling/scripts/regime_followup.py','add_calendar')+'\n\n'+function('Modeling/scripts/regime_followup.py','extended','build_state_features',[("raw = pd.read_csv(ROOT / \"data/origin/okm_augumented_2021.csv\", encoding=\"utf-8-sig\")","raw = raw_data.copy()")])+'\n\n'+function('Modeling/scripts/regime_forecast.py','model','classification_model')+'\n\n'+function('Modeling/scripts/regime_forecast.py','threshold','choose_alarm_threshold')+'\n\n'+'''
z, events=build_state_features(MODEL_SETTINGS)
show_table(events.groupby(["month","direction"]).agg(시작수=("event_id","size"),지속3시간=("sustained3",lambda x:int(x.eq(True).sum()))).reset_index(),"18_지속사건정의")
print("상승 기준:",MODEL_SETTINGS["onset_definition"]["thresholds"]["up"],"3시간 유지 기준:",MODEL_SETTINGS["onset_definition"]["thresholds"]["up"]/2)
''')
md('**정의와 해석**\n\n시간 평균이 직전 평균과 과거 6시간 중앙값보다 각각 59.5 이상 높고, 중앙값 대비 상승폭이 과거 IQR의 1.5배 이상인 시작을 후보로 삼는다. 이후 3시간이 기존 기준선보다 29.75 이상 높게 유지되면 지속 상승이다. 계속되는 같은 방향 상태는 새로운 시작으로 중복 집계하지 않는다. 분류 정답은 t+2에 확인되므로 학습 경계 이전에 확인된 정답만 사용한다.')
md('### 과거 OOF 경계 산정과 고정 모델 학습\n\n4·5·6월 예측은 각각 그 월 이전의 확인된 정답으로만 학습한다. 이 OOF 예측에서 음성 오경보율 1% 이내의 경계를 정한다. 1~6월 분류와 2~6월 회귀를 고정해 7~8월의 입력만 갱신한다.')
code('''features=MODEL_SETTINGS["classification_features"]
ct=z.loc[z.eligible_onset & z.persistence_known & z.label_confirmed_at.lt("2021-07-01")].copy()
oof_parts=[]
for month in [4,5,6]:
    boundary_month=pd.Timestamp(2021,month,1)
    tr=ct.loc[ct.label_confirmed_at.lt(boundary_month)];te=ct.loc[ct.month.eq(month)]
    fit=classification_model("logistic",MODEL_SETTINGS)
    fit.fit(tr[features],tr.sustained_onset.eq(1).astype(int))
    p=te[["timestamp","label_confirmed_at","sustained_onset"]].copy()
    p["score"]=fit.predict_proba(te[features])[:,1];p["event"]=te.sustained_onset.eq(1).astype(int);oof_parts.append(p)
oof=pd.concat(oof_parts)
alarm_boundary,oof_tp,oof_fp,fp_cap=choose_alarm_threshold(oof.event.to_numpy(int),oof.score.to_numpy(float))
clf=classification_model("logistic",MODEL_SETTINGS);clf.fit(ct[features],ct.sustained_onset.eq(1).astype(int))
late=z.loc[z.eligible_onset & z.month.ge(7)].copy()
score=clf.predict_proba(late[features])[:,1]
base=MODEL_SETTINGS["regression_features"]["B0"];expanded=MODEL_SETTINGS["regression_features"]["B1"]
extras=[n for n in expanded if n not in base and not n.endswith("classifier_available")]
zh=pd.read_csv(io.StringIO(z.loc[z.month.le(6)].to_csv(index=False)),parse_dates=["timestamp","label_confirmed_at"])
zz=pd.concat([zh,z.loc[z.month.ge(7)]],ignore_index=True)
common=zz.loc[zz.eligible_onset,["timestamp","label_confirmed_at","month","profile","sustained_onset","persistence_known"]+extras]
common=common.drop(columns="month").merge(model_frame.loc[model_frame.eligible_common,["timestamp","target_maximum"]+base],on="timestamp",validate="one_to_one")
for direction,sign in [("up",1),("down",-1)]:
    availability={}
    for month in range(2,9):
        mature=ct.loc[ct.label_confirmed_at.lt(pd.Timestamp(2021,min(month,7),1))]
        positive=mature.loc[mature.sustained_onset.eq(sign)]
        availability[month]=int(len(positive)>=8 and positive.date.nunique()>=5)
    common[direction+"_classifier_available"]=common.month.map(availability)
rt=common.loc[common.month.between(2,6)&common.persistence_known&common.label_confirmed_at.lt("2021-07-01")]
rt=pd.read_csv(io.StringIO(rt.to_csv(index=False)),parse_dates=["timestamp","label_confirmed_at"])
test=common.loc[common.month.ge(7)].copy()
predictions={}
for variant,cols in [("B0",base),("B1_all",expanded)]:
    reg=HistGradientBoostingRegressor(**MODEL_SETTINGS["hgb_parameters"])
    reg.fit(rt[cols],rt.target_maximum);predictions[variant]=reg.predict(test[cols])
scores_index=pd.Series(score,index=late.timestamp)
alarm=scores_index.loc[test.timestamp].to_numpy()>=alarm_boundary
predictions["G1_noage"]=np.where(alarm,predictions["B1_all"],predictions["B0"])
replay=pd.concat([test[["timestamp","sustained_onset"]].assign(variant=name,actual=test.target_maximum,prediction=pred) for name,pred in predictions.items()],ignore_index=True)
show_table(pd.DataFrame({"항목":["분류 학습","회귀 학습","7~8월 분류 예측","공통 회귀 시간","OOF 경계","OOF 적중","OOF 오경보"],
    "값":[len(ct),len(rt),len(late),len(test),alarm_boundary,oof_tp,oof_fp]}),"19_모델학습경계")
''')
md('**학습 결과**\n\n분류 4,336시간·회귀 3,598시간을 학습하고 분류 1,428개·회귀 3방법×1,344개 예측을 생성한다. OOF 경계는 약 0.4555다. 평가 월의 정답으로 경계를 변경하거나 7월 자료로 재학습하지 않는다. 경고가 있을 때만 B1을 선택하며 나머지는 B0를 사용한다.')
section('### 선택적 보완의 오차와 탐지 결과', '동일한 1,344시간에서 전체·상승 시작·일별 최대 시간 오차를 비교한다. 정답이 확인된 행에서만 분류 정밀도·재현율을 계산한다.',function('Modeling/scripts/regime_followup.py','detection',replacements=[('    from sklearn.metrics import average_precision_score\n','')])+'\n\n'+'''
classification=late[["timestamp","sustained_onset","persistence_known"]].copy()
classification["score"]=score;classification["alarm"]=(score>=alarm_boundary).astype(int)
classification["event"]=classification.sustained_onset.eq(1).astype(int)
class_common=classification.loc[classification.timestamp.isin(test.timestamp)&classification.persistence_known]
det=detection(class_common);show_table(pd.DataFrame([det]),"20_공통범위분류성능")
regression_rows=[]
for name,g in replay.groupby("variant"):
    for condition,p in [("전체",g),("상승 시작",g.loc[g.sustained_onset.eq(1)])]:
        regression_rows.append({"방법":name,"조건":condition,**metrics(p)})
    daily=[]
    for date,day in g.groupby(g.timestamp.dt.normalize()):
        if len(day)==24: daily.append(metrics(day.loc[day.actual.eq(day.actual.max())]))
    regression_rows.append({"방법":name,"조건":"일별 최대 시간", "hours":sum(d["hours"] for d in daily),
                            **{k:np.mean([d[k] for d in daily]) for k in ["mae","under","over","bias"]}})
regression_summary=pd.DataFrame(regression_rows);show_table(regression_summary,"20_회귀오차")
fig,axes=plt.subplots(1,2,figsize=(12,4))
regression_summary.pivot(index="조건",columns="방법",values="mae").plot.bar(ax=axes[0],color=[COLORS[0],"#999999",COLORS[1]])
axes[0].set_ylabel("MAE");axes[0].tick_params(axis="x",rotation=0)
axes[1].bar(["적중","오경보","미탐지"],[det["tp"],det["fp"],det["fn"]],color=[COLORS[0],COLORS[1],"#999999"])
axes[1].set_ylabel("시간 수");show_figure("20_성능과탐지")
replay.to_csv(OUTPUT/"모델별_테스트예측.csv",index=False,encoding="utf-8-sig")
final_predictions=test[["timestamp","target_maximum"]].rename(columns={"target_maximum":"actual_maximum"}).copy()
final_predictions["baseline_prediction"]=predictions["B0"];final_predictions["supplement_prediction"]=predictions["B1_all"]
final_predictions["final_prediction"]=predictions["G1_noage"];final_predictions["alarm"]=alarm.astype(int)
final_predictions.to_csv(PACKAGE/"테스트데이터_예측결과.csv",index=False,encoding="utf-8-sig")
''','상승 시작 MAE는 20.037→18.205(9.1% 감소), 과소예측은 19.419→16.342(15.8% 감소)다. 일별 최대 시간 MAE는 13.179→12.574, 전체는 6.335→6.317이다. 7시간만 보완하고 TP 7·FP 0·FN 6이므로 정밀도 100%, 재현율 53.8%다. 경과시간 제거는 이미 후반기 결과를 본 뒤의 수정이므로 이 개선은 탐색적 재평가로 해석한다.')
section('## 4.3 오류 해석·사례와 현장 활용', 'Logistic의 표준화 입력×계수를 더해 로짓을 정확히 분해한다. 적중·미탐지 사례는 개선폭과 실제 최대값의 중앙 순위로 선정한다. 사례를 대표 성능으로 일반화하지 않는다.', '''
scaler,linear=clf[0],clf[1]
contrib=scaler.transform(late[features])*linear.coef_[0]
logit=float(linear.intercept_[0])+contrib.sum(axis=1)
np.testing.assert_allclose(logit,clf.decision_function(late[features]),atol=1e-10)
groups={"달력":[n for n in features if n in MODEL_SETTINGS["calendar_features"]],"생산":["prior_production","prior_production_delta"],"상태":["prior_state_left_censored","prior_state_direction"]}
groups["과거 전력"]=[n for n in features if n not in sum(groups.values(),[])]
explained=classification.copy()
for name,names in groups.items(): explained[name]=contrib[:,[features.index(n) for n in names]].sum(axis=1)
explained["절편"]=float(linear.intercept_[0]);explained["로짓"]=logit
common_explained=explained.loc[explained.timestamp.isin(test.timestamp)&explained.persistence_known]
contribution_summary=pd.DataFrame([{"조건":name,"시간 수":len(g),**{col:g[col].mean() for col in list(groups)+["절편","로짓"]}}
    for name,g in [("적중 상승",common_explained.loc[common_explained.event.eq(1)&common_explained.alarm.eq(1)]),
                   ("미탐지 상승",common_explained.loc[common_explained.event.eq(1)&common_explained.alarm.eq(0)])]])
show_table(contribution_summary,"21_로짓기여")
b0=replay.loc[replay.variant.eq("B0")].set_index("timestamp")
g1=replay.loc[replay.variant.eq("G1_noage")].set_index("timestamp")
event_errors=common_explained.loc[common_explained.event.eq(1)].copy()
event_errors["actual"]=b0.loc[event_errors.timestamp,"actual"].to_numpy()
event_errors["gain"]=(b0.actual-b0.prediction).abs().loc[event_errors.timestamp].to_numpy()-(g1.actual-g1.prediction).abs().loc[event_errors.timestamp].to_numpy()
fig,axes=plt.subplots(1,2,figsize=(13,4));case_rows=[]
for ax,(name,pool,sortcol) in zip(axes,[("적중·개선",event_errors.loc[event_errors.alarm.eq(1)&event_errors.gain.gt(0)],"gain"),("미탐지",event_errors.loc[event_errors.alarm.eq(0)],"actual")]):
    pool=pool.loc[pool.timestamp.map(lambda t:pd.date_range(t-pd.Timedelta(hours=4),t+pd.Timedelta(hours=5),freq="h").isin(b0.index).all())].sort_values([sortcol,"timestamp"])
    chosen=pool.iloc[len(pool)//2];t=chosen.timestamp;times=pd.date_range(t-pd.Timedelta(hours=4),t+pd.Timedelta(hours=5),freq="h")
    ax.plot(times,b0.loc[times,"actual"],color="black",marker="o",label="실제 최대")
    ax.plot(times,b0.loc[times,"prediction"],color=COLORS[0],ls="--",label="B0")
    ax.plot(times,g1.loc[times,"prediction"],color=COLORS[1],label="선택적 보완")
    ax.axvspan(t,t+pd.Timedelta(hours=2),color="gray",alpha=.15);ax.axvline(t-pd.Timedelta(hours=1),color="gray",ls=":")
    ax.set_title(f"{name}: {t:%m/%d %H시}");ax.tick_params(axis="x",rotation=30,labelsize=8);ax.legend(fontsize=8)
    case_rows.append({"사례":name,"시각":t,"실제":b0.loc[t,"actual"],"B0":b0.loc[t,"prediction"],"보완":g1.loc[t,"prediction"],"선정 후보":len(pool)})
show_figure("21_개선과미탐지사례");show_table(pd.DataFrame(case_rows),"21_사례선정")
calendar_alarm=class_common.timestamp.dt.dayofweek.eq(0)&class_common.timestamp.dt.hour.eq(8)
calendar_control=class_common.assign(alarm=calendar_alarm.astype(int))
show_table(pd.DataFrame([{"방법":"고정 분류",**detection(class_common)},{"방법":"사후 월요일 08시 규칙",**detection(calendar_control)}]),"21_사후달력규칙")
''','적중·미탐지의 과거 전력 평균 로짓 기여는 13.86·4.16이다. 기여는 확률의 퍼센트가 아니며 인과효과도 아니다. 7/12의 실제 최대 198은 181.13→192.52로 개선되지만 8/13의 실제 207은 경고 없이 182.40으로 과소예측한다. 경고는 모두 월요일 08시에 집중되고 사후 주간 규칙도 TP 8·FP 1로 비슷하다. 무경고를 안전 판정으로 사용하지 않는다. 실제 가동 조정·전력·비용 절감은 검증된 성과가 아닌 활용 제안이다.')
section('### 재현 확인과 제출 산출물', '현재 원자료 재학습 예측을 원고에 사용한 고정 결과와 대조한다. 예측 파일·표·그림은 이 노트북 실행으로 생성된다.', '''
reference_class=pd.read_csv(PACKAGE/"reference/고정_분류예측.csv",encoding="utf-8-sig",parse_dates=["timestamp"])
reference_class=reference_class.loc[reference_class.method.eq("noage")].set_index("timestamp").loc[late.timestamp]
np.testing.assert_allclose(score,reference_class.score,atol=1e-8,rtol=1e-9)
np.testing.assert_array_equal(score>=alarm_boundary,reference_class.alarm)
reference_reg=pd.read_csv(PACKAGE/"reference/고정_회귀예측.csv",encoding="utf-8-sig",parse_dates=["timestamp"])
max_difference=0.
for variant,pred in predictions.items():
    ref=reference_reg.loc[reference_reg.variant.eq(variant)].set_index("timestamp").loc[test.timestamp]
    np.testing.assert_allclose(pred,ref.prediction,atol=1e-8,rtol=1e-9)
    max_difference=max(max_difference,float(np.max(abs(pred-ref.prediction.to_numpy()))))
assert len(ct)==4336 and len(rt)==3598 and len(late)==1428 and len(test)==1344
assert det["tp"]==7 and det["fp"]==0 and det["fn"]==6
verification=pd.DataFrame({"검증":["원자료 정상 시간","분류 예측 대조","회귀 예측 대조","회귀 최대 차이","제출 예측 파일 행"],
    "결과":[len(data),len(score),len(replay),max_difference,len(final_predictions)]})
show_table(verification,"22_재현검증")
print("제출 예측 파일:","테스트데이터_예측결과.csv")
print("전체 계산 완료. 표·그림은 output/, 전처리 자료는 data/processed/에 저장했습니다.")
''','현재 방법의 원자료 → 전처리 → 시차·정답 구성 → 학습·OOF 경계 → 예측을 재실행해 기존 분류 1,428개·회귀 4,032개 예측과 대조한다. 저장 참조 예측은 검증용이며 현재 모델 학습의 입력이 아니다. 신규 환경 설치나 과거 Optuna 전체 재탐색을 완료했다는 뜻은 아니다.')

# Preserve the rejected notebook and all copied scripts outside the submission.
backup=ROOT/'tmp'/'submission_before_readable_rebuild'
backup.mkdir(exist_ok=True)
for name in ['EDA','Analysis','Modeling','Report','실행결과','tmp','통합_분석.ipynb','README.md','requirements.txt','submission_manifest.json']:
    old=DEST/name
    if old.exists():
        target=backup/name
        if target.exists():target=backup/(datetime.datetime.now().strftime('%H%M%S')+'_'+name)
        shutil.move(str(old),str(target))
(DEST/'data/raw').mkdir(parents=True,exist_ok=True)
shutil.copy2(ROOT/'data/origin/okm_augumented_2021.csv',DEST/'data/raw/okm_augumented_2021.csv')
# The previous copied data contains reference tables and is no longer needed.
for item in list((DEST/'data').iterdir()):
    if item.name not in ['raw','processed']:
        target=backup/('old_data_'+item.name)
        if not target.exists():shutil.move(str(item),str(target))
(DEST/'reference').mkdir(exist_ok=True)
import pandas as pd
for source,name in [('Modeling/tables/regime_age_ablation/followup_classification.csv','고정_분류예측.csv'),('Modeling/tables/regime_age_ablation/followup_predictions.csv','고정_회귀예측.csv')]:
    shutil.copy2(ROOT/source,DEST/'reference'/name)
p=pd.read_csv(ROOT/'Modeling/tables/m02_rerun/predictions.csv',encoding='utf-8-sig')
p.loc[p.model.isin(['Ridge_initial','ElasticNet_initial','SVR_initial']),['model','actual','prediction']].to_csv(DEST/'reference/초기모델_개발예측.csv',index=False,encoding='utf-8-sig')
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python','version':'3.13.1'}})
for c in nb.cells:
    if c.cell_type=='code':
        tree=ast.parse(c.source)
        assert '__name__' not in c.source and '%%' not in c.source and 'MODULES[' not in c.source
nbformat.write(nb,DEST/'통합_분석.ipynb')
packages=['numpy','pandas','scipy','matplotlib','scikit-learn','xgboost','lightgbm','catboost','nbformat','nbclient','ipykernel','jupyter-client','nbconvert','jupyterlab']
(DEST/'requirements.txt').write_text('# Python 3.13\n'+'\n'.join(f'{n}=={importlib.metadata.version(n)}' for n in packages)+'\n',encoding='utf-8')
(DEST/'README.md').write_text('''# 제출 파일 실행 안내

## 파일 구성

- `통합_분석.ipynb`: 데이터 점검·전처리·EDA·Analysis·Modeling 계산, 표와 그림, 해석.
- `data/raw/okm_augumented_2021.csv`: 수정하지 않은 제공 원자료.
- `requirements.txt`: 검증에 사용한 Python 3.13 패키지 버전.
- `reference/`: 과거 초기 모델 비교와 현재 고정 예측 대조에 필요한 CSV. 현재 모델의 학습 입력은 원자료에서 구성한다.
- `테스트데이터_예측결과.csv`: 7~8월 공통 1,344시간의 최종 예측. 전체 실행 때 생성한다.
- `data/processed/`, `output/`: 노트북이 생성한 전처리 자료·결과표·그림.

제출 실행에 별도 `.py` 파일은 사용하지 않는다. `ipynb 예시.ipynb`는 사용자 제공 구성 참고 자료다.

## 가상환경과 커널 설정 (Windows PowerShell)

제출 폴더에서 Python 3.13으로 실행한다.

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe -m ipykernel install --user --name manufacturing-submit --display-name "Python (제조데이터 제출)"
.\\.venv\\Scripts\\python.exe -m jupyterlab
```

Jupyter에서 노트북을 열고 **Python (제조데이터 제출)** 커널을 선택한 뒤 **커널 재시작 및 전체 실행**을 수행한다. VS Code에서는 우측 상단 커널 선택에서 같은 환경을 선택한다.

`requirements.txt`는 패키지를 설치하며 노트북 커널을 자동 선택하지 않는다. 커널 등록은 한 번 수행하며 이후 같은 커널을 선택해 실행한다. 가상환경 자체를 제출할 필요는 없다.

등록 후 화면 없이 전체 실행하고 출력까지 저장하려면 다음 표준 명령을 사용한다. 별도 실행기 파일은 필요 없다.

```powershell
.\\.venv\\Scripts\\python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=manufacturing-submit --ExecutePreprocessor.timeout=1800 통합_분석.ipynb
```

## 분석 범위와 해석

원자료를 매번 읽고 1~8월·정상 0~23시를 필터링한다. 시간 오류 48행은 결과에 따로 기록하며 전력 0·생산량 0·큰 값은 유지한다. 이후 계산은 이 자료의 정확한 과거 시차를 사용한다. 기상 결측은 관련 계산에서만 제외한다.

과거 시간순 모델 비교는 당시 내부 검증에서 선택한 모델 설정을 고정해 재학습한다. Optuna 전체 재탐색은 수행하지 않는다. 초기 선형·SVR 비교는 당시 저장 예측의 재집계다. 최종 Logistic 및 HGB B0/B1은 원자료로 재학습하고 OOF 경계도 다시 계산한다.

7~8월은 이미 관찰한 자료의 탐색적 재평가이며 독립 최종시험으로 해석하지 않는다. 마지막 셀에서 고정 참조 예측과 수치 일치를 검증한다.

정규 CPython 3.13의 기존 설치 환경에서 전체 실행을 검증한다. 별도의 새 가상환경 설치 검증과 구분한다.
''',encoding='utf-8')
print(json.dumps({'cells':len(cells),'code_cells':sum(c.cell_type=='code' for c in cells),'backup':str(backup)},ensure_ascii=False))
