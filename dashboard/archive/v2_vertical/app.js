'use strict';
const DATA=JSON.parse(document.getElementById('demo-data').textContent);
const HOUR=3600000,QUARTER=900000,$=id=>document.getElementById(id);
const col=Object.fromEntries(DATA.columns.map((name,i)=>[name,i]));
const slots=[],hourRows=new Map();
const hourFirst=new Map();
for(const row of DATA.raw){
  const h=row[col['시간']];if(!Number.isInteger(h)||h<0||h>23)continue;
  const d=String(row[col['날짜']]);const t=Date.UTC(+d.slice(0,4),+d.slice(4,6)-1,+d.slice(6,8),h);
  hourRows.set(t,row);
  ['15분','30분','45분','60분'].forEach((k,i)=>slots.push({t:t+(i+1)*QUARTER,v:row[col[k]],h:t,s:i}));
}
slots.sort((a,b)=>a.t-b.t);
slots.forEach((s,i)=>{if(s.s===0)hourFirst.set(s.h,i);});
const forecast=new Map(DATA.forecasts.map(r=>[r[0],{
  t:r[0],p:r[1],lo:r[2],hi:r[3],score:r[4],alarm:r[5],routed:r[6],b0:r[7],b1:r[8],
  down:r[9],downAlarm:r[10],mode:r[11],drivers:r.slice(12,16)
}]));
const contexts=new Map(DATA.contexts.map(r=>[r[0],{mean:r[1],production:r[2],delta:r[3],median:r[4],range:r[5],lastMinusMean:r[6]}]));
const eventMap=new Map(DATA.events.map(r=>[r[0],{direction:r[1],delta:r[2]}]));
const pad=n=>String(n).padStart(2,'0');
const date=t=>new Date(t);
const hm=t=>pad(date(t).getUTCHours())+':'+pad(date(t).getUTCMinutes());
const ymd=t=>date(t).toISOString().slice(0,10);
const fmt=n=>n==null?'—':Number(n).toLocaleString('ko-KR',{maximumFractionDigits:2});
const pct=n=>(n*100).toFixed(1)+'%';
function lowerBound(t){let l=0,r=slots.length;while(l<r){const m=(l+r)>>1;if(slots[m].t<t)l=m+1;else r=m;}return l;}
let index=0,playing=false,lastTick=0,speed=1000,windowHours=6,plotPoints=[];
let animStart=0,fromIndex=0;
const canvas=$('chart'),ctx=canvas.getContext('2d');
const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
function seek(t){playing=false;index=Math.min(slots.length-1,lowerBound(t));fromIndex=index;animStart=0;render();}
function advance(){if(index>=slots.length-1){playing=false;render();return;}fromIndex=index;index++;animStart=performance.now();render();}
function activeForecast(){
  const t=Math.floor(slots[index].t/HOUR)*HOUR;
  const f=forecast.get(t)||(index===slots.length-1?forecast.get(slots[index].h):null);
  // The dataset starts without any previous hour. This is an observation reference,
  // never an invented model prediction or uncertainty interval.
  if(!f||f.p==null)return {t,p:slots[index].v,lo:null,hi:null,score:null,alarm:0,routed:0,b0:null,b1:null,down:null,downAlarm:0,mode:'warmup',drivers:[null,null,null,null]};
  return f;
}
function observeHour(t,now){
  if(t+HOUR>now)return null;
  const row=hourRows.get(t);return row?Math.max(...['15분','30분','45분','60분'].map(c=>row[col[c]])):null;
}
function completedDayPeak(now){
  const start=Date.UTC(date(now).getUTCFullYear(),date(now).getUTCMonth(),date(now).getUTCDate());
  let peak=null;
  for(let i=index;i>=0&&slots[i].t>=start;i--)peak=peak==null?slots[i].v:Math.max(peak,slots[i].v);
  return peak;
}
const modeName={official:'기존 평가 결과',backcast:'과거 적용 시연',extension:'추가 시간 시연',reference:'직전 시간 기준',warmup:'초기 관측 기준'};
function render(){
  const s=slots[index],previous=slots[index-1],contiguous=previous&&s.t-previous.t===QUARTER;
  const delta=contiguous?s.v-previous.v:null,f=activeForecast(),c=contexts.get(f.t);
  const d=date(s.t),weekday=['일','월','화','수','목','금','토'][d.getUTCDay()];
  $('clockDate').textContent=ymd(s.t)+' ('+weekday+')';$('clockTime').textContent=hm(s.t);$('date').value=ymd(s.t);
  $('current').textContent=fmt(s.v);$('currentSub').textContent='관측 완료 '+hm(s.t)+' · 원자료 단위';
  $('delta').textContent=delta==null?'—':(delta>0?'+':'')+fmt(delta);
  $('delta').className=delta>0?'red':delta<0?'blue':'';
  $('deltaSub').textContent=delta==null?(previous?'이전 관측과 자료 공백':'첫 관측'):delta>0?'상승 · 관측된 변화':delta<0?'하락 · 관측된 변화':'유지 · 직전 관측과 동일';
  $('prediction').textContent=fmt(f.p);
  $('predictionSub').textContent=hm(f.t)+'–'+hm(f.t+HOUR)+' · '+modeName[f.mode];
  let alarmLabel=f.alarm&&f.downAlarm?'양방향 경고':f.alarm?'지속 상승 경고':f.downAlarm?'지속 하락 경고':f.score==null?'관측 기준':'경고 없음';
  $('alarm').textContent=alarmLabel;$('alarm').className='status '+(f.alarm?'red':f.downAlarm?'blue':'');
  $('alarmSub').textContent=f.downAlarm?'하락은 개발 모델의 실험적 시연':f.alarm?'선택 보완 활성 · 보완 HGB 적용':f.score==null?'초기·공백 구간 · 모델 경고 없음':'상승 경계 0.4555 · 무경고는 안전 판정 아님';
  $('chartCaption').textContent=(previous&&!contiguous?'자료 공백을 건너뛰었습니다. 이전 관측을 유지하고 공백 경계를 표시합니다.':'날짜가 바뀌어도 연속 재생 · '+(speed/1000)+'초 = 15분 관측')+' · '+modeName[f.mode];
  $('routingTitle').textContent=f.alarm?'지속 상승 경고 → 보완 HGB 적용':'지속 상승 경고가 있을 때만 예측을 보완';
  $('routingText').textContent=f.mode==='warmup'?'첫 시간에는 과거 입력이 없습니다. 최근 관측을 기준선으로 보여주며 모델 예측구간을 만들지 않습니다.':f.mode==='reference'?'과거 입력이 부족해 직전 시간 최대값을 참고 기준으로 표시합니다. 필요한 관측이 쌓이면 모델 예측으로 전환합니다.':f.alarm?'과거 전력과 달력·생산 조건으로 지속 상승을 판정했습니다. 평상시 기본 모델을 유지하다 이 시간에만 상태 입력을 보완한 모델로 전환합니다.':'기본 HGB로 예상 최대값을 확인합니다. 전력 수준이 크게 올라 유지되기 시작할 경고가 발생하면 해당 시간에만 보완 HGB로 전환합니다.';
  $('basePrediction').textContent=fmt(f.b0);$('selectedPrediction').textContent=fmt(f.p);$('dayPeak').textContent=fmt(completedDayPeak(s.t));
  $('contextTime').textContent='직전 완료 시간 '+hm(f.t-HOUR);
  $('priorMean').textContent=fmt(c?.mean);$('pastMedian').textContent=fmt(c?.median);$('production').textContent=fmt(c?.production);
  $('calendar').textContent=weekday+'요일 · '+(d.getUTCDay()===0||d.getUTCDay()===6?'주말':'평일')+' '+hm(f.t);
  const drivers=f.drivers.map((v,i)=>({name:['과거 전력','달력 조건','생산량','과거 상태'][i],v}));
  const scale=Math.max(1,...drivers.map(x=>Math.abs(x.v??0)));
  $('driverBars').innerHTML=drivers.every(x=>x.v==null)?'<p class="note">과거 입력이 쌓이면 모델의 입력군별 판단 기여를 표시합니다.</p>':drivers.sort((a,b)=>Math.abs(b.v??0)-Math.abs(a.v??0)).map(x=>'<div class="driver-row"><span>'+x.name+'</span><div class="driver-track"><div class="driver-fill" style="width:'+((Math.abs(x.v??0)/scale)*100).toFixed(1)+'%;background:'+(x.v<0?'var(--blue)':'var(--red)')+'"></div></div><span class="amount '+(x.v<0?'blue':'red')+'">'+(x.v>0?'+':'')+fmt(x.v)+'</span></div>').join('');
  const weather=hourRows.get(f.t-HOUR);
  $('temperature').textContent=weather&&weather[col['기온']]!=null?fmt(weather[col['기온']])+' °C':'—';
  $('humidity').textContent=weather&&weather[col['습도']]!=null?fmt(weather[col['습도']])+'%':'—';
  $('wind').textContent=weather?fmt(weather[col['풍속']]):'—';$('rain').textContent=weather?fmt(weather[col['강수량']]):'—';
  $('upScore').textContent=f.score==null?'—':f.score.toFixed(3)+' / 1';$('downScore').textContent=f.down==null?'—':f.down.toFixed(3)+' / 1';
  const completedHour=Math.floor(s.t/HOUR)*HOUR-HOUR,e=eventMap.get(completedHour);
  $('observedEvent').textContent=e?(e.direction>0?'급등 관측 ':'급락 관측 ')+(e.delta>0?'+':'')+fmt(e.delta)+' · 시간 평균':'최근 완료 시간의 전환 확인';
  const rows=[];
  for(let t=f.t;rows.length<4&&t>=f.t-12*HOUR;t-=HOUR){
    const p=forecast.get(t);if(!p)continue;const actual=observeHour(t,s.t);
    rows.push('<tr><td>'+hm(t)+'–'+hm(t+HOUR)+'</td><td class="red">'+fmt(p.p)+'</td><td>'+(actual==null?'관측 중':fmt(actual))+'</td><td class="'+(p.alarm?'red':p.downAlarm?'blue':'')+'">'+(p.alarm?'상승':p.downAlarm?'하락':'없음')+'</td></tr>');
  }
  $('history').innerHTML=rows.join('');$('play').textContent=playing?'일시정지':'재생 시작';
  $('playState').textContent=playing?'전체 기간 연속 재생 중':'일시정지 · 15분씩 이동 가능';
  $('scrub').value=index;$('liveStatus').textContent=ymd(s.t)+' '+hm(s.t)+', 전력값 '+fmt(s.v)+', '+alarmLabel;
  draw(1);
}
function resize(){const b=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(b.width*ratio);canvas.height=Math.round(b.height*ratio);ctx.setTransform(ratio,0,0,ratio,0,0);draw(1);}
function draw(progress){
  const w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;
  ctx.clearRect(0,0,w,h);const mobile=w<600,left=mobile?39:53,right=mobile?12:24,top=32,bottom=43,py=h-bottom;
  const futureCount=4,pastCount=windowHours*4,total=pastCount+futureCount,plotWidth=w-left-right;
  const previous=slots[fromIndex],current=slots[index],f=activeForecast();
  const animate=progress<1&&previous&&current.t-previous.t===QUARTER;
  const visualIndex=animate?fromIndex+(index-fromIndex)*progress:index;
  const startIndex=visualIndex-pastCount,x=i=>left+(i-startIndex)/total*plotWidth;
  const boundary=x(visualIndex),lastX=w-right;
  // Fixed data scale avoids making midnight look like a chart reset.
  const max=300,min=-20,y=v=>py-(v-min)/(max-min)*(py-top);
  ctx.font='11px "Segoe UI","Malgun Gothic",sans-serif';ctx.lineWidth=1;
  for(let val=0;val<=300;val+=60){const yy=y(val);ctx.strokeStyle='#2a3b47';ctx.beginPath();ctx.moveTo(left,yy);ctx.lineTo(lastX,yy);ctx.stroke();ctx.fillStyle='#92a9b7';ctx.textAlign='right';ctx.fillText(String(val),left-8,yy+4);}
  const tickSlots=windowHours<=6?4:windowHours<=12?8:16;
  for(let i=Math.max(0,Math.ceil(startIndex));i<=index;i++){
    const s=slots[i];if(s.t%(tickSlots*QUARTER)!==0)continue;
    const xx=x(i);ctx.strokeStyle='#253641';ctx.beginPath();ctx.moveTo(xx,top);ctx.lineTo(xx,py);ctx.stroke();
    ctx.fillStyle='#92a9b7';ctx.textAlign='center';ctx.fillText(hm(s.t),xx,py+22);
    if(s.t%86400000===0){ctx.strokeStyle='#617787';ctx.setLineDash([2,4]);ctx.beginPath();ctx.moveTo(xx,top);ctx.lineTo(xx,py);ctx.stroke();ctx.setLineDash([]);ctx.font='10px "Segoe UI",sans-serif';ctx.fillText(ymd(s.t).slice(5),xx,py+35);ctx.font='11px "Segoe UI",sans-serif';}
  }
  ctx.fillStyle='#ff777807';ctx.fillRect(boundary,top,lastX-boundary,py-top);
  const first=Math.max(0,Math.floor(startIndex)-1);plotPoints=[];
  ctx.save();ctx.beginPath();ctx.rect(left,top,plotWidth,py-top);ctx.clip();
  // Past forecasts share the identical coordinates; a faint step line connects the
  // already-published hourly maxima, while the active forecast is emphasized.
  ctx.strokeStyle='#ff77787a';ctx.lineWidth=1.4;ctx.setLineDash([4,4]);
  const seen=new Set();
  for(let i=first;i<=index;i++){
    const ht=slots[i].h;if(seen.has(ht))continue;seen.add(ht);const p=forecast.get(ht);if(!p||p.p==null)continue;
    const firstIndex=hourFirst.get(ht),endIndex=firstIndex+3;
    ctx.beginPath();ctx.moveTo(x(firstIndex-1),y(p.p));ctx.lineTo(x(Math.min(endIndex,visualIndex)),y(p.p));ctx.stroke();
  }
  ctx.setLineDash([]);ctx.strokeStyle='#76e0bf';ctx.lineWidth=2.3;ctx.lineJoin='round';ctx.beginPath();let last=null;
  for(let i=first;i<=index;i++){
    const s=slots[i];let xx=x(i),v=s.v;if(i===index&&animate){xx=boundary;v=previous.v+(s.v-previous.v)*progress;}
    const yy=y(v);
    if(last&&s.t-last.t!==QUARTER){ctx.stroke();ctx.setLineDash([3,4]);ctx.strokeStyle='#687d8b';ctx.beginPath();ctx.moveTo(x(i-1),y(last.v));ctx.lineTo(xx,yy);ctx.stroke();ctx.setLineDash([]);ctx.strokeStyle='#76e0bf';ctx.beginPath();ctx.moveTo(xx,yy);
      ctx.fillStyle='#bacad4';ctx.font='10px "Malgun Gothic",sans-serif';ctx.textAlign='center';ctx.fillText('자료 공백',Math.max(left+30,xx),top+14);ctx.font='11px "Segoe UI",sans-serif';
    }else if(!last)ctx.moveTo(xx,yy);else ctx.lineTo(xx,yy);
    last=s;plotPoints.push({x:xx,y:yy,t:s.t,v:s.v});
  }
  ctx.stroke();ctx.restore();
  const endpoint=plotPoints.at(-1);if(endpoint){ctx.fillStyle='#76e0bf';ctx.beginPath();ctx.arc(endpoint.x,endpoint.y,4,0,Math.PI*2);ctx.fill();}
  ctx.strokeStyle='#718895';ctx.lineWidth=1;ctx.setLineDash([2,4]);ctx.beginPath();ctx.moveTo(boundary,top);ctx.lineTo(boundary,py);ctx.stroke();ctx.setLineDash([]);
  let futureSlots=Math.max(1,Math.min(4,Math.round((f.t+HOUR-current.t)/QUARTER)));
  const endForecast=Math.min(lastX,x(visualIndex+futureSlots));
  if(f.lo!=null&&f.hi!=null){
    ctx.fillStyle='#ff777833';ctx.fillRect(boundary,y(f.hi),endForecast-boundary,y(f.lo)-y(f.hi));
    ctx.strokeStyle='#ff777885';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(boundary,y(f.hi));ctx.lineTo(endForecast,y(f.hi));ctx.moveTo(boundary,y(f.lo));ctx.lineTo(endForecast,y(f.lo));ctx.stroke();
  }
  // Connector is presentation only; it does not fabricate predicted quarter-hour points.
  ctx.strokeStyle='#ff7778';ctx.lineWidth=2.8;ctx.setLineDash([5,4]);ctx.beginPath();
  ctx.moveTo(boundary,y(current.v));ctx.lineTo(boundary+Math.min(15,(endForecast-boundary)*.25),y(f.p));ctx.lineTo(endForecast,y(f.p));ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#ff7778';ctx.beginPath();ctx.arc(endForecast,y(f.p),4.5,0,Math.PI*2);ctx.fill();
  const labelX=Math.min(lastX-5,Math.max(boundary+8,endForecast-5));ctx.textAlign='right';ctx.font='600 '+(mobile?'13':'16')+'px "Segoe UI",sans-serif';ctx.fillStyle='#ffb1b1';ctx.fillText(fmt(f.p),labelX,Math.max(top+25,y(f.p)-12));
  ctx.font='10px "Segoe UI","Malgun Gothic",sans-serif';ctx.fillStyle='#e3a6a9';
  ctx.fillText(f.lo==null?'관측 기준':fmt(f.lo)+' ~ '+fmt(f.hi),lastX-5,top+27);
  ctx.fillStyle='#9bb1bf';ctx.textAlign='left';ctx.fillText('원자료 단위',left,17);ctx.textAlign='right';ctx.fillStyle='#ff9da0';ctx.fillText('최대값 예측',lastX-5,17);
  ctx.fillStyle='#b6c7d1';ctx.textAlign='center';ctx.fillText('지금',boundary,py+22);
  ctx.textAlign='right';ctx.fillText(hm(f.t+HOUR),lastX-3,py+22);
  canvas.setAttribute('aria-label',ymd(current.t)+' '+hm(current.t)+'까지 연속 관측 '+fmt(current.v)+', 빨간 시간 최대값 '+fmt(f.p)+', '+modeName[f.mode]);
}
function frame(now){
  if(playing&&now-lastTick>=speed){lastTick=now;advance();}
  if(!reduced&&animStart){const duration=Math.min(280,speed*.65);draw(Math.min(1,(now-animStart)/duration));if(now-animStart>=duration)animStart=0;}
  requestAnimationFrame(frame);
}
$('play').addEventListener('click',()=>{playing=!playing;lastTick=performance.now();render();});
$('step').addEventListener('click',()=>{playing=false;advance();});
$('speed').addEventListener('change',e=>{speed=+e.target.value;lastTick=performance.now();render();});
$('window').addEventListener('change',e=>{windowHours=+e.target.value;draw(1);});
$('date').addEventListener('change',e=>{if(e.target.value)seek(Date.parse(e.target.value+'T00:00:00Z'));});
$('scrub').addEventListener('input',e=>{playing=false;index=+e.target.value;fromIndex=index;animStart=0;render();});
for(const scene of DATA.scenes){const b=document.createElement('button');b.textContent=scene.label;b.dataset.scene=scene.time;b.addEventListener('click',()=>seek(Date.parse(scene.time)));$('scenes').appendChild(b);}
$('fullscreen').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{$('fullscreen').textContent='전체 화면 사용 불가';}});
canvas.addEventListener('pointermove',e=>{const b=canvas.getBoundingClientRect(),xx=e.clientX-b.left;const p=plotPoints.reduce((a,v)=>!a||Math.abs(v.x-xx)<Math.abs(a.x-xx)?v:a,null);if(!p||Math.abs(p.x-xx)>25){$('tooltip').style.display='none';return;}const tip=$('tooltip');tip.textContent=ymd(p.t)+' '+hm(p.t)+' · '+fmt(p.v);tip.style.display='block';tip.style.left=Math.max(0,Math.min(canvas.clientWidth-220,p.x+12))+'px';tip.style.top=Math.max(2,p.y-45)+'px';});
canvas.addEventListener('pointerleave',()=>{$('tooltip').style.display='none';});
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON','SUMMARY'].includes(document.activeElement.tagName))return;if(e.code==='Space'){e.preventDefault();$('play').click();}if(e.code==='ArrowRight'){e.preventDefault();$('step').click();}});
$('date').min=ymd(slots[0].t);$('date').max=ymd(slots.at(-1).t);$('scrub').max=slots.length-1;
$('startDate').textContent=ymd(slots[0].t);$('endDate').textContent=ymd(slots.at(-1).t);
$('dataMeta').innerHTML='전체 기간 '+DATA.meta.raw_rows_embedded.toLocaleString()+'행<br>15분 관측 '+slots.length.toLocaleString()+'개 · 브라우저 모델 실행 없음';
const im=DATA.meta.interval;
$('methodNote').textContent='빨간 범위는 5~6월의 저장 예측 절대오차로 정한 ±'+fmt(im.half_width)+'의 90% 경험적 예측구간입니다. 기존 7~8월 공통 평가에서 실제 포함률은 '+pct(im.followup_2021_07_08.coverage)+'였습니다. 개별 15분 값의 범위나 미래 포함률 보장이 아닙니다. 다른 시연 구간의 포함률은 검증하지 않았습니다.';
$('scopeNote').textContent='전체 관측을 1월 1일부터 재생합니다. 최종 저장 모델은 6월까지 학습했으므로 1~6월의 예측은 이후 모델을 과거에 적용한 시연이며 실시간 사전 예측 성능이 아닙니다. 처음 과거 입력이 부족한 시간은 관측·직전 시간 기준으로 표시합니다. 7~8월 기존 평가 1,344시간의 예측은 보존했고, 추가 시간과 9월은 새로 계산한 시연입니다. 지속 하락은 기존 개발 모델과 경계의 실험적 출력입니다. 저장 모델을 HTML 제작 때 한 번 실행했으며, HTML을 열 때 모델이나 서버는 실행되지 않습니다.';
seek(slots[0].t);new ResizeObserver(resize).observe(canvas);requestAnimationFrame(frame);
window.demo={getState:()=>({index,playing,time:slots[index].t,current:slots[index].v,forecast:activeForecast(),slotCount:slots.length,forecastCount:forecast.size,plotPointCount:plotPoints.length}),seek,advance,data:DATA};
