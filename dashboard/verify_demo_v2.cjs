/* Execute embedded JavaScript with a DOM/canvas adapter, not browser visual QA. */
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(path.join(__dirname,'전력_흐름_시연.html'),'utf8');
const scripts=[...html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)];
const dataText=scripts.find(m=>m[1].includes('application/json'))[2],app=scripts.find(m=>!m[1].includes('application/json'))[2];
const data=JSON.parse(dataText),ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
assert.equal(new Set(ids).size,ids.length);
const elements=new Map(),geometry=[],frames=[];
const ctx=new Proxy({}, {get(t,n){if(n in t)return t[n];return (...args)=>{geometry.push({method:n,args});};},set(t,n,v){t[n]=v;return true;}});
function element(id=''){return {id,textContent:'',innerHTML:'',value:'',className:'',style:{},dataset:{},children:[],listeners:{},tagName:'DIV',clientWidth:1320,clientHeight:405,
 addEventListener(n,fn){this.listeners[n]=fn;},appendChild(x){this.children.push(x);},setAttribute(n,v){this[n]=v;},
 getBoundingClientRect(){return {width:this.clientWidth,height:this.clientHeight,left:0,top:0};},getContext(){return ctx;},click(){this.listeners.click?.({target:this});}};}
for(const id of ids)elements.set(id,element(id));elements.get('demo-data').textContent=dataText;
const document={getElementById:id=>elements.get(id),createElement:()=>element(),activeElement:{tagName:'BODY'},listeners:{},addEventListener(n,fn){this.listeners[n]=fn;},documentElement:{}};
const sandbox={document,window:{innerWidth:1600,innerHeight:900,listeners:{},addEventListener(n,fn){this.listeners[n]=fn;}},devicePixelRatio:1,matchMedia:()=>({matches:false}),performance:{now:()=>1000},requestAnimationFrame:fn=>frames.push(fn),ResizeObserver:class{constructor(fn){this.fn=fn;}observe(){this.fn();}},console};
vm.createContext(sandbox);vm.runInContext(app,sandbox,{timeout:5000});
const demo=sandbox.window.demo,seek=s=>demo.seek(Date.parse(s)),text=id=>elements.get(id).textContent;
assert.equal(demo.getState().time,Date.parse('2021-01-01T00:15:00Z'));
assert.equal(demo.getState().forecast.mode,'warmup');assert.equal(demo.getState().forecast.lo,null);
assert.equal(demo.getState().slotCount,24480);assert.equal(demo.getState().forecastCount,6120);
seek('2021-01-01T01:00:00Z');assert.equal(demo.getState().forecast.mode,'reference');
seek('2021-01-01T06:00:00Z');assert.equal(demo.getState().forecast.mode,'backcast');assert.ok(demo.getState().forecast.score!=null);
seek('2021-07-12T07:45:00Z');assert.equal(demo.getState().forecast.t,Date.parse('2021-07-12T07:00:00Z'));
demo.advance();assert.equal(demo.getState().time,Date.parse('2021-07-12T08:00:00Z'));
assert.equal(demo.getState().forecast.alarm,1);assert.equal(demo.getState().forecast.routed,1);
assert.ok(Math.abs(demo.getState().forecast.p-192.52)<.01);assert.ok(Math.abs(demo.getState().forecast.b0-181.13)<.01);
assert.equal(text('alarm'),'지속 상승 경고');assert.ok(text('routingTitle').includes('보완'));
assert.ok(!elements.get('history').innerHTML.includes('198'));assert.ok(elements.get('driverBars').innerHTML.includes('과거 전력'));
seek('2021-07-12T08:45:00Z');assert.ok(!elements.get('history').innerHTML.includes('198'));
demo.advance();assert.ok(elements.get('history').innerHTML.includes('198'));
seek('2021-01-01T23:45:00Z');elements.get('play').click();
const before=demo.getState();demo.advance();const midnight=demo.getState();
assert.equal(midnight.time,Date.parse('2021-01-02T00:00:00Z'));assert.equal(midnight.index,before.index+1);
assert.equal(midnight.playing,true);assert.ok(midnight.plotPointCount>=24);
demo.advance();assert.equal(demo.getState().time,Date.parse('2021-01-02T00:15:00Z'));assert.equal(demo.getState().playing,true);
seek('2021-07-13T00:15:00Z');assert.equal(demo.getState().time,Date.parse('2021-07-14T00:15:00Z'));
assert.ok(demo.getState().plotPointCount>=24);assert.ok(text('chartCaption').includes('공백'));
seek('2021-09-01T08:00:00Z');assert.equal(demo.getState().forecast.mode,'extension');assert.ok(Number.isFinite(demo.getState().forecast.p));
seek('2021-08-28T19:00:00Z');assert.equal(text('current'),'0');
const scenes=elements.get('scenes').children;assert.equal(scenes.length,4);
assert.ok(scenes.every(b=>!b.textContent.includes('미탐지')&&!b.textContent.includes('시간 오류')));
scenes[0].click();demo.advance();assert.equal(demo.getState().forecast.alarm,1);
scenes[1].click();demo.advance();assert.equal(demo.getState().forecast.downAlarm,1);assert.ok(text('alarm').includes('하락'));
elements.get('speed').listeners.change({target:{value:'250'}});elements.get('window').listeners.change({target:{value:'24'}});
elements.get('date').listeners.change({target:{value:'2021-09-12'}});assert.equal(new Date(demo.getState().time).toISOString().slice(0,10),'2021-09-12');
elements.get('scrub').listeners.input({target:{value:'0'}});assert.equal(demo.getState().index,0);
const summary=vm.runInContext(`(()=>{let n=0,missing=0,modes={};for(let i=0;i<slots.length;i++){index=i;const f=activeForecast();if(f.p==null)missing++;if(f.t>slots[i].t)throw Error('Future forecast published');if(f.t+HOUR<slots[i].t)throw Error('Expired forecast');modes[f.mode]=(modes[f.mode]||0)+1;n++;}return {n,missing,modes};})()`,sandbox);
assert.equal(summary.n,24480);assert.equal(summary.missing,0);
seek('2021-01-01T23:45:00Z');elements.get('play').click();
vm.runInContext('for(let tick=0;tick<100;tick++)frame(3000+tick*1000)',sandbox);
assert.equal(demo.getState().time,Date.parse('2021-01-03T00:45:00Z'));
assert.equal(demo.getState().playing,true);
for(const width of [1320,768,366]){elements.get('chart').clientWidth=width;elements.get('chart').clientHeight=width<600?315:405;seek('2021-07-12T08:00:00Z');}
for(const item of geometry)for(const v of item.args)if(typeof v==='number')assert.ok(Number.isFinite(v),item.method);
elements.get('scrub').listeners.input({target:{value:'24478'}});elements.get('play').click();demo.advance();
assert.equal(demo.getState().index,24479);assert.ok(demo.getState().forecast.p!=null);demo.advance();assert.equal(demo.getState().playing,false);
assert.equal(/<script\s+src=|fetch\s*\(|XMLHttpRequest|WebSocket/.test(html),false);
assert.ok(html.includes('분석 검증 KPI'));assert.ok(!html.includes('<h2>예측 확인</h2>'));
const viewportChecks=[];
for(const [w,h] of [[1920,1080],[1600,900],[1440,810],[1366,768],[1280,720],[1024,768]]){
  sandbox.window.innerWidth=w;sandbox.window.innerHeight=h;sandbox.window.listeners.resize();
  const stage=elements.get('dashboard'),fw=+stage.dataset.fittedWidth,fh=+stage.dataset.fittedHeight;
  assert.ok(fw<=w+1e-7&&fh<=h+1e-7);assert.ok(Math.abs(fw/fh-16/9)<1e-10);
  assert.equal(elements.get('chart').width,elements.get('chart').clientWidth);
  viewportChecks.push({viewport:[w,h],fitted:[fw,fh],scale:+stage.dataset.scale});
}
const result={status:'PASS',verification_type:'Node DOM/canvas execution; actual browser visual rendering unverified',
 layout_viewport_checks:viewportChecks,
 observations:24480,hourly_forecasts:6120,publication:summary,checks:['Jan1 start and initial reference','full period including September',
 'midnight continuous history and playback','gap preserved as boundary without fabricated observation','hour-start publication','partial actual maximum withheld',
 'selective B1 improvement and exact warning','experimental sustained down scene','zero value','controls and final stop',
 'finite drawing at 1320/768/366px','no external runtime/network'],official_prediction_checks:data.meta.full_period_inference.official_prediction_matches};
fs.writeFileSync(path.join(__dirname,'HTML_동작검증.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
