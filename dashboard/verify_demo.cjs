/* Execute the embedded replay logic against a minimal DOM/canvas test adapter.
   This checks application behavior, not a browser rendering or screenshot. */
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(path.join(__dirname, '전력_흐름_시연.html'), 'utf8');
const scripts = [...html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)];
const dataText = scripts.find(m => m[1].includes('application/json'))[2];
const app = scripts.find(m => !m[1].includes('application/json'))[2];
const payload = JSON.parse(dataText);
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
assert.equal(new Set(ids).size, ids.length);
const elements = new Map();
const calls = [];
const context2d = new Proxy({}, {get(target, name) {
  if (name in target) return target[name];
  return (...args) => {if (name === 'fillText') calls.push(args);};
}, set(target, name, value) {target[name] = value; return true;}});
for (const id of ids) elements.set(id, {
  textContent:'', innerHTML:'', value:'', className:'', style:{}, listeners:{},
  tagName:'DIV', clientWidth:1320, clientHeight:372,
  addEventListener(name, callback) {this.listeners[name] = callback;},
  setAttribute(name, value) {this[name] = value;},
  getBoundingClientRect() {return {width:this.clientWidth,height:this.clientHeight,left:0,top:0};},
  getContext() {return context2d;},
  click() {this.listeners.click?.({target:this});},
});
elements.get('demo-data').textContent=dataText;
const scenes=[...html.matchAll(/data-scene="([^"]+)"/g)].map(m=>({dataset:{scene:m[1]},listeners:{},addEventListener(n,f){this.listeners[n]=f;}}));
const document={getElementById:id=>elements.get(id),querySelectorAll:()=>scenes,
 activeElement:{tagName:'BODY'},listeners:{},addEventListener(n,f){this.listeners[n]=f;},documentElement:{}};
const sandbox={document,window:{},devicePixelRatio:1,matchMedia:()=>({matches:false}),
 performance:{now:()=>1000},requestAnimationFrame:()=>{},ResizeObserver:class {constructor(fn){this.fn=fn;}observe(){this.fn();}},console};
vm.createContext(sandbox);
vm.runInContext(app,sandbox,{timeout:5000});
const demo=sandbox.window.demo;
const text=id=>elements.get(id).textContent;
const seek=s=>demo.seek(Date.parse(s));
assert.equal(demo.getState().slotCount,24480);
assert.equal(demo.getState().forecastCount,1344);
assert.equal(demo.getState().time,Date.parse('2021-07-12T07:45:00Z'));
assert.equal(demo.getState().forecast.t,Date.parse('2021-07-12T07:00:00Z'));
assert.ok(!elements.get('history').innerHTML.includes('198'));
demo.advance();
assert.equal(demo.getState().time,Date.parse('2021-07-12T08:00:00Z'));
assert.equal(demo.getState().forecast.t,Date.parse('2021-07-12T08:00:00Z'));
assert.equal(demo.getState().forecast.alarm,1);
assert.ok(Math.abs(demo.getState().forecast.p-192.52)<.01);
assert.equal(text('alarm'),'상승 경고');
assert.ok(elements.get('history').innerHTML.includes('관측 중'));
assert.ok(!elements.get('history').innerHTML.includes('198'));
seek('2021-07-12T08:45:00Z');
assert.ok(!elements.get('history').innerHTML.includes('198'));
demo.advance();
assert.ok(elements.get('history').innerHTML.includes('198'));
seek('2021-08-13T08:00:00Z');
assert.equal(text('alarm'),'경고 없음');
assert.ok(Math.abs(demo.getState().forecast.p-182.40)<.01);
seek('2021-01-01T01:00:00Z');
assert.equal(demo.getState().forecast,null);
assert.equal(text('prediction'),'—');
assert.equal(text('alarm'),'예측 없음');
seek('2021-09-01T01:00:00Z');
assert.equal(demo.getState().forecast,null);
seek('2021-07-13T00:15:00Z');
assert.equal(demo.getState().time,Date.parse('2021-07-14T00:15:00Z'));
assert.ok(text('chartCaption').includes('공백'));
seek('2021-08-28T19:00:00Z');
assert.equal(text('current'),'0');
assert.ok(demo.getState().forecast);
elements.get('play').click();assert.equal(demo.getState().playing,true);
elements.get('step').click();assert.equal(demo.getState().playing,false);
elements.get('speed').listeners.change({target:{value:'250'}});
elements.get('window').listeners.change({target:{value:'24'}});
elements.get('date').listeners.change({target:{value:'2021-08-13'}});
assert.equal(new Date(demo.getState().time).toISOString().slice(0,10),'2021-08-13');
elements.get('scrub').listeners.input({target:{value:'0'}});
assert.equal(demo.getState().index,0);
scenes[0].listeners.click();assert.equal(demo.getState().time,Date.parse('2021-07-12T07:45:00Z'));
// Enumerate each timeline index through the same publication rule used by the UI.
let timelineChecks=0;
for(let i=0;i<24480;i++){
 const state=vm.runInContext(`index=${i};({now:slots[index].t,active:activeForecast()||null,previous:slots[index-1]?.t})`,sandbox);
 if(state.active){assert.ok(state.active.t<=state.now);assert.ok(state.now<state.active.t+3600000);timelineChecks++;}
}
// Exercise drawing at desktop and narrow widths, reject non-finite geometry.
let drawingChecks=0;
for(const width of [1320,768,366]){
 elements.get('chart').clientWidth=width;elements.get('chart').clientHeight=width<600?300:372;
 seek('2021-07-12T08:00:00Z');drawingChecks++;
}
assert.ok(calls.length>0);
for(const call of calls){assert.ok(Number.isFinite(call[1]));assert.ok(Number.isFinite(call[2]));}
assert.equal(/<script\s+src=|fetch\s*\(|XMLHttpRequest|WebSocket/.test(html),false);
const result={status:'PASS',verification_type:'Node execution with DOM/canvas adapter; browser visual rendering not verified',
 timeline_indices_checked:24480,indices_with_forecast:timelineChecks,drawing_widths_checked:drawingChecks,
 checks:['hour-boundary forecast publication','partial-hour actual maxima withheld','completed actual maxima revealed',
 'rise warning and missed-rise examples','missing forecasts','invalid-hour gap','zero power retained',
 'play/pause/step/speed/date/scrub/scenes','finite canvas text geometry','no network or model execution'],
 raw_rows:payload.raw.length,forecast_rows:payload.forecasts.length};
fs.writeFileSync(path.join(__dirname,'HTML_동작검증.json'),JSON.stringify(result,null,2));
console.log(JSON.stringify(result));
