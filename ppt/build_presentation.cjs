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
  form:'경진대회 결과보고서 양식_일반국민,대학(원)생 부문.hwpx', submission:'제출/output/22_재현검증.csv'
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

require('./slides_content.cjs')({add,table,point,stat,call,cols,steps,figure});

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
/* === SIXTEEN SLIDE CONTENT LAYOUTS === */
.cover-sub,.cover-bottom{max-width:1320px}
.section-heading{font-size:43px;line-height:1.4;margin-bottom:26px;letter-spacing:-.035em}
.caption{font-size:24px;line-height:1.55;color:var(--muted);margin-top:18px;word-break:keep-all}
.compact-metrics{display:flex;gap:65px;border-bottom:1px solid var(--rule);padding-bottom:23px;margin-bottom:24px}
.compact-metrics .stat strong{font-size:49px}.compact-metrics .stat span{font-size:25px;margin-top:8px}
.lower-metrics{margin-top:32px;padding-top:25px;border-top:1px solid var(--rule);border-bottom:0}
.summary-band{background:var(--wash);padding:15px 20px;font-size:26px;line-height:1.55;margin-top:15px;color:var(--accent)}
.columns.evidence-split{grid-template-columns:1050px 1fr;gap:45px}
.slide:not(.cover) h2{font-size:58px;margin-bottom:32px}
.slide:not(.cover) .point{padding:17px 0}.slide:not(.cover) .point h3{font-size:30px;margin-bottom:12px}.slide:not(.cover) .point p{font-size:27px;line-height:1.6}
.slide:not(.cover) table{font-size:27px}.slide:not(.cover) th{font-size:24px;padding:13px 20px}.slide:not(.cover) td{padding:14px 20px}
.slide:not(.cover) .callout{font-size:26px;padding:18px 25px;margin-top:23px;line-height:1.55}
.pattern-page .point h3{font-size:27px}.pattern-page .point p{font-size:25px}.pattern-page .point{padding:15px 0}.pattern-page .period-label{font-size:24px;margin:19px 0 13px}.pattern-page table{font-size:25px!important}.pattern-page td{padding:11px 17px!important}.pattern-page th{font-size:23px!important;padding:12px 17px!important}.pattern-page .caption{font-size:23px;margin-top:14px}
.state-page .columns{gap:45px}.state-page td{font-size:25px}.state-page .columns>div:first-child table td:first-child{width:230px}
.input-page .columns{gap:45px}.input-page td{padding:11px 18px!important}.input-page .point h3{font-size:28px!important}.input-page .point p{font-size:25px!important}.input-page .caption{font-size:23px;margin-top:14px}
.terminal-small{height:325px;gap:10px}.terminal-small img{height:275px}.terminal-small figcaption{font-size:22px;min-height:35px}
.forecast-strip{display:grid;grid-template-columns:1fr 50px 1fr;gap:30px;align-items:center;margin-bottom:28px}.forecast-strip>div{background:var(--wash);padding:23px 27px;border-top:3px solid var(--accent)}.forecast-strip small{display:block;font-size:23px;color:var(--accent);margin-bottom:10px}.forecast-strip strong{font-size:29px;line-height:1.55;font-weight:600}.forecast-strip>span{font-size:45px;color:var(--accent)}
.model-page .point p{font-size:26px!important}.model-page .point{padding:16px 0!important}
.method-page .columns{grid-template-columns:1.12fr 1fr;gap:50px}.method-page td:first-child{white-space:nowrap}.method-page td{font-size:26px}.method-box{background:var(--ink);padding:30px;color:white}.method-box small{font-size:24px;color:#afd4d9}.method-box h3{font-size:38px;line-height:1.4;margin:15px 0 28px}.branch{border-top:1px solid #668492;padding:21px 0}.branch b{font-size:30px;font-weight:600;color:#8bd5d9}.branch p{font-size:26px;color:#c2d6df;line-height:1.55;margin-top:12px}
.diagnosis-strip{display:grid;grid-template-columns:1fr 45px 1.3fr 45px 1.2fr;align-items:center;gap:22px;padding:23px 25px;background:var(--wash);margin-bottom:27px}.diagnosis-strip small{font-size:23px;display:block;color:var(--muted);margin-bottom:10px}.diagnosis-strip strong{font-size:37px;color:var(--accent);letter-spacing:-.03em}.diagnosis-strip>span{font-size:39px;color:var(--accent)}
.error-strip{display:flex;justify-content:space-between;gap:25px;background:var(--wash);padding:16px 22px;font-size:25px;line-height:1.5;margin-bottom:15px;color:var(--muted)}.error-strip b{color:var(--accent)}
.case-wide{height:351px;gap:8px;margin-bottom:18px}.case-wide img{height:302px}.case-wide figcaption{font-size:22px;min-height:34px}.errors-page td{padding:10px 20px!important}.errors-page th{padding:11px 20px!important}.errors-page .point{padding:13px 0!important}.errors-page .point h3{font-size:29px!important}.errors-page .point p{font-size:26px!important}
.use-page .steps{gap:45px}.use-page .step-no{font-size:43px;margin-bottom:20px}.use-page .steps h3{font-size:31px;margin-bottom:15px}.use-page .steps p{font-size:26px;line-height:1.6}.use-page .steps>div{padding-top:20px}.use-page td{padding:12px 20px!important}
.reproduction-flow{display:flex;justify-content:space-between;align-items:center;background:var(--wash);padding:24px 30px;font-size:30px;gap:20px;margin-bottom:30px;color:var(--accent)}.reproduction-flow b{font-weight:400}.reproduction-page .stat{margin-bottom:25px}.reproduction-page .stat strong{font-size:53px}.reproduction-page .stat span{font-size:25px;margin-top:8px}.reproduction-page td{font-size:26px}.reproduction-page td:first-child{white-space:nowrap}
.conclusion-page .closing-thesis h3{font-size:55px}.conclusion-page .callout{margin-top:30px}
.input-page .point{padding:12px 0!important}.input-page .callout{margin-top:12px!important;padding:12px 25px!important}
.evaluation-page .forecast-strip{margin-bottom:20px}.evaluation-page .point{padding:13px 0!important}.evaluation-page .callout{margin-top:18px!important}
.model-page .point{padding:12px 0!important}.model-page .callout{margin-top:18px!important}
/* === CLOSING === */
.closing{background:#e9f2f2}.closing h2{font-size:57px}.closing-thesis{padding-left:32px;border-left:5px solid var(--accent)}.closing-thesis>span{font-size:24px;color:var(--accent)}.closing-thesis h3{font-size:59px;line-height:1.45;letter-spacing:-.03em;margin-top:18px}.closing-thesis em{font-style:normal;color:var(--accent)}.closing-metrics{display:flex;gap:70px;margin-top:42px;border-top:1px solid #b6ccd0;padding-top:28px}.closing-metrics .stat strong{font-size:55px}.closing-limit{font-size:27px;line-height:1.7;color:var(--muted);margin-top:34px}
/* === MOTION AND EXTERNAL REVIEW UI === */
.slide.visible .body{animation:enter .25s ease-out}@keyframes enter{from{opacity:.55;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}.deck-controls{bottom:14px;background:#0d202bdc;color:#edf5f6;display:flex;align-items:center;justify-content:center;flex-wrap:wrap;width:max-content;gap:12px;padding:9px 16px;border:1px solid #4b6570;border-radius:8px;font-size:13px;white-space:nowrap;max-width:96vw}.deck-controls button,.deck-controls select{background:transparent;color:inherit;border:1px solid #607784;border-radius:4px;padding:6px 10px}.deck-controls select{max-width:270px;background:#19323e}.deck-controls button:hover{background:#345462}.deck-controls .counter{font-family:'DM Mono',monospace;min-width:50px;text-align:center}.notes-panel{position:fixed;right:18px;top:18px;width:min(520px,90vw);max-height:80vh;overflow:auto;background:#fff;box-shadow:0 6px 28px #0004;padding:25px;z-index:1100;border-top:5px solid var(--accent);font-size:16px;line-height:1.8}.notes-panel[hidden]{display:none}.notes-panel h3{font-size:20px;line-height:1.6;margin-bottom:16px}.notes-panel h4{margin:20px 0 6px}.notes-panel a{color:#086676}.notes-panel button{float:right}.edit-hotzone{position:fixed;left:0;top:0;width:70px;height:60px;z-index:1150}.edit-toggle{position:fixed;top:14px;left:14px;z-index:1200;opacity:0;pointer-events:none;padding:8px 12px;background:#163846;color:#fff;border:1px solid #80a1ac;border-radius:4px}.edit-toggle.show,.edit-toggle.editing{opacity:1;pointer-events:auto}.editing [data-edit]{outline:2px dashed #5999a7;outline-offset:4px}[contenteditable=true]{cursor:text}.progress{position:fixed;top:0;left:0;height:3px;background:#70c6ce;z-index:1000}.theme-paper{--slide-bg:#fbfaf6;--accent:#284b67;--wash:#eeeee7;--stage-bg:#272f35}.theme-signal{--slide-bg:#f0ece3;--ink:#1c2644;--muted:#566174;--accent:#8b662e;--wash:#e7e1d5;--rule:#cec5b5;--stage-bg:#1c2644}.theme-signal .cover{background:#1c2644}.theme-signal .closing{background:#e7e1d5}.theme-signal .cover-mark{color:#c8a870;border-color:#c8a870}.theme-signal .cover-mark>div{border-color:#c8a870}
@media(max-width:650px){.deck-controls{gap:5px;padding:7px 8px;font-size:11px;bottom:8px}.deck-controls select{max-width:115px}.deck-controls button{padding:5px}.deck-controls .theme-select,.deck-controls #printBtn,.deck-controls #saveBtn{display:none}}
@page{size:16in 9in;margin:0}@media print{.notes-panel,.edit-hotzone,.edit-toggle,.progress{display:none!important}.slide{page-break-inside:avoid} [data-edit]{outline:none!important}}
`;
const labels={data:'데이터',eda:'EDA',analysis:'Analysis',model:'Modeling',report:'활용·차별성 원고',task:'대회 과제공개',form:'보고서 양식',submission:'제출 재현 검증'};
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
let editing=false;const key='manufacturing-power-deck-v3-16';const editable=[...document.querySelectorAll('[data-edit]')];editable.forEach((el,i)=>el.dataset.editId=i);try{const saved=JSON.parse(localStorage.getItem(key)||'{}');editable.forEach((el,i)=>{if(saved[i]!==undefined)el.innerHTML=saved[i];});}catch{}
const editor={toggle(){editing=!editing;document.body.classList.toggle('editing',editing);editable.forEach(el=>el.contentEditable=editing?'true':'false');const b=document.querySelector('#editToggle');b.classList.toggle('editing',editing);b.textContent=editing?'편집 종료 (E)':'문구 편집 (E)';}};
document.querySelector('#editToggle').onclick=()=>editor.toggle();let timer;const hot=document.querySelector('.edit-hotzone'),toggle=document.querySelector('#editToggle');[hot,toggle].forEach(el=>{el.onmouseenter=()=>{clearTimeout(timer);toggle.classList.add('show')};el.onmouseleave=()=>{timer=setTimeout(()=>{if(!editing)toggle.classList.remove('show')},400)}});hot.onclick=()=>editor.toggle();document.addEventListener('input',e=>{if(!e.target.closest('[data-edit]'))return;const saved={};editable.forEach((el,i)=>saved[i]=el.innerHTML);try{localStorage.setItem(key,JSON.stringify(saved))}catch{};});document.querySelector('#saveBtn').onclick=()=>{if(editing)editor.toggle();const cloned=document.documentElement.cloneNode(true);cloned.querySelector('body').classList.remove('editing');cloned.querySelectorAll('[contenteditable]').forEach(el=>el.removeAttribute('contenteditable'));const blob=new Blob(['<!DOCTYPE html>\\n'+cloned.outerHTML],{type:'text/html;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='발표구성_수정본.html';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),2000);};
window.deck=new SlidePresentation();
/* === REVIEW DIAGNOSTICS: READ-ONLY GEOMETRY, NO DATA/MODEL CLAIM === */
window.auditLayout=()=>{const scale=document.querySelector('.deck-stage').getBoundingClientRect().width/1920;const issues=[];for(const slide of document.querySelectorAll('.slide')){const sr=slide.getBoundingClientRect(),foot=slide.querySelector('footer').getBoundingClientRect();for(const el of slide.querySelectorAll('h2,.point,.stat,.callout,table,figure,.steps,.dataset-row,.selection,.big-question,.timeline,.routing,.two-findings,.closing-thesis,.closing-metrics,.closing-limit')){const r=el.getBoundingClientRect();if(r.left<sr.left-2||r.right>sr.right+2||r.top<sr.top-2||r.bottom>foot.top-8*scale)issues.push({slide:+slide.dataset.slide,type:'bounds',element:el.className||el.tagName});if(el.scrollHeight>el.clientHeight+3&&getComputedStyle(el).overflow!=='visible')issues.push({slide:+slide.dataset.slide,type:'overflow',element:el.className||el.tagName});}const children=[...slide.querySelector('.body').children];for(let i=0;i<children.length;i++)for(let j=i+1;j<children.length;j++){const a=children[i].getBoundingClientRect(),b=children[j].getBoundingClientRect();if(Math.min(a.right,b.right)-Math.max(a.left,b.left)>3&&Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>3)issues.push({slide:+slide.dataset.slide,type:'overlap',elements:[children[i].className,children[j].className]});}}return {slides:metadata.length,stage:[1920,1080],viewport:[innerWidth,innerHeight],images:[...document.images].every(i=>i.complete&&i.naturalWidth>0),issues};};
`;
const html=`<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>제조 전력 예측 · 발표 구성 시안</title><link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap" rel="stylesheet"><style>${css}</style></head><body><div class="progress"></div><div class="deck-viewport"><main class="deck-stage">${rendered}</main></div><nav class="deck-controls" aria-label="발표자료 탐색"><button id="prev" aria-label="이전 슬라이드">←</button><span class="counter" id="counter"></span><button id="next" aria-label="다음 슬라이드">→</button><select id="jump" aria-label="슬라이드 선택"></select><button id="noteBtn">발표·제작 메모 (N)</button><select id="theme" class="theme-select" aria-label="색상 비교"><option value="">청록·네이비</option><option value="theme-paper">흰색·잉크</option><option value="theme-signal">네이비·골드</option></select><button id="printBtn">인쇄</button><button id="saveBtn">HTML 저장</button></nav><aside class="notes-panel" hidden aria-label="발표 및 PPT 제작 메모"><button id="closeNotes">닫기</button><h3 id="noteTitle"></h3><h4>발표·제작 메모</h4><p id="noteText"></p><h4>근거 원고</h4><ul id="sourceList"></ul><p>실제 PPT는 와이드 16:9로 제작합니다. 문구 편집: E. 방향키·휠·좌우 스와이프로 이동합니다. 핵심 문구와 표는 PPT 텍스트·표로 옮기고, 기존 그림은 ppt/assets에서 가져오면 됩니다.</p></aside><div class="edit-hotzone" aria-hidden="true"></div><button id="editToggle" class="edit-toggle">문구 편집 (E)</button><script>${script}</script></body></html>`;
fs.writeFileSync(path.join(OUT,'발표구성_시안.html'),html,'utf8');
const outline=`# 발표 구성 시안 · 16장

압축 원고 전체의 핵심 정의·근거·결과·한계를 16장으로 묶었다. 이전 12장 시안에서 제작 메모로 밀렸던 분석 근거와 판단 전환을 본문에 복원했다. 와이드 16:9이며 사용자가 실제 PPT를 만드는 참고 자료다.

## 구성 원칙

- 배경·과제 선택 이유·목표를 도입부에 함께 배치한다. 선택 이유는 현재 분석 목적과 제공 변수에 근거한 설명이다.
- 서로 연결된 관측은 같은 페이지에 배치하고, 정의·방법·진단·성과는 구분한다.
- 개발 모델 비교와 후반기 탐색적 재평가를 분리하며 표본과 해석 범위를 표시한다.
- 양식 문서는 구성 참고로만 사용했다. 문서 내부의 서명·제출 등의 문구는 실행 지시로 취급하지 않았다.
- 절감은 현장 활용 제안이다. 저장된 노트북 실행 검증과 이번 HTML의 화면 확인도 구분한다.

## 페이지별 구성과 발표 메모

${metadata.map(m=>`### ${String(m.number).padStart(2,'0')}. ${m.title}

- 구분: ${m.section}
- 발표·제작 메모: ${m.note}
- 근거: ${m.sources.map(p=>`[${p}](<../${p}>)`).join(' · ')}
`).join('\n')}
## PPT로 옮길 때

HTML의 16:9 배치와 정보 묶음을 기준으로 만든다. 제목 28~34pt, 본문 18~22pt를 출발점으로 잡고 실제 PPT 화면에서 가독성을 확인한다. 표는 편집 가능한 PPT 표로 옮기고 그림은 assets의 PNG를 사용한다. 상세 조건은 발표·제작 메모에서 확인한다.

이전 시안은 archive/12장_시안 및 archive/26장_시안에 보존했다. 페이지별 통합 판단은 [내용 배분 분석](<발표_내용배분_분석.md>)에 있다.
`;
fs.writeFileSync(path.join(OUT,'슬라이드_구성안.md'),outline,'utf8');
const README=`# PPT 제작 참고 자료

[발표 구성 HTML](<발표구성_시안.html>)은 총 16장이다. 압축 원고의 핵심 정의·분석 근거·방법·성과·한계를 화면에 배치했다. 실제 PPT는 사용자가 별도로 제작한다.

- 이동: 방향키, Page Up/Down, 휠, 스와이프, 하단 슬라이드 선택.
- 발표·제작 메모: 하단 버튼 또는 N. 조건과 근거 원고 링크를 제공한다.
- 문구 편집: E. 같은 브라우저에 저장하며 HTML 저장으로 수정 파일을 내려받는다. 이전 시안과 편집 저장소를 구분했다.
- 색상: 청록·네이비, 흰색·잉크, 네이비·골드.
- 글꼴: Noto Sans KR·DM Mono, 오프라인에서 맑은 고딕 등으로 대체.
- 인쇄 규칙을 포함하지만 실제 PDF 생성은 이번 범위에 포함하지 않았다.

[슬라이드 구성안](<슬라이드_구성안.md>)과 [내용 배분 분석](<발표_내용배분_분석.md>)을 함께 참고한다. 그림 3개는 HTML에 내장했고 assets에도 복사했다. assets에는 이전 시안에서 사용한 그림 복사본도 있다. 원고 링크는 프로젝트 폴더 안에서 사용할 때 유효하다.

이전 12장·26장 시안은 archive에 보존했다. 원자료·원고·모델 결과는 변경하지 않았다. 재현성 페이지는 기존 제출 검증 기록에 근거하며 이번 HTML 작성에서 노트북을 재실행하지 않았다.

build_presentation.cjs와 slides_content.cjs가 HTML·구성안·출처 목록을 재생성한다. 화면 검증의 범위와 결과는 [검증 메모](<검증_메모.md>)에서 확인한다.
`;
fs.writeFileSync(path.join(OUT,'README.md'),README,'utf8');
const sha=file=>crypto.createHash('sha256').update(fs.readFileSync(path.join(ROOT,file))).digest('hex');
fs.writeFileSync(path.join(OUT,'source_manifest.json'),JSON.stringify({created:new Date().toISOString(),slides:slides.length,main_slides:slides.length,appendix_slides:0,embedded_figures:assets.size,sources:Object.fromEntries(Object.values(sources).map(p=>[p,sha(p)])),figures:Object.fromEntries([...assets].map(([p,n])=>[p,{copy:'assets/'+n,sha256:sha(p)}])),scope:'Sixteen-slide HTML draft; previous 12 and 26 slide drafts archived; existing results reused; no model execution'},null,2),'utf8');
fs.writeFileSync(path.join(OUT,'slides_metadata.json'),JSON.stringify(metadata,null,2),'utf8');
console.log(JSON.stringify({slides:slides.length,figures:assets.size,htmlBytes:Buffer.byteLength(html),output:path.join(OUT,'발표구성_시안.html')}));
