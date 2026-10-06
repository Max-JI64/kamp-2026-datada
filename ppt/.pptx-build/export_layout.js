/* Temporary local export control. Reads rendered HTML geometry only. */
function captureDeckLayout(){
 const round=n=>Math.round(n*1000)/1000;
 const results=[];
 for(const slide of document.querySelectorAll('.slide')){
  const root=slide.getBoundingClientRect(),scale=root.width/1920;
  const rect=el=>{const r=el.getBoundingClientRect();return {left:round((r.left-root.left)/scale),top:round((r.top-root.top)/scale),width:round(r.width/scale),height:round(r.height/scale)};};
  const color=c=>{const m=c.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);if(!m)return c;if(m[4]==='0')return 'none';return '#'+m.slice(1,4).map(x=>(+x).toString(16).padStart(2,'0')).join('');};
  const out={number:+slide.dataset.slide,background:color(getComputedStyle(slide).backgroundColor),shapes:[],texts:[],tables:[],images:[]};
  function text(el,ownOnly=false){
   const style=getComputedStyle(el),r=rect(el),pt=+parseFloat(style.paddingTop),pl=+parseFloat(style.paddingLeft),pr=+parseFloat(style.paddingRight),pb=+parseFloat(style.paddingBottom);
   let nodes=[];const walk=document.createTreeWalker(el,NodeFilter.SHOW_TEXT);let n;while(n=walk.nextNode()){if(ownOnly&&n.parentElement!==el)continue;nodes.push(n);}
   const lineMap=new Map();
   for(const n of nodes){const cs=getComputedStyle(n.parentElement);if(!n.textContent.trim())continue;for(let i=0;i<n.textContent.length;i++){const rg=document.createRange();rg.setStart(n,i);rg.setEnd(n,i+1);const rr=rg.getBoundingClientRect();if(!rr.width&&!rr.height)continue;const key=Math.round((rr.top-root.top)/scale);if(!lineMap.has(key))lineMap.set(key,[]);const arr=lineMap.get(key);const st={bold:+cs.fontWeight>=600,color:color(cs.color)};const prev=arr.at(-1);if(prev&&prev.bold===st.bold&&prev.color===st.color)prev.text+=n.textContent[i];else arr.push({text:n.textContent[i],...st});}}
   const lines=[...lineMap.entries()].sort((a,b)=>a[0]-b[0]).map(([y,runs])=>({y,runs}));
   if(!lines.length)return;
   const fs=parseFloat(style.fontSize),lh=parseFloat(style.lineHeight)||fs*1.2;
   out.texts.push({tag:el.tagName,className:el.className,position:{left:r.left+pl+parseFloat(style.borderLeftWidth),top:r.top+pt+parseFloat(style.borderTopWidth),width:r.width-pl-pr-parseFloat(style.borderLeftWidth)-parseFloat(style.borderRightWidth),height:r.height-pt-pb},fontSize:fs,lineHeight:lh,color:color(style.color),bold:+style.fontWeight>=600,align:style.textAlign,lines});
  }
  function visit(el){
   const cs=getComputedStyle(el),r=rect(el);
   if(el!==slide&&!el.closest('table')){
    const bg=color(cs.backgroundColor);if(bg!=='none'&&r.width&&r.height)out.shapes.push({position:r,fill:bg});
    for(const side of ['Top','Bottom','Left','Right']){const bw=parseFloat(cs['border'+side+'Width']);if(bw>0&&cs['border'+side+'Style']!=='none'){let p={...r};if(side==='Top')p.height=bw;if(side==='Bottom'){p.top+=p.height-bw;p.height=bw;}if(side==='Left')p.width=bw;if(side==='Right'){p.left+=p.width-bw;p.width=bw;}out.shapes.push({position:p,fill:color(cs['border'+side+'Color'])});}}
   }
   if(el.tagName==='TABLE'){
    const rows=[...el.rows].map(tr=>[...tr.cells].map(td=>{const st=getComputedStyle(td);return {text:td.innerText,position:rect(td),fontSize:parseFloat(st.fontSize),color:color(st.color),bold:td.tagName==='TH'||!!td.querySelector('b'),fill:color(st.backgroundColor),padding:{top:parseFloat(st.paddingTop),bottom:parseFloat(st.paddingBottom),left:parseFloat(st.paddingLeft),right:parseFloat(st.paddingRight)}};}));
    out.tables.push({position:r,rows,rowHeights:[...el.rows].map(tr=>rect(tr).height),colWidths:[...el.rows[0].cells].map(td=>rect(td).width)});return;
   }
   if(el.tagName==='IMG'){out.images.push({position:r,dataUrl:el.src,alt:el.alt});return;}
   if(el.classList.contains('cover-mark')){for(const c of el.children)if(c.tagName==='SPAN')text(c);return;}
   if(el.classList.contains('kicker')){text(el,true);for(const c of el.children)visit(c);return;}
   const inline=new Set(['BR','B','EM','STRONG','I','SPAN']);
   if([...el.childNodes].some(n=>n.nodeType===3&&n.textContent.trim())&&[...el.children].every(c=>inline.has(c.tagName))){text(el);return;}
   for(const c of el.children)visit(c);
  }
  visit(slide);
  results.push(out);
 }
 return {width:1920,height:1080,slides:results};
}
const exportButton=document.createElement('button');exportButton.textContent='PPT 배치 내보내기';exportButton.id='exportPptLayout';document.querySelector('.deck-controls').append(exportButton);
exportButton.onclick=async()=>{exportButton.textContent='배치 저장 중';const r=await fetch('/capture',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(captureDeckLayout())});exportButton.textContent=r.ok?'배치 저장 완료':'배치 저장 실패';};
