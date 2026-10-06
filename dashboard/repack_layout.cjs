/* Repack existing embedded data without running models or changing any predictions. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const root=__dirname,out=path.join(root,'전력_흐름_시연.html');
const before=fs.readFileSync(out,'utf8');
const data=before.match(/<script id="demo-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const sources={template:'template_wide.html',style:'style_wide.css',app:'app.js'};
let html=fs.readFileSync(path.join(root,sources.template),'utf8');
html=html.replace('__STYLE__',()=>fs.readFileSync(path.join(root,sources.style),'utf8'));
html=html.replace('__APP__',()=>fs.readFileSync(path.join(root,sources.app),'utf8'));
html=html.replace('__EMBEDDED_DATA__',()=>data);
if(/__STYLE__|__APP__|__EMBEDDED_DATA__/.test(html))throw Error('Unresolved placeholder');
const after=html.match(/<script id="demo-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
if(data!==after)throw Error('Data changed');
fs.writeFileSync(out,html,'utf8');
const report={status:'PASS',layout:'1600 x 900 design canvas (16:9), uniform viewport fitting',
 data_sha256:crypto.createHash('sha256').update(data).digest('hex'),embedded_data_identical:true,
 model_runs:0,source_files:sources,html_bytes:Buffer.byteLength(html),
 verification_scope:'Source/data identity; browser layout visual verification unavailable'};
fs.writeFileSync(path.join(root,'레이아웃_검증.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
