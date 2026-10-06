const fs=require('node:fs'),http=require('node:http'),path=require('node:path');
const page=path.join(__dirname,'layout_export.html'),output=path.join(__dirname,'ppt_layout.json');
http.createServer((req,res)=>{
 if(req.method==='POST'&&req.url==='/capture'){
  let body='';req.on('data',b=>{body+=b;if(body.length>3e6)req.destroy();});req.on('end',()=>{try{const p=JSON.parse(body);if(p.slides.length!==16)throw Error('Invalid deck');fs.writeFileSync(output,JSON.stringify(p,null,2));res.writeHead(200);res.end('saved');}catch(e){res.writeHead(400);res.end('invalid');}});return;
 }
 if(req.url==='/'&&req.method==='GET'){res.writeHead(200,{'Content-Type':'text/html; charset=utf-8'});res.end(fs.readFileSync(page));return;}
 res.writeHead(404);res.end();
}).listen(8786,'127.0.0.1',()=>console.log('PPT layout capture: http://127.0.0.1:8786/'));
