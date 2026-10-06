const fs=require('node:fs'),path=require('node:path');
const base=path.join(__dirname,'..');
let h=fs.readFileSync(path.join(base,'발표구성_시안.html'),'utf8');
h=h.replace('</body>','<script>'+fs.readFileSync(path.join(__dirname,'export_layout.js'),'utf8')+'</script></body>');
fs.writeFileSync(path.join(__dirname,'layout_export.html'),h,'utf8');
