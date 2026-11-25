const fs=require('fs'); const calcs={};
global.window={}; global.document={querySelectorAll:()=>[]}; const acc=(a,f)=>(f?a.map(f):a).filter(v=>v!=null&&!isNaN(v));
global.d3={median:(a,f)=>{const v=acc(a,f).sort((x,y)=>x-y);const n=v.length;return n%2?v[(n-1)/2]:(v[n/2-1]+v[n/2])/2}, mean:(a,f)=>{const v=acc(a,f);return v.reduce((x,y)=>x+y,0)/v.length}, sum:(a,f)=>acc(a,f).reduce((x,y)=>x+y,0), min:(a,f)=>Math.min(...acc(a,f)), max:(a,f)=>Math.max(...acc(a,f)), format:()=>String, group:(a,f)=>{const m=new Map();for(const x of a){const k=f(x);if(!m.has(k))m.set(k,[]);m.get(k).push(x)}return m}};
global.dash={calc:(id,o)=>calcs[id]=o, draw:()=>{}, colors:['a','b','c','d']};
const html=fs.readFileSync('index_v14.html','utf8'); const js=html.slice(html.indexOf('<script>')+8, html.lastIndexOf('</script>'));
eval(js);
const D={}; const load=n=>{try{return JSON.parse(fs.readFileSync('data/'+n+'.json')).rows}catch(e){return null}};
let left=Object.keys(calcs);
for(let g=0;g<50&&left.length;g++) for(const id of [...left]){const ins=calcs[id].inputs.map(n=>D[n]!==undefined?D[n]:load(n)); if(ins.some(x=>x==null)) continue; D[id]=calcs[id].fn(...ins); left=left.filter(x=>x!==id);}
console.log('unresolved', left);
const body=html.slice(html.indexOf('</style>'), html.indexOf('<script>'));
let missing=[];
const txt=body.replace(/<span data-draw="num"[^>]*data-source="(\w+)"([^>]*)>…<\/span>/g,(m,src,rest)=>{
  const g=(a)=>{const x=rest.match(new RegExp(a+'="([^"]*)"'));return x?x[1].replace(/&amp;/g,'&'):null};
  let v; if(Array.isArray(D[src])){const conds=(g('data-where')||'').split('&').filter(Boolean).map(p=>p.split('='));
  const row=D[src].find(r=>conds.every(([k,val])=>String(r[k])===val)); v=row?row[g('data-field')]:null;} else v=D[src];
  if(v==null) missing.push(src+' '+g('data-where')+' '+g('data-field'));
  return '«'+window.PLE.fmt(v,g('data-fmt'))+'»';});
const part=txt.slice(0, txt.indexOf('id="prior-art"'));
console.log(part.replace(/<style[\s\S]*?<\/style>/g,'').replace(/<[^>]+>/g,' ').replace(/\s+/g,' ').slice(0,80000)); console.log('missing', missing);
