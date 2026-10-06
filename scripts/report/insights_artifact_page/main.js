(function(){
const D=JSON.parse(document.getElementById('data').textContent);
const Y=D.meta.years, N=Y.length;
%%HELPERS%%

// ---------- shared ----------
const rname=r=>r==='Congo, D.R.'?'DR Congo':r==='Korea, Rep.'?'South Korea':r==='ROW'?'ROW (residual)':r==='Other'?'Other regions':r;
const mt=v=>fmt(v/1000,v>=10000?1:2)+' Mt';
const KIND={ore:'var(--k-ore)',concentrate:'var(--k-conc)',matte:'var(--k-matte)',anode:'var(--k-anode)',refined:'var(--k-ref)',cathode:'var(--k-cath)',scrap:'var(--k-scrap)',loss:'var(--k-loss)',stock:'var(--k-stock)',use:'var(--k-use)',process:'var(--ink)'};
const COMM={Concentrate:'var(--k-conc)',Matte:'var(--k-matte)',Anode:'var(--k-anode)',Cathode:'var(--k-cath)'};
const state={year:2024,focus:null};
function attachTipEl(wrap,svg,W,fn){const tip=document.createElement('div');tip.className='tip';tip.hidden=true;wrap.appendChild(tip);
  svg.addEventListener('mousemove',e=>{const r=svg.getBoundingClientRect();const px=(e.clientX-r.left)*W/r.width;const res=fn(e,px);if(!res){tip.hidden=true;return;}tip.hidden=false;tip.innerHTML=res.html;const ax=res.x/W*r.width;const left=ax>r.width*0.6;tip.style.top=Math.max(0,(e.clientY-r.top)-10)+'px';tip.style.left=left?'':(ax+14)+'px';tip.style.right=left?(r.width-ax+14)+'px':'';});
  svg.addEventListener('mouseleave',()=>{tip.hidden=true;fn(null);});}

// ---------- absolute stacked area ----------
function areaChart(root,o){
  const W=o.width||520,H=o.height||280,m={t:14,r:20,b:30,l:52};const iw=W-m.l-m.r,ih=H-m.t-m.b;
  const tot=sum(...o.series.map(s=>s.values));const hi=o.max||Math.max(...tot);
  const ticks=niceTicks(0,hi,o.nticks||5);const top=Math.max(hi,ticks[ticks.length-1]);
  const X=i=>m.l+iw*i/(N-1),Yy=v=>m.t+ih*(1-v/top);
  const wrap=document.createElement('div');wrap.className='chart';const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':o.aria||''});
  ticks.forEach(t=>{if(t>top)return;svg.appendChild(svgEl('line',{x1:m.l,x2:W-m.r,y1:Yy(t),y2:Yy(t),class:t===0?'axis':'grid'}));const tx=svgEl('text',{x:m.l-8,y:Yy(t)+4,'text-anchor':'end'});tx.textContent=o.yfmt?o.yfmt(t):fmt(t);svg.appendChild(tx);});
  const cum=Y.map(()=>0);
  o.series.forEach(s=>{const v=s.values.map(x=>Math.max(0,x));const up=v.map((x,i)=>cum[i]+x);const d='M'+up.map((x,i)=>X(i).toFixed(1)+','+Yy(x).toFixed(1)).join(' L')+' L'+cum.map((x,i)=>X(i).toFixed(1)+','+Yy(x).toFixed(1)).reverse().join(' L')+' Z';
    svg.appendChild(svgEl('path',{d,style:`fill:${s.color};stroke:var(--surface);stroke-width:1.5;stroke-linejoin:round`}));v.forEach((x,i)=>cum[i]+=x);});
  (o.xticks||[1995,2000,2005,2010,2015,2020,2024]).forEach(y=>{const i=Y.indexOf(y);const tx=svgEl('text',{x:X(i),y:H-m.b+18,'text-anchor':'middle'});tx.textContent=y;svg.appendChild(tx);});
  const xh=svgEl('line',{class:'xh',y1:m.t,y2:m.t+ih,x1:-10,x2:-10});svg.appendChild(xh);wrap.appendChild(svg);
  const tf=v=>o.tipfmt?o.tipfmt(v):fmt(v,0);
  attachTip(wrap,svg,W,px=>{if(px==null){xh.setAttribute('x1',-10);xh.setAttribute('x2',-10);return null;}const i=Math.max(0,Math.min(N-1,Math.round((px-m.l)/iw*(N-1))));xh.setAttribute('x1',X(i));xh.setAttribute('x2',X(i));
    return {x:X(i),html:tipRows(Y[i]+' · '+(o.unit||'kt Cu'),o.series.map(s=>({name:s.name,color:s.color,val:tf(s.values[i])})).concat([{name:'Total',color:'transparent',val:tf(tot[i])}]))};});
  root.appendChild(wrap);
  if(o.table!==false)dataTable(root,o.series.map(s=>s.name).concat(['Total']),Y.map((y,i)=>[y,...o.series.map(s=>tf(s.values[i])),tf(tot[i])]),'Year');
  return wrap;
}

// ================= PART 1: system size =================
const P=D.process_out;
const P_SERIES=[['Ore mined','Mining (ore)','var(--r1)'],['Concentrate','Concentrate','var(--r2)'],['Smelter matte','Smelter matte','var(--r3)'],['Anode','Anode','var(--r4)'],['Refined cathode','Cathode','var(--r5)'],['SX‑EW cathode','SX-EW cathode','var(--c-4)']];
const f1a=document.getElementById('f1a');
f1a.appendChild(legend(null,P_SERIES.map(([n,k,c])=>({name:n,color:c}))));
lineChart(f1a,{aria:'Process outputs along the chain, Mt Cu',series:P_SERIES.map(([n,k,c])=>({name:n,values:P[k],color:c})),
  yfmt:v=>fmt(v/1000,0),tipfmt:v=>fmt(v,0)+' kt',endVal:v=>fmt(v/1000,1),ydomain:[0,30000],nticks:6,table:false});
dataTable(f1a,Object.keys(P),Y.map((y,i)=>[y,...Object.keys(P).map(k=>fmt(P[k][i],0))]),'Year');

const SI=D.sb_in,SO=D.sb_out;
const scrapTot=sum(SI['Scrap to smelters'],SI['Scrap to refineries']);
const tailTot=sum(SO['Tailings (mill)'],SO['Tailings (SX-EW)']);
const slagTot=sum(SO['Slag (smelting)'],SO['Slag (fire ref.)'],SO['Slag (electroref.)']);
const stockTot=sum(SO['Exchange stock change'],SO['User stock change'],SO['Producer stock change']);
const f1b=document.getElementById('f1b');
f1b.appendChild(legend(null,[{name:'Ore in situ',color:'var(--k-ore)',sq:1},{name:'Scrap (smelters + refineries)',color:'var(--k-scrap)',sq:1}]));
areaChart(f1b,{aria:'Boundary inputs, Mt Cu',series:[{name:'Ore in situ',values:SI['Ore in situ (nature)'],color:'var(--k-ore)'},{name:'Scrap',values:scrapTot,color:'var(--k-scrap)'}],max:36000,yfmt:v=>fmt(v/1000,0),nticks:6});
const f1c=document.getElementById('f1c');
f1c.appendChild(legend(null,[{name:'Final use',color:'var(--k-use)',sq:1},{name:'Stock changes',color:'var(--k-stock)',sq:1},{name:'Tailings',color:'var(--k-loss)',sq:1},{name:'Slag',color:'var(--k-slag)',sq:1}]));
areaChart(f1c,{aria:'Boundary outputs, Mt Cu',series:[{name:'Final use',values:SO['Final use'],color:'var(--k-use)'},{name:'Stock changes',values:stockTot,color:'var(--k-stock)'},{name:'Tailings',values:tailTot,color:'var(--k-loss)'},{name:'Slag',values:slagTot,color:'var(--k-slag)'}],max:36000,yfmt:v=>fmt(v/1000,0),nticks:6});

(function(){const i10=Y.indexOf(2010);
  const r=(name,a,cls)=>`<tr class="${cls||''}"><td>${name}</td><td class="num">${fmt(a[0]/1000,2)}</td><td class="num">${fmt(a[i10]/1000,2)}</td><td class="num">${fmt(a[N-1]/1000,2)}</td><td class="num">${a[0]>0?'×'+(a[N-1]/a[0]).toFixed(2):'–'}</td></tr>`;
  const inTot=sum(SI['Ore in situ (nature)'],scrapTot);const outTot=sum(SO['Final use'],tailTot,slagTot,stockTot);
  document.getElementById('f1d').innerHTML=`<table><thead><tr><th>Boundary flow</th><th>1995</th><th>2010</th><th>2024</th><th>2024 / 1995</th></tr></thead><tbody>
  <tr class="sec"><td colspan="5">Inputs</td></tr>${r('Ore in situ, from nature',SI['Ore in situ (nature)'])}${r('Scrap to smelters',SI['Scrap to smelters'])}${r('Scrap to fire refineries',SI['Scrap to refineries'])}${r('Total in',inTot,'sum')}
  <tr class="sec"><td colspan="5">Outputs</td></tr>${r('Final use',SO['Final use'])}${r('Tailings (mill + SX‑EW)',tailTot)}${r('Slag (smelting, fire and electrorefining)',slagTot)}${r('Stock changes (exchange, user, producer)',stockTot)}${r('Total out',outTot,'sum')}</tbody></table>`;})();

// ================= year switch (Sankey + trade) =================
const switches=[...document.querySelectorAll('.yswitch')];
function setYear(y){state.year=y;switches.forEach(s=>s.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',+b.dataset.y===y?'true':'false')));renderSankey();renderRibbons();renderBars();}
switches.forEach(s=>s.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>setYear(+b.dataset.y))));

// ================= PART 2: Sankey =================
const lg2=document.getElementById('lg2');
legend(lg2,[{name:'Ore',color:'var(--k-ore)',sq:1},{name:'Concentrate',color:'var(--k-conc)',sq:1},{name:'Matte',color:'var(--k-matte)',sq:1},{name:'Anode',color:'var(--k-anode)',sq:1},{name:'Refined (ER, SX‑EW)',color:'var(--k-ref)',sq:1},{name:'Cathode',color:'var(--k-cath)',sq:1},{name:'Scrap',color:'var(--k-scrap)',sq:1},{name:'Tailings, slag',color:'var(--k-loss)',sq:1},{name:'Stock change',color:'var(--k-stock)',sq:1},{name:'Final use',color:'var(--k-use)',sq:1}]);
lg2.insertAdjacentHTML('beforeend','<span class="key"><i class="hatch" style="color:var(--ink2)"></i>hatched: crosses a border</span>');
function sankeyTable(root){const src=D.sankey[String(state.year)];const lab=id=>src.nodes.find(n=>n.id===id).label;
  dataTable(root,['To','Commodity','Crosses a border','kt Cu'],src.links.map(l=>[lab(l.source),lab(l.target),l.label,l.traded?'yes':'no',fmt(l.value,0)]),'From');}
function renderSankey(){
  const root=document.getElementById('f2a');root.innerHTML='';
  if(!(window.d3&&d3.sankey)){root.innerHTML='<p class="small">The Sankey needs the d3 library, which did not load; the flows are in the table below.</p>';sankeyTable(root);return;}
  const src=D.sankey[String(state.year)];
  const graph={nodes:src.nodes.map(n=>Object.assign({},n)),links:src.links.map(l=>Object.assign({},l))};
  const W=980,H=560,m={t:18,r:150,b:18,l:12};
  const sk=d3.sankey().nodeId(d=>d.id).nodeWidth(14).nodePadding(18).nodeAlign(d=>d.layer).extent([[m.l,m.t],[W-m.r,H-m.b]]).iterations(64);
  sk(graph);
  const wrap=document.createElement('div');wrap.className='chart';
  const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'World copper system '+state.year});
  const defs=svgEl('defs',{});
  ['concentrate','matte','anode','cathode'].forEach(k=>{const p=svgEl('pattern',{id:'hatch-'+k,patternUnits:'userSpaceOnUse',width:7,height:7,patternTransform:'rotate(45)'});
    const a=svgEl('rect',{width:7,height:7,style:`fill:${KIND[k]};fill-opacity:.22`});const b=svgEl('rect',{width:3,height:7,style:`fill:${KIND[k]};fill-opacity:.85`});p.appendChild(a);p.appendChild(b);defs.appendChild(p);});
  svg.appendChild(defs);
  const path=d3.sankeyLinkHorizontal();const linkEls=[];
  graph.links.forEach(l=>{const p=svgEl('path',{d:path(l),class:'sklink',style:`fill:none;stroke:${l.traded?'url(#hatch-'+l.kind+')':KIND[l.kind]};stroke-opacity:${l.traded?1:.45};stroke-width:${Math.max(1,l.width).toFixed(1)}`});p.__l=l;svg.appendChild(p);linkEls.push(p);});
  graph.nodes.forEach(n=>{const r=svgEl('rect',{x:n.x0.toFixed(1),y:n.y0.toFixed(1),width:(n.x1-n.x0).toFixed(1),height:Math.max(1.5,n.y1-n.y0).toFixed(1),rx:2,style:`fill:${KIND[n.kind]}`});r.__n=n;svg.appendChild(r);});
  const cols={};graph.nodes.forEach(n=>(cols[n.layer]=cols[n.layer]||[]).push(n));
  Object.keys(cols).forEach(k=>{const labs=cols[k].slice().sort((a,b)=>a.y0-b.y0).map(n=>({n,y:(n.y0+n.y1)/2,ty:(n.y0+n.y1)/2}));const gap=30;
    for(let i=1;i<labs.length;i++)if(labs[i].ty-labs[i-1].ty<gap)labs[i].ty=labs[i-1].ty+gap;
    const over=labs.length?labs[labs.length-1].ty+14-(H-m.b):0;if(over>0)labs.forEach(l=>l.ty-=over);
    for(let i=labs.length-2;i>=0;i--)if(labs[i+1].ty-labs[i].ty<gap)labs[i].ty=labs[i+1].ty-gap;
    labs.forEach(l=>{const n=l.n;const right=n.layer<=3||n.layer===7;const x=right?n.x1+8:n.x0-8;const anchor=right?'start':'end';
      if(Math.abs(l.ty-l.y)>2)svg.appendChild(svgEl('line',{x1:right?n.x1+2:n.x0-2,y1:l.y,x2:right?n.x1+6:n.x0-6,y2:l.ty,class:'axis'}));
      const t=svgEl('text',{x,y:l.ty-2,class:'lab halo','text-anchor':anchor});t.textContent=n.label;svg.appendChild(t);
      const v=svgEl('text',{x,y:l.ty+11,class:'val halo','text-anchor':anchor});v.textContent=mt(n.value);svg.appendChild(v);});});
  wrap.appendChild(svg);
  attachTipEl(wrap,svg,W,(e,px)=>{if(!e){linkEls.forEach(p=>p.classList.remove('dim'));return null;}
    const l=e.target.__l,n=e.target.__n;
    if(l){linkEls.forEach(p=>p.classList.toggle('dim',p.__l!==l));const s=l.source,t=l.target;const out=s.sourceLinks.reduce((a,x)=>a+x.value,0);
      return {x:px,html:tipRows(l.label+(l.traded?' · crosses a border':'')+' · '+state.year,[{name:s.label+' → '+t.label,color:KIND[l.kind],val:fmt(l.value,0)+' kt'},{name:'share of '+s.label+' output',color:'transparent',val:pct(l.value/out)}])};}
    if(n){linkEls.forEach(p=>p.classList.toggle('dim',p.__l.source!==n&&p.__l.target!==n));const inn=n.targetLinks.reduce((a,x)=>a+x.value,0),out=n.sourceLinks.reduce((a,x)=>a+x.value,0);
      return {x:px,html:tipRows(n.label+' · '+state.year,[{name:'in',color:'transparent',val:fmt(inn,0)+' kt'},{name:'out',color:'transparent',val:fmt(out,0)+' kt'}])};}
    linkEls.forEach(p=>p.classList.remove('dim'));return null;});
  root.appendChild(wrap);sankeyTable(root);
}

// ================= PART 3: scrap versus primary =================
const S=D.scrap;
const f3a=document.getElementById('f3a');
f3a.appendChild(legend(null,[{name:'World',color:'var(--ink)'},{name:'China',color:'var(--c-china)'},{name:'EU‑14',color:'var(--c-eu)'},{name:'United States',color:'var(--c-us)'},{name:'Japan',color:'var(--c-4)'},{name:'South Korea',color:'var(--c-ref)'},{name:'World, ICSG-based prior',dash:true}]));
lineChart(f3a,{series:[
  {name:'World',values:S.World.ratio,color:'var(--ink)',width:2.5},
  {name:'China',values:S.China.ratio,color:'var(--c-china)'},
  {name:'EU‑14',values:S['EU-14'].ratio,color:'var(--c-eu)'},
  {name:'United States',values:S['United States'].ratio,color:'var(--c-us)'},
  {name:'Japan',values:S.Japan.ratio,color:'var(--c-4)'},
  {name:'South Korea',values:S['South Korea'].ratio,color:'var(--c-ref)'},
  {name:'World prior',values:S.World.ratio_prior,color:'var(--muted)',dash:true,label:false}],
  tipfmt:v=>fmt(v,2),endVal:v=>fmt(v,2),yfmt:v=>fmt(v,1),ydomain:[0,1.0],nticks:5});

const gcol=r=>r==='China'?'var(--c-china)':r==='United States'?'var(--c-us)':r==='Japan'?'var(--c-4)':r==='Korea, Rep.'?'var(--c-ref)':['Germany','Poland','Spain','Bulgaria','Italy','France','Belgium','Netherlands','Austria','Sweden','Greece','Portugal','Slovakia','Luxembourg'].includes(r)?'var(--c-eu)':'var(--c-rest)';
const f3b=document.getElementById('f3b');
f3b.appendChild(legend(null,[{name:'China',color:'var(--c-china)',sq:1},{name:'EU‑14 member',color:'var(--c-eu)',sq:1},{name:'United States',color:'var(--c-us)',sq:1},{name:'Japan',color:'var(--c-4)',sq:1},{name:'South Korea',color:'var(--c-ref)',sq:1},{name:'Other',color:'var(--c-rest)',sq:1}]));
barChart(f3b,{rows:D.scrap_regions_2024.map(r=>({name:rname(r.region),v:r.share,color:gcol(r.region),
  tip:[{name:'Scrap share',color:gcol(r.region),val:pct(r.share)},{name:'Scrap, kt Cu',color:'transparent',val:fmt(r.scrap,0)},{name:'Primary, kt Cu',color:'transparent',val:fmt(r.primary,0)}],
  cells:[pct(r.share),fmt(r.scrap,0),fmt(r.primary,0)]})),max:0.5,xfmt:v=>Math.round(v*100)+' %',cols:['Scrap share','Scrap kt Cu','Primary kt Cu']});

const f3c=document.getElementById('f3c');
f3c.appendChild(legend(null,[{name:'China',color:'var(--c-china)',sq:1},{name:'EU‑14',color:'var(--c-eu)',sq:1},{name:'United States',color:'var(--c-us)',sq:1},{name:'Japan',color:'var(--c-4)',sq:1},{name:'South Korea',color:'var(--c-ref)',sq:1},{name:'Rest of world',color:'var(--c-rest)',sq:1}]));
areaChart(f3c,{aria:'Scrap feed volume by region group, kt Cu',series:[['China','var(--c-china)'],['EU-14','var(--c-eu)'],['United States','var(--c-us)'],['Japan','var(--c-4)'],['South Korea','var(--c-ref)'],['Rest of world','var(--c-rest)']].map(([g,c])=>({name:g==='EU-14'?'EU‑14':g,values:S[g].scrap,color:c})),nticks:5});

// ================= PART 4: concentration =================
const C=D.conc;const GROUPS=[['China','var(--c-china)'],['EU-14','var(--c-eu)'],['United States','var(--c-us)'],['Chile','var(--c-chile)'],['Peru','var(--c-peru)'],['Rest of world','var(--c-rest)']];
legend(document.getElementById('lg4'),GROUPS.map(([g,c])=>({name:g==='EU-14'?'EU‑14':g,color:c,sq:1})));
const STEPS=['Mining','Concentrate','Smelting','Refined cathode','Final use','Scrap feed'];
const STEPSUB={'Mining':'ore, producing region','Concentrate':'producing region','Smelting':'matte, processing region','Refined cathode':'processing region','Final use':'consuming region','Scrap feed':'secondary input, consuming region'};
const f4a=document.getElementById('f4a');
STEPS.forEach(st=>{const p=document.createElement('div');p.className='panel';p.innerHTML=`<h4>${st}</h4><p class="sub">${STEPSUB[st]} · China ${pct(C[st].groups.China[0])} → ${pct(C[st].groups.China[N-1])}</p>`;f4a.appendChild(p);
  stackChart(p,{title:st,series:GROUPS.map(([g,c])=>({name:g==='EU-14'?'EU‑14':g,color:c,values:C[st].groups[g]}))});});
(function(){const cols=[];const rows=Y.map((y,i)=>[y]);STEPS.forEach(st=>GROUPS.slice(0,5).forEach(([g])=>{cols.push(st+' · '+g);rows.forEach((r,i)=>r.push(pct(C[st].groups[g][i])));}));dataTable(f4a.parentElement,cols,rows,'Year');})();

dumbbell(document.getElementById('f4b'),{max:0.55,bands:[{from:0.15,to:0.25,label:'moderate'},{from:0.25,to:0.55,label:'high'}],
  rows:STEPS.map(st=>({name:st,a:C[st].hhi[0],b:C[st].hhi[N-1],color:'var(--ink)',
    tip:[{name:'HHI 1995',color:'var(--surface)',val:C[st].hhi[0].toFixed(3)},{name:'HHI 2024',color:'var(--ink)',val:C[st].hhi[N-1].toFixed(3)},{name:'Top‑3 share 2024',color:'transparent',val:pct(C[st].top3[N-1])},{name:'Leader 2024',color:'transparent',val:rname(C[st].leader[N-1])+' '+pct(C[st].leader_share[N-1])}],
    cells:[C[st].hhi[0].toFixed(3),C[st].hhi[N-1].toFixed(3),pct(C[st].top3[0]),pct(C[st].top3[N-1]),rname(C[st].leader[0])+' '+pct(C[st].leader_share[0]),rname(C[st].leader[N-1])+' '+pct(C[st].leader_share[N-1])]})),
  cols:['HHI 1995','HHI 2024','Top‑3 1995','Top‑3 2024','Leader 1995','Leader 2024']});
document.getElementById('f4b').insertBefore(legend(null,[{name:'1995',color:'transparent',sq:1},{name:'2024',color:'var(--ink)',sq:1}]),document.getElementById('f4b').firstChild);
document.querySelector('#f4b .legend .key i').style.cssText='background:var(--surface);border:2px solid var(--ink2);width:9px;height:9px;border-radius:50%';

const I=D.conc_import;const f4c=document.getElementById('f4c');
f4c.appendChild(legend(null,[{name:'China',color:'var(--c-china)'},{name:'EU‑14',color:'var(--c-eu)'},{name:'United States',color:'var(--c-us)'}]));
lineChart(f4c,{width:520,height:280,rightPad:112,series:[
  {name:'China',values:I.China.import_share,color:'var(--c-china)'},
  {name:'EU‑14',values:I['EU-14'].import_share,color:'var(--c-eu)'},
  {name:'United States',values:I['United States'].import_share,color:'var(--c-us)'}],
  tipfmt:v=>pct(v),endVal:v=>pct(v),yfmt:v=>Math.round(v*100)+' %',ydomain:[0,1],nticks:4});

(function(){const cell=(st,g)=>{const a=C[st].groups[g];return `<td class="num">${(a[N-1]*100).toFixed(1)} <span class="small">(${(a[0]*100).toFixed(1)})</span></td>`;};
  document.getElementById('f4d').innerHTML=`<table><thead><tr><th>Step</th><th><span class="sw" style="background:var(--c-china)"></span>China</th><th><span class="sw" style="background:var(--c-eu)"></span>EU‑14</th><th><span class="sw" style="background:var(--c-us)"></span>United States</th><th><span class="sw" style="background:var(--c-chile)"></span>Chile</th><th><span class="sw" style="background:var(--c-peru)"></span>Peru</th><th>Leader 2024</th><th>Top‑3 share</th><th>HHI</th></tr></thead><tbody>`+
  STEPS.map(st=>`<tr><td>${st}</td>${cell(st,'China')}${cell(st,'EU-14')}${cell(st,'United States')}${cell(st,'Chile')}${cell(st,'Peru')}<td>${rname(C[st].leader[N-1])} <span class="small">${pct(C[st].leader_share[N-1])}</span></td><td class="num">${pct(C[st].top3[N-1])} <span class="small">(${pct(C[st].top3[0])})</span></td><td class="num">${C[st].hhi[N-1].toFixed(3)} <span class="small">(${C[st].hhi[0].toFixed(3)})</span></td></tr>`).join('')+
  `</tbody></table><p class="small" style="margin:8px 0 0">2024 values in %, 1995 in parentheses.</p>`;})();

// ================= PART 5: trade structure =================
const T=D.trade,COMMS=T.commodities,RIB=['Concentrate','Anode','Cathode'];
function tradeOrder(){const R=T.years['2024'].regions;const named=T.order.filter(r=>r!=='Other');named.sort((a,b)=>R[b].net-R[a].net);return named.concat(['Other']);}
legend(document.getElementById('lg5a'),RIB.map(c=>({name:c,color:COMM[c],sq:1})));
function renderRibbons(){
  const root=document.getElementById('f5a');root.innerHTML='';
  const yr=T.years[String(state.year)],order=tradeOrder(),idx={};order.forEach((r,i)=>idx[r]=i);
  const nR=order.length,rowH=26,boxW=108,W=1000,m={t:42,r:6,b:6,l:6};const nc=RIB.length+1,gapW=(W-m.l-m.r-nc*boxW)/(nc-1);const H=m.t+nR*rowH+m.b;
  let maxv=0;Object.values(T.years).forEach(yy=>RIB.forEach(c=>{const ex={},im={};yy.legs[c].forEach(l=>{ex[l.f]=(ex[l.f]||0)+l.v;im[l.t]=(im[l.t]||0)+l.v;});Object.values(ex).concat(Object.values(im)).forEach(v=>{if(v>maxv)maxv=v;});}));
  const k=(rowH-6)/maxv;const colX=j=>m.l+j*(boxW+gapW),rowY=i=>m.t+i*rowH;
  const wrap=document.createElement('div');wrap.className='chart';const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'Trade-linked copper flows '+state.year});
  const ribs=[];
  RIB.forEach((c,j)=>{const legs=yr.legs[c].filter(l=>idx[l.f]!=null&&idx[l.t]!=null).map(l=>Object.assign({c},l));
    const outSum={},inSum={};legs.forEach(l=>{outSum[l.f]=(outSum[l.f]||0)+l.v;inSum[l.t]=(inSum[l.t]||0)+l.v;});
    const outOff={},inOff={};
    legs.slice().sort((a,b)=>idx[a.t]-idx[b.t]||idx[a.f]-idx[b.f]).forEach(l=>{if(outOff[l.f]==null)outOff[l.f]=rowY(idx[l.f])+rowH/2-k*outSum[l.f]/2;l.ya=outOff[l.f];outOff[l.f]+=k*l.v;});
    legs.slice().sort((a,b)=>idx[a.f]-idx[b.f]||idx[a.t]-idx[b.t]).forEach(l=>{if(inOff[l.t]==null)inOff[l.t]=rowY(idx[l.t])+rowH/2-k*inSum[l.t]/2;l.yb=inOff[l.t];inOff[l.t]+=k*l.v;});
    const xR=colX(j)+boxW,xL=colX(j+1),xm=(xR+xL)/2;
    legs.sort((a,b)=>b.v-a.v).forEach(l=>{const w=Math.max(0.8,k*l.v);const d=`M${xR},${l.ya.toFixed(1)} C${xm},${l.ya.toFixed(1)} ${xm},${l.yb.toFixed(1)} ${xL},${l.yb.toFixed(1)} L${xL},${(l.yb+w).toFixed(1)} C${xm},${(l.yb+w).toFixed(1)} ${xm},${(l.ya+w).toFixed(1)} ${xR},${(l.ya+w).toFixed(1)} Z`;
      const p=svgEl('path',{d,class:'rib',style:`fill:${COMM[c]};fill-opacity:.55`});l.share=l.v/outSum[l.f];p.__l=l;svg.appendChild(p);ribs.push(p);});
    const hx=xR+gapW/2;const t=svgEl('text',{x:hx,y:m.t-24,'text-anchor':'middle',class:'lab'});t.textContent=c;svg.appendChild(t);
    const u=svgEl('text',{x:hx,y:m.t-9,'text-anchor':'middle'});u.textContent=mt(T.ts[c].trade[Y.indexOf(state.year)])+' traded · exporter → importer';svg.appendChild(u);});
  const boxes=[];
  for(let j=0;j<nc;j++)order.forEach((r,i)=>{const x=colX(j),y=rowY(i)+2;const b=svgEl('rect',{x,y,width:boxW,height:rowH-4,rx:3,class:'rbox'+(state.focus===r?' on':'')});b.__r=r;svg.appendChild(b);boxes.push(b);
    const t=svgEl('text',{x:x+boxW/2,y:y+(rowH-4)/2+4,'text-anchor':'middle',class:'rlab'});t.textContent=rname(r);svg.appendChild(t);});
  const applyFocus=()=>ribs.forEach(p=>p.classList.toggle('dim',!!state.focus&&p.__l.f!==state.focus&&p.__l.t!==state.focus));
  applyFocus();
  svg.addEventListener('click',e=>{const r=e.target.__r;if(!r)return;state.focus=state.focus===r?null:r;boxes.forEach(b=>b.classList.toggle('on',b.__r===state.focus));applyFocus();});
  wrap.appendChild(svg);
  attachTipEl(wrap,svg,W,(e,px)=>{if(!e){applyFocus();return null;}const l=e.target.__l,r=e.target.__r;
    if(l){ribs.forEach(p=>p.classList.toggle('dim',p.__l!==l));return {x:px,html:tipRows(l.c+' · '+state.year,[{name:rname(l.f)+' → '+rname(l.t),color:COMM[l.c],val:fmt(l.v,0)+' kt'},{name:'share of '+rname(l.f)+' '+l.c.toLowerCase()+' exports',color:'transparent',val:pct(l.share)}])};}
    if(r){ribs.forEach(p=>p.classList.toggle('dim',p.__l.f!==r&&p.__l.t!==r));const R=yr.regions[r];return {x:px,html:tipRows(rname(r)+' · '+state.year+' · kt Cu',COMMS.map(c=>({name:c,color:COMM[c],val:fmt(R.exports[c],0)+' out · '+fmt(R.imports[c],0)+' in'})))};}
    applyFocus();return null;});
  root.appendChild(wrap);
  dataTable(root,['To','Commodity','kt Cu'],RIB.flatMap(c=>yr.legs[c].slice().sort((a,b)=>b.v-a.v).slice(0,25).map(l=>[rname(l.f),rname(l.t),c,fmt(l.v,0)])),'From');
}

const lg5b=document.getElementById('lg5b');legend(lg5b,COMMS.map(c=>({name:c,color:COMM[c],sq:1})));
lg5b.insertAdjacentHTML('beforeend','<span class="key"><i class="tick"></i>net trade</span>');
function renderBars(){
  const root=document.getElementById('f5b');root.innerHTML='';
  const yr=T.years[String(state.year)],order=tradeOrder();
  const rowH=22,W=1000,m={t:26,r:16,b:30,l:150};const H=m.t+order.length*rowH+m.b;const iw=W-m.l-m.r;
  const sv=o=>Object.values(o).reduce((a,b)=>a+b,0);
  let mx=0;Object.values(T.years).forEach(yy=>Object.values(yy.regions).forEach(r=>{mx=Math.max(mx,sv(r.exports),sv(r.imports));}));
  const x0=m.l+iw/2;const X=v=>x0+iw/2*v/mx;
  const wrap=document.createElement('div');wrap.className='chart';const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'Exports and imports by region '+state.year});
  niceTicks(0,mx,4).forEach(t=>{if(t>mx)return;[t,-t].forEach(v=>{if(v===0&&t!==0)return;svg.appendChild(svgEl('line',{x1:X(v),x2:X(v),y1:m.t,y2:H-m.b,class:v===0?'axis':'grid'}));const tx=svgEl('text',{x:X(v),y:H-m.b+16,'text-anchor':'middle'});tx.textContent=fmt(Math.abs(v)/1000,0);svg.appendChild(tx);});});
  const ax1=svgEl('text',{x:m.l,y:m.t-10,class:'lab'});ax1.textContent='← imports, Mt Cu';svg.appendChild(ax1);
  const ax2=svgEl('text',{x:W-m.r,y:m.t-10,'text-anchor':'end',class:'lab'});ax2.textContent='exports, Mt Cu →';svg.appendChild(ax2);
  const segs=[];
  order.forEach((r,i)=>{const y=m.t+i*rowH+3,h=rowH-6;const R=yr.regions[r];
    const lab=svgEl('text',{x:m.l-10,y:y+h/2+4,'text-anchor':'end',class:'lab'});lab.textContent=rname(r);svg.appendChild(lab);
    let xe=x0,xi=x0;COMMS.forEach(c=>{const ve=R.exports[c],vi=R.imports[c];
      if(ve>0){const w=X(ve)-x0;const s=svgEl('rect',{x:xe.toFixed(1),y,width:Math.max(w,0.5).toFixed(1),height:h,class:'seg',style:`fill:${COMM[c]};stroke:var(--surface);stroke-width:1`});s.__s={r,c,v:ve,dir:'exports'};svg.appendChild(s);segs.push(s);xe+=w;}
      if(vi>0){const w=X(vi)-x0;const s=svgEl('rect',{x:(xi-w).toFixed(1),y,width:Math.max(w,0.5).toFixed(1),height:h,class:'seg',style:`fill:${COMM[c]};stroke:var(--surface);stroke-width:1`});s.__s={r,c,v:vi,dir:'imports'};svg.appendChild(s);segs.push(s);xi-=w;}});
    svg.appendChild(svgEl('line',{x1:X(R.net).toFixed(1),x2:X(R.net).toFixed(1),y1:y-1,y2:y+h+1,style:'stroke:var(--ink);stroke-width:2'}));});
  wrap.appendChild(svg);
  attachTipEl(wrap,svg,W,(e,px)=>{if(!e)return null;const s=e.target.__s;if(!s)return null;const R=yr.regions[s.r];
    return {x:px,html:tipRows(rname(s.r)+' · '+state.year,[{name:s.c+' '+s.dir,color:COMM[s.c],val:fmt(s.v,0)+' kt'},{name:'all exports',color:'transparent',val:fmt(sv(R.exports),0)+' kt'},{name:'all imports',color:'transparent',val:fmt(sv(R.imports),0)+' kt'},{name:'net',color:'transparent',val:fmt(R.net,0)+' kt'}])};});
  root.appendChild(wrap);
  dataTable(root,COMMS.map(c=>'Exports '+c).concat(COMMS.map(c=>'Imports '+c),['Net']),order.map(r=>{const R=yr.regions[r];return [rname(r),...COMMS.map(c=>fmt(R.exports[c],0)),...COMMS.map(c=>fmt(R.imports[c],0)),fmt(R.net,0)];}),'Region');
}

const TT=T.ts;const f5c=document.getElementById('f5c');
f5c.appendChild(legend(null,COMMS.map(c=>({name:c,color:COMM[c]}))));
lineChart(f5c,{width:520,height:280,rightPad:112,series:COMMS.map(c=>({name:c,values:TT[c].trade.map((v,i)=>v/TT[c].output[i]),color:COMM[c]})),
  tipfmt:v=>pct(v),endVal:v=>pct(v),yfmt:v=>Math.round(v*100)+' %',ydomain:[0,0.7],nticks:7});
const f5d=document.getElementById('f5d');
f5d.appendChild(legend(null,COMMS.map(c=>({name:c,color:COMM[c],sq:1}))));
areaChart(f5d,{aria:'World cross-border trade by commodity, Mt Cu',series:COMMS.map(c=>({name:c,values:TT[c].trade,color:COMM[c]})),yfmt:v=>fmt(v/1000,0),nticks:5});

setYear(2024);

// ================= PART 6: end use categories (IWCC data, own 2012–2024 axis) =================
// The helpers above close over the run's 1995–2024 axis (Y, N); chapter 6 draws on the
// payload's years and leaves them untouched. Categories keep the payload's (legend) order.
const E=D.enduse,EY=E.years,EN=EY.length;
const EDASH=['','7 3','1.5 3','9 3 2 3','4 2.5 1.5 2.5'];
const ECAT=E.categories.map((c,k)=>({label:c.label,group:c.group,kt:c.kt,color:`var(--eu${k+1})`,
  dash:EDASH[E.categories.slice(0,k).filter(o=>o.group===c.group).length]}));
const ESHARE=ECAT.map(c=>c.kt.map((v,i)=>v/E.total[i]));
const pct1=v=>(v*100).toFixed(1)+' %';
const lineKey=c=>`<svg width="24" height="10" aria-hidden="true"><line x1="1" x2="23" y1="5" y2="5" style="stroke:${c.color};stroke-width:2.5${c.dash?';stroke-dasharray:'+c.dash:''}"/></svg>`;
document.getElementById('lg6').innerHTML=E.groups.map(g=>`<div class="grp"><b>${g}</b>`+ECAT.filter(c=>c.group===g).map(c=>`<span class="key">${lineKey(c)}${c.label}</span>`).join('')+'</div>').join('');
// tooltip kept inside its chart: the 17-row readouts are taller than the pointer's headroom
function tip6(wrap,svg,W,fn){const tip=document.createElement('div');tip.className='tip';tip.hidden=true;wrap.appendChild(tip);
  svg.addEventListener('mousemove',e=>{const r=svg.getBoundingClientRect();const res=fn((e.clientX-r.left)*W/r.width,(e.clientY-r.top)*W/r.width);if(!res){tip.hidden=true;return;}
    tip.hidden=false;tip.innerHTML=res.html;const ax=res.x/W*r.width,left=ax>r.width*0.6;
    tip.style.top=Math.max(0,Math.min(e.clientY-r.top-10,r.height-(tip.offsetHeight||0)))+'px';tip.style.left=left?'':(ax+14)+'px';tip.style.right=left?(r.width-ax+14)+'px':'';});
  svg.addEventListener('mouseleave',()=>{tip.hidden=true;fn(null);});}

(function(){ // chart 1: 16 category lines, world-total strip above on its own scale
  const root=document.getElementById('f6a');
  const W=960,m={t:16,r:92,b:30,l:58},hS=84,gap=26,hM=340,H=m.t+hS+gap+hM+m.b,iw=W-m.l-m.r,y0=m.t+hS+gap;
  const X=i=>m.l+iw*i/(EN-1);
  const sTop=Math.ceil(Math.max(...E.total)/5000)*5000,Ys=v=>m.t+hS*(1-v/sTop);
  const ticks=niceTicks(0,Math.max(...ECAT.flatMap(c=>c.kt)),6),mTop=ticks[ticks.length-1],Ym=v=>y0+hM*(1-v/mTop);
  const wrap=document.createElement('div');wrap.className='chart scroll';
  const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':`World copper end use by IWCC category, kt Cu, ${EY[0]}–${EY[EN-1]}`});
  const grid=(y,v)=>{svg.appendChild(svgEl('line',{x1:m.l,x2:W-m.r,y1:y,y2:y,class:v===0?'axis':'grid'}));const t=svgEl('text',{x:m.l-8,y:y+4,'text-anchor':'end'});t.textContent=fmt(v);svg.appendChild(t);};
  [0,10000,20000,30000].filter(v=>v<sTop).forEach(v=>grid(Ys(v),v));
  ticks.forEach(v=>grid(Ym(v),v));
  const st=svgEl('text',{x:m.l+6,y:m.t+2,class:'lab'});st.textContent='World total';svg.appendChild(st);
  EY.forEach((y,i)=>{if(i%2)return;const t=svgEl('text',{x:X(i),y:H-m.b+18,'text-anchor':'middle'});t.textContent=y;svg.appendChild(t);});
  const line=(vals,Yf)=>vals.map((v,i)=>(i?'L':'M')+X(i).toFixed(1)+','+Yf(v).toFixed(1)).join(' ');
  svg.appendChild(svgEl('path',{d:line(E.total,Ys),style:'fill:none;stroke:var(--ink);stroke-width:2;stroke-linejoin:round;stroke-linecap:round'}));
  svg.appendChild(svgEl('circle',{cx:X(EN-1),cy:Ys(E.total[EN-1]),r:4,style:'fill:var(--ink);stroke:var(--surface);stroke-width:2'}));
  const el=svgEl('text',{x:X(EN-1)+10,y:Ys(E.total[EN-1])+4,class:'lab'});el.textContent=fmt(E.total[EN-1])+' kt';svg.appendChild(el);
  ECAT.forEach(c=>svg.appendChild(svgEl('path',{d:line(c.kt,Ym),style:`fill:none;stroke:${c.color};stroke-width:2;stroke-linejoin:round;${c.dash?'stroke-dasharray:'+c.dash+';stroke-linecap:butt':'stroke-linecap:round'}`})));
  const xh=svgEl('line',{class:'xh',y1:m.t,y2:y0+hM,x1:-10,x2:-10});svg.appendChild(xh);
  const dot=color=>{const d=svgEl('circle',{r:4,cx:-10,cy:-10,style:`fill:${color};stroke:var(--surface);stroke-width:2;pointer-events:none`});svg.appendChild(d);return d;};
  const dots=ECAT.map(c=>dot(c.color)),dT=dot('var(--ink)');
  wrap.appendChild(svg);
  tip6(wrap,svg,W,px=>{if(px==null){xh.setAttribute('x1',-10);xh.setAttribute('x2',-10);dots.concat([dT]).forEach(d=>d.setAttribute('cx',-10));return null;}
    const i=Math.max(0,Math.min(EN-1,Math.round((px-m.l)/iw*(EN-1))));xh.setAttribute('x1',X(i));xh.setAttribute('x2',X(i));
    ECAT.forEach((c,k)=>{dots[k].setAttribute('cx',X(i));dots[k].setAttribute('cy',Ym(c.kt[i]));});dT.setAttribute('cx',X(i));dT.setAttribute('cy',Ys(E.total[i]));
    return {x:X(i),html:tipRows(EY[i]+' · kt Cu',ECAT.map(c=>({name:c.label,color:c.color,val:fmt(c.kt[i])})).concat([{name:'World total',color:'var(--ink)',val:fmt(E.total[i])}]))};});
  root.appendChild(wrap);
  dataTable(root,EY,ECAT.map(c=>[c.label,...c.kt.map(v=>fmt(v))]).concat([['World total',...E.total.map(v=>fmt(v))]]),'Category · kt Cu');
})();

(function(){ // chart 2: 100 % stacked bars, first category of the legend on top
  const root=document.getElementById('f6b');
  const W=960,m={t:12,r:16,b:30,l:58},H=440,ih=H-m.t-m.b,iw=W-m.l-m.r,slot=iw/EN,bw=Math.min(36,slot*0.55);
  const Yc=v=>m.t+ih*v; // v = cumulative share from the top
  const wrap=document.createElement('div');wrap.className='chart scroll';
  const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':`Composition of world copper end use, share of world total, ${EY[0]}–${EY[EN-1]}`});
  [0,.25,.5,.75,1].forEach(t=>{const y=Yc(1-t);svg.appendChild(svgEl('line',{x1:m.l,x2:W-m.r,y1:y,y2:y,class:t===0?'axis':'grid'}));const tx=svgEl('text',{x:m.l-8,y:y+4,'text-anchor':'end'});tx.textContent=(t*100)+' %';svg.appendChild(tx);});
  const hov=svgEl('rect',{x:-1000,y:m.t,width:slot,height:ih,style:'fill:var(--wash)'});svg.appendChild(hov);
  const segs=EY.map((y,i)=>{const x=m.l+slot*(i+.5)-bw/2;let cum=0;
    const t=svgEl('text',{x:x+bw/2,y:H-m.b+18,'text-anchor':'middle'});t.textContent=y;svg.appendChild(t);
    return ECAT.map((c,k)=>{const a=Yc(cum),b=Yc(cum+=ESHARE[k][i]),top=a+0.5,h=Math.max(0,b-a-1);
      svg.appendChild(k===0?svgEl('path',{d:`M${x},${top+h} V${top+4} a4,4 0 0 1 4,-4 H${x+bw-4} a4,4 0 0 1 4,4 V${top+h} Z`,style:`fill:${c.color}`})
        :svgEl('rect',{x,y:top,width:bw,height:h,style:`fill:${c.color}`}));
      return {a,b};});});
  wrap.appendChild(svg);
  tip6(wrap,svg,W,(px,py)=>{if(px==null){hov.setAttribute('x',-1000);return null;}
    const i=Math.max(0,Math.min(EN-1,Math.floor((px-m.l)/slot)));hov.setAttribute('x',m.l+slot*i);
    const k=segs[i].findIndex(s=>py>=s.a&&py<s.b);
    return {x:m.l+slot*(i+.5),html:tipRows(EY[i]+' · share of world total',ECAT.map((c,j)=>({name:j===k?`<span style="font-weight:600;color:var(--ink)">${c.label}</span>`:c.label,color:c.color,val:pct1(ESHARE[j][i])})).concat([{name:'World total',color:'transparent',val:fmt(E.total[i])+' kt'}]))};});
  root.appendChild(wrap);
  const groups=E.groups.map(g=>[g+' (group)',...EY.map((y,i)=>pct1(ECAT.reduce((s,c,k)=>s+(c.group===g?ESHARE[k][i]:0),0)))]);
  dataTable(root,EY,ECAT.map((c,k)=>[c.label,...ESHARE[k].map(pct1)]).concat(groups),'Category · share of world total');
})();
})();
