const fmt=(v,d)=>v==null?'–':v.toLocaleString('en-US',{minimumFractionDigits:d==null?0:d,maximumFractionDigits:d==null?0:d});
const pct=v=>(v*100).toFixed(v*100>=10?0:1)+' %';
const idx=a=>a.map(v=>v/a[0]*100);
const sum=(...as)=>Y.map((_,i)=>as.reduce((t,a)=>t+a[i],0));

function niceTicks(lo,hi,n){const r=hi-lo||1;const raw=r/n;const p=Math.pow(10,Math.floor(Math.log10(raw)));const c=raw/p;const s=(c<1.5?1:c<3.5?2:c<7.5?5:10)*p;const t=[];for(let v=Math.floor(lo/s)*s;v<=hi+1e-9;v+=s)t.push(+v.toFixed(10));if(t[t.length-1]<hi)t.push(+(t[t.length-1]+s).toFixed(10));return t;}
const svgEl=(t,a)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',t);for(const k in a)e.setAttribute(k,a[k]);return e;};
function legend(el,items){const d=el||document.createElement('div');d.className='legend';d.innerHTML=items.map(s=>`<span class="key"><i class="${s.sq?'sq':''}${s.dash?' dash':''}" style="background:${s.dash?'':s.color}"></i>${s.name}</span>`).join('');return d;}
function tipRows(title,rows){return `<b>${title}</b>`+rows.map(r=>`<div class="row"><span><i style="background:${r.color}"></i>${r.name}</span><span>${r.val}</span></div>`).join('');}
// clampBottom (opt-in): keep a tooltip taller than the pointer's headroom inside the chart
function attachTip(wrap,svg,W,onMove,clampBottom){const tip=document.createElement('div');tip.className='tip';tip.hidden=true;wrap.appendChild(tip);
  svg.addEventListener('mousemove',e=>{const r=svg.getBoundingClientRect();const px=(e.clientX-r.left)*W/r.width;const py=(e.clientY-r.top)*W/r.width;const res=onMove(px,py);if(!res){tip.hidden=true;return;}tip.hidden=false;tip.innerHTML=res.html;const ax=res.x/W*r.width;const left=ax>r.width*0.6;const top=(e.clientY-r.top)-10;tip.style.top=Math.max(0,clampBottom?Math.min(top,r.height-(tip.offsetHeight||0)):top)+'px';tip.style.left=left?'':(ax+14)+'px';tip.style.right=left?(r.width-ax+14)+'px':'';});
  svg.addEventListener('mouseleave',()=>{tip.hidden=true;onMove(null);});}
function dataTable(wrap,cols,rows,rowLabel){const d=document.createElement('details');d.className='data';d.innerHTML='<summary>Data table</summary>';const w=document.createElement('div');w.className='tblwrap';const t=document.createElement('table');t.innerHTML='<thead><tr><th>'+rowLabel+'</th>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead>';const tb=document.createElement('tbody');rows.forEach(r=>{const tr=document.createElement('tr');tr.innerHTML='<td>'+r[0]+'</td>'+r.slice(1).map(v=>'<td class="num">'+v+'</td>').join('');tb.appendChild(tr);});t.appendChild(tb);w.appendChild(t);d.appendChild(w);wrap.appendChild(d);}

// ---------- line chart ----------
function lineChart(root,o){
  const W=o.width||680,H=o.height||300,m={t:14,r:o.rightPad==null?118:o.rightPad,b:30,l:o.leftPad||46};
  const iw=W-m.l-m.r,ih=H-m.t-m.b;
  const all=o.series.flatMap(s=>s.values.filter(v=>v!=null&&isFinite(v)));
  let lo=o.ydomain?o.ydomain[0]:Math.min(o.zero===false?Infinity:0,...all),hi=o.ydomain?o.ydomain[1]:Math.max(...all);
  const ticks=niceTicks(lo,hi,o.nticks||5);lo=Math.min(lo,ticks[0]);hi=Math.max(hi,ticks[ticks.length-1]);
  const X=i=>m.l+iw*i/(N-1),Yy=v=>m.t+ih*(1-(v-lo)/(hi-lo));
  const wrap=document.createElement('div');wrap.className='chart';
  const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':o.aria||''});
  ticks.forEach(t=>{svg.appendChild(svgEl('line',{x1:m.l,x2:W-m.r,y1:Yy(t),y2:Yy(t),class:t===0||(o.baseline!=null&&t===o.baseline)?'axis':'grid'}));const tx=svgEl('text',{x:m.l-8,y:Yy(t)+4,'text-anchor':'end'});tx.textContent=o.yfmt?o.yfmt(t):fmt(t);svg.appendChild(tx);});
  (o.refLines||[]).forEach(r=>{svg.appendChild(svgEl('line',{x1:m.l,x2:W-m.r,y1:Yy(r.y),y2:Yy(r.y),class:'axis','stroke-dasharray':'3 3'}));const tx=svgEl('text',{x:W-m.r-4,y:Yy(r.y)-4,'text-anchor':'end'});tx.textContent=r.label;svg.appendChild(tx);});
  (o.bands||[]).forEach(b=>{const i0=Y.indexOf(b.from),i1=Y.indexOf(b.to);svg.appendChild(svgEl('rect',{x:X(i0),y:m.t,width:X(i1)-X(i0),height:ih,style:'fill:var(--wash)'}));});
  (o.xticks||[1995,2000,2005,2010,2015,2020,2024]).forEach((y,k,arr)=>{const i=Y.indexOf(y);const tx=svgEl('text',{x:X(i),y:H-m.b+18,'text-anchor':o.xticks?(k===0?'start':k===arr.length-1?'end':'middle'):'middle'});tx.textContent=y;svg.appendChild(tx);});
  svg.appendChild(svgEl('line',{x1:m.l,x2:W-m.r,y1:m.t+ih,y2:m.t+ih,class:'axis'}));
  // lines
  o.series.forEach(s=>{const d=s.values.map((v,i)=>(v==null?'':(i===0||s.values[i-1]==null?'M':'L')+X(i).toFixed(1)+','+Yy(v).toFixed(1))).join(' ');
    if(s.area){const a=d+` L${X(N-1).toFixed(1)},${Yy(lo)} L${X(0)},${Yy(lo)} Z`;svg.appendChild(svgEl('path',{d:a,style:`fill:${s.color};opacity:.10`}));}
    svg.appendChild(svgEl('path',{d,style:`fill:none;stroke:${s.color};stroke-width:${s.width||2};stroke-linejoin:round;stroke-linecap:round;${s.dash?'stroke-dasharray:4 4;':''}`}));});
  // end labels with collision relaxation
  if(o.endLabels!==false){const labs=o.series.filter(s=>s.label!==false).map(s=>({s,y:Yy(s.values[N-1]),ty:Yy(s.values[N-1])})).sort((a,b)=>a.y-b.y);
    const gap=13;for(let k=1;k<labs.length;k++)if(labs[k].ty-labs[k-1].ty<gap)labs[k].ty=labs[k-1].ty+gap;
    const over=labs.length?labs[labs.length-1].ty-(m.t+ih):0;if(over>0)labs.forEach(l=>l.ty-=over);
    for(let k=labs.length-2;k>=0;k--)if(labs[k+1].ty-labs[k].ty<gap)labs[k].ty=labs[k+1].ty-gap;
    labs.forEach(l=>{const x0=X(N-1);if(Math.abs(l.ty-l.y)>1)svg.appendChild(svgEl('line',{x1:x0+3,y1:l.y,x2:x0+9,y2:l.ty,class:'axis'}));
      svg.appendChild(svgEl('circle',{cx:x0,cy:l.y,r:4,style:`fill:${l.s.color};stroke:var(--surface);stroke-width:2`}));
      const t=svgEl('text',{x:x0+12,y:l.ty+4,class:'lab'});t.textContent=(l.s.label||l.s.name)+(o.endVal?' '+o.endVal(l.s.values[N-1]):'');svg.appendChild(t);});}
  // hover
  const xh=svgEl('line',{class:'xh',y1:m.t,y2:m.t+ih,x1:-10,x2:-10});svg.appendChild(xh);
  const dots=o.series.map(s=>{const c=svgEl('circle',{r:4.5,cx:-10,cy:-10,style:`fill:${s.color};stroke:var(--surface);stroke-width:2;pointer-events:none`});svg.appendChild(c);return c;});
  wrap.appendChild(svg);
  attachTip(wrap,svg,W,px=>{if(px==null){xh.setAttribute('x1',-10);xh.setAttribute('x2',-10);dots.forEach(c=>c.setAttribute('cx',-10));return null;}
    const i=Math.max(0,Math.min(N-1,Math.round((px-m.l)/iw*(N-1))));xh.setAttribute('x1',X(i));xh.setAttribute('x2',X(i));
    o.series.forEach((s,k)=>{const v=s.values[i];dots[k].setAttribute('cx',v==null?-10:X(i));dots[k].setAttribute('cy',v==null?-10:Yy(v));});
    return {x:X(i),html:tipRows(Y[i],o.series.map(s=>({name:s.name,color:s.color,val:o.tipfmt?o.tipfmt(s.values[i],s):fmt(s.values[i],1)})))};});
  root.appendChild(wrap);
  if(o.table!==false)dataTable(root,o.series.map(s=>s.name),Y.map((y,i)=>[y,...o.series.map(s=>o.tipfmt?o.tipfmt(s.values[i],s):fmt(s.values[i],1))]),'Year');
  return wrap;
}

// ---------- 100% stacked area ----------
function stackChart(root,o){
  const W=o.width||320,H=o.height||170,m={t:8,r:8,b:22,l:34};const iw=W-m.l-m.r,ih=H-m.t-m.b;
  const X=i=>m.l+iw*i/(N-1),Yy=v=>m.t+ih*(1-v);
  const wrap=document.createElement('div');wrap.className='chart';const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,role:'img'});
  [0,.25,.5,.75,1].forEach(t=>{svg.appendChild(svgEl('line',{x1:m.l,x2:W-m.r,y1:Yy(t),y2:Yy(t),class:'grid'}));const tx=svgEl('text',{x:m.l-6,y:Yy(t)+4,'text-anchor':'end'});tx.textContent=(t*100)+'%';svg.appendChild(tx);});
  const cum=Y.map(()=>0);
  o.series.forEach(s=>{const top=s.values.map((v,i)=>cum[i]+v);const d='M'+top.map((v,i)=>X(i).toFixed(1)+','+Yy(v).toFixed(1)).join(' L')+' L'+cum.map((v,i)=>X(i).toFixed(1)+','+Yy(v).toFixed(1)).reverse().join(' L')+' Z';
    svg.appendChild(svgEl('path',{d,style:`fill:${s.color};stroke:var(--surface);stroke-width:1.5;stroke-linejoin:round`}));s.values.forEach((v,i)=>cum[i]+=v);});
  [1995,2010,2024].forEach(y=>{const i=Y.indexOf(y);const tx=svgEl('text',{x:X(i),y:H-m.b+15,'text-anchor':i===0?'start':i===N-1?'end':'middle'});tx.textContent=y;svg.appendChild(tx);});
  const xh=svgEl('line',{class:'xh',y1:m.t,y2:m.t+ih,x1:-10,x2:-10});svg.appendChild(xh);wrap.appendChild(svg);
  attachTip(wrap,svg,W,px=>{if(px==null){xh.setAttribute('x1',-10);xh.setAttribute('x2',-10);return null;}const i=Math.max(0,Math.min(N-1,Math.round((px-m.l)/iw*(N-1))));xh.setAttribute('x1',X(i));xh.setAttribute('x2',X(i));
    return {x:X(i),html:tipRows(o.title+' · '+Y[i],o.series.map(s=>({name:s.name,color:s.color,val:pct(s.values[i])})))};});
  root.appendChild(wrap);
}

// ---------- horizontal bars ----------
function barChart(root,o){
  const rowH=26,W=560,m={t:6,r:70,b:26,l:130};const H=m.t+rowH*o.rows.length+m.b;const iw=W-m.l-m.r;
  const hi=o.max||Math.max(...o.rows.map(r=>r.v));const X=v=>m.l+iw*v/hi;
  const wrap=document.createElement('div');wrap.className='chart';const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,role:'img'});
  niceTicks(0,hi,4).forEach(t=>{if(t>hi)return;svg.appendChild(svgEl('line',{x1:X(t),x2:X(t),y1:m.t,y2:H-m.b,class:t===0?'axis':'grid'}));const tx=svgEl('text',{x:X(t),y:H-m.b+16,'text-anchor':'middle'});tx.textContent=o.xfmt(t);svg.appendChild(tx);});
  const bars=[];o.rows.forEach((r,k)=>{const y=m.t+rowH*k+4,h=18,w=Math.max(0,X(r.v)-m.l);
    const p=svgEl('path',{d:`M${m.l},${y} H${m.l+w-4} a4,4 0 0 1 4,4 v${h-8} a4,4 0 0 1 -4,4 H${m.l} Z`,style:`fill:${r.color}`});svg.appendChild(p);bars.push({y,h,r});
    const lab=svgEl('text',{x:m.l-8,y:y+13,'text-anchor':'end',class:'lab'});lab.textContent=r.name;svg.appendChild(lab);
    const val=svgEl('text',{x:m.l+w+6,y:y+13,class:'val'});val.textContent=o.xfmt(r.v);svg.appendChild(val);});
  wrap.appendChild(svg);
  attachTip(wrap,svg,W,(px,py)=>{if(px==null)return null;const b=bars.find(b=>py>=b.y-2&&py<=b.y+b.h+2);if(!b)return null;return {x:px,html:tipRows(b.r.name,b.r.tip)};});
  root.appendChild(wrap);
  dataTable(root,o.cols,o.rows.map(r=>[r.name,...r.cells]),'Region');
}

// ---------- dumbbell ----------
function dumbbell(root,o){
  const rowH=34,W=560,m={t:10,r:24,b:28,l:132};const H=m.t+rowH*o.rows.length+m.b;const iw=W-m.l-m.r;const hi=o.max;const X=v=>m.l+iw*v/hi;
  const wrap=document.createElement('div');wrap.className='chart';const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,role:'img'});
  (o.bands||[]).forEach(b=>{svg.appendChild(svgEl('rect',{x:X(b.from),y:m.t,width:X(b.to)-X(b.from),height:H-m.t-m.b,style:`fill:var(--wash)`}));const tx=svgEl('text',{x:X(b.from)+4,y:m.t+11});tx.textContent=b.label;svg.appendChild(tx);});
  niceTicks(0,hi,5).forEach(t=>{if(t>hi)return;svg.appendChild(svgEl('line',{x1:X(t),x2:X(t),y1:m.t,y2:H-m.b,class:t===0?'axis':'grid'}));const tx=svgEl('text',{x:X(t),y:H-m.b+16,'text-anchor':'middle'});tx.textContent=t.toFixed(2);svg.appendChild(tx);});
  const hits=[];o.rows.forEach((r,k)=>{const y=m.t+rowH*k+rowH/2+2;const lab=svgEl('text',{x:m.l-10,y:y+4,'text-anchor':'end',class:'lab'});lab.textContent=r.name;svg.appendChild(lab);
    svg.appendChild(svgEl('line',{x1:X(r.a),x2:X(r.b),y1:y,y2:y,style:'stroke:var(--axis);stroke-width:2'}));
    svg.appendChild(svgEl('circle',{cx:X(r.a),cy:y,r:5,style:'fill:var(--surface);stroke:var(--ink2);stroke-width:2'}));
    svg.appendChild(svgEl('circle',{cx:X(r.b),cy:y,r:5.5,style:`fill:${r.color};stroke:var(--surface);stroke-width:2`}));
    const v=svgEl('text',{x:X(Math.max(r.a,r.b))+10,y:y+4,class:'val'});v.textContent=r.b.toFixed(3);svg.appendChild(v);hits.push({y,r});});
  wrap.appendChild(svg);
  attachTip(wrap,svg,W,(px,py)=>{if(px==null)return null;const h=hits.find(h=>Math.abs(py-h.y)<rowH/2);if(!h)return null;return {x:px,html:tipRows(h.r.name,h.r.tip)};});
  root.appendChild(wrap);
  dataTable(root,o.cols,o.rows.map(r=>[r.name,...r.cells]),'Step');
}
