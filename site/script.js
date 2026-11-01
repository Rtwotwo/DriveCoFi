const $ = (id) => document.getElementById(id);
const dialog = $('figure-dialog');
document.addEventListener('click', (event) => {
  const button = event.target.closest('.figure-open');
  if (!button) return;
  $('dialog-image').src = button.dataset.full;
  $('dialog-image').alt = button.dataset.alt || button.querySelector('img')?.alt || '';
  $('dialog-caption').textContent = $('dialog-image').alt;
  dialog.showModal();
});
document.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', (event) => { if (event.target === dialog) dialog.close(); });
dialog.addEventListener('close', () => { $('dialog-image').removeAttribute('src'); });

function setPanel(prefix, model, src, alt) {
  const image = $(prefix + '-' + model);
  image.src = src;
  image.alt = alt;
  const button = $(prefix + '-' + model + '-open');
  button.dataset.full = src;
  button.dataset.alt = alt;
}
function initGallery(data) {
  let group = 'Straight';
  function showScene(scene) {
    $('scene-title').textContent = scene.name;
    $('scene-source').textContent = scene.source + ' · NAVSIM-v1 navtest';
    ['drivor','cofidrive'].forEach(model => setPanel('scene', model, scene[model], (model === 'drivor' ? 'DrivoR' : 'CoFiDrive') + ' · ' + scene.name));
    document.querySelectorAll('.scene-thumb').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.scene === scene.id)));
  }
  function showGroup() {
    const scenes = data.scenes.filter(x => x.group === group);
    const strip = $('scene-thumbnails');
    strip.replaceChildren();
    scenes.forEach((scene, i) => {
      const button = document.createElement('button');
      button.className = 'scene-thumb';
      button.dataset.scene = scene.id;
      button.setAttribute('aria-label', scene.name);
      const image = document.createElement('img');
      image.src = scene.cofidrive; image.alt = ''; image.loading = 'lazy';
      const text = document.createElement('span'); text.textContent = 'Scene ' + (i + 1);
      button.append(image, text);
      button.addEventListener('click', () => showScene(scene));
      strip.append(button);
    });
    showScene(scenes[0]);
  }
  $('scene-groups').addEventListener('click', event => {
    const button = event.target.closest('[data-group]');
    if (!button) return;
    group = button.dataset.group;
    $('scene-groups').querySelectorAll('button').forEach(b => {
      b.classList.toggle('active', b === button);
      b.setAttribute('aria-pressed', String(b === button));
    });
    showGroup();
  });
  function showAttention() {
    const camera = $('attention-camera').value;
    const index = Number($('attention-register').value);
    const label = $('attention-register').selectedOptions[0].textContent;
    ['drivor','cofidrive'].forEach(model => setPanel('attention', model, data.attention[camera][model][index], (model === 'drivor' ? 'DrivoR' : 'CoFiDrive') + ' · ' + camera + ' · ' + label));
  }
  $('attention-camera').addEventListener('change', showAttention);
  $('attention-register').addEventListener('change', showAttention);
  showGroup(); showAttention();
}

const NS = 'http://www.w3.org/2000/svg';
function svgNode(tag, attrs, text) {
  const node = document.createElementNS(NS, tag);
  Object.entries(attrs || {}).forEach(([key,value]) => node.setAttribute(key, String(value)));
  if (text != null) node.textContent = text;
  return node;
}
function initCSA(cases) {
  let timer = null;
  const plot = $('csa-svg');
  function stop() {
    if (timer) clearInterval(timer);
    timer = null; $('csa-play').textContent = 'Play waypoints';
  }
  function draw() {
    const scene = cases[Number($('csa-case').value)];
    const count = Number($('csa-time').value);
    $('csa-time-label').textContent = count + ' / 8 · ' + (count * .5).toFixed(1) + ' s';
    $('csa-speed').textContent = 'Initial longitudinal speed: ' + scene.initial_speed.toFixed(2) + ' m/s.';
    const all = scene.original.flat().concat(scene.augmented.flat(),scene.selected,scene.reference,[[0,0]]);
    const lat = all.map(p => -p[1]), fwd = all.map(p => p[0]);
    const xmin = Math.min(...lat)-3, xmax = Math.max(...lat)+3;
    const ymin = Math.min(...fwd)-3, ymax = Math.max(...fwd)+3;
    const scale = Math.min(580/(xmax-xmin),440/(ymax-ymin));
    const cx = (xmin+xmax)/2, cy = (ymin+ymax)/2;
    const xy = p => [350+(-p[1]-cx)*scale,260-(p[0]-cy)*scale];
    plot.replaceChildren(svgNode('rect',{x:0,y:0,width:700,height:540,fill:'#f6fafc',rx:10}));
    const gridStep = Math.max(xmax-xmin,ymax-ymin)>65 ? 10 : 5;
    for(let x=Math.ceil(xmin/gridStep)*gridStep;x<=xmax;x+=gridStep){
      const px=350+(x-cx)*scale;
      plot.append(svgNode('line',{x1:px,y1:35,x2:px,y2:485,stroke:'#dfe8ed'}));
      plot.append(svgNode('text',{x:px,y:507,'text-anchor':'middle',fill:'#6a7f8d','font-size':12},String(-x)));
    }
    for(let y=Math.ceil(ymin/gridStep)*gridStep;y<=ymax;y+=gridStep){
      const py=260-(y-cy)*scale;
      plot.append(svgNode('line',{x1:60,y1:py,x2:640,y2:py,stroke:'#dfe8ed'}));
      plot.append(svgNode('text',{x:49,y:py+4,'text-anchor':'end',fill:'#6a7f8d','font-size':12},String(y)));
    }
    plot.append(svgNode('text',{x:350,y:533,'text-anchor':'middle',fill:'#536e80','font-size':13},'Lateral position (m; left positive)'));
    plot.append(svgNode('text',{x:20,y:260,transform:'rotate(-90 20 260)','text-anchor':'middle',fill:'#536e80','font-size':13},'Forward position (m)'));
    function path(points, color, width, opacity, dots, dash) {
      const pts = [[0,0],...points.slice(0,count)].map(xy);
      const attrs={points:pts.map(p=>p.join(',')).join(' '),fill:'none',stroke:color,'stroke-width':width,'stroke-opacity':opacity,'stroke-linejoin':'round','stroke-linecap':'round'};
      if(dash) attrs['stroke-dasharray']=dash;
      plot.append(svgNode('polyline',attrs));
      if(dots) pts.slice(1).forEach((p,i)=>{
        plot.append(svgNode('circle',{cx:p[0],cy:p[1],r:4,fill:color,stroke:'#fff','stroke-width':1}));
        if(color==='#bd7e00') plot.append(svgNode('text',{x:p[0]+7,y:p[1]-5,fill:'#7e5300','font-size':11},String(i+1)));
      });
    }
    if($('show-augmented').checked) scene.augmented.forEach(p=>path(p,'#9273b5',1,.25,false));
    if($('show-original').checked) scene.original.forEach(p=>path(p,'#197aaf',1.2,.38,false));
    if($('show-reference').checked) path(scene.reference,'#268577',2.4,1,false,'6 4');
    if($('show-selected').checked) path(scene.selected,'#bd7e00',3,1,true);
    const origin=xy([0,0]);
    plot.append(svgNode('rect',{x:origin[0]-6,y:origin[1]-10,width:12,height:20,rx:3,fill:'#17364b',stroke:'#fff','stroke-width':1.5}));
    plot.append(svgNode('text',{x:origin[0]+12,y:origin[1]+15,fill:'#17364b','font-size':12},'ego'));
    plot.setAttribute('aria-label',scene.name + ', ' + count + ' of 8 future poses, original and augmented candidates in meters');
  }
  ['csa-case','show-original','show-augmented','show-selected','show-reference'].forEach(id=>$(id).addEventListener('change',()=>{stop();draw();}));
  $('csa-time').addEventListener('input',()=>{stop();draw();});
  $('csa-play').addEventListener('click',()=>{
    if(timer){stop();return;}
    $('csa-time').value=1;draw();$('csa-play').textContent='Pause';
    timer=setInterval(()=>{
      const value=Number($('csa-time').value);
      if(value>=8){stop();return;}
      $('csa-time').value=value+1;draw();
    },450);
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  draw();
}
function initResults(data) {
  data.ablations.forEach(item=>{
    const option=document.createElement('option');
    option.value=item.id;option.textContent=item.title;$('ablation-choice').append(option);
  });
  function showAblation() {
    const item=data.ablations.find(x=>x.id===$('ablation-choice').value);
    $('ablation-context').textContent=item.context;
    $('ablation-note').textContent=item.note;
    $('ablation-source').textContent=item.source+' · Reported point estimates; comparisons are within this experiment group.';
    const chart=$('ablation-chart'),table=$('ablation-table');
    chart.replaceChildren();table.replaceChildren();
    item.rows.forEach(([label,value])=>{
      const row=document.createElement('div');row.className='ablation-point-row';
      const name=document.createElement('span');name.textContent=label;
      const track=document.createElement('div');track.className='ablation-track';
      const point=document.createElement('span');point.className='ablation-dot';point.style.left=((value-91)/2*100)+'%';
      const score=document.createElement('strong');score.textContent=value.toFixed(2);
      track.append(point);row.append(name,track,score);chart.append(row);
      const tr=document.createElement('tr');
      [label,value.toFixed(2)].forEach(text=>{const td=document.createElement('td');td.textContent=text;tr.append(td);});table.append(tr);
    });
    chart.setAttribute('aria-label',item.title+': '+item.rows.map(r=>r[0]+' '+r[1]).join('; '));
  }
  function showNavhard() {
    const stage=data.navhard[$('navhard-stage').value];
    const table=$('navhard-table');table.replaceChildren();
    data.navhard.metrics.forEach((metric,i)=>{
      const tr=document.createElement('tr');
      [metric,stage.drivor[i].toFixed(1),stage.cofidrive[i].toFixed(1)].forEach(text=>{const td=document.createElement('td');td.textContent=text;tr.append(td);});
      const delta=Number((stage.cofidrive[i]-stage.drivor[i]).toFixed(1));
      const td=document.createElement('td');td.className='delta-cell';
      const track=document.createElement('div');track.className='delta-track';
      const bar=document.createElement('span');bar.className=delta<0?'negative':'positive';
      bar.style.width=Math.abs(delta)/6*50+'%';bar.style.left=delta<0?(50-Math.abs(delta)/6*50)+'%':'50%';
      const number=document.createElement('b');number.textContent=(delta>0?'+':'')+delta.toFixed(1);
      track.append(bar);td.append(track,number);tr.append(td);table.append(tr);
    });
  }
  $('ablation-choice').addEventListener('change',showAblation);
  $('navhard-stage').addEventListener('change',showNavhard);
  showAblation();showNavhard();
}
async function loadData(name) {
  const response=await fetch('./data/'+name+'.json');
  if(!response.ok)throw new Error(name+' data: '+response.status);
  return response.json();
}
Promise.all([loadData('gallery'),loadData('csa'),loadData('results')]).then(([gallery,csa,results])=>{
  initGallery(gallery);initCSA(csa);initResults(results);
  document.documentElement.dataset.explorers='ready';
}).catch(error=>{
  $('data-error').hidden=false;
  $('data-error').textContent='Interactive data could not be loaded. Reload the page to try again.';
  console.error(error);
});
