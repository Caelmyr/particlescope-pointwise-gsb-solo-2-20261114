const $ = id => document.getElementById(id);
const state = {sample:null,result:null,samples:[],catalog:null,recipes:[],selected:new Set(),view:'working',busy:false,points:[],zoom:1,dx:0,dy:0,loadToken:0};
const segmentationFields = ['method','polarity','channel','threshold','sigma','min_area','max_area','adaptive_window','adaptive_offset','percentile','opening','closing','connectivity','fill_holes','exclude_edge','split_touching','seed_distance'];
const stringFields = new Set(['method','polarity','channel']);
let loadedImage = null;
let objectURL = null;
let jobTimer = null;
const labels = {count:'颗粒数',coverage:'覆盖率 %',mean_diameter:'平均直径',median_diameter:'中位直径',unit:'单位',scale:'每像素长度',area:'面积',diameter:'等效直径',circularity:'圆度',aspect_ratio:'长宽比',solidity:'实心度',mean_intensity:'平均灰度',std_intensity:'灰度标准差',perimeter:'周长',major_axis:'长轴',minor_axis:'短轴',feret_max:'最大Feret径',feret_min:'最小Feret径',orientation:'方向角',eccentricity:'偏心率',extent:'矩形填充率',roughness:'边界粗糙度',n:'样本数',mean:'均值',std:'标准差',median:'中位数',minimum:'最小值',maximum:'最大值',q1:'第一四分位',q3:'第三四分位',iqr:'四分位距',ci95_low:'95%区间下限',ci95_high:'95%区间上限',skewness:'偏度',kurtosis:'峰度',sem:'标准误',cv_percent:'变异系数 %',number_distribution:'数量加权分布',area_distribution:'面积加权分布',volume_distribution:'体积加权分布',descriptive:'描述统计',nearest_neighbors:'最近邻距离',quadrat:'密度格网',edge_distance:'ROI边缘距离',summary:'汇总',warning:'说明',assumption:'假设',method:'方法',count_change:'数量变化',coverage_change:'覆盖率变化',pixel_agreement:'像素一致性',matching:'对象匹配',first:'样本A',second:'样本B',clusters:'聚类',noise_ids:'噪声颗粒编号',mean_difference:'均值差 A−B',mean_ratio:'均值比 A/B',wasserstein_distance:'Wasserstein距离',ks_statistic:'KS统计量',ks_p_value:'KS P值',welch_p_value:'Welch P值',cohen_d:'Cohen d',d10:'D10',d50:'D50',d90:'D90',span:'分布Span',entropy:'熵',contrast:'对比度',homogeneity:'同质性',energy:'能量',correlation:'相关性',id:'编号',included:'纳入统计',centroid_x:'质心X',centroid_y:'质心Y',true_positive:'前景正确像素',false_positive:'误检像素',false_negative:'漏检像素',iou:'IoU',dice:'Dice',precision:'精确率',recall:'召回率',pixel_accuracy:'像素准确率'};
function label(key){return labels[key] || key.replaceAll('_',' ')}
function el(tag, text, cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e}
function status(message,error=false){$('status').textContent=message;$('status').classList.toggle('error',error)}
function fmt(value){if(value===null||value===undefined)return '—';if(typeof value==='number')return Number.isInteger(value)?String(value):Number(value.toPrecision(6)).toString();if(typeof value==='boolean')return value?'是':'否';return String(value)}
async function request(url,options={}){
  const response=await fetch(url,options);
  const type=response.headers.get('content-type')||'';
  const data=type.includes('json')?await response.json():await response.blob();
  if(!response.ok)throw Error(data.error||`请求失败 (${response.status})`);
  return data;
}
function post(url,data,method='POST'){return request(url,{method,headers:{'Content-Type':'application/json'},body:JSON.stringify(data)})}
function url(path){if(!state.sample)throw Error('请先选择样本');return `/api/samples/${state.sample.id}/${path}`}
function requireSample(){if(!state.sample)throw Error('请先导入或选择样本');return state.sample}
function requireResult(){requireSample();if(!state.result)throw Error('请先运行图像分析');return state.result}
function guard(fn){return async(...args)=>{try{await fn(...args)}catch(error){status(error.message,true);const e=$('dialogBody').querySelector('.dialog-error');if(e)e.textContent=error.message}}}
async function task(message,fn){
  if(state.busy)throw Error('当前操作进行中，请稍候');
  state.busy=true;status(message);document.body.classList.add('busy');$('run').disabled=true;
  try{return await fn()}finally{state.busy=false;$('run').disabled=false;document.body.classList.remove('busy')}
}
function openDialog(title){
  if(jobTimer){clearInterval(jobTimer);jobTimer=null}
  if(objectURL){URL.revokeObjectURL(objectURL);objectURL=null}
  $('dialogTitle').textContent=title;$('dialogBody').replaceChildren();
  if(!$('dialog').open)$('dialog').showModal();
  return $('dialogBody');
}
$('closeDialog').onclick=()=>{$('dialog').close();if(jobTimer)clearInterval(jobTimer);jobTimer=null};
$('dialog').addEventListener('close',()=>{if(jobTimer)clearInterval(jobTimer);jobTimer=null});
function formField(def,value){
  const wrapper=el('label',def.label);
  let input;
  if(def.type==='select'){
    input=el('select');
    for(const option of def.options){const op=el('option',option.label);op.value=option.value;input.append(op)}
  }else if(def.type==='textarea'){
    input=el('textarea');
  }else{
    input=el('input');input.type=def.type==='array'?'text':def.type||'text';
  }
  input.name=def.name;input.dataset.fieldType=def.type;
  const initial=value===undefined?def.default:value;
  if(def.type==='checkbox'){
    input.checked=Boolean(initial);wrapper.className='check';wrapper.prepend(input);
  }else{
    input.value=def.type==='array'?JSON.stringify(initial):initial??'';
    if(def.min!==undefined&&def.min!==null)input.min=def.min;
    if(def.max!==undefined&&def.max!==null)input.max=def.max;
    if(def.type==='number')input.step=def.step||'any';
    wrapper.append(input);
  }
  return wrapper;
}
function makeForm(fields,submitLabel,onSubmit){
  const form=el('form',undefined,'form');
  for(const field of fields)form.append(formField(field));
  const submit=el('button',submitLabel,'primary');submit.type='submit';form.append(submit);
  const error=el('p','','dialog-error');form.append(error);
  form.onsubmit=guard(async e=>{
    e.preventDefault();error.textContent='';const data={};
    for(const input of form.querySelectorAll('[name]')){
      const type=input.dataset.fieldType;
      if(type==='checkbox')data[input.name]=input.checked;
      else if(type==='number'){if(input.value.trim()==='')throw Error('请填写 '+input.parentElement.textContent);data[input.name]=Number(input.value)}
      else if(type==='array'){try{data[input.name]=JSON.parse(input.value)}catch{throw Error('坐标或列表格式错误，请使用 [1,2] 格式')}}
      else data[input.name]=input.value;
    }
    submit.disabled=true;
    try{await onSubmit(data,form)}catch(e){error.textContent=e.message;throw e}finally{submit.disabled=false}
  });
  return form;
}
function num(name,label,def,min,max){return {name,label,type:'number',default:def,min,max}}
function sel(name,label,def,options){return {name,label,type:'select',default:def,options:options.map(o=>({value:o[0],label:o[1]}))}}
function arr(name,label,def){return {name,label,type:'array',default:def}}
function txt(name,label,def='',type='text'){return {name,label,type,default:def}}
function sampleOptions(){return state.samples.filter(s=>!s.archived).map(s=>[s.id,s.name])}
function parameters(){
  const data={};
  for(const name of segmentationFields){const input=$(name);data[name]=input.type==='checkbox'?input.checked:stringFields.has(name)?input.value:Number(input.value)}
  return data;
}
function setParameters(data){for(const name of segmentationFields){if(data[name]===undefined)continue;const input=$(name);if(input.type==='checkbox')input.checked=data[name];else input.value=data[name]}for(const name of ['threshold','sigma'])$(name+'Value').textContent=$(name).value}
for(const name of ['threshold','sigma'])$(name).oninput=()=>$(name+'Value').textContent=$(name).value;
async function loadLibrary(selectId=null){
  const query=new URLSearchParams({query:$('search').value,archived:$('showArchived').checked});
  if($('projectFilter').value)query.set('project_id',$('projectFilter').value);
  const samples=await request('/api/samples?'+query);
  state.samples=samples;$('total').textContent=samples.length;$('library').replaceChildren();
  for(const sample of samples){
    const card=el('div',undefined,'sample');card.classList.toggle('selected',state.sample?.id===sample.id);card.dataset.id=sample.id;
    const check=el('input');check.type='checkbox';check.checked=state.selected.has(sample.id);check.setAttribute('aria-label','批量选择 '+sample.name);
    check.onchange=()=>{check.checked?state.selected.add(sample.id):state.selected.delete(sample.id)};
    const button=el('button');button.type='button';const image=el('img');image.src=`/api/samples/${sample.id}/image/thumb?v=${sample.revision}`;image.alt=sample.name;
    button.append(image,el('span',sample.name+(sample.archived?' · 已归档':'')),el('span',`${sample.width} × ${sample.height} · ${sample.analysis_id?'已分析':'待分析'}`));
    button.onclick=guard(()=>selectSample(sample.id));card.append(button,check);$('library').append(card);
  }
  if(selectId)await selectSample(selectId);
  else if(!state.sample&&samples.length)await selectSample(samples[0].id);
}
async function loadProjects(){
  const projects=await request('/api/projects');const current=$('projectFilter').value;
  $('projectFilter').replaceChildren(new Option('全部项目',''));
  for(const project of projects)$('projectFilter').append(new Option(`${project.name} (${project.sample_count})`,project.id));
  $('projectFilter').value=current;
}
async function selectSample(key){
  if(state.busy)return status('当前操作进行中，请稍候');
  const token=++state.loadToken;const data=await request(`/api/samples/${key}`);
  if(token!==state.loadToken)return;
  state.sample=data.meta;state.view='working';state.points=[];loadedImage=null;resetView();
  $('name').textContent=data.meta.name;$('size').textContent=`${data.meta.width} × ${data.meta.height} px`;
  $('archive').textContent=data.meta.archived?'恢复':'归档';
  $('sampleInfo').textContent=`分析区域：${data.meta.roi.type}；标定：1 px = ${fmt(data.meta.calibration.scale)} ${data.meta.calibration.unit}。图像版本 ${data.meta.image_revision}，修订 ${data.meta.revision}。`;
  state.result=await request(url('result'));
  if(token!==state.loadToken)return;
  if(state.result?.parameters)setParameters(state.result.parameters);
  renderResult();showImage();
  document.querySelectorAll('.sample').forEach(c=>c.classList.toggle('selected',c.dataset.id===key));
  status(state.result?'样本与保存的分析已载入':'样本已载入，调整参数后运行分析');
}
async function refreshSample(){
  const key=requireSample().id;const data=await request(`/api/samples/${key}`);state.sample=data.meta;
  $('name').textContent=data.meta.name;$('size').textContent=`${data.meta.width} × ${data.meta.height} px`;
  $('sampleInfo').textContent=`分析区域：${data.meta.roi.type}；1 px = ${fmt(data.meta.calibration.scale)} ${data.meta.calibration.unit}；图像版本 ${data.meta.image_revision}，修订 ${data.meta.revision}。`;
  state.result=await request(url('result'));renderResult();await loadLibrary();
}
function showImage(){
  if(!state.sample)return;
  $('canvas').hidden=false;$('empty').hidden=true;
  document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===state.view));
  const image=new Image();const token=state.sample.id+state.view+state.sample.revision;
  $('canvas').dataset.token=token;
  image.onload=()=>{if($('canvas').dataset.token!==token)return;loadedImage=image;$('canvas').width=image.naturalWidth;$('canvas').height=image.naturalHeight;drawCanvas()};
  image.onerror=()=>status(state.view==='reference'?'当前样本没有参考掩膜':'图像尚未生成，请先运行分析',true);
  image.src=url(`image/${state.view}`)+`?v=${state.sample.revision}`;
}
function drawCanvas(){
  if(!loadedImage)return;const canvas=$('canvas');const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(loadedImage,0,0);
  const roi=state.sample?.roi;
  if(roi&&roi.type!=='full'){
    ctx.strokeStyle='#ffcd62';ctx.lineWidth=Math.max(2,canvas.width/350);ctx.setLineDash([8,5]);ctx.beginPath();
    if(roi.type==='rectangle')ctx.rect(roi.x,roi.y,roi.width,roi.height);
    else if(roi.type==='ellipse')ctx.ellipse(roi.x+roi.width/2,roi.y+roi.height/2,roi.width/2,roi.height/2,0,0,Math.PI*2);
    else{roi.points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath()}
    ctx.stroke();ctx.setLineDash([]);
  }
  if(state.points.length){ctx.beginPath();ctx.strokeStyle='#ffca64';ctx.fillStyle='#ffca64';ctx.lineWidth=Math.max(2,canvas.width/400);state.points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.stroke();for(const p of state.points){ctx.beginPath();ctx.arc(...p,Math.max(3,canvas.width/200),0,Math.PI*2);ctx.fill()}}
  $('pointCount').textContent=state.points.length?`已选 ${state.points.length} 个点`:'';
}
function resetView(){state.zoom=1;state.dx=state.dy=0;transform()}
function transform(){$('canvas').style.transform=`translate(${state.dx}px,${state.dy}px) scale(${state.zoom})`}
$('fit').onclick=resetView;
$('viewport').addEventListener('wheel',e=>{if(!state.sample)return;e.preventDefault();state.zoom=Math.max(.3,Math.min(10,state.zoom*(e.deltaY<0?1.12:.89)));transform()},{passive:false});
let drag=null;
$('viewport').onpointerdown=e=>{if(!state.sample||state.busy)return;if($('canvasMode').value==='pan'){drag={x:e.clientX,y:e.clientY,dx:state.dx,dy:state.dy};$('viewport').setPointerCapture(e.pointerId)}};
$('viewport').onpointermove=e=>{if(!drag)return;state.dx=drag.dx+e.clientX-drag.x;state.dy=drag.dy+e.clientY-drag.y;transform()};
$('viewport').onpointerup=$('viewport').onpointercancel=()=>drag=null;
$('canvas').onclick=guard(async e=>{
  if(state.busy||$('canvasMode').value==='pan')return;requireSample();
  if(state.view==='original'&&(loadedImage?.naturalWidth!==state.sample.width||loadedImage?.naturalHeight!==state.sample.height))throw Error('原图与当前图像尺寸不同，请切换到当前图像再测量');
  const rect=$('canvas').getBoundingClientRect();const point=[Math.max(0,Math.min(state.sample.width-1,(e.clientX-rect.left)/rect.width*$('canvas').width)),Math.max(0,Math.min(state.sample.height-1,(e.clientY-rect.top)/rect.height*$('canvas').height))];
  const mode=$('canvasMode').value;
  if(mode==='pick'){
    const particle=await post(url('pick'),{point});if(!particle)return status('该位置没有颗粒');showParticle(particle);return;
  }
  state.points.push(point);drawCanvas();
  const required={rectangle:2,line:2,angle:3,point:1,profile:2,remove_object:1};
  if(required[mode]&&state.points.length===required[mode])await finishDrawing();
});
$('canvasMode').onchange=()=>{state.points=[];drawCanvas();status('在图像上点击选择坐标；多边形完成后点击“完成绘制”')};
$('cancelDrawing').onclick=()=>{state.points=[];drawCanvas();status('绘制已取消')};
$('finishDrawing').onclick=guard(finishDrawing);
async function finishDrawing(){
  requireSample();const mode=$('canvasMode').value;const points=state.points.slice();
  if(!points.length)throw Error('请先在图像上选点');
  if(mode==='rectangle'){
    if(points.length!==2)throw Error('矩形需要两个对角点');
    const x=Math.floor(Math.min(points[0][0],points[1][0])),y=Math.floor(Math.min(points[0][1],points[1][1]));
    const width=Math.max(1,Math.ceil(Math.abs(points[0][0]-points[1][0]))),height=Math.max(1,Math.ceil(Math.abs(points[0][1]-points[1][1])));
    await mutate('保存分析区域…',()=>post(url('roi'),{revision:state.sample.revision,roi:{type:'rectangle',x,y,width,height}},'PUT'));
  }else if(mode==='polygon'){
    await mutate('保存多边形区域…',()=>post(url('roi'),{revision:state.sample.revision,roi:{type:'polygon',points}},'PUT'));
  }else if(['line','angle','measure_polygon','point'].includes(mode)){
    const type=mode==='measure_polygon'?'polygon':mode;
    await mutate('保存人工测量…',()=>post(url('annotations'),{revision:state.sample.revision,type,points,label:''}));
    state.view='working';status('人工测量已保存，可在“人工测量”查看数值和叠加');
  }else if(mode==='profile'){
    if(points.length!==2)throw Error('剖面需要两个端点');
    const output=await post(url('profile'),{start:points[0],end:points[1],samples:300,line_width:1,channel:'gray'});
    const body=openDialog('强度线剖面');renderOutput(output,body);
  }else if(['add','erase','remove_object','merge'].includes(mode)){
    const data={revision:state.sample.revision,operation:mode,points};
    if(mode==='remove_object')data.point=points[0];
    await mutate('更新掩膜并重新测量…',()=>post(url('mask'),data));state.view='overlay';
  }else throw Error('请选择绘制工具');
  state.points=[];drawCanvas();showImage();
}
function renderResult(){
  const r=state.result;$('count').textContent=r?`${r.count} / ${r.total_detected}`:'—';$('diameter').textContent=r?fmt(r.mean_diameter):'—';$('coverage').textContent=r?fmt(r.coverage)+'%':'—';$('unit').textContent=r?r.unit:state.sample?.calibration.unit||'px';$('csv').classList.toggle('disabled',!r);$('csv').href=state.sample?url('export/csv'):'#';renderRows();
}
function renderRows(){
  $('rows').replaceChildren();if(!state.result){const tr=$('rows').insertRow();const td=tr.insertCell();td.colSpan=8;td.textContent='分析后显示形态、粒径及灰度测量结果';return}
  let rows=state.result.particles.filter(r=>$('rowFilter').value==='all'||($('rowFilter').value==='included'?r.included:!r.included));
  const sort=$('sortField').value;rows=rows.slice().sort((a,b)=>a[sort]-b[sort]);
  for(const row of rows.slice(0,1000)){
    const tr=$('rows').insertRow();if(!row.included)tr.className='excluded';
    for(const value of [row.id,row.area,row.diameter,row.circularity,row.aspect_ratio,`${fmt(row.major_axis)} / ${fmt(row.minor_axis)}`,`${fmt(row.centroid_x)} / ${fmt(row.centroid_y)}`,row.included?'纳入':'排除'])tr.insertCell().textContent=fmt(value);
    tr.onclick=()=>showParticle(row);
  }
  if(rows.length>1000){const tr=$('rows').insertRow();const cell=tr.insertCell();cell.colSpan=8;cell.textContent=`页面显示前1000条，共${rows.length}条；CSV包含全部结果`}
}
$('rowFilter').onchange=$('sortField').onchange=renderRows;
async function mutate(message,fn){
  return task(message,async()=>{const result=await fn();await refreshSample();state.points=[];if(!state.result&&['overlay','mask','labels'].includes(state.view))state.view='working';showImage();status('操作完成，结果已保存');return result});
}
$('run').onclick=guard(async()=>{requireSample();await mutate('正在分割、测量并保存结果…',()=>post(url('analyze'),{...parameters(),revision:state.sample.revision}));state.view='overlay';showImage();status(`分析完成：${state.result.count} 个颗粒，覆盖率 ${fmt(state.result.coverage)}%`)});
document.querySelectorAll('[data-view]').forEach(button=>button.onclick=guard(()=>{requireSample();if(['overlay','mask','labels'].includes(button.dataset.view))requireResult();state.view=button.dataset.view;showImage()}));
$('demo').onclick=guard(async()=>{const sample=await task('生成演示图像…',()=>post('/api/demo',{kind:$('demoKind').value}));await loadLibrary(sample.id);if($('demoKind').value==='fluorescence')setParameters({polarity:'light',channel:'green',method:'otsu'});else if($('demoKind').value==='porous')setParameters({polarity:'light',method:'otsu',fill_holes:false});status('演示已载入，自带参考掩膜，可分析后评估IoU')});
$('upload').onchange=guard(async e=>{
  if(!e.target.files.length)return;const files=[...e.target.files];e.target.value='';
  const form=new FormData();for(const file of files)form.append('files',file);if($('projectFilter').value)form.append('project_id',$('projectFilter').value);
  const result=await task('正在导入图像…',()=>request('/api/samples',{method:'POST',body:form}));await loadLibrary(result.imported[0]?.id);status(`导入成功 ${result.imported.length} 张，失败 ${result.failed.length} 张`);if(result.failed.length)renderOutput(result.failed,openDialog('导入失败详情'));
});
let searchTimer;
$('search').oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(guard(()=>loadLibrary()),250)};
$('projectFilter').onchange=$('showArchived').onchange=guard(()=>loadLibrary());
$('resetROI').onclick=guard(()=>mutate('恢复全图分析区域…',()=>post(url('roi'),{revision:requireSample().revision,roi:{type:'full'}},'PUT')));
$('duplicate').onclick=guard(async()=>{const sample=await task('复制样本…',()=>post(url('duplicate'),{}));await loadLibrary(sample.id)});
$('archive').onclick=guard(async()=>{requireSample();await mutate('更新归档状态…',()=>post(url('archive'),{revision:state.sample.revision,archived:!state.sample.archived}));$('archive').textContent=state.sample.archived?'恢复':'归档';status(state.sample.archived?'样本已归档，可勾选显示已归档样本恢复':'样本已恢复')});
$('metadata').onclick=guard(()=>{
  const sample=requireSample();const body=openDialog('样本信息');body.append(makeForm([txt('name','名称',sample.name),txt('tags','标签（逗号分隔）',sample.tags.join(', ')),txt('notes','样本说明',sample.notes,'textarea'),sel('project_id','所属项目',sample.project_id||'',[['','不分组'],...Array.from($('projectFilter').options).filter(o=>o.value).map(o=>[o.value,o.textContent])])],'保存信息',async data=>{data.tags=data.tags.split(/[,，]/).map(t=>t.trim()).filter(Boolean);data.project_id=data.project_id||null;await mutate('保存样本信息…',()=>post(`/api/samples/${state.sample.id}`,{...data,revision:state.sample.revision},'PATCH'));$('dialog').close()}));
});
$('newProject').onclick=()=>{const body=openDialog('新建图像项目');body.append(makeForm([txt('name','项目名称'),txt('description','项目说明','','textarea')],'创建项目',async data=>{const project=await post('/api/projects',data);await loadProjects();$('projectFilter').value=project.id;await loadLibrary();$('dialog').close();status('项目已创建')}))};
$('projectReport').onclick=guard(()=>{const key=$('projectFilter').value;if(!key)throw Error('请先选择一个项目');window.open(`/api/projects/${key}/report`,'_blank','noopener')});
$('calibrate').onclick=guard(()=>{
  const sample=requireSample(),cal=sample.calibration;const body=openDialog('尺寸标定');body.append(el('p','输入标尺的实际长度和图中像素长度。几何操作后会重置标定、区域和人工标注。','hint'));
  const fields=[num('pixel_length','标尺像素长度',cal.pixel_length,0.000001),num('physical_length','标尺实际长度',cal.physical_length,0.000001),sel('unit','单位',cal.unit,[['px','像素'],['μm','微米 μm'],['mm','毫米 mm'],['nm','纳米 nm']])];
  body.append(makeForm(fields,'保存标定',async data=>{await mutate('保存尺寸标定…',()=>post(url('calibration'),{...data,revision:state.sample.revision},'PUT'));$('dialog').close()}));
});
$('preprocess').onclick=guard(()=>{
  requireSample();const body=openDialog('图像预处理');body.append(el('p','每次处理都保留之前的图像，可在历史中恢复。旋转、翻转、裁切和缩放会清空当前标定、ROI和人工标注。','hint'));
  const fields=[sel('operation','图像操作','gaussian',state.catalog.preprocessing.map(p=>[p.key,p.label])),num('amount','强度 / Gamma / 高斯σ',1,.01,100),num('radius','滤波 / 背景半径',3,1,100),sel('channel','通道','gray',[['gray','灰度'],['red','红'],['green','绿'],['blue','蓝'],['hue','色相'],['saturation','饱和度'],['value','明度']]),num('angle','旋转角度',90,-360,360),sel('direction','翻转方向','horizontal',[['horizontal','水平'],['vertical','垂直']]),num('width','缩放宽度',state.sample.width,16,8000),num('height','缩放高度',state.sample.height,16,8000),arr('crop_values','裁切 [x,y,width,height]',[0,0,state.sample.width,state.sample.height])];
  body.append(makeForm(fields,'应用到当前图像',async data=>{const [x,y,width,height]=data.crop_values;data.crop={x,y,width,height};await mutate('正在处理图像…',()=>post(url('enhance'),{...data,revision:state.sample.revision}));$('dialog').close()}));
});
$('filterMeasurements').onclick=guard(()=>{
  requireResult();const body=openDialog('筛选测量结果');body.append(el('p','筛选不会更改原始检测掩膜，会生成新的分析记录；留空即不限制。','hint'));
  const form=el('form',undefined,'form');const grid=el('div',undefined,'two');
  for(const [key,title] of [['area','面积'],['diameter','等效直径'],['circularity','圆度'],['aspect_ratio','长宽比'],['solidity','实心度'],['mean_intensity','平均灰度']]){
    for(const suffix of ['min','max']){const def=num(key+'_'+suffix,title+(suffix==='min'?'下限':'上限'),'');grid.append(formField(def,state.result.measurement_filters?.[def.name]??''))}
  }
  form.append(grid,formField(arr('exclude_ids','人工排除编号',state.result.measurement_filters?.exclude_ids||[])));const submit=el('button','保存筛选','primary');form.append(submit);form.onsubmit=guard(async e=>{e.preventDefault();const data={revision:state.sample.revision};for(const input of form.querySelectorAll('[name]'))data[input.name]=input.name==='exclude_ids'?JSON.parse(input.value):input.value===''?null:Number(input.value);await mutate('筛选并更新测量叠加…',()=>post(url('filter'),data));state.view='overlay';showImage();$('dialog').close()});body.append(form);
});
$('saveRecipe').onclick=guard(()=>{requireResult();const body=openDialog('保存分析配方');body.append(makeForm([txt('name','配方名称','我的颗粒分析配方'),txt('description','适用样本说明','','textarea')],'保存配方',async data=>{await post(`/api/recipes/from-sample/${state.sample.id}`,data);state.recipes=await request('/api/recipes');$('dialog').close();status('当前分割与测量筛选已保存为配方')}))});
$('referenceFile').onchange=guard(async e=>{requireSample();const file=e.target.files[0];e.target.value='';if(!file)return;const form=new FormData();form.append('file',file);await task('导入参考掩膜…',()=>request(url('reference'),{method:'POST',body:form}));status('参考掩膜已保存，可以运行参考评估')});
$('export').onclick=guard(()=>{
  requireResult();const body=openDialog('导出与分析报告');const grid=el('div',undefined,'link-grid');
  for(const [kind,title] of [['contours-svg','颗粒轮廓 SVG'],['csv','颗粒测量 CSV'],['json','完整分析 JSON'],['geojson','质心 GeoJSON'],['mask-tiff','二值掩膜 TIFF'],['labels-tiff','实例标签 TIFF'],['bundle','完整分析 ZIP'],['crops','透明颗粒裁片 ZIP'],['annotations','人工测量 CSV'],['chart','粒径分布图 PNG']]){const a=el('a','↓ '+title);a.href=url('export/'+kind);grid.append(a)}
  const report=el('a','打开可打印报告 / 另存PDF');report.href=url('report');report.target='_blank';report.rel='noopener';grid.append(report);body.append(grid,el('p','报告内含标定、分割参数、叠加图、粒径分布和测量说明。','hint'));
});
function showParticle(row){
  const body=openDialog('颗粒 #'+row.id);const image=el('img',undefined,'preview');image.src=url(`particle/${row.id}`);body.append(image);renderOutput(row,body);const button=el('button',row.included?'从统计中排除此颗粒':'重新纳入此颗粒');button.onclick=guard(async()=>{const excluded=new Set(state.result.measurement_filters?.exclude_ids||[]);row.included?excluded.add(row.id):excluded.delete(row.id);await mutate('更新颗粒筛选…',()=>post(url('filter'),{...state.result.measurement_filters,exclude_ids:[...excluded],revision:state.sample.revision}));$('dialog').close()});body.append(button);
}
$('annotations').onclick=guard(async()=>{
  requireSample();const data=await request(url('annotations'));const body=openDialog('人工测量与标注');body.append(el('p','使用画布工具选择线段、角度或多边形，点击对应端点即可保存。标注随尺寸标定更新。','hint'));
  const image=el('img',undefined,'preview');image.src=url('annotated')+'?v='+state.sample.revision;body.append(image);renderOutput(data.summary,body);
  for(const annotation of data.annotations){const card=el('div',undefined,'history-card');card.append(el('h3',`${annotation.type} · ${annotation.label||annotation.id.slice(0,8)}`));renderOutput(annotation.measurement,card);const rename=el('button','修改标签');rename.onclick=()=>{const target=openDialog('修改标注');target.append(makeForm([txt('label','标注标签',annotation.label)],'保存',async data=>{await mutate('更新标注…',()=>post(url('annotations/'+annotation.id),{...data,revision:state.sample.revision},'PATCH'));$('dialog').close()}))};const del=el('button','移除标注');del.onclick=guard(async()=>{await mutate('移除标注…',()=>post(url('annotations/'+annotation.id),{revision:state.sample.revision},'DELETE'));$('annotations').click()});card.append(rename,del);body.append(card)}
});
$('history').onclick=guard(async()=>{
  requireSample();const [history,analyses]=await Promise.all([request(url('history')),request(url('analyses'))]);const body=openDialog('图像与分析历史');
  body.append(el('h3','图像预处理历史'));const reset=el('button','恢复原始图像（当前图像另存历史）');reset.onclick=guard(async()=>{await mutate('恢复原始图像…',()=>post(url('restore'),{revision:state.sample.revision,version_id:'original'}));$('dialog').close()});body.append(reset);
  for(const entry of history.slice().reverse()){const card=el('div',undefined,'history-card');card.append(el('h3',`${entry.operation} · ${entry.time}`));const button=el('button','恢复到此操作之前');button.onclick=guard(async()=>{await mutate('恢复图像版本…',()=>post(url('restore'),{revision:state.sample.revision,version_id:entry.id}));$('dialog').close()});card.append(button);body.append(card)}
  body.append(el('h3','分析历史（保留参数与结果）'));const selects=[];
  for(const result of analyses){const card=el('div',undefined,'history-card');card.append(el('h3',`${result.created_at} · ${result.count} 个颗粒 · 图像v${result.image_revision}`));const check=el('input');check.type='checkbox';check.onchange=()=>{check.checked?selects.push(result.id):selects.splice(selects.indexOf(result.id),1)};const l=el('label',' 选择对比');l.prepend(check);card.append(l);const button=el('button','设为当前分析');button.onclick=guard(async()=>{await mutate('切换分析版本…',()=>post(url('analyses/select'),{analysis_id:result.id,revision:state.sample.revision}));state.view='overlay';showImage();$('dialog').close()});card.append(button);body.append(card)}
  const compare=el('button','比较勾选的两个分析版本');compare.onclick=guard(async()=>{if(selects.length!==2)throw Error('请勾选两个分析版本');const result=await post('/api/compare-versions',{sample_id:state.sample.id,first:selects[0],second:selects[1]});renderOutput(result,openDialog('分析版本差异'))});body.append(compare);
});
function chart(series,title,container,options={}){
  if(!series.length||!series.some(s=>s.values.length))return;
  const canvas=el('canvas');canvas.width=900;canvas.height=300;container.append(canvas);const c=canvas.getContext('2d');c.fillStyle='#10202b';c.fillRect(0,0,900,300);c.fillStyle='#cde3ea';c.font='14px sans-serif';c.fillText(title,18,24);
  const left=55,right=870,top=50,bottom=260;const values=series.flatMap(s=>s.values.map(v=>Number(v)||0));const low=options.zero===false?Math.min(...values):Math.min(0,...values);const high=Math.max(...values,low+1);const colors=['#64e4ba','#efa95c','#87b7fa'];c.strokeStyle='#3d5567';c.beginPath();c.moveTo(left,top);c.lineTo(left,bottom);c.lineTo(right,bottom);c.stroke();
  for(let i=0;i<=4;i++){const y=bottom-i*(bottom-top)/4;c.fillStyle='#9bb5c6';c.font='11px sans-serif';c.fillText(fmt(low+(high-low)*i/4).slice(0,8),3,y);c.strokeStyle='#263e4d';c.beginPath();c.moveTo(left,y);c.lineTo(right,y);c.stroke()}
  series.forEach((s,idx)=>{c.strokeStyle=colors[idx%colors.length];c.fillStyle=colors[idx%colors.length];c.lineWidth=2;c.beginPath();s.values.forEach((value,i)=>{const x=left+(right-left)*i/Math.max(s.values.length-1,1),y=bottom-(Number(value)-low)/(high-low)*(bottom-top);if(options.scatter){c.moveTo(x+2,y);c.arc(x,y,2,0,Math.PI*2)}else{i?c.lineTo(x,y):c.moveTo(x,y)}});options.scatter?c.fill():c.stroke();c.fillText(s.name,180+idx*210,25)});
}
function renderOutput(data,target,depth=0){
  if(data===null||data===undefined){target.append(el('p','暂无结果'));return}
  if(depth>5){target.append(el('p','较深层明细请下载结果 JSON'));return}
  if(Array.isArray(data)){
    if(!data.length){target.append(el('p','无条目','muted'));return}
    if(data.every(r=>r&&typeof r==='object'&&!Array.isArray(r))){
      const keys=[...new Set(data.flatMap(Object.keys))].filter(k=>data.some(r=>typeof r[k]!=='object'||r[k]===null)).slice(0,16);const wrap=el('div',undefined,'table-wrap');const table=el('table'),head=el('thead'),tr=el('tr');keys.forEach(k=>tr.append(el('th',label(k))));head.append(tr);table.append(head);const tbody=el('tbody');for(const row of data.slice(0,300)){const r=el('tr');keys.forEach(k=>r.append(el('td',fmt(row[k]))));tbody.append(r)}table.append(tbody);wrap.append(table);target.append(wrap);if(data.length>300)target.append(el('p',`显示前300条，共${data.length}条；下载JSON可查看完整结果`,'muted'));return;
    }
    if(data.every(r=>typeof r==='number'||r===null)){target.append(el('p',data.length<=60?data.map(fmt).join(', '):`数值序列 ${data.length} 项；请查看图表或下载JSON`,'muted'));return}
    for(const row of data.slice(0,30)){const section=el('div',undefined,'output-section');renderOutput(row,section,depth+1);target.append(section)}return;
  }
  if(typeof data!=='object'){target.append(el('p',fmt(data)));return}
  const plotted=new Set();
  if(data.intensity&&data.distance_px){chart([{name:'强度',values:data.intensity}],'强度沿线变化 · 横轴为等距采样',target);plotted.add('intensity');plotted.add('distance_px');plotted.add('distance')}
  if(data.percent&&data.edges){chart([{name:'比例 %',values:data.percent}],'粒径分布（等序分箱）',target);plotted.add('percent');plotted.add('edges');plotted.add('counts');if(data.cumulative_percent){chart([{name:'累计 %',values:data.cumulative_percent}],'累计粒径分布',target);plotted.add('cumulative_percent')}}
  if(data.bins&&Array.isArray(data.bins)){chart([{name:'像素数',values:data.bins}],'灰度直方图 · 0–255',target);plotted.add('bins')}
  if(data.first_percent){chart([{name:'样本A %',values:data.first_percent},{name:'样本B %',values:data.second_percent}],'双样本分布对照（相同分箱）',target);plotted.add('first_percent');plotted.add('second_percent');plotted.add('edges')}
  if(data.mean_power){chart([{name:'平均功率',values:data.mean_power}],'径向频率功率谱',target);plotted.add('mean_power');plotted.add('frequency_cycles_per_pixel')}
  if(data.g&&data.radii){chart([{name:'g(r)',values:data.g}],'径向对分布',target);plotted.add('g');plotted.add('radii')}
  if(data.l_minus_r){chart([{name:'L(r) − r',values:data.l_minus_r}],'Ripley函数',target);plotted.add('l_minus_r');plotted.add('k');plotted.add('radii')}
  if(data.normalized_radius){chart([{name:'归一化半径',values:data.normalized_radius}],'径向形状 · 0°–360°',target);plotted.add('normalized_radius');plotted.add('radius_px');plotted.add('angles_degrees')}
  if(data.colors){const swatches=el('div',undefined,'swatches');for(const color of data.colors){const swatch=el('div',`${color.hex} · ${fmt(color.percent)}%`,'swatch');swatch.style.background=color.hex;swatches.append(swatch)}target.append(swatches)}
  const metrics=el('div',undefined,'metric-grid');const nested=[];
  for(const [key,value]of Object.entries(data)){
    if(plotted.has(key))continue;
    if(value!==null&&typeof value==='object'){nested.push([key,value]);continue}
    if(['warning','assumption','interpretation','note','reason','convention','method'].includes(key)&&typeof value==='string'){target.append(el('p',label(key)+'：'+value,'hint'));continue}
    const card=el('div',undefined,'metric-card');card.append(el('small',label(key)),el('strong',fmt(value)));metrics.append(card);
  }
  if(metrics.childNodes.length)target.append(metrics);
  for(const [key,value]of nested){const section=el('details',undefined,'output-section');if(depth===0&&['descriptive','number_distribution','summary','nearest_neighbors','glcm','average','porosity','transform','after'].includes(key))section.open=true;section.append(el('summary',label(key)));renderOutput(value,section,depth+1);target.append(section)}
}
function downloadJSON(data,name='analysis-result.json'){const a=el('a');const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});const u=URL.createObjectURL(blob);a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}
$('tools').onclick=()=>openToolbox();
function openToolbox(group='全部'){
  const body=openDialog('图像分析工具箱');const groups=['全部',...new Set(state.catalog.tools.map(t=>t.group))];const nav=el('div',undefined,'group-tabs');
  for(const name of groups){const button=el('button',name);if(name===group)button.className='primary';button.onclick=()=>openToolbox(name);nav.append(button)}
  body.append(nav);const grid=el('div',undefined,'tool-list');for(const tool of state.catalog.tools.filter(t=>group==='全部'||t.group===group)){const button=el('button',undefined,'tool-card');button.append(el('small',tool.group),el('b',tool.label),el('span',tool.note));button.onclick=guard(()=>openTool(tool.key));grid.append(button)}body.append(grid);
}
function openTool(key){
  requireSample();const tool=state.catalog.tools.find(t=>t.key===key);const body=openDialog(tool.label);const layout=el('div',undefined,'dialog-grid');const controls=el('div'),output=el('div',undefined,'tool-output');controls.append(el('p',tool.note,'hint'));
  controls.append(makeForm(tool.fields,'运行 '+tool.label,async data=>{
    output.replaceChildren(el('p','正在计算…','muted'));
    if(key==='sweep')data.parameters=parameters();
    const result=await task('正在计算 '+tool.label+'…',()=>tool.method==='GET'?request(url(tool.path)):post(url(tool.path),data));
    output.replaceChildren();
    if(tool.output==='image'){
      if(objectURL)URL.revokeObjectURL(objectURL);objectURL=URL.createObjectURL(result);const img=el('img');img.src=objectURL;output.append(img);const a=el('a','↓ 下载图像');a.href=objectURL;a.download=tool.key+'.png';output.append(a);
    }else{
      if(Array.isArray(result)&&result.length&&result[0].threshold!==undefined)chart([{name:'颗粒数',values:result.map(r=>r.count)}],'阈值扫描 · 从低到高',output);
      renderOutput(result,output);const save=el('button','下载完整结果 JSON');save.onclick=()=>downloadJSON(result,tool.key+'.json');output.append(save);
    }
    status(tool.label+'完成');
  }));layout.append(controls,output);body.append(layout);
  if(tool.method==='GET')controls.querySelector('form').requestSubmit();
}
document.querySelectorAll('[data-tool]').forEach(button=>button.onclick=guard(()=>openTool(button.dataset.tool)));
$('compare').onclick=guard(()=>{
  requireSample();const body=openDialog('双样本比较 · 配准 · 拼接');if(state.samples.length<2){body.append(el('p','请先导入至少两个图像样本。','hint'));return}
  const layout=el('div',undefined,'dialog-grid'),controls=el('div'),output=el('div',undefined,'tool-output');const ids=[...state.selected];
  const fields=[sel('reference','参考样本 A',ids[0]||state.sample.id,sampleOptions()),sel('moving','移动 / 比较样本 B',ids[1]||state.samples.find(s=>s.id!==state.sample.id).id,sampleOptions()),sel('operation','操作','distribution',[['distribution','粒径 / 形态分布对照'],['registration','图像平移配准'],['mosaic','羽化拼接'],['affine','控制点仿射配准']]),sel('field','比较指标','diameter',state.catalog.measurements.map(m=>[m.key,m.label])),sel('mode','平移模式','auto',[['auto','自动估计平移'],['manual','使用手动偏移']]),num('dx','X偏移 px',0,-8000,8000),num('dy','Y偏移 px',0,-8000,8000),{name:'save',label:'配准后另存新样本',type:'checkbox',default:false},arr('source_points','移动图控制点 [[x,y],…]',[[20,20],[200,20],[20,200]]),arr('target_points','参考图控制点 [[x,y],…]',[[20,20],[200,20],[20,200]])];
  controls.append(makeForm(fields,'运行对照操作',async data=>{
    if(data.reference===data.moving)throw Error('请选择两个不同的样本');
    output.replaceChildren(el('p','正在计算…'));
    let result;
    await task('正在处理双样本图像…',async()=>{result=data.operation==='distribution'?await post('/api/compare',{first:data.reference,second:data.moving,field:data.field}):await post('/api/'+data.operation,data)});
    output.replaceChildren();renderOutput(result.report||result.transform||result.details||result,output);if(result.sample){await loadLibrary();const button=el('button','打开生成的新样本');button.onclick=guard(async()=>{await selectSample(result.sample.id);$('dialog').close()});output.append(button)}
    status('双样本操作完成');
  }));layout.append(controls,output);body.append(layout,el('p','自动配准仅估计平移；拼接采用当前偏移且要求标定完全一致。新配准样本需要重新标定。','hint'));
});
$('batch').onclick=guard(openBatch);
async function openBatch(){
  state.recipes=await request('/api/recipes');const body=openDialog('批量分析与配方');const layout=el('div',undefined,'dialog-grid'),controls=el('div'),output=el('div',undefined,'tool-output');
  const selector=el('div',undefined,'sample-select');for(const sample of state.samples.filter(s=>!s.archived)){const label=el('label',sample.name),check=el('input');check.type='checkbox';check.value=sample.id;check.checked=state.selected.has(sample.id);label.prepend(check);selector.append(label)}
  controls.append(el('h3','选择待分析样本'),selector);
  controls.append(makeForm([sel('recipe_id','分析配方',state.recipes[0]?.id,state.recipes.map(r=>[r.id,r.name]))],'开始批量分析',async data=>{const ids=[...selector.querySelectorAll('input:checked')].map(i=>i.value);await post('/api/jobs',{sample_ids:ids,recipe_id:data.recipe_id});await renderJobs(output);status('批量任务已排队，当前样本完成后再响应取消')}));
  const group=el('button','勾选样本的粒径汇总');group.onclick=guard(async()=>{const ids=[...selector.querySelectorAll('input:checked')].map(i=>i.value);const result=await post('/api/group-summary',{sample_ids:ids,field:'diameter'});renderOutput(result,openDialog('多样本粒径汇总'))});controls.append(group);
  const exportRecipe=el('a','下载当前选中配方');exportRecipe.href='/api/recipes/'+(state.recipes[0]?.id||'')+'/export';controls.querySelector('select').onchange=e=>exportRecipe.href='/api/recipes/'+e.target.value+'/export';controls.append(el('p'),exportRecipe);
  const importLabel=el('label','导入配方 JSON','file-label');const input=el('input');input.type='file';input.accept='.json';input.hidden=true;input.onchange=guard(async e=>{if(!e.target.files[0])return;const data=JSON.parse(await e.target.files[0].text());await post('/api/recipes/import',data);await openBatch();status('配方已导入')});importLabel.append(input);controls.append(el('p'),importLabel);
  layout.append(controls,output);body.append(layout);await renderJobs(output);jobTimer=setInterval(()=>renderJobs(output).catch(e=>status(e.message,true)),2500);
}
async function renderJobs(output){
  if(!$('dialog').open)return;const jobs=await request('/api/jobs');output.replaceChildren(el('h3','任务记录'));
  for(const job of jobs.slice(0,20)){const card=el('div',undefined,'job-card');card.append(el('h3',job.recipe.name+' · '+job.status),el('p',`${job.completed}/${job.items.length} 完成，${job.failed} 失败`));const bar=el('div',undefined,'bar'),span=el('span');span.style.width=job.progress_percent+'%';bar.append(span);card.append(bar);renderOutput(job.items,card);
    if(['running','queued','cancelling'].includes(job.status)){const cancel=el('button','取消后续样本');cancel.onclick=guard(async()=>{await post(`/api/jobs/${job.id}/cancel`,{});await renderJobs(output)});card.append(cancel)}else if(job.items.some(i=>['failed','interrupted','cancelled'].includes(i.status))){const retry=el('button','重试未完成样本');retry.onclick=guard(async()=>{await post(`/api/jobs/${job.id}/retry`,{});await renderJobs(output)});card.append(retry)}
    output.append(card);
  }
  if(!jobs.length)output.append(el('p','暂无任务，选择样本和配方后开始。','muted'));
}
/* 时序颗粒跟踪 */
const seq={id:null,data:null,frame:0,mode:'view',pickFrom:null,trackFilter:null,selectedTrack:null};
const seqKindLabels={ambiguous:'交叉/靠近',gap:'疑似漏检',border:'进出画面'};
const seqEventLabels={started:'新出现',ended:'消失',entered_frame:'进入画面',left_frame:'离开画面'};
Object.assign(labels,{from_frame:'起始帧',to_frame:'目标帧',dt:'时间间隔',distance:'位移',vx:'速度X',vy:'速度Y',speed:'速度',frame:'帧',time:'时间',particle_id:'颗粒编号',track_id:'轨迹'});
$('tracking').onclick=guard(openSequences);
async function openSequences(){
  const sequences=await request('/api/sequences');
  const body=openDialog('时序颗粒跟踪');
  body.append(el('p','把同一视野连续拍摄的样本按时间排列，关联各帧已分割的颗粒，查看轨迹与速度。交叉靠近、暂时漏检和进出画面不会强行连线，会列为待定对应供人工确认；所有修改只影响时序，各帧原有分析结果不变。','hint'));
  const create=el('button','＋ 新建时序','primary');create.onclick=()=>renderSequenceCreate(body);body.append(create);
  for(const s of sequences){
    const card=el('div',undefined,'history-card');
    card.append(el('h3',s.name),el('p',`${s.frame_count} 帧 · ${s.track_count} 条轨迹 · ${s.link_count} 个关联 · 间隔 ${fmt(s.interval)} ${s.interval_unit} · 更新于 ${s.updated_at}`,'muted'));
    const open=el('button','打开');open.onclick=guard(async()=>{seq.data=await request('/api/sequences/'+s.id);seq.id=s.id;seq.frame=0;seq.trackFilter=null;seq.selectedTrack=null;renderSequence(openDialog(s.name))});
    const del=el('button','删除');del.onclick=guard(async()=>{const full=await request('/api/sequences/'+s.id);await request(`/api/sequences/${s.id}`,{method:'DELETE',headers:{'Content-Type':'application/json'},body:JSON.stringify({revision:full.revision})});await openSequences()});
    card.append(open,del);body.append(card);
  }
  if(!sequences.length)body.append(el('p','还没有时序。请先为每个时间点的样本运行图像分析，再新建时序。','muted'));
}
function renderSequenceCreate(body){
  body.replaceChildren();
  const analyzed=state.samples.filter(s=>!s.archived&&s.analysis_id);
  if(analyzed.length<2){body.append(el('p','至少需要两个已分析的样本。请先对各时间点的图像运行图像分析。','hint'));return}
  body.append(el('p','勾选样本并用 ↑↓ 调整时间顺序；每帧使用样本当前的分析结果，创建后钉住该版本，之后重新分析样本不会改变已建时序。','hint'));
  const order=analyzed.map(s=>s.id).sort((a,b)=>(state.selected.has(a)?0:1)-(state.selected.has(b)?0:1));
  const chosen=new Set([...state.selected].filter(id=>analyzed.some(s=>s.id===id)));
  const list=el('div');
  const renderList=()=>{
    list.replaceChildren();
    order.forEach((id,idx)=>{
      const sample=analyzed.find(s=>s.id===id);
      const row=el('div',undefined,'seq-row');
      const check=el('input');check.type='checkbox';check.checked=chosen.has(id);
      check.onchange=()=>{check.checked?chosen.add(id):chosen.delete(id)};
      const name=el('span',`${idx+1}. ${sample.name}`);
      const up=el('button','↑');up.type='button';up.onclick=()=>{if(idx>0){[order[idx-1],order[idx]]=[order[idx],order[idx-1]];renderList()}};
      const down=el('button','↓');down.type='button';down.onclick=()=>{if(idx<order.length-1){[order[idx+1],order[idx]]=[order[idx],order[idx+1]];renderList()}};
      row.append(check,name,up,down);list.append(row);
    });
  };
  renderList();body.append(list);
  body.append(makeForm([
    txt('name','时序名称','视野时序'),
    num('interval','拍摄间隔',1,0.000001),
    sel('interval_unit','时间单位','min',[['frame','帧'],['s','秒'],['min','分钟'],['h','小时'],['d','天']]),
    num('max_distance','帧间最大位移 px',40,0.1,5000),
  ],'创建并自动关联',async data=>{
    const ids=order.filter(id=>chosen.has(id));
    if(ids.length<2)throw Error('至少勾选两个样本');
    const created=await post('/api/sequences',{name:data.name,sample_ids:ids,interval:data.interval,interval_unit:data.interval_unit,settings:{max_distance:data.max_distance}});
    seq.id=created.id;seq.data=created;seq.frame=0;seq.trackFilter=null;seq.selectedTrack=null;
    renderSequence(openDialog(created.name));
    status(`时序已创建：${created.summary.track_count} 条轨迹，${created.summary.pending_count} 个待定对应`);
  }));
}
function renderSequence(body){
  const data=seq.data;
  body.replaceChildren();
  if(data.affected&&(data.affected.changed.length||data.affected.retired.length)){
    const parts=[];
    if(data.affected.changed.length)parts.push('已重算轨迹：'+data.affected.changed.join('、'));
    if(data.affected.retired.length)parts.push('不再存在：'+data.affected.retired.join('、'));
    body.append(el('p',parts.join('；'),'hint'));
  }
  const summary=data.summary;
  const metrics=el('div',undefined,'metric-grid');
  for(const [value,title] of [[summary.frame_count,'帧数'],[summary.track_count,'轨迹数'],[summary.pending_count,'待定对应'],[summary.complete_tracks,'贯穿全程'],[summary.mean_speed==null?null:`${fmt(summary.mean_speed)} ${data.velocity_unit}`,'平均速度']]){
    if(value===null||value===undefined)continue;
    const card=el('div',undefined,'metric-card');card.append(el('small',title),el('strong',fmt(value)));metrics.append(card);
  }
  body.append(metrics);
  if(summary.calibration_warning)body.append(el('p',summary.calibration_warning,'hint'));
  const frame=data.frames[seq.frame];
  const nav=el('div',undefined,'seq-nav');
  const prev=el('button','◀ 上一帧');prev.disabled=seq.frame===0;prev.onclick=()=>{seq.frame--;renderSequence(body)};
  const next=el('button','下一帧 ▶');next.disabled=seq.frame===data.frames.length-1;next.onclick=()=>{seq.frame++;renderSequence(body)};
  nav.append(prev,el('span',`帧 ${seq.frame+1}/${data.frames.length} · t=${fmt(frame.time)} ${data.time_unit} · ${frame.particle_count} 个颗粒 · ${frame.name}`),next);
  if(!frame.is_current_analysis){
    const rebind=el('button','该帧样本已有新分析，重新绑定');
    rebind.onclick=guard(async()=>{seq.data=await post(`/api/sequences/${seq.id}/frames/${seq.frame}/rebind`,{revision:seq.data.revision});renderSequence(body);status('已重新绑定到样本当前分析，该帧关联已重算')});
    nav.append(rebind);
  }
  if(frame.image_mismatch)nav.append(el('span','该帧图像在分析后有过修改，叠加可能不对齐','muted'));
  body.append(nav);
  const modes=el('div',undefined,'seq-modes');
  for(const [mode,labeltext] of [['view','查看'],['link','手动关联'],['unlink','删除关联']]){
    const b=el('button',labeltext);b.classList.toggle('active',seq.mode===mode);
    b.onclick=()=>{seq.mode=mode;seq.pickFrom=null;renderSequence(body)};
    modes.append(b);
  }
  if(seq.mode==='link')modes.append(el('span',seq.pickFrom?`已选 帧${seq.pickFrom.frame+1} 颗粒#${seq.pickFrom.id}，切换到后续帧点击目标`:'点击本帧一个颗粒作为起点','muted'));
  body.append(modes);
  const img=el('img',undefined,'seq-overlay');
  img.src=`/api/sequences/${seq.id}/overlay/${seq.frame}?v=${data.revision}${seq.trackFilter?'&track='+seq.trackFilter:''}`;
  img.onclick=guard(async e=>{
    if(seq.mode==='view'||!img.naturalWidth)return;
    const rect=img.getBoundingClientRect();
    const x=(e.clientX-rect.left)/rect.width*img.naturalWidth,y=(e.clientY-rect.top)/rect.height*img.naturalHeight;
    const particle=await request(`/api/sequences/${seq.id}/pick?frame=${seq.frame}&x=${x}&y=${y}`);
    if(!particle)return status('该位置没有颗粒');
    if(!particle.included)return status('该颗粒在该帧分析中已被排除，不能关联');
    if(seq.mode==='link'){
      if(!seq.pickFrom){seq.pickFrom={frame:seq.frame,id:particle.id};renderSequence(body);return}
      const from=seq.pickFrom;seq.pickFrom=null;
      if(seq.frame<=from.frame)throw Error('请先切换到起始帧之后的帧再选目标');
      seq.data=await post(`/api/sequences/${seq.id}/links`,{revision:seq.data.revision,from_frame:from.frame,from_id:from.id,to_frame:seq.frame,to_id:particle.id});
      renderSequence(body);status('手动关联已保存');
    }else if(seq.mode==='unlink'){
      seq.data=await request(`/api/sequences/${seq.id}/links`,{method:'DELETE',headers:{'Content-Type':'application/json'},body:JSON.stringify({revision:seq.data.revision,from_frame:seq.frame,from_id:particle.id})});
      renderSequence(body);status('关联已删除');
    }
  });
  body.append(img);
  const pendingBox=el('div');
  pendingBox.append(el('h3',`待定对应（${data.pending.length}）`));
  if(!data.pending.length)pendingBox.append(el('p','没有待确认的对应；所有明确关联已自动完成。','muted'));
  for(const item of data.pending){
    const card=el('div',undefined,'pending-card');
    card.append(el('h3',`帧${item.frame+1} 颗粒#${item.from_id} → 帧${item.to_frame+1} · ${seqKindLabels[item.kind]}`),el('p',item.reason,'muted'));
    for(const cand of item.candidates){
      const b=el('button',`关联 → #${cand.to_id}（${fmt(cand.distance)} px）`);
      b.onclick=guard(async()=>{seq.data=await post(`/api/sequences/${seq.id}/pending/${item.key}/resolve`,{revision:seq.data.revision,action:'link',to_id:cand.to_id});renderSequence(body)});
      card.append(b);
    }
    const reject=el('button','无关联（结束/离开）');
    reject.onclick=guard(async()=>{seq.data=await post(`/api/sequences/${seq.id}/pending/${item.key}/resolve`,{revision:seq.data.revision,action:'reject'});renderSequence(body)});
    const locate=el('button','查看');
    locate.onclick=()=>{seq.frame=item.frame;renderSequence(body)};
    card.append(reject,locate);
    pendingBox.append(card);
  }
  body.append(pendingBox);
  if(data.events.length){
    const details=el('details');details.append(el('summary',`进出与起止事件（${data.events.length}）`));
    details.append(el('p',data.events.map(e=>`帧${e.frame+1} #${e.particle_id} ${seqEventLabels[e.kind]}`).join('；'),'muted'));
    body.append(details);
  }
  body.append(el('h3',`轨迹（${data.tracks.length}）`));
  const wrap=el('div',undefined,'table-wrap');
  const table=el('table'),head=el('thead'),hrow=el('tr');
  for(const title of ['轨迹','帧范围','点数','跨帧','平均速度','最大速度','净位移','路径长度'])hrow.append(el('th',title));
  head.append(hrow);table.append(head);
  const tbody=el('tbody');
  for(const track of data.tracks.slice(0,300)){
    const row=el('tr');
    for(const value of [track.id,`${track.start_frame+1}–${track.end_frame+1}`,track.point_count,track.gap_count,track.mean_speed,track.max_speed,track.net_displacement,track.path_length])row.append(el('td',fmt(value)));
    row.onclick=()=>{seq.selectedTrack=track.id;renderSequence(body)};
    tbody.append(row);
  }
  table.append(tbody);wrap.append(table);body.append(wrap);
  if(data.tracks.length>300)body.append(el('p',`显示前300条，共${data.tracks.length}条；CSV包含全部轨迹`,'muted'));
  const selected=data.tracks.find(t=>t.id===seq.selectedTrack);
  if(selected){
    const detail=el('div',undefined,'output-section');
    detail.append(el('h3',`轨迹 ${selected.id} · 速度单位 ${data.velocity_unit}`));
    if(selected.steps.length)chart([{name:'速度',values:selected.steps.map(s=>s.speed)}],`逐步速度 · ${data.velocity_unit}`,detail);
    renderOutput(selected.steps,detail);
    const focus=el('button',seq.trackFilter===selected.id?'取消突出显示':'在图上突出显示');
    focus.onclick=()=>{seq.trackFilter=seq.trackFilter===selected.id?null:selected.id;renderSequence(body)};
    const close=el('button','收起');close.onclick=()=>{seq.selectedTrack=null;renderSequence(body)};
    detail.append(focus,close);body.append(detail);
  }
  const settingsBox=el('details');settingsBox.append(el('summary','时序设置与操作'));
  settingsBox.append(makeForm([
    txt('name','时序名称',data.name),
    num('interval','拍摄间隔',data.interval,0.000001),
    sel('interval_unit','时间单位',data.interval_unit,[['frame','帧'],['s','秒'],['min','分钟'],['h','小时'],['d','天']]),
    num('max_distance','帧间最大位移 px',data.settings.max_distance,0.1,5000),
    num('margin_ratio','歧义判定比例',data.settings.margin_ratio,1,3),
    num('margin_abs','歧义判定余量 px',data.settings.margin_abs,0,100),
    {name:'gap_enabled',label:'允许跨帧（漏检）候选',type:'checkbox',default:data.settings.gap_enabled},
    num('gap_factor','跨帧距离倍数',data.settings.gap_factor,1,4),
    num('border_margin','边缘判定宽度 px',data.settings.border_margin,0,500),
  ],'保存设置并重新关联',async form=>{
    seq.data=await post(`/api/sequences/${seq.id}`,{revision:seq.data.revision,name:form.name,interval:form.interval,interval_unit:form.interval_unit,settings:{max_distance:form.max_distance,margin_ratio:form.margin_ratio,margin_abs:form.margin_abs,gap_enabled:form.gap_enabled,gap_factor:form.gap_factor,border_margin:form.border_margin}},'PATCH');
    renderSequence(body);
  }));
  const relink=el('button','按当前设置重新自动关联');
  relink.onclick=guard(async()=>{seq.data=await post(`/api/sequences/${seq.id}/relink`,{revision:seq.data.revision});renderSequence(body);status('已重新自动关联；人工关联与否决记录保留')});
  const clear=el('button','清空否决记录');
  clear.onclick=guard(async()=>{seq.data=await post(`/api/sequences/${seq.id}/rejections/clear`,{revision:seq.data.revision});renderSequence(body)});
  const csv=el('a','↓ 轨迹 CSV');csv.href=`/api/sequences/${seq.id}/export/tracks.csv`;
  const del=el('button','删除此时序');
  del.onclick=guard(async()=>{await request(`/api/sequences/${seq.id}`,{method:'DELETE',headers:{'Content-Type':'application/json'},body:JSON.stringify({revision:seq.data.revision})});await openSequences()});
  settingsBox.append(relink,clear,csv,del);
  body.append(settingsBox);
}
async function init(){
  const [catalog,recipes]=await Promise.all([request('/api/catalog'),request('/api/recipes')]);state.catalog=catalog;state.recipes=recipes;await loadProjects();await loadLibrary();
}
init().catch(e=>status(e.message,true));