let currentUnits=[], parcelGeometry=[];

const $=id=>document.getElementById(id);

$("buildingSelect").addEventListener("change",()=>{updateFloorOptions();render3D();renderTable();});
$("floorSelect").addEventListener("change",()=>{render3D();renderTable();});
$("search").addEventListener("input",()=>renderTable());

function handlePayload(res,data){
  if(!res.ok||data.error){setMessage(data.error||"Unable to process dataset.");return;}
  currentUnits=data.units; parcelGeometry=data.parcels_geometry||[];
  $("records").textContent=data.summary.records; $("parcels").textContent=data.summary.parcels;
  $("buildings").textContent=data.summary.buildings; $("floors").textContent=data.summary.max_floors;
  populateBuildings(); updateFloorOptions(); render3D(); renderTable();
  setMessage(`Generated ${data.summary.records} prototype 3D ULPIN records successfully.`);
}
function setMessage(t){$("message").textContent=t;}

function populateBuildings(){
  const select=$("buildingSelect");
  const previous=select.value||"ALL";
  const ids=[...new Set(currentUnits.map(u=>u.building_id))];
  select.innerHTML='<option value="ALL">All Buildings</option>'+ids.map(id=>`<option value="${esc(id)}">${esc(id)}</option>`).join("");
  select.value=ids.includes(previous)?previous:"ALL";
}
function updateFloorOptions(){
  const b=$("buildingSelect").value, select=$("floorSelect");
  const previous=select.value||"ALL";
  const rows=b==="ALL"?currentUnits:currentUnits.filter(u=>u.building_id===b);
  const floors=[...new Set(rows.map(u=>Number(u.floor_no)))].sort((a,b)=>a-b);
  select.innerHTML='<option value="ALL">All Floors</option>'+floors.map(f=>`<option value="${f}">Floor ${f}</option>`).join("");
  select.value=floors.includes(Number(previous))?previous:"ALL";
}

function cubeTrace(u,index){
  const x0=u.x,x1=u.x+u.dx,y0=u.y,y1=u.y+u.dy,z0=u.z,z1=u.z+u.dz;
  return {
    type:"mesh3d",
    x:[x0,x1,x1,x0,x0,x1,x1,x0],y:[y0,y0,y1,y1,y0,y0,y1,y1],z:[z0,z0,z0,z0,z1,z1,z1,z1],
    i:[0,0,0,4,4,4,1,1,2,3,3,2],j:[1,2,3,5,6,7,5,2,6,7,4,3],k:[2,3,1,6,7,5,2,6,3,4,0,7],
    opacity:.92,flatshading:false,name:u.prototype_ulpin,customdata:Array(8).fill(index),
    hovertemplate:`<b>${u.building_id} / ${u.unit_id}</b><br>Floor: ${u.floor_no}<br>Area: ${u.area_sq_m} m²<br>Elevation: ${u.unit_elevation_m} m<br>${u.prototype_ulpin}<extra></extra>`,
    showscale:false
  };
}

function floorSlab(u){
  const x0=u.x,x1=u.x+u.dx,y0=u.y,y1=u.y+u.dy,z=u.z;
  return {type:"scatter3d",mode:"lines",x:[x0,x1,x1,x0,x0],y:[y0,y0,y1,y1,y0],z:[z,z,z,z,z],
    line:{width:2},hoverinfo:"skip",showlegend:false};
}

function buildingShell(bid, units){
  const rows=units.filter(u=>u.building_id===bid);
  if(!rows.length)return null;
  const minX=Math.min(...rows.map(u=>u.x)), maxX=Math.max(...rows.map(u=>u.x+u.dx));
  const minY=Math.min(...rows.map(u=>u.y)), maxY=Math.max(...rows.map(u=>u.y+u.dy));
  const minZ=0, maxZ=Math.max(...rows.map(u=>u.z+u.dz));
  const x=[minX,maxX,maxX,minX,minX,maxX,maxX,minX],y=[minY,minY,maxY,maxY,minY,minY,maxY,maxY],z=[minZ,minZ,minZ,minZ,maxZ,maxZ,maxZ,maxZ];
  return {type:"mesh3d",x,y,z,i:[0,0,0,4,4,4,1,1,2,3,3,2],j:[1,2,3,5,6,7,5,2,6,7,4,3],k:[2,3,1,6,7,5,2,6,3,4,0,7],
    opacity:.08,flatshading:false,hoverinfo:"skip",showlegend:false};
}

function labelTrace(units){
  const labels=units.filter(u=>u.floor_no===Math.min(...units.filter(x=>x.building_id===u.building_id).map(x=>x.floor_no)))
  const seen=new Set(), x=[],y=[],z=[],text=[];
  for(const u of units){
    const key=`${u.building_id}-${u.floor_no}`;
    if(seen.has(key))continue;
    seen.add(key);
    x.push(u.x+u.dx/2);y.push(u.y+u.dy/2);z.push(u.z+u.dz+0.25);
    text.push(`${u.building_id} • F${u.floor_no}`);
  }
  return {type:"scatter3d",mode:"text",x,y,z,text,textfont:{size:11},hoverinfo:"skip",showlegend:false};
}

function render3D(){
  if(!currentUnits.length)return;
  const b=$("buildingSelect").value, f=$("floorSelect").value;
  const units=currentUnits.filter(u=>(b==="ALL"||u.building_id===b)&&(f==="ALL"||Number(u.floor_no)===Number(f)));
  const buildingIds=[...new Set(units.map(u=>u.building_id))];
  const parcels=parcelGeometry.filter(p=>buildingIds.includes(p.building_id));

  let traces=[];
  // Parcel outlines
  for(const p of parcels){
    const x=p.x,x2=p.x+p.length,y=p.y,y2=p.y+p.width;
    traces.push({type:"scatter3d",mode:"lines",x:[x,x2,x2,x,x],y:[y,y,y2,y2,y],z:[0,0,0,0,0],
      line:{width:6},name:`Parcel ${p.parcel_id}`,hovertemplate:`<b>Parcel ${p.parcel_id}</b><br>Building ${p.building_id}<extra></extra>`});
  }
  // Light building shells and apartment units
  for(const bid of buildingIds){
    const shell=buildingShell(bid,units);
    if(shell)traces.push(shell);
  }
  units.forEach((u)=>traces.push(cubeTrace(u,currentUnits.indexOf(u))));
  if(f==="ALL" || units.length<=40){
    units.forEach(u=>traces.push(floorSlab(u)));
  }
  traces.push(labelTrace(units));

  Plotly.react("plot3d",traces,{
    paper_bgcolor:"#0e1a2c",plot_bgcolor:"#0e1a2c",margin:{l:0,r:0,t:10,b:0},showlegend:false,
    scene:{
      bgcolor:"#0e1a2c",
      xaxis:{title:"X / Property Position",gridcolor:"#263650",zerolinecolor:"#263650",color:"#8295ad"},
      yaxis:{title:"Y / Property Position",gridcolor:"#263650",zerolinecolor:"#263650",color:"#8295ad"},
      zaxis:{title:"Vertical Elevation (m)",gridcolor:"#263650",zerolinecolor:"#263650",color:"#8295ad"},
      aspectmode:"data",camera:{eye:{x:1.55,y:1.45,z:1.15}}
    },
    uirevision:"keep-camera"
  },{responsive:true,displaylogo:false});

  $("plot3d").removeAllListeners?.("plotly_click");
  $("plot3d").on("plotly_click",ev=>{
    if(!ev.points?.length)return;
    const pt=ev.points[0],idx=pt.customdata;
    if(Number.isInteger(idx)&&currentUnits[idx])showDetails(currentUnits[idx]);
  });
}

function showDetails(u){
  $("emptyDetails").classList.add("hidden");$("propertyDetails").classList.remove("hidden");
  $("dUlp").textContent=u.prototype_ulpin;$("dParcel").textContent=u.parcel_id;$("dBuilding").textContent=u.building_id;
  $("dFloor").textContent=u.floor_no;$("dUnit").textContent=u.unit_id;$("dArea").textContent=`${u.area_sq_m} m²`;
  $("dUse").textContent=u.use_type;$("dLat").textContent=u.latitude;$("dLon").textContent=u.longitude;
  $("dElev").textContent=`${u.unit_elevation_m} m`;$("dOwner").textContent=u.owner_id;
}

function renderTable(){
  const q=$("search").value.trim().toLowerCase(),b=$("buildingSelect").value,f=$("floorSelect").value;
  const filtered=currentUnits.filter(u=>(b==="ALL"||u.building_id===b)&&(f==="ALL"||Number(u.floor_no)===Number(f)))
    .filter(u=>`${u.prototype_ulpin} ${u.parcel_id} ${u.building_id} ${u.unit_id}`.toLowerCase().includes(q)).slice(0,100);
  $("propertyTable").innerHTML=filtered.map(u=>`<tr data-ulpin="${esc(u.prototype_ulpin)}"><td>${esc(u.prototype_ulpin)}</td><td>${esc(u.parcel_id)}</td><td>${esc(u.building_id)}</td><td>${u.floor_no}</td><td>${esc(u.unit_id)}</td><td>${u.area_sq_m}</td><td>${esc(u.use_type)}</td></tr>`).join("");
  $("propertyTable").querySelectorAll("tr").forEach(tr=>tr.addEventListener("click",()=>{
    const u=currentUnits.find(x=>x.prototype_ulpin===tr.dataset.ulpin);if(u)showDetails(u);
  }));
}
function esc(v){return String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]));}

(async()=>{
  try {
    const res = await fetch("/sample");
    handlePayload(res, await res.json());
  } catch (e) { setMessage("Could not load the demonstration property data."); }
})();
