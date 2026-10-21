const el=id=>document.getElementById(id), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let teams=[],selected=null,request=0,loadedTeam=null,loadedTraces=null;
async function api(path){const u=new URL(path,location.origin);const run=new URLSearchParams(location.search).get('run');if(run)u.searchParams.set('run',run);const r=await fetch(u,{cache:'no-store'}),v=await r.json();if(!r.ok||v.error)throw Error(v.error||r.statusText);return v;}
function list(){const query=el('search').value.toLowerCase();el('teams').innerHTML=teams.map((t,i)=>({t,i})).filter(({t})=>JSON.stringify(t).toLowerCase().includes(query)).map(({t,i})=>`<button data-team="${i}" class="${selected===t.id?'active':''}"><b>Attempt ${i+1}</b> · ${t.agents.length}/4 agents<br><small>${esc(t.id)}<br>${esc(t.agents.map(a=>a.name).join(', '))}</small></button>`).join('');el('teams').querySelectorAll('button').forEach(b=>b.onclick=()=>show(teams[+b.dataset.team]));}
function message(m){
  const body=RLTracerTeamMessages.render(m);
  if(m.role==='system'||m.role==='user')return '<details class="initial-input"><summary>'+ (m.role==='system'?'Instructions':'Task / input')+'</summary>'+body+'</details>';
  return body;
}
function drawTeam(){
  const t=loadedTeam,traces=loadedTraces;
  if(!t||!traces)return;
  const fixedLeader=RLTracerTeamMessages.fixedLeader(traces,t.agents.map(a=>a.name));
  if(el('view-mode').value==='causal'){
    el('agents').className='causal-view';
    el('agents').innerHTML='<div class="leader-overview">'+traces.map((trace,i)=>{const summary=RLTracerTeamMessages.leaderSummary(trace,t.agents.map(a=>a.name));return summary?'<div><b>'+esc(t.agents[i].name)+RLTracerTeamMessages.leaderBadge(t.agents[i].name,fixedLeader)+'</b>'+summary+'</div>':'';}).join('')+'</div>'+RLTracerCausal.render(t,traces);
    RLTracerCausal.mount(el('agents'));
    return;
  }
  RLTracerCausal.mount(document.createElement('div'));
  el('agents').className='';
  el('agents').innerHTML=traces.map((trace,i)=>{const a=t.agents[i];return '<section class="agent"><h2>'+esc(a.name)+RLTracerTeamMessages.leaderBadge(a.name,fixedLeader)+'</h2>'+RLTracerTeamMessages.leaderSummary(trace,t.agents.map(a=>a.name))+(a.answer?'<div class="answer"><b>Final answer</b><pre>'+esc(a.answer)+'</pre></div>':'')+'<div class="conversation">'+trace.messages.map(message).join('')+'</div></section>';}).join('');
}
async function show(t){
  const seq=++request;selected=t.id;loadedTeam=null;loadedTraces=null;list();
  el('status').textContent='Loading teammates…';
  el('board').innerHTML='<h1>Team attempt · '+esc(t.id)+'</h1><p class="muted">Prompt group: '+esc(t.group_id)+' · '+t.agents.length+'/4 saved agent traces.</p><div id="agents"></div>';
  try{
    const traces=await Promise.all(t.agents.map(a=>api('/api/trajectories/'+a.id+'?team=1')));
    if(seq!==request)return;
    loadedTeam=t;loadedTraces=traces;drawTeam();
    el('status').textContent=t.agents.length+' teammates loaded';
  }catch(e){if(seq===request)el('status').textContent=e.message;}
}
async function load(){++request;loadedTeam=null;loadedTraces=null;el('board').innerHTML='<h1>Select a team attempt</h1>';el('status').textContent='Loading team attempts…';try{teams=await api('/api/steps/'+el('steps').value+'/teams?split='+el('split').value);list();el('status').textContent=teams.length+' team attempts';const team=teams.find(t=>t.id===selected)||teams[0];if(team)await show(team);else if(!teams.length)el('board').innerHTML='<h1>No team attempts in this split</h1><p>Use Original Explorer for individual trajectories or runs without team metadata.</p>';}catch(e){el('status').textContent=e.message;}}
const params=new URLSearchParams(location.search);
if(['train','eval'].includes(params.get('split')))el('split').value=params.get('split');
let initialStep=params.get('step');
async function refresh(){try{const steps=await api('/api/steps?split='+el('split').value),old=el('steps').value||initialStep;initialStep=null;el('steps').innerHTML=steps.map(s=>`<option>${s.number}</option>`).join('');if(steps.some(s=>String(s.number)===old))el('steps').value=old;else if(steps.length)el('steps').value=steps[steps.length-1].number;if(steps.length)await load();else{teams=[];list();el('board').innerHTML='<h1>No saved rollouts in this split</h1>';el('status').textContent='Waiting for saved rollouts';}}catch(e){el('status').textContent=e.message;}}
el('explorer-link').onclick=e=>{e.preventDefault();const target=new URL('/explorer',location.origin);if(params.get('run'))target.searchParams.set('run',params.get('run'));if(el('steps').value)target.searchParams.set('step',el('steps').value);target.searchParams.set('split',el('split').value);location.assign(target.href);};
el('steps').onchange=load;el('split').onchange=()=>{selected=null;refresh();};el('refresh').onclick=refresh;el('search').oninput=list;refresh();

el('view-mode').onchange=drawTeam;
