// Causal graph: local agent order plus uniquely matched send → read edges.
window.RLTracerCausal = (() => {
  const escape = value => RLTracerMessages.escapeHtml(String(value ?? ''));
  function parse(value) {
    if (typeof value !== 'string') return value;
    try { return JSON.parse(value); } catch (_) { return null; }
  }
  function inboxReads(trace) {
    const calls = new Map(), reads = [];
    for (const message of trace.messages) {
      for (const call of message.tool_calls || []) calls.set(call.id, (call.function || call).name);
      if (message.role !== 'tool') continue;
      const body = parse(message.content);
      const name = message.name || calls.get(message.tool_call_id);
      // Older traces can omit the tool name and ID; the inbox schema is distinctive.
      if (name === 'read_messages' || (!name && Array.isArray(body?.messages))) {
        reads.push({message, items: Array.isArray(body?.messages) ? body.messages : null});
      }
    }
    return reads;
  }
  function render(team, traces) {
    const logs = traces.map(t => t.metadata?.society?.events).filter(Array.isArray);
    const events = logs.sort((a,b) => b.length-a.length)[0] || [];
    if (!events.length) return '<p class="causal-note">This rollout has no shared event log. Use Agent conversations to inspect its messages.</p>';
    const lanes = team.agents.map(a => a.name);
    const fixedLeader = RLTracerTeamMessages.fixedLeader(traces, lanes);
    const reads = new Map(traces.map((trace,i) => [lanes[i], inboxReads(trace)]));
    const offsets = new Map(), sends = [], rows = [];
    const lastChoices = new Map();
    for (const [index,event] of events.entries()) {
      const who = event.sender || event.agent;
      if (!lanes.includes(who)) continue;
      const row = {index, who, event, links: [], targets: []};
      if (event.event === 'message') {
        row.used = new Set(); sends.push(row); rows.push(row);
      } else if (event.event === 'inbox_read') {
        const offset = offsets.get(who) || 0;
        row.read = (reads.get(who) || [])[offset]; offsets.set(who, offset+1);
        if (row.read?.items?.length === event.received) {
          for (const item of row.read.items) {
            const candidates = sends.filter(s => s.event.sender === item.user_id && s.event.text === item.text &&
              (s.event.to === 'all' || s.event.to === who) && s.who !== who && !s.used.has(who));
            if (candidates.length === 1) {
              const send = candidates[0]; send.used.add(who);
              row.links.push({send, item}); send.targets.push(row);
            } else row.links.push({item, ambiguous: candidates.length > 1});
          }
        } else row.unaligned = true;
        rows.push(row);
      } else if (event.event === 'discovery_tool' && event.tool === 'select_leader') {
        row.leader = event.progress?.selected_leader ?? event.arguments?.agent_name;
        row.valid = typeof row.leader === 'string' && lanes.includes(row.leader);
        row.previousChoice = lastChoices.get(who);
        if (row.previousChoice) row.previousChoice.finalChoice = false;
        row.finalChoice = true;
        lastChoices.set(who, row);
        rows.push(row);
      } else if (['agent_wait','agent_wait_end','wait_budget_exhausted','invalid_wait_timeout','message_interrupt','message_resume','idle','agent_idle','final','answer'].includes(event.event)) {
        rows.push(row);
      }
    }
    // Kahn frontiers: breadth-first layers, but wait for ALL predecessors.
    // Plain shortest-path BFS could place a read before another send it consumes.
    const successors = new Map(rows.map(row => [row, new Set()]));
    const indegree = new Map(rows.map(row => [row, 0]));
    const previous = new Map();
    function edge(from, to) {
      if (successors.get(from).has(to)) return;
      successors.get(from).add(to);
      indegree.set(to, indegree.get(to) + 1);
    }
    for (const row of rows) {
      if (previous.has(row.who)) edge(previous.get(row.who), row);
      previous.set(row.who, row);
      for (const link of row.links) if (link.send) edge(link.send, row);
    }
    const layers = [];
    let frontier = rows.filter(row => indegree.get(row) === 0);
    while (frontier.length) {
      layers.push(frontier);
      const next = [];
      for (const row of frontier) for (const child of successors.get(row)) {
        indegree.set(child, indegree.get(child) - 1);
        if (indegree.get(child) === 0) next.push(child);
      }
      frontier = next.sort((a,b) => a.index-b.index);
    }
    const card = row => {
      const event = row.event, id = 'causal-event-'+row.index;
      let title, body='', cls='';
      if (event.event === 'message') {
        title='Sent → '+event.to; cls='sent';
        body='<p>'+escape(event.text)+'</p><div class="muted">Send completed; inbox contents still need to be read.</div>';
        body+=row.targets.length ? '<p>'+row.targets.map(r => '<a href="#causal-event-'+r.index+'">Read by '+escape(r.who)+' · #'+(r.index+1)+' ↓</a>').join('<br>')+'</p>' : '<p class="muted">No uniquely matched read in the saved trace.</p>';
      } else if (event.event === 'inbox_read') {
        title='Read inbox · '+event.received+' messages'; cls='read';
        if (row.unaligned) body+='<p class="muted">Read payload could not be aligned reliably; message links omitted.</p>';
        else if (row.read) body+=RLTracerTeamMessages.render(row.read.message, {links: row.links, heading: false});
        if (event.omitted) body+='<p class="muted">'+event.omitted+' messages omitted by the harness.</p>';
      } else if (event.event === 'agent_wait' || event.event === 'agent_wait_end') {
        const ended = event.event === 'agent_wait_end';
        const reasons = {message:'message available', timeout:'timer expired',
          team_finished:'all teammates finished', teammate_finished:'teammate finished',
          no_active_teammates:'no active teammates', all_agents_waiting:'all teammates released',
          work_phase_ended:'work phase ended'};
        title = ended ? 'Wait ended · '+(reasons[event.reason] || event.reason || 'returned') : 'Wait → '+(event.to || 'all');
        cls = 'wait';
        if (ended) body += '<p class="muted">Waiting for: '+escape(event.to || 'all')+'</p>';
        if (typeof event.timeout_ms === 'number') body += '<div class="muted">Requested: '+escape(event.timeout_ms)+' ms</div>';
        const elapsed = event.waited_ms ?? (typeof event.seconds === 'number' ? event.seconds * 1000 : null);
        if (typeof elapsed === 'number') body += '<div>Actually waited: '+escape(elapsed.toFixed(1))+' ms</div>';
        if (typeof event.waits_remaining === 'number') body += '<div class="muted">Waits remaining: '+escape(event.waits_remaining)+'</div>';
        if (event.notice) body += '<p>'+escape(event.notice)+'</p>';
        if (event.reason === 'message') body += '<p class="muted">Message available; this does not read it.</p>';
      } else if (event.event === 'wait_budget_exhausted' || event.event === 'invalid_wait_timeout') {
        cls = 'wait invalid';
        title = event.event === 'wait_budget_exhausted' ? 'Wait rejected · budget exhausted' : 'Wait rejected · invalid timeout';
        body = '<p>'+ (event.event === 'wait_budget_exhausted' ? 'You cannot wait anymore; you spent your wait budget.' : 'The timeout was invalid; no wait was started.') +'</p>';
        if (typeof event.penalty === 'number') body += '<div class="muted">Reward penalty: '+escape(event.penalty)+'</div>';
      } else if (event.tool === 'select_leader') {
        title = (row.previousChoice ? 'Changed leader → ' : 'Chose leader → ') + (row.leader ?? '(missing name)');
        cls = 'leader' + (row.valid ? '' : ' invalid');
        if (row.previousChoice) body += '<p class="muted">Previously: '+escape(row.previousChoice.leader ?? '(missing name)')+'</p>';
        body += '<p>'+ (row.valid ? (row.leader === row.who ? 'Selected self.' : 'Selected teammate.') : 'Invalid selection · −0.5 unless changed.') +'</p>';
        body += '<div class="muted">'+(row.finalChoice ? 'Last recorded choice' : 'Replaced by a later choice')+'</div>';
      } else {
        const titles={message_interrupt:'Generation interrupted · inbox notification',message_resume:'Generation resumed after interruption',idle:'Idle',agent_idle:'Idle',final:'Final answer',answer:'Answer'};
        title=titles[event.event] || event.event; cls='interrupt';
        if (typeof event.latency_seconds === 'number') body='<p class="muted">Recorded latency: '+event.latency_seconds.toFixed(3)+' s</p>';
        if (event.event==='message_interrupt') body+='<p class="muted">Notification only — this does not read the message.</p>';
      }
      return '<article id="'+id+'" data-parents="'+row.links.filter(link=>link.send).map(link=>link.send.index).join(',')+'" class="causal-event '+cls+'"><div class="causal-event-index">Event #'+(row.index+1)+'</div><h4>'+escape(title)+'</h4>'+body+'</article>';
    };
    return '<p class="causal-note">Causal graph · breadth-first dependency layers. Each event, including wait start and return, follows the previous event for its agent and every matched send it reads. Wait durations are measured elapsed time; layer spacing is not wall-clock time. Parallel events share a row; a shared row does not mean simultaneous. Blue arrows connect sends to reads. Original event numbers are preserved. Matches use sender, recipient, and exact text; ambiguous matches stay unlinked.</p>'+
      '<div class="causal-graph"><svg class="causal-edges" aria-hidden="true"></svg><div class="causal-grid" style="--lanes:'+lanes.length+'"><div class="causal-heading">Layer</div>'+lanes.map(name=>'<div class="causal-heading">'+escape(name)+RLTracerTeamMessages.leaderBadge(name,fixedLeader)+'</div>').join('')+
      layers.map((layer,i)=>'<div class="causal-number">L'+(i+1)+'</div>'+lanes.map(name=>'<div class="causal-cell conversation">'+layer.filter(row=>name===row.who).map(card).join('')+'</div>').join('')).join('')+'</div></div>'+
      '<details class="causal-details"><summary>All recorded events ('+events.length+')</summary>'+RLTracerMessages.renderJsonFields(events)+'</details>';
  }
  let resizeObserver;
  function mount(root) {
    if (resizeObserver) resizeObserver.disconnect();
    const graph = root.querySelector('.causal-graph');
    if (!graph) return;
    const svg = graph.querySelector('.causal-edges');
    function drawEdges() {
      const bounds = graph.getBoundingClientRect();
      svg.setAttribute('viewBox', '0 0 '+bounds.width+' '+bounds.height);
      svg.innerHTML = '<defs><marker id="causal-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#7c9cff"/></marker></defs>';
      for (const target of graph.querySelectorAll('[data-parents]')) {
        for (const parent of target.dataset.parents.split(',').filter(Boolean)) {
          const source = graph.querySelector('#causal-event-'+parent);
          if (!source) continue;
          const from = source.getBoundingClientRect(), to = target.getBoundingClientRect();
          // Route through lane gutters so arrows never cover message text.
          const leftToRight = from.left < to.left;
          const x1 = (leftToRight ? from.right : from.left)-bounds.left;
          const x2 = (leftToRight ? to.left : to.right)-bounds.left;
          const y1 = from.top-bounds.top+14, y2 = to.top-bounds.top+14;
          const gutter1 = x1+(leftToRight ? 8 : -8);
          const gutter2 = x2+(leftToRight ? -8 : 8);
          const bridge = to.top-bounds.top-14;
          const path = document.createElementNS('http://www.w3.org/2000/svg','path');
          path.setAttribute('d', 'M '+x1+' '+y1+' H '+gutter1+' V '+bridge+' H '+gutter2+' V '+y2+' H '+x2);
          path.setAttribute('marker-end','url(#causal-arrow)');
          svg.appendChild(path);
        }
      }
    }
    resizeObserver = new ResizeObserver(drawEdges);
    resizeObserver.observe(graph);
    for (const card of graph.querySelectorAll('.causal-event')) resizeObserver.observe(card);
    drawEdges();
  }
  return {render, mount};
})();
