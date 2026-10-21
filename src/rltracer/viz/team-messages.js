// Compact inbox presentation used only by the team views.
window.RLTracerTeamMessages = {
  fixedLeader(traces, names) {
    const leaders = [...new Set(traces.map(t => t.metadata?.society?.fixed_leader)
      .filter(name => typeof name === 'string' && name.length))];
    return leaders.length === 1 && names.includes(leaders[0]) ? leaders[0] : null;
  },
  leaderBadge(name, leader) {
    return name === leader ? ' <span class="fixed-leader-badge" title="Only this agent’s correctness determines team success">Leader</span>' : '';
  },
  leaderSummary(trace, names) {
    const society = trace.metadata?.society;
    if (typeof society?.fixed_leader === 'string' && names.includes(society.fixed_leader)) {
      const leader = RLTracerMessages.escapeHtml(society.fixed_leader);
      return '<div class="leader-summary"><b>Fixed leader: </b>'+leader+
        ' · Only this agent’s correctness counts toward team success.</div>';
    }
    const discovery = society?.discovery;
    const events = society?.events || [];
    const hasLeader = events.some(e => e.tool === 'select_leader') ||
      trace.messages.some(m => m.role === 'system' && String(m.content).includes('select_leader'));
    if (!hasLeader) return '';
    const name = discovery?.selected_leader;
    const valid = typeof name === 'string' && names.includes(name);
    return '<div class="leader-summary'+(valid ? '' : ' invalid')+'"><b>Final leader: </b>'+
      RLTracerMessages.escapeHtml(String(name ?? 'Not selected')) +
      (valid ? '' : ' · −0.5')+'</div>';
  },
  render(message, {links = [], heading = true} = {}) {
    let result;
    try { result = JSON.parse(message.content); } catch (_) { /* ordinary output */ }
    const items = result?.messages;
    if (message.role !== 'tool' || !Array.isArray(items) ||
        (message.name && message.name !== 'read_messages') ||
        !items.every(item => item && typeof item.user_id === 'string' && typeof item.text === 'string')) {
      return RLTracerMessages.renderMessage(message);
    }
    const esc = RLTracerMessages.escapeHtml;
    const messages = items.map((item, index) => {
      const link = links[index];
      const sender = link?.send
        ? '<a href="#causal-event-' + link.send.index + '">From ' + esc(item.user_id) + ' · ↑ #' + (link.send.index + 1) + '</a>'
        : 'From ' + esc(item.user_id);
      const note = link && !link.send
        ? '<small>' + (link.ambiguous ? 'Ambiguous repeated send' : 'Send not matched') + '</small>' : '';
      return '<div class="inbox-item"><div class="inbox-sender">' + sender + '</div><div class="inbox-text">' + esc(item.text) + '</div>' + note + '</div>';
    }).join('');
    return '<section class="compact-inbox">' + (heading ? '<div class="inbox-title">Read messages</div>' : '') +
      (messages || '<div class="inbox-empty">No new messages.</div>') +
      (result.notice ? '<p class="muted">' + esc(String(result.notice)) + '</p>' : '') +
      '<details class="inbox-raw"><summary>Raw result</summary><pre>' + esc(message.content) + '</pre></details></section>';
  }
};
