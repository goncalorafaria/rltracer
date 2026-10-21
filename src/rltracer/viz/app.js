let markdownPreference = true;
try { markdownPreference = localStorage.getItem("rltracer-markdown") !== "off"; } catch (_) { /* storage may be disabled */ }
const state = { prompts: [], markdown: markdownPreference, currentTrace: null };
const queryParams = new URLSearchParams(window.location.search);
const selectedRun = queryParams.get("run");
const selectedStep = queryParams.get("step");

const element = (id) => document.getElementById(id);
element('team-view-link').onclick = (event) => {
  event.preventDefault();
  const target = new URL('/teams.html', location.origin);
  if (selectedRun) target.searchParams.set('run', selectedRun);
  if (element('steps').value) target.searchParams.set('step', element('steps').value);
  target.searchParams.set('split', element('split').value);
  location.assign(target.href);
};
if (['train', 'eval'].includes(queryParams.get('split'))) {
  element('split').value = queryParams.get('split');
}

const { escapeHtml, renderJsonFields, renderMarkdown } = RLTracerMessages;

function setStatus(value) {
  element("status").textContent = value;
}

async function api(path) {
  const url = new URL(path, window.location.origin);
  if (selectedRun) url.searchParams.set("run", selectedRun);
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function renderPrompts() {
  const query = element("search").value.toLowerCase();
  const prompts = state.prompts.filter((prompt) =>
    (prompt.key + " " + prompt.preview).toLowerCase().includes(query)
  );
  element("meta").textContent = prompts.length + " of " + state.prompts.length + " prompts";
  element("prompts").innerHTML = prompts.map((prompt) =>
    '<button class="prompt" data-key="' + prompt.key + '">' +
      '<span class="key">' + prompt.key + '</span>' +
      '<span class="count">' + prompt.rollout_count + '</span>' +
      '<div class="preview">' + escapeHtml(prompt.preview) + '</div>' +
    '</button>'
  ).join("");
  document.querySelectorAll(".prompt").forEach((button) => {
    button.onclick = () => loadTrajectories(button.dataset.key);
  });
}

async function loadPrompts() {
  setStatus("Indexing step and loading prompts...");
  state.prompts = await api("/api/steps/" + element("steps").value + "/prompts?split=" + element("split").value);
  renderPrompts();
  element("viewer").innerHTML = renderPassRateHistogram(state.prompts) +
    '<div class="empty"><h1>Select a prompt</h1><p>' +
    state.prompts.length + " prompt groups indexed.</p></div>";
  setStatus(state.prompts.length + " prompt groups");
}


function renderOutcome(outcome) {
  const metrics = outcome && outcome.metrics && typeof outcome.metrics === "object" ? outcome.metrics : {};
  const rewards = outcome && outcome.rewards && typeof outcome.rewards === "object" ? outcome.rewards : {};
  const correct = metrics.correct_final_label;
  const rewardRows = Object.entries(rewards).filter(([, value]) => value && typeof value === "object" && typeof value.score === "number");
  let state = rewardRows.length ? "scored" : "not scored";
  let css = rewardRows.length ? "scored" : "unscored";
  if (correct === 1 || correct === true) {
    state = "correct";
    css = "correct";
  } else if (typeof correct === "number" || correct === false) {
    state = "incorrect";
    css = "incorrect";
  }
  const pretty = value => typeof value === "number" ? value.toLocaleString(undefined, { maximumFractionDigits: 5 }) : String(value);
  const details = [
    '<span>Final-label correctness <b>' + escapeHtml(String(correct ?? "not recorded")) + "</b></span>",
    ...rewardRows.map(([name, value]) => '<span>' + escapeHtml(name) + ' <b>' + escapeHtml(pretty(value.score)) + "</b></span>"),
    typeof outcome?.is_completed === "boolean" ? '<span>Completed <b>' + (outcome.is_completed ? "yes" : "no") + "</b></span>" : "",
  ].filter(Boolean).join("");
  const metricRows = Object.entries(metrics);
  const metricPanel = metricRows.length
    ? '<details class="metric-panel"><summary>Trace metrics (' + metricRows.length + ')</summary><div class="outcome-details">' +
      metricRows.map(([name, value]) => '<span>' + escapeHtml(name) + ' <b>' + escapeHtml(pretty(value)) + "</b></span>").join("") +
      "</div></details>"
    : "";
  return '<section class="outcome ' + css + '">' +
    '<span class="outcome-state">' + state + "</span>" +
    '<div class="outcome-body"><div class="outcome-details">' + details + "</div>" + metricPanel + "</div></section>";
}

function renderMessage(message) {
  return RLTracerMessages.renderMessage(message, state.markdown);
}

function renderPassRateHistogram(prompts) {
  const bins = Array(10).fill(0);
  let scored = 0;
  for (const prompt of prompts) {
    if (typeof prompt.pass_at_1 !== "number") continue;
    bins[Math.min(Math.floor(prompt.pass_at_1 * 10), 9)] += 1;
    scored += 1;
  }
  const maximum = Math.max(...bins, 1);
  const bars = bins.map((count, index) => {
    const lower = (index / 10).toFixed(1);
    const upper = ((index + 1) / 10).toFixed(1);
    return '<div class="pass-histogram-bar" title="pass@1 ' + lower + '–' + upper + ': ' + count + ' prompts">' +
      '<b>' + count + '</b><i style="height:' + (count / maximum * 100) + '%"></i><span>' + lower + '–' + upper + '</span></div>';
  }).join("");
  return '<section class="pass-histogram"><h2>Prompt pass@1 distribution <small>step ' + element("steps").value + ' · ' + scored + ' scored prompts</small></h2>' +
    '<div class="pass-histogram-bars" role="img" aria-label="Histogram of prompt pass at one rates for the selected step">' + bars + '</div></section>';
}

async function loadTrace(id, button) {
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
  button.classList.add("active");
  setStatus("Decoding trajectory...");
  const trace = await api("/api/trajectories/" + id);
  state.currentTrace = trace;
  renderLoadedTrace(trace);
  setStatus("trajectory " + id + " · " + trace.messages.length + " messages");
}

function renderJudgeAudit(audit) {
  const judgments = Array.isArray(audit?.judgments) ? audit.judgments : [];
  if (!judgments.length) return "";
  const score = typeof audit.score === "number" ? " · score " + audit.score.toFixed(3) : "";
  const rows = judgments.map((judgment, index) => {
    const judgeTrace = judgment.trace || {};
    const messages = Array.isArray(judgeTrace.messages) ? judgeTrace.messages : [];
    const label = String(judgment.label || judgeTrace.label || "unlabeled");
    const rubric = String(judgment.text || judgeTrace.criteria || "");
    const feedback = String(judgment.feedback || judgeTrace.feedback || "");
    const weight = typeof judgment.weight === "number" ? " · weight " + judgment.weight : "";
    return '<article class="judge-rubric-card"><header><b>Rubric ' + (index + 1) + '</b><span class="judge-verdict ' + escapeHtml(label.toLowerCase()) + '">' + escapeHtml(label) + escapeHtml(weight) + '</span></header>' +
      (rubric ? '<div class="judge-rubric-text">' + renderMarkdown(rubric) + '</div>' : "") +
      (feedback ? '<div class="judge-feedback"><b>Feedback</b>' + renderMarkdown(feedback) + '</div>' : "") +
      '<details class="judge-transcript" data-judge-index="' + index + '"><summary>Judge model trace · ' + messages.length + ' messages</summary><div class="judge-transcript-body"></div></details></article>';
  }).join("");
  return '<section class="judge-audit"><h2>Judge model traces <small>' + judgments.length + ' rubric judgments · ' + escapeHtml(String(audit.status || "")) + escapeHtml(score) + '</small></h2><div class="judge-rubric-list">' + rows + '</div></section>';
}

// Replay tool IDs are assigned by the harness; never infer the handoff from a percentage.
function renderReplayConversation(messages) {
  let lastReplay = -1;
  messages.forEach((message, index) => {
    const ids = [message.tool_call_id, ...(message.tool_calls || []).map(call => call.id)];
    if (ids.some(id => typeof id === "string" && /^replay_\d+$/.test(id))) lastReplay = index;
  });
  const start = lastReplay < 0 ? -1 : messages.findIndex((message, index) => index > lastReplay && message.role === "assistant");
  return messages.map((message, index) =>
    (index === start ? '<div class="replay-handoff" role="separator" aria-label="Agent continuation begins"><span>Replayed prefix ends · Agent continuation begins</span></div>' : '') + renderMessage(message)
  ).join("");
}

function renderLoadedTrace(trace) {
  const metadata = trace.metadata || {};
  const context = metadata.task_type === "RubricHubJudgeTask"
    ? '<section class="judge-context"><h2>Judge rollout</h2><p>' +
      escapeHtml(metadata.environment || "") +
      (metadata.task_index !== null && metadata.task_index !== undefined ? " · task " + escapeHtml(String(metadata.task_index)) : "") +
      (metadata.model ? " · " + escapeHtml(metadata.model) : "") +
      '</p>' +
      (metadata.task_prompt ? '<details><summary>Judge task prompt</summary><pre>' + escapeHtml(JSON.stringify(metadata.task_prompt, null, 2)) + "</pre></details>" : "") +
      (metadata.judge_input ? '<details><summary>Judge input / rubrics</summary><pre>' + escapeHtml(typeof metadata.judge_input === "string" ? metadata.judge_input : JSON.stringify(metadata.judge_input, null, 2)) + "</pre></details>" : "") +
      (metadata.num_input_tokens !== null || metadata.num_output_tokens !== null ? '<p>Tokens in: ' + escapeHtml(String(metadata.num_input_tokens ?? "—")) + ' · tokens out: ' + escapeHtml(String(metadata.num_output_tokens ?? "—")) + "</p>" : "") +
      (metadata.calls?.length ? '<details><summary>Model calls (' + metadata.calls.length + ')</summary>' + renderJsonFields(metadata.calls) + "</details>" : "") +
      (metadata.available_tools?.length ? '<details><summary>Available tools (' + metadata.available_tools.length + ')</summary><pre>' + escapeHtml(JSON.stringify(metadata.available_tools, null, 2)) + "</pre></details>" : "") +
      '</section>'
    : "";
  const judgeAudit = renderJudgeAudit(metadata.judge_audit);
  element("trace").innerHTML = context + judgeAudit + renderOutcome(trace.outcome) + renderReplayConversation(trace.messages);
  element("trace").querySelectorAll(".judge-transcript[data-judge-index]").forEach((details) => {
    details.addEventListener("toggle", () => {
      if (!details.open || details.dataset.rendered) return;
      const judgment = metadata.judge_audit.judgments[Number(details.dataset.judgeIndex)];
      const messages = Array.isArray(judgment?.trace?.messages) ? judgment.trace.messages : [];
      details.querySelector(".judge-transcript-body").innerHTML = messages.map(renderMessage).join("");
      details.dataset.rendered = "true";
    });
  });
}
function trajectoryResult(trace) {
  const correct = trace.correct_final_label;
  if (correct === null || correct === undefined) {
    return { css: "unscored", label: "unknown" };
  }
  if (correct === 1 || correct === true) {
    return { css: "correct", label: "correct" };
  }
  if (typeof correct === "number" || correct === false) {
    return { css: "incorrect", label: "wrong" };
  }
  return { css: "unscored", label: "unscored" };
}


async function loadTrajectories(key) {
  const traces = await api(
    "/api/steps/" + element("steps").value + "/prompts/" + encodeURIComponent(key) +
    "/trajectories?split=" + element("split").value
  );
  element("viewer").innerHTML =
    '<div class="tabs">' +
    traces.map((trace, index) => {
      const result = trajectoryResult(trace);
      return '<button class="tab ' + result.css + '" data-id="' + trace.id + '">' +
        '<span class="result ' + result.css + '">' + result.label + '</span> ' +
        (trace.split ? '<span class="split">' + escapeHtml(trace.split) + "</span> " : "") +
        "trajectory " + (index + 1) + "</button>";
    }).join("") +
    '</div><div id="trace"></div>';
  document.querySelectorAll(".tab").forEach((button) => {
    button.onclick = () => loadTrace(button.dataset.id, button);
  });
  if (traces.length) await loadTrace(traces[0].id, document.querySelector(".tab"));
}

async function loadSteps() {
  setStatus("Loading rollout steps…");
  const steps = await api("/api/steps?split=" + element("split").value);
  element("steps").innerHTML = steps.map((step) =>
    '<option value="' + step.number + '">step ' + step.number + "</option>"
  ).join("");
  if (!steps.length) {
    state.prompts = [];
    renderPrompts();
    element("viewer").innerHTML = '<div class="empty"><h1>No traces</h1><p>This split has no rollout steps.</p></div>';
    return;
  }
  const requestedStep = steps.find((step) => String(step.number) === selectedStep);
  element("steps").value = (requestedStep || steps[steps.length - 1]).number;
  await loadPrompts();
}

if (!selectedRun) element("browse-link").hidden = true;
function updateMarkdownButton() {
  const button = element("markdown-toggle");
  button.textContent = "Markdown: " + (state.markdown ? "On" : "Off");
  button.setAttribute("aria-pressed", String(state.markdown));
  button.title = state.markdown ? "Show assistant text as Markdown" : "Show assistant text as plain text";
}
element("markdown-toggle").onclick = () => {
  state.markdown = !state.markdown;
  try { localStorage.setItem("rltracer-markdown", state.markdown ? "on" : "off"); } catch (_) { /* storage may be disabled */ }
  updateMarkdownButton();
  if (state.currentTrace) renderLoadedTrace(state.currentTrace);
};
updateMarkdownButton();
element("steps").onchange = loadPrompts;
element("split").onchange = loadSteps;
element("search").oninput = renderPrompts;
loadSteps().catch((error) => {
  setStatus("Error");
  element("viewer").innerHTML =
    '<div class="empty"><h1>Could not load traces</h1><p>' +
    escapeHtml(error.message) + "</p></div>";
});

fetch("/api/config").then(r => r.json()).then(config => {
  element("team-view-link").hidden = !config.team_view;
}).catch(() => { element("team-view-link").hidden = true; });

// Keep discovering live steps without resetting the user's selected rollout.
async function refreshLiveSteps() {
  try {
    const split = element("split").value;
    const steps = await api("/api/steps?split=" + split);
    if (split === element("split").value) {
      const picker = element("steps");
      const previous = picker.value;
      picker.innerHTML = steps.map(step => '<option value="' + step.number + '">step ' + step.number + '</option>').join("");
      if (steps.some(step => String(step.number) === previous)) picker.value = previous;
      else if (steps.length) { picker.value = steps[steps.length - 1].number; await loadPrompts(); }
      picker.title = "Live index: steps and rollout counts populate progressively. Reselect a step to refresh its prompts.";
    }
  } catch (_) { /* Keep the existing view during a transient refresh failure. */ }
  setTimeout(refreshLiveSteps, 5000);
}
setTimeout(refreshLiveSteps, 5000);
