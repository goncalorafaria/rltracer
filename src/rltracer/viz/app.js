const state = { prompts: [] };
const queryParams = new URLSearchParams(window.location.search);
const selectedRun = queryParams.get("run");
const selectedStep = queryParams.get("step");

const element = (id) => document.getElementById(id);

function escapeHtml(value) {
  const node = document.createElement("span");
  node.textContent = value;
  return node.innerHTML;
}

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
  let state = "not scored";
  let css = "unscored";
  if (correct === 1 || correct === true) {
    state = "correct";
    css = "correct";
  } else if (typeof correct === "number" || correct === false) {
    state = "incorrect";
    css = "incorrect";
  }
  const score = rewards.reward && typeof rewards.reward.score === "number" ? rewards.reward.score : null;
  const details = [
    '<span>Final-label correctness <b>' + escapeHtml(String(correct ?? "not recorded")) + "</b></span>",
    score === null ? "" : '<span>Aggregate reward <b>' + escapeHtml(String(score)) + "</b></span>",
    typeof outcome?.is_completed === "boolean" ? '<span>Completed <b>' + (outcome.is_completed ? "yes" : "no") + "</b></span>" : "",
  ].filter(Boolean).join("");
  return '<section class="outcome ' + css + '">' +
    '<span class="outcome-state">' + state + "</span>" +
    '<div class="outcome-details">' + details + "</div></section>";
}

function renderMessage(message) {
  const reasoning = ["reasoning_content", "reasoning", "thinking", "analysis"]
    .map((key) => message[key])
    .filter((value) => String(value ?? "").trim())
    .map((value) => String(value).trim());
  let content = String(message.content ?? "");
  content = content.replace(/<think>([\s\S]*?)(?:<\/think>|$)/gi, (_, thought) => {
    if (thought.trim()) reasoning.push(thought.trim());
    return "";
  }).trim();
  const uniqueReasoning = [...new Set(reasoning)];
  const thinking = uniqueReasoning.length
    ? '<details class="thinking"><summary>Thinking <span>' + uniqueReasoning.length +
      (uniqueReasoning.length === 1 ? " note" : " notes") +
      '</span></summary><pre>' + escapeHtml(uniqueReasoning.join("\n\n")) + "</pre></details>"
    : "";
  const calls = message.tool_calls
    ? '<pre class="calls">' + escapeHtml(JSON.stringify(message.tool_calls, null, 2)) + "</pre>"
    : "";
  const name = message.name ? " · " + message.name : "";
  return '<article class="message ' + message.role + '">' +
    "<h3>" + message.role + name + "</h3>" +
    thinking +
    (content ? "<pre>" + escapeHtml(content) + "</pre>" : "") +
    calls +
    "</article>";
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
  element("trace").innerHTML = renderOutcome(trace.outcome) + trace.messages.map(renderMessage).join("");
  setStatus("trajectory " + id + " · " + trace.messages.length + " messages");
}
function trajectoryResult(trace) {
  const correct = trace.correct_final_label;
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
element("steps").onchange = loadPrompts;
element("split").onchange = loadSteps;
element("search").oninput = renderPrompts;
loadSteps().catch((error) => {
  setStatus("Error");
  element("viewer").innerHTML =
    '<div class="empty"><h1>Could not load traces</h1><p>' +
    escapeHtml(error.message) + "</p></div>";
});
