let markdownPreference = true;
try { markdownPreference = localStorage.getItem("rltracer-markdown") !== "off"; } catch (_) { /* storage may be disabled */ }
const state = { prompts: [], markdown: markdownPreference, currentTrace: null };
const queryParams = new URLSearchParams(window.location.search);
const selectedRun = queryParams.get("run");
const selectedStep = queryParams.get("step");

const element = (id) => document.getElementById(id);

function escapeHtml(value) {
  const node = document.createElement("span");
  node.textContent = value;
  return node.innerHTML;
}

function markdownInline(source) {
  let text = escapeHtml(source);
  const code = [];
  text = text.replace(/`([^`\n]+)`/g, (_match, value) => {
    const index = code.push("<code>" + value + "</code>") - 1;
    return "\u0000" + index + "\u0000";
  });
  text = text.replace(/!?\[([^\]]+)\]\(([^)\s]+)(?:\s+"([^"]*)")?\)/g, (match, label, href) => {
    const decoded = href.replace(/&amp;/g, "&").trim();
    if (!/^(https?:|mailto:|\/|#|\.\/|\.\.\/)/i.test(decoded)) return label;
    return '<a href="' + escapeHtml(decoded) + '" target="_blank" rel="noopener noreferrer">' + label + "</a>";
  });
  text = text.replace(/\*\*(.+?)\*\*|__(.+?)__/g, (_m, a, b) => "<strong>" + (a || b) + "</strong>");
  text = text.replace(/~~(.+?)~~/g, "<del>$1</del>");
  text = text.replace(/(^|[\s(])\*([^*\n]+)\*(?=$|[\s).,!?:;])/g, "$1<em>$2</em>");
  text = text.replace(/(^|[\s(])_([^_\n]+)_(?=$|[\s).,!?:;])/g, "$1<em>$2</em>");
  text = text.replace(/&lt;(https?:\/\/[^&]+)&gt;/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
  return text.replace(/\u0000(\d+)\u0000/g, (_m, index) => code[Number(index)]);
}

function renderJsonFields(value) {
  const entries = Array.isArray(value) ? value.map((item, index) => [String(index), item]) : Object.entries(value || {});
  if (!entries.length) return '<div class="json-empty">' + (Array.isArray(value) ? "Empty array" : "Empty object") + "</div>";
  return '<div class="json-fields">' + entries.map(([key, item]) =>
    '<div class="json-field"><div class="json-key">' + escapeHtml(key) + '</div><div class="json-value">' + renderJsonValue(item) + "</div></div>"
  ).join("") + "</div>";
}

function renderJsonValue(value) {
  if (value === null) return '<span class="json-scalar json-null">null</span>';
  if (typeof value === "string") return '<span class="json-scalar json-string">' + escapeHtml(value) + "</span>";
  if (typeof value === "number" || typeof value === "boolean") return '<span class="json-scalar json-primitive">' + escapeHtml(String(value)) + "</span>";
  const isArray = Array.isArray(value);
  const size = isArray ? value.length : Object.keys(value).length;
  return '<details class="json-nested"><summary>' + (isArray ? "Array" : "Object") + " · " + size + (isArray ? " items" : " fields") + "</summary>" + renderJsonFields(value) + "</details>";
}

function renderOutputText(value) {
  const text = String(value ?? "");
  try {
    const parsed = JSON.parse(text);
    if (parsed && typeof parsed === "object") {
      return '<details class="json-output" open><summary>Structured JSON output</summary>' + renderJsonFields(parsed) +
        '</details><details class="tool-output"><summary>Raw output</summary><pre>' + escapeHtml(text) + "</pre></details>";
    }
  } catch (_) { /* tool output is ordinary text */ }
  return '<pre>' + escapeHtml(text) + "</pre>";
}

function renderMarkdown(source) {
  const lines = String(source).replace(/\r\n?/g, "\n").split("\n");
  const output = [];
  let paragraph = [];
  let listType = null;
  let listItems = [];
  let codeLines = null;
  const flushParagraph = () => {
    if (paragraph.length) output.push("<p>" + markdownInline(paragraph.join(" ")) + "</p>");
    paragraph = [];
  };
  const flushList = () => {
    if (listType) output.push("<" + listType + ">" + listItems.map(item => "<li>" + markdownInline(item) + "</li>").join("") + "</" + listType + ">");
    listType = null;
    listItems = [];
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (codeLines) {
      if (/^\s*```/.test(line)) {
        output.push("<pre class=\"markdown-code\"><code>" + escapeHtml(codeLines.join("\n")) + "</code></pre>");
        codeLines = null;
      } else codeLines.push(line);
      continue;
    }
    if (/^\s*```/.test(line)) {
      flushParagraph(); flushList(); codeLines = []; continue;
    }
    const tableSep = i + 1 < lines.length && /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(lines[i + 1]);
    if (line.includes("|") && tableSep) {
      flushParagraph(); flushList();
      const cells = row => row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(cell => cell.trim());
      const heads = cells(line); i += 1;
      let table = "<table><thead><tr>" + heads.map(cell => "<th>" + markdownInline(cell) + "</th>").join("") + "</tr></thead><tbody>";
      while (i + 1 < lines.length && lines[i + 1].includes("|") && lines[i + 1].trim()) {
        i += 1; table += "<tr>" + cells(lines[i]).map(cell => "<td>" + markdownInline(cell) + "</td>").join("") + "</tr>";
      }
      output.push(table + "</tbody></table>"); continue;
    }
    const heading = line.match(/^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (heading) {
      flushParagraph(); flushList(); const level = heading[1].length;
      output.push("<h" + level + ">" + markdownInline(heading[2]) + "</h" + level + ">"); continue;
    }
    if (/^\s{0,3}([-*_]\s*){3,}$/.test(line)) { flushParagraph(); flushList(); output.push("<hr>"); continue; }
    const item = line.match(/^\s{0,3}([-+*]|\d+[.)])\s+(.+)$/);
    if (item) {
      flushParagraph(); const type = /^\d/.test(item[1]) ? "ol" : "ul";
      if (listType && listType !== type) flushList();
      listType = type; listItems.push(item[2]); continue;
    }
    if (/^\s*>/.test(line)) {
      flushParagraph(); flushList(); const quote = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) quote.push(lines[i++].replace(/^\s*>\s?/, ""));
      i -= 1; output.push("<blockquote>" + renderMarkdown(quote.join("\n")) + "</blockquote>"); continue;
    }
    if (!line.trim()) { flushParagraph(); flushList(); continue; }
    flushList(); paragraph.push(line.trim());
  }
  if (codeLines) output.push("<pre class=\"markdown-code\"><code>" + escapeHtml(codeLines.join("\n")) + "</code></pre>");
  flushParagraph(); flushList();
  return '<div class="markdown-body">' + output.join("") + "</div>";
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
  const reasoning = ["reasoning_content", "reasoning", "thinking", "analysis"]
    .map((key) => message[key])
    .filter((value) => String(value ?? "").trim())
    .map((value) => String(value).trim());
  let content = String(message.content ?? "");
  const parsedToolCalls = [];
  content = content.replace(/<tool_call>([\s\S]*?)<\/tool_call>/gi, (_, block) => {
    const name = block.match(/<function=([^>\r\n]+)>/i)?.[1]?.trim();
    if (!name) return "";
    const args = {};
    const parameterPattern = /<parameter=([^>\r\n]+)>([\s\S]*?)<\/parameter>/gi;
    let parameter;
    while ((parameter = parameterPattern.exec(block)) !== null) {
      args[parameter[1].trim()] = parameter[2].trim();
    }
    parsedToolCalls.push({ type: "function", function: { name, arguments: args } });
    return "";
  }).trim();
  content = content.replace(/<think>([\s\S]*?)(?:<\/think>|$)/gi, (_, thought) => {
    if (thought.trim()) reasoning.push(thought.trim());
    return "";
  }).trim();
  const uniqueReasoning = [...new Set(reasoning)];
  const thinking = uniqueReasoning.length
    ? '<details class="thinking"><summary>Thinking <span>' + uniqueReasoning.length +
      (uniqueReasoning.length === 1 ? " note" : " notes") +
      '</span></summary>' + (state.markdown ? renderMarkdown(uniqueReasoning.join("\n\n")) : "<pre>" + escapeHtml(uniqueReasoning.join("\n\n")) + "</pre>") + "</details>"
    : "";
  const toolCalls = [...(Array.isArray(message.tool_calls) ? message.tool_calls : []), ...parsedToolCalls];
  const calls = toolCalls.length
    ? '<div class="tool-calls">' + toolCalls.map((call) => {
        const fn = call.function || call;
        let args = fn.arguments ?? {};
        if (typeof args === "string") {
          try { args = JSON.parse(args); } catch (_) { /* keep raw arguments */ }
        }
        const callType = call.type || (call.function ? "function" : "tool");
        const callName = fn.name || call.name || "tool call";
        const callId = call.id ? '<span class="tool-call-id">' + escapeHtml(call.id) + "</span>" : "";
        const argumentsView = typeof args === "string" ? '<pre>' + escapeHtml(args) + "</pre>" : renderJsonFields(args);
        return '<section class="tool-call"><div class="tool-call-header"><div class="tool-call-title"><span>TOOL CALL</span><b>' + escapeHtml(callName) +
          '</b></div><span class="tool-call-type">' + escapeHtml(callType) + '</span>' + callId +
          '</div><div class="tool-call-body"><div class="tool-call-section-label">Arguments</div>' + argumentsView + "</div></section>";
      }).join("") + "</div>"
    : "";
  const renderedContent = message.role === "tool"
    ? renderToolResponse(content)
    : (content ? (message.role === "assistant" && state.markdown ? renderMarkdown(content) : "<pre>" + escapeHtml(content) + "</pre>") : "");
  const messageName = message.name ? escapeHtml(message.name) : "";
  const messageId = message.tool_call_id ? '<span class="tool-call-id">' + escapeHtml(message.tool_call_id) + "</span>" : "";
  const messageHeading = message.role === "tool"
    ? '<span>TOOL RESULT</span>' + (messageName ? '<b>' + messageName + "</b>" : "") + messageId
    : escapeHtml(message.role || "message") + (messageName ? " · " + messageName : "") + messageId;
  return '<article class="message ' + message.role + '">' +
    "<h3>" + messageHeading + "</h3>" +
    thinking +
    renderedContent +
    calls +
    "</article>";
}

function renderToolResponse(content) {
  let result;
  try { result = JSON.parse(content); } catch (_) {
    return content ? "<pre>" + escapeHtml(content) + "</pre>" : "";
  }
  if (!result || typeof result !== "object" || Array.isArray(result)) {
    return '<pre>' + escapeHtml(JSON.stringify(result, null, 2)) + '</pre>';
  }
  const summary = [];
  if (result.exit_code !== undefined) summary.push("exit " + result.exit_code);
  if (typeof result.execution_time === "number") summary.push(result.execution_time.toFixed(3) + "s");
  let body = summary.length ? '<div class="tool-result-meta">' + escapeHtml(summary.join(" · ")) + "</div>" : "";
  for (const field of ["stdout", "stderr"]) {
    if (result[field]) body += '<details class="tool-output" ' + (field === "stdout" ? "open" : "") +
      '><summary>' + field + '</summary>' + renderOutputText(result[field]) + "</details>";
  }
  const other = Object.fromEntries(Object.entries(result).filter(([key]) => !["stdout", "stderr", "execution_time", "exit_code"].includes(key)));
  if (Object.keys(other).length) body += '<details class="tool-output" open><summary>Result data</summary>' + renderJsonFields(other) + "</details>";
  return body || '<pre>' + escapeHtml(JSON.stringify(result, null, 2)) + '</pre>';
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
  element("trace").innerHTML = context + judgeAudit + renderOutcome(trace.outcome) + trace.messages.map(renderMessage).join("");
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
