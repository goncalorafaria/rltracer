// Shared message formatting for Explorer and Team view.
window.RLTracerMessages = (() => {
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

function renderMessage(message, markdown = true) {
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
      '</span></summary>' + (markdown ? renderMarkdown(uniqueReasoning.join("\n\n")) : "<pre>" + escapeHtml(uniqueReasoning.join("\n\n")) + "</pre>") + "</details>"
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
    : (content ? (message.role === "assistant" && markdown ? renderMarkdown(content) : "<pre>" + escapeHtml(content) + "</pre>") : "");
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

return { escapeHtml, renderJsonFields, renderMarkdown, renderMessage };
})();
