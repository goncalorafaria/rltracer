const element = (id) => document.getElementById(id);
let current = "";

function escapeHtml(value) {
  const node = document.createElement("span");
  node.textContent = value;
  return node.innerHTML;
}

async function load() {
  element("status").textContent = "Loading checkpoint runs...";
  const response = await fetch("/api/runs");
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  element("hint").textContent = data.children.length + " checkpoint runs with run_default traces";
  element("runs").innerHTML = data.children.map((child) =>
    '<button class="run-card" data-open="' + escapeHtml(child.path) + '">' +
      '<div><h2>' + escapeHtml(child.name) + '</h2><p>run_default - ' +
      child.step_count + " saved steps, latest step " + child.latest_step + '</p></div>' +
      '<span class="open-label">Open explorer</span></button>'
  ).join("") || '<p class="empty">No output directories with a runnable run_default trace were found.</p>';
  document.querySelectorAll("[data-open]").forEach((button) => button.onclick = () => openRun(button.dataset.open));
  element("status").textContent = data.children.length + " runs";
}

function openRun(path) {
  window.location.href = "/explorer?run=" + encodeURIComponent(path || ".");
}

load().catch((error) => {
  element("status").textContent = "Error";
  element("runs").innerHTML = '<p class="empty">' + escapeHtml(error.message) + "</p>";
});
