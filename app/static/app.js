const TENANT_ID_PATTERN = /^[a-zA-Z0-9_-]{1,64}$/;
const ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt", ".xlsx"];

const thread = document.getElementById("thread");
const composer = document.getElementById("composer");
const questionInput = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");
const tenantInput = document.getElementById("tenant-input");
const attachBtn = document.getElementById("attach-btn");
const fileInput = document.getElementById("file-input");

function getStoredTenantId() {
  const stored = localStorage.getItem("rag_tenant_id");
  if (stored && TENANT_ID_PATTERN.test(stored)) return stored;
  const generated = `guest-${crypto.randomUUID()}`;
  localStorage.setItem("rag_tenant_id", generated);
  return generated;
}

tenantInput.value = getStoredTenantId();

tenantInput.addEventListener("change", () => {
  const value = tenantInput.value.trim();
  if (TENANT_ID_PATTERN.test(value)) {
    localStorage.setItem("rag_tenant_id", value);
  } else {
    tenantInput.value = getStoredTenantId();
    appendMessage("error", "Tenant must match ^[a-zA-Z0-9_-]{1,64}$ — reverted to the last valid value.");
  }
});

function currentTenantId() {
  return tenantInput.value.trim();
}

function scrollToBottom() {
  thread.scrollTop = thread.scrollHeight;
}

// role: "user" | "assistant" | "system" | "error" | "pending"
function appendMessage(role, text) {
  const bubble = document.createElement("div");
  bubble.className = `msg ${role}`;
  bubble.textContent = text;
  thread.appendChild(bubble);
  scrollToBottom();
  return bubble;
}

function renderSources(container, sources) {
  if (!sources || sources.length === 0) return;
  const details = document.createElement("details");
  details.className = "sources";
  const summary = document.createElement("summary");
  summary.textContent = `${sources.length} source${sources.length > 1 ? "s" : ""}`;
  details.appendChild(summary);

  for (const source of sources) {
    const item = document.createElement("div");
    item.className = "source-item";

    const meta = document.createElement("div");
    meta.className = "meta";
    const left = document.createElement("span");
    left.textContent = `${source.filename} — ${source.locator}`;
    const right = document.createElement("span");
    right.textContent = `score ${source.score.toFixed(3)}`;
    meta.appendChild(left);
    meta.appendChild(right);
    item.appendChild(meta);
    details.appendChild(item);
  }
  container.appendChild(details);
}

function renderAnswer(bubble, response) {
  bubble.classList.remove("pending");
  bubble.textContent = response.answer;
  if (!response.grounded) {
    bubble.classList.add("refused");
  } else {
    renderSources(bubble, response.sources);
  }
}

async function sendQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;

  const tenantId = currentTenantId();
  if (!TENANT_ID_PATTERN.test(tenantId)) {
    appendMessage("error", "Fix the tenant field before sending a question.");
    return;
  }

  appendMessage("user", question);
  questionInput.value = "";
  setBusy(true);

  const pending = appendMessage("assistant pending", "Thinking…");

  try {
    const res = await fetch(`/tenants/${encodeURIComponent(tenantId)}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const body = await res.json().catch(() => null);
    if (!res.ok) {
      pending.remove();
      appendMessage("error", (body && body.detail) || `Request failed (${res.status}).`);
      return;
    }
    renderAnswer(pending, body);
  } catch (err) {
    pending.remove();
    appendMessage("error", "Couldn't reach the server. If it's been idle, it may still be waking up — try again shortly.");
  } finally {
    setBusy(false);
  }
}

async function uploadFile(file) {
  const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!ALLOWED_EXTENSIONS.includes(extension)) {
    appendMessage("error", `Unsupported file type "${extension}". Allowed: ${ALLOWED_EXTENSIONS.join(", ")}`);
    return;
  }

  const tenantId = currentTenantId();
  if (!TENANT_ID_PATTERN.test(tenantId)) {
    appendMessage("error", "Fix the tenant field before uploading a document.");
    return;
  }

  setBusy(true);
  const pending = appendMessage("system pending", `Uploading ${file.name}…`);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`/tenants/${encodeURIComponent(tenantId)}/documents`, {
      method: "POST",
      body: formData,
    });
    const body = await res.json().catch(() => null);
    if (!res.ok) {
      pending.remove();
      appendMessage("error", (body && body.detail) || `Upload failed (${res.status}).`);
      return;
    }
    pending.classList.remove("pending");
    pending.textContent = `📄 ${body.filename} — ${body.chunks_indexed} chunk${body.chunks_indexed === 1 ? "" : "s"} indexed`;
  } catch (err) {
    pending.remove();
    appendMessage("error", "Couldn't reach the server. If it's been idle, it may still be waking up — try again shortly.");
  } finally {
    setBusy(false);
  }
}

function setBusy(busy) {
  sendBtn.disabled = busy;
  questionInput.disabled = busy;
  attachBtn.disabled = busy;
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  sendQuestion();
});

attachBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  fileInput.value = "";
  if (file) uploadFile(file);
});
