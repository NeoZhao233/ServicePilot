const state = {
  userId: "USER-001",
  sessionId: null,
  documentId: null,
  eventCount: 0,
};

const messages = document.querySelector("#messages");
const trace = document.querySelector("#trace");
const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const submit = form.querySelector("button");

const labels = {
  logistics: "物流查询",
  refund: "退货退款",
  damaged_item: "商品破损",
  human_handoff: "人工转接",
  cancel: "取消操作",
  unknown: "待识别",
  document_qa: "PDF 问答",
};

function appendMessage(role, content, citations = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;
  article.appendChild(bubble);
  messages.appendChild(article);

  if (citations.length) {
    const row = document.createElement("div");
    row.className = "citations";
    citations.forEach((citation) => {
      const pill = document.createElement("span");
      pill.className = "citation";
      pill.textContent = `依据 · ${citation}`;
      row.appendChild(pill);
    });
    messages.appendChild(row);
  }
  messages.scrollTop = messages.scrollHeight;
  return article;
}

function renderEvents(events) {
  if (state.eventCount === 0) {
    trace.className = "";
    trace.replaceChildren();
  }
  events.forEach((event) => {
    state.eventCount += 1;
    const item = document.createElement("article");
    const isError = event.type === "tool" && event.detail.ok === false;
    item.className = `trace-item ${event.type}${isError ? " error" : ""}`;

    const heading = document.createElement("div");
    heading.className = "trace-title";
    const name = document.createElement("strong");
    name.textContent = event.name;
    const type = document.createElement("span");
    type.textContent = event.type;
    heading.append(name, type);

    const detail = document.createElement("pre");
    detail.className = "trace-detail";
    detail.textContent = JSON.stringify(event.detail, null, 2);
    item.append(heading, detail);
    trace.appendChild(item);
  });
  document.querySelector("#event-count").textContent = `${state.eventCount} steps`;
  trace.scrollTop = trace.scrollHeight;
}

function updateState(data) {
  document.querySelector("#intent").textContent = labels[data.intent] || data.intent || "—";
  document.querySelector("#order").textContent = data.order_id || "—";
  document.querySelector("#pending").textContent = data.pending_action || "—";
  document.querySelector("#session-label").textContent = data.session_id;
}

async function sendMessage(content) {
  const text = content.trim();
  if (!text || submit.disabled) return;
  appendMessage("user", text);
  input.value = "";
  input.style.height = "auto";
  submit.disabled = true;
  const pending = appendMessage("assistant", state.documentId ? "正在检索当前 PDF…" : "正在检查订单与售后政策…");
  pending.querySelector(".bubble").classList.add("typing");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: state.userId,
        message: text,
        session_id: state.sessionId,
        document_id: state.documentId,
      }),
    });
    const data = await response.json();
    pending.remove();
    if (!response.ok) throw new Error(data.detail || data.error || "请求失败");
    state.sessionId = data.session_id;
    if (data.trace_id) document.querySelector("#trace-viewer-link").href = `/traces?trace_id=${encodeURIComponent(data.trace_id)}`;
    const article = appendMessage("assistant", data.message, data.citations);
    (data.pdf_evidence || []).forEach((evidence) => {
      (evidence.images || []).forEach((url) => {
        const image = document.createElement("img");
        image.className = "pdf-evidence-image";
        image.alt = `PDF 第 ${evidence.page_number} 页图片证据`;
        image.src = url;
        article.appendChild(image);
      });
    });
    renderEvents(data.events);
    updateState(data);
  } catch (error) {
    pending.remove();
    appendMessage("assistant", `暂时无法处理：${error.message}`);
  } finally {
    submit.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(input.value);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 100)}px`;
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.dataset.prompt));
});

const historyDialog = document.querySelector("#history-dialog");
const historyList = document.querySelector("#history-list");

document.querySelector("#history-button").addEventListener("click", async () => {
  historyList.replaceChildren();
  const loading = document.createElement("div");
  loading.className = "history-empty";
  loading.textContent = "正在读取历史会话…";
  historyList.appendChild(loading);
  historyDialog.showModal();
  try {
    const response = await fetch(`/api/sessions?user_id=${encodeURIComponent(state.userId)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "读取失败");
    historyList.replaceChildren();
    if (!data.sessions.length) {
      loading.textContent = "还没有历史会话";
      historyList.appendChild(loading);
      return;
    }
    data.sessions.forEach((session) => {
      const button = document.createElement("button");
      button.className = "history-item";
      const meta = document.createElement("div");
      meta.className = "history-meta";
      const id = document.createElement("strong");
      id.textContent = session.order_id || "未关联订单";
      const intent = document.createElement("span");
      intent.textContent = labels[session.intent] || session.intent;
      meta.append(id, intent);
      const preview = document.createElement("p");
      preview.textContent = session.last_message;
      button.append(meta, preview);
      button.addEventListener("click", () => restoreSession(session.session_id));
      historyList.appendChild(button);
    });
  } catch (error) {
    loading.textContent = `历史会话读取失败：${error.message}`;
    historyList.replaceChildren(loading);
  }
});

document.querySelector("#history-close").addEventListener("click", () => historyDialog.close());

async function restoreSession(sessionId) {
  const response = await fetch(
    `/api/sessions/${encodeURIComponent(sessionId)}?user_id=${encodeURIComponent(state.userId)}`,
  );
  const data = await response.json();
  if (!response.ok) {
    historyList.textContent = data.error || "会话恢复失败";
    return;
  }
  state.sessionId = data.session_id;
  state.documentId = null;
  state.eventCount = 0;
  messages.replaceChildren();
  data.turns.forEach((turn) => appendMessage(turn.role, turn.content));
  renderEvents(data.events);
  updateState(data);
  await refreshPdfDocuments();
  historyDialog.close();
}

document.querySelector("#reset").addEventListener("click", () => {
  state.sessionId = null;
  state.documentId = null;
  state.eventCount = 0;
  messages.replaceChildren();
  appendMessage("assistant", "新会话已创建。请告诉我订单号和遇到的问题。");
  trace.className = "trace-empty";
  trace.replaceChildren();
  const icon = document.createElement("span");
  icon.className = "trace-icon";
  icon.textContent = "⌁";
  const text = document.createElement("p");
  text.textContent = "发送一条消息后，这里会展示意图路由、知识检索、规则判断和工具调用。";
  trace.append(icon, text);
  document.querySelector("#event-count").textContent = "0 steps";
  document.querySelector("#session-label").textContent = "新会话";
  document.querySelector("#intent").textContent = "—";
  document.querySelector("#order").textContent = "—";
  document.querySelector("#pending").textContent = "—";
  renderPdfDocuments([]);
});

const pdfFile = document.querySelector("#pdf-file");
const pdfSelect = document.querySelector("#pdf-select");
const pdfStatus = document.querySelector("#pdf-status");
const pdfUpload = document.querySelector("#pdf-upload");
const pdfDelete = document.querySelector("#pdf-delete");

function renderPdfDocuments(documents) {
  pdfSelect.replaceChildren();
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = "未选择文档";
  pdfSelect.appendChild(empty);
  documents.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.document_id;
    option.textContent = item.filename;
    pdfSelect.appendChild(option);
  });
  pdfSelect.value = state.documentId || "";
  pdfDelete.disabled = !state.documentId;
}

async function refreshPdfDocuments() {
  if (!state.sessionId) {
    renderPdfDocuments([]);
    return;
  }
  const query = new URLSearchParams({ user_id: state.userId, session_id: state.sessionId });
  const response = await fetch(`/api/pdf/documents?${query}`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "读取文档列表失败");
  renderPdfDocuments(data.documents);
}

pdfUpload.addEventListener("click", () => pdfFile.click());
pdfFile.addEventListener("change", async () => {
  const file = pdfFile.files[0];
  if (!file) return;
  pdfUpload.disabled = true;
  pdfStatus.textContent = "正在解析 PDF…";
  try {
    const form = new FormData();
    form.append("user_id", state.userId);
    if (state.sessionId) form.append("session_id", state.sessionId);
    form.append("file", file);
    const response = await fetch("/api/pdf/upload", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "上传失败");
    state.sessionId = data.session_id;
    state.documentId = data.document_id;
    await refreshPdfDocuments();
    pdfStatus.textContent = `已解析：${data.counts.text} 段文字、${data.counts.table} 张表格、${data.counts.image} 张图片`;
    document.querySelector("#session-label").textContent = state.sessionId;
    appendMessage("assistant", `已读取 ${data.filename}。请提问；回答会标注页码。`);
  } catch (error) {
    pdfStatus.textContent = error.message;
  } finally {
    pdfUpload.disabled = false;
    pdfFile.value = "";
  }
});

pdfSelect.addEventListener("change", () => {
  state.documentId = pdfSelect.value || null;
  pdfDelete.disabled = !state.documentId;
  pdfStatus.textContent = state.documentId ? "正在使用所选 PDF" : "已切换到普通售后对话";
});

pdfDelete.addEventListener("click", async () => {
  if (!state.documentId) return;
  const query = new URLSearchParams({ user_id: state.userId, session_id: state.sessionId });
  const response = await fetch(`/api/pdf/${state.documentId}?${query}`, { method: "DELETE" });
  const data = await response.json();
  if (!response.ok) {
    pdfStatus.textContent = data.detail || "删除失败";
    return;
  }
  state.documentId = null;
  await refreshPdfDocuments();
  pdfStatus.textContent = "文档已删除";
});
