from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from pydantic import BaseModel

from kg_search import KnowledgeGraphSearch


DEFAULT_MODEL = "qwen-plus"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

load_dotenv()

app = FastAPI(title="数据结构知识图谱问答助手")
search_engine = KnowledgeGraphSearch()


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    matches: list[dict[str, Any]]
    model: str
    direct_evidence: bool
    error: str | None = None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.post("/api/ask", response_model=AskResponse)
async def ask(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        return AskResponse(
            answer="请输入一个关于数据结构的问题。",
            matches=[],
            model=get_model_name(),
            direct_evidence=False,
        )

    matches = search_engine.search(question)
    direct_evidence = search_engine.has_direct_evidence(matches)
    match_payload = [match.to_dict() for match in matches]
    model = get_model_name()

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        return AskResponse(
            answer=missing_key_answer(matches, direct_evidence),
            matches=match_payload,
            model=model,
            direct_evidence=direct_evidence,
            error="DASHSCOPE_API_KEY is not configured.",
        )

    try:
        answer = await call_bailian(question, matches, direct_evidence, api_key, model)
    except ImportError:
        return AskResponse(
            answer="已完成知识图谱检索，但缺少 openai 依赖，无法调用百炼。请先安装 requirements.txt 中的依赖。",
            matches=match_payload,
            model=model,
            direct_evidence=direct_evidence,
            error="openai package is not installed.",
        )
    except Exception as exc:  # pragma: no cover - depends on remote API/network.
        return AskResponse(
            answer=f"已完成知识图谱检索，但调用百炼失败：{exc}",
            matches=match_payload,
            model=model,
            direct_evidence=direct_evidence,
            error=str(exc),
        )

    return AskResponse(
        answer=answer,
        matches=match_payload,
        model=model,
        direct_evidence=direct_evidence,
    )


async def call_bailian(
    question: str,
    matches: list,
    direct_evidence: bool,
    api_key: str,
    model: str,
) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=os.getenv("BAILIAN_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL,
    )
    graph_context = search_engine.build_context(question, matches)
    coverage_note = (
        "知识图谱已检索到直接相关证据。"
        if direct_evidence
        else "知识图谱没有检索到足够强的直接证据，需要先说明覆盖不足。"
    )
    completion = await client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个面向数据结构课程的知识图谱问答助手。"
                    "回答必须先基于给定知识图谱三元组和证据句。"
                    "允许使用通用数据结构知识补充，但补充内容必须用“【常识补充】”标明。"
                    "不要把补充知识伪装成图谱证据；如果图谱证据不足，先明确说明。"
                    "回答使用简洁中文。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题：{question}\n\n"
                    f"图谱覆盖情况：{coverage_note}\n\n"
                    f"检索到的知识图谱内容：\n{graph_context}\n\n"
                    "请按以下结构回答：\n"
                    "1. 【图谱依据】概括图谱中直接支持答案的内容。\n"
                    "2. 【回答】直接回答用户问题。\n"
                    "3. 【常识补充】如需补充图谱之外的通用知识，在这里写；不需要则写“无”。"
                ),
            },
        ],
    )
    return completion.choices[0].message.content or ""


def get_model_name() -> str:
    return os.getenv("BAILIAN_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def missing_key_answer(matches: list, direct_evidence: bool) -> str:
    if matches:
        prefix = (
            "已完成知识图谱检索，并找到了直接相关证据。"
            if direct_evidence
            else "已完成知识图谱检索，但未找到足够强的直接证据。"
        )
    else:
        prefix = "知识图谱暂未检索到匹配关系。"
    return (
        f"{prefix}\n\n"
        "当前未配置 DASHSCOPE_API_KEY，所以没有调用阿里云百炼生成最终回答。"
        "请设置环境变量后重新提问。"
    )


INDEX_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>数据结构知识图谱问答助手</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f5;
      --panel: #ffffff;
      --ink: #18201c;
      --muted: #66736d;
      --line: #dbe2dd;
      --green: #246b4f;
      --blue: #315f9c;
      --amber: #9a6218;
      --soft-green: #e7f2ec;
      --soft-blue: #e8eef8;
      --soft-amber: #fff4df;
      font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
    }

    .shell {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(280px, 380px);
      min-height: 100vh;
    }

    main {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      min-width: 0;
      border-right: 1px solid var(--line);
    }

    header {
      padding: 22px 28px 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.76);
      backdrop-filter: blur(12px);
    }

    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.3;
      letter-spacing: 0;
    }

    .subtle {
      margin-top: 6px;
      color: var(--muted);
      font-size: 14px;
    }

    .chat {
      overflow-y: auto;
      padding: 26px 28px 18px;
    }

    .message {
      width: min(760px, 100%);
      margin-bottom: 16px;
      padding: 15px 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      white-space: pre-wrap;
      line-height: 1.65;
      box-shadow: 0 10px 28px rgba(24, 32, 28, 0.05);
    }

    .message.user {
      margin-left: auto;
      border-color: #bfd6c8;
      background: var(--soft-green);
    }

    .message.assistant {
      margin-right: auto;
    }

    .composer {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      padding: 16px 28px 22px;
      border-top: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.86);
    }

    textarea {
      width: 100%;
      min-height: 52px;
      max-height: 150px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      font: inherit;
      line-height: 1.5;
      color: var(--ink);
      background: #fff;
      outline: none;
    }

    textarea:focus {
      border-color: var(--green);
      box-shadow: 0 0 0 3px rgba(36, 107, 79, 0.14);
    }

    button {
      min-width: 92px;
      height: 52px;
      border: 0;
      border-radius: 8px;
      background: var(--green);
      color: #fff;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }

    button:disabled {
      cursor: wait;
      opacity: 0.62;
    }

    aside {
      min-width: 0;
      padding: 22px;
      overflow-y: auto;
      background: #fbfcfa;
    }

    .panel-title {
      margin: 0 0 14px;
      font-size: 17px;
      line-height: 1.35;
    }

    .match {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 13px;
      margin-bottom: 12px;
    }

    .triple {
      font-weight: 700;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }

    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 10px 0;
    }

    .pill {
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
      color: var(--blue);
      background: var(--soft-blue);
    }

    .pill.conf {
      color: var(--amber);
      background: var(--soft-amber);
    }

    .evidence {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }

    .empty {
      color: var(--muted);
      line-height: 1.7;
    }

    @media (max-width: 860px) {
      .shell {
        grid-template-columns: 1fr;
      }

      main {
        min-height: 68vh;
        border-right: 0;
      }

      aside {
        border-top: 1px solid var(--line);
      }
    }

    @media (max-width: 560px) {
      header,
      .chat,
      .composer,
      aside {
        padding-left: 16px;
        padding-right: 16px;
      }

      .composer {
        grid-template-columns: 1fr;
      }

      button {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <main>
      <header>
        <h1>数据结构知识图谱问答助手</h1>
        <div class="subtle" id="status">模型：qwen-plus</div>
      </header>

      <section class="chat" id="chat" aria-live="polite">
        <div class="message assistant">可以问我：数据结构分为哪些类型？链表支持哪些操作？Prim算法用于什么？</div>
      </section>

      <form class="composer" id="form">
        <textarea id="question" name="question" placeholder="输入问题" autocomplete="off"></textarea>
        <button id="submit" type="submit">发送</button>
      </form>
    </main>

    <aside>
      <h2 class="panel-title">检索命中</h2>
      <div id="matches" class="empty">等待提问。</div>
    </aside>
  </div>

  <script>
    const form = document.querySelector("#form");
    const questionInput = document.querySelector("#question");
    const submitButton = document.querySelector("#submit");
    const chat = document.querySelector("#chat");
    const matches = document.querySelector("#matches");
    const status = document.querySelector("#status");

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const question = questionInput.value.trim();
      if (!question) {
        questionInput.focus();
        return;
      }

      appendMessage("user", question);
      questionInput.value = "";
      submitButton.disabled = true;
      submitButton.textContent = "生成中";

      try {
        const response = await fetch("/api/ask", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({question}),
        });
        const data = await response.json();
        status.textContent = `模型：${data.model}${data.error ? " · 需要配置" : ""}`;
        appendMessage("assistant", data.answer || "没有返回回答。");
        renderMatches(data.matches || []);
      } catch (error) {
        appendMessage("assistant", `请求失败：${error}`);
      } finally {
        submitButton.disabled = false;
        submitButton.textContent = "发送";
        questionInput.focus();
      }
    });

    questionInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    function appendMessage(role, text) {
      const node = document.createElement("div");
      node.className = `message ${role}`;
      node.textContent = text;
      chat.appendChild(node);
      chat.scrollTop = chat.scrollHeight;
    }

    function renderMatches(items) {
      matches.innerHTML = "";
      if (!items.length) {
        matches.className = "empty";
        matches.textContent = "没有检索到匹配关系。";
        return;
      }
      matches.className = "";
      for (const item of items) {
        const block = document.createElement("section");
        block.className = "match";
        const evidence = (item.evidences || []).map((entry) => entry.text).filter(Boolean).slice(0, 2).join(" / ");
        block.innerHTML = `
          <div class="triple">${escapeHtml(item.subject)} → ${escapeHtml(item.predicate)} → ${escapeHtml(item.object)}</div>
          <div class="meta">
            <span class="pill">${escapeHtml(item.subject_category)} / ${escapeHtml(item.object_category)}</span>
            <span class="pill conf">置信度 ${Number(item.confidence || 0).toFixed(2)}</span>
            <span class="pill">检索分 ${Number(item.score || 0).toFixed(2)}</span>
          </div>
          <p class="evidence">${escapeHtml(evidence || "无证据句")}</p>
        `;
        matches.appendChild(block);
      }
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }
  </script>
</body>
</html>
"""
