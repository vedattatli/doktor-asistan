from __future__ import annotations

import re
from typing import Any

import streamlit as st

from services.backend_bridge import ask_question


TOPK_SOURCE_RE = re.compile(
    r"^\s*-\s*\d+\.\s*score=(?P<score>[^|]+)\|\s*doc_name=(?P<doc_name>[^|]+)\|\s*"
    r"page=(?P<page>\d+)\s*\|\s*chunk_id=(?P<chunk_id>[^\s|]+)\s*$",
    flags=re.MULTILINE,
)
CONTEXT_SOURCE_RE = re.compile(
    r"\[SOURCE\s+doc=(?P<doc_name>.+?)\s+page=(?P<page>\d+)\s+chunk_id=(?P<chunk_id>[^\]]+)\]\s*"
    r"(?P<text>.*?)(?=(?:\n\[SOURCE\s+doc=)|\Z)",
    flags=re.DOTALL,
)
QUICK_ACTIONS: list[dict[str, str]] = [
    {
        "id": "clinical_summary",
        "label": "Klinik Özet",
        "query": "Klinik özet hazırla.",
        "task_instruction": (
            "From the provided CONTEXT only:\n"
            "Create a clinical summary in Turkish with these sections:\n"
            "1) Hasta kimliği (ad/yaş/cinsiyet varsa)\n"
            "2) Başvuru nedeni / şikayet (varsa)\n"
            "3) Özgeçmiş / ilaçlar / alerji (varsa)\n"
            "4) Önemli bulgular (lab/radyoloji)\n"
            "5) Plan / öneri / takip (belgede yazıyorsa)\n\n"
            "If any section is not present, write \"Belgede bilgi yok.\"\n"
            "Do not invent."
        ),
    },
    {
        "id": "timeline",
        "label": "Kronoloji",
        "query": "Kronoloji çıkar.",
        "task_instruction": (
            "From the provided CONTEXT only:\n"
            "Extract all dates/times and present a timeline in chronological order.\n"
            "For each item:\n"
            "- Date/time\n"
            "- Event / test / report\n"
            "- 1 line evidence quote\n"
            "If no dates: \"Belgede bilgi yok.\""
        ),
    },
    {
        "id": "lab_abnormalities",
        "label": "Lab Anormallikleri",
        "query": "Lab anormalliklerini çıkar.",
        "task_instruction": (
            "From the provided CONTEXT only:\n"
            "List abnormal lab results as a table:\n"
            "Test | Value | Unit | Reference | Interpretation (only \"high/low/normal\" based on reference)\n"
            "If references are missing, do not interpret; just list.\n"
            "No guessing."
        ),
    },
]


def _init_messages() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def _parse_sources_from_raw(raw: str) -> list[dict[str, Any]]:
    if not raw:
        return []

    sources: list[dict[str, Any]] = []
    source_by_chunk: dict[str, dict[str, Any]] = {}

    for match in TOPK_SOURCE_RE.finditer(raw):
        score_raw = match.group("score").strip()
        doc_name = match.group("doc_name").strip()
        page_raw = match.group("page").strip()
        chunk_id = match.group("chunk_id").strip()
        try:
            score = float(score_raw)
        except ValueError:
            score = None
        try:
            page = int(page_raw)
        except ValueError:
            page = 0

        item = {
            "score": score,
            "doc_name": doc_name,
            "page": page,
            "chunk_id": chunk_id,
            "snippet": "",
        }
        sources.append(item)
        source_by_chunk[chunk_id] = item

    context_text = ""
    if "Context Preview:" in raw:
        context_text = raw.split("Context Preview:", 1)[1]
        if "\n\nAnswer:" in context_text:
            context_text = context_text.split("\n\nAnswer:", 1)[0]
        elif "Answer:" in context_text:
            context_text = context_text.split("Answer:", 1)[0]

    for match in CONTEXT_SOURCE_RE.finditer(context_text):
        doc_name = match.group("doc_name").strip()
        chunk_id = match.group("chunk_id").strip()
        try:
            page = int(match.group("page").strip())
        except ValueError:
            page = 0

        snippet = " ".join(match.group("text").split()).strip()
        if len(snippet) > 240:
            snippet = snippet[:240].rstrip() + "..."

        target = source_by_chunk.get(chunk_id)
        if target is None:
            target = {
                "score": None,
                "doc_name": doc_name,
                "page": page,
                "chunk_id": chunk_id,
                "snippet": "",
            }
            sources.append(target)
            source_by_chunk[chunk_id] = target

        if snippet and not target.get("snippet"):
            target["snippet"] = snippet

    return sources


def _render_sources(sources: list[dict[str, Any]]) -> None:
    with st.expander("📚 Referans Dokümanlar & Kanıtlar", expanded=False):
        if not sources:
            st.write("Kaynak bulunamadı.")
            return

        for index, source in enumerate(sources, start=1):
            doc_name = str(source.get("doc_name", "") or "-")
            page = source.get("page", "-")
            chunk_id = str(source.get("chunk_id", "") or "-")
            score = source.get("score")
            if isinstance(score, (int, float)):
                score_text = f"{float(score):.4f}"
            else:
                score_text = "-"
            st.markdown(f"📄 **{doc_name}** — Sayfa: {page} (Skor: {score_text})")
            st.caption(f"chunk_id: `{chunk_id}`")
            snippet = str(source.get("snippet", "") or "").strip()
            if snippet:
                st.caption(snippet)
            if index < len(sources):
                st.divider()


def _render_quick_actions() -> dict[str, str] | None:
    st.caption("Hazır Görevler")
    cols = st.columns(len(QUICK_ACTIONS))
    for idx, action in enumerate(QUICK_ACTIONS):
        key = f"quick_action_{action['id']}"
        if cols[idx].button(action["label"], use_container_width=True, key=key):
            return action
    return None


def render_chat() -> None:
    _init_messages()

    for msg in st.session_state["messages"]:
        role = str(msg.get("role", "assistant"))
        with st.chat_message(role):
            st.markdown(str(msg.get("content", "")))
            if role != "assistant":
                continue

            raw = str(msg.get("raw", "") or "")
            stored_sources = msg.get("sources")
            if isinstance(stored_sources, list):
                sources = stored_sources
            else:
                sources = _parse_sources_from_raw(raw)

            _render_sources(sources)
            if raw:
                with st.expander("Gelişmiş Debug (Raw)"):
                    st.code(raw)

    selected_action = _render_quick_actions()
    question = st.chat_input("Sorunuzu yazın...")
    if not selected_action and not question:
        return

    user_text = question or ""
    query_text = question or ""
    task_instruction = None
    if selected_action:
        user_text = str(selected_action.get("label", "Hazır görev"))
        query_text = str(selected_action.get("query", "")).strip() or user_text
        task_instruction = str(selected_action.get("task_instruction", "")).strip()

    st.session_state["messages"].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Cevap hazırlanıyor..."):
            result = ask_question(query_text, task_instruction=task_instruction)
        answer = result.get("answer", "Yanıt bulunamadı.")
        raw = result.get("raw", "")
        sources = _parse_sources_from_raw(raw)
        st.markdown(answer)
        _render_sources(sources)
        with st.expander("Gelişmiş Debug (Raw)"):
            st.code(raw)

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "sources": sources, "raw": raw}
    )
