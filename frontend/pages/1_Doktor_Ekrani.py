from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st

from components.layout import render_page_header
from services.backend_bridge import ask_question


REPO_ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = REPO_ROOT / "data" / "pdfs"
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024

TOPK_SOURCE_RE = re.compile(
    r"^\s*-\s*\d+\.\s*score=(?P<score>-?\d+(?:\.\d+)?)\s*\|\s*doc_name=(?P<doc_name>[^|]+)\|\s*"
    r"page=(?P<page>\d+)\s*\|\s*chunk_id=(?P<chunk_id>[^\s|]+)\s*$",
    flags=re.MULTILINE,
)
CONTEXT_SOURCE_RE = re.compile(
    r"\[SOURCE\s+doc=(?P<doc_name>.+?)\s+page=(?P<page>\d+)\s+chunk_id=(?P<chunk_id>[^\]]+)\]\s*"
    r"(?P<text>.*?)(?=(?:\n\[SOURCE\s+doc=)|\Z)",
    flags=re.DOTALL,
)


def _save_uploaded_files(
    uploaded_files: list[st.runtime.uploaded_file_manager.UploadedFile],
) -> tuple[int, int]:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    rejected = 0
    for uploaded in uploaded_files:
        raw_name = str(getattr(uploaded, "name", "") or "")
        base_name = Path(raw_name).name
        base_name = os.path.basename(base_name).replace(" ", "_")

        if not base_name or not base_name.lower().endswith(".pdf"):
            st.error(f"Geçersiz dosya türü: {raw_name}")
            rejected += 1
            continue

        size_bytes = getattr(uploaded, "size", None)
        if size_bytes is None:
            size_bytes = len(uploaded.getbuffer())
        if int(size_bytes) > MAX_UPLOAD_SIZE_BYTES:
            st.error(f"Dosya çok büyük (>{MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB): {base_name}")
            rejected += 1
            continue

        target_path = PDF_DIR / f"{uuid4()}_{base_name}"
        target_path.write_bytes(uploaded.getbuffer())
        saved += 1
    return saved, rejected


def _init_messages() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def _parse_sources_from_raw(raw: str) -> list[dict[str, Any]]:
    if not raw:
        return []

    sources: list[dict[str, Any]] = []
    by_chunk_id: dict[str, dict[str, Any]] = {}

    for match in TOPK_SOURCE_RE.finditer(raw):
        doc_name = match.group("doc_name").strip()
        chunk_id = match.group("chunk_id").strip()
        try:
            page = int(match.group("page"))
        except ValueError:
            page = 0
        try:
            score = float(match.group("score"))
        except ValueError:
            score = 0.0

        item = {
            "doc_name": doc_name,
            "page": page,
            "chunk_id": chunk_id,
            "score": score,
            "snippet": "",
        }
        sources.append(item)
        by_chunk_id[chunk_id] = item

    context_section = ""
    if "Context Preview:" in raw:
        context_section = raw.split("Context Preview:", 1)[1]
        if "\n\nAnswer:" in context_section:
            context_section = context_section.split("\n\nAnswer:", 1)[0]
        elif "Answer:" in context_section:
            context_section = context_section.split("Answer:", 1)[0]

    for match in CONTEXT_SOURCE_RE.finditer(context_section):
        chunk_id = match.group("chunk_id").strip()
        snippet = " ".join(match.group("text").split()).strip()
        if len(snippet) > 240:
            snippet = snippet[:240].rstrip() + "..."

        target = by_chunk_id.get(chunk_id)
        if target and snippet and not target.get("snippet"):
            target["snippet"] = snippet

    return sources


def _render_sources_for_assistant(sources: list[dict[str, Any]]) -> None:
    with st.expander("📚 Referans Dokümanlar & Kanıtlar", expanded=False):
        if not sources:
            st.write("Kaynak bulunamadı.")
            return

        for source in sources:
            doc_name = str(source.get("doc_name", "-"))
            page = source.get("page", "-")
            score = float(source.get("score", 0.0))
            chunk_id = str(source.get("chunk_id", "-"))
            st.markdown(f"📄 **{doc_name}** - Sayfa: {page} (Skor: {score:.4f})")
            st.caption(f"chunk_id: {chunk_id}")
            snippet = str(source.get("snippet", "")).strip()
            if snippet:
                st.caption(snippet)
            st.markdown("---")


def render_chat() -> None:
    _init_messages()

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_sources_for_assistant(msg.get("sources", []))
                raw = msg.get("raw")
                if raw:
                    with st.expander("Gelişmiş Debug (Raw)"):
                        st.code(raw)

    question = st.chat_input("Sorunuzu yazın...")
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Cevap hazırlanıyor..."):
            result = ask_question(question)

        answer = result.get("answer", "Yanıt bulunamadı.")
        raw = result.get("raw", "")
        sources = _parse_sources_from_raw(raw)

        st.markdown(answer)
        _render_sources_for_assistant(sources)
        with st.expander("Gelişmiş Debug (Raw)"):
            st.code(raw)

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "raw": raw,
            "sources": sources,
        }
    )


def main() -> None:
    render_page_header("🩺 Doktor Asistanı")

    with st.sidebar:
        st.subheader("PDF Yükleme")
        uploaded_files = st.file_uploader(
            "PDF dosyalarını seçin",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            saved_count, rejected_count = _save_uploaded_files(uploaded_files)
            if saved_count > 0:
                st.success(
                    f"{saved_count} dosya kaydedildi. "
                    "Dosyalar kaydedildi. İşlenmesi için Admin panelinden ingest çalıştırın."
                )
            if rejected_count > 0:
                st.warning(f"{rejected_count} dosya güvenlik kontrolünden geçmedi.")
        else:
            st.info("İsterseniz bir veya daha fazla PDF yükleyin.")

    render_chat()


if __name__ == "__main__":
    main()
