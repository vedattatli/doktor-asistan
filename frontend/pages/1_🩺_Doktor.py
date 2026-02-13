from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st

from components.layout import render_page_header
from services.backend_bridge import (
    ask_question,
    detect_query_mode,
    get_docs_stats,
    get_index_status,
    get_loaded_documents,
    ingest_and_build_index,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = REPO_ROOT / "data" / "pdfs"
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
MAX_FILES_PER_BATCH = 10
BOOT_CLEAN_FLAG = "_boot_clean_done"
RESET_NOTICE_KEY = "_doctor_reset_notice"
NEEDS_INDEX_PROMPT_KEY = "_doctor_needs_index_prompt"
MESSAGES_KEY = "messages"
LATEST_SOURCES_KEY = "_doctor_latest_sources"
LATEST_CONTEXT_KEY = "_doctor_latest_context"
LATEST_RAW_KEY = "_doctor_latest_raw"
PATIENT_CARD_KEY = "_doctor_patient_card"
PATIENT_CARD_SIG_KEY = "_doctor_patient_card_sig"
PATIENT_CARD_SOURCES_KEY = "_doctor_patient_card_sources"
AUTO_INDEX_KEY = "_doctor_auto_index_enabled"

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

TASK_SHORT_SUMMARY = (
    "From the provided CONTEXT only:\n"
    "Create a short clinical summary in Turkish (doctor-friendly).\n"
    "Sections:\n"
    "1) Hasta kimligi\n"
    "2) Basvuru nedeni / sikayet\n"
    "3) Onemli bulgular\n"
    "4) Plan / takip\n"
    "If any section is missing write: \"Belgede bilgi yok.\"\n"
    "Do not invent."
)
TASK_TIMELINE = (
    "From the provided CONTEXT only:\n"
    "Extract dates/times and create a chronological timeline.\n"
    "For each item:\n"
    "- Date/time\n"
    "- Event / test / report\n"
    "- 1 line evidence quote\n"
    "If no dates: \"Belgede bilgi yok.\""
)
TASK_LAB_ABNORMALITIES = (
    "From the provided CONTEXT only:\n"
    "List abnormal lab results as a table:\n"
    "Test | Value | Unit | Reference | Interpretation (only high/low/normal based on reference)\n"
    "If references are missing, do not interpret; just list.\n"
    "No guessing."
)
TASK_GRAPH_TEMPLATE = (
    "From the provided CONTEXT only:\n"
    "Extract all dated values for test: {test_name}.\n"
    "Output as a compact chronological table:\n"
    "Date | Test | Value | Unit | Reference\n"
    "If dates or values are missing write exactly: \"Belgede bilgi yok.\"\n"
    "Do not invent."
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


def _normalize_uploaded_files(uploaded: Any) -> list[st.runtime.uploaded_file_manager.UploadedFile]:
    if uploaded is None:
        return []
    if isinstance(uploaded, list):
        return uploaded
    return [uploaded]


def _clear_chat_and_upload_state() -> None:
    st.session_state[MESSAGES_KEY] = []
    for key in list(st.session_state.keys()):
        if key in {"uploaded_files_cache", "upload_form_key"} or key.startswith("uploader_"):
            st.session_state.pop(key, None)
    st.session_state["upload_form_key"] = str(uuid4())
    st.session_state[LATEST_SOURCES_KEY] = []
    st.session_state[LATEST_CONTEXT_KEY] = ""
    st.session_state[LATEST_RAW_KEY] = ""
    st.session_state[NEEDS_INDEX_PROMPT_KEY] = False
    st.session_state[PATIENT_CARD_KEY] = {
        "name": "Belgede bilgi yok.",
        "age": "Belgede bilgi yok.",
        "sex": "Belgede bilgi yok.",
        "protocol_no": "Belgede bilgi yok.",
        "report_date": "Belgede bilgi yok.",
    }
    st.session_state[PATIENT_CARD_SIG_KEY] = ""
    st.session_state[PATIENT_CARD_SOURCES_KEY] = []


def _ensure_upload_form_key() -> str:
    key = str(st.session_state.get("upload_form_key", "") or "")
    if not key:
        key = str(uuid4())
        st.session_state["upload_form_key"] = key
    return key


def _handle_boot_clean(auto_clean_enabled: bool) -> None:
    if st.session_state.get(BOOT_CLEAN_FLAG) is not None:
        return
    st.session_state[BOOT_CLEAN_FLAG] = True
    if not auto_clean_enabled:
        return

    _clear_chat_and_upload_state()
    st.session_state[RESET_NOTICE_KEY] = "Oturum başlangıcında chat ve upload seçimleri sıfırlandı."
    st.rerun()


def _render_reset_notice() -> None:
    notice = st.session_state.pop(RESET_NOTICE_KEY, None)
    if notice:
        st.success(str(notice))


def _init_state() -> None:
    if MESSAGES_KEY not in st.session_state:
        st.session_state[MESSAGES_KEY] = []
    if LATEST_SOURCES_KEY not in st.session_state:
        st.session_state[LATEST_SOURCES_KEY] = []
    if LATEST_CONTEXT_KEY not in st.session_state:
        st.session_state[LATEST_CONTEXT_KEY] = ""
    if LATEST_RAW_KEY not in st.session_state:
        st.session_state[LATEST_RAW_KEY] = ""
    if NEEDS_INDEX_PROMPT_KEY not in st.session_state:
        st.session_state[NEEDS_INDEX_PROMPT_KEY] = False
    if PATIENT_CARD_KEY not in st.session_state:
        st.session_state[PATIENT_CARD_KEY] = {
            "name": "Belgede bilgi yok.",
            "age": "Belgede bilgi yok.",
            "sex": "Belgede bilgi yok.",
            "protocol_no": "Belgede bilgi yok.",
            "report_date": "Belgede bilgi yok.",
        }
    if PATIENT_CARD_SIG_KEY not in st.session_state:
        st.session_state[PATIENT_CARD_SIG_KEY] = ""
    if PATIENT_CARD_SOURCES_KEY not in st.session_state:
        st.session_state[PATIENT_CARD_SOURCES_KEY] = []
    if AUTO_INDEX_KEY not in st.session_state:
        st.session_state[AUTO_INDEX_KEY] = True


def _extract_context_from_raw(raw: str) -> str:
    text = str(raw or "")
    if "Context Preview:" not in text:
        return ""
    section = text.split("Context Preview:", 1)[1]
    if "\n\nAnswer:" in section:
        section = section.split("\n\nAnswer:", 1)[0]
    elif "Answer:" in section:
        section = section.split("Answer:", 1)[0]
    return section.strip()


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

    context_text = _extract_context_from_raw(raw)
    for match in CONTEXT_SOURCE_RE.finditer(context_text):
        doc_name = match.group("doc_name").strip()
        chunk_id = match.group("chunk_id").strip()
        try:
            page = int(match.group("page").strip())
        except ValueError:
            page = 0

        snippet = " ".join(match.group("text").split()).strip()
        if len(snippet) > 280:
            snippet = snippet[:280].rstrip() + "..."

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


def _strip_sources_footer(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw
    marker = re.search(r"\n\s*\n?(?:Kaynaklar|Referans)\s*:?", raw, flags=re.IGNORECASE)
    if not marker:
        return raw
    return raw[: marker.start()].strip()


def _default_patient_card() -> dict[str, str]:
    return {
        "name": "Belgede bilgi yok.",
        "age": "Belgede bilgi yok.",
        "sex": "Belgede bilgi yok.",
        "protocol_no": "Belgede bilgi yok.",
        "report_date": "Belgede bilgi yok.",
    }


def _compute_patient_card_signature(docs: list[dict[str, Any]], index_status: dict[str, Any]) -> str:
    doc_names = sorted(str(doc.get("doc_name", "") or "") for doc in docs)
    docs_signature = "|".join(doc_names[:80]) + f"|count={len(doc_names)}"
    index_ready = bool(index_status.get("ready", False))
    index_meta = REPO_ROOT / "out" / "index" / "meta.jsonl"
    index_mtime = index_meta.stat().st_mtime if index_meta.exists() else 0.0
    return f"index_ready={index_ready}|index_mtime={index_mtime}|docs={docs_signature}"


def _parse_patient_card_answer(answer_text: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    text = str(answer_text or "").strip()
    if not text or text == "Belgede bilgi yok.":
        return _default_patient_card(), []

    payload: dict[str, Any] | None = None
    try:
        maybe = json.loads(text)
        if isinstance(maybe, dict):
            payload = maybe
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                maybe = json.loads(match.group(0))
                if isinstance(maybe, dict):
                    payload = maybe
            except Exception:
                payload = None

    if payload is None:
        return _default_patient_card(), []

    card = _default_patient_card()
    for key in ["name", "age", "sex", "protocol_no", "report_date"]:
        value = str(payload.get(key, "") or "").strip()
        card[key] = value if value else "Belgede bilgi yok."

    raw_sources = payload.get("sources", [])
    parsed_sources: list[dict[str, Any]] = []
    if isinstance(raw_sources, list):
        for src in raw_sources:
            if not isinstance(src, dict):
                continue
            parsed_sources.append(
                {
                    "doc_name": str(src.get("doc", "") or ""),
                    "page": int(src.get("page", 0) or 0),
                    "chunk_id": str(src.get("chunk_id", "") or ""),
                }
            )
    return card, parsed_sources


def _refresh_patient_card(docs: list[dict[str, Any]], index_status: dict[str, Any], force: bool = False) -> dict[str, str]:
    docs_loaded = len(docs) > 0
    index_ready = bool(index_status.get("ready", False))
    if not docs_loaded or not index_ready:
        card = _default_patient_card()
        st.session_state[PATIENT_CARD_KEY] = card
        st.session_state[PATIENT_CARD_SOURCES_KEY] = []
        st.session_state[PATIENT_CARD_SIG_KEY] = _compute_patient_card_signature(docs, index_status)
        return card

    signature = _compute_patient_card_signature(docs, index_status)
    cached_sig = str(st.session_state.get(PATIENT_CARD_SIG_KEY, "") or "")
    if not force and cached_sig == signature:
        return dict(st.session_state.get(PATIENT_CARD_KEY, _default_patient_card()))

    result = ask_question(
        "Hasta kartı bilgilerini çıkar: hastanın adı soyadı, yaş, cinsiyet, protokol no, rapor tarihi",
        auto_build_if_missing=False,
    )
    answer_text = str(result.get("answer", "") or "")
    card, sources = _parse_patient_card_answer(answer_text)
    st.session_state[PATIENT_CARD_KEY] = card
    st.session_state[PATIENT_CARD_SOURCES_KEY] = sources
    st.session_state[PATIENT_CARD_SIG_KEY] = signature
    return card


def _render_patient_card(docs: list[dict[str, Any]], index_status: dict[str, Any]) -> None:
    title_col, action_col = st.columns([6, 1])
    title_col.subheader("Hasta Kartı")
    refresh_clicked = action_col.button("🔄 Yenile", use_container_width=True, key="refresh_patient_card")

    docs_loaded = len(docs) > 0
    index_ready = bool(index_status.get("ready", False))

    if not docs_loaded:
        st.info("Hasta kartı için önce PDF yükleyin.")
    elif not index_ready and not bool(st.session_state.get(AUTO_INDEX_KEY, True)):
        st.info("Hasta kartı için önce index oluştur.")

    patient = _refresh_patient_card(docs, index_status, force=refresh_clicked)
    cols = st.columns(5)
    fields = [
        ("Ad Soyad", "name"),
        ("Yaş", "age"),
        ("Cinsiyet", "sex"),
        ("Protokol No", "protocol_no"),
        ("Rapor Tarihi", "report_date"),
    ]
    with st.container(border=True):
        for col, (label, key) in zip(cols, fields):
            col.caption(label)
            col.markdown(f"**{patient.get(key, 'Belgede bilgi yok.')}**")


def _build_index_with_feedback() -> None:
    try:
        with st.spinner("Index oluşturuluyor, lütfen bekleyin..."):
            output = ingest_and_build_index()
        st.session_state["last_ingest_output"] = output
        st.session_state[RESET_NOTICE_KEY] = "Index oluşturma tamamlandı."
        st.session_state[NEEDS_INDEX_PROMPT_KEY] = False
    except Exception as exc:
        st.session_state["last_ingest_output"] = str(exc)
        st.session_state[RESET_NOTICE_KEY] = f"Index oluşturma hatası: {exc}"
    st.rerun()


def _render_sidebar(index_status: dict[str, Any], docs: list[dict[str, Any]], docs_stats: dict[str, Any]) -> None:
    with st.sidebar:
        auto_index_enabled = st.checkbox("Otomatik index oluştur", value=True, key=AUTO_INDEX_KEY)

        st.subheader("PDF Yükleme")
        upload_form_key = _ensure_upload_form_key()
        uploader_key = f"uploader_{upload_form_key}"
        with st.form(key=f"upload_form_{upload_form_key}"):
            uploaded = st.file_uploader(
                "PDF dosyalarını seçin",
                type=["pdf"],
                accept_multiple_files=True,
                key=uploader_key,
            )
            submitted = st.form_submit_button("📤 Dosyaları Kaydet", use_container_width=True)

        if submitted:
            files = _normalize_uploaded_files(uploaded)
            if not files:
                st.info("Kaydetmek için en az bir PDF seçin.")
            else:
                batch_rejected = 0
                if len(files) > MAX_FILES_PER_BATCH:
                    batch_rejected = len(files) - MAX_FILES_PER_BATCH
                    files = files[:MAX_FILES_PER_BATCH]
                    st.info(
                        f"Bir seferde en fazla {MAX_FILES_PER_BATCH} dosya işlenir. "
                        f"{batch_rejected} dosya bu turda alınmadı."
                    )

                with st.spinner("Dosyalar kaydediliyor..."):
                    saved_count, rejected_count = _save_uploaded_files(files)
                total_rejected = rejected_count + batch_rejected

                if saved_count > 0:
                    st.success(f"{saved_count} PDF kaydedildi.")
                if total_rejected > 0:
                    st.warning(f"{total_rejected} dosya güvenlik kontrolünden geçmedi.")
                if saved_count > 0 and auto_index_enabled:
                    try:
                        with st.spinner("PDF işlendi, index otomatik oluşturuluyor..."):
                            output = ingest_and_build_index()
                        st.session_state["last_ingest_output"] = output
                        st.session_state[NEEDS_INDEX_PROMPT_KEY] = False
                        st.session_state[RESET_NOTICE_KEY] = "PDF kaydedildi ve index otomatik güncellendi."
                    except Exception as exc:
                        st.session_state["last_ingest_output"] = str(exc)
                        st.session_state[RESET_NOTICE_KEY] = f"Otomatik index oluşturma hatası: {exc}"
                elif saved_count > 0 and not auto_index_enabled:
                    st.session_state[RESET_NOTICE_KEY] = "PDF kaydedildi. Index için 'Index Oluştur' kullanın."

                st.session_state["upload_form_key"] = str(uuid4())
                st.rerun()

        st.subheader("Index Durumu")
        if bool(index_status.get("ready", False)):
            st.success("Index hazır.")
        else:
            st.info("Index henüz hazır değil.")
        st.caption(
            f"Doküman: {int(docs_stats.get('total_docs', 0))} | "
            f"OCR kullanılan: {int(docs_stats.get('ocr_used_docs', 0))}"
        )
        if st.button("🔧 Index Oluştur", type="primary", use_container_width=True):
            _build_index_with_feedback()

        st.subheader("Yüklü Dokümanlar")
        if docs:
            max_items = 20
            for doc in docs[:max_items]:
                doc_name = str(doc.get("doc_name", "-") or "-")
                doc_type = str(doc.get("doc_type", "unknown") or "unknown")
                pages = int(doc.get("total_pages", 0) or 0)
                st.caption(f"{doc_name} | {doc_type} | sayfa={pages}")
            if len(docs) > max_items:
                st.caption(f"... +{len(docs) - max_items} doküman")
        else:
            st.info("Henüz PDF yüklenmedi. PDF yükleyip ardından index oluşturabilirsiniz.")


def _append_message(role: str, content: str, raw: str = "", sources: list[dict[str, Any]] | None = None) -> None:
    content_raw = str(content or "")
    content_display = content_raw
    if role == "assistant":
        stripped = _strip_sources_footer(content_raw)
        if stripped:
            content_display = stripped
    st.session_state[MESSAGES_KEY].append(
        {
            "role": role,
            "content": content_display,
            "content_raw": content_raw,
            "raw": raw,
            "sources": sources or [],
        }
    )


def _process_question(
    display_text: str,
    query_text: str,
    task_instruction: str | None,
    docs_loaded: bool,
    index_ready: bool,
    auto_index_enabled: bool,
) -> None:
    _append_message("user", display_text)
    query_mode = "document_qa" if task_instruction else detect_query_mode(query_text)

    if query_mode == "document_qa" and not docs_loaded:
        answer = "Henüz PDF yüklenmedi. Soldan PDF yükleyip sonra index oluşturabilirsiniz."
        _append_message("assistant", answer)
        st.session_state[NEEDS_INDEX_PROMPT_KEY] = False
        return

    if query_mode == "document_qa" and not index_ready:
        if auto_index_enabled:
            try:
                with st.spinner("Index hazır değil, otomatik oluşturuluyor..."):
                    output = ingest_and_build_index()
                st.session_state["last_ingest_output"] = output
                index_ready = bool(get_index_status().get("ready", False))
                st.session_state[NEEDS_INDEX_PROMPT_KEY] = not index_ready
            except Exception as exc:
                st.session_state["last_ingest_output"] = str(exc)
                _append_message("assistant", "Önce index oluştur.")
                st.session_state[NEEDS_INDEX_PROMPT_KEY] = True
                return
            if not index_ready:
                _append_message("assistant", "Önce index oluştur.")
                st.session_state[NEEDS_INDEX_PROMPT_KEY] = True
                return
        else:
            answer = "Önce index oluştur."
            _append_message("assistant", answer)
            st.session_state[NEEDS_INDEX_PROMPT_KEY] = True
            return

    result = ask_question(
        query_text,
        task_instruction=task_instruction,
        auto_build_if_missing=False,
    )
    answer = str(result.get("answer", "Yanıt bulunamadı.") or "Yanıt bulunamadı.")
    raw = str(result.get("raw", "") or "")
    sources = _parse_sources_from_raw(raw)
    _append_message("assistant", answer, raw=raw, sources=sources)

    context = _extract_context_from_raw(raw)
    st.session_state[LATEST_SOURCES_KEY] = sources
    st.session_state[LATEST_RAW_KEY] = raw
    if context:
        st.session_state[LATEST_CONTEXT_KEY] = context

    st.session_state[NEEDS_INDEX_PROMPT_KEY] = bool(result.get("index_missing", False))


def _render_chat() -> None:
    st.subheader("Soru / Chat")

    for msg in st.session_state[MESSAGES_KEY]:
        role = str(msg.get("role", "assistant"))
        with st.chat_message(role):
            st.markdown(str(msg.get("content", "")))
            if role == "assistant":
                sources = msg.get("sources", [])
                if isinstance(sources, list) and sources:
                    with st.expander("Kaynağı Göster", expanded=False):
                        for source in sources[:8]:
                            doc_name = str(source.get("doc_name", "") or "-")
                            page = source.get("page", "-")
                            chunk_id = str(source.get("chunk_id", "") or "-")
                            snippet = str(source.get("snippet", "") or "").strip()
                            st.caption(f"doc={doc_name} | page={page} | chunk_id={chunk_id}")
                            if snippet:
                                st.write(snippet)

    graph_test = st.text_input("Grafik için test adı", placeholder="Örn: CRP", key="doctor_graph_test")
    action_cols = st.columns(4)

    pending: tuple[str, str, str | None] | None = None
    if action_cols[0].button("Kısa Özet", use_container_width=True):
        pending = ("Kısa Özet", "Kısa özet çıkar.", TASK_SHORT_SUMMARY)
    if action_cols[1].button("Kronoloji", use_container_width=True):
        pending = ("Kronoloji", "Kronoloji çıkar.", TASK_TIMELINE)
    if action_cols[2].button("Lab Anormallikleri", use_container_width=True):
        pending = ("Lab Anormallikleri", "Lab anormalliklerini çıkar.", TASK_LAB_ABNORMALITIES)
    if action_cols[3].button("Grafik", use_container_width=True):
        test_name = graph_test.strip()
        if not test_name:
            st.info("Grafik için önce test adı girin.")
        else:
            pending = (
                f"Grafik: {test_name.upper()}",
                f"{test_name} testinin zaman içi değişimini çıkar.",
                TASK_GRAPH_TEMPLATE.format(test_name=test_name),
            )

    question = st.chat_input("Sorunuzu yazın...")
    if question:
        pending = (question, question, None)

    docs_loaded = len(get_loaded_documents()) > 0
    index_ready = bool(get_index_status().get("ready", False))
    auto_index_enabled = bool(st.session_state.get(AUTO_INDEX_KEY, True))
    if pending:
        display_text, query_text, task_instruction = pending
        _process_question(
            display_text=display_text,
            query_text=query_text,
            task_instruction=task_instruction,
            docs_loaded=docs_loaded,
            index_ready=index_ready,
            auto_index_enabled=auto_index_enabled,
        )
        st.rerun()

    if st.session_state.get(NEEDS_INDEX_PROMPT_KEY) and not index_ready and not auto_index_enabled:
        st.warning("Önce index oluştur.")
        if st.button("Şimdi Index Oluştur", type="primary", key="build_index_inline"):
            _build_index_with_feedback()


def main() -> None:
    render_page_header("🩺 Doktor Asistanı")
    _init_state()
    _render_reset_notice()
    docs = get_loaded_documents()
    index_status = get_index_status()
    docs_stats = get_docs_stats()

    _render_sidebar(index_status=index_status, docs=docs, docs_stats=docs_stats)

    if not docs:
        st.info("Başlamak için soldan PDF yükleyin.")

    _render_patient_card(docs, index_status)
    st.markdown("---")
    _render_chat()


if __name__ == "__main__":
    main()
