from __future__ import annotations

from typing import Any

import streamlit as st


def render_metrics(stats: dict[str, Any]) -> None:
    total_docs = int(stats.get("total_docs", 0))
    ocr_used_docs = int(stats.get("ocr_used_docs", 0))
    doc_type_counts = stats.get("doc_type_counts", {})

    col1, col2 = st.columns(2)
    col1.metric("Toplam Doküman", total_docs)
    col2.metric("OCR Kullanılan", ocr_used_docs)

    st.subheader("Doküman Türü Dağılımı")
    if isinstance(doc_type_counts, dict) and doc_type_counts:
        st.bar_chart(doc_type_counts)
    else:
        st.info("Gösterilecek doc_type verisi bulunamadı.")
