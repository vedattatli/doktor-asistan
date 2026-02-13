from __future__ import annotations

import streamlit as st


def render_page_header(title: str, subtitle: str | None = None) -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)
