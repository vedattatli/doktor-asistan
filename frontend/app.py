from __future__ import annotations

import streamlit as st

from components.layout import render_page_header


st.set_page_config(
    page_title="Akıllı Doktor Asistanı",
    layout="wide",
)


def _render_sidebar_navigation() -> None:
    st.sidebar.title("Navigasyon")
    if hasattr(st.sidebar, "page_link"):
        st.sidebar.page_link("pages/1_🩺_Doktor.py", label="Doktor", icon="🩺")
        st.sidebar.page_link("pages/2_⚙️_Admin.py", label="Admin", icon="⚙️")
    else:
        st.sidebar.markdown("- Doktor\n- Admin")


def main() -> None:
    _render_sidebar_navigation()
    render_page_header(
        "Akıllı Doktor Asistanı",
        "Sol menüden Doktor veya Admin sayfasını seçin.",
    )


if __name__ == "__main__":
    main()
