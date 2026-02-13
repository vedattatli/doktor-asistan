from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from components.layout import render_page_header
from components.metrics_cards import render_metrics
from services.backend_bridge import get_audit_logs, get_docs_stats, ingest_and_build_index


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_FILE = REPO_ROOT / ".ingest.lock"
STALE_SECONDS = 1800


def _cleanup_stale_lock() -> None:
    if not LOCK_FILE.exists():
        return

    try:
        lock_age = time.time() - LOCK_FILE.stat().st_mtime
    except Exception as exc:
        st.warning(f"Lock dosyası kontrol edilemedi: {exc}")
        return

    if lock_age <= STALE_SECONDS:
        return

    try:
        LOCK_FILE.unlink(missing_ok=True)
        st.info("Stale lock temizlendi, tekrar deneyebilirsiniz.")
    except Exception as exc:
        st.warning(f"Stale lock silinemedi: {exc}")


def main() -> None:
    render_page_header("⚙️ Yönetim Paneli")

    _cleanup_stale_lock()
    lock_exists = LOCK_FILE.exists()

    st.subheader("Veritabanı Güncelleme")
    if lock_exists:
        st.info("Sistem şu anda index oluşturuyor.")

    if st.button("Veritabanını Güncelle", type="primary", disabled=lock_exists):
        if LOCK_FILE.exists():
            st.warning("Ingest zaten çalışıyor.")
            return

        try:
            LOCK_FILE.touch(exist_ok=False)
        except FileExistsError:
            st.warning("Ingest zaten çalışıyor.")
            return
        except Exception as exc:
            st.error(f"Lock dosyası oluşturulamadı: {exc}")
            return

        try:
            with st.spinner("Index oluşturuluyor, lütfen bekleyin..."):
                output = ingest_and_build_index()
                st.session_state["last_ingest_output"] = output
                st.success("İşlem tamamlandı.")
        except Exception as exc:
            st.session_state["last_ingest_output"] = str(exc)
            st.error(f"İşlem sırasında hata: {exc}")
        finally:
            try:
                LOCK_FILE.unlink(missing_ok=True)
            except Exception:
                pass

    if "last_ingest_output" in st.session_state:
        with st.expander("Son ingest çıktısı", expanded=False):
            st.code(st.session_state["last_ingest_output"])

    st.subheader("Doküman Metrikleri")
    render_metrics(get_docs_stats())

    st.subheader("Son 50 Audit Log")
    logs = get_audit_logs(limit=50)
    if logs:
        st.code("\n".join(logs))
    else:
        st.info("Audit log bulunamadı.")


if __name__ == "__main__":
    main()
