from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path
from typing import Any, Callable, TypedDict

from backend.ocr_helper import ocr_full_page, should_use_ocr_fallback
from pypdf import PdfReader

EMPTY_TEXT_PAGE = "EMPTY_TEXT_PAGE"
GIBBERISH_MIN_LEN = 300
GIBBERISH_MAX_WEIRD_RATIO = 0.25
GIBBERISH_PATTERN = "+DVWD"


class PageData(TypedDict):
    page: int
    text: str
    warnings: list[str]


OCRUsedCallback = Callable[[dict[str, Any]], None]


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return normalized.replace("\r\n", "\n").replace("\r", "\n")


def _extract_with_pypdf(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    page_texts: list[str] = []
    for page in reader.pages:
        page_texts.append(_normalize_text(page.extract_text() or ""))
    return page_texts


def _extract_with_pymupdf(path: Path) -> list[str] | None:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError:
        return None

    page_texts: list[str] = []
    with fitz.open(str(path)) as document:
        for page in document:
            page_texts.append(_normalize_text(page.get_text("text") or ""))
    return page_texts


def _is_gibberish(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < GIBBERISH_MIN_LEN:
        return True

    total_chars = max(1, len(text))
    allowed_chars = sum(1 for ch in text if ch.isalnum() or ch.isspace())
    weird_ratio = (total_chars - allowed_chars) / total_chars
    if weird_ratio > GIBBERISH_MAX_WEIRD_RATIO:
        return True

    return GIBBERISH_PATTERN in text


def _join_page_texts(page_texts: list[str]) -> str:
    return "\n\n".join(page_texts)


def _to_page_data(page_texts: list[str]) -> list[PageData]:
    pages: list[PageData] = []
    for page_no, text in enumerate(page_texts, start=1):
        warnings: list[str] = []
        if not text.strip():
            warnings.append(EMPTY_TEXT_PAGE)
        pages.append({"page": page_no, "text": text, "warnings": warnings})
    return pages


def extract_pages(pdf_path: str | Path, on_ocr_used: OCRUsedCallback | None = None) -> list[PageData]:
    path = Path(pdf_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"PDF path must be a file: {path}")

    primary_pages = _extract_with_pypdf(path)
    primary_text = _join_page_texts(primary_pages)

    chosen_pages = primary_pages
    if _is_gibberish(primary_text):
        fallback_pages = _extract_with_pymupdf(path)
        if fallback_pages is not None:
            fallback_text = _join_page_texts(fallback_pages)
            fallback_improves_quality = (not _is_gibberish(fallback_text)) or (
                len(fallback_text.strip()) > len(primary_text.strip())
            )
            if fallback_improves_quality:
                chosen_pages = fallback_pages

    chosen_text = _join_page_texts(chosen_pages)
    should_use_ocr, reason = should_use_ocr_fallback(chosen_text)
    if should_use_ocr:
        ocr_text = ocr_full_page(path, reason=reason, on_used=on_ocr_used)
        if ocr_text:
            if chosen_pages:
                chosen_pages = [ocr_text, *chosen_pages[1:]]
            else:
                chosen_pages = [ocr_text]

    return _to_page_data(chosen_pages)


def read_pdf(path: str | Path, on_ocr_used: OCRUsedCallback | None = None) -> str:
    pages = extract_pages(path, on_ocr_used=on_ocr_used)
    return _join_page_texts([page["text"] for page in pages])


def _print_preview(pages: list[PageData], max_pages: int = 2, max_chars: int = 300) -> None:
    for item in pages[:max_pages]:
        preview = item["text"][:max_chars]
        print(f"Page {item['page']} | warnings={item['warnings']}")
        print(preview)
        print("-" * 40)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract text page-by-page from a PDF file.")
    parser.add_argument("pdf_path", help="Path to a PDF file")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    extracted = extract_pages(args.pdf_path)
    _print_preview(extracted, max_pages=2, max_chars=300)
