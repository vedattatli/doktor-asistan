from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Callable

PRINTABLE_GIBBERISH_THRESHOLD = 0.18
DIAGNOSIS_BLOCK_MIN_CHARS = 120
DIAGNOSIS_BLOCK_WEIRD_RATIO_THRESHOLD = 0.20
DIAGNOSIS_BLOCK_MAX_LETTER_RATIO = 0.65
DIAGNOSIS_BLOCK_MAX_LINES = 13  # hit line + next 12 lines
PATHOLOGY_HEADER_RE = re.compile(r"(PATOLOJ[Iİ]|PATHOLOGY)\s*(RAPORU|REPORT)?", re.IGNORECASE)
DIAGNOSIS_RE = re.compile(r"(TANI|ICD|SONU[ÇC]|DIAGNOSIS)", re.IGNORECASE)
TANI_RE = re.compile(r"TAN[Iİ]", re.IGNORECASE)
ALLOWED_PUNCT = ".,;:!?()[]{}%+-/\\'\""
OCRUsedCallback = Callable[[dict[str, Any]], None]


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return normalized.replace("\r\n", "\n").replace("\r", "\n")


def _weird_printable_ratio(text: str) -> float:
    if not text:
        return 1.0
    weird_chars = sum(
        1
        for ch in text
        if not (ch.isalnum() or ch.isspace() or ch in ALLOWED_PUNCT)
    )
    return weird_chars / max(1, len(text))


def is_gibberish_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return _weird_printable_ratio(stripped) > PRINTABLE_GIBBERISH_THRESHOLD


def _diagnosis_block_is_gibberish(text: str) -> bool:
    lines = text.splitlines()
    if not lines:
        return False

    hit_index = next((idx for idx, line in enumerate(lines) if TANI_RE.search(line)), None)
    if hit_index is None:
        hit_index = next((idx for idx, line in enumerate(lines) if DIAGNOSIS_RE.search(line)), None)
    if hit_index is None:
        return False

    block_lines = lines[hit_index : hit_index + DIAGNOSIS_BLOCK_MAX_LINES]
    block = "\n".join(block_lines).strip()
    if len(block) < DIAGNOSIS_BLOCK_MIN_CHARS:
        return False

    # In encoded diagnosis blocks, digit/punctuation-heavy noise is common; treat them as noise for this check.
    block_for_ratio = "".join("¤" if (ch.isdigit() or ch in ALLOWED_PUNCT) else ch for ch in block)
    weird_ratio = _weird_printable_ratio(block_for_ratio)
    letter_ratio = sum(1 for ch in block if ch.isalpha()) / max(1, len(block))
    return weird_ratio >= DIAGNOSIS_BLOCK_WEIRD_RATIO_THRESHOLD and letter_ratio <= DIAGNOSIS_BLOCK_MAX_LETTER_RATIO


def should_use_ocr_fallback(text: str) -> tuple[bool, str]:
    normalized = _normalize_text(text)
    has_pathology_header = bool(PATHOLOGY_HEADER_RE.search(normalized))
    has_diagnosis_keyword = bool(DIAGNOSIS_RE.search(normalized))

    if is_gibberish_text(normalized):
        return True, "gibberish"

    if has_pathology_header and has_diagnosis_keyword and _diagnosis_block_is_gibberish(normalized):
        return True, "diagnosis_block_gibberish"

    if has_pathology_header and not has_diagnosis_keyword:
        return True, "pathology_no_diag_keyword"
    return False, ""


def ocr_full_page(
    pdf_path: str | Path,
    lang: str = "tur",
    dpi: int = 300,
    reason: str = "",
    on_used: OCRUsedCallback | None = None,
) -> str | None:
    path = Path(pdf_path).expanduser()
    try:
        import fitz  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
        import pytesseract  # type: ignore[import-not-found]
    except Exception:
        return None

    try:
        with fitz.open(str(path)) as document:
            if document.page_count < 1:
                return None
            page = document.load_page(0)
            matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            pix = page.get_pixmap(matrix=matrix, alpha=False)

        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            image = image[:, :, :3]

        text = pytesseract.image_to_string(image, lang=lang, config="--oem 3 --psm 6")
        normalized = _normalize_text(text or "")
        if not normalized.strip():
            return None

        if on_used is not None:
            try:
                on_used(
                    {
                        "pdf_path": str(path),
                        "reason": reason,
                        "dpi": dpi,
                        "lang": lang,
                        "ocr_chars": len(normalized),
                    }
                )
            except Exception as exc:
                print(f"[ocr_helper] on_used callback failed: {exc}", file=sys.stderr)

        return normalized
    except Exception:
        return None
