#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any


KEYWORDS = ["TANI", "ICD", "SONUC", "DIAGNOSIS", "PATOLOJIK", "PATOLOJIK TANI"]
ALNUM_RE = re.compile(r"[^A-Z0-9]+")
DIAGNOSIS_LINE_RE = re.compile(r"(TANI|ICD|SONUÇ|SONUC|DIAGNOSIS)", flags=re.IGNORECASE)


def _normalize_token(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = folded.upper().replace("İ", "I")
    return ALNUM_RE.sub("", folded)


def _load_deps() -> tuple[Any, Any, Any, Any]:
    missing: list[str] = []

    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None
        missing.append("numpy")

    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None
        missing.append("opencv-python")

    try:
        import pytesseract  # type: ignore
    except Exception:
        pytesseract = None
        missing.append("pytesseract")

    fitz = None
    pdf2image = None
    try:
        import fitz as _fitz  # type: ignore

        fitz = _fitz
    except Exception:
        try:
            import pdf2image as _pdf2image  # type: ignore

            pdf2image = _pdf2image
        except Exception:
            missing.append("pymupdf or pdf2image")

    if missing:
        raise RuntimeError(
            "Missing dependencies: "
            + ", ".join(missing)
            + "\nInstall example: .venv/bin/pip install numpy opencv-python pytesseract pymupdf"
        )

    return np, cv2, pytesseract, (fitz or pdf2image)


def _extract_first_page_image(pdf_path: Path, dpi: int, np: Any, cv2: Any, pdf_backend: Any) -> Any:
    # PyMuPDF path
    if pdf_backend.__name__ == "fitz":
        fitz = pdf_backend
        with fitz.open(str(pdf_path)) as doc:
            if doc.page_count < 1:
                raise RuntimeError(f"PDF has no pages: {pdf_path}")
            page = doc.load_page(0)
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            image_bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            image_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return image_bgr

    # pdf2image path
    pdf2image = pdf_backend
    pages = pdf2image.convert_from_path(str(pdf_path), dpi=dpi, first_page=1, last_page=1)
    if not pages:
        raise RuntimeError(f"Failed to render first page for: {pdf_path}")
    rgb = np.array(pages[0])
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _merge_boxes(boxes: list[tuple[int, int, int, int]], x_gap: int = 30, y_gap: int = 18) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []

    merged = boxes[:]
    changed = True
    while changed:
        changed = False
        out: list[tuple[int, int, int, int]] = []
        while merged:
            x1, y1, x2, y2 = merged.pop(0)
            merged_any = False
            keep: list[tuple[int, int, int, int]] = []
            for ox1, oy1, ox2, oy2 in merged:
                overlap_x = not (x2 + x_gap < ox1 or ox2 + x_gap < x1)
                overlap_y = not (y2 + y_gap < oy1 or oy2 + y_gap < y1)
                if overlap_x and overlap_y:
                    x1, y1 = min(x1, ox1), min(y1, oy1)
                    x2, y2 = max(x2, ox2), max(y2, oy2)
                    merged_any = True
                    changed = True
                else:
                    keep.append((ox1, oy1, ox2, oy2))
            merged = keep
            out.append((x1, y1, x2, y2))
            if merged_any:
                # restart for transitive merge
                merged = out + merged
                out = []
                break
        merged = out

    merged.sort(key=lambda b: (b[1], b[0]))
    return merged


def _detect_text_regions(image_bgr: Any, cv2: Any) -> tuple[list[tuple[int, int, int, int]], Any]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    min_area = max(120, int(w * h * 0.00004))

    raw_boxes: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh < min_area:
            continue
        if bw < 25 or bh < 10:
            continue
        raw_boxes.append((x, y, x + bw, y + bh))

    merged = _merge_boxes(raw_boxes)
    return merged, binary


def _ocr_page(image_bgr: Any, pytesseract: Any, cv2: Any, lang: str) -> tuple[str, dict[str, list[Any]]]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    config = "--oem 3 --psm 6"
    full_text = pytesseract.image_to_string(rgb, lang=lang, config=config)
    data = pytesseract.image_to_data(
        rgb,
        lang=lang,
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    return full_text, data


def _keyword_hits(data: dict[str, list[Any]]) -> list[tuple[int, int, int, int, str]]:
    norm_keywords = {_normalize_token(item) for item in KEYWORDS}
    tokens = data.get("text", [])
    lefts = data.get("left", [])
    tops = data.get("top", [])
    widths = data.get("width", [])
    heights = data.get("height", [])

    hits: list[tuple[int, int, int, int, str]] = []
    for i, raw in enumerate(tokens):
        token = _normalize_token(str(raw))
        if not token:
            continue
        # Match exact, prefix, or containing keyword parts.
        matched = any(
            token == kw
            or token.startswith(kw)
            or kw in token
            or (kw.startswith(token) and len(token) >= 3)
            for kw in norm_keywords
        )
        if not matched:
            continue
        x = int(lefts[i])
        y = int(tops[i])
        w = int(widths[i])
        h = int(heights[i])
        if w <= 0 or h <= 0:
            continue
        hits.append((x, y, x + w, y + h, str(raw)))

    hits.sort(key=lambda item: (item[1], item[0]))
    return hits


def _choose_crop_box(
    image_bgr: Any,
    regions: list[tuple[int, int, int, int]],
    hits: list[tuple[int, int, int, int, str]],
) -> tuple[int, int, int, int]:
    h, w = image_bgr.shape[:2]

    if not hits:
        if regions:
            selected = regions[: min(6, len(regions))]
            x1 = min(b[0] for b in selected)
            y1 = min(b[1] for b in selected)
            x2 = max(b[2] for b in selected)
            y2 = max(b[3] for b in selected)
        else:
            return (0, 0, w, h)
    else:
        hx1, hy1, hx2, hy2, _ = hits[0]
        if regions:
            cy = (hy1 + hy2) // 2
            idx = min(range(len(regions)), key=lambda i: abs(((regions[i][1] + regions[i][3]) // 2) - cy))
            start = max(0, idx - 1)
            end = min(len(regions), idx + 5)
            selected = regions[start:end]
            x1 = min(min(b[0] for b in selected), hx1)
            y1 = min(min(b[1] for b in selected), hy1)
            x2 = max(max(b[2] for b in selected), hx2)
            y2 = max(max(b[3] for b in selected), hy2)
        else:
            x1, y1, x2, y2 = hx1, hy1, hx2, hy2

        # Diagnosis statements often continue below the heading.
        y2 = max(y2, hy2 + int(0.22 * h))

    margin_x = int(0.03 * w)
    margin_y = int(0.02 * h)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    if x2 <= x1 or y2 <= y1:
        return (0, 0, w, h)
    return (x1, y1, x2, y2)


def _clean_excerpt(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return "(no text)"

    norm_keywords = [_normalize_token(item) for item in KEYWORDS]
    hit_indexes = [
        idx
        for idx, line in enumerate(lines)
        if any(kw in _normalize_token(line) for kw in norm_keywords)
    ]

    if not hit_indexes:
        excerpt_lines = lines[:12]
    else:
        keep: set[int] = set()
        for idx in hit_indexes:
            for j in range(max(0, idx - 3), min(len(lines), idx + 4)):
                keep.add(j)
        excerpt_lines = [lines[i] for i in sorted(keep)]

    compacted = [re.sub(r"\s+", " ", line) for line in excerpt_lines]
    return "\n".join(compacted)


def _full_page_excerpt(text: str, context_radius: int = 5) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    if not lines:
        return "(no text)"

    hit_indexes = [idx for idx, line in enumerate(lines) if DIAGNOSIS_LINE_RE.search(line)]
    if not hit_indexes:
        fallback = lines[:40]
        block = "\n".join(f"[{idx + 1:04d}] {line}" for idx, line in enumerate(fallback))
        return "No diagnosis keyword found\n\n" + block

    chunks: list[str] = []
    for hit_idx in hit_indexes:
        start = max(0, hit_idx - context_radius)
        end = min(len(lines), hit_idx + context_radius + 1)
        context_block = "\n".join(f"[{i + 1:04d}] {lines[i]}" for i in range(start, end))
        chunks.append(f"--- Hit line {hit_idx + 1:04d} ---\n{context_block}")

    return "\n\n".join(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR diagnosis area from first page of a PDF")
    parser.add_argument("pdf_path", help="PDF file path")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI (default: 300)")
    parser.add_argument("--lang", default="tur", help="Tesseract language (default: tur)")
    parser.add_argument(
        "--full-page",
        action="store_true",
        help="Bypass crop/region logic and run OCR on the full first page",
    )
    parser.add_argument(
        "--save-crop",
        action="store_true",
        help="Save cropped diagnosis image to /tmp for debugging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf_path).expanduser()

    if not pdf_path.exists() or not pdf_path.is_file():
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    try:
        np, cv2, pytesseract, pdf_backend = _load_deps()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        _ = pytesseract.get_tesseract_version()
    except Exception as exc:
        print(
            "Error: Tesseract binary not found or not working. "
            "Install system package (e.g. sudo apt install tesseract-ocr tesseract-ocr-tur).",
            file=sys.stderr,
        )
        print(f"Details: {exc}", file=sys.stderr)
        return 1

    try:
        image_bgr = _extract_first_page_image(pdf_path, dpi=args.dpi, np=np, cv2=cv2, pdf_backend=pdf_backend)
        full_text, ocr_data = _ocr_page(image_bgr, pytesseract, cv2, lang=args.lang)
    except Exception as exc:
        print(f"Error: extraction/OCR failed: {exc}", file=sys.stderr)
        return 1

    out_full = Path("/tmp") / f"ocr_full_{pdf_path.name}.txt"
    out_full.write_text(full_text, encoding="utf-8")

    if args.full_page:
        print(f"PDF: {pdf_path}")
        print(f"Backend: {pdf_backend.__name__}")
        print(f"Image size: {image_bgr.shape[1]}x{image_bgr.shape[0]} @ {args.dpi} DPI")
        print(f"Keyword hits: {len(_keyword_hits(ocr_data))}")
        print("\nFull-page OCR text excerpt:\n")
        print(_full_page_excerpt(full_text, context_radius=5))
        print(f"\nSaved full OCR text: {out_full}")
        return 0

    regions, _ = _detect_text_regions(image_bgr, cv2)
    hits = _keyword_hits(ocr_data)
    crop_box = _choose_crop_box(image_bgr, regions, hits)
    x1, y1, x2, y2 = crop_box
    crop = image_bgr[y1:y2, x1:x2]

    try:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_text = pytesseract.image_to_string(crop_rgb, lang=args.lang, config="--oem 3 --psm 6")
    except Exception as exc:
        print(f"Error: cropped OCR failed: {exc}", file=sys.stderr)
        return 1

    cleaned = _clean_excerpt(crop_text)

    print(f"PDF: {pdf_path}")
    print(f"Backend: {pdf_backend.__name__}")
    print(f"Image size: {image_bgr.shape[1]}x{image_bgr.shape[0]} @ {args.dpi} DPI")
    print(f"Text regions detected (OpenCV): {len(regions)}")
    print(f"Keyword hits: {len(hits)}")
    if hits:
        print("Matched tokens:", ", ".join(h[4] for h in hits[:8]))
    else:
        print("Matched tokens: none (fallback crop used)")
    print(f"Crop box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    print("\nDiagnosis-focused OCR text:\n")
    print(cleaned)
    print(f"\nSaved full OCR text: {out_full}")

    if args.save_crop:
        crop_path = Path("/tmp") / f"ocr_crop_{pdf_path.stem}.png"
        cv2.imwrite(str(crop_path), crop)
        print(f"Saved crop image: {crop_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
