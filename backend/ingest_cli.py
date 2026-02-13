from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.audit import log_event
from backend.audit_logger import log_event as log_best_effort_event
from backend.pdf_reader import extract_pages
from backend.text_cleaner import clean_text
from backend.anonymizer import mask
from backend.chunker import chunk_text
from backend.router import guess_profile
from backend.vector_store import build_index
from backend.writer import write_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDFs: reader -> cleaner -> anonymizer -> chunker -> writer.")
    parser.add_argument(
        "--input",
        required=True,
        help="PDF file path or directory containing PDF files",
    )
    parser.add_argument(
        "--outdir",
        default="out/",
        help="Output directory",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=150,
        help="Chunk overlap in characters",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build retrieval index after ingest",
    )
    parser.add_argument(
        "--indexdir",
        default=None,
        help="Index directory path (default: <outdir>/index)",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def collect_pdf_files(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() == ".pdf" else []

    pdfs = [
        path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"
    ]
    return sorted(pdfs, key=lambda path: str(path).lower())


def process_document(
    pdf_path: Path,
    chunk_size: int,
    overlap: int,
    outdir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    doc_meta: dict[str, Any] = {}

    def _on_ocr_used(meta: dict[str, Any]) -> None:
        reason = str(meta.get("reason", ""))
        doc_meta["ocr_used"] = True
        doc_meta["ocr_reason"] = reason

        log_best_effort_event(
            outdir,
            {
                "event": "OCR_USED",
                "pdf_path": str(meta.get("pdf_path", pdf_path)),
                "doc_name": pdf_path.name,
                "reason": reason,
                "dpi": int(meta.get("dpi", 300)),
                "lang": str(meta.get("lang", "tur")),
                "ocr_chars": int(meta.get("ocr_chars", 0)),
            },
        )

    pages = extract_pages(pdf_path, on_ocr_used=_on_ocr_used)
    try:
        warnings_total = sum(len(page.get("warnings", [])) for page in pages)
        warnings_unique = sorted({str(w) for page in pages for w in page.get("warnings", [])})
    except Exception:
        warnings_total = None
        warnings_unique = None

    joined_text = "\n\n".join(str(page.get("text", "")) for page in pages)
    preview_text = joined_text[:3000]
    try:
        doc_type = guess_profile("", preview_text)
    except Exception:
        doc_type = "general"

    doc_chunks: list[dict[str, Any]] = []
    doc_warnings: list[dict[str, Any]] = []
    mask_totals = {"tckn": 0, "phone": 0, "date": 0, "total": 0}

    for page_obj in pages:
        page_no = int(page_obj["page"])
        page_warnings = [str(item) for item in page_obj.get("warnings", [])]
        for code in page_warnings:
            doc_warnings.append({"page": page_no, "code": code})

        original_text = str(page_obj.get("text", ""))
        cleaned_text = clean_text(original_text)
        masked_text, mask_stats = mask(cleaned_text)
        mask_totals["tckn"] += int(mask_stats["tckn"])
        mask_totals["phone"] += int(mask_stats["phone"])
        mask_totals["date"] += int(mask_stats["date"])
        mask_totals["total"] += int(mask_stats["total"])
        page_chunks = chunk_text(masked_text, chunk_size=chunk_size, overlap=overlap)

        for chunk in page_chunks:
            doc_chunks.append(
                {
                    "chunk_id": str(uuid4()),
                    "doc_name": pdf_path.name,
                    "doc_path": str(pdf_path),
                    "page": page_no,
                    "text_original_len": len(cleaned_text),
                    "text_masked": chunk,
                    "created_at": utc_now_iso(),
                    "warnings": page_warnings,
                }
            )

    doc_summary: dict[str, Any] = {
        "doc_name": pdf_path.name,
        "doc_path": str(pdf_path),
        "doc_type": doc_type,
        "total_pages": len(pages),
        "total_chunks": len(doc_chunks),
        "mask_totals": mask_totals,
        "warnings": warnings_unique if warnings_unique is not None else doc_warnings,
        "warnings_detailed": doc_warnings,
    }
    if warnings_total is not None:
        doc_summary["warnings_total"] = warnings_total
    doc_summary.update(doc_meta)
    return doc_chunks, doc_summary


def write_docs_summary(
    outdir: Path,
    input_path: Path,
    doc_summaries: list[dict[str, Any]],
    total_chunks: int,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    docs_json_path = outdir / "docs.json"

    all_warnings: list[dict[str, Any]] = []
    total_pages = 0
    for summary in doc_summaries:
        total_pages += int(summary["total_pages"])
        warning_source = summary.get("warnings_detailed", summary.get("warnings", []))
        for warning in warning_source:
            if isinstance(warning, str):
                warning_item = {
                    "doc_name": summary["doc_name"],
                    "doc_path": summary["doc_path"],
                    "page": 0,
                    "code": warning,
                }
                all_warnings.append(warning_item)
                continue
            warning_item = {
                "doc_name": summary["doc_name"],
                "doc_path": summary["doc_path"],
                "page": warning.get("page", 0),
                "code": warning.get("code", "UNKNOWN_WARNING"),
            }
            if "detail" in warning:
                warning_item["detail"] = warning["detail"]
            all_warnings.append(warning_item)

    payload = {
        "created_at": utc_now_iso(),
        "input": str(input_path),
        "outdir": str(outdir),
        "total_docs": len(doc_summaries),
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "warnings": all_warnings,
        "docs": doc_summaries,
    }

    with docs_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return docs_json_path


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    outdir = Path(args.outdir).expanduser()
    indexdir = Path(args.indexdir).expanduser() if args.indexdir else outdir / "index"

    log_event(
        outdir,
        "START",
        {
            "input": str(input_path),
            "outdir": str(outdir),
            "indexdir": str(indexdir),
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "build_index": args.build_index,
            "python_version": sys.version.split()[0],
        },
    )

    if args.chunk_size <= 0:
        print("chunk-size must be > 0", file=sys.stderr)
        return 1
    if args.overlap < 0:
        print("overlap must be >= 0", file=sys.stderr)
        return 1
    if args.overlap >= args.chunk_size:
        print("overlap must be smaller than chunk-size", file=sys.stderr)
        return 1

    try:
        files = collect_pdf_files(input_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Input: {input_path}")
    print(f"Outdir: {outdir}")
    log_event(
        outdir,
        "FILES_FOUND",
        {
            "count": len(files),
            "files": [str(path) for path in files],
        },
    )

    if not files:
        print("No PDF files found.")
        log_event(
            outdir,
            "END",
            {
                "total_docs": 0,
                "total_pages": 0,
                "total_chunks": 0,
                "total_warnings": 0,
            },
        )
        return 0

    print(f"Found {len(files)} PDF file(s).")
    all_chunks: list[dict[str, Any]] = []
    doc_summaries: list[dict[str, Any]] = []

    for index, file in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] Processing: {file}")
        log_event(
            outdir,
            "DOC_START",
            {
                "doc_name": file.name,
                "doc_path": str(file),
            },
        )
        try:
            doc_chunks, doc_summary = process_document(
                file,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                outdir=outdir,
            )
            log_event(
                outdir,
                "PDF_READ",
                {
                    "doc_name": file.name,
                    "total_pages": int(doc_summary["total_pages"]),
                },
            )
            log_event(
                outdir,
                "MASK_STATS",
                {
                    "doc_name": file.name,
                    "totals": doc_summary["mask_totals"],
                },
            )
            log_event(
                outdir,
                "CHUNK_COUNT",
                {
                    "doc_name": file.name,
                    "total_chunks": int(doc_summary["total_chunks"]),
                },
            )
        except Exception as exc:
            print(f"Failed to process {file}: {exc}", file=sys.stderr)
            log_event(
                outdir,
                "DOC_ERROR",
                {
                    "doc_name": file.name,
                    "error_message": str(exc),
                },
            )
            doc_chunks = []
            doc_summary = {
                "doc_name": file.name,
                "doc_path": str(file),
                "total_pages": 0,
                "total_chunks": 0,
                "mask_totals": {"tckn": 0, "phone": 0, "date": 0, "total": 0},
                "error": str(exc),
                "warnings": [{"page": 0, "code": "PROCESSING_ERROR", "detail": str(exc)}],
            }
        finally:
            log_event(
                outdir,
                "DOC_END",
                {
                    "doc_name": file.name,
                },
            )

        all_chunks.extend(doc_chunks)
        doc_summaries.append(doc_summary)

    chunks_path = outdir / "chunks.jsonl"
    written_chunks = write_chunks(chunks_path, all_chunks)
    docs_path = write_docs_summary(outdir, input_path, doc_summaries, written_chunks)
    log_event(
        outdir,
        "WRITE_DONE",
        {
            "chunks_jsonl_path": str(chunks_path),
            "docs_json_path": str(docs_path),
            "total_chunks": written_chunks,
        },
    )

    total_pages = sum(int(doc["total_pages"]) for doc in doc_summaries)
    total_warnings = sum(int(doc.get("warnings_total", len(doc.get("warnings", [])))) for doc in doc_summaries)

    if args.build_index:
        print(f"Building index: {indexdir}")
        try:
            index_info = build_index(chunks_path, indexdir)
        except Exception as exc:
            log_event(
                outdir,
                "INDEX_BUILD_ERROR",
                {
                    "indexdir": str(indexdir),
                    "error_message": str(exc),
                },
            )
            print(f"Index build failed: {exc}", file=sys.stderr)
            return 1

        log_event(
            outdir,
            "INDEX_BUILD_DONE",
            {
                "indexdir": str(indexdir),
                "chunk_count": int(index_info["chunk_count"]),
                "skipped_garbage": int(index_info["skipped_garbage"]),
                "matrix_rows": int(index_info["matrix_rows"]),
                "matrix_cols": int(index_info["matrix_cols"]),
            },
        )
        print(
            "Index build complete. "
            f"chunks={index_info['chunk_count']} skipped_garbage={index_info['skipped_garbage']} "
            f"matrix_shape=({index_info['matrix_rows']}, {index_info['matrix_cols']})"
        )

    log_event(
        outdir,
        "END",
        {
            "total_docs": len(doc_summaries),
            "total_pages": total_pages,
            "total_chunks": written_chunks,
            "total_warnings": total_warnings,
        },
    )
    print(f"Done. Pages={total_pages} Chunks={written_chunks} Warnings={total_warnings}")
    print(f"- {chunks_path}")
    print(f"- {docs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
