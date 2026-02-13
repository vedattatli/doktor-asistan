from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ingest quality report from docs.json and chunks.jsonl.")
    parser.add_argument("--outdir", default="out/", help="Output directory produced by ingest pipeline")
    return parser.parse_args()


def _normalize_one_line(text: str, max_chars: int = 200) -> str:
    normalized = " ".join(text.split())
    return normalized[:max_chars]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object in {path}")
    return data


def _read_warning_items(docs_payload: dict[str, Any]) -> list[dict[str, Any]]:
    warnings = docs_payload.get("warnings")
    if isinstance(warnings, list):
        return [item for item in warnings if isinstance(item, dict)]

    fallback: list[dict[str, Any]] = []
    docs = docs_payload.get("docs", [])
    if isinstance(docs, list):
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            for warning in doc.get("warnings", []):
                if not isinstance(warning, dict):
                    continue
                fallback.append(
                    {
                        "doc_name": doc.get("doc_name", ""),
                        "doc_path": doc.get("doc_path", ""),
                        "page": warning.get("page", 0),
                        "code": warning.get("code", "UNKNOWN_WARNING"),
                    }
                )
    return fallback


def _print_general_summary(docs_payload: dict[str, Any]) -> None:
    print("== General Summary ==")
    print(f"total_docs: {int(docs_payload.get('total_docs', 0))}")
    print(f"total_pages: {int(docs_payload.get('total_pages', 0))}")
    print(f"total_chunks: {int(docs_payload.get('total_chunks', 0))}")
    print(f"total_warnings: {len(_read_warning_items(docs_payload))}")
    print()


def _print_warning_breakdown(warning_items: list[dict[str, Any]]) -> None:
    print("== Warning Breakdown ==")
    if not warning_items:
        print("No warnings found.")
        print()
        return

    by_code = Counter(str(item.get("code", "UNKNOWN_WARNING")) for item in warning_items)
    print("By code:")
    for code, count in sorted(by_code.items(), key=lambda x: (-x[1], x[0])):
        print(f"- {code}: {count}")

    print("Samples (first 10):")
    for item in warning_items[:10]:
        doc_name = str(item.get("doc_name", ""))
        page = item.get("page", 0)
        code = str(item.get("code", "UNKNOWN_WARNING"))
        print(f"- {doc_name} | page={page} | code={code}")
    print()


def _print_doc_table(docs_payload: dict[str, Any]) -> None:
    print("== Document Table ==")
    docs = docs_payload.get("docs", [])
    if not isinstance(docs, list) or not docs:
        print("No document rows found.")
        print()
        return

    rows: list[tuple[str, int, int, int]] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        doc_name = str(doc.get("doc_name", ""))
        pages = int(doc.get("total_pages", 0))
        chunks = int(doc.get("total_chunks", 0))
        warnings_count = len(doc.get("warnings", [])) if isinstance(doc.get("warnings"), list) else 0
        rows.append((doc_name, pages, chunks, warnings_count))

    name_width = max(len("doc_name"), *(len(row[0]) for row in rows))
    print(
        f"{'doc_name'.ljust(name_width)} | {'pages':>5} | {'chunks':>6} | {'warnings_count':>14}"
    )
    print(f"{'-' * name_width}-+-------+--------+---------------")
    for doc_name, pages, chunks, warnings_count in rows:
        print(f"{doc_name.ljust(name_width)} | {pages:>5} | {chunks:>6} | {warnings_count:>14}")
    print()


def _print_chunk_stats(chunks_path: Path) -> None:
    print("== Chunk Stats ==")
    total_chunks = 0
    total_len = 0
    min_len: int | None = None
    max_len: int | None = None
    short_count = 0
    doc_counter: Counter[str] = Counter()
    preview_rows: list[tuple[str, int, int, str]] = []

    with chunks_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSONL at {chunks_path}:{line_no}")

            if not isinstance(obj, dict):
                continue

            text = str(obj.get("text_masked", ""))
            text_len = len(text)
            doc_name = str(obj.get("doc_name", ""))
            page = int(obj.get("page", 0))

            total_chunks += 1
            total_len += text_len
            min_len = text_len if min_len is None else min(min_len, text_len)
            max_len = text_len if max_len is None else max(max_len, text_len)
            if text_len < 50:
                short_count += 1
            doc_counter[doc_name] += 1

            if len(preview_rows) < 2:
                preview_rows.append((doc_name, page, text_len, _normalize_one_line(text, max_chars=200)))

    if total_chunks == 0:
        print("No chunks found.")
        print()
        return

    avg_len = total_len / total_chunks
    print(f"min_len: {min_len}")
    print(f"max_len: {max_len}")
    print(f"avg_len: {avg_len:.2f}")
    print(f"short_chunks(len<50): {short_count}")
    print("Top 5 docs by chunk count:")
    for doc_name, count in doc_counter.most_common(5):
        print(f"- {doc_name}: {count}")
    print()

    print("== Preview (First 2 Chunks) ==")
    if not preview_rows:
        print("No preview rows.")
        return

    for doc_name, page, text_len, preview in preview_rows:
        print(f"- {doc_name} | page={page} | len={text_len} | {preview}")


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    docs_path = outdir / "docs.json"
    chunks_path = outdir / "chunks.jsonl"

    if not docs_path.exists():
        print(f"Error: required file not found: {docs_path}", file=sys.stderr)
        return 1
    if not chunks_path.exists():
        print(f"Error: required file not found: {chunks_path}", file=sys.stderr)
        return 1

    try:
        docs_payload = _load_json(docs_path)
        warning_items = _read_warning_items(docs_payload)

        _print_general_summary(docs_payload)
        _print_warning_breakdown(warning_items)
        _print_doc_table(docs_payload)
        _print_chunk_stats(chunks_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
