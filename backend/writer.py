from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4


class ChunkRecord(TypedDict):
    chunk_id: str
    doc_name: str
    doc_path: str
    page: int
    text_original_len: int
    text_masked: str
    created_at: str
    warnings: list[str]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_warnings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _build_record(chunk_object: Mapping[str, Any]) -> ChunkRecord:
    text_masked = str(chunk_object.get("text_masked", ""))

    if "text_original_len" in chunk_object:
        text_original_len = int(chunk_object["text_original_len"])
    elif "text_original" in chunk_object:
        text_original_len = len(str(chunk_object["text_original"]))
    else:
        text_original_len = len(text_masked)

    created_at_value = chunk_object.get("created_at")
    if isinstance(created_at_value, datetime):
        created_at = created_at_value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    elif created_at_value:
        created_at = str(created_at_value)
    else:
        created_at = _utc_now_iso()

    return {
        "chunk_id": str(chunk_object.get("chunk_id", uuid4())),
        "doc_name": str(chunk_object.get("doc_name", "")),
        "doc_path": str(chunk_object.get("doc_path", "")),
        "page": int(chunk_object.get("page", 0)),
        "text_original_len": text_original_len,
        "text_masked": text_masked,
        "created_at": created_at,
        "warnings": _normalize_warnings(chunk_object.get("warnings")),
    }


def write_chunks(out_path: str | Path, chunk_objects: Iterable[Mapping[str, Any]]) -> int:
    path = Path(out_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as f:
        for chunk_object in chunk_objects:
            record = _build_record(chunk_object)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


if __name__ == "__main__":
    sample_chunks: list[dict[str, Any]] = [
        {
            "doc_name": "ornek1.pdf",
            "doc_path": "data/ornek1.pdf",
            "page": 1,
            "text_original": "Hasta TCKN: 12345678901",
            "text_masked": "Hasta TCKN: [TCKN]",
            "warnings": [],
        },
        {
            "doc_name": "ornek1.pdf",
            "doc_path": "data/ornek1.pdf",
            "page": 2,
            "text_masked": "",
            "warnings": ["EMPTY_TEXT_PAGE"],
        },
    ]

    output = Path("out/chunks.jsonl")
    written = write_chunks(output, sample_chunks)
    print(f"Wrote {written} chunk(s) -> {output}")
