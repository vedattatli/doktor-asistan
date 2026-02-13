from __future__ import annotations

import re
from collections.abc import Callable

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?;:])\s+")


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _split_paragraphs(text: str) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    return [part.strip() for part in re.split(r"\n\s*\n+", normalized) if part.strip()]


def _merge_units(
    units: list[str],
    chunk_size: int,
    separator: str,
    oversized_fallback: Callable[[str], list[str]],
) -> list[str]:
    merged: list[str] = []
    current = ""

    for raw_unit in units:
        unit = raw_unit.strip()
        if not unit:
            continue

        if len(unit) > chunk_size:
            if current:
                merged.append(current)
                current = ""
            merged.extend(oversized_fallback(unit))
            continue

        if not current:
            current = unit
            continue

        candidate = f"{current}{separator}{unit}"
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            merged.append(current)
            current = unit

    if current:
        merged.append(current)

    return merged


def _split_by_characters(text: str, chunk_size: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        split_at = end

        if end < text_len:
            nearest_space = text.rfind(" ", start, end)
            if nearest_space > start + max(20, chunk_size // 3):
                split_at = nearest_space

        piece = text[start:split_at].strip()
        if piece:
            chunks.append(piece)

        start = split_at
        while start < text_len and text[start].isspace():
            start += 1

    return chunks


def _split_large_paragraph(paragraph: str, chunk_size: int) -> list[str]:
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if len(lines) > 1:
        return _merge_units(
            units=lines,
            chunk_size=chunk_size,
            separator="\n",
            oversized_fallback=lambda unit: _split_large_paragraph(unit, chunk_size),
        )

    sentences = [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(paragraph) if s.strip()]
    if len(sentences) > 1:
        return _merge_units(
            units=sentences,
            chunk_size=chunk_size,
            separator=" ",
            oversized_fallback=lambda unit: _split_by_characters(unit, chunk_size),
        )

    return _split_by_characters(paragraph, chunk_size)


def _build_overlap_prefix(previous_chunk: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    return previous_chunk[-overlap:].strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
        else:
            units.extend(_split_large_paragraph(paragraph, chunk_size))

    chunks: list[str] = []
    current = ""
    separator = "\n\n"

    for unit in units:
        if not current:
            current = unit
            continue

        candidate = f"{current}{separator}{unit}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        chunks.append(current)
        prefix = _build_overlap_prefix(current, overlap)

        if prefix:
            prefixed = f"{prefix}{separator}{unit}"
            if len(prefixed) <= chunk_size:
                current = prefixed
                continue

            max_prefix = max(0, chunk_size - len(unit) - len(separator))
            if max_prefix > 0:
                trimmed_prefix = prefix[-max_prefix:].strip()
                if trimmed_prefix:
                    current = f"{trimmed_prefix}{separator}{unit}"
                    continue

        current = unit

    if current:
        chunks.append(current)

    return chunks


if __name__ == "__main__":
    sample_text = (
        "Giris bolumu: Bu metin chunker davranisini gostermek icin yazildi.\n\n"
        "Ikinci paragraf: Kisa bir paragraf ve birkac ek bilgi iceriyor.\n\n"
        "Uzun paragraf: "
        + "Bu satir chunk_size sinirina yaklasmak icin tekrar ediyor. " * 45
        + "\n"
        + "Ayni paragraf icinde ikinci uzun satir da fallbacki tetikler. " * 25
    )

    produced_chunks = chunk_text(sample_text, chunk_size=1000, overlap=150)
    print(f"Produced chunk count: {len(produced_chunks)}")
    for idx, chunk in enumerate(produced_chunks[:2], start=1):
        print(f"Chunk {idx} length: {len(chunk)}")
