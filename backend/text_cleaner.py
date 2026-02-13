from __future__ import annotations

import re

CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
ALLOWED_PUNCT = ".,;:!?()[]{}%+-/\\'\""
MAX_NORMALIZED_LEN = 2500
HIGH_WEIRD_RATIO_THRESHOLD = 0.22


def _weird_ratio(text: str) -> float:
    if not text:
        return 0.0
    weird_chars = sum(
        1
        for ch in text
        if not (ch.isalnum() or ch.isspace() or ch in ALLOWED_PUNCT)
    )
    return weird_chars / max(1, len(text))


def _normalize_noisy_text(text: str) -> str:
    normalized = "".join(
        ch if (ch.isalnum() or ch.isspace() or ch in ALLOWED_PUNCT) else " "
        for ch in text
    )
    # Keep line structure while normalizing noisy content.
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n\s*\n+", "\n\n", normalized)
    normalized = normalized.strip()
    if len(normalized) > MAX_NORMALIZED_LEN:
        normalized = normalized[:MAX_NORMALIZED_LEN].rstrip()
    return normalized


def clean_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = CONTROL_CHAR_PATTERN.sub(" ", normalized)

    cleaned_lines = []
    for line in normalized.split("\n"):
        compact_line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(compact_line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n\s*\n+", "\n\n", cleaned_text)
    cleaned_text = cleaned_text.strip()

    if _weird_ratio(cleaned_text) > HIGH_WEIRD_RATIO_THRESHOLD:
        cleaned_text = _normalize_noisy_text(cleaned_text)

    return cleaned_text


if __name__ == "__main__":
    sample_text = """
    Hasta   Adı:     Ahmet    Yılmaz


      Tanı:   Kronik   bronşit



    Notlar:   Kontrol   2   hafta sonra.
    """

    print("Before:")
    print(sample_text)
    print("-" * 40)
    print("After:")
    print(clean_text(sample_text))
