#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running as: python scripts/diagnose_pdf_quality.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.pdf_reader import read_pdf

PATTERNS = [
    "TANI",
    "ICD",
    "SONUÇ",
    "DIAGNOSIS",
    "TANI (",
    "TANI ICD",
]
CONTEXT_RADIUS = 3


@dataclass
class PatternResult:
    pattern: str
    line_hits: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose PDF text extraction quality by searching key diagnosis patterns."
    )
    parser.add_argument("pdf_path", help="Path to PDF file")
    return parser.parse_args()


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def find_pattern_hits(lines: list[str], pattern: str) -> PatternResult:
    regex = re.compile(re.escape(pattern), flags=re.IGNORECASE)
    hits = [idx for idx, line in enumerate(lines, start=1) if regex.search(line)]
    return PatternResult(pattern=pattern, line_hits=hits)


def build_context(lines: list[str], line_no: int, radius: int = CONTEXT_RADIUS) -> str:
    start = max(1, line_no - radius)
    end = min(len(lines), line_no + radius)
    block = [f"  [{idx:04d}] {lines[idx - 1]}" for idx in range(start, end + 1)]
    return "\n".join(block)


def build_report(pdf_path: Path, text: str) -> str:
    lines = text.splitlines()
    results = [find_pattern_hits(lines, pattern) for pattern in PATTERNS]

    found_any = any(result.line_hits for result in results)

    parts: list[str] = []
    parts.append(f"PDF: {pdf_path}")
    parts.append(f"Total chars: {len(text)}")
    parts.append(f"Total lines: {len(lines)}")
    parts.append(f"Any pattern found: {'YES' if found_any else 'NO'}")
    parts.append("")

    for result in results:
        parts.append(f"Pattern: {result.pattern}")
        parts.append(f"Found: {'YES' if result.line_hits else 'NO'}")

        if not result.line_hits:
            parts.append("Raw matching lines: (none)")
            parts.append("Line contexts (±3): (none)")
            parts.append("")
            continue

        parts.append("Raw matching lines:")
        for line_no in result.line_hits:
            parts.append(f"  [{line_no:04d}] {lines[line_no - 1]}")

        parts.append("Line contexts (±3):")
        for line_no in result.line_hits:
            parts.append(f"- Match line {line_no:04d}")
            parts.append(build_context(lines, line_no, radius=CONTEXT_RADIUS))

        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf_path).expanduser()

    if not pdf_path.exists():
        print(f"Error: PDF file does not exist: {pdf_path}", file=sys.stderr)
        return 1
    if not pdf_path.is_file():
        print(f"Error: PDF path must be a file: {pdf_path}", file=sys.stderr)
        return 1

    try:
        text = read_pdf(pdf_path)
    except Exception as exc:
        print(f"Error: failed to read PDF: {exc}", file=sys.stderr)
        return 1

    report = build_report(pdf_path, text)
    output_path = Path("/tmp") / f"pdf_quality_{safe_filename(pdf_path.name)}.txt"
    output_path.write_text(report, encoding="utf-8")

    print(report, end="")
    print(f"Saved report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
