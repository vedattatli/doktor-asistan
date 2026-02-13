from __future__ import annotations

import re
from typing import TypedDict


class LabResult(TypedDict):
    test: str
    value: str
    unit: str
    reference: str


_NUM = r"[-+]?\d+(?:[.,]\d+)?"
_REF = rf"{_NUM}\s*[-–]\s*{_NUM}"
_TEST = r"[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ0-9()%#.,/_ -]{1,80}"
_UNIT = r"[A-Za-zÇĞİÖŞÜçğıöşüµμ/%^0-9._-]+"

_TEST_FIRST_RE = re.compile(
    rf"^(?P<test>{_TEST})\s+(?P<value>{_NUM})\s*(?P<unit>{_UNIT})?\s*(?P<reference>{_REF})?$"
)
_VALUE_FIRST_RE = re.compile(
    rf"^(?P<value>{_NUM})\s*(?P<unit>{_UNIT})?\s*(?P<reference>{_REF})?\s+(?P<test>{_TEST})$"
)


def _normalize_line(line: str) -> str:
    line = line.strip()
    # Some extracted rows stick reference and test without a separator (e.g. 106GLUKOZ).
    line = re.sub(r"(?<=\d)(?=[A-ZÇĞİÖŞÜ])", " ", line)
    line = re.sub(r"\s+", " ", line)
    return line


def _is_valid_test_name(value: str) -> bool:
    if not value:
        return False
    if len(value) > 90:
        return False
    if not any(ch.isalpha() for ch in value):
        return False
    return True


def _add_result(
    results: list[LabResult],
    seen: set[tuple[str, str, str, str]],
    test: str,
    value: str,
    unit: str | None,
    reference: str | None,
) -> None:
    test_clean = test.strip(" :-")
    if not _is_valid_test_name(test_clean):
        return

    value_clean = value.strip()
    unit_clean = (unit or "").strip()
    reference_clean = (reference or "").strip()
    key = (test_clean.upper(), value_clean, unit_clean, reference_clean)
    if key in seen:
        return
    seen.add(key)
    results.append(
        {
            "test": test_clean,
            "value": value_clean,
            "unit": unit_clean,
            "reference": reference_clean,
        }
    )


def parse_lab_results(text: str) -> list[dict]:
    if not text or not text.strip():
        return []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    results: list[LabResult] = []
    seen: set[tuple[str, str, str, str]] = set()

    for raw_line in normalized.splitlines():
        line = _normalize_line(raw_line)
        if not line or line.startswith("[SOURCE "):
            continue

        match = _TEST_FIRST_RE.match(line)
        if match:
            _add_result(
                results,
                seen,
                test=match.group("test"),
                value=match.group("value"),
                unit=match.group("unit"),
                reference=match.group("reference"),
            )
            continue

        match = _VALUE_FIRST_RE.match(line)
        if match:
            _add_result(
                results,
                seen,
                test=match.group("test"),
                value=match.group("value"),
                unit=match.group("unit"),
                reference=match.group("reference"),
            )

    return results

