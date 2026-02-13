from __future__ import annotations

import re
from typing import TypedDict

TCKN_PATTERN = re.compile(
    r"""
    (?<!\d)                # no digit before
    [1-9]\d{10}            # 11-digit TCKN
    (?!\d)                 # no digit after
    """,
    re.VERBOSE,
)

PHONE_PATTERN = re.compile(
    r"""
    (?<!\d)                                # no digit before
    (?:(?:\+90|0090)\s*)?                  # optional country code
    (?:0\s*)?                              # optional trunk prefix
    \(?\d{3}\)?[\s.-]*                     # first 3 digits
    \d{3}[\s.-]*                           # second 3 digits
    \d{2}[\s.-]*                           # next 2 digits
    \d{2}                                  # last 2 digits
    (?!\d)                                 # no digit after
    """,
    re.VERBOSE,
)

DATE_PATTERN = re.compile(
    r"""
    (?<!\d)
    (?:
        (?:0?[1-9]|[12]\d|3[01])          # day
        [./-]
        (?:0?[1-9]|1[0-2])                # month
        [./-]
        (?:19|20)\d{2}                    # year
      |
        (?:19|20)\d{2}                    # year
        [./-]
        (?:0?[1-9]|1[0-2])                # month
        [./-]
        (?:0?[1-9]|[12]\d|3[01])          # day
    )
    (?!\d)
    """,
    re.VERBOSE,
)


class MaskStats(TypedDict):
    tckn: int
    phone: int
    date: int
    total: int


def mask(text: str) -> tuple[str, MaskStats]:
    stats: MaskStats = {"tckn": 0, "phone": 0, "date": 0, "total": 0}
    masked_text = text

    masked_text, stats["tckn"] = TCKN_PATTERN.subn("[TCKN]", masked_text)
    masked_text, stats["phone"] = PHONE_PATTERN.subn("[PHONE]", masked_text)
    masked_text, stats["date"] = DATE_PATTERN.subn("[DATE]", masked_text)
    stats["total"] = stats["tckn"] + stats["phone"] + stats["date"]

    return masked_text, stats


if __name__ == "__main__":
    sample_text = (
        "Hasta TCKN: 12345678901\n"
        "Telefon: +90 532 123 45 67\n"
        "Randevu Tarihi: 12.03.2024\n"
        "Alternatif Tarih: 2025-01-30\n"
    )

    masked, stats_dict = mask(sample_text)

    print("Masked text:")
    print(masked)
    print("-" * 40)
    print("stats_dict:")
    print(stats_dict)
