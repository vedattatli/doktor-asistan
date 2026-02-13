from __future__ import annotations

import re

LAB_HINT_RE = re.compile(
    r"(HEMOGRAM|\bCBC\b|B[İI]YOK[İI]MYA|LABORATUVAR|REFERANS\s+ARALI[ĞG]I|GLUKOZ|HBA1C|\bLDL\b|\bCRP\b|\bTSH\b|KREAT[İI]N[İI]N|[ÜU]RE|\bAST\b|\bALT\b)",
    re.IGNORECASE,
)
PATHOLOGY_HINT_RE = re.compile(r"(PATOLOJ[İI]\s*RAPORU|TANI\s*\(ICD-O|ICD-O\s*kodları)", re.IGNORECASE)
RADIOLOGY_HINT_RE = re.compile(
    r"(RADYOLOJ[İI]|\bMR\b|\bBT\b|TOMOGRAF[İI]|ULTRASON)",
    re.IGNORECASE,
)
EPICRISIS_HINT_RE = re.compile(r"(EP[İI]KR[İI]Z|TABURCU|YATIŞ|YATIS|[ÖO]ZET[İI])", re.IGNORECASE)
ENDOSCOPY_HINT_RE = re.compile(r"(ENDOSKOP[İI]|GASTROSKOP[İI]|KOLONOSKOP[İI])", re.IGNORECASE)
SURGERY_HINT_RE = re.compile(r"(AMEL[İI]YAT|OPERASYON|ANESTEZ[İI]|CERRAH[İI])", re.IGNORECASE)


def guess_profile(query: str, context: str) -> str:
    # Query overrides context for explicit intent.
    if LAB_HINT_RE.search(query):
        return "lab"
    if PATHOLOGY_HINT_RE.search(query):
        return "pathology"
    if RADIOLOGY_HINT_RE.search(query):
        return "radiology"
    if EPICRISIS_HINT_RE.search(query):
        return "epicrisis"
    if ENDOSCOPY_HINT_RE.search(query):
        return "endoscopy"
    if SURGERY_HINT_RE.search(query):
        return "surgery"

    # Fallback to context scan with existing priority.
    if LAB_HINT_RE.search(context):
        return "lab"
    if PATHOLOGY_HINT_RE.search(context):
        return "pathology"
    if RADIOLOGY_HINT_RE.search(context):
        return "radiology"
    if EPICRISIS_HINT_RE.search(context):
        return "epicrisis"
    if ENDOSCOPY_HINT_RE.search(context):
        return "endoscopy"
    if SURGERY_HINT_RE.search(context):
        return "surgery"
    return "general"


def guess_profile_from_context(context: str) -> str:
    return guess_profile("", context)
