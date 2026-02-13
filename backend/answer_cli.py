from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

from backend.audit_logger import log_event as log_best_effort_event
from backend.lab_parser import parse_lab_results
from backend.profile_loader import load_profile, load_profile_with_department
from backend.router import guess_profile


QUERY_MODE_DOCUMENT_QA = "document_qa"
QUERY_MODE_SMALLTALK = "smalltalk"


@lru_cache(maxsize=8)
def _load_docs_payload_cached(docs_json_path: str) -> dict:
    try:
        with Path(docs_json_path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _load_docs_payload(indexdir: Path) -> dict:
    docs_json_path = indexdir.parent / "docs.json"
    return _load_docs_payload_cached(str(docs_json_path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer questions from retrieval index (stub or ollama mode)")
    parser.add_argument("--indexdir", default="out/index", help="Index directory path")
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument("--topk", type=int, default=None, help="Top-K retrieved chunks (overrides profile)")
    parser.add_argument("--min-score", type=float, default=None, help="Minimum score to include chunk in context")
    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.55,
        help="Minimum cosine similarity for embedding retrieval",
    )
    parser.add_argument(
        "--retrieval",
        choices=["tfidf", "embedding"],
        default="tfidf",
        help="Retrieval backend to use",
    )
    parser.add_argument("--mode", choices=["stub", "ollama"], default="stub", help="Answer mode")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama model name")
    parser.add_argument("--profile", default="auto", help="Profile name (auto, general, pathology, ...)")
    parser.add_argument("--department", default=None, help="Department profile override (e.g., gastro)")
    parser.add_argument("--profiles-dir", default="backend/profiles", help="Directory containing YAML profiles")
    parser.add_argument(
        "--task-instruction",
        default="",
        help="Optional structured task instruction (e.g. clinical summary, timeline, lab abnormalities)",
    )
    return parser.parse_args()


def _build_context(results: list[dict]) -> str:
    blocks: list[str] = []
    for item in results:
        block = (
            f"[SOURCE doc={item.get('doc_name', '')} "
            f"page={item.get('page', 0)} "
            f"chunk_id={item.get('chunk_id', '')}]\n"
            f"{item.get('text_masked', '')}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def _normalize_preview(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "..."


def _format_sources_block(results: list[dict], limit: int = 8) -> str:
    if not results:
        return "Kaynaklar:\n- Belgede bilgi yok."

    lines = ["Kaynaklar:"]
    for item in results[: max(1, int(limit))]:
        lines.append(
            f"- doc={item.get('doc_name', '')} "
            f"page={item.get('page', 0)} "
            f"chunk_id={item.get('chunk_id', '')}"
        )
    return "\n".join(lines)


def _sources_as_items(results: list[dict], limit: int = 8) -> list[dict]:
    items: list[dict] = []
    for item in results[: max(1, int(limit))]:
        items.append(
            {
                "doc": str(item.get("doc_name", "") or ""),
                "page": int(item.get("page", 0) or 0),
                "chunk_id": str(item.get("chunk_id", "") or ""),
            }
        )
    return items


def _guarded_prompt(query: str, context: str, prompt_rules: list[str], task_instruction: str = "") -> str:
    rules_section = ""
    if prompt_rules:
        rules_section = "".join(f"- {rule}\n" for rule in prompt_rules if str(rule).strip())
    task_section = ""
    cleaned_task_instruction = str(task_instruction or "").strip()
    if cleaned_task_instruction:
        task_section = f"TASK TEMPLATE:\n{cleaned_task_instruction}\n\n"

    return (
        "SYSTEM / INSTRUCTIONS:\n"
        f"{rules_section}"
        "- CRITICAL: Answer using ONLY the provided CONTEXT from the PDF(s).\n"
        "- If the answer is not explicitly stated in the context, reply exactly: Belgede bilgi yok.\n"
        "- In that no-answer case, output only that sentence.\n"
        "- Do NOT guess and do NOT invent.\n"
        "- Do NOT use medical general knowledge unless user explicitly asks for general explanation.\n"
        "- Keep the answer concise and doctor-friendly.\n"
        "- If answer exists in context, include one short supporting quote from context in one line.\n"
        "- Quote format: Alinti: \"...\"\n"
        "- If user asks for lab values, extract them clearly.\n"
        "- If user asks for patient name/date/history, extract them from the text.\n"
        "- Tani koyma, tedavi onerme, karar verme.\n"
        "- If TASK TEMPLATE is provided, follow it strictly.\n"
        "- If answer exists in context, end with 'Kaynaklar:' and list doc/page/chunk_id.\n\n"
        f"{task_section}"
        "USER:\n"
        f"Soru: {query}\n"
        "CONTEXT:\n"
        f"{context}"
    )


def _run_ollama(model: str, prompt: str) -> str:
    try:
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Ollama calismiyor olabilir: 'ollama' komutu bulunamadi.") from exc

    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit_code={proc.returncode}"
        raise RuntimeError(f"Ollama calismiyor olabilir: {detail}")

    answer = proc.stdout.strip()
    if not answer:
        raise RuntimeError("LLM returned empty response.")
    return answer


def _resolve_topk_min_score(args: argparse.Namespace, profile: dict) -> tuple[int, float]:
    topk = int(args.topk) if args.topk is not None else int(profile.get("topk", 5))
    min_score = float(args.min_score) if args.min_score is not None else float(profile.get("min_score", 0.0))
    return topk, min_score


def _filter_results(results: list[dict], min_score: float, include_zero_scores: bool = False) -> list[dict]:
    if include_zero_scores:
        filtered = [r for r in results if float(r.get("score", 0.0)) >= min_score]
        if not filtered and results:
            filtered = [results[0]]
        return filtered

    positive = [r for r in results if float(r.get("score", 0.0)) > 0.0]
    filtered = [r for r in positive if float(r.get("score", 0.0)) >= min_score]
    if not filtered and positive:
        filtered = [positive[0]]
    return filtered


def _search_results(
    indexdir: Path,
    query: str,
    topk: int,
    retrieval: str,
    min_sim: float,
    allowed_doc_types: set[str] | None = None,
) -> list[dict]:
    try:
        from backend.vector_store import search_embedding, search_index
    except ModuleNotFoundError as exc:
        return _search_results_fallback(indexdir, query, topk, allowed_doc_types=allowed_doc_types)

    if retrieval == "embedding":
        return search_embedding(
            indexdir,
            query,
            topk,
            min_sim=min_sim,
            allowed_doc_types=allowed_doc_types,
        )
    return search_index(indexdir, query, topk, allowed_doc_types=allowed_doc_types)


def _search_results_fallback(
    indexdir: Path,
    query: str,
    topk: int,
    allowed_doc_types: set[str] | None = None,
) -> list[dict]:
    meta_path = indexdir / "meta.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing index file: {meta_path}")

    token_re = re.compile(r"[A-Za-z0-9ÇĞİÖŞÜçğıöşü]+")
    query_tokens = {token.lower() for token in token_re.findall(str(query or ""))}
    if not query_tokens:
        query_tokens = {str(query or "").strip().lower()}

    doc_type_by_name: dict[str, str] = {}
    try:
        docs_payload = _load_docs_payload(indexdir)
        docs = docs_payload.get("docs", [])
        if isinstance(docs, list):
            for doc in docs:
                if not isinstance(doc, dict):
                    continue
                doc_name = str(doc.get("doc_name", "")).strip()
                doc_type = str(doc.get("doc_type", "")).strip().lower()
                if doc_name and doc_type:
                    doc_type_by_name[doc_name] = doc_type
    except Exception:
        pass

    results: list[dict] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if not isinstance(item, dict):
                continue

            doc_name = str(item.get("doc_name", "")).strip()
            if allowed_doc_types:
                doc_type = doc_type_by_name.get(doc_name, "")
                if doc_type and doc_type not in allowed_doc_types:
                    continue

            text_masked = str(item.get("text_masked", "") or "")
            text_tokens = {token.lower() for token in token_re.findall(text_masked)}
            overlap = len(query_tokens.intersection(text_tokens))
            coverage = overlap / max(1, len(query_tokens))
            if str(query or "").strip().lower() in text_masked.lower():
                coverage += 0.2

            score = float(min(1.0, coverage))
            results.append(
                {
                    "doc_name": doc_name,
                    "page": int(item.get("page", 0) or 0),
                    "chunk_id": str(item.get("chunk_id", "") or ""),
                    "text_masked": text_masked,
                    "score": score,
                }
            )

    results.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    return results[: max(1, int(topk))]


def _doc_type_inventory(indexdir: Path) -> dict[str, int]:
    try:
        docs_payload = _load_docs_payload(indexdir)
        docs = docs_payload.get("docs", [])
        if not isinstance(docs, list):
            return {}

        counts: dict[str, int] = {}
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            doc_type = str(doc.get("doc_type", "")).strip().lower()
            if not doc_type:
                continue
            counts[doc_type] = counts.get(doc_type, 0) + 1
        return counts
    except Exception:
        return {}


def _format_doc_type_inventory(counts: dict[str, int]) -> str:
    ordered = ["radiology", "pathology", "endoscopy", "epicrisis", "surgery"]
    parts = [f"{name}={counts.get(name, 0)}" for name in ordered]
    extras = [name for name in sorted(counts.keys()) if name not in set(ordered)]
    parts.extend(f"{name}={counts[name]}" for name in extras)
    return " ".join(parts)


def _allowed_doc_types_for_profile(
    profile_name: str,
    doc_type_counts: dict[str, int] | None = None,
) -> list[str]:
    key = (profile_name or "").strip().lower()
    counts = doc_type_counts or {}

    if key in {"", "general", "auto"}:
        if counts:
            return sorted(counts.keys())
        return ["lab", "radiology", "pathology", "endoscopy", "epicrisis", "surgery"]

    if key == "radiology":
        if counts.get("radiology", 0) == 0:
            return ["epicrisis"]
        return ["radiology", "epicrisis"]
    if key == "pathology":
        if counts.get("pathology", 0) == 0:
            return ["endoscopy", "epicrisis"]
        return ["pathology", "endoscopy", "epicrisis"]
    if key == "endoscopy":
        if counts.get("endoscopy", 0) == 0:
            return ["epicrisis"]
        return ["endoscopy", "epicrisis"]
    if key == "surgery":
        if counts.get("surgery", 0) == 0:
            return ["epicrisis"]
        return ["surgery", "epicrisis"]
    if key == "epicrisis":
        return ["epicrisis"]

    mapping = {
        "radiology": ["radiology", "epicrisis"],
        "pathology": ["pathology", "endoscopy", "epicrisis"],
        "endoscopy": ["endoscopy", "epicrisis"],
        "surgery": ["surgery", "epicrisis"],
        "epicrisis": ["epicrisis"],
    }
    if key in mapping:
        return mapping[key]
    return [key or "general"]


def _log_profile_selected(
    indexdir: Path,
    query: str,
    base_profile: str,
    department: str | None,
    effective_profile: str,
) -> None:
    try:
        outdir = indexdir.parent
        event = {
            "event": "PROFILE_SELECTED",
            "query": query,
            "indexdir": str(indexdir),
            "base_profile": base_profile,
            "department": department or "none",
            "effective_profile": effective_profile,
            "profile": effective_profile,
        }
        log_best_effort_event(outdir, event)
    except Exception:
        pass


def _print_ocr_used(indexdir: Path, context_results: list[dict]) -> None:
    try:
        docs_payload = _load_docs_payload(indexdir)
        docs = docs_payload.get("docs", [])
        used_doc_names = {str(item.get("doc_name", "")) for item in context_results if item.get("doc_name")}
        ocr_doc_names = [
            str(doc.get("doc_name", ""))
            for doc in docs
            if str(doc.get("doc_name", "")) in used_doc_names and bool(doc.get("ocr_used", False))
        ]

        if ocr_doc_names:
            print(f"OCR used: yes ({ocr_doc_names[0]})")
        else:
            print("OCR used: no")
    except Exception:
        print("OCR used: unknown")


def _index_inventory(indexdir: Path) -> tuple[int, int] | None:
    try:
        docs_payload = _load_docs_payload(indexdir)
        docs = docs_payload.get("docs", [])
        if not isinstance(docs, list):
            return None

        all_doc_names: set[str] = set()
        ocr_doc_names: set[str] = set()
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            doc_name = str(doc.get("doc_name", "")).strip()
            if not doc_name:
                continue
            all_doc_names.add(doc_name)
            if bool(doc.get("ocr_used", False)):
                ocr_doc_names.add(doc_name)

        return len(all_doc_names), len(ocr_doc_names)
    except Exception:
        return None


LAB_TOKEN_WHITELIST = {
    "HGB",
    "WBC",
    "RBC",
    "PLT",
    "HCT",
    "MCV",
    "MCH",
    "MCHC",
    "RDW",
    "CRP",
    "TSH",
    "LDL",
    "HDL",
    "AST",
    "ALT",
    "ALP",
    "GGT",
    "URE",
    "BUN",
    "CREA",
    "GLU",
    "GLUKOZ",
    "HBA1C",
    "NA",
    "K",
    "CL",
    "CA",
    "MG",
    "FE",
    "FERRI",
    "VITD",
    "B12",
}

METADATA_KEYWORDS = [
    "ad",
    "soyad",
    "isim",
    "adı",
    "adı soyadı",
    "tc",
    "kimlik",
    "doğum",
    "doğum tarihi",
    "cinsiyet",
    "protokol",
    "dosya",
    "istem",
    "rapor tarihi",
    "tarih",
    "poliklinik",
    "klinik",
    "bölüm",
    "doktor",
    "şikayet",
    "başvuru",
]

DOC_QUERY_TOKEN_KEYWORDS = {
    "ad",
    "adı",
    "soyad",
    "isim",
    "hasta",
    "öykü",
    "oyku",
    "özet",
    "ozet",
    "tanı",
    "tani",
    "değer",
    "deger",
    "sonuç",
    "sonuc",
    "rapor",
    "tarih",
    "değerler",
    "degerler",
    "değeri",
    "degeri",
    "patoloji",
    "radyoloji",
    "epikriz",
    "endoskopi",
    "ameliyat",
    "protokol",
    "dosya",
    "istem",
    "cinsiyet",
    "doğum",
    "dogum",
    "lab",
    "laboratuvar",
    "hgb",
    "hba1c",
    "crp",
    "wbc",
    "rbc",
    "plt",
    "tsh",
    "glu",
    "glukoz",
    "ast",
    "alt",
    "ldl",
    "hdl",
}

DOC_QUERY_PHRASE_KEYWORDS = [
    "adı soyadı",
    "ad soyad",
    "doğum tarihi",
    "dogum tarihi",
    "hasta adı",
    "hasta soyadı",
    "lab sonucu",
    "lab değer",
    "lab deger",
]

SMALLTALK_GREETINGS = {
    "selam",
    "merhaba",
    "slm",
    "hey",
    "günaydın",
    "gunaydin",
    "iyi akşamlar",
    "iyi aksamlar",
}

SMALLTALK_WHAT_ARE_YOU_DOING = [
    "ne yapıyorsun",
    "ne yapiyorsun",
    "napıyorsun",
    "napiyorsun",
    "ne yaparsın",
    "ne yapabilirsin",
]


def _is_metadata_question(q: str) -> bool:
    s = (q or "").lower()
    return any(keyword in s for keyword in METADATA_KEYWORDS)


def _is_patient_name_question(query: str) -> bool:
    q = str(query or "").lower()
    tokens = [
        "adı",
        "soyadı",
        "ad soyad",
        "ad soyadı",
        "isim",
        "hasta adı",
        "hasta ismi",
        "hasta adı soyadı",
    ]
    return any(token in q for token in tokens)


def _is_patient_card_question(query: str) -> bool:
    q = str(query or "").lower()
    return any(token in q for token in ["hasta kartı", "hasta karti", "patient card"])


TR_LOWER_OVERRIDES = str.maketrans({"I": "ı", "İ": "i"})
NAME_STOP_TOKEN_PATTERN = re.compile(
    r"\b(?:TC\s*Kimlik|TC|T\.?C\.?|Protokol|Dosya|İstem|Istem|Rapor|Doğum|Dogum)\b.*$",
    flags=re.IGNORECASE,
)
PREFERRED_METADATA_DOC_KEYWORDS = ("rapor", "epikriz", "sonuç", "sonuc")
DATE_DMY_RE = re.compile(r"\b([0-3]?\d)[./-]([01]?\d)[./-](\d{2}|\d{4})\b")
DATE_YMD_RE = re.compile(r"\b(\d{4})[./-]([01]?\d)[./-]([0-3]?\d)\b")


def _tr_lower(s: str) -> str:
    return str(s or "").translate(TR_LOWER_OVERRIDES).lower()


def _tr_upper(s: str) -> str:
    out: list[str] = []
    for ch in str(s or ""):
        if ch == "i":
            out.append("İ")
        elif ch == "ı":
            out.append("I")
        else:
            out.append(ch.upper())
    return "".join(out)


def _tr_title_case(s: str) -> str:
    normalized = re.sub(r"\s+", " ", str(s or "")).strip()
    if not normalized:
        return ""

    words: list[str] = []
    for word in normalized.split(" "):
        parts = re.split(r"([\-'])", word)
        out_parts: list[str] = []
        for part in parts:
            if not part:
                continue
            if part in {"-", "'"}:
                out_parts.append(part)
                continue
            lower_word = _tr_lower(part)
            out_parts.append(_tr_upper(lower_word[:1]) + lower_word[1:])
        words.append("".join(out_parts))
    return " ".join(words)


# Turkish title-case sanity samples:
# _tr_title_case("ESRA OZKENT") == "Esra Ozkent"
# _tr_title_case("ESRA ÖZKENT") == "Esra Özkent"
# _tr_title_case("İBRAHİM") == "İbrahim"
# _tr_title_case("IŞIK") == "Işık"
# _tr_title_case("ALİ") == "Ali"


def _is_plausible_name_word(token: str) -> bool:
    cleaned = str(token or "").strip(" .,:;|_/")
    if not cleaned:
        return False
    parts = re.split(r"[-']", cleaned)
    if not parts:
        return False
    letter_count = 0
    for part in parts:
        if not part:
            return False
        if not re.fullmatch(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", part):
            return False
        letter_count += len(part)
    return letter_count >= 2


def _normalize_name_candidate(raw: str) -> str | None:
    if not raw:
        return None

    candidate = str(raw).replace("/", " ").replace("’", "'")
    candidate = re.sub(r"\s+", " ", candidate).strip(" :;,-_|")
    candidate = NAME_STOP_TOKEN_PATTERN.sub("", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" :;,-_|")
    if not candidate:
        return None

    words = [w.strip(" :;,-_|") for w in candidate.split() if _is_plausible_name_word(w)]
    if len(words) < 2:
        return None

    candidate = _tr_title_case(" ".join(words))
    return candidate or None


def _extract_patient_name_from_context(context: str) -> str | None:
    text = str(context or "")
    if not text:
        return None

    patterns = [
        r"(?:Hastanın\s+Adı\s*,?\s*Soyadı|Hastanin\s+Adi\s*,?\s*Soyadi)\s*[:\-]?\s*([^\n]+)",
        r"(?:Hasta\s+Adı\s*Soyadı|Hasta\s+Adi\s+Soyadi)\s*[:\-]?\s*([^\n]+)",
        r"(?:Ad\s*Soyad)\s*[:\-]?\s*([^\n]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            name = _normalize_name_candidate(match.group(1))
            if name:
                return name
    return None


def _extract_patient_age_from_context(context: str) -> str | None:
    text = str(context or "")
    if not text:
        return None

    patterns = [
        r"\b(\d{1,3})\s*(?:YAŞ|YAS)\b",
        r"(?:Yaş|YAS|YAŞ)\s*[:\-]?\s*(\d{1,3})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return str(match.group(1))
    return None


def _extract_patient_sex_from_context(context: str) -> str | None:
    text = str(context or "")
    if not text:
        return None

    match = re.search(
        r"(?:Cinsiyet|Sex)\s*[:\-]?\s*(Kadın|Erkek|K|E|Female|Male)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    token = str(match.group(1)).strip().lower()
    if token in {"kadın", "k", "female"}:
        return "Kadın"
    if token in {"erkek", "e", "male"}:
        return "Erkek"
    return match.group(1)


def _extract_report_date_from_context(context: str) -> str | None:
    text = str(context or "")
    if not text:
        return None

    match = re.search(
        r"(?:Rapor\s*Tarihi|Tarih)\s*[:\-]?\s*([0-3]?\d[./-][01]?\d[./-](?:\d{4}|\d{2}))",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return str(match.group(1))


def _extract_protocol_no_from_context(context: str) -> str | None:
    text = str(context or "")
    if not text:
        return None

    patterns = [
        r"Protokol\s*/\s*Dosya\s*/\s*İstem\s*No\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9/\-]*)",
        r"(?:Protokol(?:\s*No|\s*Numarası)?|Dosya\s*No|İstem\s*No|Istem\s*No)\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9/\-]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = str(match.group(1) or "").strip(" .,:;|")
        if value and re.search(r"[A-Za-z0-9]", value) and re.search(r"\d", value):
            return value
    return None


def _doc_key(doc_name: str) -> str:
    return str(doc_name or "").strip().lower()


def _parse_date_value(year_raw: str, month_raw: str, day_raw: str) -> date | None:
    try:
        year = int(year_raw)
        month = int(month_raw)
        day = int(day_raw)
    except ValueError:
        return None
    if len(str(year_raw)) == 2:
        year += 2000
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _extract_dates_from_text(text: str) -> list[date]:
    found: set[date] = set()
    content = str(text or "")
    if not content:
        return []

    for year_raw, month_raw, day_raw in DATE_YMD_RE.findall(content):
        parsed = _parse_date_value(year_raw, month_raw, day_raw)
        if parsed:
            found.add(parsed)
    for day_raw, month_raw, year_raw in DATE_DMY_RE.findall(content):
        parsed = _parse_date_value(year_raw, month_raw, day_raw)
        if parsed:
            found.add(parsed)
    return sorted(found)


def _iter_string_values(value: object, depth: int = 0) -> list[str]:
    if depth > 5:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for item in value.values():
            strings.extend(_iter_string_values(item, depth + 1))
        return strings
    if isinstance(value, list):
        strings = []
        for item in value:
            strings.extend(_iter_string_values(item, depth + 1))
        return strings
    return []


def _update_doc_latest_date(doc_dates: dict[str, date], doc_name: str, dates: list[date]) -> None:
    if not dates:
        return
    key = _doc_key(doc_name)
    if not key:
        return
    best = max(dates)
    current = doc_dates.get(key)
    if current is None or best > current:
        doc_dates[key] = best


def _collect_doc_dates_from_docs_payload(docs_payload: dict) -> dict[str, date]:
    doc_dates: dict[str, date] = {}
    docs = docs_payload.get("docs", [])
    if not isinstance(docs, list):
        return doc_dates
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        doc_name = str(doc.get("doc_name", "") or "").strip()
        if not doc_name:
            continue
        for value in _iter_string_values(doc):
            _update_doc_latest_date(doc_dates, doc_name, _extract_dates_from_text(value))
    return doc_dates


def _collect_doc_dates_from_results(doc_dates: dict[str, date], results: list[dict], limit: int = 40) -> None:
    sample_limit = max(1, min(int(limit), len(results)))
    for item in results[:sample_limit]:
        if not isinstance(item, dict):
            continue
        doc_name = str(item.get("doc_name", "") or "").strip()
        if not doc_name:
            continue
        text = str(item.get("text_masked", "") or "")
        _update_doc_latest_date(doc_dates, doc_name, _extract_dates_from_text(text))


def _is_preferred_metadata_doc(doc_name: str) -> bool:
    lowered = str(doc_name or "").lower()
    return any(keyword in lowered for keyword in PREFERRED_METADATA_DOC_KEYWORDS)


def _prioritize_results_for_metadata(indexdir: Path, results: list[dict]) -> list[dict]:
    if not results:
        return []

    doc_dates: dict[str, date] = {}
    try:
        docs_payload = _load_docs_payload(indexdir)
        doc_dates = _collect_doc_dates_from_docs_payload(docs_payload)
    except Exception:
        doc_dates = {}
    _collect_doc_dates_from_results(doc_dates, results, limit=40)

    indexed = list(enumerate(results))

    def sort_key(entry: tuple[int, dict]) -> tuple[int, int, int, float, int]:
        index, item = entry
        doc_name = str(item.get("doc_name", "") or "")
        key = _doc_key(doc_name)
        best_date = doc_dates.get(key)
        preferred_rank = 0 if _is_preferred_metadata_doc(doc_name) else 1
        date_rank = 0 if best_date else 1
        date_ord = -(best_date.toordinal() if best_date else -1)
        try:
            score = -float(item.get("score", 0.0))
        except Exception:
            score = 0.0
        return (preferred_rank, date_rank, date_ord, score, index)

    return [item for _, item in sorted(indexed, key=sort_key)]


def _build_patient_card_payload(context: str, results: list[dict]) -> dict:
    return {
        "name": _extract_patient_name_from_context(context) or "Belgede bilgi yok.",
        "age": _extract_patient_age_from_context(context) or "Belgede bilgi yok.",
        "sex": _extract_patient_sex_from_context(context) or "Belgede bilgi yok.",
        "protocol_no": _extract_protocol_no_from_context(context) or "Belgede bilgi yok.",
        "report_date": _extract_report_date_from_context(context) or "Belgede bilgi yok.",
        "sources": _sources_as_items(results),
    }


def _detect_query_mode(query: str) -> str:
    q = str(query or "").strip()
    if not q:
        return QUERY_MODE_SMALLTALK

    up = q.upper()
    low = q.lower()
    token_pattern = r"\b[0-9A-ZÇĞİÖŞÜa-zçğıöşü]+\b"
    tokens = set(re.findall(token_pattern, q))
    up_tokens = set(re.findall(r"\b[A-Z0-9ÇĞİÖŞÜ]+\b", up))

    if any(token in up_tokens for token in LAB_TOKEN_WHITELIST):
        return QUERY_MODE_DOCUMENT_QA
    if any(token.lower() in DOC_QUERY_TOKEN_KEYWORDS for token in tokens):
        return QUERY_MODE_DOCUMENT_QA
    if any(phrase in low for phrase in DOC_QUERY_PHRASE_KEYWORDS):
        return QUERY_MODE_DOCUMENT_QA

    return QUERY_MODE_SMALLTALK


def _smalltalk_response(query: str) -> str:
    q = str(query or "").strip().lower()
    if any(greet in q for greet in SMALLTALK_GREETINGS):
        return "Selam! İstersen PDF’den hasta adı/özet/lab anormallikleri çıkarayım. Ne arıyorsun?"
    if any(p in q for p in SMALLTALK_WHAT_ARE_YOU_DOING):
        return "Buradayım. PDF yüklersen özet, kronoloji, değer çıkarma yapabilirim."
    return "Buradayım. İstersen PDF’den hasta adı, özet, kronoloji veya lab değerlerini çıkarabilirim."


def _extract_lab_target_token(query: str) -> str | None:
    candidates = re.findall(r"\b[A-Z]{2,6}\b", str(query).upper())
    for candidate in candidates:
        if candidate in LAB_TOKEN_WHITELIST:
            return candidate
        if candidate.startswith("D") and len(candidate) > 1 and candidate[1:] in LAB_TOKEN_WHITELIST:
            return candidate[1:]
    return None


LAB_TARGET_LINE_RE = re.compile(
    r"(?P<value>[-+]?\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>[A-Za-zÇĞİÖŞÜçğıöşüµμ/%^0-9._-]+)?\s*"
    r"(?P<ref>\d+(?:[.,]\d+)?\s*[-–]\s*\d+(?:[.,]\d+)?)?",
    re.UNICODE,
)

LAB_UNIT_PREFERENCE: dict[str, list[str]] = {
    "WBC": ["10^3/μL", "10^3/UL", "10^3"],
    "PLT": ["10^3/μL", "10^3/UL", "10^3"],
    "RBC": ["10^6/μL", "10^6/UL", "10^6"],
    "HGB": ["G/DL"],
    "HCT": ["%"],
    "GLU": ["MG/DL"],
    "GLUKOZ": ["MG/DL"],
    "CRP": ["MG/L"],
    "TSH": ["μIU/ML", "UIU/ML"],
}


def _lab_target_variants(target: str) -> list[str]:
    t = (target or "").strip().upper()
    if not t:
        return []
    prefixes = ["", "D", "Y"]
    variants: list[str] = []
    for prefix in prefixes:
        variants.append(prefix + t)
    return variants


def _rerank_lab_tfidf_zero_scores(results: list[dict], target: str) -> list[dict]:
    if not results:
        return results

    try:
        max_score = max(float(r.get("score", 0.0)) for r in results)
    except Exception:
        max_score = 0.0

    if max_score > 0.0:
        return results

    variants = _lab_target_variants(target)
    if not variants:
        return results

    def has_target(r: dict) -> bool:
        text = str(r.get("text_masked", "") or "")
        up = text.upper()
        return any(v in up for v in variants)

    hits = [r for r in results if has_target(r)]
    misses = [r for r in results if not has_target(r)]
    if not hits:
        return results
    return hits + misses


def _pick_best_lab_match(
    matches: list[re.Match],
    token_pos: int,
    line_upper: str,
    target: str,
) -> re.Match | None:
    if not matches:
        return None

    tgt = (target or "").upper()

    def score(match: re.Match) -> tuple[int, int, int, int]:
        reference = (match.group("ref") or "").strip()
        unit = (match.group("unit") or "").strip()
        has_ref = 2 if reference else 0
        has_unit = 1 if unit else 0

        bonus = 0
        if tgt == "RBC":
            unit_upper = unit.upper()
            if "10^6" in unit_upper or "10^6" in line_upper:
                bonus += 5
            if "HPF" in line_upper:
                bonus -= 4
            if "ERITROS" in line_upper or "ERİTROS" in line_upper:
                bonus += 2

        dist = abs(token_pos - match.end())
        return (bonus + has_ref + has_unit, -dist, match.end(), 1 if unit else 0)

    return max(matches, key=score)


def _extract_target_from_context(context: str, target: str) -> dict | None:
    if not context or not target:
        return None

    t = target.upper()
    variants = _lab_target_variants(t)
    if not variants:
        return None

    best_candidate: dict | None = None
    best_key: tuple[int, int, int, int, int] | None = None

    for line_index, raw in enumerate(context.splitlines()):
        line = raw.strip()
        if not line:
            continue

        up = line.upper()

        def _find_token_pos(text: str, token: str) -> int:
            match = re.search(rf"(?<![A-Z0-9]){re.escape(token)}(?![A-Z0-9])", text)
            return match.start() if match else -1

        positions: list[int] = []
        for variant in variants:
            pos = _find_token_pos(up, variant)
            if pos >= 0:
                positions.append(pos)
        if not positions:
            continue

        pos = min(positions)
        matches = list(LAB_TARGET_LINE_RE.finditer(line))
        if not matches:
            continue

        left_matches = [m for m in matches if m.end() <= pos]
        line_upper = line.upper()
        chosen = _pick_best_lab_match(
            left_matches,
            pos,
            line_upper,
            target,
        ) or _pick_best_lab_match(
            matches,
            pos,
            line_upper,
            target,
        )
        if not chosen:
            continue

        value = (chosen.group("value") or "").strip()
        if not value:
            continue

        unit = (chosen.group("unit") or "").strip()
        reference = (chosen.group("ref") or "").strip()
        unit_up = unit.upper().replace("µ", "U").replace("μ", "U").replace("Μ", "U")
        preferred_units = [
            p.upper().replace("µ", "U").replace("μ", "U").replace("Μ", "U")
            for p in LAB_UNIT_PREFERENCE.get(target.upper(), [])
        ]
        unit_match = 1 if any(p in unit_up for p in preferred_units) else 0
        candidate_key = (
            unit_match,
            line_index,
            1 if reference else 0,
            1 if unit else 0,
            chosen.end(),
        )
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_candidate = {
                "test": target,
                "value": value,
                "unit": unit,
                "reference": reference,
            }

    return best_candidate


def _format_lab_answer_from_context(context: str, query: str) -> str | None:
    target = _extract_lab_target_token(query)
    if target:
        direct = _extract_target_from_context(context, target)
        if direct and direct.get("value"):
            lines = [
                f"TEST: {direct.get('test', '-')}",
                f"DEĞER: {direct.get('value', '-')}",
                f"BİRİM: {direct.get('unit') or '-'}",
                f"REFERANS: {direct.get('reference') or '-'}",
            ]
            return "\n".join(lines)

    rows = parse_lab_results(context)
    if not rows:
        return None

    if target:
        matched_rows = [
            row
            for row in rows
            if target in str(row.get("test", "")).upper()
        ]
        if matched_rows:
            selected_rows = matched_rows[:3]
            lines: list[str] = []
            for row in selected_rows:
                lines.append(f"TEST: {row.get('test', '-')}")
                lines.append(f"DEĞER: {row.get('value', '-')}")
                lines.append(f"BİRİM: {row.get('unit', '') or '-'}")
                lines.append(f"REFERANS: {row.get('reference', '') or '-'}")

            other_count = max(0, len(rows) - len(selected_rows))
            if other_count:
                lines.append(f"Diğer sonuçlar: {other_count} satır")
            return "\n".join(lines)

    lines: list[str] = []
    for row in rows[:10]:
        lines.append(f"TEST: {row.get('test', '-')}")
        lines.append(f"DEĞER: {row.get('value', '-')}")
        lines.append(f"BİRİM: {row.get('unit', '') or '-'}")
        lines.append(f"REFERANS: {row.get('reference', '') or '-'}")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    task_instruction = str(args.task_instruction or "").strip()
    query_mode = QUERY_MODE_DOCUMENT_QA if task_instruction else _detect_query_mode(args.query)

    print(f"Selected query mode: {query_mode}")
    if task_instruction:
        print("Selected task template: enabled")
    if query_mode == QUERY_MODE_SMALLTALK:
        print()
        print("Answer:")
        print(_smalltalk_response(args.query))
        return 0

    indexdir = Path(args.indexdir).expanduser()

    profiles_dir = Path(args.profiles_dir).expanduser()
    requested_profile = str(args.profile or "general").strip().lower() or "general"
    force_lab_profile = requested_profile == "lab"
    if requested_profile == "auto" and _extract_lab_target_token(args.query):
        force_lab_profile = True
    base_profile_name = requested_profile
    if requested_profile == "auto":
        base_profile_name = "general"
    if force_lab_profile:
        base_profile_name = "lab"
    base_profile = load_profile(base_profile_name, profiles_dir)
    selected_profile = base_profile
    topk, min_score = _resolve_topk_min_score(args, base_profile)

    if topk <= 0:
        print("Error: --topk must be > 0", file=sys.stderr)
        return 1
    if min_score < 0:
        print("Error: --min-score must be >= 0", file=sys.stderr)
        return 1
    if args.min_sim < 0:
        print("Error: --min-sim must be >= 0", file=sys.stderr)
        return 1

    try:
        results = _search_results(indexdir, args.query, topk, args.retrieval, args.min_sim)
    except FileNotFoundError as exc:
        print("Error: index files not found. Önce build et:", file=sys.stderr)
        print(
            f"  .venv/bin/python -m backend.retrieve_cli build --chunks out/chunks.jsonl --indexdir {indexdir}",
            file=sys.stderr,
        )
        print(f"Missing: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    filtered = _filter_results(results, min_score)
    context = _build_context(filtered)

    if requested_profile == "auto" and not force_lab_profile:
        guessed_name = guess_profile(args.query, context)
        guessed_profile = load_profile(guessed_name, profiles_dir)
        guessed_topk, guessed_min_score = _resolve_topk_min_score(args, guessed_profile)
        if guessed_topk <= 0:
            print("Error: --topk must be > 0", file=sys.stderr)
            return 1
        if guessed_min_score < 0:
            print("Error: --min-score must be >= 0", file=sys.stderr)
            return 1

        base_profile = guessed_profile
        selected_profile = guessed_profile
        base_profile_name = guessed_name
        print(f"Guessed doc type: {guessed_name}")
        if guessed_topk != topk or guessed_min_score != min_score:
            topk, min_score = guessed_topk, guessed_min_score
            try:
                results = _search_results(indexdir, args.query, topk, args.retrieval, args.min_sim)
            except FileNotFoundError as exc:
                print("Error: index files not found. Önce build et:", file=sys.stderr)
                print(
                    f"  .venv/bin/python -m backend.retrieve_cli build --chunks out/chunks.jsonl --indexdir {indexdir}",
                    file=sys.stderr,
                )
                print(f"Missing: {exc}", file=sys.stderr)
                return 1
            except Exception as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            filtered = _filter_results(results, min_score)
            context = _build_context(filtered)

    selected_profile = load_profile_with_department(base_profile_name, args.department, profiles_dir)
    if base_profile_name.strip().lower() != "general":
        selected_name = str(selected_profile.get("name", "general")).strip().lower()
        if selected_name == "general":
            selected_profile = base_profile
    effective_profile_name = str(selected_profile.get("name", "general"))
    retrieval_mode = args.retrieval
    allowed_doc_types: set[str] | None = None
    allowed_doc_types_display: list[str] | None = None
    doc_type_counts: dict[str, int] | None = None

    if retrieval_mode in {"embedding", "tfidf"}:
        doc_type_counts = _doc_type_inventory(indexdir)
        allowed_doc_types_display = _allowed_doc_types_for_profile(effective_profile_name, doc_type_counts)
        allowed_doc_types = set(allowed_doc_types_display)
        if effective_profile_name.strip().lower() == "lab":
            allowed_doc_types_display = ["lab"]
            allowed_doc_types = {"lab"}
        try:
            results = _search_results(
                indexdir,
                args.query,
                topk,
                retrieval_mode,
                args.min_sim,
                allowed_doc_types=allowed_doc_types,
            )
        except FileNotFoundError as exc:
            print("Error: index files not found. Önce build et:", file=sys.stderr)
            print(
                f"  .venv/bin/python -m backend.retrieve_cli build --chunks out/chunks.jsonl --indexdir {indexdir}",
                file=sys.stderr,
            )
            print(f"Missing: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        if effective_profile_name == "lab" and retrieval_mode == "tfidf" and len(results) == 0:
            print("Lab fallback to embedding (tfidf empty)")
            try:
                embedding_results = _search_results(
                    indexdir,
                    args.query,
                    topk,
                    "embedding",
                    args.min_sim,
                    allowed_doc_types=allowed_doc_types,
                )
                results = embedding_results
                retrieval_mode = "embedding"
            except FileNotFoundError:
                # Keep TF-IDF empty result set when embedding index is unavailable.
                pass
            except Exception as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1

        # Radiology can fallback to epicrisis-only if mixed filter returns nothing.
        if retrieval_mode == "embedding" and not results and effective_profile_name == "radiology":
            try:
                results = _search_results(
                    indexdir,
                    args.query,
                    topk,
                    retrieval_mode,
                    args.min_sim,
                    allowed_doc_types={"epicrisis"},
                )
            except FileNotFoundError as exc:
                print("Error: index files not found. Önce build et:", file=sys.stderr)
                print(
                    f"  .venv/bin/python -m backend.retrieve_cli build --chunks out/chunks.jsonl --indexdir {indexdir}",
                    file=sys.stderr,
                )
                print(f"Missing: {exc}", file=sys.stderr)
                return 1
            except Exception as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1

        if effective_profile_name.strip().lower() == "lab" and retrieval_mode == "tfidf" and results:
            try:
                max_score = max(float(item.get("score", 0.0)) for item in results)
            except Exception:
                max_score = 0.0

            target = _extract_lab_target_token(args.query) or ""
            if max_score <= 0.0 and target:
                widened_topk = max(topk, 80)
                widened = _search_results(
                    indexdir,
                    args.query,
                    widened_topk,
                    "tfidf",
                    args.min_sim,
                    allowed_doc_types=allowed_doc_types,
                )
                widened = _rerank_lab_tfidf_zero_scores(widened, target)
                results = widened[:topk]
            else:
                results = _rerank_lab_tfidf_zero_scores(results, target)

        include_zero_scores = effective_profile_name.strip().lower() == "lab"
        filtered = _filter_results(results, min_score, include_zero_scores=include_zero_scores)
        context = _build_context(filtered)

    print(f"Selected base profile: {base_profile_name}")
    print(f"Selected department: {args.department or 'none'}")
    print(f"Selected effective profile: {effective_profile_name}")
    _log_profile_selected(
        indexdir=indexdir,
        query=args.query,
        base_profile=base_profile_name,
        department=args.department,
        effective_profile=effective_profile_name,
    )

    embedding_max_score: float | None = None
    if retrieval_mode == "embedding":
        embedding_max_score = max((float(item.get("score", 0.0)) for item in results), default=0.0)
        print(f"Embedding max_score: {embedding_max_score:.4f}")
        print(f"Doc_type inventory: {_format_doc_type_inventory(doc_type_counts or {})}")
    if allowed_doc_types_display:
        allowed_display = ", ".join(allowed_doc_types_display)
        print(f"Allowed doc_types: {allowed_display}")

    try:
        chunk_count: int | None = None
        try:
            docs_payload = _load_docs_payload(indexdir)
            chunk_count = int(docs_payload.get("total_chunks", 0))
        except Exception:
            chunk_count = None

        profile_version = str(selected_profile.get("profile_version", "v1"))
        retrieval_event = {
            "event": "RETRIEVAL_DECISION",
            "query": args.query,
            "retrieval": retrieval_mode,
            "base_profile": base_profile_name,
            "effective_profile": effective_profile_name,
            "profile_version": profile_version,
            "department": args.department or "none",
            "doc_type_inventory": doc_type_counts or {},
            "chunk_count": chunk_count,
            "allowed_doc_types": allowed_doc_types_display or [],
            "topk": topk,
            "min_score": min_score,
            "min_sim": args.min_sim if retrieval_mode == "embedding" else None,
            "embedding_max_score": embedding_max_score if retrieval_mode == "embedding" else None,
            "retrieved": len(results),
            "used_for_context": len(filtered),
        }
        log_best_effort_event(indexdir.parent, retrieval_event)
    except Exception:
        pass

    print("Top-K Sources:")
    if retrieval_mode == "embedding":
        print(f"Retrieval min_sim: {args.min_sim}")
    print(f"- Retrieved: {len(results)}, Used_for_context: {len(filtered)}")
    if not results:
        print("- No results.")
    else:
        for idx, item in enumerate(results, start=1):
            score = float(item.get("score", 0.0))
            print(
                f"- {idx}. score={score:.4f} | doc_name={item.get('doc_name', '')} "
                f"| page={item.get('page', 0)} | chunk_id={item.get('chunk_id', '')}"
            )
    if effective_profile_name.strip().lower() == "lab" and len(results) > 0 and len(filtered) == 0:
        debug_scores = [round(float(item.get("score", 0.0)), 4) for item in results[:3]]
        debug_doc_names = [str(item.get("doc_name", "")) for item in results[:3]]
        first_text_len = len(str(results[0].get("text_masked", "")))
        print(f"LAB_DEBUG: retrieved={len(results)} used_for_context={len(filtered)}")
        print(f"LAB_DEBUG: scores={debug_scores}")
        print(f"LAB_DEBUG: doc_names={debug_doc_names}")
        print(f"LAB_DEBUG: first_text_len={first_text_len}")
    _print_ocr_used(indexdir, filtered)

    is_patient_card_query = _is_patient_card_question(args.query)
    is_patient_name_query = _is_patient_name_question(args.query)
    is_metadata_query = is_patient_card_query or is_patient_name_query

    metadata_results: list[dict] = []
    metadata_context = ""
    if is_metadata_query:
        source_pool = filtered if filtered else results
        metadata_results = _prioritize_results_for_metadata(indexdir, source_pool)
        metadata_context = _build_context(metadata_results)

    if not filtered:
        if is_patient_card_query:
            print()
            print("Answer:")
            payload = _build_patient_card_payload(metadata_context, metadata_results)
            print(json.dumps(payload, ensure_ascii=False))
            return 0
        if is_patient_name_query:
            print()
            print("Answer:")
            extracted_name = _extract_patient_name_from_context(metadata_context)
            if extracted_name:
                print(f"{extracted_name}\n\n{_format_sources_block(metadata_results)}")
            else:
                print("Belgede bilgi yok.")
            return 0

        print()
        profile_for_message = effective_profile_name
        if profile_for_message not in {"radiology", "epicrisis", "endoscopy", "pathology", "surgery"}:
            profile_for_message = base_profile_name

        if profile_for_message == "radiology":
            print("Radyoloji/MR/BT raporu için index'te ilgili içerik bulunamadı.")
            print("İpucu: Radiology PDF'i ekleyip ingest'i tekrar çalıştırın.")
        elif profile_for_message == "epicrisis":
            print("Epikriz dokümanı için index'te ilgili içerik bulunamadı.")
        elif profile_for_message == "endoscopy":
            print("Endoskopi raporu için index'te ilgili içerik bulunamadı.")
        elif profile_for_message == "pathology":
            print("Patoloji raporu için index'te ilgili içerik bulunamadı.")
        elif profile_for_message == "surgery":
            print("Cerrahi/ameliyat raporu için index'te ilgili içerik bulunamadı.")
        else:
            print("No relevant context found.")

        inventory = _index_inventory(indexdir)
        if inventory is not None:
            total_docs, ocr_used_docs = inventory
            print(f"Index inventory: total_docs={total_docs}, ocr_used_docs={ocr_used_docs}")
        print()
        print("Answer:")
        print("Belgede bilgi yok.")
        return 0

    context_for_answer = metadata_context if is_metadata_query else context
    sources_for_answer = metadata_results if is_metadata_query else filtered

    print()
    print("Context Preview:")
    print(_normalize_preview(context_for_answer, max_chars=800))

    print()
    print("Answer:")
    if is_patient_card_query:
        payload = _build_patient_card_payload(context_for_answer, sources_for_answer)
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    if is_patient_name_query:
        extracted_name = _extract_patient_name_from_context(context_for_answer)
        if extracted_name:
            print(f"{extracted_name}\n\n{_format_sources_block(sources_for_answer)}")
        else:
            print("Belgede bilgi yok.")
        return 0

    if args.mode == "stub":
        print("LLM disabled (stub mode).")
        return 0

    prompt_rules = [str(rule) for rule in selected_profile.get("prompt_rules", [])]
    prompt = _guarded_prompt(
        args.query,
        context_for_answer,
        prompt_rules=prompt_rules,
        task_instruction=task_instruction,
    )
    try:
        answer = _run_ollama(args.model, prompt)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    normalized_answer = answer.strip()
    is_no_info = normalized_answer == "Belgede bilgi yok."
    if not is_no_info and "Kaynaklar:" not in normalized_answer:
        answer = f"{normalized_answer}\n\n{_format_sources_block(sources_for_answer)}"

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
