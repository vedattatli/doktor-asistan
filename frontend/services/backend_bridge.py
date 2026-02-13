from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "out"
INDEX_DIR = OUT_DIR / "index"
CHUNKS_PATH = OUT_DIR / "chunks.jsonl"
PDF_INPUT_DIR = REPO_ROOT / "data" / "pdfs"
INDEX_REQUIRED_FILES = ("vectorizer.pkl", "matrix.npz", "meta.jsonl")
LAB_TOKENS = {
    "HGB",
    "WBC",
    "RBC",
    "PLT",
    "CRP",
    "TSH",
    "GLU",
    "GLUKOZ",
    "AST",
    "ALT",
    "LDL",
    "HDL",
    "NA",
    "K",
}
META_KEYWORDS = [
    "ad",
    "soyad",
    "isim",
    "doğum",
    "cinsiyet",
    "protokol",
    "dosya",
    "istem",
    "rapor tarihi",
    "poliklinik",
    "klinik",
    "doktor",
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


def get_python_path() -> str:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def run_command(cmd: list[str]) -> str:
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(detail or f"Command failed with exit code {completed.returncode}")
    return completed.stdout


def _missing_index_files() -> list[str]:
    missing: list[str] = []
    for name in INDEX_REQUIRED_FILES:
        path = INDEX_DIR / name
        if not path.exists():
            missing.append(str(path))
    return missing


def _ensure_index_ready() -> None:
    if not _missing_index_files():
        return

    if CHUNKS_PATH.exists():
        build_index()
    elif PDF_INPUT_DIR.exists():
        ingest_and_build_index()

    missing = _missing_index_files()
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            "Error: index files not found. Önce build et:\n"
            f"  {get_python_path()} -m backend.retrieve_cli build --chunks out/chunks.jsonl --indexdir out/index\n"
            f"Missing: Missing index file(s): {missing_str}"
        )


def _parse_answer_block(stdout: str) -> str:
    marker = "Answer:"
    if marker not in stdout:
        if "No relevant context found." in stdout:
            return "Belgede açık ifade yok."
        return stdout.strip() or "Yanıt bulunamadı."
    return stdout.split(marker, 1)[1].strip() or "Yanıt bulunamadı."


def _detect_intent(query: str) -> str:
    q = (query or "").strip()
    up = q.upper()
    low = q.lower()
    query_tokens = set(re.findall(r"\b[A-Z0-9ÇĞİÖŞÜ]+\b", up))

    if any(token in query_tokens for token in LAB_TOKENS):
        return "lab"
    if any(keyword in low for keyword in META_KEYWORDS):
        return "meta"
    return "other"


def _detect_query_mode(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "smalltalk"

    up = q.upper()
    low = q.lower()
    tokens = set(re.findall(r"\b[0-9A-ZÇĞİÖŞÜa-zçğıöşü]+\b", q))
    lower_tokens = {t.lower() for t in tokens}
    query_tokens = set(re.findall(r"\b[A-Z0-9ÇĞİÖŞÜ]+\b", up))

    if any(token in query_tokens for token in LAB_TOKENS):
        return "document_qa"
    if any(keyword in lower_tokens for keyword in DOC_QUERY_TOKEN_KEYWORDS):
        return "document_qa"
    if any(phrase in low for phrase in DOC_QUERY_PHRASE_KEYWORDS):
        return "document_qa"
    return "smalltalk"


def _smalltalk_reply(query: str) -> str:
    q = (query or "").strip().lower()
    if any(greet in q for greet in SMALLTALK_GREETINGS):
        return "Selam! İstersen PDF’den hasta adı/özet/lab anormallikleri çıkarayım. Ne arıyorsun?"
    if any(p in q for p in SMALLTALK_WHAT_ARE_YOU_DOING):
        return "Buradayım. PDF yüklersen özet, kronoloji, değer çıkarma yapabilirim."
    return "Buradayım. PDF yüklersen özet, kronoloji, değer çıkarma yapabilirim."


def detect_query_mode(query: str) -> str:
    return _detect_query_mode(query)


def ask_question(
    query: str,
    task_instruction: str | None = None,
    auto_build_if_missing: bool = True,
) -> dict[str, Any]:
    task_instruction_clean = str(task_instruction or "").strip()
    query_mode = "document_qa" if task_instruction_clean else _detect_query_mode(query)
    if query_mode == "smalltalk":
        answer = _smalltalk_reply(query)
        return {
            "answer": answer,
            "raw": f"Selected query mode: smalltalk\n\nAnswer:\n{answer}",
            "intent": "smalltalk",
            "query_mode": query_mode,
            "index_missing": False,
        }

    try:
        intent_query = query
        if task_instruction_clean:
            intent_query = f"{query}\n{task_instruction_clean}"
        intent = _detect_intent(intent_query)
        missing = _missing_index_files()
        if missing and not auto_build_if_missing:
            return {
                "answer": "Önce index oluştur.",
                "raw": (
                    "Error: index files not found. Önce build et:\n"
                    f"Missing: Missing index file(s): {', '.join(missing)}"
                ),
                "intent": intent,
                "query_mode": query_mode,
                "index_missing": True,
            }
        _ensure_index_ready()
        profile = "auto"
        rewritten_query = query

        if intent == "lab":
            profile = "lab"
        elif intent == "meta":
            profile = "auto"
            rewritten_query = (
                query
                + "\nLütfen raporun üst kısmındaki kimlik/başlık alanından aynen çıkar: "
                "Ad Soyad, Doğum Tarihi, Cinsiyet, Protokol/Dosya/İstem No, "
                "Rapor Tarihi, Poliklinik/Klinik."
            )

        cmd = [
            get_python_path(),
            "-m",
            "backend.answer_cli",
            "--indexdir",
            "out/index",
            "--profile",
            profile,
            "--mode",
            "ollama",
            "--model",
            "qwen2.5:7b",
            "--retrieval",
            "tfidf",
            "--query",
            rewritten_query,
        ]
        if task_instruction_clean:
            cmd.extend(["--task-instruction", task_instruction_clean])
        raw = run_command(cmd)
    except Exception as exc:  # best-effort for UI
        intent_query = query
        if task_instruction_clean:
            intent_query = f"{query}\n{task_instruction_clean}"
        intent = _detect_intent(intent_query)
        raw = str(exc)
    return {
        "answer": _parse_answer_block(raw),
        "raw": raw,
        "intent": intent,
        "query_mode": query_mode,
        "index_missing": False,
    }


def run_ingest() -> str:
    cmd = [
        get_python_path(),
        "-m",
        "backend.ingest_cli",
        "--input",
        "data/pdfs",
        "--outdir",
        "out",
    ]
    return run_command(cmd)


def build_index() -> str:
    cmd = [
        get_python_path(),
        "-m",
        "backend.retrieve_cli",
        "build",
        "--chunks",
        "out/chunks.jsonl",
        "--indexdir",
        "out/index",
    ]
    return run_command(cmd)


def ingest_and_build_index() -> str:
    ingest_output = run_ingest()
    build_output = build_index()
    return (
        "### Ingest Output ###\n"
        f"{ingest_output.strip()}\n\n"
        "### Build Output ###\n"
        f"{build_output.strip()}\n"
    )


def get_index_status() -> dict[str, Any]:
    missing = _missing_index_files()
    return {
        "ready": len(missing) == 0,
        "missing_files": missing,
        "missing_count": len(missing),
    }


def get_loaded_documents(limit: int = 200) -> list[dict[str, Any]]:
    docs_json = OUT_DIR / "docs.json"
    max_items = max(1, int(limit))
    merged: dict[str, dict[str, Any]] = {}

    if PDF_INPUT_DIR.exists():
        for path in sorted(PDF_INPUT_DIR.glob("*.pdf")):
            merged[path.name] = {
                "doc_name": path.name,
                "doc_type": "unknown",
                "total_pages": 0,
                "total_chunks": 0,
            }

    if docs_json.exists():
        try:
            with docs_json.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            docs = payload.get("docs", [])
            if isinstance(docs, list):
                for item in docs:
                    if not isinstance(item, dict):
                        continue
                    doc_name = str(item.get("doc_name", "")).strip()
                    if not doc_name:
                        continue
                    merged[doc_name] = {
                        "doc_name": doc_name,
                        "doc_type": str(item.get("doc_type", "general")).strip().lower() or "general",
                        "total_pages": int(item.get("total_pages", 0) or 0),
                        "total_chunks": int(item.get("total_chunks", 0) or 0),
                    }
        except Exception:
            pass

    ordered_names = sorted(merged.keys())
    return [merged[name] for name in ordered_names[:max_items]]


def get_docs_stats() -> dict[str, Any]:
    docs_json = OUT_DIR / "docs.json"
    if not docs_json.exists():
        return {"total_docs": 0, "doc_type_counts": {}, "ocr_used_docs": 0}

    with docs_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    docs = payload.get("docs", [])
    if not isinstance(docs, list):
        return {"total_docs": 0, "doc_type_counts": {}, "ocr_used_docs": 0}

    doc_type_counts: Counter[str] = Counter()
    ocr_used_docs = 0
    for item in docs:
        if not isinstance(item, dict):
            continue
        doc_type = str(item.get("doc_type", "")).strip().lower() or "general"
        doc_type_counts[doc_type] += 1
        if bool(item.get("ocr_used", False)):
            ocr_used_docs += 1

    return {
        "total_docs": len(docs),
        "doc_type_counts": dict(doc_type_counts),
        "ocr_used_docs": ocr_used_docs,
    }


def get_audit_logs(limit: int = 50) -> list[str]:
    audit_jsonl = OUT_DIR / "audit.jsonl"
    if not audit_jsonl.exists():
        return []

    with audit_jsonl.open("r", encoding="utf-8") as f:
        lines = deque((line.rstrip("\n") for line in f if line.strip()), maxlen=limit)
    return list(lines)
