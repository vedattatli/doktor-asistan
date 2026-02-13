from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, TypedDict
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ChunkMeta(TypedDict):
    doc_name: str
    page: int
    chunk_id: str
    text_masked: str


class BuildInfo(TypedDict):
    chunk_count: int
    skipped_garbage: int
    garbage_samples_path: str
    vocab_size: int
    matrix_rows: int
    matrix_cols: int
    vectorizer_path: str
    matrix_path: str
    meta_path: str


class EmbeddingBuildInfo(TypedDict):
    chunk_count: int
    skipped_garbage: int
    embedding_model: str
    embedding_dim: int
    embeddings_path: str
    embeddings_meta_path: str


class SearchResult(TypedDict):
    score: float
    doc_name: str
    page: int
    chunk_id: str
    text_masked: str


MIN_TEXT_LEN = 40
MIN_ALNUM_RATIO = 0.35
REPLACEMENT_CHAR = "�"
WEIRD_RATIO_THRESHOLD = 0.18
ALLOWED_WEIRD_PUNCT = ".,;:!?()[]{}%+-/\\'\""
ENCODING_GARBAGE_MIN_LEN = 80
NON_ASCII_RATIO_THRESHOLD = 0.12
TR_LETTER_RATIO_THRESHOLD = 0.08
ENCODING_CONTROL_RATIO_THRESHOLD = 0.02
ENCODING_TR_SOFT_RATIO_THRESHOLD = 0.50
ALLOWED_TR = set("abcçdefgğhıijklmnoöprsştuüvyz" + "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ")
HIGH_WEIRD_CONTROL_RATIO_THRESHOLD = 0.01
HIGH_WEIRD_NON_ASCII_RATIO_THRESHOLD = 0.25
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_DEFAULT_HOST = "http://127.0.0.1:11434"


def _alnum_ratio(text: str) -> float:
    letters_digits = sum(1 for ch in text if ch.isalnum())
    return letters_digits / max(1, len(text))


def garbage_reason(text: str) -> str | None:
    if not text or not text.strip():
        return "empty"
    if len(text) < MIN_TEXT_LEN:
        return "too_short"

    text_len = max(1, len(text))
    ratio = _alnum_ratio(text)
    if REPLACEMENT_CHAR in text:
        return "replacement_char"

    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    non_ascii_ratio = non_ascii / text_len
    tr_letters = sum(1 for ch in text if ch in ALLOWED_TR)
    tr_letter_ratio = tr_letters / text_len
    control_chars = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\t\r")
    control_ratio = control_chars / text_len
    if (
        len(text) >= ENCODING_GARBAGE_MIN_LEN
        and non_ascii_ratio > NON_ASCII_RATIO_THRESHOLD
        and tr_letter_ratio < TR_LETTER_RATIO_THRESHOLD
    ):
        return "encoding_garbage"

    # Some PDF extraction artifacts keep non-ASCII low but still look like encoded noise.
    if (
        len(text) >= ENCODING_GARBAGE_MIN_LEN
        and control_ratio > ENCODING_CONTROL_RATIO_THRESHOLD
        and tr_letter_ratio < ENCODING_TR_SOFT_RATIO_THRESHOLD
    ):
        return "encoding_garbage"

    weird_chars = sum(
        1
        for ch in text
        if not (ch.isalnum() or ch.isspace() or ch in ALLOWED_WEIRD_PUNCT)
    )
    weird_ratio = weird_chars / text_len
    if weird_ratio > WEIRD_RATIO_THRESHOLD and (
        control_ratio > HIGH_WEIRD_CONTROL_RATIO_THRESHOLD
        or non_ascii_ratio > HIGH_WEIRD_NON_ASCII_RATIO_THRESHOLD
    ):
        return "high_weird_ratio"

    if ratio < MIN_ALNUM_RATIO:
        return "low_alnum_ratio"

    return None


def is_garbage_text(text: str) -> bool:
    return garbage_reason(text) is not None


def _index_paths(indexdir: Path) -> tuple[Path, Path, Path]:
    return indexdir / "vectorizer.pkl", indexdir / "matrix.npz", indexdir / "meta.jsonl"


def _embedding_index_paths(indexdir: Path) -> tuple[Path, Path]:
    return indexdir / "embeddings.npy", indexdir / "embeddings_meta.json"


def _parse_chunk_line(line: str, line_no: int, source: Path) -> ChunkMeta | None:
    stripped = line.strip()
    if not stripped:
        return None

    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {source}:{line_no}: {exc}") from exc

    if not isinstance(obj, dict):
        raise ValueError(f"Invalid row type in {source}:{line_no}: expected object")

    try:
        page = int(obj.get("page", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid page value in {source}:{line_no}") from exc

    return {
        "doc_name": str(obj.get("doc_name", "")),
        "page": page,
        "chunk_id": str(obj.get("chunk_id", "")),
        "text_masked": str(obj.get("text_masked", "")),
    }


def _normalize_chunk_obj(obj: dict[str, Any]) -> ChunkMeta:
    try:
        page = int(obj.get("page", 0))
    except (TypeError, ValueError):
        page = 0
    return {
        "doc_name": str(obj.get("doc_name", "")),
        "page": page,
        "chunk_id": str(obj.get("chunk_id", "")),
        "text_masked": str(obj.get("text_masked", "")),
    }


def load_chunks_from_jsonl(chunks_path: str | Path) -> list[ChunkMeta]:
    chunks_file = Path(chunks_path).expanduser()
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
    if not chunks_file.is_file():
        raise ValueError(f"Chunks path must be a file: {chunks_file}")

    chunks: list[ChunkMeta] = []
    with chunks_file.open("r", encoding="utf-8") as in_f:
        for line_no, line in enumerate(in_f, start=1):
            meta = _parse_chunk_line(line, line_no, chunks_file)
            if meta is None:
                continue
            chunks.append(meta)
    return chunks


def _ollama_host() -> str:
    host = os.getenv("OLLAMA_HOST", OLLAMA_DEFAULT_HOST).strip()
    if not host:
        return OLLAMA_DEFAULT_HOST
    if not host.startswith("http://") and not host.startswith("https://"):
        host = "http://" + host
    return host.rstrip("/")


def _post_ollama_json(api_path: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(
        url=f"{_ollama_host()}{api_path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError) as exc:
        raise RuntimeError(f"Ollama embed request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama embed response is not valid JSON") from exc


def _embeddings_via_embed_api(texts: list[str], model: str) -> np.ndarray:
    response = _post_ollama_json("/api/embed", {"model": model, "input": texts})
    embeddings = response.get("embeddings")
    if not isinstance(embeddings, list) or not embeddings:
        raise RuntimeError("Ollama /api/embed returned invalid embeddings payload")

    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise RuntimeError("Ollama /api/embed returned non-matrix embeddings")
    return matrix


def _embeddings_via_legacy_api(texts: list[str], model: str) -> np.ndarray:
    vectors: list[list[float]] = []
    for text in texts:
        response = _post_ollama_json("/api/embeddings", {"model": model, "prompt": text})
        embedding = response.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError("Ollama /api/embeddings returned invalid embedding payload")
        vectors.append([float(item) for item in embedding])

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise RuntimeError("Ollama /api/embeddings returned non-matrix embeddings")
    return matrix


def _embed_texts_with_ollama(texts: list[str], model: str = OLLAMA_EMBED_MODEL) -> np.ndarray:
    if not texts:
        raise ValueError("texts must not be empty")
    try:
        return _embeddings_via_embed_api(texts, model)
    except RuntimeError:
        return _embeddings_via_legacy_api(texts, model)


def build_embedding_index(chunks: list[dict[str, Any]], indexdir: Path | str) -> EmbeddingBuildInfo:
    index_path = Path(indexdir).expanduser()
    index_path.mkdir(parents=True, exist_ok=True)
    embeddings_path, embeddings_meta_path = _embedding_index_paths(index_path)

    filtered: list[ChunkMeta] = []
    skipped_garbage = 0
    for raw_chunk in chunks:
        if not isinstance(raw_chunk, dict):
            continue
        meta = _normalize_chunk_obj(raw_chunk)
        if garbage_reason(meta["text_masked"]) is not None:
            skipped_garbage += 1
            continue
        filtered.append(meta)

    if not filtered:
        raise ValueError(
            f"No usable chunk rows found for embedding index after filtering "
            f"(skipped_garbage={skipped_garbage})"
        )

    text_rows = [item["text_masked"] for item in filtered]
    embeddings = _embed_texts_with_ollama(text_rows, model=OLLAMA_EMBED_MODEL)
    if embeddings.shape[0] != len(filtered):
        raise ValueError(
            "Embedding row count mismatch: "
            f"embeddings={embeddings.shape[0]} filtered_chunks={len(filtered)}"
        )

    np.save(embeddings_path, embeddings.astype(np.float32))
    payload = {
        "model": OLLAMA_EMBED_MODEL,
        "chunk_count": len(filtered),
        "chunk_ids": [item["chunk_id"] for item in filtered],
        "items": filtered,
    }
    with embeddings_meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return {
        "chunk_count": len(filtered),
        "skipped_garbage": skipped_garbage,
        "embedding_model": OLLAMA_EMBED_MODEL,
        "embedding_dim": int(embeddings.shape[1]),
        "embeddings_path": str(embeddings_path),
        "embeddings_meta_path": str(embeddings_meta_path),
    }


def build_index(chunks_path: str | Path, indexdir: str | Path) -> BuildInfo:
    chunks_file = Path(chunks_path).expanduser()
    index_path = Path(indexdir).expanduser()

    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
    if not chunks_file.is_file():
        raise ValueError(f"Chunks path must be a file: {chunks_file}")

    index_path.mkdir(parents=True, exist_ok=True)
    vectorizer_path, matrix_path, meta_path = _index_paths(index_path)
    garbage_samples_path = index_path / "garbage_samples.jsonl"

    chunk_count = 0
    skipped_garbage = 0
    with (
        chunks_file.open("r", encoding="utf-8") as in_f,
        meta_path.open("w", encoding="utf-8") as meta_f,
        garbage_samples_path.open("a", encoding="utf-8") as garbage_f,
    ):
        for line_no, line in enumerate(in_f, start=1):
            meta = _parse_chunk_line(line, line_no, chunks_file)
            if meta is None:
                continue

            reason = garbage_reason(meta["text_masked"])
            if reason is not None:
                skipped_garbage += 1
                sample = {
                    "reason": reason,
                    "doc_name": meta["doc_name"],
                    "page": meta["page"],
                    "chunk_id": meta["chunk_id"],
                    "text_preview": meta["text_masked"][:200],
                }
                garbage_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                continue

            meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            chunk_count += 1

    if chunk_count == 0:
        raise ValueError(
            f"No usable chunk rows found in {chunks_file} after filtering "
            f"(skipped_garbage={skipped_garbage})"
        )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(_iter_texts_from_meta(meta_path))

    with vectorizer_path.open("wb") as f:
        pickle.dump(vectorizer, f)
    save_npz(matrix_path, matrix)

    print(f"skipped_garbage: {skipped_garbage}")

    return {
        "chunk_count": chunk_count,
        "skipped_garbage": skipped_garbage,
        "garbage_samples_path": str(garbage_samples_path),
        "vocab_size": len(vectorizer.vocabulary_),
        "matrix_rows": int(matrix.shape[0]),
        "matrix_cols": int(matrix.shape[1]),
        "vectorizer_path": str(vectorizer_path),
        "matrix_path": str(matrix_path),
        "meta_path": str(meta_path),
    }


def _iter_texts_from_meta(meta_path: Path):
    with meta_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parsed = _parse_chunk_line(line, line_no, meta_path)
            if parsed is None:
                continue
            yield parsed["text_masked"]


def _load_meta(meta_path: Path) -> list[ChunkMeta]:
    meta: list[ChunkMeta] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parsed = _parse_chunk_line(line, line_no, meta_path)
            if parsed is None:
                continue
            meta.append(parsed)
    return meta


def _load_index(indexdir: str | Path) -> tuple[TfidfVectorizer, csr_matrix, list[ChunkMeta]]:
    index_path = Path(indexdir).expanduser()
    vectorizer_path, matrix_path, meta_path = _index_paths(index_path)

    missing: list[str] = []
    if not vectorizer_path.exists():
        missing.append(str(vectorizer_path))
    if not matrix_path.exists():
        missing.append(str(matrix_path))
    if not meta_path.exists():
        missing.append(str(meta_path))
    if missing:
        raise FileNotFoundError("Missing index file(s): " + ", ".join(missing))

    with vectorizer_path.open("rb") as f:
        vectorizer = pickle.load(f)
    matrix = load_npz(matrix_path)
    meta = _load_meta(meta_path)

    if matrix.shape[0] != len(meta):
        raise ValueError(
            f"Index mismatch: matrix rows={matrix.shape[0]} but meta rows={len(meta)}"
        )

    return vectorizer, matrix, meta


def search_index(
    indexdir: str | Path,
    query: str,
    topk: int = 5,
    allowed_doc_types: set[str] | None = None,
) -> list[SearchResult]:
    if topk <= 0:
        raise ValueError("topk must be greater than 0")
    if not query or not query.strip():
        raise ValueError("query cannot be empty")

    vectorizer, matrix, meta = _load_index(indexdir)
    if matrix.shape[0] == 0:
        return []

    if allowed_doc_types is not None:
        allowed = {str(item).strip().lower() for item in allowed_doc_types if str(item).strip()}
        if not allowed:
            return []

        doc_types_by_name = _load_doc_types_by_name(indexdir)
        keep_indices = [
            idx
            for idx, chunk in enumerate(meta)
            if _resolve_doc_type(doc_types_by_name, chunk["doc_name"]) in allowed
        ]
        if not keep_indices:
            _print_doc_name_mismatch_debug(indexdir, meta)
            return []

        matrix = matrix[keep_indices]
        meta = [meta[idx] for idx in keep_indices]
        if matrix.shape[0] == 0:
            return []

    query_vec = vectorizer.transform([query])
    scores = np.asarray(cosine_similarity(query_vec, matrix)).ravel()

    k = min(topk, len(scores))
    if k == len(scores):
        top_indices = np.argsort(-scores)
    else:
        partition = np.argpartition(-scores, k - 1)[:k]
        top_indices = partition[np.argsort(-scores[partition])]

    results: list[SearchResult] = []
    for idx in top_indices:
        item = meta[int(idx)]
        results.append(
            {
                "score": float(scores[int(idx)]),
                "doc_name": item["doc_name"],
                "page": item["page"],
                "chunk_id": item["chunk_id"],
                "text_masked": item["text_masked"],
            }
        )

    return results


def _load_embedding_index(indexdir: str | Path) -> tuple[np.ndarray, list[ChunkMeta]]:
    index_path = Path(indexdir).expanduser()
    embeddings_path, embeddings_meta_path = _embedding_index_paths(index_path)

    missing: list[str] = []
    if not embeddings_path.exists():
        missing.append(str(embeddings_path))
    if not embeddings_meta_path.exists():
        missing.append(str(embeddings_meta_path))
    if missing:
        raise FileNotFoundError("Missing index file(s): " + ", ".join(missing))

    embeddings = np.load(embeddings_path)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.ndim != 2:
        raise ValueError(f"Invalid embeddings shape: {embeddings.shape}")

    with embeddings_meta_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        raw_items = payload.get("items", [])
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raise ValueError(f"Invalid embeddings metadata format: {embeddings_meta_path}")

    if not isinstance(raw_items, list):
        raise ValueError(f"Invalid embeddings metadata rows: {embeddings_meta_path}")

    items = [_normalize_chunk_obj(item) for item in raw_items if isinstance(item, dict)]
    if embeddings.shape[0] != len(items):
        raise ValueError(
            "Embedding index mismatch: "
            f"embeddings rows={embeddings.shape[0]} but metadata rows={len(items)}"
        )

    return embeddings.astype(np.float32), items


def _load_doc_types_by_name(indexdir: str | Path) -> dict[str, str]:
    docs_json_path = Path(indexdir).expanduser().parent / "docs.json"
    if not docs_json_path.exists() or not docs_json_path.is_file():
        return {}

    try:
        with docs_json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}

    docs = payload.get("docs", [])
    if not isinstance(docs, list):
        return {}

    def _norm(value: str) -> str:
        return value.strip().lower()

    def _basename(value: str) -> str:
        return Path(value.replace("\\", "/")).name

    mapping: dict[str, str] = {}
    for item in docs:
        if not isinstance(item, dict):
            continue
        doc_name = str(item.get("doc_name", "")).strip()
        doc_path = str(item.get("doc_path", "")).strip()
        doc_type = str(item.get("doc_type", "")).strip().lower()
        if not doc_type:
            continue

        if doc_name:
            doc_name_key = _norm(doc_name)
            if doc_name_key:
                mapping[doc_name_key] = doc_type

        if doc_path:
            base = _basename(doc_path)
            base_key = _norm(base)
            stem_base_key = _norm(Path(base).stem)
            if base_key:
                mapping[base_key] = doc_type
            if stem_base_key:
                mapping[stem_base_key] = doc_type
    return mapping


def _resolve_doc_type(doc_types_by_name: dict[str, str], chunk_doc_name: str) -> str:
    chunk_doc = str(chunk_doc_name).strip()
    if not chunk_doc:
        return ""

    normalized = chunk_doc.replace("\\", "/")
    base = Path(normalized).name
    stem_base = Path(base).stem
    candidates = [chunk_doc.lower(), base.lower(), stem_base.lower()]
    for key in candidates:
        if not key:
            continue
        doc_type = doc_types_by_name.get(key)
        if doc_type:
            return doc_type
    return ""


def _load_docs_json_doc_name_samples(indexdir: str | Path, limit: int = 5) -> list[str]:
    docs_json_path = Path(indexdir).expanduser().parent / "docs.json"
    if not docs_json_path.exists() or not docs_json_path.is_file():
        return []

    try:
        with docs_json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []

    docs = payload.get("docs", [])
    if not isinstance(docs, list):
        return []

    samples: list[str] = []
    for item in docs:
        if not isinstance(item, dict):
            continue
        name = str(item.get("doc_name", "")).strip()
        if not name:
            continue
        samples.append(name)
        if len(samples) >= limit:
            break
    return samples


def _print_doc_name_mismatch_debug(indexdir: str | Path, meta: list[ChunkMeta]) -> None:
    meta_samples = [str(item.get("doc_name", "")).strip() for item in meta[:5]]
    docs_samples = _load_docs_json_doc_name_samples(indexdir, limit=5)
    print(f"first 5 meta doc_name examples: {meta_samples}")
    print(f"first 5 docs.json doc_name examples: {docs_samples}")
    print("doc_name mismatch suspected")


def search_embedding(
    indexdir: str | Path,
    query: str,
    topk: int = 5,
    min_sim: float = 0.20,
    allowed_doc_types: set[str] | None = None,
) -> list[SearchResult]:
    if topk <= 0:
        raise ValueError("topk must be greater than 0")
    if not query or not query.strip():
        raise ValueError("query cannot be empty")
    if min_sim < 0:
        raise ValueError("min_sim must be >= 0")

    embeddings, meta = _load_embedding_index(indexdir)
    if embeddings.shape[0] == 0:
        return []

    if allowed_doc_types is not None:
        allowed = {str(item).strip().lower() for item in allowed_doc_types if str(item).strip()}
        if not allowed:
            return []

        doc_types_by_name = _load_doc_types_by_name(indexdir)
        keep_indices = [
            idx
            for idx, chunk in enumerate(meta)
            if _resolve_doc_type(doc_types_by_name, chunk["doc_name"]) in allowed
        ]
        if not keep_indices:
            _print_doc_name_mismatch_debug(indexdir, meta)
            return []

        embeddings = embeddings[keep_indices]
        meta = [meta[idx] for idx in keep_indices]
        if embeddings.shape[0] == 0:
            return []

    query_embedding = _embed_texts_with_ollama([query], model=OLLAMA_EMBED_MODEL)[0]
    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    doc_norms = np.linalg.norm(embeddings, axis=1)
    query_norm = float(np.linalg.norm(query_embedding))
    denom = np.maximum(doc_norms * max(query_norm, 1e-12), 1e-12)
    scores = np.asarray((embeddings @ query_embedding) / denom).ravel()
    best_score = float(np.max(scores)) if scores.size else 0.0
    if best_score < min_sim:
        return []

    k = min(topk, len(scores))
    if k == len(scores):
        top_indices = np.argsort(-scores)
    else:
        partition = np.argpartition(-scores, k - 1)[:k]
        top_indices = partition[np.argsort(-scores[partition])]

    results: list[SearchResult] = []
    for idx in top_indices:
        item = meta[int(idx)]
        results.append(
            {
                "score": float(scores[int(idx)]),
                "doc_name": item["doc_name"],
                "page": item["page"],
                "chunk_id": item["chunk_id"],
                "text_masked": item["text_masked"],
            }
        )
    return results
