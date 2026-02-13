#!/usr/bin/env bash
set -u
set -o pipefail

STATUS="PASS"
WARN_COUNT=0
FAIL_COUNT=0

print_info() {
  printf '[INFO] %s\n' "$1"
}

print_warn() {
  printf '[WARN] %s\n' "$1"
  WARN_COUNT=$((WARN_COUNT + 1))
  if [ "$STATUS" != "FAIL" ]; then
    STATUS="WARN"
  fi
}

print_fail() {
  printf '[FAIL] %s\n' "$1"
  FAIL_COUNT=$((FAIL_COUNT + 1))
  STATUS="FAIL"
}

print_section() {
  printf '\n== %s ==\n' "$1"
}

finalize() {
  print_section "SONUC"
  printf 'STATUS: %s\n' "$STATUS"
  printf 'WARN_COUNT: %s\n' "$WARN_COUNT"
  printf 'FAIL_COUNT: %s\n' "$FAIL_COUNT"

  if [ "$STATUS" = "FAIL" ]; then
    exit 1
  fi
  exit 0
}

print_section "RAG DIAGNOSE"
printf 'Repo: %s\n' "$(pwd)"

# 1) Repo root check
print_section "1) Repo Root Kontrolu"
if [ ! -d backend ]; then
  print_fail "Bu dizin repo root gibi gorunmuyor (backend/ bulunamadi)."
  printf 'Cozum: cd /home/bim/Desktop/doktor-asistan\n'
  finalize
fi
print_info "backend/ bulundu."

# 2) Python venv check
print_section "2) Python/Venv Kontrolu"
PYTHON_BIN=".venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  print_fail "$PYTHON_BIN bulunamadi veya calistirilabilir degil."
  printf 'Cozum: python3 -m venv .venv && source .venv/bin/activate\n'
  finalize
fi
print_info "$PYTHON_BIN bulundu."

# 3) Artifacts check
print_section "3) Ingest/Index Artifacts Kontrolu"
CHUNKS_PATH="out/chunks.jsonl"
INDEX_DIR="out/index"
VECTORIZER_PATH="$INDEX_DIR/vectorizer.pkl"
MATRIX_PATH="$INDEX_DIR/matrix.npz"
META_PATH="$INDEX_DIR/meta.jsonl"

if [ ! -f "$CHUNKS_PATH" ]; then
  print_fail "$CHUNKS_PATH bulunamadi."
fi

if [ ! -d "$INDEX_DIR" ]; then
  print_fail "$INDEX_DIR dizini bulunamadi."
fi

if [ -d "$INDEX_DIR" ]; then
  if [ ! -f "$VECTORIZER_PATH" ] || [ ! -f "$MATRIX_PATH" ] || [ ! -f "$META_PATH" ]; then
    print_warn "Index dizini var ama temel dosyalardan biri eksik (vectorizer.pkl/matrix.npz/meta.jsonl)."
  else
    print_info "Index dosyalari bulundu (vectorizer.pkl, matrix.npz, meta.jsonl)."
  fi
fi

if [ ! -f "$CHUNKS_PATH" ] || [ ! -d "$INDEX_DIR" ]; then
  printf 'Cozum (tek komut):\n'
  printf '  .venv/bin/python -m backend.ingest_cli --input data/pdfs --outdir out --build-index --indexdir out/index\n'
fi

# If chunks missing, skip content-based checks but still give recommendations.
if [ ! -f "$CHUNKS_PATH" ]; then
  print_warn "chunks.jsonl olmadigi icin metin/garbage kontrolleri atlandi."
  print_section "6) Onerilen Komutlar"
  printf '.venv/bin/python -m backend.answer_cli --indexdir out/index --query "patoloji raporunda tani nedir?" --topk 5 --min-score 0 --mode stub\n'
  printf '.venv/bin/python -m backend.answer_cli --indexdir out/index --query "patoloji raporunda tani nedir?" --topk 5 --min-score 0 --mode ollama --model qwen2.5:7b\n'
  finalize
fi

# 4) Diagnosis term check
print_section "4) Tani Metni Kontrolu"
if command -v rg >/dev/null 2>&1; then
  DIAG_HITS="$(rg -n -m 20 -e 'TANI|TAN[Iİ]|DIAG|DIAGNOS' "$CHUNKS_PATH" || true)"
else
  DIAG_HITS="$(grep -nEi 'TANI|TAN[Iİ]|DIAG|DIAGNOS' "$CHUNKS_PATH" | head -n 20 || true)"
fi

if [ -z "$DIAG_HITS" ]; then
  print_warn "Tanisal anahtar kelime eslesmesi yok. PDF extraction bozuk olabilir."
else
  print_info "Tanisal anahtar kelimeler bulundu (ilk eslesmeler):"
  printf '%s\n' "$DIAG_HITS"
fi

# 5) Gibberish check
print_section "5) Gibberish Kontrolu"
if command -v rg >/dev/null 2>&1; then
  GIBBERISH_HITS="$(rg -n -m 20 -e '\+DVWD' -e '\\\\{4,}' -e '[A-Z0-9+/]{40,}' "$CHUNKS_PATH" || true)"
else
  GIBBERISH_HITS="$(grep -nE '\+DVWD|\\\\\\\\{4,}|[A-Z0-9+/]{40,}' "$CHUNKS_PATH" | head -n 20 || true)"
fi

if [ -n "$GIBBERISH_HITS" ]; then
  print_warn "Gibberish/garip pattern bulundu; font/glyph mapping sorunu olabilir."
  printf '%s\n' "$GIBBERISH_HITS"
else
  print_info "Belirgin gibberish pattern bulunamadi."
fi

# 6) Suggested commands
print_section "6) Onerilen Komutlar"
printf '.venv/bin/python -m backend.answer_cli --indexdir out/index --query "patoloji raporunda tani nedir?" --topk 5 --min-score 0 --mode stub\n'
printf '.venv/bin/python -m backend.answer_cli --indexdir out/index --query "patoloji raporunda tani nedir?" --topk 5 --min-score 0 --mode ollama --model qwen2.5:7b\n'

finalize
