#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PDF_DIR="data/pdfs"
OUT_DIR="out"
QUERY=""
MODE="stub"

print_usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_rag.sh --pdf-dir <DIR> --out-dir <DIR> --query "<QUESTION>" --mode <stub|ollama>

Required args:
  --query      Question text
  --mode       stub or ollama

Optional args:
  --pdf-dir    Default: data/pdfs
  --out-dir    Default: out
  -h, --help   Show this help

Examples:
  bash scripts/run_rag.sh --pdf-dir data/pdfs --out-dir out --query "patoloji raporunda tanı nedir?" --mode stub
  bash scripts/run_rag.sh --pdf-dir data/pdfs --out-dir out --query "patoloji raporunda tanı nedir?" --mode ollama
USAGE
}

log_info() {
  printf '[INFO] %s\n' "$1"
}

log_warn() {
  printf '[WARN] %s\n' "$1"
}

log_error() {
  printf '[ERROR] %s\n' "$1" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --pdf-dir)
      [ "$#" -ge 2 ] || { log_error "--pdf-dir requires a value"; print_usage; exit 1; }
      PDF_DIR="$2"
      shift 2
      ;;
    --out-dir)
      [ "$#" -ge 2 ] || { log_error "--out-dir requires a value"; print_usage; exit 1; }
      OUT_DIR="$2"
      shift 2
      ;;
    --query)
      [ "$#" -ge 2 ] || { log_error "--query requires a value"; print_usage; exit 1; }
      QUERY="$2"
      shift 2
      ;;
    --mode)
      [ "$#" -ge 2 ] || { log_error "--mode requires a value"; print_usage; exit 1; }
      MODE="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      log_error "Unknown argument: $1"
      print_usage
      exit 1
      ;;
  esac
done

if [ -z "$QUERY" ]; then
  log_error "--query is required"
  print_usage
  exit 1
fi

if [ "$MODE" != "stub" ] && [ "$MODE" != "ollama" ]; then
  log_error "--mode must be 'stub' or 'ollama'"
  exit 1
fi

cd "$REPO_ROOT"

if [ ! -d "backend" ]; then
  log_error "Repo root validation failed: backend/ not found at ${REPO_ROOT}"
  exit 1
fi

PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  log_error "Python not found at ${PYTHON_BIN}"
  log_error "Create venv first: python3 -m venv .venv"
  exit 1
fi

if [ "${VIRTUAL_ENV:-}" != "${REPO_ROOT}/.venv" ]; then
  if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
    log_info "Activated virtualenv at ${REPO_ROOT}/.venv"
  else
    log_warn "Virtualenv activate script not found; continuing with explicit ${PYTHON_BIN}"
  fi
fi

PDF_DIR_ABS="$(realpath -m "$PDF_DIR")"
OUT_DIR_ABS="$(realpath -m "$OUT_DIR")"
INDEX_DIR="${OUT_DIR_ABS}/index"
CHUNKS_PATH="${OUT_DIR_ABS}/chunks.jsonl"

if [ ! -d "$PDF_DIR_ABS" ]; then
  log_error "PDF directory not found: ${PDF_DIR_ABS}"
  exit 1
fi

log_info "Repo root: ${REPO_ROOT}"
log_info "PDF dir: ${PDF_DIR_ABS}"
log_info "Out dir: ${OUT_DIR_ABS}"
log_info "Mode: ${MODE}"

log_info "Step 1/2: ingest + index build"
"$PYTHON_BIN" -m backend.ingest_cli \
  --input "$PDF_DIR_ABS" \
  --outdir "$OUT_DIR_ABS" \
  --build-index \
  --indexdir "$INDEX_DIR"
INGEST_EXIT=$?
if [ "$INGEST_EXIT" -ne 0 ]; then
  log_error "ingest_cli failed with exit code ${INGEST_EXIT}"
  exit "$INGEST_EXIT"
fi

required_index_files=(
  "${INDEX_DIR}/vectorizer.pkl"
  "${INDEX_DIR}/matrix.npz"
  "${INDEX_DIR}/meta.jsonl"
)

missing=()
for f in "${required_index_files[@]}"; do
  if [ ! -f "$f" ]; then
    missing+=("$f")
  fi
done

if [ "${#missing[@]}" -gt 0 ]; then
  log_warn "Index files are missing after ingest. Rebuilding index automatically."
  printf 'Missing:\n'
  printf '  %s\n' "${missing[@]}"

  if [ ! -f "$CHUNKS_PATH" ]; then
    log_error "chunks.jsonl not found: ${CHUNKS_PATH}"
    exit 1
  fi

  "$PYTHON_BIN" -m backend.retrieve_cli build --chunks "$CHUNKS_PATH" --indexdir "$INDEX_DIR"
  BUILD_EXIT=$?
  if [ "$BUILD_EXIT" -ne 0 ]; then
    log_error "retrieve_cli build failed with exit code ${BUILD_EXIT}"
    exit "$BUILD_EXIT"
  fi
fi

for f in "${required_index_files[@]}"; do
  if [ ! -f "$f" ]; then
    log_error "Index still missing after recovery: ${f}"
    exit 1
  fi
done

log_info "Step 2/2: answer query"
ANSWER_LOG="$(mktemp)"
trap 'rm -f "${ANSWER_LOG}"' EXIT

if [ "$MODE" = "ollama" ]; then
  "$PYTHON_BIN" -m backend.answer_cli \
    --indexdir "$INDEX_DIR" \
    --query "$QUERY" \
    --topk 5 \
    --mode ollama \
    --model qwen2.5:7b | tee "$ANSWER_LOG"
  ANSWER_EXIT=${PIPESTATUS[0]}
else
  "$PYTHON_BIN" -m backend.answer_cli \
    --indexdir "$INDEX_DIR" \
    --query "$QUERY" \
    --topk 5 \
    --mode stub | tee "$ANSWER_LOG"
  ANSWER_EXIT=${PIPESTATUS[0]}
fi

if [ "$ANSWER_EXIT" -ne 0 ]; then
  log_error "answer_cli failed with exit code ${ANSWER_EXIT}"
  exit "$ANSWER_EXIT"
fi

printf '\n=== Source Chunk IDs ===\n'
if command -v rg >/dev/null 2>&1; then
  rg -o 'chunk_id=[0-9a-fA-F-]{36}' "$ANSWER_LOG" | sed 's/chunk_id=//' | awk '!seen[$0]++' || true
else
  grep -oE 'chunk_id=[0-9a-fA-F-]{36}' "$ANSWER_LOG" | sed 's/chunk_id=//' | awk '!seen[$0]++' || true
fi

printf '\n=== Final Answer ===\n'
awk 'found{print} /^Answer:/{found=1}' "$ANSWER_LOG"

log_info "Pipeline completed successfully."
