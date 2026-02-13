#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: Python not found at $PYTHON_BIN" >&2
  exit 1
fi

cd "$ROOT_DIR"

run_step() {
  local title="$1"
  shift
  echo
  echo "==> $title"
  "$@"
}

echo
echo "==> 1) ingest"
INGEST_LOG="$(mktemp)"
trap 'rm -f "$INGEST_LOG"' EXIT
"$PYTHON_BIN" -m backend.ingest_cli --input data/pdfs --outdir out --build-index --indexdir out/index | tee "$INGEST_LOG"

FOUND_COUNT="$(grep -Eo 'Found[[:space:]]+[0-9]+[[:space:]]+PDF file\(s\)\.' "$INGEST_LOG" | tail -n1 | grep -Eo '[0-9]+' || true)"
if ! [[ "$FOUND_COUNT" =~ ^[0-9]+$ ]]; then
  echo "Error: ingest output does not include a valid 'Found X PDF file(s).' line" >&2
  exit 1
fi
echo "found_pdfs: $FOUND_COUNT"

echo
echo "==> 2) docs count"
DOC_COUNT="$("$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

docs_path = Path("out/docs.json")
if not docs_path.exists():
    raise SystemExit(1)

with docs_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

print(len(data.get("docs", [])))
PY
)"
echo "docs: $DOC_COUNT"
if ! [[ "$DOC_COUNT" =~ ^[0-9]+$ ]] || [[ "$DOC_COUNT" -le 0 ]]; then
  echo "Error: docs count is invalid: $DOC_COUNT" >&2
  exit 1
fi
if [[ "$DOC_COUNT" -ne "$FOUND_COUNT" ]]; then
  echo "Error: docs count mismatch (found_pdfs=$FOUND_COUNT, docs=$DOC_COUNT)" >&2
  exit 1
fi
echo "docs_count_check: PASS ($DOC_COUNT == $FOUND_COUNT)"

run_step "3) lab query" \
  "$PYTHON_BIN" -m backend.answer_cli --indexdir out/index --query "hemogram sonuçlarında HGB kaç?" --mode stub --retrieval tfidf

run_step "4) MR query" \
  "$PYTHON_BIN" -m backend.answer_cli --indexdir out/index --query "MR raporunda sonuç nedir?" --mode stub --retrieval tfidf

run_step "5) patoloji query" \
  "$PYTHON_BIN" -m backend.answer_cli --indexdir out/index --query "patoloji raporunda tanı nedir?" --mode stub --retrieval tfidf

echo
echo "Regression test completed successfully."
