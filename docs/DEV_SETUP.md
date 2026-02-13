# Developer Setup

This repo uses a local virtual environment at `.venv`.

## Recommended (Option A) - Remove `langchain-huggingface`

Reason: current backend pipeline does not import `langchain_huggingface`; it uses custom ingestion + TF-IDF retrieval.

```bash
.venv/bin/python -m pip uninstall -y langchain-huggingface
.venv/bin/python -m pip check
```

## Option B - If you need it, pin `huggingface-hub` to compatible range

Use this only if you need `langchain_huggingface` integrations.

```bash
.venv/bin/python -m pip install "huggingface-hub>=0.33.4,<1.0.0" --upgrade
.venv/bin/python -m pip check
```

Risk: may downgrade transitive behavior expected by latest `transformers` / `sentence-transformers`.

## End-to-end pipeline (single command for ingest + index)

```bash
.venv/bin/python -m backend.ingest_cli --input data/pdfs --outdir out --build-index --indexdir out/index
```

Then ask:

```bash
.venv/bin/python -m backend.answer_cli --indexdir out/index --query "patoloji raporunda tanı nedir?" --topk 5 --mode ollama --model qwen2.5:7b
```

## OCR dependencies (full-page extraction)

Install system dependencies:

```bash
sudo apt install tesseract-ocr tesseract-ocr-tur poppler-utils
```

Install Python dependencies:

```bash
.venv/bin/pip install opencv-python pytesseract pymupdf pillow
```

Note: `poppler-utils` provides `pdftoppm`, needed when using `pdf2image` fallback.

Full-page OCR test:

```bash
.venv/bin/python scripts/ocr_extract.py data/pdfs/2.pdf --dpi 400 --lang tur+eng --full-page | tee /tmp/ocr_run.log
```

Keyword context check:

```bash
rg -n -C 5 "TANI|ICD|TANI \\(" /tmp/ocr_full_2.pdf.txt || true
```
