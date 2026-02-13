# Setup

Bu doküman Ubuntu üzerinde projeyi hızlıca ayağa kaldırmak için copy-paste odaklı komutları içerir.

## 1) Ön koşullar

- Ubuntu (22.04+ önerilir)
- Python 3.10+ ve `venv`
- Ollama (local LLM için)

Kontrol:

```bash
python3 --version
ollama --version
ollama list
```

## 2) Virtualenv aktivasyonu

Repo kökünde:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

Not: Komutlarda belirsizlik yaşamamak için doğrudan `.venv/bin/python` kullanabilirsiniz.

## 3) Telemetry kapatma

Bu oturum için:

```bash
export ANONYMIZED_TELEMETRY=false
export CHROMA_TELEMETRY=false
```

Kalıcı yapmak için `~/.bashrc` içine ekleyin:

```bash
echo 'export ANONYMIZED_TELEMETRY=false' >> ~/.bashrc
echo 'export CHROMA_TELEMETRY=false' >> ~/.bashrc
```

## 4) End-to-end pipeline

Ingest + index build (tek komut):

```bash
.venv/bin/python -m backend.ingest_cli --input data/pdfs --outdir out --build-index --indexdir out/index
```

Soru-cevap (stub):

```bash
.venv/bin/python -m backend.answer_cli --indexdir out/index --query "patoloji raporunda tanı nedir?" --topk 5 --mode stub
```

Soru-cevap (ollama):

```bash
.venv/bin/python -m backend.answer_cli --indexdir out/index --query "patoloji raporunda tanı nedir?" --topk 5 --mode ollama --model qwen2.5:7b
```

## 5) Dependency notu

### Option A (önerilen): `langchain-huggingface` kaldır

Pipeline şu an custom ingestion + TF-IDF retrieval ile çalışıyor; `langchain_huggingface` import edilmiyorsa kaldırmak daha temizdir.

```bash
.venv/bin/python -m pip uninstall -y langchain-huggingface
.venv/bin/python -m pip check
```

### Option B (sadece gerekirse): `huggingface-hub` pinle

`langchain-huggingface` zorunluysa hub sürümünü uyumlu aralığa sabitleyin.

```bash
.venv/bin/python -m pip install "huggingface-hub>=0.33.4,<1.0.0" --upgrade
.venv/bin/python -m pip check
```

## 6) Troubleshooting

`python: command not found` veya farklı interpreter sorunu:

```bash
.venv/bin/python -m backend.ingest_cli --help
```

`out/index` dosyaları eksikse (`vectorizer.pkl`, `matrix.npz`, `meta.jsonl`):

```bash
.venv/bin/python -m backend.retrieve_cli build --chunks out/chunks.jsonl --indexdir out/index
```

Alternatif olarak ingest sırasında otomatik index üretin:

```bash
.venv/bin/python -m backend.ingest_cli --input data/pdfs --outdir out --build-index --indexdir out/index
```
