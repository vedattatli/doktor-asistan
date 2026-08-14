# Doktor Asistanı v1

Türkçe tıbbi PDF dokümanlarında (laboratuvar, patoloji, radyoloji, epikriz) kaynak göstererek soru-cevap yapan yerel sistem. Hasta verisi makineden çıkmaz.

> **Not:** Bu sürümün devamı [doktor-asistan_v2](https://github.com/vedattatli/doktor-asistan_v2) olarak sıfırdan yazılmıştır. v2 kapsamı laboratuvar raporlarıyla sınırlar, buna karşılık deterministik ayrıştırma ve grafik desteği ekler. Bu depo referans olarak korunmaktadır.

## Ne yapıyor

- **PDF ve OCR** — PyMuPDF, tesseract ve OpenCV ile metin çıkarma
- **Kişisel veri maskeleme** — TC kimlik numarası, telefon ve tarih bilgilerini anonimleştirir
- **Vektör arama** — TF-IDF, opsiyonel embedding desteğiyle
- **Doküman tipi yönlendirme** — gelen dokümanın tipini tahmin eder, 8 farklı YAML profilinden uygun olanı uygular
- **Denetim kaydı** — yapılan işlemler loglanır
- **Arayüz** — komut satırı araçları ve çok sayfalı Streamlit arayüzü

## Teknoloji

Python · Ollama (qwen2.5:7b) · PyMuPDF · tesseract · OpenCV · scikit-learn · Streamlit

## Çalıştırma

Kurulum adımları için [docs/SETUP.md](docs/SETUP.md) dosyasına bakın.

```bash
python -m backend.ingest_cli    # doküman yükleme
python -m backend.answer_cli    # soru sorma
```

## Ölçek

Yaklaşık 6.045 satır.
