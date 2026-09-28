# Türkçe Metin Duygu Sınıflandırma (BERTurk, 8 sınıf)

Türkçe metinleri 8 duyguya ayıran BERTurk tabanlı sınıflandırıcı: veri seti derleme, eğitim, istatistiksel
değerlendirme ve Streamlit demosu.

**Sınıflar:** mutluluk · üzüntü · öfke · korku · şaşkınlık · tiksinti · sevgi · nötr
(tek kaynak: `src/data/labels.py`)

## Sonuçlar

Test seti: 8.875 insan etiketli örnek. 3 seed ortalaması; düzen seçimi yalnızca val F1 ile yapıldı.
Tüm sayılar: [`results/v2/ablation_summary.md`](results/v2/ablation_summary.md),
istatistik testleri: [`results/v2/stats/stats_report.md`](results/v2/stats/stats_report.md).

### Modeller (aynı veri, aynı bütçe)

| Model | Test macro-F1 | BERTurk ile fark (%95 GA) |
|---|---|---|
| **BERTurk** (`gold_none`, seçilen) | **0.812 ± 0.001** | — |
| XLM-RoBERTa-base | 0.797 ± 0.002 | −0.016 [−0.021, −0.010] |
| mBERT | 0.780 ± 0.002 | −0.033 [−0.039, −0.026] |
| TF-IDF + Logistic Regression | 0.767 | −0.045 [−0.054, −0.037] |

Tüm farklar anlamlı (paired bootstrap B=10.000 + Holm, p < 0.001).

### Kaynak bazlı test F1 (tek sayı yanıltıcıdır)

| Kaynak | n | BERTurk (%95 GA) | TF-IDF |
|---|---|---|---|
| GoEmotions TR (çeviri, Reddit) | 4.775 | **0.616** [0.592, 0.638] | 0.553 |
| TREMO — 2/3 oy uzlaşısı | 855 | **0.806** [0.782, 0.830] | 0.728 |
| TREMO — 3/3 oy uzlaşısı | 2.660 | 0.961 [0.955, 0.968] | 0.925 |
| Türkçe tweet | 585 | 0.978 [0.968, 0.988] | 0.966 |

TREMO-consensus ve tweet örnekleri çoğunlukla duyguyu doğrudan adlandırır ("korkarım", "şaşırdım"); kelime
tabanlı TF-IDF bile 0.93-0.97 alır. **Gerçekçi performans GoEmotions (~0.62) ve TREMO-majority (~0.81) satırlarıdır.**

### Ablation (tek etken değiştirilerek)

| Değişiklik | Etki | %95 GA | Sonuç |
|---|---|---|---|
| Loss ağırlığını kaldırmak | +0.008 | [+0.003, +0.012] | anlamlı iyileşme → varsayılan ağırlıksız |
| Weak veri eklemek (sınıf başına ≤5.000) | +0.001 | [−0.003, +0.005] | etkisiz |
| Silver veri eklemek | −0.018 | [−0.023, −0.013] | anlamlı zarar |
| Öğrenme hızı 1e-5 / 3e-5 (2e-5 yerine) | −0.001 / −0.001 | sıfırı içeriyor | fark yok |
| Öğrenme hızı 5e-5 | −0.006 | [−0.010, −0.003] | anlamlı zarar |

### Weak veri neden etkisiz?

`gold_weak` düzeni gold'a 14.514 kural tabanlı etiketli yorum ekler (sınıf başına ≤5.000: mutluluk 5.000, üzüntü 4.347,
şaşkınlık 1.740, öfke 1.248, tiksinti 1.140, sevgi 959, korku 80, **nötr 0**). Eğitim verisi %35 büyür ama test F1
değişmez (+0.001, p=0.60); sınıf bazlı değişimler ±0.006 içinde (tek istisna tiksinti +0.016), kaynak bazlı ±0.003.

Nedenini görmek için yalnızca gold ile eğitilmiş modelin (`gold_s42`) bu 14.514 weak örneği nasıl etiketlediğine bakıldı:

| Sınıf | Gold modeli weak etiketle aynı fikirde |
|---|---|
| mutluluk | 0.98 |
| üzüntü | 0.84 |
| korku | 0.66 |
| öfke | 0.65 |
| sevgi | 0.58 |
| şaşkınlık | 0.57 |
| tiksinti | 0.54 |
| **toplam** | **0.80** (uyuşanlarda ortalama güven 0.92) |

1. **Yeni bilgi taşımıyor.** Weak etiketler metindeki duygu kelimesinden üretilir ("memnunum", "üzüldüm"); gold ile
   eğitilen model bu kelimeleri zaten öğrenmiştir. Örneklerin %80'ini eklemeden önce de yüksek güvenle doğru tahmin
   eder, bu yüzden bunlardan gelen gradyan neredeyse sıfırdır. (TF-IDF'in kolay kaynaklarda 0.93-0.97 alması da
   bu kelime ipuçlarının zaten güçlü olduğunu gösterir.)
2. **Kalan %20 büyük ölçüde belirsiz ya da gürültülü.** Ürün/film/otel yorumları tek bir duygudan çok bir
   *değerlendirme* ifade eder; kural tek bir kelimeye bakar. Örnekler: *"bayıldım"* → kural sevgi der, gold
   tanımına göre mutluluk; *"bu kadar çabuk gelmesini beklemiyordum … tavsiye ederim"* → kural şaşkınlık, genel
   ton mutluluk; *"iğrenç ötesi … aşçı süper, teşekkür ederiz"* → kural tiksinti, metin karışık. Bu örnekler
   modeli gold tanımından uzaklaştırır; az sayıdaki gerçek yeni bilginin katkısını götürür.
3. **Zayıf sınıflara ulaşmıyor.** En zayıf sınıflar nötr (0.74) ve sevgi (0.64). Weak veri nötr örneği hiç
   içermez (sentiment kaynaklarının "nötr"ü Wikipedia cümleleriydi ve atıldı); sevgi örneklerinin çoğu "bayıldım"
   kalıbıdır ve modelle yalnızca %58 uyuşur.
4. **Alan farklı.** Weak verinin tamamı yorumlardan gelir; test kaynaklarının hiçbiri (anı cümleleri, Reddit, tweet)
   yorum değildir. Kazanılabilecek alan genellemesi test setinde ölçülmez.

Sonuç: weak veri varsayılan eğitimden çıkarıldı (`gold_none`). Fayda sağlaması için kelime ipucu taşımayan,
nötr/sevgi gibi zayıf sınıflara yönelik ve modelin henüz bilmediği örnekler gerekir (ör. insan etiketi ya da
güçlü bir öğretmen modelle seçici pseudo-label).

### Sınıf ve hata analizi

En zayıf sınıflar **sevgi (0.64)**, **nötr (0.74)** ve **öfke (0.75)**. En sık karışıklık öfke ↔ nötr (~200'er örnek)
ve nötr → mutluluk. Sevgi ve nötrün gold örneklerinin tamamı çeviridir. GoEmotions'taki yüksek güvenli hataların
önemli kısmı etiket gürültüsüdür (ör. *"Bu beni üzüyor"* → etiket: nötr). Ana dil test örneklerinin yalnızca
%0,4'ü nötr/sevgi tahmin ediliyor, yani model "çeviri dili → nötr/sevgi" kısa yolunu öğrenmemiş.
Olasılıklar temperature scaling ile kalibre (ECE 0.052 → 0.018).

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU'nuza uygun CUDA sürümü
pip install -r requirements.txt
```

İlk `build_v2` çalıştırmasında TREMO Kaggle'dan (`mansuralp/tremo`), diğer kaynaklar HuggingFace Hub'dan indirilir.

## Çalıştırma sırası

```bash
# 1. Veri setini derle (deterministik) → data/v2/*.parquet + build_report.json
python -m src.data.build_v2

# 2. Eğitimler: ablation + baseline'lar + öğrenme hızı taraması (tamamlananlar atlanır)
python -m src.training.run_ablation_v2
python -m src.training.run_ablation_v2 --groups ablation      # yalnızca bir grup
python -m src.training.train_v2 --data gold --imbalance none --seed 42 --save-model   # tek çalıştırma

# 3. İstatistik ve hata analizi → results/v2/stats/
python -m src.evaluation.stats_v2

# 4. Demo (config.yaml → inference.model_dir modelini yükler)
streamlit run demo/app.py

# Testler
python -m pytest tests
```

Windows PowerShell 5.1'de `&&` çalışmaz; komutları tek tek çalıştırın.

## Veri

Ayrıntılar: [`data/v2/README.md`](data/v2/README.md). Özet:

| Kalite | Kaynak | Lisans | Kullanım |
|---|---|---|---|
| gold | **TREMO** (Tocoglu & Alpkocak, 2018) — kişisel anılar, ana dil | yalnızca ticari olmayan kullanım | train/val/test |
| gold | **Türkçe tweet duygu** (Güven vd., 2019) — ana dil | belirsiz | train/val/test |
| gold | **GoEmotions TR** — İngilizceden makine çevirisi (Reddit) | Apache-2.0 (orijinal) | train/val/test |
| weak | Ürün/film yorumları (winvoker, mteb, fthbrmnby), kural tabanlı sentiment → duygu | kaynağa göre | yalnızca ablation |
| silver | `nihalenc/turkish-8class-emotion-dataset` — süreci belgelenmemiş | belirsiz | yalnızca ablation |

- val ve test **yalnızca gold** veriden oluşur; yakın kopyalar MinHash-LSH ile gruplanıp aynı split'e konur.
- TREMO lisansı nedeniyle `data/v2/*.parquet` repoya konmaz; `build_v2` ile yeniden üretilir.
- Eğitilmiş model ağırlıkları (`models/v2/`, ~440 MB) repoda değildir.

## Proje yapısı

```
src/data/          labels.py (sınıflar), sources.py (indirme), weak_labeling.py, build_v2.py
src/training/      train_v2.py (tek çalıştırma), run_ablation_v2.py (tüm deneyler + özet)
src/evaluation/    tfidf_baseline.py, stats_v2.py (bootstrap, McNemar, hata analizi)
src/inference/     emotion_classifier.py (kalibre olasılıklarla tahmin)
src/response/      response_adapter.py (duyguya göre yanıt profili)
demo/app.py        Streamlit arayüzü
results/v2/        deney çıktıları: runs/<düzen>_s<seed>/, ablation_summary.md, stats/
legacy/            v1 (10 sınıf) kodu ve sonuçları — bakımı yapılmıyor, bkz. legacy/README.md
```

## Yöntem

- Model: `dbmdz/bert-base-turkish-cased`, max 128 token, en fazla 4 epoch (en iyi val epoch'u), bf16. Tüm düzenlerde aynı eğitim bütçesi.
- Her düzen 3 seed (42, 43, 44) ile eğitilir; sonuçlar ortalama ± standart sapma olarak verilir.
- **Model ve hiperparametre seçimi yalnızca val macro-F1 ile yapılır**; test en sonda bir kez değerlendirilir.
- Olasılıklar val üzerinde temperature scaling ile kalibre edilir (demo kalibre olasılık gösterir).
- Karşılaştırmalar: paired bootstrap (B=10.000, seed ortalamalı) %95 güven aralığı, seed bazlı McNemar, Holm düzeltmesi.

## Bilinen sınırlamalar

- **nötr ve sevgi sınıflarının gold örneklerinin tamamı makine çevirisidir** (GoEmotions). Bu iki sınıf en zayıf
  sınıflardır ve ana dil metinlerdeki performansları doğrudan ölçülemez.
- **Tek bir genel skor yanıltıcıdır.** TREMO-consensus örneklerinin büyük kısmı etiketin kök kelimesini içerir
  ("korkarım", "şaşırdım") ve kolaydır; gerçekçi performans kaynak bazlı tabloda görülür.
- Alanlar farklıdır: TREMO kişisel anı cümleleri, GoEmotions Reddit yorumları, tweet'ler. Başka alanlarda
  (ürün yorumu, haber, konuşma dili) performans farklı olabilir.
- Tek etiketli sınıflandırmadır; karışık duygular tek sınıfa indirgenir.

## Lisans

Kod: MIT (`LICENSE`). Veri kaynaklarının lisansları kendilerine aittir; TREMO ve ondan türetilen modeller
yalnızca ticari olmayan amaçlarla kullanılabilir.

