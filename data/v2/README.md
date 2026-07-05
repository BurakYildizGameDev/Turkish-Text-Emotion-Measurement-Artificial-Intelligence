# v2 Veri Seti — 8 Sınıflı Türkçe Duygu

Yeniden üretmek için (tüm kaynaklar otomatik indirilir, çıktı deterministiktir):

```bash
python -m src.data.build_v2            # data/v2/*.parquet + build_report.json
python -m src.data.build_v2 --no-silver
```

Parquet dosyaları repoya konmaz (TREMO lisansı ticari olmayan kullanım içindir). Tüm sayılar
`build_report.json` dosyasındadır; aşağıdaki tablolar oradan alınmıştır.

## Sınıflar

`mutluluk, üzüntü, öfke, korku, şaşkınlık, tiksinti, sevgi, nötr` — tek kaynak: `src/data/labels.py`.
v1'deki **gurur** ve **utanç** çıkarıldı (test'te 8 ve 37 örnek vardı, tek kaynakları makine çevirisiydi).

## Dosyalar ve etiket kalitesi

| Dosya | Kalite | Satır | Kullanım |
|---|---|---|---|
| `train.parquet` | gold | 41.465 | eğitim |
| `val.parquet` | gold | 8.867 | model seçimi |
| `test.parquet` | gold | 8.875 | **yalnızca nihai değerlendirme** |
| `train_weak.parquet` | weak | 105.211 | eğitime **eklenebilir** (sınıf başına üst sınırla) |
| `train_silver.parquet` | silver | 27.882 | varsayılan olarak **kullanılmaz** (ablation için) |

**val ve test yalnızca insan etiketli veriden oluşur**; weak/silver örnekleri asla değerlendirmeye girmez.

## Kaynaklar

### Gold (insan etiketli)
| Kaynak | Örnek | Not |
|---|---|---|
| **TREMO** (Tocoglu & Alpkocak, 2018) | 25.989 | Kişisel anılar, 6 duygu. `ValidatedEmotion` kullanıldı; 1.361 "Ambigious" atıldı. `subsource`: `tremo/consensus` (3/3 oy, 19.462) / `tremo/majority` (2/3, 6.527). Kaggle `mansuralp/tremo`'dan indirilir |
| **Türkçe tweet duygu** (Güven vd., 2019) | 3.998 | 5 duygu, dengeli, ana dili Türkçe |
| **GoEmotions TR** | 32.558 / 43.410 | İngilizceden makine çevirisi (`translated=True`). Yalnızca **etiket-tutarlı** örnekler: 8.235'i atılan bir etiket içerdiği, 2.606'sı farklı sınıflara giden çoklu etiket taşıdığı için çıkarıldı |

GoEmotions eşlemesi (Ekman gruplaması + sevgi): admiration, amusement, excitement, joy, optimism, relief, gratitude → mutluluk ·
love, caring → sevgi · anger, annoyance, disapproval → öfke · sadness, grief, disappointment, remorse → üzüntü ·
fear, nervousness → korku · disgust → tiksinti · surprise → şaşkınlık · neutral → nötr.
**Atılanlar:** pride, embarrassment (sınıf kaldırıldı), confusion, curiosity, realization (şaşkınlıktan kavramsal olarak farklı), approval, desire (belirgin sınıf yok).

### Weak (sentiment → duygu, kural tabanlı)
winvoker (ürün/mağaza/film yorumları, tweet), mteb_product, mteb_movie, fthbrmnby — toplam 563.570 olumlu/olumsuz yorum.
Kurallar: `src/data/weak_labeling.py`.

- **Polarite kısıtı:** olumlu → {mutluluk, sevgi, şaşkınlık}; olumsuz → {öfke, üzüntü, korku, tiksinti, şaşkınlık}
- Metin **tam olarak bir** duygunun açık ifadesini içermeli; eşleşme yok ya da birden fazla ise örnek atılır
- Olumsuzlama ("memnun değilim", "endişe etmeyin") ve yanıltıcı ifadeler ("korku filmi", "korkunç", "inanılmaz") dikkate alınır
- ≤ 500 karakter (etiketi belirleyen kelime modelin gördüğü 128 token içinde kalsın)
- Film kaynaklarında korku atılır (içerik tasvir eder, duygu değil — 148 örnek)
- Etiketlenen: 190.483 → tekilleştirme sonrası **105.211**

**Atılan sentiment verisi:** winvoker'ın "Notr" sınıfının tamamı (170.413 Wikipedia cümlesi + 504 anlamsız metin, ör. *"Integer non velit."*),
boun-tabilab (fthbrmnby ile birebir aynı), TTC4900 haberleri (v1'de hepsi "nötr" sayılıyordu).

**Kural isabeti** (kurallar gold veriye uygulandı; gold etiketle uyuşma oranı):

| Sınıf | Tahmin | İsabet |
|---|---|---|
| mutluluk | 6.192 | 0.954 |
| öfke | 2.184 | 0.950 |
| korku | 2.047 | 0.950 |
| tiksinti | 1.733 | 0.947 |
| üzüntü | 2.055 | 0.902 |
| şaşkınlık | 2.080 | 0.842 |
| sevgi | 894 | 0.767 |
| **genel** | %29 etiketlendi | **0.923** |

Sevgi kalıpları gold **train** split'inde ölçülerek seçildi (test'e bakılmadı). Bu isabet gold veride ölçüldüğü için
ürün/film yorumu alanındaki gerçek isabet farklı olabilir.

### Silver
`nihal_8class` (HF `nihalenc/turkish-8class-emotion-dataset`): etiketleme süreci belgelenmemiş. **4.160 örneği gold
veriyle birebir aynı**, 248'i val/test ile yakın kopya çıktı → GoEmotions/TREMO'dan türetilmiş. Aynı İngilizce cümlenin
farklı çevirisi metin eşleşmesiyle yakalanamayacağından test sızıntısı riski taşır. Kopyalar atıldı (kalan 27.882), ancak dosya
varsayılan eğitimde kullanılmamalıdır.

## Tekilleştirme ve split

1. Normalize anahtar (Türkçe küçük harf, noktalama yok): aynı metin farklı etiketle geliyorsa **hepsi atılır**
   (135 metin / 646 satır — çoğu TREMO'daki "ölüm", "sınav" gibi kısa ifadeler); aynı etiketle gelen 2.692 kopya atıldı.
2. Yakın kopyalar **tüm çiftler arasında** MinHash-LSH (karakter 5-gram, Jaccard ≥ 0.8) ile gruplanır — 462 grup, 1.009 örnek.
3. Gruplar `kaynak|etiket` tabakalı 70/15/15 bölünür; bir grup asla iki split'e düşmez.
4. weak/silver'dan gold ile tam kopya ve val/test ile yakın kopya olanlar atılır.

Doğrulama: train∩val, train∩test, val∩test, weak∩test normalize kesişimleri **0**; iki ayrı derleme **birebir aynı** dosyaları üretir.

## Sınıf dağılımı (gold)

| Sınıf | train | val | test | test kaynakları |
|---|---|---|---|---|
| mutluluk | 10.256 | 2.186 | 2.191 | GoEmo 1.369 · TREMO 703 · tweet 119 |
| nötr | 8.853 | 1.895 | 1.895 | **yalnızca GoEmo** |
| öfke | 6.654 | 1.428 | 1.422 | GoEmo 647 · TREMO 658 · tweet 117 |
| üzüntü | 5.153 | 1.104 | 1.101 | GoEmo 304 · TREMO 680 · tweet 117 |
| korku | 3.561 | 764 | 762 | GoEmo 82 · TREMO 564 · tweet 116 |
| şaşkınlık | 3.036 | 647 | 654 | GoEmo 104 · TREMO 434 · tweet 116 |
| tiksinti | 2.549 | 543 | 551 | GoEmo 75 · TREMO 476 |
| sevgi | 1.403 | 300 | 299 | **yalnızca GoEmo** |

## Bilinen sınırlamalar

- **nötr ve sevgi gold örneklerinin tamamı makine çevirisi (GoEmotions).** Model "çeviri dili → nötr/sevgi" kısa yolunu
  öğrenebilir. Değerlendirmede kaynak bazlı skorlar ayrıca raporlanmalı; weak veri sevgi için ana dilde 959 örnek ekler.
- **Alan farkı:** TREMO kişisel anı cümleleri, GoEmotions Reddit yorumları, weak veri ürün/film yorumları.
- **weak veri dengesiz:** 105.211 örneğin 95.697'si mutluluk ("memnunum", "teşekkürler"). Eğitimde sınıf başına üst sınır şart.
- **weak etiketler anahtar kelimeye dayanır:** Model bu kelimeleri kısa yol olarak öğrenebilir; bu nedenle test yalnızca gold'dur ve
  weak verinin katkısı Faz 4'te ablation ile ölçülmelidir (gold-only vs gold+weak).
- Tweet ve TREMO'nun lisansları sırasıyla belirsiz ve ticari olmayan kullanımdır.
