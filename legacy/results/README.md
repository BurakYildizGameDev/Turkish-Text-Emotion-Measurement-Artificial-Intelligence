# Sonuçlar — Okumadan Önce

Bu klasördeki sayılar **tek seed** ile, **farklı eğitim bütçeleriyle** üretilmiştir.
Aşağıdaki uyarılar okunmadan deneyler arasında sonuç çıkarılmamalıdır.
Son düzeltme: 2026-09-28 (bkz. `../summary_of_fixes.txt` → "REVİZYON v3").

## Eğitim bütçeleri

| Deney | Eğitim verisi | Adım | Süre | Test F1-Macro | Test Acc |
|---|---|---|---|---|---|
| exp0 | tam veri, `max_steps=500` (~0.17 epoch), ağırlıksız | 500 | 11 dk | 0.3774 | 0.9178 |
| exp1 | ~41k (`--max-per-class 10000`), 2 epoch | 648 | 7 dk | 0.3617 | 0.8082 |
| exp2 | ~61k (max_per_class + oversample), 2 epoch | 954 | 8 dk | 0.4629 | 0.8743 |
| exp3 | **~385k (tam veri)**, 2 epoch | **6018** | 30 dk | 0.4376 | 0.8151 |
| exp4 | ~61k (max_per_class + oversample), 2 epoch | 954 | 8 dk | 0.4597 | 0.8761 |

Tüm deneyler aynı test setinde (`data/splits/master_splits/test.parquet`, 82.532 örnek) değerlendirilmiştir.

## Geçerli / geçersiz karşılaştırmalar

| Karşılaştırma | Geçerli mi | Neden |
|---|---|---|
| exp2 vs exp1 | ✅ | Aynı bütçe; fark oversampling + CL ağırlığından |
| **exp4 vs exp2** | ✅ | **Lexicon etkisinin tek geçerli ölçümü: −0.0032 → ölçülebilir katkı yok** |
| exp4 vs exp1 | ✅ | Aynı bütçe |
| exp3 vs exp1 | ❌ | exp3 ~10 kat fazla veri/adım |
| exp3 vs exp2 / exp4 | ❌ | Farklı bütçe |
| exp0 vs herhangi biri | ❌ | exp0 yalnızca ~0.17 epoch |

Her `exp*/metrics_report.json` içindeki karşılaştırma bloklarında `karsilastirma_gecerli_mi` ve `egitim_butcesi` alanları vardır.

## Dosya bazında uyarılar

- **`augmented_*.json`, `augmented_comparison_table.csv`** — Model iyileşmesi **değildir**. Test setine
  şablonla üretilmiş 200 cümle (100 gurur + 100 utanç) eklenmiştir; cümlelerin çoğu etiket kelimesini içerir.
  Gurur/utanç F1 artışı test setinin kolaylaşmasından gelir. Orijinal test sonuçları raporlanmalıdır.
- **`clean_test_results.json`, `clean_exp*_results.json`** — Bu dosyaları ve `test_clean.parquet`'i üreten script
  repoda yoktur; hangi 806 örneğin neden çıkarıldığı doğrulanamaz.
- **`group_aware_*`** — Group-aware split winvoker'ı 487k → 50k'ya indirir; sınıf dağılımı değiştiği için
  orijinal test sonuçlarıyla doğrudan karşılaştırılamaz. Eğitim ayarları da farklıdır (max_length=64, max_per_class=30000).
- **`kfold/`** — `max_per_class=5000` ile ~23k örnekli fold'lar; nihai modellerin koşullarını temsil etmez.
  `lr_sensitivity.json` 1 epoch/~100 sn'lik çalışmalardır (lr=1e-5'te accuracy %0.9) — LR duyarlılığını değil yakınsama hızını ölçer.
- **`baseline_comparison/`** — BiLSTM baseline'ı **eğitilmemiştir** ve 5.000 örnekle test edilmiştir; BERT modelleri 82.532 örnekle.
- **`statistical_tests/`** — `summary_of_fixes.txt`'te bahsedilen bu klasör repoda yoktur.

## Genel sınırlamalar

- Test setinde gurur **8**, utanç **37** örnek vardır; bu sınıfların F1 değerleri ve dolayısıyla macro-F1 çok gürültülüdür.
- Verinin ~%89'u sentiment (olumlu/olumsuz) verisidir ve doğrudan mutluluk/üzüntü/nötr olarak etiketlenmiştir;
  bu üç sınıftaki yüksek skorlar duygu tanımayı değil kaynak/tür ayrımını yansıtabilir.
- Eğitilmiş model ağırlıkları (`exp*/best_model/`) repoda yoktur; sonuçlar yeniden eğitim olmadan tekrar üretilemez.
