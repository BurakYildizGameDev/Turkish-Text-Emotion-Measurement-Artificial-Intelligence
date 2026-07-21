# v2 Deney Özeti (ablation + baseline + öğrenme hızı)

Seed'ler: [42, 43, 44] · ortalama ± standart sapma · düzen seçimi: **val_f1_macro ortalaması (test seçimde kullanılmaz)**
Seçilen düzen: **gold_none**

| Düzen | Train | Val F1-macro | Test F1-macro | Test Acc | Çeviri kısa yolu | ECE önce → sonra |
|---|---|---|---|---|---|---|
| `tfidf_lr` — TF-IDF (kelime+karakter n-gram) + Logistic Regression, yalnızca gold | 41,465 | 0.7640 ± 0.0000 | 0.7670 ± 0.0000 | 0.7741 ± 0.0000 | 0.018 | 0.027 → 0.020 |
| `gold` — BERTurk, yalnızca gold, ağırlıklı loss | 41,465 | 0.8042 ± 0.0009 | 0.8047 ± 0.0004 | 0.8104 ± 0.0018 | 0.003 | 0.074 → 0.018 |
| `gold_none` — BERTurk, yalnızca gold, ağırlıksız loss | 41,465 | 0.8159 ± 0.0011 | 0.8122 ± 0.0014 | 0.8179 ± 0.0020 | 0.004 | 0.052 → 0.018 |
| `gold_weak` — BERTurk, gold + weak (sınıf başına ≤5000), ağırlıklı loss | 55,979 | 0.8071 ± 0.0018 | 0.8057 ± 0.0012 | 0.8103 ± 0.0008 | 0.004 | 0.086 → 0.019 |
| `gold_silver` — BERTurk, gold + silver (nihal_8class), ağırlıklı loss | 69,347 | 0.7888 ± 0.0022 | 0.7866 ± 0.0029 | 0.7949 ± 0.0021 | 0.003 | 0.089 → 0.025 |
| `mbert_none` — mBERT (çok dilli), yalnızca gold, ağırlıksız | 41,465 | 0.7816 ± 0.0012 | 0.7796 ± 0.0022 | 0.7874 ± 0.0018 | 0.006 | 0.073 → 0.026 |
| `xlmr_none` — XLM-RoBERTa-base, yalnızca gold, ağırlıksız | 41,465 | 0.8026 ± 0.0012 | 0.7966 ± 0.0016 | 0.8044 ± 0.0015 | 0.004 | 0.067 → 0.027 |
| `gold_none_lr1e-5` — BERTurk gold_none, lr=1e-5 | 41,465 | 0.8125 ± 0.0011 | 0.8109 ± 0.0018 | 0.8175 ± 0.0020 | 0.004 | 0.048 → 0.021 |
| `gold_none_lr3e-5` — BERTurk gold_none, lr=3e-5 | 41,465 | 0.8142 ± 0.0025 | 0.8109 ± 0.0021 | 0.8160 ± 0.0025 | 0.003 | 0.064 → 0.021 |
| `gold_none_lr5e-5` — BERTurk gold_none, lr=5e-5 | 41,465 | 0.8117 ± 0.0026 | 0.8058 ± 0.0009 | 0.8114 ± 0.0015 | 0.003 | 0.076 → 0.024 |

## Sınıf bazlı test F1 (ortalama)

| Sınıf | `tfidf_lr` | `gold` | `gold_none` | `gold_weak` | `gold_silver` | `mbert_none` | `xlmr_none` | `gold_none_lr1e-5` | `gold_none_lr3e-5` | `gold_none_lr5e-5` |
|---|---|---|---|---|---|---|---|---|---|---|
| mutluluk | 0.830 | 0.874 | 0.878 | 0.871 | 0.867 | 0.852 | 0.873 | 0.878 | 0.876 | 0.871 |
| üzüntü | 0.788 | 0.842 | 0.849 | 0.843 | 0.818 | 0.804 | 0.823 | 0.850 | 0.846 | 0.841 |
| öfke | 0.704 | 0.750 | 0.754 | 0.752 | 0.753 | 0.711 | 0.732 | 0.754 | 0.751 | 0.749 |
| korku | 0.877 | 0.892 | 0.893 | 0.889 | 0.887 | 0.859 | 0.874 | 0.891 | 0.894 | 0.889 |
| şaşkınlık | 0.840 | 0.855 | 0.865 | 0.849 | 0.795 | 0.837 | 0.850 | 0.862 | 0.861 | 0.854 |
| tiksinti | 0.851 | 0.858 | 0.882 | 0.874 | 0.871 | 0.842 | 0.861 | 0.889 | 0.876 | 0.875 |
| sevgi | 0.538 | 0.638 | 0.635 | 0.636 | 0.606 | 0.606 | 0.624 | 0.623 | 0.643 | 0.632 |
| nötr | 0.709 | 0.728 | 0.742 | 0.731 | 0.697 | 0.726 | 0.735 | 0.741 | 0.740 | 0.736 |

## Kaynak bazlı test F1 (kaynağın kendi sınıfları üzerinden, ortalama)

| Kaynak | `tfidf_lr` | `gold` | `gold_none` | `gold_weak` | `gold_silver` | `mbert_none` | `xlmr_none` | `gold_none_lr1e-5` | `gold_none_lr3e-5` | `gold_none_lr5e-5` |
|---|---|---|---|---|---|---|---|---|---|---|
| goemotions_tr | 0.553 | 0.617 | 0.616 | 0.619 | 0.575 | 0.585 | 0.612 | 0.624 | 0.616 | 0.607 |
| tremo/consensus | 0.925 | 0.960 | 0.961 | 0.957 | 0.959 | 0.927 | 0.945 | 0.960 | 0.961 | 0.958 |
| tremo/majority | 0.728 | 0.802 | 0.806 | 0.804 | 0.797 | 0.751 | 0.767 | 0.806 | 0.803 | 0.793 |
| tweet_emotion | 0.966 | 0.978 | 0.978 | 0.977 | 0.992 | 0.970 | 0.974 | 0.974 | 0.979 | 0.980 |

Çeviri kısa yolu: TREMO/tweet (ana dil) test örneklerinin nötr veya sevgi tahmin edilme oranı. Bu kaynaklarda gold nötr/sevgi olmadığı için her biri hatadır.
