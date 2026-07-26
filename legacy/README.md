# legacy/ — v1 arşivi (bakımı yapılmıyor)

Bu klasör projenin ilk sürümünü (v1: 10 sınıf, TREMO + çeviri karışımı, tek seed) arşivler.
Güncel hat v2'dir (8 sınıf) ve kök `README.md`'de anlatılır; v2 kodu buradaki hiçbir dosyayı kullanmaz.

| Klasör | İçerik |
|---|---|
| `scripts/` | Eski kök dizin scriptleri: `train_exp0..4.py`, `run_*.py`, `app.py`, `data_engine.py`, `error_analysis.py`, `shap_interpretability.py` vb. |
| `src_v1/` | v1 eğitim (`train.py`) ve veri indirme (`download_dataset.py`) modülleri |
| `results/` | v1 deney çıktıları (`exp0..4`, `kfold`, `final`, ...). Okumadan önce `results/README.md`'deki uyarılara bakın |
| `logs/` | v1 eğitim günlüğü |
| `summary_of_fixes.txt` | v1 dönemindeki onarım notları |

## Neden çalışmıyor

- Scriptler proje kökünde çalışacak şekilde yazılmıştı (`Path(__file__).parent` = kök); taşındıktan sonra yolları geçersizdir.
- v1 verisi (`data/processed/`, `data/splits/`, `data/raw/combined_raw.parquet`, ~400 MB) boyutu nedeniyle repodan ve
  git geçmişinden çıkarıldı. (Adına rağmen `tremo_splits` TREMO metni içermez; v1 verisinde TREMO yoktu.)
- `config.yaml` artık v2 şemasındadır (8 sınıf); v1'in `training`/`data` blokları kaldırıldı.

v1 verisi ve eski 3 sınıflı modeller git geçmişinden de temizlendiği için v1 sonuçları bu repodan yeniden üretilemez.
