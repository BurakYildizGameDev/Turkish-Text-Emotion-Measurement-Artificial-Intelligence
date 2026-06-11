import pandas as pd
import numpy as np
import os
import hashlib
import sys
from pathlib import Path

# Windows terminal encoding fix
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Yollar çalışma dizininden bağımsız olsun diye proje köküne göre
ROOT      = Path(__file__).resolve().parent
CSV_PATH  = ROOT / "results" / "turkce_duygu_analizi_ham.csv"
TEST_PATH = ROOT / "data" / "splits" / "master_splits" / "test.parquet"

print("=== VERI HAZIRLIK RAPORU ===\n")

# 1. CSV Oku
df_csv = pd.read_csv(CSV_PATH, encoding="utf-8")
csv_total = len(df_csv)
gurur_count = int((df_csv["label"] == "gurur").sum())
utanc_count = int((df_csv["label"] == "utanç").sum())
print(f"CSV okundu: {csv_total} satir")
print(f"  Gurur: {gurur_count} ornek")
print(f"  Utanc: {utanc_count} ornek\n")

# 2. Kalite Kontrolu
rejected = {"bos_metin": 0, "cok_kisa": 0, "duplicate_csv": 0, "duplicate_test": 0}

# Bos metin
bos_mask = df_csv["text"].isna() | (df_csv["text"].str.strip() == "")
rejected["bos_metin"] = int(bos_mask.sum())
df_csv = df_csv[~bos_mask].copy()

# Cok kisa (<5 kelime)
df_csv["word_count"] = df_csv["text"].str.split().str.len()
kisa_mask = df_csv["word_count"] < 5
rejected["cok_kisa"] = int(kisa_mask.sum())
df_csv = df_csv[~kisa_mask].copy()

# CSV ici duplicate
dup_csv = df_csv["text"].duplicated(keep="first")
rejected["duplicate_csv"] = int(dup_csv.sum())
df_csv = df_csv[~dup_csv].copy()

print(f"Kalite kontrolu sonrasi CSV: {len(df_csv)} satir")
print(f"  Elendi - Bos metin: {rejected['bos_metin']}")
print(f"  Elendi - Cok kisa (<5 kelime): {rejected['cok_kisa']}")
print(f"  Elendi - CSV ici duplicate: {rejected['duplicate_csv']}\n")

# 3. Test.parquet ile duplicate kontrolu
test_df = pd.read_parquet(TEST_PATH)
test_texts = set(test_df["text"].str.strip().str.lower())
dup_test = df_csv["text"].str.strip().str.lower().isin(test_texts)
rejected["duplicate_test"] = int(dup_test.sum())
df_csv = df_csv[~dup_test].copy()

print(f"Test seti ile duplicate kontrolu sonrasi: {len(df_csv)} satir")
print(f"  Elendi - Test seti ile duplicate: {rejected['duplicate_test']}\n")

# 4. 100 gurur + 100 utanc sec
df_gurur_pool = df_csv[df_csv["label"] == "gurur"]
df_utanc_pool = df_csv[df_csv["label"] == "utanç"]

n_gurur = min(100, len(df_gurur_pool))
n_utanc = min(100, len(df_utanc_pool))

df_gurur = df_gurur_pool.sample(n=n_gurur, random_state=42)
df_utanc = df_utanc_pool.sample(n=n_utanc, random_state=42)

print(f"Secilen ornekler:")
print(f"  Gurur: {len(df_gurur)} / {len(df_gurur_pool)} mevcut (hedef: 100)")
print(f"  Utanc: {len(df_utanc)} / {len(df_utanc_pool)} mevcut (hedef: 100)\n")

# 5. Label mapping (test.parquet formatina uygun)
label_norm_map = {"gurur": "gurur", "utanç": "utanc"}
label_id_map = {"gurur": 9, "utanç": 8}

df_new = pd.concat([df_gurur, df_utanc], ignore_index=True)
df_new["label_norm"] = df_new["label"].map(label_norm_map)
df_new["label_id"] = df_new["label"].map(label_id_map)
df_new["source"] = "manual_collection"
df_new = df_new[["text", "label_norm", "source", "label_id", "label"]]

# 6. Birlestir (orijinal DEGISTIRILMIYOR)
test_augmented = pd.concat([test_df, df_new], ignore_index=True)

orig_gurur = int((test_df["label"] == "gurur").sum())
orig_utanc = int((test_df["label"] == "utanç").sum())
new_gurur = int((test_augmented["label"] == "gurur").sum())
new_utanc = int((test_augmented["label"] == "utanç").sum())

print("=== BIRLESTIRME SONUCU ===")
print(f"Orijinal test.parquet: {len(test_df)} satir")
print(f"Yeni eklenen: {len(df_new)} satir")
print(f"test_augmented toplam: {len(test_augmented)} satir\n")
print(f"Gurur: {orig_gurur} -> {new_gurur} (+{len(df_gurur)})")
print(f"Utanc: {orig_utanc} -> {new_utanc} (+{len(df_utanc)})\n")

# 7. Kaydet
out_path = ROOT / "data" / "splits" / "test_augmented.parquet"
out_path.parent.mkdir(parents=True, exist_ok=True)
test_augmented.to_parquet(out_path, index=False)
print(f"KAYDEDILDI: {out_path}")

# Kayıt amaçlı hash — önceki bir değerle karşılaştırılmadığı için "değişmedi" iddiası yapılmaz
with open(TEST_PATH, "rb") as f:
    orig_hash = hashlib.md5(f.read()).hexdigest()
print(f"Orijinal test.parquet MD5: {orig_hash}\n")

total_rejected = sum(rejected.values())
print("=== OZET ===")
print(f"CSV toplam: {csv_total} satir")
print(f"Toplam elenen: {total_rejected}")
print(f"  - Bos metin: {rejected['bos_metin']}")
print(f"  - Cok kisa (<5 kelime): {rejected['cok_kisa']}")
print(f"  - CSV ici duplicate: {rejected['duplicate_csv']}")
print(f"  - Test seti ile duplicate: {rejected['duplicate_test']}")
print(f"Eklenen gurur: {len(df_gurur)}, eklenen utanc: {len(df_utanc)}")
print(f"test_augmented.parquet toplam: {len(test_augmented)} satir")
