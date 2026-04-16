# Commit 40: feat(data): add CLI threshold tuning to run_duplicate_analysis
"""
Exact duplicate (MD5) + Near-duplicate (MinHash/LSH) analizi.
GPU işlemi yok, eğitim yok — sadece veri kalite kontrolü.
"""

import sys, re, json, hashlib, os
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from datasketch import MinHash, MinHashLSH

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "data" / "splits" / "master_splits"
OUT    = ROOT / "results" / "duplicate_analysis_report.json"

MINHASH_PERMS = 128
NEAR_DUP_THRESHOLD = 0.85
SHINGLE_SIZE = 3       # kelime n-gramları

# ─── Normalizer ──────────────────────────────────────────────────────────────

_URL_RE    = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_PUNCT_RE  = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RE  = re.compile(r"\s+")
_TR_LOWER  = str.maketrans("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ",
                            "abcçdefgğhıijklmnoöprsştuüvyz")

def normalize(text: str) -> str:
    text = text.translate(_TR_LOWER)
    text = _URL_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()

# ─── Yardımcılar ─────────────────────────────────────────────────────────────

def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def shingles(text: str, k: int = SHINGLE_SIZE) -> set:
    words = text.split()
    if len(words) < k:
        return {text}
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}

def build_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=MINHASH_PERMS)
    for s in shingles(text):
        m.update(s.encode("utf-8"))
    return m

# ─── Veri yükleme ────────────────────────────────────────────────────────────

print("=" * 60)
print("ADIM 0 — Veriler yükleniyor...")
print("=" * 60)

train_df = pd.read_parquet(SPLITS / "train.parquet")
val_df   = pd.read_parquet(SPLITS / "val.parquet")
test_df  = pd.read_parquet(SPLITS / "test.parquet")

print(f"  train : {len(train_df):,} satır")
print(f"  val   : {len(val_df):,} satır")
print(f"  test  : {len(test_df):,} satır")

# ─── Normaliz ────────────────────────────────────────────────────────────────

print("\nNormalize ediliyor...")
train_norm = train_df["text"].fillna("").map(normalize)
val_norm   = val_df["text"].fillna("").map(normalize)
test_norm  = test_df["text"].fillna("").map(normalize)
print("  Normalizasyon tamamlandı.")

# ─── ADIM 1: Exact Duplicate (MD5) ───────────────────────────────────────────

print("\n" + "=" * 60)
print("ADIM 1 — Exact Duplicate Kontrolü (MD5 Hash)")
print("=" * 60)

print("  Hash'ler hesaplanıyor...")
train_hashes = set(train_norm.map(md5))
val_hashes   = set(val_norm.map(md5))
test_hashes  = set(test_norm.map(md5))

# Overlap
tt_exact  = len(train_hashes & test_hashes)
tv_exact  = len(train_hashes & val_hashes)
vt_exact  = len(val_hashes   & test_hashes)

print(f"  Train ∩ Test : {tt_exact:,} exact duplicate")
print(f"  Train ∩ Val  : {tv_exact:,} exact duplicate")
print(f"  Val   ∩ Test : {vt_exact:,} exact duplicate")

# ─── ADIM 2: Near-Duplicate (MinHash LSH) ────────────────────────────────────

print("\n" + "=" * 60)
print("ADIM 2 — Near-Duplicate Kontrolü (MinHash LSH, eşik=0.85)")
print("=" * 60)

# Sadece train-test çifti için LSH (en kritik sızıntı riski)
# val-test ve train-val da yapılır ama train-test öncelikli

def find_near_dups(
    query_series: pd.Series,
    query_label: str,
    index_series: pd.Series,
    index_label: str,
    threshold: float = NEAR_DUP_THRESHOLD,
) -> tuple[int, list]:
    """
    query_series içindeki her metni index_series'e karşı LSH ile sorgular.
    Dönüş: (near_dup_count, sample_pairs_list)
    """
    print(f"  LSH index oluşturuluyor ({index_label}, {len(index_series):,} metin)...")
    lsh = MinHashLSH(threshold=threshold, num_perm=MINHASH_PERMS)

    for i, text in enumerate(index_series):
        m = build_minhash(text)
        lsh.insert(f"{index_label}_{i}", m)
        if (i + 1) % 50_000 == 0:
            print(f"    {i+1:,}/{len(index_series):,} index edildi...")
    print(f"  Index tamamlandı. Şimdi {query_label} sorgulanıyor ({len(query_series):,} metin)...")

    near_dup_pairs = []
    seen = set()

    for j, text in enumerate(query_series):
        m = build_minhash(text)
        results = lsh.query(m)
        for res in results:
            pair_key = (j, res)
            if pair_key not in seen:
                seen.add(pair_key)
                # İndeks numarasını çıkar
                idx_num = int(res.split("_", 2)[-1])
                near_dup_pairs.append({
                    "query_idx":   j,
                    "index_key":   res,
                    "query_text":  text[:120],
                    "index_text":  index_series.iloc[idx_num][:120],
                })
        if (j + 1) % 20_000 == 0:
            print(f"    {j+1:,}/{len(query_series):,} sorgulandı, {len(near_dup_pairs)} çift bulundu...")

    return len(near_dup_pairs), near_dup_pairs[:10]


# Train-Test (en önemli)
nd_train_test_count, nd_train_test_samples = find_near_dups(
    test_norm,  "test",
    train_norm, "train",
)
print(f"\n  Train-Test near-duplicate: {nd_train_test_count:,}")

# Train-Val
nd_train_val_count, nd_train_val_samples = find_near_dups(
    val_norm,   "val",
    train_norm, "train",
)
print(f"  Train-Val near-duplicate : {nd_train_val_count:,}")

# Val-Test
nd_val_test_count, nd_val_test_samples = find_near_dups(
    test_norm, "test",
    val_norm,  "val",
)
print(f"  Val-Test near-duplicate  : {nd_val_test_count:,}")

# ─── ADIM 3: Raporu kaydet ───────────────────────────────────────────────────

print("\n" + "=" * 60)
print("ADIM 3 — Rapor kaydediliyor...")
print("=" * 60)

report = {
    "dataset_sizes": {
        "train": len(train_df),
        "val":   len(val_df),
        "test":  len(test_df),
    },
    "exact_duplicates": {
        "train_test": tt_exact,
        "train_val":  tv_exact,
        "val_test":   vt_exact,
    },
    "near_duplicates": {
        "train_test": nd_train_test_count,
        "train_val":  nd_train_val_count,
        "val_test":   nd_val_test_count,
    },
    "threshold": NEAR_DUP_THRESHOLD,
    "minhash_perms": MINHASH_PERMS,
    "shingle_size": SHINGLE_SIZE,
    "sample_pairs_train_test": nd_train_test_samples,
}

os.makedirs(str(OUT.parent), exist_ok=True)
with open(str(OUT), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"  Kaydedildi: {OUT}")

# ─── Özet ────────────────────────────────────────────────────────────────────

total_exact    = tt_exact + tv_exact + vt_exact
total_near_dup = nd_train_test_count + nd_train_val_count + nd_val_test_count
leak_risk = "YÜKSEK" if (tt_exact > 0 or nd_train_test_count > 100) else \
            "DÜŞÜK"  if (nd_train_test_count > 0) else "YOK"

print("\n" + "=" * 60)
print("ÖZET RAPORU")
print("=" * 60)
print(f"Exact Duplicate:")
print(f"  Train ∩ Test : {tt_exact:,}")
print(f"  Train ∩ Val  : {tv_exact:,}")
print(f"  Val   ∩ Test : {vt_exact:,}")
print(f"  TOPLAM       : {total_exact:,}")
print()
print(f"Near-Duplicate (eşik={NEAR_DUP_THRESHOLD}):")
print(f"  Train-Test : {nd_train_test_count:,}")
print(f"  Train-Val  : {nd_train_val_count:,}")
print(f"  Val-Test   : {nd_val_test_count:,}")
print(f"  TOPLAM     : {total_near_dup:,}")
print()
print(f"Veri Sızıntısı Riski: {leak_risk}")
if nd_train_test_samples:
    print()
    print("En benzer çiftlerden örnekler (Train-Test):")
    for i, p in enumerate(nd_train_test_samples[:5], 1):
        print(f"  [{i}] TEST : {p['query_text'][:80]}")
        print(f"      TRAIN: {p['index_text'][:80]}")
        print()
