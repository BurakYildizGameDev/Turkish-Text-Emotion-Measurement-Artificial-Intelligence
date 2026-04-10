"""
Near-Duplicate Kontrol Scripti
Eğitim başlamadan önce çalıştırılacak ön kontrol.
ÇALIŞTIRMA: python check_near_duplicates.py
Veriyi DEĞİŞTİRMEZ, sadece raporlar.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from pathlib import Path

THRESHOLD = 0.85
SAMPLE_SIZE = 5000
REPORT_PATH = "results/near_duplicate_report.json"


def check_intra_source(df):
    """Her kaynak içinde near-duplicate tarar."""
    report = {}
    for source in df['source'].unique():
        src_df = df[df['source'] == source].copy()
        if len(src_df) < 2:
            continue
        sample_idx = np.random.choice(len(src_df), min(SAMPLE_SIZE, len(src_df)), replace=False)
        texts = src_df['text'].fillna('').iloc[sample_idx].tolist()
        vec = TfidfVectorizer(max_features=5000)
        tfidf = vec.fit_transform(texts)
        sim = cosine_similarity(tfidf)
        np.fill_diagonal(sim, 0)
        dup_pairs = int((sim > THRESHOLD).sum() / 2)
        report[source] = {
            "sample_size": len(texts),
            "near_dup_pairs": dup_pairs,
            "risk_ratio": round(dup_pairs / max(len(texts), 1), 4)
        }
        print(f"[{source}] Örnek: {len(texts)} | Near-dup çifti: {dup_pairs}")
    return report


def check_train_test_leakage(df):
    """Train ile test arasında sızıntı riski tarar."""
    if 'split' not in df.columns:
        print("UYARI: 'split' kolonu bulunamadı, parquet dosyalarından yükleniyor...")
        return {}

    train_texts = df[df['split'] == 'train']['text'].sample(
        min(3000, len(df[df['split'] == 'train'])), random_state=42).tolist()
    test_texts = df[df['split'] == 'test']['text'].sample(
        min(1000, len(df[df['split'] == 'test'])), random_state=42).tolist()

    vec = TfidfVectorizer(max_features=5000)
    vec.fit(train_texts + test_texts)
    train_tfidf = vec.transform(train_texts)
    test_tfidf = vec.transform(test_texts)

    cross_sim = cosine_similarity(test_tfidf, train_tfidf)
    risk_ratio = float((cross_sim > THRESHOLD).any(axis=1).mean())

    print(f"\n[Train-Test Sızıntı Riski] Threshold={THRESHOLD} | Risk altındaki test örneği: %{risk_ratio*100:.1f}")
    return {"leakage_risk_ratio": round(risk_ratio, 4), "threshold": THRESHOLD}


def check_cross_split_leakage():
    """
    Ayrı parquet split dosyaları varsa train-test sızıntısını ölçer.
    master_splits/train.parquet ve test.parquet kullanır.
    """
    splits_dir = Path("data/splits/master_splits")
    train_path = splits_dir / "train.parquet"
    test_path  = splits_dir / "test.parquet"

    if not train_path.exists() or not test_path.exists():
        print("UYARI: Split parquet dosyaları bulunamadı.")
        return {}

    train_df = pd.read_parquet(train_path)
    test_df  = pd.read_parquet(test_path)

    train_sample = train_df['text'].fillna('').sample(
        min(3000, len(train_df)), random_state=42).tolist()
    test_sample  = test_df['text'].fillna('').sample(
        min(1000, len(test_df)), random_state=42).tolist()

    vec = TfidfVectorizer(max_features=5000)
    vec.fit(train_sample + test_sample)
    train_tfidf = vec.transform(train_sample)
    test_tfidf  = vec.transform(test_sample)

    cross_sim  = cosine_similarity(test_tfidf, train_tfidf)
    risk_ratio = float((cross_sim > THRESHOLD).any(axis=1).mean())

    print(f"[Cross-Split Sızıntı] Threshold={THRESHOLD} | Risk altındaki test örneği: %{risk_ratio*100:.1f}")
    return {
        "train_sample_size": len(train_sample),
        "test_sample_size":  len(test_sample),
        "leakage_risk_ratio": round(risk_ratio, 4),
        "threshold": THRESHOLD
    }


def main():
    print("=== Near-Duplicate Kontrol Başlıyor ===\n")

    # Veri yükle
    parquet_path = Path("data/processed/master_dataset.parquet")
    if not parquet_path.exists():
        print(f"HATA: {parquet_path} bulunamadı. Önce 'python data_engine.py --rebuild' çalıştırın.")
        return

    df = pd.read_parquet(parquet_path)
    print(f"Toplam örnek: {len(df)}\n")

    # Kaynak içi near-duplicate kontrol
    print("--- Kaynak İçi Near-Duplicate Analizi ---")
    intra_report = check_intra_source(df)

    # Train-test sızıntı kontrolü (split kolonu varsa)
    print("\n--- Train-Test Sızıntı Kontrolü (split kolonu) ---")
    leakage_report = check_train_test_leakage(df)

    # Parquet split dosyalarından sızıntı kontrolü
    print("\n--- Cross-Split Sızıntı Kontrolü (parquet) ---")
    cross_split_report = check_cross_split_leakage()

    # Raporu kaydet
    full_report = {
        "date": "2026-06-14",
        "threshold": THRESHOLD,
        "intra_source": intra_report,
        "train_test_leakage": leakage_report,
        "cross_split_leakage": cross_split_report,
        "status": "RAPOR TAMAMLANDI — veri değiştirilmedi"
    }
    Path("results").mkdir(exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    print(f"\nRapor kaydedildi: {REPORT_PATH}")


if __name__ == "__main__":
    main()
