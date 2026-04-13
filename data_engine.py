"""
Master Data Engine — 10 Sınıflı Türkçe Duygu Analizi
=====================================================
Desteklenen Kaynaklar:
  1. TREMO       — tocoglu/tremo (HF) veya data/raw/tremo_local.csv (yerel)
  2. TURTED      — Turkish Twitter Emotion Dataset (yerel CSV veya HF)
  3. winvoker    — winvoker/turkish-sentiment-analysis-dataset
  4. TTC4900     — savasy/ttc4900 (Türkçe haber → Nötr)
  5. GoEmotions TR — AnasAlokla/multilingual_go_emotions (TR filtresi)
  6. (ek)        — fthbrmnby, boun, mteb ürün/film yorumları

Çıktı:
  data/processed/master_dataset.parquet   — tam veri, label_id dahil
  data/processed/master_stats.json        — sınıf dağılımı + ağırlıklar
  data/processed/sampler_weights.npy      — WeightedRandomSampler için ağırlık dizisi

ÖNEMLİ: Repodaki veri (data/processed/combined_clean.parquet ve master_splits/)
bu script ile DEĞİL, src/data/download_dataset.py ile üretilmiştir. Bu script
hiçbir eğitim scriptinden import edilmez; TREMO ve TURTED verideki kaynaklar
arasında yoktur (gerçek kaynaklar: winvoker, goemotions_tr, mteb_movie,
mteb_product, ttc4900, fthbrmnby).

RTX 5070 Ti için: bf16=True (fp16 değil), batch_size=64, gradient_accumulation_steps=2  # config.yaml:52 ile tutarlı — bf16 kullanılıyor
"""

import os
import ast
import json
import re
import yaml
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Optional

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"

# ─── Sabitler ───────────────────────────────────────────────────────────────

EMOTIONS = [
    "mutluluk", "üzüntü", "öfke", "korku",
    "şaşkınlık", "tiksinti", "sevgi", "nötr", "utanç", "gurur",
]
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL = {i: e for i, e in enumerate(EMOTIONS)}
NUM_LABELS = len(EMOTIONS)

# İç işlemde accent-free normalize form kullanılır, son aşamada Türkçe'ye çevrilir.
_NORM = {
    "mutluluk":  "mutluluk",
    "uzuntu":    "üzüntü",
    "ofke":      "öfke",
    "korku":     "korku",
    "saskınlık": "şaşkınlık",
    "saskınlik": "şaşkınlık",
    "tiksinti":  "tiksinti",
    "sevgi":     "sevgi",
    "notr":      "nötr",
    "nötr":      "nötr",
    "utanc":     "utanç",
    "utanç":     "utanç",
    "gurur":     "gurur",
    # ─── Unicode-tam ───
    "üzüntü":    "üzüntü",
    "öfke":      "öfke",
    "şaşkınlık": "şaşkınlık",
}

# winvoker / binary sentiment
_SENTIMENT_MAP = {
    "positive": "mutluluk",
    "negative": "üzüntü",
    "neutral":  "nötr",
    "notr":     "nötr",
    "pos":      "mutluluk",
    "neg":      "üzüntü",
    "1":        "mutluluk",
    "0":        "üzüntü",
    1:          "mutluluk",
    0:          "üzüntü",
}

# GoEmotions 28 → 10
_GO_TO_10 = {
    0:  "mutluluk",   # admiration
    1:  "mutluluk",   # amusement
    2:  "öfke",       # anger
    3:  "öfke",       # annoyance
    4:  "mutluluk",   # approval
    5:  "sevgi",      # caring
    6:  "şaşkınlık",  # confusion
    7:  "şaşkınlık",  # curiosity
    8:  "sevgi",      # desire
    9:  "üzüntü",     # disappointment
    10: "öfke",       # disapproval
    11: "tiksinti",   # disgust
    12: "utanç",      # embarrassment
    13: "mutluluk",   # excitement
    14: "korku",      # fear
    15: "sevgi",      # gratitude
    16: "üzüntü",     # grief
    17: "mutluluk",   # joy
    18: "sevgi",      # love
    19: "korku",      # nervousness
    20: "mutluluk",   # optimism
    21: "gurur",      # pride
    22: "şaşkınlık",  # realization
    23: "mutluluk",   # relief
    24: "üzüntü",     # remorse
    25: "üzüntü",     # sadness
    26: "şaşkınlık",  # surprise
    27: "nötr",       # neutral
}

# TURTED etiketleri
_TURTED_MAP = {
    "joy":       "mutluluk",
    "sadness":   "üzüntü",
    "anger":     "öfke",
    "fear":      "korku",
    "surprise":  "şaşkınlık",
    "disgust":   "tiksinti",
    "love":      "sevgi",
    "neutral":   "nötr",
    "shame":     "utanç",
    "pride":     "gurur",
    # Türkçe varyantlar
    "mutluluk":  "mutluluk",
    "uzuntu":    "üzüntü",
    "ofke":      "öfke",
    "korku":     "korku",
    "saskınlık": "şaşkınlık",
    "tiksinti":  "tiksinti",
    "sevgi":     "sevgi",
    "notr":      "nötr",
    "utanc":     "utanç",
    "gurur":     "gurur",
}


# ─── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────

def _normalize_label(raw) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    return _NORM.get(s) or _SENTIMENT_MAP.get(s) or _TURTED_MAP.get(s)


def _clean_text(text: str) -> str:
    text = str(text).strip()
    # URL kaldır
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # @mention kaldır
    text = re.sub(r"@\w+", "", text)
    # çoklu boşluk
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_hf(dataset_name: str, **kwargs):
    from datasets import load_dataset
    return load_dataset(dataset_name, **kwargs)


# ─── Kaynak Yükleyiciler ─────────────────────────────────────────────────────

def load_tremo() -> pd.DataFrame:
    """TREMO: HuggingFace'den veya yerel CSV'den yükler."""
    local_csv = ROOT / "data" / "raw" / "tremo_local.csv"
    rows = []

    if local_csv.exists():
        print(f"  [TREMO] Yerel CSV bulundu: {local_csv}")
        df = pd.read_csv(local_csv)
        # Sütun adlarını normalize et
        df.columns = [c.lower().strip() for c in df.columns]
        text_col = next((c for c in df.columns if "text" in c or "yorum" in c or "cumle" in c), None)
        label_col = next((c for c in df.columns if "label" in c or "etiket" in c or "duygu" in c), None)
        if text_col and label_col:
            for _, row in df.iterrows():
                norm = _normalize_label(row[label_col])
                if norm and len(str(row[text_col]).strip()) >= 10:
                    rows.append({"text": _clean_text(row[text_col]), "label": norm, "source": "tremo_local"})
        print(f"  [TREMO] {len(rows)} örnek (yerel)")
    else:
        print("  [TREMO] Yerel CSV yok, HuggingFace deneniyor: tocoglu/tremo")
        try:
            ds = _load_hf("tocoglu/tremo")
            for split_name in ds.keys():
                for row in ds[split_name]:
                    norm = _normalize_label(row.get("label") or row.get("emotion"))
                    text = row.get("text") or row.get("sentence") or ""
                    if norm and len(text.strip()) >= 10:
                        rows.append({"text": _clean_text(text), "label": norm, "source": "tremo_hf"})
            print(f"  [TREMO] {len(rows)} örnek (HuggingFace)")
        except Exception as e:
            print(f"  [TREMO] UYARI: Yüklenemedi ({e})")
            print("  [TREMO] → data/raw/tremo_local.csv dosyasını projeye ekleyebilirsiniz.")

    return pd.DataFrame(rows)


def load_turted() -> pd.DataFrame:
    """
    TURTED: Önce yerel CSV, ardından bilinen HF adlarını dener.
    Yerel dosya yolu: data/raw/turted.csv  (sütunlar: text, emotion)
    """
    local_csv = ROOT / "data" / "raw" / "turted.csv"
    rows = []

    if local_csv.exists():
        print(f"  [TURTED] Yerel CSV bulundu: {local_csv}")
        df = pd.read_csv(local_csv)
        df.columns = [c.lower().strip() for c in df.columns]
        text_col = next((c for c in df.columns if "text" in c or "tweet" in c), None)
        label_col = next((c for c in df.columns if "label" in c or "emotion" in c or "duygu" in c), None)
        if text_col and label_col:
            for _, row in df.iterrows():
                norm = _normalize_label(row[label_col])
                if norm and len(str(row[text_col]).strip()) >= 5:
                    rows.append({"text": _clean_text(row[text_col]), "label": norm, "source": "turted_local"})
        print(f"  [TURTED] {len(rows)} örnek (yerel)")
    else:
        hf_candidates = [
            "turkish-nlp/turted",
            "savasy/turkish-emotion",
            "basakgultekin/turkish_tweets_emotion",
        ]
        for hf_name in hf_candidates:
            try:
                print(f"  [TURTED] Deneniyor: {hf_name}")
                ds = _load_hf(hf_name)
                for split_name in ds.keys():
                    for row in ds[split_name]:
                        norm = _normalize_label(row.get("label") or row.get("emotion") or row.get("target"))
                        text = row.get("text") or row.get("tweet") or row.get("sentence") or ""
                        if norm and len(text.strip()) >= 5:
                            rows.append({"text": _clean_text(text), "label": norm, "source": f"turted_{hf_name.split('/')[-1]}"})
                if rows:
                    print(f"  [TURTED] {len(rows)} örnek ({hf_name})")
                    break
            except Exception:
                continue
        if not rows:
            print("  [TURTED] UYARI: Bulunamadı. data/raw/turted.csv ekleyebilirsiniz.")

    return pd.DataFrame(rows)


def load_winvoker() -> pd.DataFrame:
    """winvoker — 489k Türkçe ürün yorumu (positive/negative/notr)."""
    print("  [winvoker] İndiriliyor...")
    rows = []
    try:
        ds = _load_hf("winvoker/turkish-sentiment-analysis-dataset")
        for split_name in ds.keys():
            df_s = ds[split_name].to_pandas()
            for _, row in df_s.iterrows():
                norm = _normalize_label(row.get("label"))
                text = str(row.get("text", "")).strip()
                if norm and len(text) >= 10:
                    rows.append({"text": _clean_text(text), "label": norm, "source": "winvoker"})
        print(f"  [winvoker] {len(rows)} örnek")
    except Exception as e:
        print(f"  [winvoker] HATA: {e}")
    return pd.DataFrame(rows)


def load_ttc4900() -> pd.DataFrame:
    """TTC4900 — Türkçe haber kategorileri → hepsi Nötr."""
    print("  [TTC4900] İndiriliyor...")
    rows = []
    try:
        ds = _load_hf("savasy/ttc4900")
        for split_name in ds.keys():
            df_s = ds[split_name].to_pandas()
            for _, row in df_s.iterrows():
                text = str(row.get("text", "")).strip()
                if len(text) >= 20:
                    rows.append({"text": _clean_text(text), "label": "nötr", "source": "ttc4900"})
        print(f"  [TTC4900] {len(rows)} örnek (hepsi → nötr)")
    except Exception as e:
        print(f"  [TTC4900] HATA: {e}")
    return pd.DataFrame(rows)


def load_goemotions_tr() -> pd.DataFrame:
    """GoEmotions TR — 28 sınıf → 10 sınıfa dönüştürülür."""
    print("  [GoEmotions TR] İndiriliyor...")
    rows = []
    try:
        ds = _load_hf("AnasAlokla/multilingual_go_emotions")
        split_name = list(ds.keys())[0]
        for row in ds[split_name]:
            if row.get("language") != "tr":
                continue
            text = str(row.get("text", "")).strip()
            if len(text) < 10:
                continue
            try:
                raw_labels = row.get("labels", "[]")
                go_ids = ast.literal_eval(raw_labels) if isinstance(raw_labels, str) else raw_labels
            except Exception:
                continue
            for gid in go_ids:
                norm = _GO_TO_10.get(int(gid))
                if norm:
                    rows.append({"text": _clean_text(text), "label": norm, "source": "goemotions_tr"})
                    break
        print(f"  [GoEmotions TR] {len(rows)} örnek")
    except Exception as e:
        print(f"  [GoEmotions TR] HATA: {e}")

    df = pd.DataFrame(rows)

    # ============================================================
    # KALİTE FİLTRESİ — EĞİTİM ÖNCESİ AKTİF EDİLECEK
    # Aşağıdaki satırların başındaki # işaretini kaldırarak aktif et.
    # Makine çevirisi kaynaklı gürültüyü azaltmak için tasarlanmıştır.
    # ============================================================
    # Türkçe karakter oranı filtresi (bozuk çeviri tespiti)
    # turkish_chars = set('çğışöüÇĞİŞÖÜ')
    # df['tr_char_ratio'] = df['text'].apply(
    #     lambda x: sum(c in turkish_chars for c in str(x)) / max(len(str(x)), 1)
    # )
    # before = len(df)
    # df = df[df['tr_char_ratio'] > 0.01].drop(columns=['tr_char_ratio'])
    # print(f"[GoEmotions TR] Kalite filtresi: {before} → {len(df)} örnek kaldı")
    #
    # Minimum sözcük sayısı filtresi (15 sözcük altı çeviriler genelde bozuk)
    # df = df[df['text'].str.split().str.len() >= 15]
    #
    # Multi-label güven skoru — tek etiketli örnekler daha güvenilirdir:
    # (Şu an ilk etiket alınıyor; aşağıdaki blok önceliklendirme için kullanılabilir)
    # df_single = df[df['source'] == 'goemotions_tr']  # tek etiket seçimi için ek alan gerekir
    # print(f"[GoEmotions TR] Tek etiket: {len(df_single)} | Çok etiket: {len(df) - len(df_single)}")
    # ============================================================

    return df


def load_extra_sources() -> pd.DataFrame:
    """Ek kaynaklar: fthbrmnby, boun, mteb ürün/film yorumları."""
    rows = []

    extras = [
        ("fthbrmnby/turkish_product_reviews",  "sentence", "sentiment"),
        ("boun-tabilab/Turkish-Product-Reviews", "text",    "label"),
        ("mteb/turkish_product_sentiment",      "text",    "label"),
        ("mteb/turkish_movie_sentiment",        "text",    "label"),
    ]

    for hf_name, text_col, label_col in extras:
        try:
            print(f"  [EK] {hf_name} indiriliyor...")
            ds = _load_hf(hf_name)
            for split_name in ds.keys():
                df_s = ds[split_name].to_pandas()
                # Split'e özel yerel kolon adları — önceki split'te bulunan alternatif
                # adın sonraki split'e sızmaması için tanımlı adlar üzerine yazılmaz.
                t_col, l_col = text_col, label_col
                if t_col not in df_s.columns or l_col not in df_s.columns:
                    alt_text = next((c for c in df_s.columns if "text" in c.lower()), None)
                    alt_label = next((c for c in df_s.columns if "label" in c.lower() or "sentiment" in c.lower()), None)
                    if alt_text:
                        t_col = alt_text
                    if alt_label:
                        l_col = alt_label
                for _, row in df_s.iterrows():
                    norm = _normalize_label(row.get(l_col))
                    text = str(row.get(t_col, "")).strip()
                    if norm and len(text) >= 10:
                        rows.append({"text": _clean_text(text), "label": norm, "source": hf_name.split("/")[-1]})
            print(f"  [EK] {hf_name}: {sum(1 for r in rows if r['source'] == hf_name.split('/')[-1])} örnek")
        except Exception as e:
            print(f"  [EK] {hf_name} atlandı: {e}")

    return pd.DataFrame(rows)


# ─── WeightedRandomSampler Ağırlık Hesabı ───────────────────────────────────

def compute_sample_weights(label_ids: np.ndarray, num_classes: int = NUM_LABELS) -> np.ndarray:
    """
    Her örnek için örnekleme ağırlığı hesaplar.
    Sınıf ağırlığı = N_toplam / (N_sınıf * num_classes)
    Bu değerleri PyTorch WeightedRandomSampler'a doğrudan verebilirsiniz.
    """
    counts = np.bincount(label_ids, minlength=num_classes).astype(float)
    # Sıfır bölmeden kaçın
    counts = np.where(counts == 0, 1, counts)
    class_weights = len(label_ids) / (counts * num_classes)
    sample_weights = class_weights[label_ids]
    return sample_weights


# ─── İstatistik Raporu ──────────────────────────────────────────────────────

def print_stats(df: pd.DataFrame, title: str = "") -> dict:
    total = len(df)
    counts = Counter(df["label"])
    print(f"\n{'═'*60}")
    if title:
        print(f"  {title}")
    print(f"{'═'*60}")
    stats = {}
    for emo in EMOTIONS:
        c = counts.get(emo, 0)
        bar = "█" * int(c / max(total, 1) * 30)
        print(f"  {emo:12s} {c:7d} ({c/max(total,1)*100:5.1f}%)  {bar}")
        stats[emo] = {"count": c, "ratio": round(c / max(total, 1), 4)}
    print(f"  {'TOPLAM':12s} {total:7d}")
    print(f"{'═'*60}")

    # Kaynak dağılımı
    src_counts = Counter(df["source"])
    print("\n  Kaynak Dağılımı:")
    for src, cnt in sorted(src_counts.items(), key=lambda x: -x[1]):
        print(f"    {src:40s}: {cnt:7d}")

    return stats


# ─── Ana Fonksiyon ───────────────────────────────────────────────────────────

def build_master_dataset(
    force_rebuild: bool = False,
    max_per_class: Optional[int] = None,
    min_text_len: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Tüm kaynakları birleştirir, temizler, label_id ekler ve kaydeder.

    Args:
        force_rebuild: True ise önbellek yoksayılır.
        max_per_class: Sınıf başına maksimum örnek sayısı (None = sınırsız).
        min_text_len: Minimum metin uzunluğu (karakter).
        seed: Rastgelelik tohumu.

    Returns:
        pd.DataFrame — 'text', 'label', 'label_id', 'source' sütunları
    """
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = processed_dir / "master_dataset.parquet"
    out_stats   = processed_dir / "master_stats.json"
    out_weights = processed_dir / "sampler_weights.npy"

    if out_parquet.exists() and not force_rebuild:
        print(f"[CACHE] Mevcut veri yükleniyor: {out_parquet}")
        df = pd.read_parquet(out_parquet)
        print_stats(df, "MEVCUT VERİ DAĞILIMI")
        return df

    print("\n" + "="*60)
    print("  MASTER DATA ENGINE — Veri Yükleme Başlıyor")
    print("="*60)

    # ── 1. Tüm kaynaklardan veri yükle ──────────────────────────
    loaders = [
        ("TREMO",        load_tremo),
        ("TURTED",       load_turted),
        ("winvoker",     load_winvoker),
        ("TTC4900",      load_ttc4900),
        ("GoEmotions TR", load_goemotions_tr),
        ("Ek Kaynaklar", load_extra_sources),
    ]

    frames = []
    for name, loader_fn in loaders:
        print(f"\n[{name}] Yükleniyor...")
        try:
            df_src = loader_fn()
            if df_src is not None and len(df_src) > 0:
                frames.append(df_src)
                print(f"  ✓ {len(df_src)} örnek eklendi")
            else:
                print(f"  ✗ Boş veri döndü")
        except Exception as e:
            print(f"  ✗ HATA: {e}")

    if not frames:
        raise RuntimeError("Hiçbir kaynaktan veri yüklenemedi!")

    # ── 2. Birleştir ──────────────────────────────────────────────
    df = pd.concat(frames, ignore_index=True)
    print(f"\n[BİRLEŞTİRME] Toplam ham örnek: {len(df)}")

    # ── 3. Temizle ────────────────────────────────────────────────
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= min_text_len]
    df = df.dropna(subset=["label"])
    df = df[df["label"].isin(EMOTIONS)]

    # Duplikat kaldır
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    print(f"[TEMİZLEME] {before - len(df)} duplikat kaldırıldı. Kalan: {len(df)}")

    # ── 4. Sınıf başına kap (opsiyonel) ──────────────────────────
    if max_per_class is not None:
        rng = np.random.default_rng(seed)
        parts = []
        for emo in EMOTIONS:
            subset = df[df["label"] == emo]
            if len(subset) > max_per_class:
                idx = rng.choice(len(subset), max_per_class, replace=False)
                subset = subset.iloc[idx]
            parts.append(subset)
        df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"[KAPLAMA] max_per_class={max_per_class} → {len(df)} örnek")

    # ── 5. Label ID ekle ──────────────────────────────────────────
    df["label_id"] = df["label"].map(LABEL2ID)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)

    # Sıra karıştır
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # ── 6. WeightedRandomSampler ağırlıkları ──────────────────────
    label_ids = df["label_id"].values
    sample_weights = compute_sample_weights(label_ids)
    np.save(out_weights, sample_weights)
    print(f"[AĞIRLIK] Sampler ağırlıkları kaydedildi: {out_weights}")

    # ── 7. İstatistikleri kaydet ──────────────────────────────────
    stats = print_stats(df, "MASTER VERİ DAĞILIMI (NİHAİ)")

    # Sınıf ağırlıklarını da kaydet
    counts = np.bincount(label_ids, minlength=NUM_LABELS).astype(float)
    counts = np.where(counts == 0, 1, counts)
    class_weights = len(label_ids) / (counts * NUM_LABELS)

    full_stats = {
        "total": len(df),
        "num_labels": NUM_LABELS,
        "emotions": EMOTIONS,
        "label2id": LABEL2ID,
        "distribution": stats,
        "class_weights": {
            EMOTIONS[i]: round(float(class_weights[i]), 4)
            for i in range(NUM_LABELS)
        },
        "source_counts": dict(Counter(df["source"])),
    }

    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(full_stats, f, ensure_ascii=False, indent=2)
    print(f"[İSTATİSTİK] Kaydedildi: {out_stats}")

    # ── 8. Parquet'e kaydet ────────────────────────────────────────
    df.to_parquet(out_parquet, index=False)
    print(f"\n[KAYIT] Master veri seti kaydedildi: {out_parquet}")
    print(f"[KAYIT] Toplam: {len(df)} örnek | {NUM_LABELS} sınıf\n")

    return df


# ─── Eğitime Hazır Splitter ──────────────────────────────────────────────────

def make_splits(
    df: Optional[pd.DataFrame] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple:
    """
    Stratified train/val/test split.

    Returns:
        (df_train, df_val, df_test)
    """
    from sklearn.model_selection import train_test_split

    if df is None:
        df = build_master_dataset()

    df_trainval, df_test = train_test_split(
        df, test_size=test_size,
        stratify=df["label_id"], random_state=seed,
    )
    val_ratio = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_trainval, test_size=val_ratio,
        stratify=df_trainval["label_id"], random_state=seed,
    )

    splits_dir = ROOT / "data" / "splits" / "master_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    for name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
        path = splits_dir / f"{name}.parquet"
        subset.reset_index(drop=True).to_parquet(path, index=False)

    # Orijinal master-dataset indekslerini kaydet — WeightedRandomSampler için gerekli.
    # sampler_weights.npy tam dataset pozisyonuna göre hesaplanır; reset_index sonrası bu
    # bilgi kaybolur. Dosyadan yükleyerek doğru ağırlıkları seçmek için kullanılır.
    np.save(splits_dir / "train_orig_indices.npy", df_train.index.to_numpy())
    np.save(splits_dir / "val_orig_indices.npy",   df_val.index.to_numpy())
    np.save(splits_dir / "test_orig_indices.npy",  df_test.index.to_numpy())

    print(f"\n[SPLIT] Train:{len(df_train)}  Val:{len(df_val)}  Test:{len(df_test)}")
    print(f"[SPLIT] Orijinal indeksler kaydedildi: {splits_dir}/*_orig_indices.npy")
    print(f"[SPLIT] Dosyalar: {splits_dir}")

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


# ─── PyTorch Dataset Sınıfı ─────────────────────────────────────────────────

class TurkishEmotionDataset:
    """
    Hafif PyTorch Dataset sarmalayıcısı.
    from torch.utils.data import DataLoader ile kullanılır.

    Örnek:
        from data_engine import TurkishEmotionDataset, build_master_dataset
        from torch.utils.data import DataLoader, WeightedRandomSampler
        import numpy as np

        df_train, df_val, df_test = make_splits()
        # make_splits() reset_index(drop=True) ile döndürür — orijinal indeksler kaybolur.
        # Ağırlıklar için make_splits() tarafından kaydedilen indeks dosyasını kullan:
        weights = np.load("data/processed/sampler_weights.npy")
        # reset_index(drop=True) sonrası index 0'dan başlar, weights doğru eşleşir
        train_orig_idx = np.load("data/splits/master_splits/train_orig_indices.npy")
        train_weights = weights[train_orig_idx]  # orijinal master-dataset pozisyonları
        train_ds = TurkishEmotionDataset(df_train, tokenizer, max_length=128)
        sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
        loader  = DataLoader(train_ds, batch_size=64, sampler=sampler, pin_memory=True)
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.texts = df["text"].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", encoding["input_ids"] * 0).squeeze(0),
            "labels":         self.labels[idx],
        }


# ─── Konfigürasyon Yardımcısı ────────────────────────────────────────────────

def get_class_weights_tensor(stats_path: Optional[Path] = None):
    """
    WeightedCrossEntropyLoss için sınıf ağırlıklarını torch.Tensor olarak döndürür.
    Kullanım:
        weights = get_class_weights_tensor()
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    """
    import torch
    if stats_path is None:
        stats_path = ROOT / "data" / "processed" / "master_stats.json"

    if not stats_path.exists():
        raise FileNotFoundError(f"İstatistik dosyası yok: {stats_path}  (önce build_master_dataset çalıştırın)")

    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)

    weights = [stats["class_weights"][emo] for emo in EMOTIONS]
    return torch.tensor(weights, dtype=torch.float32)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Master Data Engine — 10 Sınıf Türkçe Duygu")
    parser.add_argument("--rebuild",       action="store_true", help="Önbelleği yoksay, sıfırdan oluştur")
    parser.add_argument("--max-per-class", type=int, default=None, help="Sınıf başına max örnek")
    parser.add_argument("--split",         action="store_true", help="Train/val/test split oluştur")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    df = build_master_dataset(
        force_rebuild=args.rebuild,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )

    if args.split:
        make_splits(df, seed=args.seed)
