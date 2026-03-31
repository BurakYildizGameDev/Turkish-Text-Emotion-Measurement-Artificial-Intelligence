"""
10-Sinifli Turkce Duygu Analizi - Veri Toplama Modulu
======================================================
Kaynaklar (calisma durumu dogrulanmis):
  1. AnasAlokla/multilingual_go_emotions  -- 43k Turkce, 28 GoEmotions etiketi
  2. winvoker/turkish-sentiment-analysis-dataset -- 489k, Positive/Negative/Notr
  3. fthbrmnby/turkish_product_reviews    -- 235k, pozitif/negatif urun yorumlari
  4. boun-tabilab/Turkish-Product-Reviews -- 235k, pozitif/negatif urun yorumlari
  5. savasy/ttc4900                       -- 4.9k, Turkce haber metinleri (notr)
  6. mteb/turkish_product_sentiment       -- 5.6k
  7. mteb/turkish_movie_sentiment         -- 10.5k

Hedef: 10 duygu sinifi, 500k+ satir
  mutluluk | uzuntu | ofke | korku | saskınlık | tiksinti | sevgi | notr | utanc | gurur
"""

import os
import json
import ast
import yaml
import pandas as pd
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from collections import Counter

ROOT       = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"


# ─── Config ──────────────────────────────────────────────────
def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── GoEmotions (28 sinif) -> 10 Duygu ───────────────────────
# https://arxiv.org/abs/2005.00547
GO_TO_10 = {
    0:  "mutluluk",   # admiration
    1:  "mutluluk",   # amusement
    2:  "ofke",       # anger
    3:  "ofke",       # annoyance
    4:  "mutluluk",   # approval
    5:  "sevgi",      # caring
    6:  "saskınlık",  # confusion
    7:  "saskınlık",  # curiosity
    8:  "sevgi",      # desire
    9:  "uzuntu",     # disappointment
    10: "ofke",       # disapproval
    11: "tiksinti",   # disgust
    12: "utanc",      # embarrassment
    13: "mutluluk",   # excitement
    14: "korku",      # fear
    15: "sevgi",      # gratitude
    16: "uzuntu",     # grief
    17: "mutluluk",   # joy
    18: "sevgi",      # love
    19: "korku",      # nervousness
    20: "mutluluk",   # optimism
    21: "gurur",      # pride
    22: "saskınlık",  # realization
    23: "mutluluk",   # relief
    24: "uzuntu",     # remorse
    25: "uzuntu",     # sadness
    26: "saskınlık",  # surprise
    27: "notr",       # neutral
}

# Tum kaynaklarda gecerli siniflar (config ile eslesecek)
LABEL_NORMALIZE = {
    # winvoker
    "positive": "mutluluk",
    "negative": "uzuntu",
    "notr":     "notr",
    "neutral":  "notr",
    # genel
    "mutluluk":  "mutluluk",
    "uzuntu":    "uzuntu",
    "ofke":      "ofke",
    "korku":     "korku",
    "saskınlık": "saskınlık",
    "tiksinti":  "tiksinti",
    "sevgi":     "sevgi",
    "notr":      "notr",
    "utanc":     "utanc",
    "gurur":     "gurur",
    # binary sentiment (urun/film yorumlari)
    "pos":       "mutluluk",
    "neg":       "uzuntu",
    1:            "mutluluk",
    0:            "uzuntu",
    "1":          "mutluluk",
    "0":          "uzuntu",
}

# config'deki Turkce etiket -> normalize etikete
TURKCE_TO_NORM = {
    "mutluluk":  "mutluluk",
    "uzuntu":    "uzuntu",
    "ofke":      "ofke",
    "korku":     "korku",
    "saskınlık": "saskınlık",
    "tiksinti":  "tiksinti",
    "sevgi":     "sevgi",
    "notr":      "notr",
    "utanc":     "utanc",
    "gurur":     "gurur",
}

# config label2id anahtarlarini normalized forma cevir
CONFIG_NORM_TO_LABEL = {
    "mutluluk":  "mutluluk",
    "uzuntu":    "uzuntu",
    "ofke":      "ofke",
    "korku":     "korku",
    "saskınlık": "saskınlık",
    "tiksinti":  "tiksinti",
    "sevgi":     "sevgi",
    "notr":      "notr",
    "utanc":     "utanc",
    "gurur":     "gurur",
}


# ─── Kaynak 1: GoEmotions TR ──────────────────────────────────
def load_goemotions_tr() -> pd.DataFrame:
    print("\n[1/7] AnasAlokla/multilingual_go_emotions (TR) indiriliyor...")
    ds = load_dataset("AnasAlokla/multilingual_go_emotions")
    split = list(ds.keys())[0]
    all_rows = []
    for i in range(len(ds[split])):
        row = ds[split][i]
        if row.get("language") != "tr":
            continue
        text = row.get("text")
        if not text or not isinstance(text, str):
            continue
        text = text.strip()
        if len(text) < 10:
            continue
        try:
            raw_labels = row.get("labels", "[]")
            go_ids = ast.literal_eval(raw_labels) if isinstance(raw_labels, str) else raw_labels
        except Exception:
            continue
        # Tum etiketleri ekle (coklu etiket destegi)
        added = False
        for gid in go_ids:
            norm = GO_TO_10.get(int(gid))
            if norm:
                all_rows.append({"text": text, "label_norm": norm, "source": "goemotions_tr"})
                added = True
                break
    df = pd.DataFrame(all_rows)
    print(f"   -> {len(df)} ornek (TR filtresi sonrasi)")
    return df


# ─── Kaynak 2: winvoker ───────────────────────────────────────
def load_winvoker() -> pd.DataFrame:
    print("\n[2/7] winvoker/turkish-sentiment-analysis-dataset indiriliyor...")
    ds = load_dataset("winvoker/turkish-sentiment-analysis-dataset")
    frames = []
    for sname in ds.keys():
        df = ds[sname].to_pandas()[["text", "label"]]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["label_norm"] = df["label"].str.lower().str.strip().map(
        lambda x: LABEL_NORMALIZE.get(x)
    )
    df = df.dropna(subset=["label_norm"])
    df["source"] = "winvoker"
    print(f"   -> {len(df)} ornek")
    return df[["text", "label_norm", "source"]]


# ─── Kaynak 3: fthbrmnby/turkish_product_reviews ─────────────
def load_fthbrmnby() -> pd.DataFrame:
    print("\n[3/7] fthbrmnby/turkish_product_reviews indiriliyor...")
    ds = load_dataset("fthbrmnby/turkish_product_reviews")
    split = list(ds.keys())[0]
    df = ds[split].to_pandas().rename(columns={"sentence": "text", "sentiment": "label"})
    df["label_norm"] = df["label"].map(LABEL_NORMALIZE)
    df = df.dropna(subset=["label_norm", "text"])
    df["source"] = "fthbrmnby"
    print(f"   -> {len(df)} ornek")
    return df[["text", "label_norm", "source"]]


# ─── Kaynak 4: boun-tabilab/Turkish-Product-Reviews ──────────
def load_boun() -> pd.DataFrame:
    print("\n[4/7] boun-tabilab/Turkish-Product-Reviews indiriliyor...")
    ds = load_dataset("boun-tabilab/Turkish-Product-Reviews")
    frames = [ds[s].to_pandas() for s in ds.keys()]
    df = pd.concat(frames, ignore_index=True)
    df["label_norm"] = df["label"].map(LABEL_NORMALIZE)
    df = df.dropna(subset=["label_norm", "text"])
    df["source"] = "boun"
    print(f"   -> {len(df)} ornek")
    return df[["text", "label_norm", "source"]]


# ─── Kaynak 5: savasy/ttc4900 (haber -> notr) ────────────────
def load_ttc4900() -> pd.DataFrame:
    print("\n[5/7] savasy/ttc4900 indiriliyor (notr metin)...")
    ds = load_dataset("savasy/ttc4900")
    split = list(ds.keys())[0]
    df = ds[split].to_pandas()
    # Tum kategoriler notr olarak etiketlenir
    df["label_norm"] = "notr"
    df["source"] = "ttc4900"
    df = df.rename(columns={"text": "text"})
    df = df[df["text"].str.len() >= 20]
    print(f"   -> {len(df)} ornek (hepsi notr)")
    return df[["text", "label_norm", "source"]]


# ─── Kaynak 6: mteb/turkish_product_sentiment ─────────────────
def load_mteb_product() -> pd.DataFrame:
    print("\n[6/7] mteb/turkish_product_sentiment indiriliyor...")
    ds = load_dataset("mteb/turkish_product_sentiment")
    frames = [ds[s].to_pandas() for s in ds.keys()]
    df = pd.concat(frames, ignore_index=True)
    df["label_norm"] = df["label"].map(LABEL_NORMALIZE)
    df = df.dropna(subset=["label_norm", "text"])
    df["source"] = "mteb_product"
    print(f"   -> {len(df)} ornek")
    return df[["text", "label_norm", "source"]]


# ─── Kaynak 7: mteb/turkish_movie_sentiment ───────────────────
def load_mteb_movie() -> pd.DataFrame:
    print("\n[7/7] mteb/turkish_movie_sentiment indiriliyor...")
    ds = load_dataset("mteb/turkish_movie_sentiment")
    frames = [ds[s].to_pandas() for s in ds.keys()]
    df = pd.concat(frames, ignore_index=True)
    df["label_norm"] = df["label"].map(LABEL_NORMALIZE)
    df = df.dropna(subset=["label_norm", "text"])
    df["source"] = "mteb_movie"
    print(f"   -> {len(df)} ornek")
    return df[["text", "label_norm", "source"]]


# ─── Birlestir & Normalize ────────────────────────────────────
def combine_sources(dfs: list) -> pd.DataFrame:
    df = pd.concat(dfs, ignore_index=True)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 10]
    # Duplikatlari kaldir
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    print(f"\n[DEDUP] {before - len(df)} duplikat kaldirildi. Kalan: {len(df)}")
    return df.reset_index(drop=True)


# ─── Dagilim Raporu ───────────────────────────────────────────
def print_distribution(df: pd.DataFrame, title: str = ""):
    print(f"\n{'='*55}")
    if title:
        print(f"  {title}")
    print(f"{'='*55}")
    total = len(df)
    counts = Counter(df["label_norm"])
    for label in sorted(counts.keys()):
        c = counts[label]
        bar = "#" * int(c / total * 40)
        print(f"  {label:12s}: {c:7d} ({c/total*100:5.1f}%)  {bar}")
    print(f"  {'TOPLAM':12s}: {total:7d}")
    print(f"{'='*55}")

    print("\n  Kaynak dagilimi:")
    for src, cnt in sorted(Counter(df["source"]).items(), key=lambda x: -x[1]):
        print(f"    {src:35s}: {cnt:7d}")


# ─── Label ID Ekle ────────────────────────────────────────────
def add_label_ids(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    label2id = config["emotions"]["label2id"]
    # config anahtarlari Turkce, normalize etiketi eslestir
    norm_to_id = {
        "mutluluk":  label2id["mutluluk"],
        "uzuntu":    label2id["\u00fcz\u00fcnt\u00fc"],
        "ofke":      label2id["\u00f6fke"],
        "korku":     label2id["korku"],
        "saskınlık": label2id["\u015fa\u015fk\u0131nl\u0131k"],
        "tiksinti":  label2id["tiksinti"],
        "sevgi":     label2id["sevgi"],
        "notr":      label2id["n\u00f6tr"],
        "utanc":     label2id["utan\u00e7"],
        "gurur":     label2id["gurur"],
    }
    df["label_id"] = df["label_norm"].map(norm_to_id)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)
    return df


# ─── Train / Val / Test Split ─────────────────────────────────
def split_dataset(df: pd.DataFrame, config: dict) -> DatasetDict:
    test_size = config["data"]["test_size"]
    val_size  = config["data"]["val_size"]
    seed      = config["training"]["seed"]

    df_trainval, df_test = train_test_split(
        df, test_size=test_size, stratify=df["label_id"], random_state=seed
    )
    val_ratio = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_trainval, test_size=val_ratio, stratify=df_trainval["label_id"], random_state=seed
    )
    print(f"\n[SPLIT] Train:{len(df_train)}  Val:{len(df_val)}  Test:{len(df_test)}")
    return DatasetDict({
        "train":      Dataset.from_pandas(df_train.reset_index(drop=True)),
        "validation": Dataset.from_pandas(df_val.reset_index(drop=True)),
        "test":       Dataset.from_pandas(df_test.reset_index(drop=True)),
    })


# ─── Kaydet ───────────────────────────────────────────────────
def save_splits(dataset_dict: DatasetDict, config: dict):
    splits_dir = ROOT / config["data"]["splits_dir"]
    splits_dir.mkdir(parents=True, exist_ok=True)
    out = splits_dir / "tremo_splits"
    dataset_dict.save_to_disk(str(out))
    print(f"[SAVE] Split'ler kaydedildi: {out}")

    stats = {}
    for sname, sdata in dataset_dict.items():
        df_s = sdata.to_pandas()
        stats[sname] = {
            "total": len(df_s),
            "distribution": df_s["label_norm"].value_counts().to_dict(),
            "sources": df_s["source"].value_counts().to_dict(),
        }
    spath = splits_dir / "dataset_stats.json"
    with open(spath, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] Istatistikler kaydedildi: {spath}")


# ─── Ana Fonksiyon ────────────────────────────────────────────
def prepare_dataset() -> DatasetDict:
    config  = load_config()
    raw_dir = ROOT / config["data"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache   = raw_dir / "combined_raw.parquet"

    # Ham birlesik veriyi cache'le
    if cache.exists():
        print(f"[CACHE] Ham veri mevcut: {cache}")
        df_raw = pd.read_parquet(cache)
    else:
        dfs = []
        loaders = [
            load_goemotions_tr,
            load_winvoker,
            load_fthbrmnby,
            load_boun,
            load_ttc4900,
            load_mteb_product,
            load_mteb_movie,
        ]
        for fn in loaders:
            try:
                dfs.append(fn())
            except Exception as e:
                print(f"[WARN] {fn.__name__} basarisiz: {e}")

        df_raw = combine_sources(dfs)
        df_raw.to_parquet(cache, index=False)
        print(f"[SAVE] Ham veri kaydedildi: {cache}")

    print_distribution(df_raw, "HAM VERI DAGILIMI (7 KAYNAK)")

    # Label ID ekle
    df = add_label_ids(df_raw, config)

    # Islemis veriyi kaydet
    processed_dir = ROOT / config["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    clean_path = processed_dir / "combined_clean.parquet"
    df.to_parquet(clean_path, index=False)
    print(f"[SAVE] Temiz veri kaydedildi: {clean_path}")

    print_distribution(df, "TEMIZLENMIS VERI DAGILIMI")

    # Split
    dataset_dict = split_dataset(df, config)

    # Kaydet
    save_splits(dataset_dict, config)

    return dataset_dict


if __name__ == "__main__":
    prepare_dataset()
