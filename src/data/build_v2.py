# Commit 199: feat(v2/data): integrate MinHash-LSH near-duplicate clustering into build_v2.py
"""
v2 veri seti derleyicisi — 8 sınıflı Türkçe duygu veri seti.

Kullanım:
    python -m src.data.build_v2

Çıktılar (data/v2/):
    train.parquet         gold, %70  ─┐
    val.parquet           gold, %15   ├ grup-farkındalıklı, kaynak×etiket tabakalı
    test.parquet          gold, %15  ─┘
    train_weak.parquet    sentiment verisinden kural tabanlı etiket — yalnızca eğitim
    train_silver.parquet  kaynağı doğrulanamayan veri (nihal_8class) — varsayılan olarak kullanılmaz
    build_report.json     tüm sayılar, atılma nedenleri, kural isabeti

Tasarım ilkeleri (v1 hatalarına karşılık):
  - val/test yalnızca insan etiketli (gold) veriden oluşur; weak/silver asla test'e girmez.
  - Tam kopyalar normalize edilmiş metinle bulunur; aynı metin farklı etiketle geliyorsa
    hepsi atılır (v1 sessizce ilkini tutuyordu).
  - Yakın kopyalar TÜM çiftler arasında MinHash-LSH ile gruplanır; grup bütünüyle tek
    split'e gider (v1 yalnızca train↔test çiftlerine bakıyordu).
  - weak/silver örneklerinden val/test'e tam veya yakın kopyası olanlar atılır.
  - Her adım deterministiktir (sabit seed, Python hash() kullanılmaz).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import train_test_split

from src.data import sources
from src.data.labels import EMOTIONS, LABEL2ID, dedup_key
from src.data.weak_labeling import matched_emotions, weak_label

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "v2"

SEED = 42
NUM_PERM = 128
LSH_THRESHOLD = 0.8     # karakter 5-gram Jaccard
SHINGLE = 5
WEAK_MAX_CHARS = 500    # etiketi belirleyen kelime 128 token'ın dışında kalmasın
SOURCE_PRIORITY = {"tremo": 0, "tweet_emotion": 1, "goemotions_tr": 2}
MOVIE_SOURCES = {"winvoker/HUMIR", "mteb_movie"}


def log(msg: str) -> None:
    print(msg, flush=True)


# ─── Yakın kopya ─────────────────────────────────────────────────────────────

def minhash(key: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM, seed=SEED)
    s = key if len(key) >= SHINGLE else key.ljust(SHINGLE)
    m.update_batch([s[i:i + SHINGLE].encode("utf-8") for i in range(len(s) - SHINGLE + 1)])
    return m


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def near_dup_groups(keys: list[str]) -> np.ndarray:
    """Her örnek için yakın-kopya grup kimliği (tüm çiftler arasında)."""
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    uf = UnionFind(len(keys))
    for i, k in enumerate(keys):
        m = minhash(k)
        for j in lsh.query(m):
            uf.union(i, int(j))
        lsh.insert(str(i), m)
    return np.array([uf.find(i) for i in range(len(keys))])


# ─── Tam kopya ───────────────────────────────────────────────────────────────

def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aynı anahtar + farklı etiket → hepsi atılır. Aynı etiket → öncelikli kaynak tutulur."""
    df = df.assign(_key=df["text"].map(dedup_key))
    df = df[df["_key"].str.len() > 0]
    n_labels = df.groupby("_key")["label"].transform("nunique")
    conflict = df[n_labels > 1]
    df = df[n_labels == 1]
    before = len(df)
    df = (df.assign(_prio=df["source"].map(SOURCE_PRIORITY).fillna(9))
            .sort_values(["_prio"], kind="stable")
            .drop_duplicates("_key")
            .drop(columns="_prio"))
    return df, {
        "celiskili_etiket_ornek": int(len(conflict)),
        "celiskili_etiket_metin": int(conflict["_key"].nunique()),
        "tam_kopya_atilan": int(before - len(df)),
    }


# ─── Split ───────────────────────────────────────────────────────────────────

def group_split(df: pd.DataFrame) -> pd.Series:
    """Grupları source|label ile tabakalı 70/15/15 böler. Döndürür: her satır için split adı."""
    g = (df.groupby("_group")
           .agg(strat=("strat", "first"))
           .reset_index())
    counts = g["strat"].value_counts()
    # 3'ten az grubu olan tabakalar tabakalamayı bozar → etiket düzeyine indirgenir
    g.loc[g["strat"].map(counts) < 3, "strat"] = g["strat"].str.split("|").str[1]
    tr, rest = train_test_split(g, test_size=0.30, stratify=g["strat"], random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=rest["strat"], random_state=SEED)
    split_of = {**dict.fromkeys(tr["_group"], "train"),
                **dict.fromkeys(va["_group"], "val"),
                **dict.fromkeys(te["_group"], "test")}
    return df["_group"].map(split_of)


# ─── Kural isabeti ───────────────────────────────────────────────────────────

def evaluate_rules(gold: pd.DataFrame) -> dict:
    """
    weak_labeling kurallarını altın veride çalıştırır. Polarite altın etiketten türetilir
    (şaşkınlık ve nötr için kısıt yok). Tek eşleşme veren örneklerde isabet ölçülür.
    """
    pos = {"mutluluk", "sevgi"}
    rows = []
    for text, label in zip(gold["text"], gold["label"]):
        if label in pos:
            pred = weak_label(text, "positive")
        elif label in ("öfke", "üzüntü", "korku", "tiksinti"):
            pred = weak_label(text, "negative")
        else:
            found = matched_emotions(text)
            pred = found.pop() if len(found) == 1 else None
        rows.append((label, pred))
    ev = pd.DataFrame(rows, columns=["gold", "pred"])
    labeled = ev.dropna(subset=["pred"])
    per_class = {}
    for emo in EMOTIONS:
        p = labeled[labeled["pred"] == emo]
        per_class[emo] = {
            "tahmin_sayisi": int(len(p)),
            "isabet": round(float((p["gold"] == emo).mean()), 3) if len(p) else None,
            "kapsama": round(float((labeled["gold"] == emo).sum() / max((ev["gold"] == emo).sum(), 1)), 3),
        }
    return {
        "aciklama": "Kurallar altin veriye uygulandi; isabet = kuralin verdigi etiketin altin etiketle uyusma orani.",
        "etiketlenen_oran": round(len(labeled) / len(ev), 3),
        "genel_isabet": round(float((labeled["gold"] == labeled["pred"]).mean()), 3),
        "sinif_bazli": per_class,
        "karisiklik": {f"{g}->{p}": int(n) for (g, p), n in
                       labeled[labeled["gold"] != labeled["pred"]].value_counts().head(15).items()},
    }


# ─── Ana akış ────────────────────────────────────────────────────────────────

def build(include_silver: bool = True) -> dict:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"seed": SEED, "etiketler": EMOTIONS,
                    "parametreler": {"lsh_threshold": LSH_THRESHOLD, "num_perm": NUM_PERM,
                                     "shingle_char": SHINGLE, "weak_max_chars": WEAK_MAX_CHARS}}

    # 1. Gold kaynaklar
    log("[1/6] Gold kaynaklar yükleniyor...")
    tremo = sources.load_tremo()
    tweet = sources.load_tweet_emotion()
    goemo, go_stats = sources.load_goemotions_tr()
    gold = pd.concat([tremo, tweet, goemo], ignore_index=True)
    report["kaynak_ham"] = {
        "tremo": {"kalan": len(tremo), **tremo["subsource"].value_counts().to_dict()},
        "tweet_emotion": {"kalan": len(tweet)},
        "goemotions_tr": go_stats,
    }
    log(f"      tremo={len(tremo):,}  tweet={len(tweet):,}  goemotions={len(goemo):,}")

    # 2. Tam kopya + çelişkili etiket
    log("[2/6] Tam kopyalar ve çelişkili etiketler temizleniyor...")
    gold, dup_stats = drop_exact_duplicates(gold)
    report["gold_tekillestirme"] = dup_stats
    log(f"      {dup_stats}")

    # 3. Yakın kopya grupları + split
    log(f"[3/6] Yakın kopya grupları (MinHash-LSH, {len(gold):,} örnek)...")
    gold = gold.reset_index(drop=True)
    gold["_group"] = near_dup_groups(gold["_key"].tolist())
    sizes = gold["_group"].value_counts()
    report["yakin_kopya"] = {"grup_sayisi": int(len(sizes)),
                             "coklu_grup": int((sizes > 1).sum()),
                             "coklu_gruptaki_ornek": int(sizes[sizes > 1].sum())}
    gold["strat"] = gold["source"] + "|" + gold["label"]
    gold["split"] = group_split(gold)
    assert gold.groupby("_group")["split"].nunique().max() == 1, "grup split'ler arasında bölündü"

    # 4. Weak (sentiment → duygu)
    log("[4/6] Sentiment verisi yükleniyor ve kurallarla etiketleniyor...")
    sent = sources.load_all_sentiment()
    report["weak_ham"] = {"toplam": len(sent), "kaynak": sent["subsource"].value_counts().to_dict()}
    sent = sent[sent["text"].str.len() <= WEAK_MAX_CHARS].copy()
    report["weak_ham"]["uzunluk_filtresi_sonrasi"] = len(sent)
    sent["label"] = [weak_label(t, p) for t, p in zip(sent["text"], sent["polarity"])]
    weak = sent.dropna(subset=["label"]).copy()
    # Film yorumlarında korku kelimeleri çoğunlukla filmin içeriğini anlatır
    # ("dehşetli sahneler"), yorumcunun duygusunu değil → bu kaynaklarda korku atılır.
    movie_fear = weak["subsource"].isin(MOVIE_SOURCES) & (weak["label"] == "korku")
    weak = weak[~movie_fear]
    report["weak_ham"]["film_korku_atilan"] = int(movie_fear.sum())
    weak["quality"] = "weak"
    weak["translated"] = False
    weak["orig_label"] = weak["polarity"]
    weak = weak[sources.COLUMNS]
    report["weak_ham"]["etiketlenen"] = len(weak)

    frames = {"weak": weak}
    if include_silver:
        log("      silver (nihal_8class) yükleniyor...")
        frames["silver"] = sources.load_nihal_8class()

    # 5. weak/silver: kendi içinde tekilleştir, gold ile tam/yakın kopyaları at
    log("[5/6] weak/silver tekilleştirme ve val/test sızıntı kontrolü...")
    gold_keys = set(gold["_key"])
    heldout = gold[gold["split"] != "train"]
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    for i, k in enumerate(heldout["_key"]):
        lsh.insert(str(i), minhash(k))
    report["ek_egitim_verisi"] = {}
    for name, df in frames.items():
        n0 = len(df)
        df, st = drop_exact_duplicates(df)
        in_gold = df["_key"].isin(gold_keys)
        df = df[~in_gold]
        near = np.array([bool(lsh.query(minhash(k))) for k in df["_key"]], dtype=bool)
        df = df[~near]
        frames[name] = df
        report["ek_egitim_verisi"][name] = {
            "baslangic": n0, **st,
            "gold_ile_tam_kopya": int(in_gold.sum()),
            "val_test_ile_yakin_kopya": int(near.sum()),
            "kalan": len(df),
        }
        log(f"      {name}: {report['ek_egitim_verisi'][name]}")

    # 6. Kaydet + rapor
    log("[6/6] Kaydediliyor...")

    def finalize(df: pd.DataFrame) -> pd.DataFrame:
        out = df[sources.COLUMNS].copy()
        out["label_id"] = out["label"].map(LABEL2ID).astype(int)
        return out.reset_index(drop=True)

    for split in ("train", "val", "test"):
        finalize(gold[gold["split"] == split]).to_parquet(OUT_DIR / f"{split}.parquet", index=False)
    finalize(frames["weak"]).to_parquet(OUT_DIR / "train_weak.parquet", index=False)
    if "silver" in frames:
        finalize(frames["silver"]).to_parquet(OUT_DIR / "train_silver.parquet", index=False)

    def dist(df: pd.DataFrame) -> dict:
        return {"toplam": len(df),
                "sinif": {e: int((df["label"] == e).sum()) for e in EMOTIONS},
                "kaynak_sinif": {s: {e: int(n) for e, n in g["label"].value_counts().items()}
                                 for s, g in df.groupby("subsource")}}

    report["dagilim"] = {s: dist(gold[gold["split"] == s]) for s in ("train", "val", "test")}
    report["dagilim"]["train_weak"] = dist(frames["weak"])
    if "silver" in frames:
        report["dagilim"]["train_silver"] = dist(frames["silver"])

    log("      kural isabeti altın veride ölçülüyor...")
    report["kural_isabeti"] = evaluate_rules(gold)
    report["sure_sn"] = round(time.time() - t0, 1)
    (OUT_DIR / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"Bitti ({report['sure_sn']} sn) → {OUT_DIR}")
    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="v2 8-sınıf Türkçe duygu veri seti")
    ap.add_argument("--no-silver", action="store_true", help="nihal_8class dosyasını üretme")
    build(include_silver=not ap.parse_args().no_silver)
