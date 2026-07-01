"""
v2 veri kaynakları. Her yükleyici aynı şemada DataFrame döndürür:

    text        : temizlenmiş metin
    label       : src.data.labels.EMOTIONS içinden etiket
    source      : kaynak adı (tremo, tweet_emotion, goemotions_tr, ...)
    subsource   : kaynak içi alt küme (ör. winvoker/urun_yorumlari) — yoksa source ile aynı
    quality     : gold | weak | silver  (bkz. labels.QUALITIES)
    translated  : metin makine çevirisi mi
    orig_label  : kaynaktaki ham etiket (izlenebilirlik için)

Sentiment yükleyicileri (load_sentiment_*) ek olarak `polarity` sütunu döndürür ve
etiketlemeyi build_v2.py, weak_labeling.weak_label ile yapar.
"""

from __future__ import annotations

import ast
import glob
import os
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd

from src.data.labels import EMOTIONS, clean_text

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
HF_HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))

COLUMNS = ["text", "label", "source", "subsource", "quality", "translated", "orig_label"]


def _finalize(df: pd.DataFrame, min_chars: int) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df[df["text"].str.len() >= min_chars]
    assert df["label"].isin(EMOTIONS).all(), set(df["label"]) - set(EMOTIONS)
    return df.reset_index(drop=True)


# ─── TREMO ────────────────────────────────────────────────────────────────────
# Tocoglu & Alpkocak (2018), J. Inf. Sci. — ticari olmayan kullanım içindir, bu yüzden
# ham dosya repoya konmaz; ilk çalıştırmada Kaggle'dan indirilir.
TREMO_URL = "https://www.kaggle.com/api/v1/datasets/download/mansuralp/tremo"
TREMO_DIR = RAW_DIR / "tremo"
TREMO_MAP = {
    "Happy": "mutluluk", "Sadness": "üzüntü", "Anger": "öfke",
    "Fear": "korku", "Disgust": "tiksinti", "Surprise": "şaşkınlık",
}


def _ensure_tremo() -> Path:
    xml_path = TREMO_DIR / "TREMODATA.xml"
    if xml_path.exists():
        return xml_path
    TREMO_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = TREMO_DIR / "tremo.zip"
    print(f"  [TREMO] İndiriliyor: {TREMO_URL}")
    urllib.request.urlretrieve(TREMO_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extract("TREMODATA.xml", TREMO_DIR)
    return xml_path


def load_tremo() -> pd.DataFrame:
    """
    ValidatedEmotion kullanılır (3 annotatörün doğruladığı etiket). "Ambigious"
    (Condition=Reject, 1.361 kayıt) atılır. Uzlaşı türü subsource'a yazılır:
    tremo/consensus (3/3) veya tremo/majority (2/3).
    """
    root = ET.parse(_ensure_tremo()).getroot()
    rows = []
    for doc in root:
        validated = (doc.findtext("ValidatedEmotion") or "").strip()
        if validated not in TREMO_MAP:
            continue
        cond = (doc.findtext("Condition") or "").strip()
        rows.append({
            "text": doc.findtext("Entry") or "",
            "label": TREMO_MAP[validated],
            "source": "tremo",
            "subsource": "tremo/consensus" if cond == "Consensus" else "tremo/majority",
            "quality": "gold",
            "translated": False,
            "orig_label": validated,
        })
    return _finalize(pd.DataFrame(rows, columns=COLUMNS), min_chars=3)


# ─── Türkçe tweet duygu veri seti ────────────────────────────────────────────
# Güven, Diri & Çakaloğlu (2019), ASYU. HF yükleyicisi dosyayı okuyamadığı için CSV
# doğrudan ayrıştırılır (satır sonundaki son virgülden sonrası etikettir).
TWEET_MAP = {"kizgin": "öfke", "korku": "korku", "mutlu": "mutluluk",
             "surpriz": "şaşkınlık", "uzgun": "üzüntü"}


def _hf_snapshot_file(repo: str, pattern: str) -> Path:
    files = glob.glob(str(HF_HUB / f"datasets--{repo.replace('/', '--')}" / "snapshots" / "*" / pattern))
    if not files:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo, pattern, repo_type="dataset"))
    return Path(files[0])


def load_tweet_emotion() -> pd.DataFrame:
    path = _hf_snapshot_file("anilguven/turkish_tweet_emotion_dataset", "Turkish_Tweet_Dataset.csv")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "," not in line:
            continue
        text, raw = line.rsplit(",", 1)
        raw = raw.strip()
        if raw in TWEET_MAP:
            rows.append({"text": text, "label": TWEET_MAP[raw], "source": "tweet_emotion",
                         "subsource": "tweet_emotion", "quality": "gold",
                         "translated": False, "orig_label": raw})
    return _finalize(pd.DataFrame(rows, columns=COLUMNS), min_chars=5)


# ─── GoEmotions TR ───────────────────────────────────────────────────────────
GO_NAMES = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
]
# GoEmotions makalesindeki Ekman gruplaması temel alınır; "sevgi" sınıfı için love/caring ayrılır.
# None = atılır:
#   pride, embarrassment      → gurur/utanç sınıfları kaldırıldı
#   confusion, curiosity,
#   realization               → şaşkınlık ile kavramsal olarak farklı (v1'de tartışmalı eşlemeydi)
#   approval, desire          → belirgin bir duygu sınıfına karşılık gelmiyor
GO_TO_V2 = {
    "admiration": "mutluluk", "amusement": "mutluluk", "excitement": "mutluluk",
    "joy": "mutluluk", "optimism": "mutluluk", "relief": "mutluluk", "gratitude": "mutluluk",
    "love": "sevgi", "caring": "sevgi",
    "anger": "öfke", "annoyance": "öfke", "disapproval": "öfke",
    "sadness": "üzüntü", "grief": "üzüntü", "disappointment": "üzüntü", "remorse": "üzüntü",
    "fear": "korku", "nervousness": "korku",
    "disgust": "tiksinti",
    "surprise": "şaşkınlık",
    "neutral": "nötr",
    "pride": None, "embarrassment": None, "confusion": None, "curiosity": None,
    "realization": None, "approval": None, "desire": None,
}
assert set(GO_TO_V2) == set(GO_NAMES)


def load_goemotions_tr() -> tuple[pd.DataFrame, dict]:
    """
    Yalnızca etiket-tutarlı örnekler alınır: tüm ham etiketleri AYNI v2 sınıfına
    eşlenen ve hiçbiri atılan bir etiket olmayan örnekler. (v1 çok etiketli örneklerde
    sadece ilk etiketi alıyordu.) İkinci dönüş değeri atılma istatistiğidir.
    """
    from datasets import load_dataset
    ds = load_dataset("AnasAlokla/multilingual_go_emotions")
    df = ds[list(ds.keys())[0]].to_pandas()
    df = df[df["language"] == "tr"]
    stats = {"toplam_tr": len(df), "atilan_etiket_iceriyor": 0, "celiskili_cok_etiket": 0}
    rows = []
    for text, raw in zip(df["text"], df["labels"]):
        ids = ast.literal_eval(raw) if isinstance(raw, str) else list(raw)
        names = [GO_NAMES[int(i)] for i in ids]
        targets = {GO_TO_V2[n] for n in names}
        if None in targets:
            stats["atilan_etiket_iceriyor"] += 1
            continue
        if len(targets) != 1:
            stats["celiskili_cok_etiket"] += 1
            continue
        rows.append({"text": text, "label": targets.pop(), "source": "goemotions_tr",
                     "subsource": "goemotions_tr", "quality": "gold",
                     "translated": True, "orig_label": "+".join(names)})
    out = _finalize(pd.DataFrame(rows, columns=COLUMNS), min_chars=5)
    stats["kalan"] = len(out)
    return out, stats


# ─── nihalenc 8 sınıf (silver) ───────────────────────────────────────────────
# Etiketleme süreci ve kaynağı belgelenmemiş; bir kısmı GoEmotions'ın farklı bir çevirisi
# gibi görünüyor (aynı İngilizce cümlenin başka çevirisi → metin eşleşmesiyle yakalanamayan
# test sızıntısı riski). Bu yüzden "silver": varsayılan eğitimde kullanılmaz.
NIHAL_MAP = {"mutluluk": "mutluluk", "uzuntu": "üzüntü", "ofke": "öfke", "korku": "korku",
             "saskinlik": "şaşkınlık", "igrenme": "tiksinti",
             "minnet": None, "pismanlik": None}   # minnet/pişmanlık 8 sınıfta karşılıksız


def load_nihal_8class() -> pd.DataFrame:
    path = _hf_snapshot_file("nihalenc/turkish-8class-emotion-dataset", "turkish-8class-emotion-dataset.csv")
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", on_bad_lines="skip", engine="python")
    label_cols = [c for c in df.columns if c in NIHAL_MAP]
    df = df[df[label_cols].sum(axis=1) == 1]
    df["orig_label"] = df[label_cols].idxmax(axis=1)
    df["label"] = df["orig_label"].map(NIHAL_MAP)
    df = df.dropna(subset=["label"])
    df = df.assign(source="nihal_8class", subsource="nihal_8class", quality="silver", translated=None)
    return _finalize(df[COLUMNS], min_chars=5)


# ─── Sentiment kaynakları (weak etiketleme için) ─────────────────────────────
SENT_COLUMNS = ["text", "polarity", "source", "subsource"]


def load_sentiment_winvoker() -> pd.DataFrame:
    """
    winvoker/turkish-sentiment-analysis-dataset ile birebir aynı içeriğe sahip, ancak alt
    kaynak sütunu da bulunan WhiteAngelss kopyası kullanılır. Atılanlar:
      wiki   (170.413) — "Notr" sınıfının tamamı Wikipedia cümleleri, duygu nötrü değil
      random (504)     — "Integer non velit." gibi anlamsız metinler, "Notr" etiketli
    """
    from datasets import load_dataset
    ds = load_dataset("WhiteAngelss/Turkce-Duygu-Analizi-Dataset")
    df = pd.concat([ds[k].to_pandas() for k in ds], ignore_index=True)
    df = df[df["label"].isin(["Positive", "Negative"]) & ~df["dataset"].isin(["wiki", "random"])]
    return pd.DataFrame({
        "text": df["text"].values,
        "polarity": df["label"].str.lower().values,
        "source": "winvoker",
        "subsource": ("winvoker/" + df["dataset"]).values,
    })


def load_sentiment_binary(repo: str, name: str, text_col: str, label_col: str) -> pd.DataFrame:
    """0 = olumsuz, 1 = olumlu (örneklerle doğrulandı)."""
    from datasets import load_dataset
    ds = load_dataset(repo)
    df = pd.concat([ds[k].to_pandas() for k in ds], ignore_index=True)
    assert set(df[label_col].unique()) <= {0, 1}, df[label_col].unique()
    return pd.DataFrame({
        "text": df[text_col].values,
        "polarity": df[label_col].map({0: "negative", 1: "positive"}).values,
        "source": name, "subsource": name,
    })


def load_all_sentiment() -> pd.DataFrame:
    parts = [
        load_sentiment_winvoker(),
        load_sentiment_binary("mteb/turkish_product_sentiment", "mteb_product", "text", "label"),
        load_sentiment_binary("mteb/turkish_movie_sentiment", "mteb_movie", "text", "label"),
        # fthbrmnby ve boun-tabilab aynı 235.165 satırlık veridir; biri yeterli
        load_sentiment_binary("fthbrmnby/turkish_product_reviews", "fthbrmnby", "sentence", "sentiment"),
    ]
    df = pd.concat(parts, ignore_index=True)
    df["text"] = df["text"].astype(str).map(clean_text)
    return df[df["text"].str.len() >= 15].reset_index(drop=True)
