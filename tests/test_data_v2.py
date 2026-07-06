# Commit 203: test(v2): add tests for Turkish case folding and deduplication key generation
"""v2 veri hattı testleri: python -m pytest tests/"""

import pytest

from src.data.labels import EMOTIONS, dedup_key, tr_lower
from src.data.sources import GO_NAMES, GO_TO_V2
from src.data.weak_labeling import weak_label


@pytest.mark.parametrize("text, polarity, expected", [
    ("Üründen çok memnunum, herkese tavsiye ederim", "positive", "mutluluk"),
    ("Bu telefona bayıldım", "positive", "sevgi"),
    ("memnun değilim, kargo rezalet", "negative", "öfke"),
    ("Maalesef beklediğim gibi çıkmadı", "negative", "üzüntü"),
    ("Ürün iğrenç kokuyor", "negative", "tiksinti"),
    ("şarj ederken ısınıyor, yangın çıkacak diye korkuyorum", "negative", "korku"),
    ("hic beklemiyordum bu kadar iyi olacagini", "positive", "şaşkınlık"),
    ("filmi izledim cok uzuldum sonunda agladim", "negative", "üzüntü"),   # ASCII yazım
    ("Işık çok güzel", "positive", "mutluluk"),                           # Türkçe büyük I
    # Etiket verilmemesi gerekenler
    ("korku filmi diye gittik komedi çıktı", "negative", None),           # tür, duygu değil
    ("korkunç bir film, vakit kaybı", "negative", None),                  # pekiştireç
    ("inanılmaz güzel bir ürün", "positive", None),                       # pekiştireç
    ("endişe etmeyin ürün sağlam", "positive", None),                     # olumsuzlama
    ("harika bir ürün çok sevdim", "positive", None),                     # iki duygu → belirsiz
    ("çok rezalet", "positive", None),                                    # polarite kısıtı
    ("beğenmedim", "negative", None),                                     # açık duygu ifadesi yok
])
def test_weak_label(text, polarity, expected):
    assert weak_label(text, polarity) == expected


def test_label_schema():
    assert len(EMOTIONS) == 8
    assert "gurur" not in EMOTIONS and "utanç" not in EMOTIONS


def test_config_matches_labels():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["emotions"]["labels"] == EMOTIONS
    assert {int(k): v for k, v in cfg["emotions"]["id2label"].items()} == dict(enumerate(EMOTIONS))


def test_goemotions_mapping_complete():
    assert set(GO_TO_V2) == set(GO_NAMES) and len(GO_NAMES) == 28
    assert set(v for v in GO_TO_V2.values() if v) <= set(EMOTIONS)
    assert GO_TO_V2["pride"] is None and GO_TO_V2["embarrassment"] is None


def test_turkish_lower_and_dedup_key():
    assert tr_lower("UMUT IŞIK İZMİR") == "umut ışık izmir"
    assert dedup_key("Harika!!  Çok   GÜZEL.") == dedup_key("harika çok güzel")
