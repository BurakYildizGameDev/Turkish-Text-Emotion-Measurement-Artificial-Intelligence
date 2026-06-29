# Commit 190: feat(v2/labels): add tr_lower, clean_text, and dedup_key Unicode utilities
"""
Etiket şemasının tek kaynağı (v2 veri seti).

Tüm v2 scriptleri etiketleri buradan import etmelidir; dosya başına ayrı
ASCII_TO_ID sözlükleri tutulmaz.

v1'den farklar:
  - gurur ve utanç çıkarıldı (test'te 8 ve 37 örnek vardı; tek kaynak makine çevirisiydi)
  - etiketler her yerde Unicode Türkçe yazılır ("nötr", "şaşkınlık"); ASCII varyant kullanılmaz
"""

from __future__ import annotations

import re

EMOTIONS: list[str] = [
    "mutluluk", "üzüntü", "öfke", "korku", "şaşkınlık", "tiksinti", "sevgi", "nötr",
]
LABEL2ID: dict[str, int] = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL: dict[int, str] = {i: e for i, e in enumerate(EMOTIONS)}
NUM_LABELS: int = len(EMOTIONS)

# Etiket kalitesi — split ve eğitimde hangi örneğin nerede kullanılabileceğini belirler.
#   gold   : insan etiketli (TREMO, tweet, GoEmotions). val/test yalnızca bunlardan oluşur.
#   weak   : sentiment verisinden kural tabanlı türetilmiş etiket. Yalnızca train.
#   silver : kaynağı/etiketleme süreci doğrulanamayan veri. Varsayılan olarak kullanılmaz.
QUALITIES = ("gold", "weak", "silver")

# Python'un str.lower() fonksiyonu Türkçe I/İ'yi yanlış çevirir ("I" -> "i", "İ" -> "i̇").
_TR_LOWER = str.maketrans("IİÇĞÖŞÜ", "ıiçğöşü")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def tr_lower(text: str) -> str:
    """Türkçe kurallarına uygun küçük harfe çevirme."""
    return text.translate(_TR_LOWER).lower()


def clean_text(text: str) -> str:
    """Model girdisi için hafif temizlik: URL ve @mention silinir, boşluklar sadeleşir."""
    text = _URL_RE.sub(" ", str(text))
    text = _MENTION_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def dedup_key(text: str) -> str:
    """Tekilleştirme anahtarı: küçük harf, noktalama yok, tek boşluk."""
    text = _PUNCT_RE.sub(" ", tr_lower(text))
    return _SPACE_RE.sub(" ", text).strip()
