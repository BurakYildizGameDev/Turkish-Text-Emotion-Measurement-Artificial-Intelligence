"""
Sentiment (olumlu/olumsuz) verisinden kural tabanlı duygu etiketi türetme.

v1'deki hata: tüm olumlu yorumlar "mutluluk", tüm olumsuz yorumlar "üzüntü" sayılıyordu.
Burada bir yoruma ancak şu koşullarda duygu etiketi verilir:

  1. Polarite kısıtı — olumlu yorum yalnızca {mutluluk, sevgi, şaşkınlık},
     olumsuz yorum yalnızca {öfke, üzüntü, korku, tiksinti, şaşkınlık} olabilir.
  2. Metin, izin verilen duygulardan TAM OLARAK BİRİNE ait açık bir ifade içermelidir.
     Hiç eşleşme yoksa veya birden fazla duygu eşleşiyorsa örnek atılır.
  3. Eşleşmeden hemen sonra olumsuzlama ("değil", "etmeyin", ...) geliyorsa eşleşme sayılmaz.

Bu etiketler "weak" kalitededir: yalnızca train'de kullanılır, val/test'e girmez.
Kuralların isabeti, altın veride (TREMO/tweet/GoEmotions) ölçülüp build raporuna yazılır.

Eşleştirme hem Türkçe hem ASCII yazımı yakalamak için metin ve kalıplar ASCII'ye
katlanarak yapılır ("üzgünüm" ve "uzgunum" aynı kalıba düşer).
"""

from __future__ import annotations

import re

from src.data.labels import tr_lower

_ASCII = str.maketrans("çğıöşüâîû", "cgiosuaiu")

POSITIVE_ALLOWED = ("mutluluk", "sevgi", "şaşkınlık")
NEGATIVE_ALLOWED = ("öfke", "üzüntü", "korku", "tiksinti", "şaşkınlık")

# Her kalıp kelime başına sabitlenir. Kalıplar Türkçe yazılır, derlenirken ASCII'ye katlanır.
_LEXICON: dict[str, list[str]] = {
    "mutluluk": [
        r"mutlu(yum|yuz|ydum|ydu|luk|lukla)?\b",
        r"sevin(dim|dik|diriyor|dirdi|dirici|çliyim|çli|ç|çle)\b",
        r"memnun(um|uz|dum|duk|iyet)?\b",
        r"memnun kal(dım|dık)\b",
        r"keyif(le|li|liydi)\b",
        r"eğlen(celi|dim|dik|ceydi)\b",
        r"harika(ydı|dır|sınız)?\b",
        r"mükemmel\w*",
        r"muhteşem\w*",
        r"(çok|gayet|son derece) güzel\w*",
        r"teşekkür(ler| ederim| ediyorum)\b",
        # GoEmotions'ta admiration → mutluluk; gold train'de "hayran" %80 mutluluk
        r"hayran (kaldım|kaldık|oldum)\b",
        r"hayranıyım\b",
    ],
    # Kalıplar gold TRAIN split'inde ölçülerek seçildi (test'e bakılmadı):
    #   "(çok|en) sevdiğim" %36, "aşık oldum" %23 sevgi isabetiyle çıkarıldı.
    "sevgi": [
        r"sev(iyorum|iyoruz|dim|dik)\b",
        r"bayıl(dım|dık|ıyorum|ıyoruz)\b",
        r"aşığım\b",
        r"tapıyorum\b",
    ],
    "şaşkınlık": [
        r"şaşır(dım|dık|ttı|tı|tıcı|tıcıydı)\b",
        r"şaşkın(ım|ız|lık|lıkla)?\b",
        r"beklemiyordum\b",
        r"beklemediğim\b",
        r"inanamadım\b",
        r"hayret(ler)?\b",
        r"şok (oldum|olduk|edici)\b",
        r"vay (be|canına)\b",
    ],
    "öfke": [
        r"rezalet\w*",
        r"sinir(lendim|lendik|lendirdi|lendirici|imi bozdu|imi bozuyor| bozucu|liyim| oldum)\b",
        r"kızgın\w*",
        r"kız(dım|dık|ıyorum)\b",
        r"öfke\w*",
        r"çıldır(dım|acağım|tıyor|ttı|ttınız)\b",
        r"saygısız\w*",
        r"dolandırıcı\w*",
        r"dolandır(dı|dılar|ıldım|ılıyor)\b",
        r"yazıklar olsun\b",
        r"utanmaz\w*",
        r"kandır(dı|dılar|ıldım|ıyorlar|ıyorsunuz)\b",
        r"terbiyesiz\w*",
        r"sahtekar\w*",
        r"gerizekalı\w*",
        r"allah kahretsin\b",
    ],
    "üzüntü": [
        r"üzgün\w*",
        r"üzül(düm|dük|üyorum|üyoruz)\b",
        r"üzücü\w*",
        r"hayal kırıklığı\w*",
        r"maalesef\b",
        r"ne yazık ki\b",
        r"yazık oldu\b",
        r"keşke\b",
        r"pişman(ım|ız|dım|dık|lık)?\b",
        r"ağla(dım|dık|ttı|yarak|mak)\b",
        r"hüzün\w*",
        r"kahr(oldum|olduk)\b",
        r"moral(im|imiz) bozuldu\b",
    ],
    "tiksinti": [
        r"iğrenç\w*",
        r"iğren(dim|dik|iyorum)\b",
        r"tiksin\w*",
        r"tiksindir\w*",
        r"mide(mi)? bulandır\w*",
        r"midem bulandı\b",
        r"kus(tum|acak|acaktım|turucu)\b",
        r"leş\b",
        r"(pis|kötü|berbat|ağır) (bir )?koku\w*",
        r"kokuyor\b",
    ],
    "korku": [
        r"kork(tum|tuk|uyorum|uyoruz|arım|utucu|utucuydu)\b",
        r"endişe(li|leniyorum|lendim|lendik)?\b",
        r"kaygı(lı|lıyım|landım)?\b",
        r"tedirgin\w*",
        r"panik(ledim|ledik)?\b",
        r"dehşet\w*",
        r"ürkütücü\b",
        r"tehlike(li)?\b",
        r"yangın çık\w*",
        r"elektrik çarp\w*",
    ],
}

# Eşleşmeden önce metinden silinen yanıltıcı ifadeler
#   "korku filmi" türdür, duygu değildir; "korkunç" ve "inanılmaz" çoğu zaman pekiştireçtir.
_STOP_PHRASES = [r"korku (film|filmi|filmleri|filmlerini|türü)\w*", r"korkunç\w*", r"inanılmaz\w*"]

_NEGATIONS = {
    "değil", "değildi", "değilim", "değiliz", "yok", "etmeyin", "etmedim", "etmedik",
    "olmadım", "olmadık", "olmadı", "kalmadım", "kalmadık", "kalmadı",
}


def _fold(text: str) -> str:
    return tr_lower(text).translate(_ASCII)


_COMPILED: dict[str, re.Pattern] = {
    emo: re.compile(r"(?<!\w)(?:" + "|".join(_fold(p) for p in pats) + r")")
    for emo, pats in _LEXICON.items()
}
_STOP_RE = re.compile(r"(?<!\w)(?:" + "|".join(_fold(p) for p in _STOP_PHRASES) + r")")
_NEG_FOLDED = {_fold(n) for n in _NEGATIONS}


def _negated(text: str, end: int) -> bool:
    """Eşleşmeden sonraki ilk iki kelimede olumsuzlama var mı?"""
    following = text[end:end + 40].split()[:2]
    return any(w.strip(".,!?;:") in _NEG_FOLDED for w in following)


def matched_emotions(text: str, allowed: tuple[str, ...] | None = None) -> set[str]:
    """Metinde açık ifadesi bulunan (ve olumsuzlanmamış) duyguların kümesi."""
    t = _STOP_RE.sub(" ", _fold(text))
    found = set()
    for emo, pat in _COMPILED.items():
        if allowed is not None and emo not in allowed:
            continue
        if any(not _negated(t, m.end()) for m in pat.finditer(t)):
            found.add(emo)
    return found


def weak_label(text: str, polarity: str) -> str | None:
    """
    polarity: "positive" | "negative". Tek bir duygu eşleşirse onu, aksi halde None döndürür.
    """
    allowed = POSITIVE_ALLOWED if polarity == "positive" else NEGATIVE_ALLOWED
    found = matched_emotions(text, allowed)
    return found.pop() if len(found) == 1 else None
