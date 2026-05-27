"""
Experiment 3 — BERTurk + Lexicon Informed Architecture
=======================================================
Hibrit Model:
  BERTurk [CLS] (768-dim)
       +
  Lexicon (NRC categories) Features (10-dim)  ← Türkçe kelime sözlüğü
       ↓  Concatenate
  778-dim vektör
       ↓  Linear(778, 10) + Dropout
  10 sınıf çıkışı

NRC Kategorileri (sabit sıra):
  0:anger  1:anticipation  2:disgust  3:fear  4:joy
  5:negative  6:positive  7:sadness  8:surprise  9:trust

ÖNEMLİ: Sözlük, NRC Emotion Lexicon'un kendisi DEĞİLDİR. NRC'nin 10 kategori
şeması kullanılarak elle hazırlanmış ~600 kelimelik bir Türkçe listedir
(_build_turkish_lexicon). Raporlarda "NRC Lexicon" diye anılmamalıdır.

Exp3'ün amacı: Kelime düzeyindeki duygu bilgisinin BERT'in
bağlamsal temsilini ne kadar güçlendirdiğini ölçmek.

Çıktılar (results/exp3/):
  confusion_matrix.png
  training_progress.png
  radar_chart.png          ← Exp1 + Exp2 + Exp3 karşılaştırması
  lexicon_importance.png   ← Lexikon feature analizi
  metrics_report.json      ← Exp1/2/3 karşılaştırma özeti
  best_model/
"""

import os
import sys
import json
import re
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from math import pi
from pathlib import Path
from typing import Optional, Dict, List, Any
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, WeightedRandomSampler

from transformers import (
    AutoTokenizer,
    BertModel,
    BertConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
import evaluate
from training_analytics import EnhancedEpochLogger as EpochLogger, run_full_analytics

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ─── Sabitler ───────────────────────────────────────────────────────────────

ROOT      = Path(__file__).resolve().parent
RESULTS   = ROOT / "results" / "exp3"
EXP1_JSON = ROOT / "results" / "exp1" / "metrics_report.json"
EXP2_JSON = ROOT / "results" / "exp2" / "metrics_report.json"
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
SEED       = 42

EMOTIONS = [
    "mutluluk", "üzüntü", "öfke", "korku",
    "şaşkınlık", "tiksinti", "sevgi", "nötr", "utanç", "gurur",
]
ASCII_TO_ID = {
    "mutluluk": 0, "uzuntu": 1, "ofke": 2, "korku": 3,
    "saskınlık": 4, "tiksinti": 5, "sevgi": 6, "notr": 7,
    "utanc": 8, "gurur": 9,
    "üzüntü": 1, "öfke": 2, "şaşkınlık": 4, "nötr": 7, "utanç": 8,
}
LABEL2ID   = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL   = {i: e for i, e in enumerate(EMOTIONS)}
NUM_LABELS = 10
LEXICON_DIM = 10

# NRC kategorileri (sabit sıra — model mimarisini belirler)
NRC_CATS = [
    "anger",        # 0
    "anticipation", # 1
    "disgust",      # 2
    "fear",         # 3
    "joy",          # 4
    "negative",     # 5
    "positive",     # 6
    "sadness",      # 7
    "surprise",     # 8
    "trust",        # 9
]


# ─── Türkçe NRC Duygu Sözlüğü ───────────────────────────────────────────────
# Format: { kelime: [a,ant,d,f,j,neg,pos,s,sur,tr] }  (her değer 0.0-1.0)
# Yaklaşık 600 kelime — 10 NRC kategorisi eşlemesi

def _build_turkish_lexicon() -> Dict[str, np.ndarray]:
    """
    Kategorilere ayrılmış Türkçe kelime listelerinden
    kelime -> 10-dim NRC vektörü haritası oluşturur.
    """
    # Her liste: (kelime, [a,ant,d,f,j,neg,pos,s,sur,tr]) şeklinde
    # a=anger ant=anticipation d=disgust f=fear j=joy
    # neg=negative pos=positive s=sadness sur=surprise tr=trust

    # --- ANGER (0) ----
    anger_words = [
        "kızgın","sinirli","öfkeli","öfke","kızdım","kızıyorum","kızıyor",
        "nefret","nefreti","hayal kırıklığı","bezdim","bıktım","lanet",
        "saldırı","saldırgan","kavga","dövüş","bağırdım","bağırıyorum",
        "küfür","hakaret","aşağıladı","incitti","kırıcı","rezalet",
        "haksız","zorba","zorbalık","taciz","rahatsız","iğneledi",
        "kışkırttı","saldırdı","tehdit","şiddet","şiddetli","agresif",
        "sert","acımasız","vahşet","gaddar","zalim","zalimlik","cani",
        "hunhar","intikam","öç","kin","düşmanca","galeyan","hiddet",
        "gazap","hınç","kin","isyan","kıyamet","taşkınlık","taşkın",
    ]

    # --- ANTICIPATION (1) ---
    anticipation_words = [
        "umut","umutlu","bekliyorum","bekliyoruz","bekledi","merak",
        "meraklı","ileriye","gelecek","merakla","heyecanla","sabırsız",
        "belki","umarım","inşallah","ümit","hayaller","plan","hazırlık",
        "yakında","hedef","amaç","niyet","girişim","proje","fırsat",
        "şans","olasılık","seçenek","keşke","keşif","deney","araştırma",
        "sorgu","soru","ilginç","enteresan","çarpıcı","geliştirme",
        "ilerleme","büyüme","dönüşüm","değişim","yenilik","yeni",
        "modernleşme","evrim","atılım","sıçrama","ivme","potansiyel",
        "beklenti","arzu","istek","dilek","temennisi",
    ]

    # --- DISGUST (2) ---
    disgust_words = [
        "iğrenç","tiksinç","iğrenti","tiksinti","mide bulandırıcı",
        "pis","kirli","aşağılık","alçak","rezil","rezillik","kötü koku",
        "utanç verici","tiksintili","kusacağım","midem bulandı",
        "çirkin","çirkef","leş","kokmuş","bozulmuş","çürümüş",
        "pislik","kirletilmiş","ahlaksız","ahlaksızlık","sapkın",
        "sapkınlık","kötülük","iblis","şeytan","şeytanca","iğreniyor",
        "tiksiniyorum","tiksiniyor","murdarlık","necaset","murdarca",
        "kötü","namussuz","namussuzluk","hain","hainlik","düşkün",
        "düşkünlük","müstehcen","edepsiz","edepsizlik","adi","adice",
    ]

    # --- FEAR (3) ---
    fear_words = [
        "korku","korkuyorum","endişe","kaygı","dehşet","panik",
        "tedirgin","korktum","ürktüm","panikledim","tehlike",
        "tehlikeli","tehdit","korkutucu","ölüm","yaralanma","felaket",
        "titredim","titriyorum","kaçtım","saklandım","gizlendim",
        "çığlık","haykırdım","alarm","uyarı","risk","güvensiz",
        "korumasız","savunmasız","çaresiz","tuzak","baskı","eziliyorum",
        "korkunç","korkunçluk","dehşetli","dehşetengiz","ürkütücü",
        "telaş","telaşlı","heyecan","heyecanlı","sinirlilik","gerginlik",
        "stres","stresli","anksiyete","fobi","fobik","paranoya",
        "paranoyak","şüphe","kuşku","vesvese","endişeli","kaygılı",
    ]

    # --- JOY (4) ---
    joy_words = [
        "mutlu","sevinç","neşe","güzel","harika","mükemmel","süper",
        "muhteşem","keyif","şen","coşku","sevindim","sevindi","seviniyor",
        "mutluluk","neşeli","kutlama","bayram","güldüm","eğlence","zevk",
        "hoş","tatlı","sevimli","memnun","başarı","kazandım","kazandı",
        "yendi","galip","şampiyon","gülümsüyorum","gülümsedim",
        "neşeyle","mutlulukla","coşkuyla","sevgiyle","minnetle",
        "minnet","teşekkür","tebrikler","aferin","bravo","harikasın",
        "güzellik","zafer","şükran","şükür","sevindirici","tatmin",
        "tatminkâr","başarılı","olumlu","güzel haber","iyi haber",
        "muhteşem","olağanüstü","nefis","lezzetli","keyifli","eğlenceli",
        "hoşça","keyifle","zevkle","eğlenerek","coşarak","sevinçle",
    ]

    # --- NEGATIVE (5) ---
    negative_words = [
        "kötü","berbat","yanlış","hata","sorun","problem","başarısız",
        "olmadı","yapamadım","kaybettim","mahvoldu","olmaz","imkansız",
        "çöktü","battı","işe yaramaz","faydasız","boşuna","nafile",
        "beyhude","bozuldu","hasar","zarar","kayıp","ziyan","acıklı",
        "üzücü","talihsiz","şanssız","bedbaht","mutsuz","mutsuzluk",
        "sıkıntı","dert","bela","trajik","facıa","rezalet","utanç",
        "ayıp","kötü","yanlış","hatalı","eksik","yetersiz","zayıf",
        "başarısız","mağlup","yenilgi","kayıp","trajedi","trajik",
        "felâket","mahkum","mahkumiyet","ceza","acı","ıstırap",
    ]

    # --- POSITIVE (6) ---
    positive_words = [
        "iyi","güzel","mükemmel","harika","süper","oldu","başardım",
        "kazandım","evet","kesinlikle","tabii","elbette","doğru",
        "ilerliyorum","güçlüyüm","sağlıklı","başarılı","verimli",
        "üretken","faydalı","yararlı","çözüm","çözüldü","halloldu",
        "tamamlandı","sevindim","memnun","memnunum","memnuniyetle",
        "gururla","onurla","şerefle","saygıyla","sevgiyle","nezaketle",
        "cömertçe","içtenlikle","dürüstçe","şeffafça","adilce",
        "başarıyla","zaferle","muzafferen","galipçe","kazanarak",
        "gelişiyorum","yükseliyorum","büyüyorum","öğreniyorum",
        "keşfediyorum","yaratıyorum","inşa ediyorum","yapıyorum",
        "tamamlıyorum","bitiriyorum","sunuyorum","paylaşıyorum",
    ]

    # --- SADNESS (7) ---
    sadness_words = [
        "üzgün","ağladım","hüzün","mutsuz","acı","yas","yalnız",
        "kayıp","kaybettim","ayrılık","özlem","hasret","dert","çile",
        "elem","ağıt","ağlamak","üzüntü","mahzun","keder","gözyaşı",
        "sızı","ah","vah","yazık","iç sıkıntısı","can sıkıntısı",
        "bunalım","depresyon","yıkıldım","çöküntü","bitkinlik",
        "tükenmişlik","yorgunluk","umutsuzluk","kimsesiz","terkedilmiş",
        "yalnızlık","ıssızlık","boşluk","anlamsızlık","karanlık",
        "sessizlik","derin üzüntü","ağıt","hüzünlü","hüzünlendim",
        "keder","kederli","gamlı","gam","keder","matemli","meyus",
        "umutsuz","ümitsiz","çaresizce","yıkıcı","parçalandım",
    ]

    # --- SURPRISE (8) ---
    surprise_words = [
        "şaşırdım","inanılmaz","beklenmedik","hayret","vay","şaşkın",
        "şok","sürpriz","şaşırıcı","şok edici","beklemiyordum",
        "beklemedim","aniden","birdenbire","nasıl olur","mucize",
        "olağanüstü","fevkalade","olağandışı","alışılmamış","tuhaf",
        "garip","acayip","yabancı","keşfettim","fark ettim","gördüm ki",
        "meğer","meğerki","demek ki","anlıyorum","anladım","kavradım",
        "farkındayım","şaşırıcı","şaşırtıcı","hayret verici","inanmıyorum",
        "inanamıyorum","inanması güç","akıl almaz","akıl sır ermez",
        "gizemli","gizem","sır","duyulmamış","görülmemiş","eşsiz",
        "benzersiz","emsalsiz","nadir","ender","özel","has","özgün",
    ]

    # --- TRUST (9) ---
    trust_words = [
        "güven","inanıyorum","güvenilir","emin","dürüst","adil",
        "şeffaf","inanç","sadakat","bağlılık","destek","güveniyorum",
        "inanıyorum sana","söz veriyorum","taahhüt","garanti","sağlam",
        "güçlü","istikrarlı","tutarlı","bağlı","sadık","dürüstçe",
        "açıkça","şeffaflıkla","destek oluyorum","yanındayım",
        "beraber","birlikte","dayanışma","yardım","yardımcı",
        "işbirliği","ortaklık","beraberlik","ittifak","birlik",
        "dostluk","arkadaşlık","kardeşlik","aile","sevgi","saygı",
        "hürmet","itimat","emniyet","güvence","söz","ahit","yemin",
        "doğruluk","dürüstlük","açıklık","şeffaflık","güvenlik",
    ]

    word_lists = [
        anger_words, anticipation_words, disgust_words, fear_words, joy_words,
        negative_words, positive_words, sadness_words, surprise_words, trust_words,
    ]

    # Türkçe -> ASCII yaklaşık dönüşüm tablosu (çekim eki sorununu azaltır)
    _tr2ascii = str.maketrans(
        "çğışöüÇĞİŞÖÜ",
        "cgisouCGISOI",
    )

    def _variants(w: str):
        """Bir kelime için hem Unicode hem ASCII formunu döndür."""
        return {w, w.translate(_tr2ascii)}

    lexicon: Dict[str, np.ndarray] = {}
    for cat_idx, words in enumerate(word_lists):
        for word in words:
            word = word.strip().lower()
            # çok-kelimeli ifadeleri atla (token bazlı eşleşme yapılamaz)
            if not word or " " in word:
                continue
            for variant in _variants(word):
                if variant not in lexicon:
                    lexicon[variant] = np.zeros(LEXICON_DIM, dtype=np.float32)
                lexicon[variant][cat_idx] = 1.0

    return lexicon


TURKISH_LEXICON: Dict[str, np.ndarray] = _build_turkish_lexicon()
print(f"[LEXIKON] {len(TURKISH_LEXICON)} kelime yüklendi ({LEXICON_DIM} NRC kategorisi)")


# ─── Lexicon Extractor ───────────────────────────────────────────────────────

class LexiconExtractor:
    """
    Bir cümle için 10-boyutlu NRC duygu özellik vektörü hesaplar.

    Yöntem:
      1. Metni tokenize et (küçük harf + noktalama temizle)
      2. Her kelime için sözlükte ara
      3. Eşleşen vektörleri topla
      4. Kelime sayısına göre normalize et (per-word average)
      5. L2 normalize et (birim vektör)
    """

    def __init__(self, lexicon: Optional[Dict[str, np.ndarray]] = None):
        self.lexicon = lexicon if lexicon is not None else TURKISH_LEXICON
        self._punct_re = re.compile(r"[^\w\s]", re.UNICODE)
        # Hızlı kök araması: ilk 4-7 karakter -> lexikon vektörü
        # Türkçe çekim eklerini yaklaşık olarak yönetir
        self._stem_index: Dict[str, np.ndarray] = {}
        for key, vec in self.lexicon.items():
            for stem_len in (4, 5, 6, 7):
                if len(key) >= stem_len:
                    stem = key[:stem_len]
                    if stem not in self._stem_index:
                        self._stem_index[stem] = vec.copy()
                    else:
                        self._stem_index[stem] = self._stem_index[stem] + vec

    # Türkçe büyük harf->küçük harf (İ->i, I->ı dahil)
    _TR_UPPER = str.maketrans("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ",
                               "abcçdefgğhıijklmnoöprsştuüvyz")
    # Türkçe -> ASCII yaklaşık (prefix eşleştirme için)
    _TR_ASCII = str.maketrans("çğışöüîâû", "cgisouiau")

    def _tr_lower(self, text: str) -> str:
        return text.translate(self._TR_UPPER)

    def _tokenize(self, text: str) -> List[str]:
        text = self._tr_lower(text)
        text = self._punct_re.sub(" ", text)
        return [t for t in text.split() if len(t) >= 2]

    def extract(self, text: str) -> np.ndarray:
        """Tek metin -> 10-dim NRC vektörü."""
        tokens = self._tokenize(text)
        vec = np.zeros(LEXICON_DIM, dtype=np.float32)
        n_hits = 0

        for token in tokens:
            hit = None
            # 1. Tam eşleşme (Unicode)
            if token in self.lexicon:
                hit = self.lexicon[token]
            # 2. ASCII-normalize edilmiş tam eşleşme
            elif (ascii_tok := token.translate(self._TR_ASCII)) in self.lexicon:
                hit = self.lexicon[ascii_tok]
            # 3. Hızlı kök araması — token'ın 4-7 karakterlik önek indeksinde ara
            elif len(token) >= 4:
                for stem_len in (7, 6, 5, 4):
                    stem = token[:stem_len] if len(token) >= stem_len else token
                    if stem in self._stem_index:
                        hit = self._stem_index[stem]
                        break

            if hit is not None:
                vec += hit
                n_hits += 1

        # Kelime başına ortalama (sıfır bölmeden kaçın)
        if len(tokens) > 0:
            vec /= len(tokens)

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm

        return vec

    def batch_extract(self, texts: List[str], verbose: bool = True) -> np.ndarray:
        """Metin listesi -> (N, 10) matris."""
        n = len(texts)
        out = np.zeros((n, LEXICON_DIM), dtype=np.float32)
        step = max(1, n // 10)
        for i, text in enumerate(texts):
            out[i] = self.extract(text)
            if verbose and i % step == 0:
                print(f"  Lexicon features: {i}/{n} ({i/n*100:.0f}%)", end="\r")
        if verbose:
            print(f"  Lexicon features: {n}/{n} (100%) - tamamlandi")
        return out

    def coverage_report(self, texts: List[str]) -> dict:
        """Sözlük kapsama istatistikleri."""
        total_tokens = 0
        hit_tokens   = 0
        texts_with_hits = 0
        for text in texts:
            tokens = self._tokenize(text)
            hits = sum(1 for t in tokens if t in self.lexicon)
            total_tokens += len(tokens)
            hit_tokens   += hits
            if hits > 0:
                texts_with_hits += 1
        return {
            "total_texts":      len(texts),
            "texts_with_hits":  texts_with_hits,
            "coverage_rate":    texts_with_hits / max(len(texts), 1),
            "token_hit_rate":   hit_tokens / max(total_tokens, 1),
            "total_tokens":     total_tokens,
            "hit_tokens":       hit_tokens,
        }


# ─── Hibrit Model ────────────────────────────────────────────────────────────

class LexiconBERTModel(nn.Module):
    """
    BERTurk [CLS] (768) ++ Lexikon (10) -> Linear(778, 10) -> Logit

    forward() -> SequenceClassifierOutput (HF Trainer uyumlu)
    """

    def __init__(
        self,
        bert_model_name: str = MODEL_NAME,
        num_labels: int = NUM_LABELS,
        lexicon_dim: int = LEXICON_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_labels  = num_labels
        self.lexicon_dim = lexicon_dim

        self.bert    = BertModel.from_pretrained(bert_model_name)
        hidden_size  = self.bert.config.hidden_size          # 768
        combined_dim = hidden_size + lexicon_dim             # 778

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(combined_dim, num_labels)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        # Model config bilgisi (HF Trainer için)
        self.config = self.bert.config
        self.config.num_labels = num_labels

    def forward(
        self,
        input_ids:        Optional[torch.Tensor] = None,
        attention_mask:   Optional[torch.Tensor] = None,
        token_type_ids:   Optional[torch.Tensor] = None,
        lexicon_features: Optional[torch.Tensor] = None,
        labels:           Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:

        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_vec = self.dropout(bert_out.last_hidden_state[:, 0, :])  # (B, 768)

        if lexicon_features is not None:
            lex = lexicon_features.to(cls_vec.device).float()
            combined = torch.cat([cls_vec, lex], dim=-1)             # (B, 778)
        else:
            # Fallback: lexikon olmadan sadece BERT
            zero_lex = torch.zeros(cls_vec.size(0), self.lexicon_dim, device=cls_vec.device)
            combined = torch.cat([cls_vec, zero_lex], dim=-1)

        logits = self.classifier(combined)                           # (B, 10)

        # Loss yok — WeightedTrainer.compute_loss() hesaplar
        return SequenceClassifierOutput(loss=None, logits=logits)


# Eski transformers sürümlerinin kaydettiği, modelde karşılığı olmayan zararsız buffer'lar
_IGNORABLE_UNEXPECTED = ("position_ids",)


def load_lexicon_bert_weights(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """
    state_dict'i yükler; eksik veya beklenmeyen anahtar varsa hata fırlatır.

    strict=False tek başına kullanıldığında uyumsuz ağırlıklar (ör. farklı mimari)
    sessizce atlanır ve model rastgele classifier ile "eğitilmiş" gibi görünür.
    """
    result = model.load_state_dict(state, strict=False)
    unexpected = [k for k in result.unexpected_keys
                  if not k.endswith(_IGNORABLE_UNEXPECTED)]
    if result.missing_keys or unexpected:
        raise RuntimeError(
            "Ağırlıklar model mimarisiyle uyuşmuyor — "
            f"eksik: {result.missing_keys[:5]} (toplam {len(result.missing_keys)}), "
            f"beklenmeyen: {unexpected[:5]} (toplam {len(unexpected)})"
        )


# ─── Dataset ────────────────────────────────────────────────────────────────

class LexiconEmotionDataset(Dataset):
    """
    Tokenize edilmiş metin + precomputed 10-dim lexikon özelliği döner.
    Lexikon özellikleri init aşamasında toplu hesaplanır (batch_extract).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        extractor: LexiconExtractor,
        max_length: int = 128,
    ):
        self.texts      = df["text"].tolist()
        self.labels     = df["label_id"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

        print(f"  Lexikon özellikleri hesaplanıyor ({len(self.texts)} örnek)...")
        self.lex_feats = extractor.batch_extract(self.texts, verbose=True)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"]           = torch.tensor(self.labels[idx], dtype=torch.long)
        item["lexicon_features"] = torch.tensor(self.lex_feats[idx], dtype=torch.float32)
        return item


# ─── Data Collator ───────────────────────────────────────────────────────────

class LexiconDataCollator(DataCollatorWithPadding):
    """
    DataCollatorWithPadding'i uzatır:
      - Metin token'larını dinamik padding ile toplar
      - lexicon_features tensörünü ayrıca stack eder
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        lex_feats = torch.stack([f.pop("lexicon_features") for f in features])
        batch = super().__call__(features)
        batch["lexicon_features"] = lex_feats
        return batch


# ─── Ağırlık Hesabı ─────────────────────────────────────────────────────────

def compute_class_weights(label_ids: np.ndarray) -> torch.Tensor:
    counts = np.bincount(label_ids, minlength=NUM_LABELS).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    w = len(label_ids) / (counts * NUM_LABELS)
    return torch.tensor(w, dtype=torch.float32)


def compute_sample_weights(label_ids: np.ndarray) -> np.ndarray:
    counts = np.bincount(label_ids, minlength=NUM_LABELS).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    class_w = len(label_ids) / (counts * NUM_LABELS)
    return class_w[label_ids]


# ─── Weighted Trainer ────────────────────────────────────────────────────────

class LexiconWeightedTrainer(Trainer):
    def __init__(self, *args, train_sample_weights=None, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._sample_weights = train_sample_weights
        self._class_weights  = class_weights

    def _get_train_sampler(self, dataset=None):
        if self._sample_weights is not None:
            return WeightedRandomSampler(
                weights=torch.from_numpy(self._sample_weights).double(),
                num_samples=len(self._sample_weights),
                replacement=True,
            )
        return super()._get_train_sampler(dataset)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self._class_weights.to(outputs.logits.device) if self._class_weights is not None else None
        loss = nn.CrossEntropyLoss(weight=weight)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# ─── Epoch Logger ────────────────────────────────────────────────────────────

# ─── Metrik Fonksiyonu ───────────────────────────────────────────────────────

def build_compute_metrics():
    acc_m = evaluate.load("accuracy")
    f1_m  = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds    = np.argmax(logits, axis=-1)
        acc      = acc_m.compute(predictions=preds, references=labels)["accuracy"]
        f1_macro = f1_m.compute(predictions=preds, references=labels, average="macro")["f1"]
        f1_wt    = f1_m.compute(predictions=preds, references=labels, average="weighted")["f1"]
        f1_per   = f1_m.compute(predictions=preds, references=labels, average=None)["f1"]
        per_cls  = {f"f1_{ID2LABEL[i]}": float(f1_per[i]) for i in range(len(f1_per))}
        return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_wt, **per_cls}

    return compute_metrics


# ─── Görselleştirme ──────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    labels_short = [e[:8] for e in EMOTIONS]

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    for ax, data, title, fmt in [
        (axes[0], cm,      "Ham Sayilar",        "d"),
        (axes[1], cm_norm, "Normalize (Oransal)", ".2f"),
    ]:
        sns.heatmap(
            data, annot=True, fmt=fmt,
            xticklabels=labels_short, yticklabels=labels_short,
            cmap="Purples", linewidths=0.4, ax=ax, annot_kws={"size": 8},
        )
        ax.set_xlabel("Tahmin Edilen", fontsize=11)
        ax.set_ylabel("Gercek", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)

    fig.suptitle("Experiment 3 — BERTurk + Lexicon | Confusion Matrix",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Confusion matrix: {save_path}")


def save_training_curves(history: dict, save_path: Path):
    epochs = history["epoch"]
    if not epochs:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history["train_loss"], "o-", color="#e74c3c", label="Train Loss",  lw=2)
    ax1.plot(epochs, history["val_loss"],   "s-", color="#8e44ad", label="Val Loss",    lw=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss Egrisi", fontweight="bold"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, history["val_f1_macro"], "o-", color="#2ecc71", label="Val F1-Macro", lw=2)
    ax2.plot(epochs, history["val_acc"],       "s-", color="#9b59b6", label="Val Accuracy", lw=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Skor"); ax2.set_ylim(0, 1)
    ax2.set_title("Validation Metrikleri", fontweight="bold"); ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.suptitle("Experiment 3 — BERTurk + Lexicon | Egitim Ilerlemesi",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Training curves: {save_path}")


def save_radar_chart(
    exp3_per_class: dict,
    save_path: Path,
):
    """Exp1 / Exp2 / Exp3 — 10 sınıf F1 örümcek grafiği."""

    def _load_per_class(json_path: Path) -> dict:
        if not json_path.exists():
            return {}
        with open(json_path, encoding="utf-8") as f:
            d = json.load(f)
        return {e: d["per_class"].get(e, {}).get("f1", 0.0) for e in EMOTIONS}

    exp1_f1 = _load_per_class(EXP1_JSON)
    exp2_f1 = _load_per_class(EXP2_JSON)
    exp3_f1 = {e: exp3_per_class.get(e, {}).get("f1", 0.0) for e in EMOTIONS}

    labels = [e[:9] for e in EMOTIONS]
    N      = len(labels)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]   # kapat

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=8, color="grey")
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

    specs = [
        (exp1_f1, "#95a5a6", "Exp1 — Baseline"),
        (exp2_f1, "#3498db", "Exp2 — Cross-Lingual"),
        (exp3_f1, "#8e44ad", "Exp3 — Lexicon"),
    ]
    for scores_dict, color, label in specs:
        if not scores_dict:
            continue
        vals = [scores_dict.get(e, 0.0) for e in EMOTIONS]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2.5, linestyle="solid", color=color, label=label)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_title(
        "Experiment 1 / 2 / 3 — 10 Sinif F1 Karsilastirmasi\n(Radar Chart)",
        size=14, fontweight="bold", y=1.08,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Radar chart: {save_path}")


def save_lexicon_importance(
    lex_features: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
) -> dict:
    """
    Lexikon feature analizi:
      A) Per-class ortalama lexikon profili (ısı haritası)
      B) Logistic Regression sadece lexikon ile — separability testi
      C) NRC kategori katkı raporu
    """
    # ── A. Per-class ortalama lexikon profili ─────────────────────
    profile = np.zeros((NUM_LABELS, LEXICON_DIM), dtype=float)
    for cls_id in range(NUM_LABELS):
        mask = (y_true == cls_id)
        if mask.sum() > 0:
            profile[cls_id] = lex_features[mask].mean(axis=0)

    # ── B. Logistic Regression (sadece lexikon) ────────────────────
    mask_nz = np.any(lex_features != 0, axis=1)  # sıfır vektör olmayanlar
    X = lex_features[mask_nz]
    y = y_true[mask_nz]
    lr_f1 = 0.0
    lr_acc = 0.0
    if len(X) > 50:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        lr = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced", random_state=SEED)
        lr.fit(X_tr, y_tr)
        y_pred_lr = lr.predict(X_te)
        lr_f1  = float(f1_score(y_te, y_pred_lr, average="macro",  zero_division=0))
        lr_acc = float(accuracy_score(y_te, y_pred_lr))

    # ── C. Görsel ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Sol: per-class lexikon profili
    ax = axes[0]
    df_hm = pd.DataFrame(
        profile,
        index=[e[:9] for e in EMOTIONS],
        columns=[c[:5] for c in NRC_CATS],
    )
    sns.heatmap(
        df_hm, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, annot_kws={"size": 9},
    )
    ax.set_title("Sinif Basi Ortalama Sozluk Profili (NRC kategorileri)\n(Deger ne kadar yuksek -> o duygu o sınıfta ne kadar belirgin)",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("NRC Kategori", fontsize=10)
    ax.set_ylabel("Duygu Sinifi", fontsize=10)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)

    # Sağ: Logistic Regression vs Hybrid model bar karşılaştırma
    ax2 = axes[1]
    hybrid_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bars = ax2.bar(
        ["Sadece Lexikon\n(Logistic Reg.)", "BERTurk\n+ Lexikon (Exp3)"],
        [lr_f1, hybrid_f1],
        color=["#f39c12", "#8e44ad"],
        width=0.4, edgecolor="white",
    )
    for bar, val in zip(bars, [lr_f1, hybrid_f1]):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.4f}",
                 ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("F1-Macro", fontsize=12)
    ax2.set_title("Lexikon Tek Basina vs\nBERT + Lexikon Entegrasyonu",
                  fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    iyilesme = hybrid_f1 - lr_f1
    sign = "+" if iyilesme >= 0 else ""
    ax2.text(0.5, 0.92, f"BERT katkilari: {sign}{iyilesme:.4f}",
             ha="center", va="center", transform=ax2.transAxes,
             fontsize=11, color="#27ae60" if iyilesme >= 0 else "#c0392b",
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#27ae60" if iyilesme>=0 else "#c0392b"))

    fig.suptitle("Experiment 3 — Lexikon Feature Importance Analizi",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Lexikon importance: {save_path}")

    return {
        "lexicon_only_f1_macro":  round(lr_f1, 4),
        "lexicon_only_accuracy":  round(lr_acc, 4),
        "hybrid_f1_macro":        round(hybrid_f1, 4),
        "bert_contribution":      round(hybrid_f1 - lr_f1, 4),
        "per_class_lexicon_profile": {
            EMOTIONS[i]: {
                NRC_CATS[j]: round(float(profile[i, j]), 4)
                for j in range(LEXICON_DIM)
            }
            for i in range(NUM_LABELS)
        },
    }


def save_metrics_report(
    y_true, y_pred, history: dict, lex_stats: dict, save_path: Path
) -> dict:
    report_dict = classification_report(
        y_true, y_pred, target_names=EMOTIONS, output_dict=True, zero_division=0,
    )
    per_class = {
        emo: {
            "precision": round(report_dict[emo]["precision"], 4),
            "recall":    round(report_dict[emo]["recall"],    4),
            "f1":        round(report_dict[emo]["f1-score"],  4),
            "support":   int(report_dict[emo]["support"]),
        }
        for emo in EMOTIONS if emo in report_dict
    }

    output = {
        "experiment":        "exp3_lexicon_informed",
        "model":             f"{MODEL_NAME} + NRC Turkish Lexicon",
        "architecture":      "BERTurk[CLS](768) ++ Lexikon(10) -> Linear(778,10)",
        "lexicon_dim":       LEXICON_DIM,
        "nrc_categories":    NRC_CATS,
        "test_accuracy":     round(float(accuracy_score(y_true, y_pred)), 4),
        "test_f1_macro":     round(float(f1_score(y_true, y_pred, average="macro",    zero_division=0)), 4),
        "test_f1_weighted":  round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "per_class":         per_class,
        "lexicon_analysis":  lex_stats,
        "training_history":  history,
        "comparison":        {},
    }

    # Exp1 & Exp2 karşılaştırması
    for exp_name, exp_path in [("exp1", EXP1_JSON), ("exp2", EXP2_JSON)]:
        if exp_path.exists():
            with open(exp_path, encoding="utf-8") as f:
                ref = json.load(f)
            delta_macro = output["test_f1_macro"] - ref.get("test_f1_macro", 0)
            sign = "+" if delta_macro >= 0 else ""
            output["comparison"][exp_name] = {
                "f1_macro_ref":   ref.get("test_f1_macro"),
                "f1_macro_exp3":  output["test_f1_macro"],
                "f1_macro_delta": round(delta_macro, 4),
                "per_class_delta": {
                    emo: round(
                        per_class.get(emo, {}).get("f1", 0.0) -
                        ref.get("per_class", {}).get(emo, {}).get("f1", 0.0),
                        4
                    )
                    for emo in EMOTIONS
                },
            }
        else:
            output["comparison"][exp_name] = {"note": f"{exp_path.name} bulunamadi"}

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[VIZ] Metrics report: {save_path}")

    # Konsol özet
    print("\n" + "-"*70)
    print(f"  {'Duygu':12s} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Destek':>8}  vs Exp1  vs Exp2")
    print("-"*70)
    comp1 = output["comparison"].get("exp1", {}).get("per_class_delta", {})
    comp2 = output["comparison"].get("exp2", {}).get("per_class_delta", {})
    for emo in EMOTIONS:
        m  = per_class.get(emo, {})
        d1 = comp1.get(emo, 0.0)
        d2 = comp2.get(emo, 0.0)
        s1 = ("+" if d1 >= 0 else "") + f"{d1:.3f}"
        s2 = ("+" if d2 >= 0 else "") + f"{d2:.3f}"
        print(f"  {emo:12s} {m.get('precision',0):8.4f} {m.get('recall',0):8.4f} "
              f"{m.get('f1',0):8.4f} {m.get('support',0):8d}  {s1:>7}  {s2:>7}")
    print("-"*70)
    dm1 = output["comparison"].get("exp1", {}).get("f1_macro_delta", "N/A")
    dm2 = output["comparison"].get("exp2", {}).get("f1_macro_delta", "N/A")
    s1 = ("+" if isinstance(dm1, float) and dm1>=0 else "") + (f"{dm1:.4f}" if isinstance(dm1,float) else str(dm1))
    s2 = ("+" if isinstance(dm2, float) and dm2>=0 else "") + (f"{dm2:.4f}" if isinstance(dm2,float) else str(dm2))
    print(f"  {'MACRO':12s} {'':>8} {'':>8} {output['test_f1_macro']:8.4f} {'':>8}  {s1:>7}  {s2:>7}")
    print(f"  {'ACCURACY':12s} {'':>8} {'':>8} {output['test_accuracy']:8.4f}")
    print(f"\n  [LEXIKON] Yalnizca lexikon F1-Macro : {lex_stats['lexicon_only_f1_macro']:.4f}")
    print(f"  [LEXIKON] Hibrit F1-Macro           : {lex_stats['hybrid_f1_macro']:.4f}")
    b = lex_stats['bert_contribution']
    sign = "+" if b>=0 else ""
    print(f"  [LEXIKON] BERT katkilari             : {sign}{b:.4f}")
    print("-"*70 + "\n")

    return output


# ─── Veri Fonksiyonları ──────────────────────────────────────────────────────

def load_dataframe() -> pd.DataFrame:
    master = ROOT / "data" / "processed" / "master_dataset.parquet"
    legacy = ROOT / "data" / "processed" / "combined_clean.parquet"
    if master.exists():
        df = pd.read_parquet(master)
        if "label_id" not in df.columns:
            df["label_id"] = df["label"].map(LABEL2ID)
    elif legacy.exists():
        df = pd.read_parquet(legacy)
        if "label_id" not in df.columns:
            df["label_id"] = df["label_norm"].map(ASCII_TO_ID)
        if "label" not in df.columns:
            df["label"] = df["label_id"].map(ID2LABEL)
    else:
        raise FileNotFoundError("Veri bulunamadi. python data_engine.py --rebuild")
    df = df.dropna(subset=["text","label_id"])
    df["label_id"] = df["label_id"].astype(int)
    df = df[df["label_id"].between(0, NUM_LABELS-1)]
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 5].reset_index(drop=True)
    return df


def make_splits(df: pd.DataFrame, test_size=0.15, val_size=0.15):
    splits_dir = ROOT / "data" / "splits" / "master_splits"
    tp, vp, tep = splits_dir/"train.parquet", splits_dir/"val.parquet", splits_dir/"test.parquet"
    if tp.exists() and vp.exists() and tep.exists():
        print("[SPLIT] Mevcut split'ler kullaniliyor (karsilastirabilirlik icin)")
        df_train = pd.read_parquet(tp)
        df_val   = pd.read_parquet(vp)
        df_test  = pd.read_parquet(tep)
        for d in [df_train, df_val, df_test]:
            if "label" not in d.columns:
                d["label"] = d["label_id"].map(ID2LABEL)
            if "source" not in d.columns:
                d["source"] = "unknown"
    else:
        splits_dir.mkdir(parents=True, exist_ok=True)
        df_tv, df_test = train_test_split(df, test_size=test_size, stratify=df["label_id"], random_state=SEED)
        val_r = val_size / (1.0 - test_size)
        df_train, df_val = train_test_split(df_tv, test_size=val_r, stratify=df_tv["label_id"], random_state=SEED)
        for name, subset in [("train",df_train),("val",df_val),("test",df_test)]:
            subset.reset_index(drop=True).to_parquet(splits_dir/f"{name}.parquet", index=False)

    print(f"[SPLIT] Train:{len(df_train):,}  Val:{len(df_val):,}  Test:{len(df_test):,}")
    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


# ─── Ana Eğitim Fonksiyonu ───────────────────────────────────────────────────

def train(epochs: int = 5, max_per_class: int = None):
    set_seed(SEED)
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*65)
    print("  EXPERIMENT 3 — BERTURK + LEXICON INFORMED ARCHITECTURE")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Mimari: BERTurk[CLS](768) ++ Lexikon({LEXICON_DIM}) -> Linear(778,10)")
    print("="*65)

    # ── 1. Veri ──────────────────────────────────────────────────
    df = load_dataframe()
    df_train, df_val, df_test = make_splits(df)

    if max_per_class:
        df_train = (
            df_train.groupby("label_id", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max_per_class), random_state=42))
            .reset_index(drop=True)
        )
        print(f"[FAST] max_per_class={max_per_class} -> egitim: {len(df_train):,} ornek")

    # Sınıf dağılımı özeti
    cnt = Counter(df_train["label_id"].tolist())
    print("\n[DAGILIM] Egitim seti:")
    for i, emo in enumerate(EMOTIONS):
        c = cnt.get(i, 0)
        print(f"  {emo:12s}: {c:7,d}")

    # ── 2. Lexikon & Tokenizer ────────────────────────────────────
    print(f"\n[LEXIKON] Extractor olusturuluyor...")
    extractor = LexiconExtractor()
    cov = extractor.coverage_report(df_train["text"].tolist()[:5000])
    print(f"  Sozluk: {len(extractor.lexicon)} kelime")
    print(f"  Kapsama: %{cov['coverage_rate']*100:.1f} metin | %{cov['token_hit_rate']*100:.1f} token")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── 3. Dataset (lexikon özellikleri toplu hesaplanır) ──────────
    print("\n[DATASET] Veri setleri hazirlanıyor...")
    train_ds = LexiconEmotionDataset(df_train, tokenizer, extractor, max_length=128)
    val_ds   = LexiconEmotionDataset(df_val,   tokenizer, extractor, max_length=128)
    test_ds  = LexiconEmotionDataset(df_test,  tokenizer, extractor, max_length=128)

    # ── 4. Ağırlıklar ─────────────────────────────────────────────
    train_labels   = df_train["label_id"].values
    sample_weights = compute_sample_weights(train_labels)
    class_weights  = compute_class_weights(train_labels)

    print("\n[AGIRLIK] Sinif agirlikları:")
    for i, emo in enumerate(EMOTIONS):
        print(f"  {emo:12s}: {float(class_weights[i]):.4f}")

    # ── 5. Model ──────────────────────────────────────────────────
    print(f"\n[MODEL] Hibrit model olusturuluyor...")
    model = LexiconBERTModel(
        bert_model_name=MODEL_NAME,
        num_labels=NUM_LABELS,
        lexicon_dim=LEXICON_DIM,
        dropout=0.1,
    )
    total_p    = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bert_p     = sum(p.numel() for p in model.bert.parameters())
    head_p     = sum(p.numel() for p in model.classifier.parameters())
    print(f"  BERT parametresi   : {bert_p:,}")
    print(f"  Classifier (778->10): {head_p:,}  (768+10={768+LEXICON_DIM} -> 10)")
    print(f"  Toplam egitim.     : {trainable:,}")

    # ── 6. Training Arguments ─────────────────────────────────────
    best_model_dir = RESULTS / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = len(df_train) // (64 * 2)
    warmup_steps    = int(0.1 * steps_per_epoch * epochs)

    training_args = TrainingArguments(
        output_dir=str(best_model_dir),
        fp16=True, bf16=False,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=2,
        gradient_checkpointing=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        num_train_epochs=epochs,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=100,
        logging_first_step=True,
        report_to=[],
        seed=SEED,
        remove_unused_columns=False,  # lexicon_features silinmemeli!
    )

    epoch_logger  = EpochLogger()
    data_collator = LexiconDataCollator(tokenizer=tokenizer)

    trainer = LexiconWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(),
        train_sample_weights=sample_weights,
        class_weights=class_weights,
        callbacks=[epoch_logger, EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── 7. Eğitim ─────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  EGITIM BASLIYOR")
    print(f"  Efektif batch : {64*2} | Adım/epoch: ~{steps_per_epoch}")
    print(f"  Warmup adımı  : {warmup_steps}")
    print(f"  Lexikon boyutu: {LEXICON_DIM} (NRC kategorisi)")
    print("="*65 + "\n")

    train_result = trainer.train()
    print(f"\n[TAMAMLANDI] Adim:{train_result.global_step} | "
          f"Kayıp:{train_result.training_loss:.4f} | "
          f"Sure:{train_result.metrics['train_runtime']/60:.1f}dk")

    # ── 8. Test ───────────────────────────────────────────────────
    print("\n[TEST] Test seti degerlendiriliyor...")
    test_output = trainer.predict(test_ds)
    y_pred = np.argmax(test_output.predictions, axis=-1)
    y_true = test_output.label_ids

    # Test seti lexikon özellikleri (feature importance için)
    test_lex = test_ds.lex_feats

    # ── 9. Görseller & Raporlar ───────────────────────────────────
    print("\n[VIZ] Gorseller olusturuluyor...")
    save_confusion_matrix(y_true, y_pred, RESULTS / "confusion_matrix.png")
    save_training_curves(epoch_logger.history, RESULTS / "training_progress.png")

    lex_stats = save_lexicon_importance(test_lex, y_true, y_pred, RESULTS / "lexicon_importance.png")
    report    = save_metrics_report(y_true, y_pred, epoch_logger.history, lex_stats, RESULTS / "metrics_report.json")
    save_radar_chart(report["per_class"], RESULTS / "radar_chart.png")

    # ── 9b. Kapsamlı Analitik ─────────────────────────────────────
    run_full_analytics(
        y_true=y_true,
        y_pred=y_pred,
        logits_or_probs=test_output.predictions,
        epoch_logger=epoch_logger,
        train_result=train_result,
        out_dir=RESULTS,
        exp_name="Exp3 Lexicon Informed",
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        tokenizer=tokenizer,
        extra_info={"lexicon_dim": LEXICON_DIM, "lexicon_analysis": lex_stats},
    )

    # ── 10. Model Kaydet ──────────────────────────────────────────
    torch.save(model.state_dict(), best_model_dir / "lexicon_bert_weights.pt")
    tokenizer.save_pretrained(str(best_model_dir))
    with open(best_model_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "bert_model_name": MODEL_NAME,
            "num_labels": NUM_LABELS,
            "lexicon_dim": LEXICON_DIM,
            "nrc_categories": NRC_CATS,
            "architecture": "BERTurk[CLS](768) ++ Lexikon(10) -> Linear(778,10)",
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[KAYIT] Model agırlıkları: {best_model_dir}/lexicon_bert_weights.pt")
    print(f"[KAYIT] Sonuclar: {RESULTS}")
    return trainer


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exp3 — BERTurk + Lexicon Informed")
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--epochs",        type=int, default=5)
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Sinif basi max ornek (hizli egitim, orn: 10000)")
    parser.add_argument("--lexicon-test",  action="store_true",
                        help="Sadece lexikon extractor'i test et, egitme")
    args = parser.parse_args()

    if args.lexicon_test:
        ext = LexiconExtractor()
        test_texts = [
            "Bugün çok mutluyum, harika bir gün!",
            "Korkunç bir şey oldu, çok korktum.",
            "İğrenç bir davranış, tiksindim.",
            "Çünkü biz dahiyiz, gurur duyuyorum.",
            "Kimse onların hakkı olmadığını söylemedi. Hala utanç verici.",
        ]
        print("\nLexikon Extractor Testi:")
        for text in test_texts:
            vec = ext.extract(text)
            top = sorted(enumerate(vec), key=lambda x: -x[1])[:3]
            cats = ", ".join(f"{NRC_CATS[i]}={v:.3f}" for i, v in top if v > 0)
            print(f"  '{text[:50]}...'")
            print(f"    Top NRC: [{cats}]")
        sys.exit(0)

    splits_dir = ROOT / "data" / "splits" / "master_splits"
    if args.rebuild_data or not (splits_dir / "train.parquet").exists():
        df = load_dataframe()
        make_splits(df)

    train(epochs=args.epochs, max_per_class=args.max_per_class)
