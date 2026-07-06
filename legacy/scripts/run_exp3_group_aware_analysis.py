"""
EXP3 Group-Aware Test Set — Full Inference & Analysis
======================================================
1. Classification report (precision, recall, F1 per class)
2. Confusion matrix (PNG)
3. ROC curves — one-vs-rest per class (PNG)
4. Precision-Recall curves per class (PNG)
5. McNemar test: EXP3 vs EXP1
6. Bootstrap CI (10000 iter) for F1-Macro difference EXP3 - EXP1

Outputs: results/group_aware_eval/exp3_analysis/
"""

import os
import sys
import json
import time
import warnings
import logging

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2
from scipy.special import softmax

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    accuracy_score,
)
from sklearn.preprocessing import label_binarize

# ─── Paths ───────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).resolve().parent
SPLITS  = ROOT / "data" / "splits" / "group_aware_splits"
OUT_DIR = ROOT / "results" / "group_aware_eval" / "exp3_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXP3_WEIGHTS = ROOT / "results" / "exp3" / "best_model" / "lexicon_bert_weights.pt"
EXP3_TOK     = ROOT / "results" / "exp3" / "best_model"
EXP1_MODEL   = ROOT / "results" / "exp1" / "best_model"

BERT_BASE   = "dbmdz/bert-base-turkish-cased"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED        = 42
NUM_LABELS  = 10
LEXICON_DIM = 10

EMOTIONS = [
    "mutluluk", "üzüntü", "öfke", "korku",
    "şaşkınlık", "tiksinti", "sevgi", "nötr", "utanç", "gurur",
]
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL = {i: e for i, e in enumerate(EMOTIONS)}

NRC_CATS = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "negative", "positive", "sadness", "surprise", "trust",
]

set_seed(SEED)
print(f"[DEVICE] {DEVICE}")
print(f"[OUTPUT] {OUT_DIR}")

# ─── Turkish NRC Lexicon ─────────────────────────────────────────────────────

def _build_turkish_lexicon():
    anger_words = [
        "kızgın","sinirli","öfkeli","öfke","kızdım","kızıyorum","kızıyor",
        "nefret","nefreti","hayal kırıklığı","bezdim","bıktım","lanet",
        "saldırı","saldırgan","kavga","dövüş","bağırdım","bağırıyorum",
        "küfür","hakaret","aşağıladı","incitti","kırıcı","rezalet",
        "haksız","zorba","zorbalık","taciz","rahatsız","iğneledi",
        "kışkırttı","saldırdı","tehdit","şiddet","şiddetli","agresif",
        "sert","acımasız","vahşet","gaddar","zalim","zalimlik","cani",
        "hunhar","intikam","öç","kin","düşmanca","galeyan","hiddet",
        "gazap","hınç","isyan","kıyamet","taşkınlık","taşkın",
    ]
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
    ]
    negative_words = [
        "kötü","berbat","yanlış","hata","sorun","problem","başarısız",
        "olmadı","yapamadım","kaybettim","mahvoldu","olmaz","imkansız",
        "çöktü","battı","işe yaramaz","faydasız","boşuna","nafile",
        "beyhude","bozuldu","hasar","zarar","kayıp","ziyan","acıklı",
        "üzücü","talihsiz","şanssız","bedbaht","mutsuz","mutsuzluk",
        "sıkıntı","dert","bela","trajik","facıa","rezalet","utanç",
        "ayıp","hatalı","eksik","yetersiz","zayıf","mağlup","yenilgi",
        "trajedi","felâket","mahkum","mahkumiyet","ceza","acı","ıstırap",
    ]
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
    ]
    sadness_words = [
        "üzgün","ağladım","hüzün","mutsuz","acı","yas","yalnız",
        "kayıp","kaybettim","ayrılık","özlem","hasret","dert","çile",
        "elem","ağıt","ağlamak","üzüntü","mahzun","keder","gözyaşı",
        "sızı","ah","vah","yazık","iç sıkıntısı","can sıkıntısı",
        "bunalım","depresyon","yıkıldım","çöküntü","bitkinlik",
        "tükenmişlik","yorgunluk","umutsuzluk","kimsesiz","terkedilmiş",
        "yalnızlık","ıssızlık","boşluk","anlamsızlık","karanlık",
        "sessizlik","derin üzüntü","hüzünlü","hüzünlendim",
        "keder","kederli","gamlı","gam","matemli","meyus",
        "umutsuz","ümitsiz","çaresizce","yıkıcı","parçalandım",
    ]
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
    _tr2ascii = str.maketrans("çğışöüÇĞİŞÖÜ", "cgisouCGISOI")

    def _variants(w):
        return {w, w.translate(_tr2ascii)}

    lexicon = {}
    for cat_idx, words in enumerate(word_lists):
        for word in words:
            word = word.strip().lower()
            if not word or " " in word:
                continue
            for variant in _variants(word):
                if variant not in lexicon:
                    lexicon[variant] = np.zeros(LEXICON_DIM, dtype=np.float32)
                lexicon[variant][cat_idx] = 1.0
    return lexicon


TURKISH_LEXICON = _build_turkish_lexicon()
print(f"[LEXICON] {len(TURKISH_LEXICON)} words loaded")


# ─── Lexicon Extractor ───────────────────────────────────────────────────────

import re as _re

class LexiconExtractor:
    def __init__(self):
        self.lexicon = TURKISH_LEXICON
        self._punct_re = _re.compile(r"[^\w\s]", _re.UNICODE)
        self._stem_index = {}
        for key, vec in self.lexicon.items():
            for stem_len in (4, 5, 6, 7):
                if len(key) >= stem_len:
                    stem = key[:stem_len]
                    if stem not in self._stem_index:
                        self._stem_index[stem] = vec.copy()
                    else:
                        self._stem_index[stem] = self._stem_index[stem] + vec

    _TR_UPPER = str.maketrans("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ",
                               "abcçdefgğhıijklmnoöprsştuüvyz")
    _TR_ASCII = str.maketrans("çğışöüîâû", "cgisouiau")

    def _tr_lower(self, text):
        return text.translate(self._TR_UPPER)

    def _tokenize(self, text):
        text = self._tr_lower(text)
        text = self._punct_re.sub(" ", text)
        return [t for t in text.split() if len(t) >= 2]

    def extract(self, text):
        tokens = self._tokenize(text)
        vec = np.zeros(LEXICON_DIM, dtype=np.float32)
        for token in tokens:
            hit = None
            if token in self.lexicon:
                hit = self.lexicon[token]
            elif (ascii_tok := token.translate(self._TR_ASCII)) in self.lexicon:
                hit = self.lexicon[ascii_tok]
            elif len(token) >= 4:
                for sl in (7, 6, 5, 4):
                    stem = token[:sl] if len(token) >= sl else token
                    if stem in self._stem_index:
                        hit = self._stem_index[stem]
                        break
            if hit is not None:
                vec += hit
        if len(tokens) > 0:
            vec /= len(tokens)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm
        return vec

    def batch_extract(self, texts):
        out = np.zeros((len(texts), LEXICON_DIM), dtype=np.float32)
        step = max(1, len(texts) // 20)
        for i, t in enumerate(texts):
            out[i] = self.extract(t)
            if i % step == 0:
                print(f"  Lexicon features: {i}/{len(texts)} ({i/len(texts)*100:.0f}%)", end="\r")
        print(f"  Lexicon features: {len(texts)}/{len(texts)} (100%)          ")
        return out


# ─── Model Architecture ──────────────────────────────────────────────────────

from transformers import BertModel

class LexiconBERTModel(nn.Module):
    def __init__(self, bert_model_name=BERT_BASE, num_labels=NUM_LABELS,
                 lexicon_dim=LEXICON_DIM, dropout=0.1):
        super().__init__()
        self.num_labels  = num_labels
        self.lexicon_dim = lexicon_dim
        self.bert    = BertModel.from_pretrained(bert_model_name)
        combined_dim = self.bert.config.hidden_size + lexicon_dim
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(combined_dim, num_labels)
        self.config = self.bert.config
        self.config.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                lexicon_features=None, labels=None, **kwargs):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        cls_vec = self.dropout(bert_out.last_hidden_state[:, 0, :])
        if lexicon_features is not None:
            lex = lexicon_features.to(cls_vec.device).float()
            combined = torch.cat([cls_vec, lex], dim=-1)
        else:
            zero_lex = torch.zeros(cls_vec.size(0), self.lexicon_dim, device=cls_vec.device)
            combined = torch.cat([cls_vec, zero_lex], dim=-1)
        logits = self.classifier(combined)
        return SequenceClassifierOutput(loss=None, logits=logits)


# ─── Datasets ────────────────────────────────────────────────────────────────

class LexiconDataset(Dataset):
    def __init__(self, df, tokenizer, lex_feats, max_length=128):
        self.texts      = df["text"].tolist()
        self.labels     = df["label_id"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.lex_feats  = lex_feats

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=self.max_length,
                             truncation=True, padding=False)
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"]           = torch.tensor(self.labels[idx], dtype=torch.long)
        item["lexicon_features"] = torch.tensor(self.lex_feats[idx], dtype=torch.float32)
        return item


class LexiconCollator(DataCollatorWithPadding):
    def __call__(self, features):
        lex = torch.stack([f.pop("lexicon_features") for f in features])
        batch = super().__call__(features)
        batch["lexicon_features"] = lex
        return batch


class SimpleDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts  = df["text"].tolist()
        self.labels = df["label_id"].tolist()
        self.tok    = tokenizer
        self.max_len = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], max_length=self.max_len,
                       truncation=True, padding=False)
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ─── Inference Helpers ───────────────────────────────────────────────────────

def run_exp3_inference(test_df):
    print("\n" + "="*60)
    print("  EXP3 — BERTurk + NRC Lexicon Inference")
    print(f"  Weights: {EXP3_WEIGHTS}")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(str(EXP3_TOK))

    print("\n[LEXICON] Extracting features for test set...")
    extractor = LexiconExtractor()
    lex_feats = extractor.batch_extract(test_df["text"].tolist())

    model = LexiconBERTModel(BERT_BASE, NUM_LABELS, LEXICON_DIM)
    state_dict = torch.load(str(EXP3_WEIGHTS), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval().to(DEVICE)
    print(f"[MODEL] EXP3 weights loaded -> {DEVICE}")

    test_ds  = LexiconDataset(test_df, tokenizer, lex_feats, max_length=128)
    collator = LexiconCollator(tokenizer=tokenizer)
    loader   = DataLoader(test_ds, batch_size=64, collate_fn=collator,
                          num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    all_logits, all_labels = [], []
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            labels = batch.pop("labels")
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            out    = model(**inputs)
            all_logits.append(out.logits.cpu().float().numpy())
            all_labels.append(labels.numpy())
            if i % 10 == 0:
                print(f"  Batch {i+1}/{len(loader)}", end="\r")

    print(f"  Inference done in {time.time()-t0:.1f}s          ")

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.argmax(logits, axis=-1)
    probs  = softmax(logits, axis=-1)

    del model
    torch.cuda.empty_cache()
    return y_true, y_pred, probs


def run_exp1_inference(test_df):
    print("\n" + "="*60)
    print("  EXP1 — BERTurk Baseline Inference")
    print(f"  Path: {EXP1_MODEL}")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(str(EXP1_MODEL))
    model = AutoModelForSequenceClassification.from_pretrained(str(EXP1_MODEL))
    model.eval().to(DEVICE)
    print(f"[MODEL] EXP1 loaded -> {DEVICE}")

    test_ds  = SimpleDataset(test_df, tokenizer, max_length=128)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader   = DataLoader(test_ds, batch_size=64, collate_fn=collator,
                          num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    all_logits, all_labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            labels = batch.pop("labels")
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            out    = model(**inputs)
            all_logits.append(out.logits.cpu().float().numpy())
            all_labels.append(labels.numpy())
            if i % 10 == 0:
                print(f"  Batch {i+1}/{len(loader)}", end="\r")

    print(f"  Inference done in {time.time()-t0:.1f}s          ")

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.argmax(logits, axis=-1)
    probs  = softmax(logits, axis=-1)

    del model
    torch.cuda.empty_cache()
    return y_true, y_pred, probs


# ─── Analysis Functions ───────────────────────────────────────────────────────

def print_classification_report(y_true, y_pred, model_name):
    print(f"\n{'='*70}")
    print(f"  CLASSIFICATION REPORT — {model_name}")
    print(f"{'='*70}")
    report = classification_report(
        y_true, y_pred,
        target_names=EMOTIONS,
        digits=4,
        zero_division=0,
    )
    print(report)
    return classification_report(
        y_true, y_pred,
        target_names=EMOTIONS,
        output_dict=True,
        zero_division=0,
    )


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    labels  = [e[:9] for e in EMOTIONS]

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    for ax, data, title, fmt in [
        (axes[0], cm,      "Raw Counts",         "d"),
        (axes[1], cm_norm, "Row-Normalized",     ".2f"),
    ]:
        sns.heatmap(
            data, annot=True, fmt=fmt,
            xticklabels=labels, yticklabels=labels,
            cmap="Blues", linewidths=0.4, ax=ax, annot_kws={"size": 8},
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)

    fig.suptitle(f"{model_name} — Group-Aware Test Set | Confusion Matrix",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Confusion matrix: {save_path}")


def plot_roc_curves(y_true, probs, model_name, save_path):
    y_bin = label_binarize(y_true, classes=list(range(NUM_LABELS)))
    cols  = 5
    rows  = 2
    fig, axes = plt.subplots(rows, cols, figsize=(20, 9))
    axes = axes.flatten()

    all_auc = {}
    for i, (emotion, ax) in enumerate(zip(EMOTIONS, axes)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc     = auc(fpr, tpr)
        all_auc[emotion] = round(float(roc_auc), 4)

        ax.plot(fpr, tpr, lw=2, color="#8e44ad",
                label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("FPR", fontsize=9)
        ax.set_ylabel("TPR", fontsize=9)
        ax.set_title(emotion, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    macro_auc = np.mean(list(all_auc.values()))
    fig.suptitle(f"{model_name} — ROC Curves (One-vs-Rest) | Macro AUC = {macro_auc:.4f}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] ROC curves: {save_path}")
    print(f"  Per-class AUC: {all_auc}")
    print(f"  Macro AUC: {macro_auc:.4f}")
    return all_auc, macro_auc


def plot_pr_curves(y_true, probs, model_name, save_path):
    y_bin = label_binarize(y_true, classes=list(range(NUM_LABELS)))
    cols  = 5
    rows  = 2
    fig, axes = plt.subplots(rows, cols, figsize=(20, 9))
    axes = axes.flatten()

    all_ap = {}
    for i, (emotion, ax) in enumerate(zip(EMOTIONS, axes)):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ap = average_precision_score(y_bin[:, i], probs[:, i])
        all_ap[emotion] = round(float(ap), 4)

        support_frac = float(y_bin[:, i].mean())
        ax.plot(recall, precision, lw=2, color="#27ae60",
                label=f"AP = {ap:.3f}")
        ax.axhline(support_frac, color="navy", lw=1, linestyle="--",
                   alpha=0.6, label=f"Baseline = {support_frac:.3f}")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall", fontsize=9)
        ax.set_ylabel("Precision", fontsize=9)
        ax.set_title(emotion, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    mean_ap = np.mean(list(all_ap.values()))
    fig.suptitle(f"{model_name} — Precision-Recall Curves | mAP = {mean_ap:.4f}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] PR curves: {save_path}")
    print(f"  Per-class AP: {all_ap}")
    print(f"  Mean AP: {mean_ap:.4f}")
    return all_ap, mean_ap


def mcnemar_test(y_true, y_pred_a, y_pred_b, name_a="EXP3", name_b="EXP1"):
    """
    McNemar test: compares two classifiers on the same test set.
    Contingency table:
      n00 = both wrong
      n01 = A wrong, B right
      n10 = A right, B wrong
      n11 = both right
    Statistic: (|n01 - n10| - 1)^2 / (n01 + n10)  [with continuity correction]
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    n00 = int(np.sum(~correct_a & ~correct_b))  # both wrong
    n01 = int(np.sum(~correct_a &  correct_b))  # A wrong, B right
    n10 = int(np.sum( correct_a & ~correct_b))  # A right, B wrong
    n11 = int(np.sum( correct_a &  correct_b))  # both right

    print(f"\n{'='*60}")
    print(f"  McNEMAR TEST: {name_a} vs {name_b}")
    print(f"{'='*60}")
    print(f"  Contingency table (rows=A={name_a}, cols=B={name_b}):")
    print(f"              B wrong   B right")
    print(f"  A wrong  :  {n00:7d}  {n01:7d}")
    print(f"  A right  :  {n10:7d}  {n11:7d}")
    print(f"\n  n01 (A wrong, B right) = {n01}")
    print(f"  n10 (A right, B wrong) = {n10}")

    discordant = n01 + n10
    if discordant == 0:
        print("  [NOTE] No discordant pairs — models are identical on this test set.")
        return {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
                "chi2": 0.0, "p_value": 1.0, "significant_0.05": False}

    # With continuity correction (Edwards)
    chi2_stat = (abs(n01 - n10) - 1.0) ** 2 / discordant
    p_value   = float(1 - chi2.cdf(chi2_stat, df=1))

    print(f"\n  chi2 (with continuity correction) = {chi2_stat:.4f}")
    print(f"  p-value                           = {p_value:.6f}")
    print(f"  Significant (alpha=0.05)          = {p_value < 0.05}")

    if p_value < 0.001:
        sig_str = "p < 0.001 ***"
    elif p_value < 0.01:
        sig_str = f"p = {p_value:.4f} **"
    elif p_value < 0.05:
        sig_str = f"p = {p_value:.4f} *"
    else:
        sig_str = f"p = {p_value:.4f} (not significant)"
    print(f"  {sig_str}")

    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "discordant_pairs": discordant,
        "chi2_continuity_corrected": round(chi2_stat, 4),
        "p_value": round(p_value, 6),
        "significant_0.05": bool(p_value < 0.05),
        "significance_label": sig_str,
    }


def bootstrap_f1_macro_difference(
    y_true, y_pred_a, y_pred_b,
    name_a="EXP3", name_b="EXP1",
    n_iter=10000, ci_level=0.95, seed=42
):
    rng    = np.random.default_rng(seed)
    n      = len(y_true)
    diffs  = np.empty(n_iter)

    print(f"\n{'='*60}")
    print(f"  BOOTSTRAP CI — F1-Macro Difference: {name_a} - {name_b}")
    print(f"  Iterations: {n_iter:,}  |  CI level: {ci_level:.0%}")
    print(f"{'='*60}")

    observed_a  = f1_score(y_true, y_pred_a, average="macro", zero_division=0)
    observed_b  = f1_score(y_true, y_pred_b, average="macro", zero_division=0)
    observed_diff = observed_a - observed_b
    print(f"  Observed F1-Macro {name_a} : {observed_a:.4f}")
    print(f"  Observed F1-Macro {name_b} : {observed_b:.4f}")
    print(f"  Observed difference       : {observed_diff:+.4f}")
    print(f"\n  Running {n_iter:,} bootstrap iterations...")

    t0 = time.time()
    for i in range(n_iter):
        idx   = rng.integers(0, n, size=n)
        f1_a  = f1_score(y_true[idx], y_pred_a[idx], average="macro", zero_division=0)
        f1_b  = f1_score(y_true[idx], y_pred_b[idx], average="macro", zero_division=0)
        diffs[i] = f1_a - f1_b

    alpha  = 1 - ci_level
    lo     = float(np.percentile(diffs, 100 * alpha / 2))
    hi     = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    se     = float(np.std(diffs, ddof=1))
    elapsed = time.time() - t0

    print(f"  Done in {elapsed:.1f}s")
    print(f"\n  95% CI (percentile): [{lo:+.4f}, {hi:+.4f}]")
    print(f"  Bootstrap SE        : {se:.4f}")
    print(f"  Contains zero       : {lo <= 0 <= hi}")

    return {
        "model_a": name_a,
        "model_b": name_b,
        "observed_f1_macro_a": round(float(observed_a),    4),
        "observed_f1_macro_b": round(float(observed_b),    4),
        "observed_difference":  round(float(observed_diff), 4),
        "bootstrap_iterations": n_iter,
        "ci_level": ci_level,
        "ci_lower": round(lo, 4),
        "ci_upper": round(hi, 4),
        "bootstrap_se": round(se, 4),
        "ci_contains_zero": bool(lo <= 0 <= hi),
        "bootstrap_diffs": diffs,
    }


def plot_bootstrap_distribution(bootstrap_result, save_path):
    diffs = bootstrap_result["bootstrap_diffs"]
    lo    = bootstrap_result["ci_lower"]
    hi    = bootstrap_result["ci_upper"]
    obs   = bootstrap_result["observed_difference"]
    name_a = bootstrap_result["model_a"]
    name_b = bootstrap_result["model_b"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(diffs, bins=80, color="#3498db", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(obs, color="#e74c3c", lw=2.5, label=f"Observed diff = {obs:+.4f}")
    ax.axvline(lo,  color="#f39c12", lw=2, linestyle="--", label=f"95% CI lower = {lo:+.4f}")
    ax.axvline(hi,  color="#f39c12", lw=2, linestyle="--", label=f"95% CI upper = {hi:+.4f}")
    ax.axvline(0,   color="black",   lw=1.5, linestyle=":", alpha=0.8, label="Zero (null)")

    ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1],
                     lo, hi, alpha=0.12, color="#f39c12")

    ax.set_xlabel(f"F1-Macro Difference ({name_a} - {name_b})", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        f"Bootstrap Distribution of F1-Macro Difference\n"
        f"{name_a} vs {name_b} | n=10,000 iterations | 95% CI [{lo:+.4f}, {hi:+.4f}]",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Bootstrap distribution: {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("  EXP3 GROUP-AWARE ANALYSIS — Full Inference + Statistical Tests")
    print("="*70)

    # ── 1. Load test data ────────────────────────────────────────────────────
    print(f"\n[DATA] Loading group-aware test split: {SPLITS / 'test.parquet'}")
    test_df = pd.read_parquet(SPLITS / "test.parquet")

    if "label_id" not in test_df.columns:
        test_df["label_id"] = test_df["label"].map(LABEL2ID)
    test_df["label_id"] = test_df["label_id"].astype(int)
    test_df = test_df.dropna(subset=["text", "label_id"]).reset_index(drop=True)
    test_df["text"] = test_df["text"].astype(str).str.strip()

    print(f"  Test samples: {len(test_df):,}")
    print("\n  Label distribution:")
    for label, cnt in test_df["label"].value_counts().items():
        print(f"    {label}: {cnt:,}")

    # ── 2. EXP3 Inference ────────────────────────────────────────────────────
    y_true_exp3, y_pred_exp3, probs_exp3 = run_exp3_inference(test_df)

    # ── 3. EXP1 Inference ────────────────────────────────────────────────────
    y_true_exp1, y_pred_exp1, probs_exp1 = run_exp1_inference(test_df)

    assert np.array_equal(y_true_exp3, y_true_exp1), "y_true mismatch between EXP3 and EXP1!"
    y_true = y_true_exp3

    # ── 4. Save predictions ──────────────────────────────────────────────────
    np.save(OUT_DIR / "exp3_y_true.npy",  y_true)
    np.save(OUT_DIR / "exp3_y_pred.npy",  y_pred_exp3)
    np.save(OUT_DIR / "exp3_probs.npy",   probs_exp3)
    np.save(OUT_DIR / "exp1_y_pred.npy",  y_pred_exp1)
    np.save(OUT_DIR / "exp1_probs.npy",   probs_exp1)
    print(f"\n[SAVED] Predictions -> {OUT_DIR}")

    # ── 5. Classification Report ─────────────────────────────────────────────
    report_exp3 = print_classification_report(y_true, y_pred_exp3, "EXP3 (BERTurk + NRC Lexicon)")
    report_exp1 = print_classification_report(y_true, y_pred_exp1, "EXP1 (BERTurk Baseline)")

    # Save classification reports as JSON
    with open(OUT_DIR / "exp3_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report_exp3, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "exp1_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report_exp1, f, ensure_ascii=False, indent=2)

    # ── 6. Confusion Matrix ──────────────────────────────────────────────────
    print("\n[VIZ] Generating confusion matrix...")
    plot_confusion_matrix(y_true, y_pred_exp3, "EXP3 (BERTurk + NRC Lexicon)",
                          OUT_DIR / "exp3_confusion_matrix.png")

    # ── 7. ROC Curves ────────────────────────────────────────────────────────
    print("\n[VIZ] Generating ROC curves...")
    roc_auc_dict, macro_auc = plot_roc_curves(
        y_true, probs_exp3, "EXP3 (BERTurk + NRC Lexicon)",
        OUT_DIR / "exp3_roc_curves.png"
    )

    # ── 8. Precision-Recall Curves ───────────────────────────────────────────
    print("\n[VIZ] Generating Precision-Recall curves...")
    ap_dict, mean_ap = plot_pr_curves(
        y_true, probs_exp3, "EXP3 (BERTurk + NRC Lexicon)",
        OUT_DIR / "exp3_pr_curves.png"
    )

    # ── 9. McNemar Test ──────────────────────────────────────────────────────
    mcnemar_result = mcnemar_test(y_true, y_pred_exp3, y_pred_exp1, "EXP3", "EXP1")

    # ── 10. Bootstrap CI ─────────────────────────────────────────────────────
    bootstrap_result = bootstrap_f1_macro_difference(
        y_true, y_pred_exp3, y_pred_exp1, "EXP3", "EXP1",
        n_iter=10000, ci_level=0.95, seed=SEED
    )
    plot_bootstrap_distribution(
        bootstrap_result,
        OUT_DIR / "exp3_vs_exp1_bootstrap_ci.png"
    )

    # ── 11. Summary Report ────────────────────────────────────────────────────
    summary = {
        "experiment":    "EXP3_group_aware_analysis",
        "model":         "BERTurk + NRC Turkish Lexicon",
        "test_split":    "data/splits/group_aware_splits/test.parquet",
        "n_test":        int(len(test_df)),
        "exp3_metrics": {
            "accuracy":     round(float(accuracy_score(y_true, y_pred_exp3)), 4),
            "f1_macro":     round(float(f1_score(y_true, y_pred_exp3, average="macro",    zero_division=0)), 4),
            "f1_weighted":  round(float(f1_score(y_true, y_pred_exp3, average="weighted", zero_division=0)), 4),
            "macro_auc":    round(float(macro_auc), 4),
            "mean_ap":      round(float(mean_ap), 4),
        },
        "exp1_metrics": {
            "accuracy":    round(float(accuracy_score(y_true, y_pred_exp1)), 4),
            "f1_macro":    round(float(f1_score(y_true, y_pred_exp1, average="macro",    zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(y_true, y_pred_exp1, average="weighted", zero_division=0)), 4),
        },
        "per_class_roc_auc": roc_auc_dict,
        "per_class_ap":      ap_dict,
        "mcnemar_test":      {k: v for k, v in mcnemar_result.items()},
        "bootstrap_ci": {
            k: v for k, v in bootstrap_result.items()
            if k != "bootstrap_diffs"
        },
    }

    out_json = OUT_DIR / "analysis_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\n  EXP3 (BERTurk + NRC Lexicon):")
    print(f"    Accuracy     : {summary['exp3_metrics']['accuracy']:.4f}")
    print(f"    F1-Macro     : {summary['exp3_metrics']['f1_macro']:.4f}")
    print(f"    F1-Weighted  : {summary['exp3_metrics']['f1_weighted']:.4f}")
    print(f"    Macro AUC    : {summary['exp3_metrics']['macro_auc']:.4f}")
    print(f"    Mean AP      : {summary['exp3_metrics']['mean_ap']:.4f}")
    print(f"\n  EXP1 (BERTurk Baseline):")
    print(f"    Accuracy     : {summary['exp1_metrics']['accuracy']:.4f}")
    print(f"    F1-Macro     : {summary['exp1_metrics']['f1_macro']:.4f}")
    print(f"    F1-Weighted  : {summary['exp1_metrics']['f1_weighted']:.4f}")
    print(f"\n  McNemar Test (EXP3 vs EXP1):")
    print(f"    chi2 (cc)    : {mcnemar_result['chi2_continuity_corrected']:.4f}")
    print(f"    p-value      : {mcnemar_result['p_value']:.6f}")
    print(f"    Significant  : {mcnemar_result['significant_0.05']}")
    print(f"\n  Bootstrap CI (F1-Macro EXP3 - EXP1, n=10000):")
    print(f"    Observed diff: {bootstrap_result['observed_difference']:+.4f}")
    print(f"    95% CI       : [{bootstrap_result['ci_lower']:+.4f}, {bootstrap_result['ci_upper']:+.4f}]")
    print(f"    Contains zero: {bootstrap_result['ci_contains_zero']}")
    print(f"\n  All outputs saved to: {OUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
