"""
SHAP Yorumlanabilirlik Modülü — BERTurk + Sözlük (NRC kategorili) Hibrit Model
===================================================================
Hangi kelimeler duygu kararını nasıl etkiliyor?

Üç analiz yöntemi:
  1. SHAP Partition Explainer  → token düzeyinde katkı (gerçek SHAP)
  2. Lexikon-Lineer Analizi    → sözlük × sınıflandırıcı ağırlıkları
                                  (eğitime gerek yok, anında sonuç)
  3. Gradyan × Giriş           → BERT için gradient-based saliency

Çıktılar (results/interpretability/):
  force_plot_<duygu>.html       — etkileşimli SHAP katkı grafiği
  force_plot_<duygu>.png        — baskı kalitesi katkı çubuğu
  summary_plot_<duygu>.png      — çok örnek SHAP özet grafiği
  lexicon_weights_<duygu>.png   — sözlük × ağırlık ısı haritası
  top_keywords_all.json         — her duygu için en etkili 5 kelime
  gradient_saliency_<duygu>.png — BERT gradient saliency haritası

Kullanım:
  python shap_interpretability.py --text "Bu film harika bir deneyimdi!"
  python shap_interpretability.py --emotion gurur --top-words
  python shap_interpretability.py --demo-all
  python shap_interpretability.py --full-report
"""

import sys
import re
import json
import time
import warnings
import argparse
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec

import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent
OUTDIR  = ROOT / "results" / "interpretability"
OUTDIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

# ─── Bileşenleri import et ──────────────────────────────────────────────────

try:
    from train_exp3_lexicon import (
        LexiconBERTModel,
        LexiconExtractor,
        TURKISH_LEXICON,
        NRC_CATS,
        MODEL_NAME,
        NUM_LABELS,
        LEXICON_DIM,
        EMOTIONS,
        load_lexicon_bert_weights,
    )
    _IMPORTS_OK = True
except Exception as e:
    print(f"[UYARI] train_exp3_lexicon import hatasi: {e}")
    _IMPORTS_OK = False

try:
    from transformers import AutoTokenizer
    _TRANSFORMERS_OK = True
except ImportError:
    _TRANSFORMERS_OK = False
    print("[UYARI] transformers yuklu degil — pip install transformers")

# SHAP isteğe bağlı
try:
    import shap
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False
    print("[BILGI] shap yuklu degil — Lexikon+Gradient analizi aktif, SHAP devre disi.")
    print("        SHAP icin: pip install shap")


# ─── Sabitler ───────────────────────────────────────────────────────────────

EMOTIONS = [
    "mutluluk", "üzüntü", "öfke", "korku",
    "şaşkınlık", "tiksinti", "sevgi", "nötr", "utanç", "gurur",
]
NRC_CATS = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "negative", "positive", "sadness", "surprise", "trust",
]
EMOTION_COLORS = {
    "mutluluk":  "#f1c40f",
    "üzüntü":    "#3498db",
    "öfke":      "#e74c3c",
    "korku":     "#9b59b6",
    "şaşkınlık": "#1abc9c",
    "tiksinti":  "#27ae60",
    "sevgi":     "#e91e63",
    "nötr":      "#95a5a6",
    "utanç":     "#e67e22",
    "gurur":     "#2c3e50",
}
NRC_TR = {
    "anger": "Öfke", "anticipation": "Beklenti", "disgust": "Tiksinti",
    "fear": "Korku", "joy": "Neşe", "negative": "Negatif",
    "positive": "Pozitif", "sadness": "Üzüntü", "surprise": "Şaşkınlık",
    "trust": "Güven",
}

# Hoca sunumu için her duyguya örnek cümleler
DEMO_TEXTS: Dict[str, List[str]] = {
    "mutluluk":  [
        "Bu gün hayatımın en güzel günü, çok mutluyum!",
        "Sınavı geçtim, sevinçten harika hissediyorum.",
        "Ailemle geçirdiğimiz tatil muhteşemdi.",
    ],
    "üzüntü":    [
        "Köpeğimi kaybettim, çok üzgünüm.",
        "Bu ayrılık gerçekten çok ağır geldi bana.",
        "Yalnız hissediyorum, hiç kimse yok yanımda.",
    ],
    "öfke":      [
        "Bu kadar haksızlığa nasıl göz yumarsın, lanet olsun!",
        "İnanılmaz sinirli ve öfkeliyim, artık kaldıramıyorum!",
        "Zorba davranışına bir son verilmeli, rezalet!",
    ],
    "korku":     [
        "Karanlık gecede yalnız kalmaktan çok korkuyorum.",
        "O ses beni ürküttü, panikledim.",
        "Tehlike sinyalleri var, endişeliyim.",
    ],
    "şaşkınlık": [
        "İnanılmaz, bunu hiç beklemiyordum! Şaşırdım.",
        "Vay be, nasıl olur, mucize gibi bir şey!",
        "Meğer her şey bambaşkaymış, hayret.",
    ],
    "tiksinti":  [
        "Bu pislik durumdan iğrendim, çok iğrenç!",
        "Rezil bir davranış, insanlık dışı ve tiksintici.",
        "O koku korkunçtu, midem bulandı.",
    ],
    "sevgi":     [
        "Seni çok seviyorum, hayatımdaki en özel kişisin.",
        "Annem her şeyim, ona olan sevgim sonsuz.",
        "Bu güzel ilişki kalbimi dolduruyor.",
    ],
    "nötr":      [
        "Hava bugün 18 derece.",
        "Toplantı saat 14:00'te başlayacak.",
        "Rapor salı günü teslim edilecek.",
    ],
    "utanç":     [
        "O an yaptığım hatadan dolayı hâlâ utanıyorum.",
        "Herkesin önünde mahcup oldum, yere geçmek istedim.",
        "Bu hareketi utanç verici buluyorum.",
    ],
    "gurur":     [
        "Oğlum diplomasını aldı, onunla gurur duyuyorum!",
        "Milli takımımız şampiyon oldu, gururla göğsüm kabarıyor.",
        "Yıllarca çalıştım ve başardım, kendimle gurur duyuyorum.",
    ],
}


# ─── Model Yükleyici ─────────────────────────────────────────────────────────

class ModelLoader:
    """Eğitilmiş ağırlıkları yükler; bulunamazsa rastgele başlatılmış model döner."""

    def __init__(self):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model     = None
        self.tokenizer = None
        self.extractor = None
        self._loaded   = False

    def load(self) -> bool:
        if not (_IMPORTS_OK and _TRANSFORMERS_OK):
            print("[HATA] Gerekli paketler eksik.")
            return False

        print(f"[YÜKLEME] Cihaz: {self.device}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.extractor = LexiconExtractor()

        # Model mimarisini oluştur
        self.model = LexiconBERTModel(
            bert_model_name=MODEL_NAME,
            num_labels=NUM_LABELS,
            lexicon_dim=LEXICON_DIM,
        )

        # Ağırlık arama sırası: exp4 > exp3 > rastgele
        weight_candidates = [
            ROOT / "results" / "exp4" / "best_model" / "master_hybrid_weights.pt",
            ROOT / "results" / "exp4" / "best_model" / "pytorch_model.bin",
            ROOT / "results" / "exp4" / "best_model" / "model.safetensors",
            ROOT / "results" / "exp3" / "best_model" / "lexicon_bert_weights.pt",
            ROOT / "results" / "exp3" / "best_model" / "pytorch_model.bin",
            ROOT / "results" / "exp3" / "best_model" / "model.safetensors",
        ]

        loaded_from = None
        for candidate in weight_candidates:
            if candidate.exists():
                try:
                    if candidate.suffix in (".bin", ".pt"):
                        state = torch.load(candidate, map_location="cpu")
                    else:
                        from safetensors.torch import load_file
                        state = load_file(str(candidate))
                    load_lexicon_bert_weights(self.model, state)
                    loaded_from = candidate
                    break
                except Exception as e:
                    print(f"  [UYARI] {candidate.name} yüklenemedi: {e}")

        if loaded_from:
            print(f"[YÜKLEME] Ağırlıklar: {loaded_from.relative_to(ROOT)}")
        else:
            print("[UYARI] Eğitilmiş ağırlık bulunamadı — rastgele başlatıldı (demo modu)")

        self.model.eval()
        self.model.to(self.device)
        self._loaded = True
        return True

    @property
    def ready(self) -> bool:
        return self._loaded


# ─── Predict Wrapper ─────────────────────────────────────────────────────────

class PredictWrapper:
    """
    SHAP uyumlu predict fonksiyonu.
    texts: List[str] → np.ndarray (N, 10) prob

    Maskeli metinler için lexikon özellikleri yeniden hesaplanır —
    bu, doğru SHAP değerleri için kritiktir.
    """

    def __init__(self, loader: ModelLoader, batch_size: int = 16):
        self.loader     = loader
        self.batch_size = batch_size

    def __call__(self, texts) -> np.ndarray:
        """texts: list of str → (N, 10) float32 numpy array"""
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        texts = [str(t) for t in texts]

        all_probs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            probs = self._forward_batch(batch)
            all_probs.append(probs)

        return np.vstack(all_probs)

    def _forward_batch(self, texts: List[str]) -> np.ndarray:
        device    = self.loader.device
        tokenizer = self.loader.tokenizer
        extractor = self.loader.extractor
        model     = self.loader.model

        # BERT tokenization
        enc = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        # Lexikon özellikleri — maskeli metin için yeniden hesapla
        lex_list = [extractor.extract(t) for t in texts]
        lex_tensor = torch.tensor(
            np.stack(lex_list), dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                token_type_ids=enc.get("token_type_ids"),
                lexicon_features=lex_tensor,
            )
        return F.softmax(out.logits, dim=-1).cpu().numpy()


# ─── Gradient Saliency ───────────────────────────────────────────────────────

def compute_gradient_saliency(
    text: str,
    target_idx: int,
    loader: ModelLoader,
) -> Tuple[List[str], np.ndarray]:
    """
    ∂output[target] / ∂embedding * embedding → word importance scores.

    Subword tokenler kelime düzeyinde toplanır (ortalama).
    Döner: (word_list, saliency_scores)
    """
    device    = loader.device
    tokenizer = loader.tokenizer
    extractor = loader.extractor
    model     = loader.model

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        return_offsets_mapping=True,
    )
    offset_mapping = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    lex = torch.tensor(extractor.extract(text), dtype=torch.float32).unsqueeze(0).to(device)

    # embedding katmanına hook
    embeddings = model.bert.embeddings.word_embeddings(enc["input_ids"])  # (1, T, 768)
    embeddings = embeddings.detach().requires_grad_(True)

    # forward pass
    bert_out  = model.bert(
        inputs_embeds=embeddings,
        attention_mask=enc["attention_mask"],
        token_type_ids=enc.get("token_type_ids"),
    )
    cls_vec  = model.dropout(bert_out.last_hidden_state[:, 0, :])
    combined = torch.cat([cls_vec, lex], dim=-1)
    logits   = model.classifier(combined)

    # hedef sınıfa göre gradient
    score = logits[0, target_idx]
    score.backward()

    grads = embeddings.grad[0].cpu().numpy()         # (T, 768)
    saliency = np.abs(grads).sum(axis=-1)             # (T,)

    # Subword → kelime dönüşümü
    tokens    = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].cpu().tolist())
    words, word_scores = _aggregate_subwords(tokens, saliency, offset_mapping, text)

    # L2 normalize
    norm = np.linalg.norm(word_scores)
    if norm > 1e-8:
        word_scores = word_scores / norm

    return words, word_scores


def _aggregate_subwords(
    tokens: List[str],
    scores: np.ndarray,
    offsets: List[Tuple[int, int]],
    original_text: str,
) -> Tuple[List[str], np.ndarray]:
    """BERTurk [CLS] tokenlerini orijinal kelimelere topla."""
    words, word_scores = [], []
    current_word, current_scores = "", []
    prev_end = 0

    for token, score, (start, end) in zip(tokens, scores, offsets):
        if token in ("[CLS]", "[SEP]", "[PAD]") or start == end:
            if current_word:
                words.append(current_word)
                word_scores.append(np.mean(current_scores))
                current_word, current_scores = "", []
            continue

        # Kelime sınırı: bir önceki token ile arasında boşluk var mı?
        if start > prev_end and current_word:
            words.append(current_word)
            word_scores.append(np.mean(current_scores))
            current_word, current_scores = "", []

        # "##" prefix'li subwordleri temizle (BERTurk kullanır)
        clean = token.lstrip("#")
        current_word += clean
        current_scores.append(score)
        prev_end = end

    if current_word:
        words.append(current_word)
        word_scores.append(np.mean(current_scores))

    return words, np.array(word_scores, dtype=np.float32)


# ─── Lexikon–Ağırlık Analizi ─────────────────────────────────────────────────

def analyze_lexicon_weights(
    model: "LexiconBERTModel",
) -> Dict[str, np.ndarray]:
    """
    Sınıflandırıcının lexikon bloğu ağırlıklarını analiz eder.

    classifier.weight: (10, 778)
    Lexikon bloğu: [:, 768:778]  → (10, 10) — her duygu × her NRC kategorisi

    Döner: {"emotion": np.ndarray(10,), ...}  — her duygu için NRC katkı vektörü
    """
    W = model.classifier.weight.detach().cpu().numpy()  # (10, 778)
    W_lex = W[:, 768:]                                  # (10, 10) lexikon blok

    result = {}
    for i, emotion in enumerate(EMOTIONS):
        result[emotion] = W_lex[i]   # (10,) — NRC kanal ağırlıkları
    return result


def top_keywords_per_emotion(
    model: "LexiconBERTModel",
    n: int = 5,
    method: str = "lexicon_linear",
) -> Dict[str, List[dict]]:
    """
    Her duygu için en etkili N kelimeyi bulur.

    method="lexicon_linear":
      Her kelime için  importance = W_lex[emotion] · lexicon[word]
      → sınıflandırıcı ağırlıkları × NRC vektörü skoru
      → Eğitimden bağımsız, anında çalışır.

    Döner:
      {
        "gurur": [
          {"word": "gurur", "score": 0.82, "nrc_channels": ["trust","positive"]},
          ...
        ],
        ...
      }
    """
    from train_exp3_lexicon import TURKISH_LEXICON

    W     = model.classifier.weight.detach().cpu().numpy()  # (10, 778)
    W_lex = W[:, 768:]                                      # (10, 10)
    b     = model.classifier.bias.detach().cpu().numpy()    # (10,)

    keywords: Dict[str, List[dict]] = {}

    for i, emotion in enumerate(EMOTIONS):
        w_vec = W_lex[i]  # (10,) — bu duygu için NRC ağırlıkları

        scored_words = []
        for word, lex_vec in TURKISH_LEXICON.items():
            # lex_vec: 0/1 ikili NRC vektörü
            score = float(np.dot(w_vec, lex_vec))
            if score == 0.0:
                continue
            # Hangi NRC kanalları aktif?
            active_nrc = [NRC_CATS[j] for j in range(10) if lex_vec[j] > 0.5]
            scored_words.append({
                "word":         word,
                "score":        score,
                "nrc_channels": active_nrc,
            })

        # En yüksek skorlu N kelime
        scored_words.sort(key=lambda x: -x["score"])
        keywords[emotion] = scored_words[:n]

    return keywords


# ─── Görselleştirme: Force / Waterfall Plot ──────────────────────────────────

def plot_word_contributions(
    words:      List[str],
    scores:     np.ndarray,
    base_prob:  float,
    final_prob: float,
    emotion:    str,
    title:      str = "",
    save_path:  Optional[Path] = None,
) -> plt.Figure:
    """
    Kelime katkılarını gösteren yatay waterfall (şelale) çubuğu çizer.

    Pozitif katkı (duyguyu destekleyen) → kırmızı/turuncu
    Negatif katkı (duyguyu körelten)    → mavi
    """
    # En büyük mutlak etki ile ilk 15 kelimeyi göster
    n_show = min(15, len(words))
    order  = np.argsort(np.abs(scores))[::-1][:n_show]
    words_s  = [words[j]  for j in order]
    scores_s = scores[order]

    # Görüntü için en küçükten en büyüğe sırala
    sort_idx = np.argsort(scores_s)
    words_s  = [words_s[j]  for j in sort_idx]
    scores_s = scores_s[sort_idx]

    colors = ["#e74c3c" if s > 0 else "#3498db" for s in scores_s]

    emotion_color = EMOTION_COLORS.get(emotion, "#2c3e50")

    fig, (ax_bar, ax_info) = plt.subplots(
        1, 2,
        figsize=(13, max(5, n_show * 0.45 + 2)),
        gridspec_kw={"width_ratios": [3, 1]},
        facecolor="#fafafa",
    )

    # ─── Sol: Waterfall / Contribution Bars ────────────────────────────────
    ax_bar.set_facecolor("#fafafa")
    bars = ax_bar.barh(
        range(len(words_s)),
        scores_s,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        height=0.72,
    )
    ax_bar.set_yticks(range(len(words_s)))
    ax_bar.set_yticklabels(words_s, fontsize=11)
    ax_bar.axvline(0, color="#2c3e50", linewidth=1.2, alpha=0.7)
    ax_bar.set_xlabel("Katkı Skoru (Gradient × Giriş)", fontsize=10)
    ax_bar.set_title(
        title or f"Kelime Katkıları — {emotion.upper()}",
        fontsize=13, fontweight="bold", pad=10
    )
    ax_bar.spines[["top","right"]].set_visible(False)

    # Değer etiketleri
    for bar, sc in zip(bars, scores_s):
        ax_bar.text(
            sc + (0.01 if sc >= 0 else -0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{sc:+.3f}",
            va="center",
            ha="left" if sc >= 0 else "right",
            fontsize=8.5,
            color="#2c3e50",
        )

    # ─── Sağ: Özet Bilgi Kutusu ────────────────────────────────────────────
    ax_info.set_facecolor(emotion_color)
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis("off")

    info_lines = [
        ("Tahmin Duygu", emotion),
        ("Temel Olasılık", f"{base_prob:.2%}"),
        ("Final Olasılık", f"{final_prob:.2%}"),
        ("Değişim", f"{final_prob - base_prob:+.2%}"),
        ("Analiz Yöntemi", "Gradient Saliency"),
        ("Model", "Exp4 Hybrid"),
    ]
    ax_info.text(0.5, 0.96, "ÖZET", ha="center", va="top",
                 fontsize=12, fontweight="bold", color="white")
    y = 0.83
    for label, val in info_lines:
        ax_info.text(0.08, y,  f"{label}:", fontsize=8.5, color="white", alpha=0.8)
        ax_info.text(0.08, y - 0.055, val, fontsize=10, fontweight="bold", color="white")
        y -= 0.14

    # Legend
    pos_patch = mpatches.Patch(color="#e74c3c", label="Duyguyu destekler (+)")
    neg_patch = mpatches.Patch(color="#3498db", label="Duyguyu köreltir (-)")
    ax_bar.legend(handles=[pos_patch, neg_patch], loc="lower right",
                  fontsize=9, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [KAYIT] {save_path.relative_to(ROOT)}")

    return fig


# ─── Görselleştirme: SHAP Force Plot (HTML) ──────────────────────────────────

def save_shap_force_html(
    shap_values,       # shap.Explanation object
    emotion_idx: int,
    text: str,
    save_path: Path,
):
    """SHAP force plot'u etkileşimli HTML olarak kaydeder."""
    if not _SHAP_OK:
        return

    sv_for_class = shap_values[:, :, emotion_idx]
    shap.initjs()
    force = shap.plots.force(sv_for_class[0], show=False, matplotlib=False)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>SHAP Force Plot — {EMOTIONS[emotion_idx]}</title>
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; background: #f8f9fa; padding: 20px; }}
    .header {{ color: #2c3e50; margin-bottom: 16px; }}
    .meta {{ color: #7f8c8d; font-size: 13px; margin-bottom: 12px; }}
    .text-box {{ background: white; border-left: 4px solid {EMOTION_COLORS[EMOTIONS[emotion_idx]]};
                 padding: 10px 14px; border-radius: 4px; margin-bottom: 18px;
                 font-size: 15px; color: #2c3e50; }}
  </style>
</head>
<body>
  <h2 class="header">SHAP Katkı Analizi — {EMOTIONS[emotion_idx].upper()}</h2>
  <div class="meta">Kırmızı kelimeler duyguyu destekler | Mavi kelimeler duyguyu köreltir</div>
  <div class="text-box">"{text}"</div>
  {force.html()}
</body>
</html>"""

    save_path.write_text(html_content, encoding="utf-8")
    print(f"  [KAYIT] {save_path.relative_to(ROOT)}")


# ─── Görselleştirme: SHAP Summary Plot ───────────────────────────────────────

def plot_shap_summary(
    shap_values_list: List[np.ndarray],
    feature_names:    List[str],
    emotion:          str,
    save_path:        Optional[Path] = None,
) -> plt.Figure:
    """
    Çok örnek SHAP özet çubuğu — en büyük ortalama |SHAP| olan kelimeler.
    shap_values_list: list of (n_tokens,) arrays, bir duygu için
    feature_names:    list of str (token/kelime adları)
    """
    # Tüm örneklerdeki kelimeleri topla
    word_scores: Dict[str, List[float]] = {}
    for sv, fn in zip(shap_values_list, feature_names):
        for word, score in zip(fn, sv):
            word_scores.setdefault(word, []).append(score)

    # Ortalama |SHAP| değeri
    word_mean = {w: np.mean(np.abs(v)) for w, v in word_scores.items()}
    word_sign = {w: np.mean(v) for w, v in word_scores.items()}

    # En büyük 20 kelime
    top_words = sorted(word_mean.items(), key=lambda x: -x[1])[:20]
    names  = [w for w, _ in top_words]
    values = [word_mean[w] for _, w in [(n, n) for n in names]]
    signs  = [word_sign[n] for n in names]
    colors = ["#e74c3c" if s > 0 else "#3498db" for s in signs]

    # Küçükten büyüğe sırala (en büyük en üstte)
    order  = np.argsort(values)
    names  = [names[i]  for i in order]
    values = [values[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.42 + 2)), facecolor="#fafafa")
    ax.set_facecolor("#fafafa")
    ax.barh(range(len(names)), values, color=colors, edgecolor="white", linewidth=0.7, height=0.72)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10.5)
    ax.set_xlabel("Ortalama |SHAP| Değeri", fontsize=10)
    ax.set_title(
        f"SHAP Özet — {emotion.upper()} ({len(shap_values_list)} örnek)",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.spines[["top","right"]].set_visible(False)

    pos_patch = mpatches.Patch(color="#e74c3c", label="Ortalama pozitif katkı")
    neg_patch = mpatches.Patch(color="#3498db", label="Ortalama negatif katkı")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [KAYIT] {save_path.relative_to(ROOT)}")
    return fig


# ─── Görselleştirme: Lexikon Ağırlık Isı Haritası ────────────────────────────

def plot_lexicon_weight_heatmap(
    model: "LexiconBERTModel",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    10×10 ısı haritası: duygu sınıfı × NRC kategorisi için sınıflandırıcı ağırlıkları.
    Her hücre: classifier.weight[duygu, 768+nrc_idx]
    """
    W_lex = analyze_lexicon_weights(model)   # {emotion: (10,) ndarray}
    matrix = np.stack([W_lex[e] for e in EMOTIONS], axis=0)  # (10, 10)

    vmax = np.abs(matrix).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(11, 8), facecolor="white")

    im = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")
    plt.colorbar(im, ax=ax, label="Ağırlık Değeri", shrink=0.8)

    ax.set_xticks(range(10))
    ax.set_xticklabels([NRC_TR[c] for c in NRC_CATS], rotation=40, ha="right", fontsize=10)
    ax.set_yticks(range(10))
    ax.set_yticklabels(EMOTIONS, fontsize=11)

    # Hücre değerleri
    for i in range(10):
        for j in range(10):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(v) < vmax * 0.6 else "white",
                    fontweight="bold" if abs(v) > vmax * 0.5 else "normal")

    ax.set_title(
        "Sınıflandırıcı Lexikon Blok Ağırlıkları\n"
        "Duygu Sınıfı × NRC Kategorisi",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("NRC Duygu Kategorisi", fontsize=11)
    ax.set_ylabel("Tahmin Edilen Duygu", fontsize=11)

    # Izgara
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [KAYIT] {save_path.relative_to(ROOT)}")
    return fig


# ─── Görselleştirme: En Etkili 5 Kelime (Hoca Sunumu) ────────────────────────

def plot_top_keywords_panel(
    keywords: Dict[str, List[dict]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Her duygu için en etkili 5 kelimeyi 2×5 grid panelde gösterir.
    Hoca sunumu için optimize edilmiş büyük yazı tipli görsel.
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 9), facecolor="#1a1a2e")
    fig.suptitle(
        "Her Duygu İçin En Etkili 5 Kelime\n"
        "BERTurk + Sözlük (NRC kategorili) Hibrit Modeli",
        fontsize=16, fontweight="bold", color="white", y=1.02,
    )

    for idx, (emotion, ax) in enumerate(zip(EMOTIONS, axes.flat)):
        color = EMOTION_COLORS[emotion]
        ax.set_facecolor("#16213e")
        ax.spines[:].set_visible(False)

        words_data = keywords.get(emotion, [])
        if not words_data:
            ax.text(0.5, 0.5, "Veri yok", ha="center", va="center",
                    color="gray", transform=ax.transAxes)
            ax.set_title(emotion, color=color, fontweight="bold", fontsize=11)
            continue

        words_list = [d["word"]  for d in words_data]
        scores     = [d["score"] for d in words_data]

        # Normalize scores for bar width
        max_s = max(abs(s) for s in scores) or 1.0
        norm_s = [s / max_s for s in scores]

        y_pos = list(range(len(words_list)))
        bar_colors = [color if s >= 0 else "#e74c3c" for s in norm_s]

        bars = ax.barh(y_pos, norm_s, color=bar_colors, alpha=0.85,
                       edgecolor="#0f3460", linewidth=0.5, height=0.65)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(words_list, fontsize=10.5, color="white")
        ax.set_xlim(-0.05, 1.15)
        ax.set_xticks([])
        ax.axvline(0, color="white", linewidth=0.8, alpha=0.4)

        # NRC kanalları
        for bar, word_d in zip(bars, words_data):
            nrc_str = ", ".join(word_d.get("nrc_channels", [])[:2])
            ax.text(
                bar.get_width() + 0.03,
                bar.get_y() + bar.get_height() / 2,
                nrc_str,
                va="center", ha="left",
                fontsize=7.5, color="#bdc3c7",
            )

        ax.set_title(
            emotion.upper(),
            color=color, fontweight="bold", fontsize=11, pad=6,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="#1a1a2e")
        print(f"  [KAYIT] {save_path.relative_to(ROOT)}")
    return fig


# ─── Görselleştirme: Gradient Saliency Isı Haritası ─────────────────────────

def plot_gradient_saliency_heatmap(
    text:      str,
    words:     List[str],
    scores:    np.ndarray,
    emotion:   str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Token saliency'yi metin üzerinde ısı haritası olarak gösterir.
    Her kelime kutusunu normalize edilmiş skor ile boyar.
    """
    max_s = scores.max() if scores.max() > 1e-8 else 1.0
    norm_s = scores / max_s

    n_words = len(words)
    cols = min(8, n_words)
    rows = (n_words + cols - 1) // cols

    fig, ax = plt.subplots(figsize=(max(10, cols * 1.6), rows * 1.3 + 1.5),
                           facecolor="white")
    ax.axis("off")

    cmap = plt.cm.YlOrRd
    emotion_color = EMOTION_COLORS.get(emotion, "#e74c3c")

    cell_w, cell_h = 1.0 / cols, 1.0 / (rows + 0.5)

    for i, (word, score) in enumerate(zip(words, norm_s)):
        row = i // cols
        col = i % cols
        x   = col * cell_w
        y   = 1.0 - (row + 1) * cell_h * (rows + 0.5) / rows

        bg_color = cmap(float(score))
        rect = mpatches.FancyBboxPatch(
            (x + 0.01, y + 0.01),
            cell_w * 0.92, cell_h * 0.85,
            boxstyle="round,pad=0.01",
            facecolor=bg_color,
            edgecolor="#cccccc",
            linewidth=0.5,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        text_color = "white" if score > 0.55 else "#2c3e50"
        ax.text(
            x + cell_w / 2,
            y + cell_h * 0.5,
            word,
            ha="center", va="center",
            fontsize=10, fontweight="bold" if score > 0.4 else "normal",
            color=text_color,
            transform=ax.transAxes,
        )
        ax.text(
            x + cell_w / 2,
            y + cell_h * 0.15,
            f"{score:.2f}",
            ha="center", va="center",
            fontsize=7, color=text_color, alpha=0.8,
            transform=ax.transAxes,
        )

    ax.set_title(
        f"Gradient Saliency Isı Haritası — {emotion.upper()}\n"
        f'"{textwrap.shorten(text, 70)}"',
        fontsize=12, fontweight="bold", pad=10,
        color="#2c3e50",
    )

    # Renk çubuğu (ax.transAxes koordinatında sahte)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.03, pad=0.02, shrink=0.5)
    cbar.set_label("Normalize Saliency Skoru", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [KAYIT] {save_path.relative_to(ROOT)}")
    return fig


# ─── Ana SHAP Açıklayıcı ─────────────────────────────────────────────────────

class SHAPInterpreter:
    """
    BERTurk + Lexikon Hibrit Model için tam SHAP + Gradient yorumlayıcı.

    Özellikler:
      - SHAP Partition Explainer (SHAP yüklüyse)
      - Gradient × Input Saliency (her zaman)
      - Lexikon-Lineer analizi (eğitim gerekmez)
      - Top-5 keyword per emotion
    """

    def __init__(self, loader: ModelLoader):
        self.loader  = loader
        self.predict = PredictWrapper(loader)
        self._explainer = None

    def _get_shap_explainer(self):
        """SHAP explainer'ı oluştur (ilk çağrıda, lazy init)."""
        if self._explainer is not None:
            return self._explainer
        if not _SHAP_OK:
            return None

        masker = shap.maskers.Text(self.loader.tokenizer)
        self._explainer = shap.Explainer(
            self.predict,
            masker,
            output_names=EMOTIONS,
            algorithm="partition",
        )
        return self._explainer

    # ── SHAP Analizi ──────────────────────────────────────────────────────────

    def explain_shap(
        self,
        text: str,
        max_evals: int = 300,
        batch_size: int = 16,
    ):
        """
        Bir metin için SHAP değerlerini hesaplar.
        Döner: shap.Explanation nesnesi (SHAP yüklü değilse None)
        """
        explainer = self._get_shap_explainer()
        if explainer is None:
            return None

        print(f"  [SHAP] Hesaplanıyor (max_evals={max_evals})...")
        t0 = time.time()
        shap_values = explainer([text], max_evals=max_evals, batch_size=batch_size)
        print(f"  [SHAP] Tamamlandı: {time.time()-t0:.1f}s")
        return shap_values

    def force_plot(
        self,
        text:       str,
        emotion:    str,
        max_evals:  int = 300,
        save_html:  bool = True,
        save_png:   bool = True,
    ) -> dict:
        """
        Bir metin ve duygu için force plot üretir.
        HTML (etkileşimli) + PNG (baskı kalitesi) kaydeder.
        """
        emo_idx = EMOTIONS.index(emotion)

        # Olasılıkları hesapla
        probs = self.predict([text])[0]
        final_prob = float(probs[emo_idx])
        base_prob  = 1.0 / len(EMOTIONS)  # uniform prior

        outputs = {}

        # 1. SHAP (varsa)
        if _SHAP_OK and save_html:
            shap_values = self.explain_shap(text, max_evals=max_evals)
            if shap_values is not None:
                html_path = OUTDIR / f"force_plot_{emotion}.html"
                save_shap_force_html(shap_values, emo_idx, text, html_path)
                outputs["shap_html"] = str(html_path)

        # 2. Gradient Saliency (PNG)
        if save_png:
            try:
                words, scores = compute_gradient_saliency(text, emo_idx, self.loader)
                png_path = OUTDIR / f"force_plot_{emotion}.png"
                plot_word_contributions(
                    words=words,
                    scores=scores,
                    base_prob=base_prob,
                    final_prob=final_prob,
                    emotion=emotion,
                    title=f'Kelime Katkıları — {emotion.upper()}\n"{textwrap.shorten(text, 60)}"',
                    save_path=png_path,
                )
                plt.close("all")
                outputs["gradient_png"] = str(png_path)

                # Isı haritası
                heat_path = OUTDIR / f"gradient_saliency_{emotion}.png"
                plot_gradient_saliency_heatmap(text, words, scores, emotion, heat_path)
                plt.close("all")
                outputs["saliency_heatmap"] = str(heat_path)

            except Exception as e:
                print(f"  [UYARI] Gradient hesaplaması başarısız: {e}")

        return outputs

    def summary_plot(
        self,
        texts:     List[str],
        emotion:   str,
        max_evals: int = 200,
    ) -> Optional[Path]:
        """
        Birden fazla örnek için SHAP özet grafiği (varsa SHAP).
        Yoksa gradient bazlı özet üretir.
        """
        emo_idx   = EMOTIONS.index(emotion)
        save_path = OUTDIR / f"summary_plot_{emotion}.png"

        if _SHAP_OK:
            explainer = self._get_shap_explainer()
            if explainer:
                print(f"  [SHAP Summary] {len(texts)} örnek, duygu={emotion}...")
                shap_values = explainer(texts, max_evals=max_evals, batch_size=8)

                # Token bazlı SHAP değerlerini kelime düzeyinde topla
                all_svs, all_words = [], []
                for i in range(len(texts)):
                    sv  = shap_values[i, :, emo_idx].values
                    tks = shap_values[i, :, emo_idx].data
                    all_svs.append(sv)
                    all_words.append(tks.tolist())

                plot_shap_summary(all_svs, all_words, emotion, save_path)
                plt.close("all")
                return save_path

        # Fallback: Gradient tabanlı özet
        print(f"  [Gradient Summary] {len(texts)} örnek, duygu={emotion}...")
        all_svs, all_words = [], []
        for text in texts:
            try:
                words, scores = compute_gradient_saliency(text, emo_idx, self.loader)
                all_svs.append(scores)
                all_words.append(words)
            except Exception:
                continue

        if all_svs:
            plot_shap_summary(all_svs, all_words, emotion, save_path)
            plt.close("all")

        return save_path

    # ── Lexikon-Lineer Anahtar Kelime Analizi ─────────────────────────────────

    def top_keywords(self, n: int = 5) -> Dict[str, List[dict]]:
        """Her duygu için en etkili N kelimeyi döner (lexikon × ağırlık)."""
        return top_keywords_per_emotion(self.loader.model, n=n)

    # ── Tam Rapor ─────────────────────────────────────────────────────────────

    def full_report(self, shap_max_evals: int = 200):
        """
        Tüm 10 duygu için kapsamlı yorumlanabilirlik raporu üretir.
        results/interpretability/ altına kaydeder.
        """
        print(f"\n{'='*60}")
        print("  TAM YORUMLANABILIRLIK RAPORU")
        print(f"{'='*60}")
        print(f"  Cikti: {OUTDIR}\n")

        t_start = time.time()

        # ── 1. Lexikon Ağırlık Isı Haritası ───────────────────────────────────
        print("[1/5] Lexikon ağırlık ısı haritası...")
        plot_lexicon_weight_heatmap(
            self.loader.model,
            OUTDIR / "lexicon_weights_heatmap.png",
        )
        plt.close("all")

        # ── 2. En Etkili 5 Kelime Paneli ──────────────────────────────────────
        print("[2/5] En etkili 5 kelime (tüm duygular)...")
        keywords = self.top_keywords(n=5)
        plot_top_keywords_panel(keywords, OUTDIR / "top_keywords_panel.png")
        plt.close("all")

        # JSON olarak da kaydet
        kw_path = OUTDIR / "top_keywords_all.json"
        with open(kw_path, "w", encoding="utf-8") as f:
            json.dump(keywords, f, ensure_ascii=False, indent=2)
        print(f"  [KAYIT] {kw_path.relative_to(ROOT)}")

        # ── 3. Her Duygu için Force Plot ──────────────────────────────────────
        print("[3/5] Her duygu için force plot (gradient saliency)...")
        for emotion, texts in DEMO_TEXTS.items():
            text = texts[0]  # her duygunun ilk örnek cümlesi
            print(f"  -> {emotion}: \"{textwrap.shorten(text, 50)}\"")
            self.force_plot(
                text     = text,
                emotion  = emotion,
                save_html = _SHAP_OK,
                save_png  = True,
                max_evals = shap_max_evals,
            )

        plt.close("all")

        # ── 4. SHAP Summary Plot (seçilmiş duygular) ──────────────────────────
        print("[4/5] SHAP / Gradient summary plot (gurur, öfke, mutluluk)...")
        for emotion in ["gurur", "öfke", "mutluluk"]:
            texts = DEMO_TEXTS[emotion]
            self.summary_plot(texts, emotion, max_evals=shap_max_evals)
        plt.close("all")

        # ── 5. Tüm Duyguların Lexikon Top-5 Çubuk Grafiği ────────────────────
        print("[5/5] Duygu bazlı lexikon kelime çubukları...")
        self._plot_per_emotion_bars(keywords)
        plt.close("all")

        # ── Rapor Özeti ───────────────────────────────────────────────────────
        elapsed = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"  RAPOR TAMAMLANDI — {elapsed:.1f}s")
        print(f"  Cikti: {OUTDIR}")
        print(f"{'='*60}\n")

        # Hoca sunumu için özet yazdır
        self._print_professor_summary(keywords)

    def _plot_per_emotion_bars(self, keywords: Dict[str, List[dict]]):
        """Her duygu için ayrı bir kelime çubuğu grafiği kaydeder."""
        for emotion, words_data in keywords.items():
            if not words_data:
                continue
            words_list = [d["word"]  for d in words_data]
            scores     = [d["score"] for d in words_data]
            max_s = max(abs(s) for s in scores) or 1.0
            norm_s = [s / max_s for s in scores]

            color = EMOTION_COLORS[emotion]
            fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="white")
            ax.set_facecolor("#f8f9fa")

            bars = ax.barh(range(len(words_list)), norm_s[::-1],
                           color=color, alpha=0.85, edgecolor="white",
                           linewidth=0.8, height=0.65)
            ax.set_yticks(range(len(words_list)))
            ax.set_yticklabels(words_list[::-1], fontsize=11)
            ax.set_xlim(0, 1.2)
            ax.set_xlabel("Normalize Skor", fontsize=9)
            ax.set_title(f"En Etkili Kelimeler — {emotion.upper()}", fontsize=12,
                         fontweight="bold", color=color, pad=8)
            ax.spines[["top","right","bottom"]].set_visible(False)
            ax.axvline(0, color="#7f8c8d", linewidth=0.8)

            for bar, word_d in zip(bars, reversed(words_data)):
                nrc = ", ".join(word_d.get("nrc_channels", [])[:2])
                ax.text(
                    bar.get_width() + 0.03,
                    bar.get_y() + bar.get_height() / 2,
                    nrc,
                    va="center", ha="left", fontsize=8, color="#7f8c8d",
                )

            plt.tight_layout()
            save_path = OUTDIR / f"top_keywords_{emotion.replace('ş','s').replace('ü','u').replace('ö','o').replace('ı','i').replace('ğ','g').replace('ç','c')}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  [KAYIT] {save_path.relative_to(ROOT)}")
            plt.close(fig)

    def _print_professor_summary(self, keywords: Dict[str, List[dict]]):
        """Terminal'e hoca sunumu için yapılandırılmış özet yazar."""
        print("\n" + "="*65)
        print("  HOCA SUNUMU — HER DUYGU İÇİN EN ETKİLİ 5 KELİME")
        print("  (Lexikon × Sınıflandırıcı Ağırlık Analizi)")
        print("="*65)
        for emotion in EMOTIONS:
            words_data = keywords.get(emotion, [])
            word_str   = ", ".join(
                f'"{d["word"]}" ({d["score"]:+.3f})' for d in words_data
            )
            print(f"\n  {emotion.upper():<12}:")
            print(f"    {word_str}")
            nrc_mentioned = set()
            for d in words_data:
                nrc_mentioned.update(d.get("nrc_channels", []))
            if nrc_mentioned:
                print(f"    NRC kanalları: {', '.join(sorted(nrc_mentioned))}")
        print("="*65 + "\n")


# ─── Tek Metin Analizi ───────────────────────────────────────────────────────

def analyze_text(
    text:        str,
    loader:      ModelLoader,
    target_emo:  Optional[str] = None,
    save_shap:   bool = True,
):
    """Tek metin için tam analiz ve görselleştirme."""
    predict = PredictWrapper(loader)
    probs   = predict([text])[0]

    print(f"\nMetin: \"{text}\"")
    print(f"\nOlasılıklar:")
    order = np.argsort(probs)[::-1]
    for idx in order:
        bar = "█" * int(probs[idx] * 40)
        print(f"  {EMOTIONS[idx]:<12} {probs[idx]:.4f}  {bar}")

    dominant_idx = int(np.argmax(probs))
    dominant_emo = EMOTIONS[dominant_idx]
    target_emo   = target_emo or dominant_emo

    print(f"\nAnaliz hedefi: {target_emo.upper()}")

    interp = SHAPInterpreter(loader)
    outputs = interp.force_plot(text, target_emo, save_html=save_shap)

    print(f"\nKaydedilen dosyalar:")
    for k, v in outputs.items():
        print(f"  {k}: {Path(v).relative_to(ROOT)}")

    return probs, outputs


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SHAP Yorumlanabilirlik — BERTurk + Sözlük (NRC kategorili)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Ornekler:
          python shap_interpretability.py --text "Bu film harikaydi!"
          python shap_interpretability.py --text "Korktum ve kaçtım" --emotion korku
          python shap_interpretability.py --top-words
          python shap_interpretability.py --demo-all
          python shap_interpretability.py --full-report
          python shap_interpretability.py --heatmap
        """),
    )
    parser.add_argument("--text",       type=str,   default=None,
                        help="Analiz edilecek metin")
    parser.add_argument("--emotion",    type=str,   default=None,
                        choices=EMOTIONS,
                        help="Hedef duygu (varsayılan: tahmin edilen dominant duygu)")
    parser.add_argument("--top-words",  action="store_true",
                        help="Her duygu için en etkili 5 kelimeyi çıkar ve kaydet")
    parser.add_argument("--demo-all",   action="store_true",
                        help="Tüm demo metinleri için analiz çalıştır")
    parser.add_argument("--full-report",action="store_true",
                        help="Kapsamlı yorumlanabilirlik raporu üret")
    parser.add_argument("--heatmap",    action="store_true",
                        help="Lexikon ağırlık ısı haritası üret")
    parser.add_argument("--shap-evals", type=int,   default=300,
                        help="SHAP max_evals (daha yüksek = daha doğru ama yavaş)")
    parser.add_argument("--no-shap",    action="store_true",
                        help="SHAP'ı devre dışı bırak (sadece gradient/lexikon)")

    args = parser.parse_args()

    if args.no_shap:
        global _SHAP_OK
        _SHAP_OK = False

    # Model yükle
    loader = ModelLoader()
    if not loader.load():
        print("[HATA] Model yüklenemedi.")
        sys.exit(1)

    if args.full_report:
        interp = SHAPInterpreter(loader)
        interp.full_report(shap_max_evals=args.shap_evals)

    elif args.top_words:
        interp  = SHAPInterpreter(loader)
        kw      = interp.top_keywords(n=5)
        interp._print_professor_summary(kw)
        plot_top_keywords_panel(kw, OUTDIR / "top_keywords_panel.png")
        plt.close("all")
        kw_path = OUTDIR / "top_keywords_all.json"
        with open(kw_path, "w", encoding="utf-8") as f:
            json.dump(kw, f, ensure_ascii=False, indent=2)
        print(f"\n[KAYIT] {kw_path.relative_to(ROOT)}")

    elif args.heatmap:
        plot_lexicon_weight_heatmap(
            loader.model,
            OUTDIR / "lexicon_weights_heatmap.png",
        )
        plt.close("all")

    elif args.demo_all:
        interp = SHAPInterpreter(loader)
        for emotion, texts in DEMO_TEXTS.items():
            text = texts[0]
            print(f"\n--- {emotion.upper()} ---")
            interp.force_plot(
                text=text, emotion=emotion,
                save_html=_SHAP_OK, save_png=True,
                max_evals=args.shap_evals,
            )
        plt.close("all")

    elif args.text:
        analyze_text(
            text       = args.text,
            loader     = loader,
            target_emo = args.emotion,
            save_shap  = _SHAP_OK,
        )

    else:
        parser.print_help()
        print("\n[BILGI] Hizli baslangic:")
        print("  python shap_interpretability.py --top-words      # aninda calisir")
        print("  python shap_interpretability.py --full-report    # tam rapor")
        print("  python shap_interpretability.py --text 'Cok mutluyum!'")


if __name__ == "__main__":
    main()
