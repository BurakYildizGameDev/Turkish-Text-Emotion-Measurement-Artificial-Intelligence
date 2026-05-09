"""
Error Analysis — Türkçe Duygu Analizi
======================================
Test setinde modelin yanlış yaptığı örnekleri analiz eder.

Dürüst Bir Uyarı:
  Bu test setinde sınıf dağılımı son derece eşitsizdir:
    gurur=8, utanç=37, korku=95 örnek
  Bu nedenle azınlık sınıflarındaki hata analizi istatistiksel olarak
  güvenilir değildir. Bulgular "model başarısız" değil,
  "veri yetersiz" şeklinde yorumlanmalıdır.

Çıktılar (results/error_analysis/):
  error_report.csv            — tüm yanlış örnekler
  error_report_confident.csv  — yüksek güvenlikli hatalar (>70%)
  confusion_heatmap.png       — normalize edilmiş karmaşıklık matrisi
  top_confusion_pairs.png     — en çok karışan duygu çiftleri
  per_class_metrics.png       — sınıf bazlı F1/Precision/Recall
  confidence_histogram.png    — doğru vs yanlış tahmin güvenlikleri
  pair_analysis_top3.png      — en kötü 3 çiftin derinlemesine analizi
  error_analysis_report.json  — tam yapılandırılmış rapor

Kullanım:
  python error_analysis.py                     # tam analiz
  python error_analysis.py --max-samples 5000  # hızlı ön analiz
  python error_analysis.py --no-model          # sadece veri analizi
  python error_analysis.py --batch-size 128    # GPU bellek ayarı
"""

import sys
import io
import re
import json
import time
import warnings
import argparse
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict

# Windows cp1254 codec sorununu önle
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent
OUTDIR  = ROOT / "results" / "error_analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

# ─── Bileşenler ──────────────────────────────────────────────────────────────

try:
    from transformers import AutoTokenizer
    from train_exp3_lexicon import (
        LexiconBERTModel, LexiconExtractor,
        MODEL_NAME, NUM_LABELS, LEXICON_DIM, NRC_CATS,
        load_lexicon_bert_weights,
    )
    _COMPONENTS_OK = True
except Exception as e:
    print(f"[UYARI] Bileşen yüklenemedi: {e}")
    _COMPONENTS_OK = False

# ─── Sabitler ────────────────────────────────────────────────────────────────

EMOTIONS = [
    "mutluluk", "üzüntü", "öfke", "korku",
    "şaşkınlık", "tiksinti", "sevgi", "nötr", "utanç", "gurur",
]
EMO2ID = {e: i for i, e in enumerate(EMOTIONS)}
ID2EMO = {i: e for i, e in enumerate(EMOTIONS)}

EMOTION_COLORS = {
    "mutluluk":  "#f1c40f", "üzüntü":    "#3498db", "öfke":      "#e74c3c",
    "korku":     "#9b59b6", "şaşkınlık": "#1abc9c", "tiksinti":  "#27ae60",
    "sevgi":     "#e91e63", "nötr":      "#95a5a6", "utanç":     "#e67e22",
    "gurur":     "#2c3e50",
}

# Türkçe stop words (analiz için çıkarılır)
TR_STOPWORDS = {
    "ve","ile","bir","bu","da","de","mi","mu","mü","ki","ama","için",
    "çok","daha","olan","gibi","ne","en","o","ben","sen","biz","siz",
    "hem","ya","veya","ya","değil","var","yok","bu","şu","ise","kadar",
    "olan","oldu","olan","edildi","göre","sonra","önce","her","tüm",
    "hiç","bile","zaten","artık","nasıl","neden","çünkü","eğer","hep",
}

# Test splitleri için minimum güvenilir örnek sayısı
MIN_RELIABLE_SAMPLES = 30

# ─── Veri Yükleme ────────────────────────────────────────────────────────────

def load_test_data(max_samples: Optional[int] = None) -> pd.DataFrame:
    """Test split'i yükler. Eğer yoksa processed dataset'ten örnek alır."""
    test_path = ROOT / "data" / "splits" / "master_splits" / "test.parquet"

    if test_path.exists():
        print(f"[VERİ] Test split yükleniyor: {test_path.relative_to(ROOT)}")
        df = pd.read_parquet(test_path)
    else:
        processed = ROOT / "data" / "processed" / "master_dataset.parquet"
        if not processed.exists():
            processed = ROOT / "data" / "processed" / "combined_clean.parquet"
        if not processed.exists():
            raise FileNotFoundError(
                "Test split veya processed dataset bulunamadı.\n"
                "Çalıştır: python data_engine.py --rebuild --split"
            )
        print(f"[VERİ] Test split yok, processed dataset'ten %15 örnek alınıyor...")
        df = pd.read_parquet(processed)
        df = df.sample(frac=0.15, random_state=42).reset_index(drop=True)

    # label_id sütununu doğrula / oluştur
    if "label_id" not in df.columns:
        # label sütunundan eşleştir
        label_map = {}
        for i, e in enumerate(EMOTIONS):
            label_map[e] = i
        # Normalize edilmiş etiket varyantları
        ascii_map = {
            "uzuntu": 1, "ofke": 2, "saskınlık": 4, "notr": 7, "utanc": 8,
        }
        def _to_id(label):
            l = str(label).strip().lower()
            if l in label_map:   return label_map[l]
            if l in ascii_map:   return ascii_map[l]
            return -1
        df["label_id"] = df["label"].apply(_to_id)
        df = df[df["label_id"] >= 0].reset_index(drop=True)

    # Geçersiz label_id'leri temizle
    df = df[df["label_id"].between(0, 9)].reset_index(drop=True)

    if max_samples and len(df) > max_samples:
        # Stratified örnekleme
        df = (
            df.groupby("label_id", group_keys=False)
            .apply(lambda g: g.sample(
                min(len(g), max(1, int(max_samples * len(g) / len(df)))),
                random_state=42,
            ))
            .reset_index(drop=True)
        )

    print(f"[VERİ] {len(df):,} örnek yüklendi")
    vc = df["label_id"].value_counts().sort_index()
    for lid, cnt in vc.items():
        emo    = ID2EMO.get(int(lid), "?")
        caveat = " ⚠️ Az örnek" if cnt < MIN_RELIABLE_SAMPLES else ""
        print(f"  {emo:<12} {cnt:>6,}{caveat}")
    return df


# ─── Model Yükleme ───────────────────────────────────────────────────────────

def load_model(device: torch.device):
    """Eğitilmiş ağırlıkları yükler. Bulunamazsa None döner ve uyarır."""
    if not _COMPONENTS_OK:
        return None, None, None, False

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    extractor = LexiconExtractor()
    model     = LexiconBERTModel(
        bert_model_name=MODEL_NAME,
        num_labels=NUM_LABELS,
        lexicon_dim=LEXICON_DIM,
    )

    candidates = [
        ROOT / "results" / "exp4" / "best_model" / "master_hybrid_weights.pt",
        ROOT / "results" / "exp4" / "best_model" / "pytorch_model.bin",
        ROOT / "results" / "exp4" / "best_model" / "model.safetensors",
        ROOT / "results" / "exp3" / "best_model" / "lexicon_bert_weights.pt",
        ROOT / "results" / "exp3" / "best_model" / "pytorch_model.bin",
        ROOT / "results" / "exp3" / "best_model" / "model.safetensors",
    ]

    for path in candidates:
        if path.exists():
            try:
                if path.suffix in (".bin", ".pt"):
                    state = torch.load(path, map_location="cpu")
                else:
                    from safetensors.torch import load_file
                    state = load_file(str(path))
                load_lexicon_bert_weights(model, state)
                print(f"[MODEL] Ağırlıklar yüklendi: {path.relative_to(ROOT)}")
                model.to(device).eval()
                return model, tokenizer, extractor, True
            except Exception as e:
                print(f"[UYARI] {path.name}: {e}")

    print(
        "\n" + "!"*65 +
        "\n  DİKKAT: Eğitilmiş model ağırlığı bulunamadı!" +
        "\n  Model rastgele başlatılmış — tahminler rastgeledir." +
        "\n  Bu script çalışır ama sonuçlar anlamsızdır." +
        "\n  Modeli eğitmek için: python run_experiments.py --only 4" +
        "\n" + "!"*65 + "\n"
    )
    model.to(device).eval()
    return model, tokenizer, extractor, False


# ─── Batch Inference ─────────────────────────────────────────────────────────

class SimpleTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, extractor, max_length=128):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.extractor  = extractor
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = str(self.texts[idx])
        label = int(self.labels[idx])
        enc   = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        lex = torch.tensor(self.extractor.extract(text), dtype=torch.float32)
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "lex":            lex,
            "label":          torch.tensor(label, dtype=torch.long),
        }


def run_inference(
    model, tokenizer, extractor, df: pd.DataFrame,
    device: torch.device, batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test seti üzerinde batch inference çalıştırır.
    Döner: (all_preds, all_probs) — shape (N,) ve (N, 10)
    """
    dataset = SimpleTextDataset(
        df["text"].tolist(), df["label_id"].tolist(),
        tokenizer, extractor,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds, all_probs = [], []
    n_batches = len(loader)

    print(f"[INF] {len(df):,} örnek, batch={batch_size}, {n_batches} batch")
    t0 = time.time()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i % max(1, n_batches // 10) == 0:
                pct = i / n_batches * 100
                elapsed = time.time() - t0
                eta     = (elapsed / max(i, 1)) * (n_batches - i)
                print(f"  {pct:5.1f}%  ({i}/{n_batches})  ETA: {eta:.0f}s", end="\r")

            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                lexicon_features=batch["lex"].to(device),
            )
            probs = F.softmax(out.logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_probs.append(probs)
            all_preds.append(preds)

    elapsed = time.time() - t0
    all_preds = np.concatenate(all_preds)
    all_probs = np.vstack(all_probs)

    print(f"  100.0%  ({n_batches}/{n_batches})  Toplam: {elapsed:.1f}s         ")
    return all_preds, all_probs


# ─── Hata Raporu ─────────────────────────────────────────────────────────────

def build_error_report(
    df:       pd.DataFrame,
    preds:    np.ndarray,
    probs:    np.ndarray,
) -> pd.DataFrame:
    """
    Yanlış sınıflandırılmış örnekleri filtreler ve raporlar.
    """
    true_labels = df["label_id"].values
    is_wrong    = preds != true_labels

    error_df = pd.DataFrame({
        "Cümle Metni":          df["text"].values[is_wrong],
        "Gerçek Etiket":        [ID2EMO[int(t)] for t in true_labels[is_wrong]],
        "Tahmin Edilen Etiket": [ID2EMO[int(p)] for p in preds[is_wrong]],
        "Gerçek ID":            true_labels[is_wrong],
        "Tahmin ID":            preds[is_wrong],
        "Olasılık Puanı":       probs[is_wrong].max(axis=1).round(4),
        "Gerçek Sınıf Olasılığı": [
            round(float(probs[is_wrong][i, true_labels[is_wrong][i]]), 4)
            for i in range(is_wrong.sum())
        ],
        "Kaynak":               df["source"].values[is_wrong] if "source" in df.columns else "bilinmiyor",
        "Metin Uzunluğu":       [len(str(t).split()) for t in df["text"].values[is_wrong]],
    })

    error_df.sort_values("Olasılık Puanı", ascending=False, inplace=True)
    error_df.reset_index(drop=True, inplace=True)
    return error_df


# ─── Top-K Karışma Çiftleri ──────────────────────────────────────────────────

def analyze_confusion_pairs(
    error_df:     pd.DataFrame,
    true_labels:  np.ndarray,
    preds:        np.ndarray,
    probs:        np.ndarray,
    top_k:        int = 10,
) -> List[dict]:
    """
    En çok karışan duygu çiftlerini bulur ve her biri için
    istatistiksel özellikler çıkarır.
    """
    # Karışma çifti sayımı (yanlışlar arasında)
    pair_counts: Counter = Counter()
    for t, p in zip(true_labels, preds):
        if t != p:
            pair_counts[(int(t), int(p))] += 1

    top_pairs = pair_counts.most_common(top_k)
    results   = []

    for (tid, pid), count in top_pairs:
        true_emo = ID2EMO[tid]
        pred_emo = ID2EMO[pid]

        # Bu çiftteki örnekleri filtrele
        mask = error_df[
            (error_df["Gerçek ID"] == tid) &
            (error_df["Tahmin ID"] == pid)
        ]

        # Ortak kelimeler
        texts_for_pair = mask["Cümle Metni"].tolist()
        common_words   = _top_words(texts_for_pair, top_n=10)

        # Güvenlik istatistikleri
        conf_scores = mask["Olasılık Puanı"].values
        true_probs  = mask["Gerçek Sınıf Olasılığı"].values
        avg_len     = mask["Metin Uzunluğu"].mean()

        # Örnek sayısı uyarısı
        n_true_total = int((true_labels == tid).sum())
        is_reliable  = count >= MIN_RELIABLE_SAMPLES

        results.append({
            "true_emotion":      true_emo,
            "pred_emotion":      pred_emo,
            "count":             count,
            "n_true_total":      n_true_total,
            "error_rate":        round(count / max(n_true_total, 1), 4),
            "avg_confidence":    round(float(conf_scores.mean()), 4),
            "avg_true_prob":     round(float(true_probs.mean()), 4),
            "avg_text_length":   round(float(avg_len), 1),
            "high_conf_errors":  int((conf_scores > 0.7).sum()),
            "top_common_words":  common_words,
            "is_reliable":       is_reliable,
            "reliability_note":  (
                "Yeterli örnek" if is_reliable
                else f"DİKKAT: {count} örnek < {MIN_RELIABLE_SAMPLES} eşik"
            ),
            "hypothesis":        _generate_hypothesis(true_emo, pred_emo),
        })

    return results


def _top_words(texts: List[str], top_n: int = 10) -> List[str]:
    """Bir metin listesinden en sık geçen anlamlı kelimeleri döner."""
    punct_re = re.compile(r"[^\w\s]", re.UNICODE)
    counter  = Counter()
    for text in texts:
        clean = punct_re.sub(" ", str(text).lower())
        words = [w for w in clean.split() if len(w) >= 3 and w not in TR_STOPWORDS]
        counter.update(words)
    return [w for w, _ in counter.most_common(top_n)]


def _generate_hypothesis(true_emo: str, pred_emo: str) -> str:
    """
    İki duygu arasındaki karışmanın olası sebebini açıklar.
    sözlük (NRC kategorili) örtüşmesine ve dilsel benzerliğe dayanır.
    """
    known_pairs = {
        frozenset(["mutluluk", "sevgi"]):
            "Her ikisi de 'joy' ve 'positive' NRC kanallarını paylaşır; "
            "olumlu bağlamda ifade edilen sevgi cümleleri mutluluk gibi görünür.",
        frozenset(["mutluluk", "gurur"]):
            "Başarı ve kutlama ifadeleri her iki sınıfta da görülür. "
            "Gurur eğitim verisi son derece az (test=8).",
        frozenset(["üzüntü", "korku"]):
            "Her ikisi de negatif valans taşır; 'negative' ve 'sadness'/'fear' "
            "NRC kanalları örtüşür.",
        frozenset(["öfke", "tiksinti"]):
            "Her ikisi de yüksek negatif yoğunluk içerir. "
            "'anger' ve 'disgust' NRC kanalları zaman zaman aynı kelimelerde aktiftir.",
        frozenset(["utanç", "üzüntü"]):
            "Utanç, üzüntü belirteçleriyle sıklıkla birlikte kullanılır. "
            "Test setinde utanç=37 örnek — güvenilir analiz için yetersiz.",
        frozenset(["şaşkınlık", "mutluluk"]):
            "Olumlu sürpriz ifadeleri ('harika', 'inanılmaz') mutluluk sınıfında "
            "çok daha fazla örnekle temsil edilir.",
        frozenset(["nötr", "mutluluk"]):
            "Nötr sınıf eğitim setinde aşırı temsil edilir. "
            "Düşük duygusallıktaki mutluluk cümleleri nötr gibi görünür.",
        frozenset(["korku", "öfke"]):
            "Tehdit içeren durumlarda hem korku hem öfke aktive olabilir; "
            "cümle bağlamı kısa ise ayrıştırmak güçtür.",
    }
    key = frozenset([true_emo, pred_emo])
    return known_pairs.get(key, (
        f"'{true_emo}' → '{pred_emo}' çifti için önceden tanımlanmış hipotez yok. "
        "Ortak kelimeler ve NRC örtüşmesi incelenmeli."
    ))


# ─── Görselleştirme ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    true_labels: np.ndarray,
    preds:       np.ndarray,
    save_path:   Path,
    is_trained:  bool = True,
):
    """Normalize edilmiş karmaşıklık matrisi."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(true_labels, preds, labels=list(range(10)))
    # Satır toplamına göre normalize
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0)

    fig, ax = plt.subplots(figsize=(13, 10), facecolor="white")

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Normalize Sayım (satır toplamına göre)", fontsize=10)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(EMOTIONS, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(EMOTIONS, fontsize=11)
    ax.set_xlabel("Tahmin Edilen Etiket", fontsize=12, fontweight="bold")
    ax.set_ylabel("Gerçek Etiket",        fontsize=12, fontweight="bold")

    title = "Normalize Karmaşıklık Matrisi"
    if not is_trained:
        title += "\n⚠️ UYARI: Eğitilmemiş model — matris rastgele dağılımı yansıtır"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    for i in range(10):
        for j in range(10):
            v     = cm_norm[i, j]
            raw   = cm[i, j]
            color = "white" if v > 0.55 else "#2c3e50"
            text  = f"{v:.2f}\n({raw})" if raw > 0 else ""
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=7.5, color=color)

    # Az örnekli sınıfları işaretle
    low_sample_emos = ["gurur", "utanç", "korku", "tiksinti"]
    for i, emo in enumerate(EMOTIONS):
        if emo in low_sample_emos:
            ax.get_yticklabels()[i].set_color("#e74c3c")
            ax.get_yticklabels()[i].set_fontweight("bold")

    fig.text(0.01, 0.01,
             "Kırmızı etiketler = test setinde az örnek (≤95). Analiz istatistiksel olarak güvenilir değil.",
             fontsize=8, color="#e74c3c")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  [KAYIT] {save_path.relative_to(ROOT)}")
    plt.close(fig)


def plot_top_confusion_pairs(
    pairs:     List[dict],
    save_path: Path,
    is_trained: bool = True,
):
    """En çok karışan duygu çiftlerini yatay bar ile gösterir."""
    n     = len(pairs)
    fig, (ax_main, ax_info) = plt.subplots(
        1, 2, figsize=(14, max(5, n * 0.7 + 2)),
        gridspec_kw={"width_ratios": [3, 2]},
        facecolor="white",
    )

    labels   = [f'{p["true_emotion"]}  →  {p["pred_emotion"]}' for p in pairs]
    counts   = [p["count"]             for p in pairs]
    errors   = [p["error_rate"] * 100  for p in pairs]
    reliable = [p["is_reliable"]       for p in pairs]
    colors   = ["#e74c3c" if r else "#f39c12" for r in reliable]

    y_pos = range(n)
    bars  = ax_main.barh(y_pos, counts, color=colors, edgecolor="white",
                         linewidth=0.7, height=0.72)
    ax_main.set_yticks(y_pos)
    ax_main.set_yticklabels(labels, fontsize=10.5)
    ax_main.set_xlabel("Yanlış Sınıflandırma Sayısı", fontsize=10)
    ax_main.spines[["top","right"]].set_visible(False)

    title = "En Çok Karışan Duygu Çiftleri"
    if not is_trained:
        title += "\n(Eğitilmemiş model — anlamsız)"
    ax_main.set_title(title, fontsize=12, fontweight="bold", pad=10)

    for bar, cnt, err, rel in zip(bars, counts, errors, reliable):
        ax_main.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{cnt:,}  ({err:.1f}%)",
            va="center", ha="left", fontsize=9,
            color="#2c3e50",
        )
        if not rel:
            ax_main.text(
                bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                "Az örnek",
                va="center", ha="center", fontsize=7.5,
                color="white", fontweight="bold",
            )

    rel_patch = mpatches.Patch(color="#e74c3c", label=f"Güvenilir (≥{MIN_RELIABLE_SAMPLES} örnek)")
    low_patch = mpatches.Patch(color="#f39c12", label=f"Az örnekli (<{MIN_RELIABLE_SAMPLES})")
    ax_main.legend(handles=[rel_patch, low_patch], loc="lower right", fontsize=9)

    # Sağ panel: en kötü 3 çiftin hipotezi
    ax_info.axis("off")
    ax_info.set_facecolor("#fafafa")
    y = 0.97
    ax_info.text(0.05, y, "En Problemli Karışmalar — Hipotez",
                 fontsize=11, fontweight="bold", color="#2c3e50",
                 transform=ax_info.transAxes, va="top")
    y -= 0.07

    for p in pairs[:3]:
        hypo = textwrap.fill(p["hypothesis"], width=42)
        color_emo = EMOTION_COLORS.get(p["true_emotion"], "#333")
        ax_info.text(
            0.05, y,
            f'"{p["true_emotion"]}" → "{p["pred_emotion"]}"',
            fontsize=9.5, fontweight="bold", color=color_emo,
            transform=ax_info.transAxes, va="top",
        )
        y -= 0.05
        for line in hypo.split("\n"):
            ax_info.text(0.07, y, line, fontsize=8.2, color="#555",
                         transform=ax_info.transAxes, va="top")
            y -= 0.04
        y -= 0.04

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  [KAYIT] {save_path.relative_to(ROOT)}")
    plt.close(fig)


def plot_per_class_metrics(
    true_labels: np.ndarray,
    preds:       np.ndarray,
    save_path:   Path,
    is_trained:  bool = True,
):
    """Sınıf bazlı F1, Precision, Recall ve örnek sayısı."""
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f, support = precision_recall_fscore_support(
        true_labels, preds, labels=list(range(10)), zero_division=0
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="white")

    x = np.arange(10)
    w = 0.28

    ax = axes[0]
    ax.bar(x - w,   p, w, label="Precision", color="#3498db", alpha=0.85)
    ax.bar(x,       r, w, label="Recall",    color="#e74c3c", alpha=0.85)
    ax.bar(x + w,   f, w, label="F1",        color="#2ecc71", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTIONS, rotation=40, ha="right", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Skor", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_facecolor("#fafafa")
    ax.spines[["top","right"]].set_visible(False)
    title0 = "Sınıf Bazlı Metrikler"
    if not is_trained:
        title0 += " (⚠️ Eğitilmemiş)"
    ax.set_title(title0, fontsize=12, fontweight="bold")

    # Değer etiketleri
    for i, (pi, ri, fi) in enumerate(zip(p, r, f)):
        if fi > 0.01:
            ax.text(i + w, fi + 0.02, f"{fi:.2f}", ha="center", fontsize=7.5, color="#2c3e50")

    ax2 = axes[1]
    support_colors = ["#e74c3c" if s < MIN_RELIABLE_SAMPLES else "#3498db" for s in support]
    bars = ax2.bar(x, support, color=support_colors, alpha=0.8, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(EMOTIONS, rotation=40, ha="right", fontsize=10)
    ax2.set_ylabel("Test Örnek Sayısı", fontsize=11)
    ax2.set_yscale("log")
    ax2.set_facecolor("#fafafa")
    ax2.spines[["top","right"]].set_visible(False)
    ax2.set_title("Test Seti Sınıf Dağılımı\n(Kırmızı = Az Örnek → Güvenilmez)",
                  fontsize=12, fontweight="bold")

    for bar, s in zip(bars, support):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.15,
            str(s), ha="center", fontsize=9, fontweight="bold",
            color="#e74c3c" if s < MIN_RELIABLE_SAMPLES else "#2c3e50",
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  [KAYIT] {save_path.relative_to(ROOT)}")
    plt.close(fig)


def plot_confidence_histogram(
    true_labels: np.ndarray,
    preds:       np.ndarray,
    probs:       np.ndarray,
    save_path:   Path,
    is_trained:  bool = True,
):
    """Doğru ve yanlış tahminlerin güven dağılımı."""
    correct = preds == true_labels
    conf    = probs.max(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")

    ax = axes[0]
    ax.hist(conf[correct],  bins=40, color="#2ecc71", alpha=0.7, label="Doğru",  density=True)
    ax.hist(conf[~correct], bins=40, color="#e74c3c", alpha=0.7, label="Yanlış", density=True)
    ax.set_xlabel("Model Güven Skoru (max softmax prob)", fontsize=11)
    ax.set_ylabel("Yoğunluk", fontsize=11)
    ax.legend(fontsize=11)
    ax.set_facecolor("#fafafa")
    ax.spines[["top","right"]].set_visible(False)
    title0 = "Güven Dağılımı: Doğru vs Yanlış"
    if not is_trained:
        title0 += "\n(⚠️ Eğitilmemiş — anlamsız)"
    ax.set_title(title0, fontsize=12, fontweight="bold")

    ax.axvline(0.5, color="#7f8c8d", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(0.7, color="#e67e22", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(0.51, ax.get_ylim()[1] * 0.95, "0.5", color="#7f8c8d", fontsize=8)
    ax.text(0.71, ax.get_ylim()[1] * 0.95, "0.7", color="#e67e22", fontsize=8)

    # Yüksek güvenlikli hata analizi
    high_conf_wrong = (~correct) & (conf > 0.7)
    n_hcw = high_conf_wrong.sum()
    n_wrong = (~correct).sum()
    n_total = len(preds)

    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_facecolor("#fafafa")

    acc   = correct.mean() * 100
    stats = [
        ("Toplam Örnek",          f"{n_total:,}"),
        ("Doğru Tahmin",          f"{correct.sum():,}  ({acc:.1f}%)"),
        ("Yanlış Tahmin",         f"{n_wrong:,}  ({100-acc:.1f}%)"),
        ("Ortalama Güven (Doğru)",  f"{conf[correct].mean():.3f}"),
        ("Ortalama Güven (Yanlış)", f"{conf[~correct].mean():.3f}"),
        ("Yüksek Güvenli Hatalar", f"{n_hcw:,}  ({n_hcw/max(n_wrong,1)*100:.1f}% of errors)"),
        ("",                       ""),
        ("⚠️ Yüksek Güvenli Hata", "Modelin en tehlikeli hatası:"),
        ("  >70% emin ama yanlış", f"{n_hcw:,} örnek"),
    ]
    y = 0.95
    for label, val in stats:
        if label == "":
            y -= 0.04
            continue
        ax2.text(0.05, y, f"{label}:", fontsize=10, color="#555",
                 transform=ax2.transAxes, va="top")
        ax2.text(0.6,  y, val, fontsize=10, fontweight="bold", color="#2c3e50",
                 transform=ax2.transAxes, va="top")
        y -= 0.09

    ax2.set_title("Özet İstatistikler", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  [KAYIT] {save_path.relative_to(ROOT)}")
    plt.close(fig)


def plot_pair_analysis(
    pairs:     List[dict],
    error_df:  pd.DataFrame,
    save_path: Path,
    top_n:     int = 3,
):
    """En kötü N çifti için derinlemesine analiz paneli."""
    n_panels = min(top_n, len(pairs))
    if n_panels == 0:
        return

    fig, axes = plt.subplots(
        n_panels, 2, figsize=(14, n_panels * 4.5 + 1),
        facecolor="white",
    )
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        "Derinlemesine Çift Analizi — En Problemli 3 Karışma",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for i, pair in enumerate(pairs[:n_panels]):
        ax_words, ax_stats = axes[i]
        te = pair["true_emotion"]
        pe = pair["pred_emotion"]
        tc = EMOTION_COLORS.get(te, "#333")
        pc = EMOTION_COLORS.get(pe, "#666")

        # Sol: en yaygın kelimeler
        words = pair["top_common_words"][:8]
        if words:
            ax_words.barh(
                range(len(words)), [1.0] * len(words),
                color=[tc] * len(words), alpha=0.75,
                edgecolor="white", height=0.65,
            )
            ax_words.set_yticks(range(len(words)))
            ax_words.set_yticklabels(words, fontsize=11)
            ax_words.set_xticks([])
            ax_words.set_title(
                f'"{te}" → "{pe}"  ({pair["count"]} hata)\n'
                f'Ortak Kelimeler',
                fontsize=11, fontweight="bold",
                color="#2c3e50",
            )
        else:
            ax_words.text(0.5, 0.5, "Ortak kelime yok",
                          ha="center", va="center", transform=ax_words.transAxes)
        ax_words.spines[:].set_visible(False)
        ax_words.set_facecolor("#fafafa")

        # Sağ: istatistikler
        ax_stats.axis("off")
        ax_stats.set_facecolor("#fafafa")

        rows = [
            ("Gerçek Duygu",          te,  tc),
            ("Tahmin Edilen",         pe,  pc),
            ("Hata Sayısı",           f'{pair["count"]:,}', "#2c3e50"),
            ("Hata Oranı",            f'{pair["error_rate"]*100:.1f}%', "#e74c3c"),
            ("Ort. Tahmin Güveni",    f'{pair["avg_confidence"]:.3f}', "#2c3e50"),
            ("Ort. Gerçek Sınıf Olasılığı", f'{pair["avg_true_prob"]:.3f}', "#2c3e50"),
            ("Yüksek Güvenli Hata",   f'{pair["high_conf_errors"]}', "#e74c3c" if pair["high_conf_errors"] > 0 else "#2c3e50"),
            ("Ortalama Cümle Uzunluğu", f'{pair["avg_text_length"]:.0f} kelime', "#2c3e50"),
            ("Güvenilirlik",          pair["reliability_note"], "#e74c3c" if not pair["is_reliable"] else "#27ae60"),
        ]
        y_pos = 0.92
        for label, val, color in rows:
            ax_stats.text(0.04, y_pos, f"{label}:", fontsize=9.5,
                          color="#777", transform=ax_stats.transAxes, va="top")
            ax_stats.text(0.50, y_pos, val, fontsize=9.5, fontweight="bold",
                          color=color, transform=ax_stats.transAxes, va="top")
            y_pos -= 0.10

        # Hipotez
        hypo = textwrap.fill(pair["hypothesis"], width=38)
        ax_stats.text(0.04, y_pos - 0.02, "Hipotez:", fontsize=9,
                      color="#7f8c8d", transform=ax_stats.transAxes, va="top",
                      fontstyle="italic")
        y_pos -= 0.10
        for line in hypo.split("\n"):
            ax_stats.text(0.04, y_pos, line, fontsize=8.5, color="#555",
                          transform=ax_stats.transAxes, va="top")
            y_pos -= 0.075

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  [KAYIT] {save_path.relative_to(ROOT)}")
    plt.close(fig)


# ─── Metin Raporu ────────────────────────────────────────────────────────────

def generate_text_report(
    df:          pd.DataFrame,
    error_df:    pd.DataFrame,
    true_labels: np.ndarray,
    preds:       np.ndarray,
    probs:       np.ndarray,
    pairs:       List[dict],
    is_trained:  bool,
) -> str:
    """Terminal'e ve dosyaya yazdırılan kapsamlı metin raporu."""
    from sklearn.metrics import classification_report, accuracy_score, f1_score

    acc      = accuracy_score(true_labels, preds)
    f1_macro = f1_score(true_labels, preds, average="macro", zero_division=0)
    f1_wt    = f1_score(true_labels, preds, average="weighted", zero_division=0)
    n_wrong  = (preds != true_labels).sum()
    conf     = probs.max(axis=1)
    hcw      = ((preds != true_labels) & (conf > 0.7)).sum()

    report = []
    sep    = "=" * 70

    report.append(sep)
    report.append("  TÜRKÇE DUYGU ANALİZİ — HATA ANALİZİ RAPORU")
    report.append(sep)

    if not is_trained:
        report.append("")
        report.append("  ⚠️⚠️⚠️  UYARI: MODELİN EĞİTİLMİŞ AĞIRLIKLARI YOK  ⚠️⚠️⚠️")
        report.append("  Aşağıdaki tüm metrikler ANLAMLIDIR çünkü pipeline çalışıyor,")
        report.append("  ancak SAYISAL DEĞERLER anlamsızdır (rastgele tahmin).")
        report.append("  Modeli eğitmek için: python run_experiments.py --only 4")
        report.append("")

    report.append(f"\n  Toplam Test Örneği   : {len(df):,}")
    report.append(f"  Doğru Tahmin         : {(preds==true_labels).sum():,}  ({acc*100:.2f}%)")
    report.append(f"  Yanlış Tahmin        : {n_wrong:,}  ({(1-acc)*100:.2f}%)")
    report.append(f"  F1-Macro             : {f1_macro:.4f}")
    report.append(f"  F1-Weighted          : {f1_wt:.4f}")
    report.append(f"  Yüksek Güvenli Hata  : {hcw:,}  (>70% emin ama yanlış)")

    # Sınıf dağılım uyarısı
    report.append(f"\n{'-'*70}")
    report.append("  VERİ ASIMETRIS UYARISI (test setinde az örnek):")
    report.append(f"{'-'*70}")
    vc = pd.Series(true_labels).value_counts().sort_index()
    for lid in range(10):
        n   = int(vc.get(lid, 0))
        emo = ID2EMO[lid]
        flag = " ← GÜVENİLMEZ SONUÇLAR" if n < MIN_RELIABLE_SAMPLES else ""
        report.append(f"  {emo:<12}  {n:>6,} örnek{flag}")

    # Sınıf bazlı metrikler
    report.append(f"\n{'-'*70}")
    report.append("  SINIF BAZLI METRİKLER:")
    report.append(f"{'-'*70}")
    cr = classification_report(
        true_labels, preds,
        labels=list(range(10)), target_names=EMOTIONS,
        zero_division=0,
    )
    report.append(cr)

    # Top-K karışma çiftleri
    report.append(f"{'-'*70}")
    report.append(f"  TOP-{len(pairs)} KARIŞMA ÇİFTLERİ:")
    report.append(f"{'-'*70}")
    for rank, p in enumerate(pairs, 1):
        report.append(
            f"\n  {rank}. {p['true_emotion']:<12} → {p['pred_emotion']:<12}"
            f"  ({p['count']:,} hata,  {p['error_rate']*100:.1f}% hata oranı)"
        )
        report.append(f"     Ort. Güven       : {p['avg_confidence']:.3f}")
        report.append(f"     Yük. Güv. Hata   : {p['high_conf_errors']:,}")
        report.append(f"     Ort. Metin Uzun. : {p['avg_text_length']:.0f} kelime")
        report.append(f"     Güvenilirlik     : {p['reliability_note']}")
        report.append(f"     Ortak Kelimeler  : {', '.join(p['top_common_words'][:6])}")
        hypo = textwrap.fill(p['hypothesis'], width=60, initial_indent="     Hipotez: ", subsequent_indent="              ")
        report.append(hypo)

    # Yüksek güvenlikli hatalar
    hce = error_df[error_df["Olasılık Puanı"] > 0.7]
    report.append(f"\n{'-'*70}")
    report.append(f"  EN TEHLİKELİ HATALAR (>70% güven, {len(hce):,} örnek):")
    report.append(f"{'-'*70}")
    for _, row in hce.head(5).iterrows():
        report.append(
            f"\n  [{row['Gerçek Etiket']} → {row['Tahmin Edilen Etiket']}]"
            f"  güven={row['Olasılık Puanı']:.3f}"
        )
        report.append(f"  \"{textwrap.shorten(str(row['Cümle Metni']), 90)}\"")

    report.append(f"\n{sep}")
    report.append("  ÇIKTI DOSYALARI:")
    report.append(f"  {OUTDIR}")
    report.append(sep)

    return "\n".join(report)


# ─── Ana Fonksiyon ────────────────────────────────────────────────────────────

def run_analysis(
    max_samples:  Optional[int] = None,
    batch_size:   int = 64,
    no_model:     bool = False,
    top_k_pairs:  int = 10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BAŞLAT] Cihaz: {device}")

    # ── 1. Veri ───────────────────────────────────────────────────────────────
    df = load_test_data(max_samples)

    # ── 2. Model ──────────────────────────────────────────────────────────────
    if no_model:
        print("[MOD] Sadece veri analizi (--no-model)")
        _run_data_only_analysis(df)
        return

    model, tokenizer, extractor, is_trained = load_model(device)

    if model is None:
        print("[HATA] Model bileşenleri yüklenemedi.")
        sys.exit(1)

    # ── 3. Inference ──────────────────────────────────────────────────────────
    print(f"\n[INF] Çıkarım başlatılıyor...")
    preds, probs = run_inference(model, tokenizer, extractor, df, device, batch_size)
    true_labels  = df["label_id"].values

    # ── 4. Hata Raporu CSV ────────────────────────────────────────────────────
    print("\n[CSV] Hata raporu oluşturuluyor...")
    error_df = build_error_report(df, preds, probs)

    csv_path = OUTDIR / "error_report.csv"
    error_df[[
        "Cümle Metni", "Gerçek Etiket",
        "Tahmin Edilen Etiket", "Olasılık Puanı",
        "Gerçek Sınıf Olasılığı", "Metin Uzunluğu", "Kaynak",
    ]].to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  [KAYIT] {csv_path.relative_to(ROOT)}  ({len(error_df):,} yanlış örnek)")

    # Yüksek güvenli hatalar ayrı dosyaya
    hce = error_df[error_df["Olasılık Puanı"] > 0.7]
    hce_path = OUTDIR / "error_report_confident.csv"
    hce[[
        "Cümle Metni", "Gerçek Etiket",
        "Tahmin Edilen Etiket", "Olasılık Puanı", "Kaynak",
    ]].to_csv(hce_path, index=False, encoding="utf-8-sig")
    print(f"  [KAYIT] {hce_path.relative_to(ROOT)}  ({len(hce):,} yüksek güvenli hata)")

    # ── 5. Karışma Çifti Analizi ──────────────────────────────────────────────
    print("\n[ANALİZ] Top-K karışma çiftleri hesaplanıyor...")
    pairs = analyze_confusion_pairs(error_df, true_labels, preds, probs, top_k=top_k_pairs)

    # ── 6. Görselleştirmeler ──────────────────────────────────────────────────
    print("\n[GÖRSEL] Grafikler üretiliyor...")

    plot_confusion_matrix(
        true_labels, preds,
        OUTDIR / "confusion_heatmap.png",
        is_trained,
    )
    plot_top_confusion_pairs(
        pairs,
        OUTDIR / "top_confusion_pairs.png",
        is_trained,
    )
    plot_per_class_metrics(
        true_labels, preds,
        OUTDIR / "per_class_metrics.png",
        is_trained,
    )
    plot_confidence_histogram(
        true_labels, preds, probs,
        OUTDIR / "confidence_histogram.png",
        is_trained,
    )
    plot_pair_analysis(
        pairs, error_df,
        OUTDIR / "pair_analysis_top3.png",
        top_n=3,
    )

    # ── 7. JSON Raporu ────────────────────────────────────────────────────────
    from sklearn.metrics import accuracy_score, f1_score

    json_report = {
        "is_trained_model":  is_trained,
        "test_samples":      len(df),
        "n_errors":          int((preds != true_labels).sum()),
        "accuracy":          round(float(accuracy_score(true_labels, preds)), 4),
        "f1_macro":          round(float(f1_score(true_labels, preds, average="macro", zero_division=0)), 4),
        "f1_weighted":       round(float(f1_score(true_labels, preds, average="weighted", zero_division=0)), 4),
        "high_conf_errors":  int(((preds != true_labels) & (probs.max(axis=1) > 0.7)).sum()),
        "class_sample_counts": {
            ID2EMO[i]: int((true_labels == i).sum())
            for i in range(10)
        },
        "unreliable_classes": [
            ID2EMO[i]
            for i in range(10)
            if (true_labels == i).sum() < MIN_RELIABLE_SAMPLES
        ],
        "top_confusion_pairs": [
            {k: v for k, v in p.items() if k != "top_common_words"}
            for p in pairs
        ],
        "top_confusion_pairs_with_words": pairs,
    }

    json_path = OUTDIR / "error_analysis_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2)
    print(f"  [KAYIT] {json_path.relative_to(ROOT)}")

    # ── 8. Metin Raporu ───────────────────────────────────────────────────────
    print("")
    text_report = generate_text_report(
        df, error_df, true_labels, preds, probs, pairs, is_trained
    )
    print(text_report)

    report_path = OUTDIR / "error_analysis_summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text_report)
    print(f"\n  [KAYIT] {report_path.relative_to(ROOT)}")

    print(f"\n[TAMAM] Tüm dosyalar: {OUTDIR}")
    return json_report


def _run_data_only_analysis(df: pd.DataFrame):
    """Model olmadan sadece veri istatistiklerini analiz eder."""
    print("\n[VERİ ANALİZİ] Model gerektirmeden sınıf dağılımı analizi")

    vc     = df["label_id"].value_counts().sort_index()
    total  = len(df)
    report = ["", "=" * 60, "  VERİ ASIMETRIS RAPORU (model olmadan)", "=" * 60, ""]

    for lid in range(10):
        n   = int(vc.get(lid, 0))
        emo = ID2EMO[lid]
        pct = n / total * 100
        bar = "█" * int(pct / 2)
        flag = "  ← GÜVENİLMEZ" if n < MIN_RELIABLE_SAMPLES else ""
        report.append(f"  {emo:<12}  {n:>7,}  ({pct:5.1f}%)  {bar}{flag}")

    report.append("")
    report.append(
        "  ⚠️ Bu asimetri, azınlık sınıflarındaki hata analizini\n"
        "  istatistiksel olarak geçersiz kılar. Gurur (n=8) ve\n"
        "  utanç (n=37) için tek bir model hatası dahi %12-3\n"
        "  hata oranı değişimine yol açabilir."
    )
    print("\n".join(report))


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Error Analysis — Türkçe Duygu Analizi Hata Raporu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Ornekler:
          python error_analysis.py                      # tam analiz
          python error_analysis.py --max-samples 5000   # hizli on analiz
          python error_analysis.py --no-model            # sadece veri analizi
          python error_analysis.py --batch-size 128      # GPU bellek ayari
          python error_analysis.py --top-k 15            # top-15 cift
        """),
    )
    parser.add_argument("--max-samples",  type=int,  default=None,
                        help="Maksimum test örneği (None=tümü)")
    parser.add_argument("--batch-size",   type=int,  default=64,
                        help="Inference batch büyüklüğü")
    parser.add_argument("--no-model",     action="store_true",
                        help="Model yüklemeden sadece veri analizi yap")
    parser.add_argument("--top-k",        type=int,  default=10,
                        help="Raporlanacak karışma çifti sayısı")

    args = parser.parse_args()

    run_analysis(
        max_samples  = args.max_samples,
        batch_size   = args.batch_size,
        no_model     = args.no_model,
        top_k_pairs  = args.top_k,
    )
