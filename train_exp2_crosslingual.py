# Commit 119: feat(training): save best model checkpoint for exp2 crosslingual run
"""
Experiment 2 — BERTurk + Cross-Lingual Enhanced
=================================================
Temel Hipotez:
  Azınlık sınıflar (Gurur=55, Utanç=246, Tiksinti=580, Korku=633 örnek)
  sadece GoEmotions TR (cross-lingual) kaynağından geliyor.
  Bu kaynağı stratejik oversample ile güçlendirmek minority F1'ı artırır.

ÖNEMLİ: "Cross-lingual" adına rağmen burada diller arası transfer (çok dilli
model, İngilizce veriden öğrenme vb.) yoktur. Yapılan işlem, GoEmotions TR
(İngilizceden makine çevirisi) kaynaklı azınlık örneklerinin rastgele
kopyalanarak çoğaltılmasıdır (random oversampling).

Exp1'den Farklar:
  1. CrossLingualEnhancer  — minority sınıflar hedef sayıya çoğaltılır
  2. Differential weighting — cross-lingual örneklere ek ağırlık katsayısı
  3. Comparison report     — Exp1 baseline ile F1-delta karşılaştırması
  4. f1_comparison.png     — Exp1 vs Exp2 yan yana sınıf bazlı bar grafik

Donanım : RTX 5070 Ti  ->  fp16=True, batch=64, grad_accum=2

Çıktılar (results/exp2/):
  confusion_matrix.png   — normalize ısı haritası (gurur/utanç TP vurgusu)
  training_progress.png  — loss + f1 eğrileri
  f1_comparison.png      — Exp1 vs Exp2 sınıf bazlı F1 karşılaştırması
  tp_minority_report.png — Gurur & Utanç True-Positive analizi
  metrics_report.json    — per-class metrikler + Exp1 delta karşılaştırması
  best_model/
"""

import os
import sys
import json
import warnings
import logging

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, WeightedRandomSampler

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.model_selection import train_test_split
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

ROOT       = Path(__file__).resolve().parent
RESULTS    = ROOT / "results" / "exp2"
EXP1_JSON  = ROOT / "results" / "exp1" / "metrics_report.json"
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

# Cross-lingual kaynak adı
CROSSLINGUAL_SOURCE = "goemotions_tr"

# Sınıf başına hedef oversample sayısı (None = orijinal)
# Sadece cross-lingual kaynaklı azınlık sınıflara uygulanır
OVERSAMPLE_TARGETS: Dict[str, int] = {
    "gurur":     3_500,   # 55   -> 3500  (~63x)
    "utanç":     4_000,   # 246  -> 4000  (~16x)
    "tiksinti":  4_000,   # 580  -> 4000  (~7x)
    "korku":     4_000,   # 633  -> 4000  (~6x)
    "şaşkınlık": 6_000,   # 4403 -> 6000  (~1.4x)
    "öfke":      6_000,   # 5346 -> 6000  (~1.1x)
}

# Cross-lingual örneklere ek ağırlık katsayısı (WeightedRandomSampler)
CL_WEIGHT_MULTIPLIER = 2.5

# ─── Veri Yükleme ───────────────────────────────────────────────────────────

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
        raise FileNotFoundError("Veri bulunamadı. `python data_engine.py --rebuild` çalıştırın.")

    df = df.dropna(subset=["text", "label_id", "source"])
    df["label_id"] = df["label_id"].astype(int)
    df = df[df["label_id"].between(0, NUM_LABELS - 1)]
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 5].reset_index(drop=True)
    return df


def make_splits(df: pd.DataFrame, test_size=0.15, val_size=0.15):
    splits_dir = ROOT / "data" / "splits" / "master_splits"

    # Mevcut split'leri yeniden kullan (Exp1 ile karşılaştırılabilirlik)
    train_p = splits_dir / "train.parquet"
    val_p   = splits_dir / "val.parquet"
    test_p  = splits_dir / "test.parquet"

    if train_p.exists() and val_p.exists() and test_p.exists():
        print("[SPLIT] Mevcut Exp1 split'leri kullanılıyor (karşılaştırılabilirlik için)")
        df_train = pd.read_parquet(train_p)
        df_val   = pd.read_parquet(val_p)
        df_test  = pd.read_parquet(test_p)
        # label sütunu yoksa ekle
        for d in [df_train, df_val, df_test]:
            if "label" not in d.columns:
                d["label"] = d["label_id"].map(ID2LABEL)
            if "source" not in d.columns:
                d["source"] = "unknown"
    else:
        splits_dir.mkdir(parents=True, exist_ok=True)
        df_tv, df_test = train_test_split(
            df, test_size=test_size, stratify=df["label_id"], random_state=SEED
        )
        val_r = val_size / (1.0 - test_size)
        df_train, df_val = train_test_split(
            df_tv, test_size=val_r, stratify=df_tv["label_id"], random_state=SEED
        )
        for name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
            subset.reset_index(drop=True).to_parquet(splits_dir / f"{name}.parquet", index=False)

    print(f"[SPLIT] Train:{len(df_train):,}  Val:{len(df_val):,}  Test:{len(df_test):,}")
    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


# ─── Cross-Lingual Enhancer ─────────────────────────────────────────────────

class CrossLingualEnhancer:
    """
    Azınlık sınıf örneklerini (tümü GoEmotions TR = cross-lingual kaynaklı)
    hedef sayıya ulaşana kadar rastgele tekrarlayarak çoğaltır.

    Oversample sonrası train setinin kaynak özeti ve sınıf dağılımı loglanır.
    """

    def __init__(
        self,
        targets: Dict[str, int] = OVERSAMPLE_TARGETS,
        cl_source: str = CROSSLINGUAL_SOURCE,
        seed: int = SEED,
    ):
        self.targets   = targets
        self.cl_source = cl_source
        self.rng       = np.random.default_rng(seed)

    def enhance(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """df_train'e oversample edilmiş cross-lingual örnekleri ekler."""
        print(f"\n[CL-ENHANCE] Cross-Lingual oversample başlıyor...")
        print(f"  Kaynak: '{self.cl_source}'")

        before_total = len(df_train)
        extra_frames = []

        for emo, target in self.targets.items():
            label_id = LABEL2ID[emo]

            # Bu sınıfın cross-lingual örnekleri
            cl_mask = (
                (df_train["label_id"] == label_id) &
                (df_train["source"] == self.cl_source)
            )
            cl_examples = df_train[cl_mask]

            # Bu sınıfın mevcut toplam sayısı
            current_total = (df_train["label_id"] == label_id).sum()
            current_cl    = len(cl_examples)

            if current_total >= target:
                print(f"  {emo:12s}: zaten {current_total:,} ≥ hedef {target:,}, atlandı")
                continue
            if current_cl == 0:
                print(f"  {emo:12s}: cross-lingual örnek yok, atlandı")
                continue

            needed = target - current_total
            # Yerine koyarak örnekle
            idx = self.rng.integers(0, current_cl, size=needed)
            extra = cl_examples.iloc[idx].copy()
            extra["source"] = f"{self.cl_source}_aug"  # augmente edildi etiketi
            extra_frames.append(extra)

            print(
                f"  {emo:12s}: {current_total:,} -> {current_total + needed:,} "
                f"(+{needed:,} cross-lingual kopyasi, orijinal={current_cl})"
            )

        if extra_frames:
            df_aug = pd.concat([df_train] + extra_frames, ignore_index=True)
            df_aug = df_aug.sample(frac=1, random_state=SEED).reset_index(drop=True)
        else:
            df_aug = df_train.copy()

        after_total = len(df_aug)
        print(f"\n  Toplam: {before_total:,} -> {after_total:,} (+{after_total - before_total:,} örnek)\n")
        return df_aug

    def log_distribution(self, df: pd.DataFrame, title: str = ""):
        if title:
            print(f"\n[DAĞILIM] {title}")
        cnt = Counter(df["label_id"].tolist())
        total = len(df)
        src_cl   = (df["source"].str.startswith(self.cl_source)).sum()
        src_orig = total - src_cl
        for i, emo in enumerate(EMOTIONS):
            c = cnt.get(i, 0)
            bar = "█" * min(int(c / max(total, 1) * 25), 25)
            print(f"  {emo:12s}: {c:7,d} ({c/max(total,1)*100:5.1f}%)  {bar}")
        print(f"  {'TOPLAM':12s}: {total:7,d}")
        print(f"  Cross-lingual: {src_cl:,}  |  Diğer: {src_orig:,}")


# ─── Ağırlık Hesabı ─────────────────────────────────────────────────────────

def compute_class_weights(label_ids: np.ndarray) -> torch.Tensor:
    counts = np.bincount(label_ids, minlength=NUM_LABELS).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    weights = len(label_ids) / (counts * NUM_LABELS)
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(
    df: pd.DataFrame,
    cl_source: str = CROSSLINGUAL_SOURCE,
    cl_multiplier: Optional[float] = None,
) -> np.ndarray:
    """
    WeightedRandomSampler ağırlıkları:
      base = N_total / (N_class * num_classes)
      cross-lingual minority örneklere ek cl_multiplier katsayısı uygulanır.

    cl_multiplier verilmezse çağrı anındaki CL_WEIGHT_MULTIPLIER kullanılır
    (varsayılan argüman tanım anında bağlandığı için --cl-multiplier etkisiz kalıyordu).
    """
    if cl_multiplier is None:
        cl_multiplier = CL_WEIGHT_MULTIPLIER
    label_ids = df["label_id"].values
    counts = np.bincount(label_ids, minlength=NUM_LABELS).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    class_w = len(label_ids) / (counts * NUM_LABELS)
    sample_w = class_w[label_ids].astype(float)

    # Cross-lingual örneklere ek boost
    is_cl = df["source"].str.startswith(cl_source).values
    # Sadece azınlık sınıflara (gurur, utanç, tiksinti, korku) ek boost
    minority_ids = {LABEL2ID[e] for e in ["gurur", "utanç", "tiksinti", "korku"]}
    is_minority = np.isin(label_ids, list(minority_ids))
    boost_mask = is_cl & is_minority
    sample_w[boost_mask] *= cl_multiplier

    print(f"[AĞIRLIK] CL-boost ({cl_multiplier}x) uygulanan örnek: {boost_mask.sum():,}")
    return sample_w


# ─── PyTorch Dataset ────────────────────────────────────────────────────────

class EmotionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.texts      = df["text"].tolist()
        self.labels     = df["label_id"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

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
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ─── Weighted Trainer ───────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
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


# ─── Epoch Logger ───────────────────────────────────────────────────────────

# ─── Metrik Fonksiyonu ───────────────────────────────────────────────────────

def build_compute_metrics():
    acc_m = evaluate.load("accuracy")
    f1_m  = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
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
        (axes[0], cm,      "Ham Sayılar",       "d"),
        (axes[1], cm_norm, "Normalize (Oransal)", ".2f"),
    ]:
        sns.heatmap(
            data, annot=True, fmt=fmt,
            xticklabels=labels_short, yticklabels=labels_short,
            cmap="Blues", linewidths=0.4, ax=ax,
            annot_kws={"size": 8},
        )
        ax.set_xlabel("Tahmin Edilen", fontsize=11)
        ax.set_ylabel("Gerçek", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

    # Gurur ve Utanç satırlarını vurgula
    for idx in [LABEL2ID["utanç"], LABEL2ID["gurur"]]:
        for ax in axes:
            ax.add_patch(plt.Rectangle(
                (0, idx), NUM_LABELS, 1,
                fill=False, edgecolor="#e74c3c", lw=2.5, zorder=5,
            ))

    fig.suptitle("Experiment 2 — BERTurk + Cross-Lingual Enhanced | Confusion Matrix\n"
                 "(Kırmızı çerçeve: Utanç & Gurur sınıfları)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Confusion matrix: {save_path}")


def save_training_curves(history: dict, save_path: Path):
    epochs = history["epoch"]
    if not epochs:
        print("[VIZ] Epoch verisi yok, grafik atlandı.")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history["train_loss"], "o-", color="#e74c3c", label="Train Loss", lw=2)
    ax1.plot(epochs, history["val_loss"],   "s-", color="#3498db", label="Val Loss",   lw=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss Eğrileri", fontweight="bold")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, history["val_f1_macro"], "o-", color="#2ecc71", label="Val F1-Macro", lw=2)
    ax2.plot(epochs, history["val_acc"],       "s-", color="#9b59b6", label="Val Accuracy", lw=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Skor")
    ax2.set_title("Validation Metrikleri", fontweight="bold")
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 1)
    fig.suptitle("Experiment 2 — BERTurk + Cross-Lingual | Eğitim İlerlemesi", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Training curves: {save_path}")


def save_f1_comparison(exp2_per_class: dict, save_path: Path):
    """Exp1 vs Exp2 sınıf bazlı F1 bar grafiği."""
    exp1_f1 = {}
    if EXP1_JSON.exists():
        with open(EXP1_JSON, encoding="utf-8") as f:
            exp1_data = json.load(f)
        exp1_f1 = {e: exp1_data["per_class"][e]["f1"] for e in EMOTIONS if e in exp1_data.get("per_class", {})}
    else:
        print(f"[VIZ] Exp1 JSON bulunamadı ({EXP1_JSON}), karşılaştırma grafiği tek sütunlu olacak")

    x      = np.arange(NUM_LABELS)
    width  = 0.38
    colors_exp1 = ["#95a5a6"] * NUM_LABELS
    colors_exp2 = []
    minority_ids = {LABEL2ID[e] for e in ["gurur", "utanç", "tiksinti", "korku"]}
    for i in range(NUM_LABELS):
        colors_exp2.append("#e74c3c" if i in minority_ids else "#3498db")

    fig, ax = plt.subplots(figsize=(15, 6))

    if exp1_f1:
        vals1 = [exp1_f1.get(e, 0.0) for e in EMOTIONS]
        bars1 = ax.bar(x - width/2, vals1, width, label="Exp1 — Baseline", color=colors_exp1, alpha=0.85, edgecolor="white")

    vals2 = [exp2_per_class.get(e, {}).get("f1", 0.0) for e in EMOTIONS]
    bars2 = ax.bar(
        x + width/2 if exp1_f1 else x,
        vals2, width,
        label="Exp2 — Cross-Lingual",
        color=colors_exp2, alpha=0.9, edgecolor="white",
    )

    # Delta etiketleri
    if exp1_f1:
        for i, (v1, v2) in enumerate(zip(vals1, vals2)):
            delta = v2 - v1
            color = "#27ae60" if delta >= 0 else "#c0392b"
            sign  = "+" if delta >= 0 else ""
            ax.annotate(
                f"{sign}{delta:.2f}",
                xy=(x[i] + width/2, v2 + 0.01),
                ha="center", va="bottom", fontsize=7.5,
                color=color, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([e[:9] for e in EMOTIONS], rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 Skoru", fontsize=11)
    ax.set_title("Experiment 1 vs Experiment 2 — Sınıf Bazlı F1 Karşılaştırması\n"
                 "(Kırmızı: Azınlık sınıflar, Yeşil delta = iyileşme)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    red_patch  = mpatches.Patch(color="#e74c3c", label="Exp2 — Azınlık (Cross-Lingual)")
    blue_patch = mpatches.Patch(color="#3498db", label="Exp2 — Çoğunluk")
    grey_patch = mpatches.Patch(color="#95a5a6", label="Exp1 — Baseline")
    ax.legend(handles=[grey_patch, blue_patch, red_patch], loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] F1 karşılaştırma: {save_path}")


def save_tp_minority_report(y_true, y_pred, save_path: Path):
    """Gurur & Utanç için True-Positive oranı ve karışım analizi."""
    focus_emotions = ["utanç", "gurur", "tiksinti", "korku"]
    focus_ids      = [LABEL2ID[e] for e in focus_emotions]
    cm             = confusion_matrix(y_true, y_pred)
    cm_norm        = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, len(focus_emotions), figsize=(5 * len(focus_emotions), 5))

    for ax, emo, eid in zip(axes, focus_emotions, focus_ids):
        # Bu sınıfın tahmin dağılımı (gerçekte bu sınıf olan örnekler)
        true_mask = (np.array(y_true) == eid)
        total_true = true_mask.sum()
        if total_true == 0:
            ax.text(0.5, 0.5, f"{emo}\n(test seti yok)", ha="center", va="center", fontsize=12)
            ax.axis("off")
            continue

        pred_for_true = np.array(y_pred)[true_mask]
        pred_counts   = Counter(pred_for_true)

        labels_sorted = sorted(pred_counts.keys())
        values        = [pred_counts[l] / total_true for l in labels_sorted]
        bar_colors    = ["#2ecc71" if l == eid else "#e74c3c" for l in labels_sorted]
        bar_labels    = [ID2LABEL[l][:7] for l in labels_sorted]

        bars = ax.bar(range(len(labels_sorted)), values, color=bar_colors, edgecolor="white")
        ax.set_xticks(range(len(labels_sorted)))
        ax.set_xticklabels(bar_labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Oran", fontsize=9)
        ax.set_title(
            f"'{emo}' Tahmin Dağılımı\n"
            f"Gerçek: {total_true} | TP={pred_counts.get(eid, 0)/total_true:.2%}",
            fontsize=9, fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Azınlık Sınıf True-Positive Analizi — Exp2 Cross-Lingual",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] TP minority analiz: {save_path}")


def save_metrics_report(y_true, y_pred, history: dict, save_path: Path) -> dict:
    report_dict = classification_report(
        y_true, y_pred, target_names=EMOTIONS, output_dict=True, zero_division=0,
    )

    per_class = {}
    for emo in EMOTIONS:
        if emo in report_dict:
            per_class[emo] = {
                "precision": round(report_dict[emo]["precision"], 4),
                "recall":    round(report_dict[emo]["recall"],    4),
                "f1":        round(report_dict[emo]["f1-score"],  4),
                "support":   int(report_dict[emo]["support"]),
            }

    output = {
        "experiment":        "exp2_crosslingual_enhanced",
        "model":             MODEL_NAME,
        "cross_lingual_source": CROSSLINGUAL_SOURCE,
        "oversample_targets": OVERSAMPLE_TARGETS,
        "cl_weight_multiplier": CL_WEIGHT_MULTIPLIER,
        "test_accuracy":    round(float(accuracy_score(y_true, y_pred)), 4),
        "test_f1_macro":    round(float(f1_score(y_true, y_pred, average="macro",    zero_division=0)), 4),
        "test_f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "per_class": per_class,
        "training_history": history,
        "comparison_vs_exp1": {},
    }

    # ── Exp1 ile karşılaştırma ────────────────────────────────────
    if EXP1_JSON.exists():
        with open(EXP1_JSON, encoding="utf-8") as f:
            exp1 = json.load(f)

        delta_macro = output["test_f1_macro"] - exp1.get("test_f1_macro", 0)
        output["comparison_vs_exp1"]["f1_macro_delta"]    = round(delta_macro, 4)
        output["comparison_vs_exp1"]["f1_macro_exp1"]     = exp1.get("test_f1_macro", None)
        output["comparison_vs_exp1"]["f1_macro_exp2"]     = output["test_f1_macro"]
        output["comparison_vs_exp1"]["accuracy_delta"]    = round(
            output["test_accuracy"] - exp1.get("test_accuracy", 0), 4
        )

        per_class_delta = {}
        exp1_pc = exp1.get("per_class", {})
        for emo in EMOTIONS:
            f1_e1 = exp1_pc.get(emo, {}).get("f1", 0.0)
            f1_e2 = per_class.get(emo, {}).get("f1", 0.0)
            delta = round(f1_e2 - f1_e1, 4)
            per_class_delta[emo] = {
                "exp1_f1": round(f1_e1, 4),
                "exp2_f1": round(f1_e2, 4),
                "delta":   delta,
                "improved": delta > 0,
            }
        output["comparison_vs_exp1"]["per_class_delta"] = per_class_delta

        # Özel: Gurur & Utanç TP
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        output["comparison_vs_exp1"]["minority_tp_rates"] = {
            emo: round(float(cm_norm[LABEL2ID[emo], LABEL2ID[emo]]), 4)
            for emo in ["gurur", "utanç", "tiksinti", "korku"]
            if (np.array(y_true) == LABEL2ID[emo]).any()
        }
    else:
        output["comparison_vs_exp1"]["note"] = "Exp1 JSON bulunamadı, delta hesaplanamadı."

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[VIZ] Metrics report: {save_path}")

    # ── Konsol özet ──────────────────────────────────────────────
    print("\n" + "─"*75)
    print(f"  {'Duygu':12s} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Destek':>8} {'∆Exp1':>8}")
    print("─"*75)
    delta_info = output["comparison_vs_exp1"].get("per_class_delta", {})
    for emo in EMOTIONS:
        m  = per_class.get(emo, {})
        d  = delta_info.get(emo, {}).get("delta", 0.0)
        sign = "+" if d >= 0 else ""
        flag = " ◄ CL" if emo in OVERSAMPLE_TARGETS else ""
        print(
            f"  {emo:12s} {m.get('precision',0):8.4f} {m.get('recall',0):8.4f} "
            f"{m.get('f1',0):8.4f} {m.get('support',0):8d} {sign}{d:7.4f}{flag}"
        )
    print("─"*75)
    dm = output["comparison_vs_exp1"].get("f1_macro_delta", "N/A")
    sign = ("+" if isinstance(dm, float) and dm >= 0 else "")
    print(f"  {'MACRO':12s} {'':>8} {'':>8} {output['test_f1_macro']:8.4f} {'':>8} {sign}{dm}")
    print(f"  {'WEIGHTED':12s} {'':>8} {'':>8} {output['test_f1_weighted']:8.4f}")
    print(f"  {'ACCURACY':12s} {'':>8} {'':>8} {output['test_accuracy']:8.4f}")
    print("─"*75)

    tp_rates = output["comparison_vs_exp1"].get("minority_tp_rates", {})
    if tp_rates:
        print("\n  [AZINLIK SINIF TP ORANLARI]")
        for emo, tp in tp_rates.items():
            bar = "█" * int(tp * 20)
            print(f"  {emo:12s}: {tp:.4f}  {bar}")

    return output


# ─── Ana Eğitim Fonksiyonu ───────────────────────────────────────────────────

def train(epochs: int = 5, max_per_class: int = None):
    set_seed(SEED)
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*65)
    print("  EXPERIMENT 2 — BERTURK + CROSS-LINGUAL ENHANCED")
    print("="*65)

    # ── 1. Veri ──────────────────────────────────────────────────
    df = load_dataframe()
    df_train_base, df_val, df_test = make_splits(df)

    # Hızlı eğitim: çoğunluk sınıfları cap'le, azınlıklar CL ile büyüyecek
    if max_per_class:
        df_train_base = (
            df_train_base.groupby("label_id", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max_per_class), random_state=42))
            .reset_index(drop=True)
        )
        print(f"[FAST] max_per_class={max_per_class} -> base: {len(df_train_base):,} ornek")

    # ── 2. Cross-Lingual Enhancement ─────────────────────────────
    enhancer = CrossLingualEnhancer()
    df_train = enhancer.enhance(df_train_base)
    enhancer.log_distribution(df_train, "EĞİTİM SETİ (OVERSAMPLE SONRASI)")

    # ── 3. Tokenizer & Dataset ───────────────────────────────────
    print(f"\n[MODEL] Tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = EmotionDataset(df_train, tokenizer, max_length=128)
    val_ds   = EmotionDataset(df_val,   tokenizer, max_length=128)
    test_ds  = EmotionDataset(df_test,  tokenizer, max_length=128)

    # ── 4. Ağırlıklar ─────────────────────────────────────────────
    sample_weights = compute_sample_weights(df_train)
    class_weights  = compute_class_weights(df_train["label_id"].values)

    print("\n[AĞIRLIK] Sınıf ağırlıkları (CrossEntropyLoss):")
    for i, emo in enumerate(EMOTIONS):
        cw_val = float(class_weights[i])
        flag = " ← CL-enhanced" if emo in OVERSAMPLE_TARGETS else ""
        print(f"  {emo:12s}: {cw_val:.4f}{flag}")

    # ── 5. Model ──────────────────────────────────────────────────
    print(f"\n[MODEL] Yükleniyor: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        ignore_mismatched_sizes=True,
    )

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
    )

    epoch_logger  = EpochLogger()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedTrainer(
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
    print("  EĞİTİM BAŞLIYOR")
    print(f"  Efektif batch : {64 * 2} | Adım/epoch: ~{steps_per_epoch}")
    print(f"  Warmup adımı  : {warmup_steps}")
    print(f"  CL Boost      : {CL_WEIGHT_MULTIPLIER}x (minority cross-lingual)")
    print("="*65 + "\n")

    train_result = trainer.train()
    print(f"\n[TAMAMLANDI] Adım:{train_result.global_step} | "
          f"Kayıp:{train_result.training_loss:.4f} | "
          f"Süre:{train_result.metrics['train_runtime']/60:.1f}dk")

    # ── 8. Test ───────────────────────────────────────────────────
    print("\n[TEST] Test seti değerlendiriliyor...")
    test_output = trainer.predict(test_ds)
    y_pred = np.argmax(test_output.predictions, axis=-1)
    y_true = test_output.label_ids

    # ── 9. Görseller & Rapor ──────────────────────────────────────
    print("\n[VIZ] Görseller oluşturuluyor...")
    save_confusion_matrix(y_true, y_pred, RESULTS / "confusion_matrix.png")
    save_training_curves(epoch_logger.history, RESULTS / "training_progress.png")
    report = save_metrics_report(y_true, y_pred, epoch_logger.history, RESULTS / "metrics_report.json")
    save_f1_comparison(report["per_class"], RESULTS / "f1_comparison.png")
    save_tp_minority_report(y_true, y_pred, RESULTS / "tp_minority_report.png")

    # ── 9b. Kapsamlı Analitik ─────────────────────────────────────
    run_full_analytics(
        y_true=y_true,
        y_pred=y_pred,
        logits_or_probs=test_output.predictions,
        epoch_logger=epoch_logger,
        train_result=train_result,
        out_dir=RESULTS,
        exp_name="Exp2 Cross-Lingual Enhanced",
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        tokenizer=tokenizer,
        extra_info={"cl_weight_multiplier": CL_WEIGHT_MULTIPLIER,
                    "oversample_targets": OVERSAMPLE_TARGETS},
    )

    # ── 10. Model Kaydet ──────────────────────────────────────────
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    print(f"\n[KAYIT] Model: {best_model_dir}")
    print(f"[KAYIT] Sonuçlar: {RESULTS}")

    return trainer


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exp2 — BERTurk + Cross-Lingual Enhanced")
    parser.add_argument("--rebuild-data",    action="store_true", help="Splitleri yeniden oluştur")
    parser.add_argument("--epochs",          type=int,   default=5)
    parser.add_argument("--max-per-class",   type=int,   default=None,
                        help="Sinif basi max ornek (hizli egitim, orn: 10000)")
    parser.add_argument("--cl-multiplier",   type=float, default=CL_WEIGHT_MULTIPLIER,
                        help="Cross-lingual minority örnek ağırlık katsayısı (varsayılan: 2.5)")
    args = parser.parse_args()

    if args.cl_multiplier != CL_WEIGHT_MULTIPLIER:
        CL_WEIGHT_MULTIPLIER = args.cl_multiplier
        print(f"[CONFIG] CL multiplier: {CL_WEIGHT_MULTIPLIER}")

    splits_dir = ROOT / "data" / "splits" / "master_splits"
    if args.rebuild_data or not (splits_dir / "train.parquet").exists():
        df = load_dataframe()
        make_splits(df)

    train(epochs=args.epochs, max_per_class=args.max_per_class)
