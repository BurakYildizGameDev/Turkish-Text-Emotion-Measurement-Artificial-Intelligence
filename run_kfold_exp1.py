# Commit 147: exp(validation): execute 4-fold cross-validation and save fold metrics
"""
EXP1 — 3-Fold Stratified Cross-Validation
==========================================
Baseline BERTurk: WeightedRandomSampler + WeightedCrossEntropyLoss
Parametreler: max_per_class=5000, epochs=2, n_folds=3

Ciktilar (results/kfold/):
  fold_comparison_bar.png   — grouped bar chart (3 fold x 3 metrik)
  per_class_f1_kfold.png    — sinif bazli F1 ortalama +- std
  kfold_boxplot.png         — F1-Makro boxplot (3 fold)
  kfold_summary.txt         — sayisal ozet + sure tahmini
  kfold_results.json        — tam sonuclar JSON
  fold_N/fold_N_metrics.json — her fold icin detay
"""

import os
import sys
import json
import time
import shutil
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
import seaborn as sns
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, WeightedRandomSampler

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
import evaluate

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ─── Sabitler ─────────────────────────────────────────────────

ROOT          = Path(__file__).resolve().parent
RESULTS       = ROOT / "results" / "kfold"
MODEL_NAME    = "dbmdz/bert-base-turkish-cased"
SEED          = 42
N_SPLITS      = 3
EPOCHS        = 2
MAX_PER_CLASS = 5000
BATCH_SIZE    = 64
GRAD_ACCUM    = 2

EMOTIONS = [
    "mutluluk", "üzüntü", "öfke", "korku",
    "şaşkınlık", "tiksinti", "sevgi", "nötr", "utanç", "gurur",
]
ASCII_TO_ID = {
    "mutluluk": 0, "uzuntu": 1, "ofke": 2, "korku": 3,
    "saskınlık": 4, "tiksinti": 5, "sevgi": 6, "notr": 7,
    "utanc": 8, "gurur": 9,
    # Unicode formlar
    "üzüntü": 1, "öfke": 2,
    "şaşkınlık": 4, "nötr": 7, "utanç": 8,
}
LABEL2ID  = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL  = {i: e for i, e in enumerate(EMOTIONS)}
NUM_LABELS = 10


# ─── Veri ─────────────────────────────────────────────────────

def load_dataframe() -> pd.DataFrame:
    master = ROOT / "data" / "processed" / "master_dataset.parquet"
    legacy = ROOT / "data" / "processed" / "combined_clean.parquet"

    if master.exists():
        print(f"[DATA] master_dataset.parquet yukleniyor")
        df = pd.read_parquet(master)
        if "label_id" not in df.columns:
            df["label_id"] = df["label"].map(LABEL2ID)
    elif legacy.exists():
        print(f"[DATA] combined_clean.parquet yukleniyor")
        df = pd.read_parquet(legacy)
        if "label_id" not in df.columns:
            df["label_id"] = df["label_norm"].map(ASCII_TO_ID)
        if "label" not in df.columns:
            df["label"] = df["label_id"].map(ID2LABEL)
    else:
        raise FileNotFoundError(
            "Veri bulunamadi! Once `python data_engine.py --rebuild` calistirin."
        )

    df = df.dropna(subset=["text", "label_id"])
    df["label_id"] = df["label_id"].astype(int)
    df = df[df["label_id"].between(0, NUM_LABELS - 1)]
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 5].reset_index(drop=True)
    print(f"[DATA] Toplam: {len(df):,} ornek yuklendi")
    return df


def cap_per_class(df: pd.DataFrame, max_per_class: int) -> pd.DataFrame:
    groups = []
    for label_id, group in df.groupby("label_id"):
        if len(group) > max_per_class:
            group = group.sample(max_per_class, random_state=SEED)
        groups.append(group)
    capped = pd.concat(groups, ignore_index=True)
    return capped.sample(frac=1, random_state=SEED).reset_index(drop=True)


def estimate_training_time(total_samples: int) -> float:
    """Tahmini egitim suresi (dakika)."""
    train_per_fold = total_samples * (N_SPLITS - 1) / N_SPLITS
    effective_batch = BATCH_SIZE * GRAD_ACCUM
    steps_per_epoch = max(1, int(train_per_fold / effective_batch))
    total_steps = steps_per_epoch * EPOCHS * N_SPLITS
    # RTX 5070 Ti, fp16, BERT-base: ~50ms/step (hesaplama) + overhead
    compute_min = total_steps * 0.05 / 60
    overhead_min = N_SPLITS * 2.5  # model yukleme + eval + tokenization per fold
    return compute_min + overhead_min


# ─── Dataset ──────────────────────────────────────────────────

class EmotionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.texts   = df["text"].tolist()
        self.labels  = df["label_id"].tolist()
        self.tok     = tokenizer
        self.max_len = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding=False,
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ─── Agirliklar ───────────────────────────────────────────────

def compute_class_weights(label_ids: np.ndarray) -> torch.Tensor:
    counts = np.bincount(label_ids, minlength=NUM_LABELS).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    return torch.tensor(len(label_ids) / (counts * NUM_LABELS), dtype=torch.float32)


def compute_sample_weights(label_ids: np.ndarray) -> np.ndarray:
    counts = np.bincount(label_ids, minlength=NUM_LABELS).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    class_w = len(label_ids) / (counts * NUM_LABELS)
    return class_w[label_ids]


# ─── Custom Trainer ───────────────────────────────────────────

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
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        weight  = self._class_weights.to(outputs.logits.device) if self._class_weights is not None else None
        loss    = nn.CrossEntropyLoss(weight=weight)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# ─── Metrikler ────────────────────────────────────────────────

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
        per_class = {f"f1_{ID2LABEL[i]}": float(f1_per[i]) for i in range(len(f1_per))}
        return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_wt, **per_class}

    return compute_metrics


# ─── Tek Fold Egitimi ─────────────────────────────────────────

def run_fold(fold_idx: int, df_train: pd.DataFrame, df_val: pd.DataFrame,
             tokenizer, fold_dir: Path) -> dict:
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx+1}/{N_SPLITS}  BASLIYOR")
    print(f"  Train: {len(df_train):,}  |  Val/Test: {len(df_val):,}")
    print(f"{'='*60}\n")

    set_seed(SEED + fold_idx)
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Tmp checkpoint dizini (fold bittikten sonra silinecek)
    tmp_dir = fold_dir / "tmp_ckpt"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    train_ds = EmotionDataset(df_train, tokenizer)
    val_ds   = EmotionDataset(df_val,   tokenizer)

    train_labels   = df_train["label_id"].values
    sample_weights = compute_sample_weights(train_labels)
    class_weights  = compute_class_weights(train_labels)

    # Sinif dagılımı
    cnt = Counter(train_labels.tolist())
    print("[TRAIN DAGILI MI]")
    for i, emo in enumerate(EMOTIONS):
        c = cnt.get(i, 0)
        print(f"  {emo:12s}: {c:6,d}")

    # Her fold icin sifirdan model yukle
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        ignore_mismatched_sizes=True,
    )

    steps_per_epoch = max(1, len(df_train) // (BATCH_SIZE * GRAD_ACCUM))
    warmup_steps    = max(1, int(0.1 * steps_per_epoch * EPOCHS))

    training_args = TrainingArguments(
        output_dir=str(tmp_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=GRAD_ACCUM,
        fp16=True,
        bf16=False,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        logging_steps=50,
        logging_first_step=True,
        report_to=[],
        seed=SEED + fold_idx,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

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
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # Final tahmin
    pred_out = trainer.predict(val_ds)
    y_pred   = np.argmax(pred_out.predictions, axis=-1)
    y_true   = pred_out.label_ids

    f1_macro  = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    f1_wt     = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    acc       = float(accuracy_score(y_true, y_pred))

    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = {EMOTIONS[i]: float(f1_per_class[i]) for i in range(NUM_LABELS)}

    metrics = {
        "fold":        fold_idx + 1,
        "f1_macro":    f1_macro,
        "accuracy":    acc,
        "f1_weighted": f1_wt,
        "per_class_f1": per_class_f1,
        "train_size":  int(len(df_train)),
        "val_size":    int(len(df_val)),
        "duration_s":  elapsed,
    }

    # Fold metriklerini kaydet
    metrics_path = fold_dir / f"fold_{fold_idx+1}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n[FOLD {fold_idx+1} SONUC]")
    print(f"  F1-Makro   : {f1_macro:.4f}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  F1-Weighted: {f1_wt:.4f}")
    print(f"  Sure       : {elapsed/60:.1f} dk")

    # VRAM temizle, tmp checkpoint sil
    del model, trainer
    torch.cuda.empty_cache()
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return metrics


# ─── Gorsellestirme ───────────────────────────────────────────

def plot_fold_comparison_bar(fold_metrics: list, save_path: Path):
    """Grouped bar chart: 3 fold × 3 metrik."""
    metric_labels = ["F1-Makro", "Accuracy", "F1-Weighted"]
    metric_keys   = ["f1_macro", "accuracy", "f1_weighted"]
    fold_names    = [f"Fold {fm['fold']}" for fm in fold_metrics]
    colors        = ["#3498db", "#2ecc71", "#e74c3c"]

    x     = np.arange(len(metric_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 7))

    for i, (fm, color) in enumerate(zip(fold_metrics, colors)):
        vals = [fm[k] for k in metric_keys]
        bars = ax.bar(
            x + i * width, vals, width,
            label=fold_names[i], color=color, alpha=0.85,
            edgecolor="white", linewidth=0.8,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.007,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold",
            )

    # Ortalama cizgisi (F1-Makro icin)
    mean_f1 = np.mean([fm["f1_macro"] for fm in fold_metrics])
    ax.axhline(
        y=mean_f1, xmin=0.02, xmax=0.38,
        color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.7,
        label=f"F1-Makro Ort. ({mean_f1:.3f})",
    )

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=13)
    ax.set_ylabel("Skor", fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "EXP1 Baseline BERTurk — 3-Fold CV Karsilastirmasi\n"
        f"(max_per_class={MAX_PER_CLASS}, epochs={EPOCHS})",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] fold_comparison_bar.png kaydedildi")


def plot_per_class_f1_kfold(fold_metrics: list, save_path: Path):
    """Per-class F1 mean +- std across 3 folds."""
    means, stds, labels_short = [], [], []
    for emo in EMOTIONS:
        vals = [fm["per_class_f1"].get(emo, 0.0) for fm in fold_metrics]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
        labels_short.append(emo[:9])

    x = np.arange(len(EMOTIONS))

    # Renk: dusuk F1 -> kirmizi, yuksek -> mavi
    norm_vals = np.array(means)
    cmap = plt.cm.RdYlGn
    bar_colors = [cmap(v) for v in norm_vals]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(
        x, means, yerr=stds, capsize=6,
        color=bar_colors, alpha=0.85,
        error_kw={"linewidth": 2, "color": "#2c3e50", "capthick": 2},
        edgecolor="white", linewidth=0.8,
    )

    for bar, m, s in zip(bars, means, stds):
        ypos = bar.get_height() + s + 0.012
        ax.text(
            bar.get_x() + bar.get_width() / 2, ypos,
            f"{m:.3f}", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color="#2c3e50",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, fontsize=11, rotation=30, ha="right")
    ax.set_ylabel("F1 Skoru (Ortalama ± Std Dev)", fontsize=13)
    ax.set_ylim(0, 1.20)
    ax.set_title(
        "EXP1 Baseline BERTurk — Sinif Bazli F1 (3-Fold Ortalama)\n"
        "Hata cubuklari = standart sapma",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(
        y=np.mean(means), color="#3498db", linestyle="--",
        linewidth=1.5, alpha=0.7,
        label=f"Ortalama F1 ({np.mean(means):.3f})",
    )
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] per_class_f1_kfold.png kaydedildi")


def plot_kfold_boxplot(fold_metrics: list, save_path: Path):
    """F1-Makro dagilimini gosteren boxplot (3 fold)."""
    f1_values = [fm["f1_macro"] for fm in fold_metrics]
    fold_labels = [f"Fold {fm['fold']}" for fm in fold_metrics]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Boxplot (3 nokta icin de gecerli istatistiksel gosterim)
    bp = ax.boxplot(
        f1_values,
        positions=[1],
        widths=0.5,
        patch_artist=True,
        medianprops={"color": "#e74c3c", "linewidth": 3},
        boxprops={"facecolor": "#3498db", "alpha": 0.6},
        whiskerprops={"linewidth": 2},
        capprops={"linewidth": 2},
        flierprops={"marker": "o", "markerfacecolor": "#e74c3c", "markersize": 8},
    )

    # Bireysel fold degerlerini numaralandirilmis nokta olarak ekle
    rng = np.random.default_rng(42)
    for j, (v, lbl) in enumerate(zip(f1_values, fold_labels)):
        jitter = rng.uniform(-0.08, 0.08)
        ax.scatter(1 + jitter, v, zorder=5, s=100, color="#2c3e50", alpha=0.9)
        ax.annotate(
            lbl, xy=(1 + jitter, v),
            xytext=(1 + jitter + 0.12, v),
            fontsize=10, fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": "gray", "lw": 1},
        )

    # Ortalama cizgisi
    mean_f1 = np.mean(f1_values)
    std_f1  = np.std(f1_values)
    ax.axhline(y=mean_f1, color="#27ae60", linestyle="--", linewidth=2,
               label=f"Ortalama: {mean_f1:.3f} ± {std_f1:.3f}")

    ax.set_xticks([1])
    ax.set_xticklabels(["F1-Makro"], fontsize=13)
    ax.set_ylabel("Skor", fontsize=13)
    ax.set_xlim(0.4, 2.0)
    ax.set_ylim(max(0, mean_f1 - 0.25), min(1.0, mean_f1 + 0.25))
    ax.set_title(
        "EXP1 Baseline BERTurk\nF1-Makro Dagilimi (3-Fold CV)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] kfold_boxplot.png kaydedildi")


def write_kfold_summary(fold_metrics: list, total_time: float, save_path: Path):
    f1_macros = [fm["f1_macro"]    for fm in fold_metrics]
    accs      = [fm["accuracy"]    for fm in fold_metrics]
    f1_wts    = [fm["f1_weighted"] for fm in fold_metrics]

    lines = [
        "=" * 65,
        "  EXP1 BASELINE BERTURK — 3-FOLD CV OZET",
        "=" * 65,
        f"  Model          : dbmdz/bert-base-turkish-cased",
        f"  max_per_class  : {MAX_PER_CLASS}",
        f"  Epochs/fold    : {EPOCHS}",
        f"  N Folds        : {N_SPLITS}",
        f"  Batch          : {BATCH_SIZE} x grad_accum={GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM} efektif",
        "",
        "  --- Her Fold Sonuclari ---",
    ]

    for fm in fold_metrics:
        lines += [
            f"",
            f"  Fold {fm['fold']}  (train={fm['train_size']:,}  val/test={fm['val_size']:,})",
            f"    F1-Makro   : {fm['f1_macro']:.4f}",
            f"    Accuracy   : {fm['accuracy']:.4f}",
            f"    F1-Weighted: {fm['f1_weighted']:.4f}",
            f"    Sure       : {fm['duration_s']/60:.1f} dakika",
        ]

    lines += [
        "",
        "=" * 65,
        "  --- Ortalama +- Standart Sapma ---",
        "",
        f"  F1-Makro   : {np.mean(f1_macros):.3f} +- {np.std(f1_macros):.3f}",
        f"  Accuracy   : {np.mean(accs):.3f} +- {np.std(accs):.3f}",
        f"  F1-Weighted: {np.mean(f1_wts):.3f} +- {np.std(f1_wts):.3f}",
        "",
        f"  Toplam egitim suresi  : {total_time/60:.1f} dakika",
        f"  Fold basina ortalama  : {total_time/60/N_SPLITS:.1f} dakika",
        "",
        "=" * 65,
        "  --- Sinif Bazli F1 (3-Fold Ortalama) ---",
        "",
    ]

    for emo in EMOTIONS:
        vals = [fm["per_class_f1"].get(emo, 0.0) for fm in fold_metrics]
        bar = "#" * int(np.mean(vals) * 20)
        lines.append(
            f"  {emo:12s}: {np.mean(vals):.3f} +- {np.std(vals):.3f}  {bar}"
        )

    lines += [
        "",
        "=" * 65,
    ]

    text = "\n".join(lines)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"\n[OZET] kfold_summary.txt kaydedildi: {save_path}")
    print(text)


# ─── Ana Fonksiyon ────────────────────────────────────────────

def main():
    t_global = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  EXP1 — 3-FOLD STRATIFIED CROSS-VALIDATION")
    print(f"  max_per_class={MAX_PER_CLASS}  epochs={EPOCHS}  n_folds={N_SPLITS}")
    print("=" * 65)

    # ── 1. Veri yukle ve cap uygula ───────────────────────────
    df = load_dataframe()
    df_capped = cap_per_class(df, MAX_PER_CLASS)

    print(f"\n[CAP] max_per_class={MAX_PER_CLASS} sonrasi: {len(df_capped):,} ornek")
    cnt = Counter(df_capped["label_id"].tolist())
    print("[DAGIL]")
    for i, emo in enumerate(EMOTIONS):
        c = cnt.get(i, 0)
        print(f"  {emo:12s}: {c:6,d}")

    # ── Sure tahmini ──────────────────────────────────────────
    est_min = estimate_training_time(len(df_capped))
    print(f"\n[SURE TAHMINI] Yaklasik {est_min:.0f}-{est_min*1.4:.0f} dakika")
    print(f"  (RTX 5070 Ti, fp16, batch={BATCH_SIZE}x{GRAD_ACCUM}, 3 fold x {EPOCHS} epoch)")
    print(f"  Toplam ~ {N_SPLITS} fold x {est_min/N_SPLITS:.0f} dk/fold")

    # ── 2. Tokenizer (tum foldlar paylasiyor) ─────────────────
    print(f"\n[TOKENIZER] Yukleniyor: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── 3. StratifiedKFold ────────────────────────────────────
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    X   = df_capped.index.values
    y   = df_capped["label_id"].values

    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        df_train = df_capped.iloc[train_idx].reset_index(drop=True)
        df_val   = df_capped.iloc[val_idx].reset_index(drop=True)
        fold_dir = RESULTS / f"fold_{fold_idx + 1}"

        metrics = run_fold(fold_idx, df_train, df_val, tokenizer, fold_dir)
        fold_metrics.append(metrics)

        # Anlık ilerleme ozeti
        f1s = [fm["f1_macro"] for fm in fold_metrics]
        print(f"\n[ILERLEME] {len(fold_metrics)}/{N_SPLITS} fold tamamlandi")
        if len(fold_metrics) > 1:
            print(f"  Simdiye kadar F1-Makro: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}")

    total_time = time.time() - t_global

    # ── 4. Tam sonuclari kaydet ───────────────────────────────
    f1_macros = [fm["f1_macro"]    for fm in fold_metrics]
    accs      = [fm["accuracy"]    for fm in fold_metrics]
    f1_wts    = [fm["f1_weighted"] for fm in fold_metrics]

    all_metrics = {
        "experiment": "exp1_baseline_berturk_kfold",
        "config": {
            "model": MODEL_NAME,
            "max_per_class": MAX_PER_CLASS,
            "epochs": EPOCHS,
            "n_splits": N_SPLITS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
        },
        "fold_results": fold_metrics,
        "summary": {
            "f1_macro_mean":    float(np.mean(f1_macros)),
            "f1_macro_std":     float(np.std(f1_macros)),
            "accuracy_mean":    float(np.mean(accs)),
            "accuracy_std":     float(np.std(accs)),
            "f1_weighted_mean": float(np.mean(f1_wts)),
            "f1_weighted_std":  float(np.std(f1_wts)),
        },
        "total_duration_s": total_time,
    }
    with open(RESULTS / "kfold_results.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    # ── 5. Gorseller ─────────────────────────────────────────
    print("\n[VIZ] Grafikler olusturuluyor...")
    plot_fold_comparison_bar(fold_metrics, RESULTS / "fold_comparison_bar.png")
    plot_per_class_f1_kfold(fold_metrics,  RESULTS / "per_class_f1_kfold.png")
    plot_kfold_boxplot(fold_metrics,       RESULTS / "kfold_boxplot.png")
    write_kfold_summary(fold_metrics, total_time, RESULTS / "kfold_summary.txt")

    # ── 6. Final cikti ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL SONUCLAR")
    print("=" * 65)
    for fm in fold_metrics:
        print(f"  Fold {fm['fold']}: F1-Makro={fm['f1_macro']:.4f}  "
              f"Acc={fm['accuracy']:.4f}  F1-Wt={fm['f1_weighted']:.4f}")
    print("  " + "-" * 50)
    print(f"  F1-Makro   : {np.mean(f1_macros):.3f} ± {np.std(f1_macros):.3f}")
    print(f"  Accuracy   : {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"  F1-Weighted: {np.mean(f1_wts):.3f} ± {np.std(f1_wts):.3f}")
    print(f"  Toplam sure: {total_time/60:.1f} dakika")
    print(f"  Sonuclar   : {RESULTS}")
    print("=" * 65)


if __name__ == "__main__":
    main()
