# Commit 93: feat(training): add cosine learning rate schedule to train_exp1
"""
Experiment 1 — Baseline BERTurk
================================
Model   : dbmdz/bert-base-turkish-cased
Veri    : data/processed/master_dataset.parquet  (yoksa combined_clean.parquet)
Sampler : WeightedRandomSampler (azınlık sınıf boost)
Kayıp   : WeightedCrossEntropyLoss
Donanım : RTX 5070 Ti  ->  fp16=True, batch=64, grad_accum=2

Çıktılar (results/exp1/):
  confusion_matrix.png   — normalize edilmiş ısı haritası
  training_progress.png  — loss + f1 eğrileri
  metrics_report.json    — per-class precision/recall/f1 + macro/weighted
  best_model/            — en iyi checkpoint
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
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

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
RESULTS    = ROOT / "results" / "exp1"
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
SEED       = 42

EMOTIONS = [
    "mutluluk", "üzüntü", "öfke", "korku",
    "şaşkınlık", "tiksinti", "sevgi", "nötr", "utanç", "gurur",
]
# combined_clean.parquet içindeki ASCII form -> ID eşlemesi
ASCII_TO_ID = {
    "mutluluk":  0, "uzuntu": 1, "ofke":      2, "korku":   3,
    "saskınlık": 4, "tiksinti": 5, "sevgi":   6, "notr":    7,
    "utanc":     8, "gurur": 9,
    # Unicode formlar da kabul et
    "üzüntü": 1, "öfke": 2, "şaşkınlık": 4, "nötr": 7, "utanç": 8,
}
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL  = {i: e for i, e in enumerate(EMOTIONS)}
NUM_LABELS = 10

# ─── Veri Yükleme ───────────────────────────────────────────────────────────

def load_dataframe() -> pd.DataFrame:
    """master_dataset.parquet -> combined_clean.parquet öncelik sırası."""
    master = ROOT / "data" / "processed" / "master_dataset.parquet"
    legacy = ROOT / "data" / "processed" / "combined_clean.parquet"

    if master.exists():
        print(f"[DATA] master_dataset.parquet yükleniyor ({master})")
        df = pd.read_parquet(master)
        # label sütunu Unicode, label_id zaten var
        if "label_id" not in df.columns:
            df["label_id"] = df["label"].map(LABEL2ID)
    elif legacy.exists():
        print(f"[DATA] combined_clean.parquet yükleniyor ({legacy})")
        df = pd.read_parquet(legacy)
        # label_norm sütunu ASCII, label_id zaten var
        if "label_id" not in df.columns:
            df["label_id"] = df["label_norm"].map(ASCII_TO_ID)
        # Görüntü için unicode label ekle
        if "label" not in df.columns:
            id_to_emo = ID2LABEL
            df["label"] = df["label_id"].map(id_to_emo)
    else:
        raise FileNotFoundError(
            "Veri bulunamadı! Önce `python data_engine.py --rebuild` çalıştırın."
        )

    df = df.dropna(subset=["text", "label_id"])
    df["label_id"] = df["label_id"].astype(int)
    df = df[df["label_id"].between(0, NUM_LABELS - 1)]
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 5]
    df = df.reset_index(drop=True)
    print(f"[DATA] Toplam: {len(df):,} örnek")
    return df


def make_splits(df: pd.DataFrame, test_size=0.15, val_size=0.15, force: bool = False):
    """
    Stratified split, splits/master_splits/ altına kaydeder.
    Diskte split varsa (ve force=False ise) onu kullanır — Exp2/3/4 ile aynı test seti garanti edilir.
    """
    splits_dir = ROOT / "data" / "splits" / "master_splits"
    paths = [splits_dir / f"{n}.parquet" for n in ("train", "val", "test")]
    if not force and all(p.exists() for p in paths):
        print("[SPLIT] Mevcut split'ler kullaniliyor (karsilastirabilirlik icin)")
        df_train, df_val, df_test = (pd.read_parquet(p) for p in paths)
        for d in (df_train, df_val, df_test):
            if "label" not in d.columns:
                d["label"] = d["label_id"].map(ID2LABEL)
        print(f"[SPLIT] Train:{len(df_train):,}  Val:{len(df_val):,}  Test:{len(df_test):,}")
        return df_train, df_val, df_test

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


# ─── PyTorch Dataset ────────────────────────────────────────────────────────

class EmotionDataset(Dataset):
    """HuggingFace Trainer ile uyumlu hafif Dataset."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.texts   = df["text"].tolist()
        self.labels  = df["label_id"].tolist()
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


# ─── Ağırlık Hesabı ─────────────────────────────────────────────────────────

def compute_class_weights(label_ids: np.ndarray) -> torch.Tensor:
    """WeightedCrossEntropyLoss için sınıf ağırlıkları."""
    counts = np.bincount(label_ids, minlength=NUM_LABELS).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    weights = len(label_ids) / (counts * NUM_LABELS)
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(label_ids: np.ndarray) -> np.ndarray:
    """WeightedRandomSampler için örnek başına ağırlık."""
    counts = np.bincount(label_ids, minlength=NUM_LABELS).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    class_w = len(label_ids) / (counts * NUM_LABELS)
    return class_w[label_ids]


# ─── Custom Trainer (WeightedRandomSampler + WeightedLoss) ──────────────────

class WeightedTrainer(Trainer):
    """
    İki ekleme:
      1. _get_train_sampler() -> WeightedRandomSampler (azınlık boost)
      2. compute_loss()       -> WeightedCrossEntropyLoss
    """

    def __init__(self, *args, train_sample_weights=None, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._sample_weights = train_sample_weights
        self._class_weights  = class_weights

    def _get_train_sampler(self, dataset=None):
        if self._sample_weights is not None:
            print("[SAMPLER] WeightedRandomSampler aktif.")
            return WeightedRandomSampler(
                weights=torch.from_numpy(self._sample_weights).double(),
                num_samples=len(self._sample_weights),
                replacement=True,
            )
        return super()._get_train_sampler(dataset)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self._class_weights is not None:
            weight = self._class_weights.to(logits.device)
        else:
            weight = None

        loss = nn.CrossEntropyLoss(weight=weight)(logits, labels)

        if return_outputs:
            return loss, outputs
        return loss


# ─── Metrik Fonksiyonu ───────────────────────────────────────────────────────

def build_compute_metrics():
    acc_metric = evaluate.load("accuracy")
    f1_metric  = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        acc      = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
        f1_macro = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
        f1_wt    = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
        f1_per   = f1_metric.compute(predictions=preds, references=labels, average=None)["f1"]

        per_class = {f"f1_{ID2LABEL[i]}": float(f1_per[i]) for i in range(len(f1_per))}

        return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_wt, **per_class}

    return compute_metrics


# ─── Görselleştirme ──────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    labels_short = [e[:8] for e in EMOTIONS]

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

    fig.suptitle("Experiment 1 — Baseline BERTurk | Confusion Matrix", fontsize=15, fontweight="bold")
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

    # Loss
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss",  color="#e74c3c", linewidth=2)
    ax1.plot(epochs, history["val_loss"],   "s-", label="Val Loss",    color="#3498db", linewidth=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss Eğrileri", fontweight="bold")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # F1 + Accuracy
    ax2.plot(epochs, history["val_f1_macro"], "o-", label="Val F1-Macro", color="#2ecc71", linewidth=2)
    ax2.plot(epochs, history["val_acc"],       "s-", label="Val Accuracy", color="#9b59b6", linewidth=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Skor")
    ax2.set_title("Validation Metrikleri", fontweight="bold")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    fig.suptitle("Experiment 1 — Baseline BERTurk | Eğitim İlerlemesi", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Training curves: {save_path}")


def save_metrics_report(y_true, y_pred, history: dict, save_path: Path):
    report_dict = classification_report(
        y_true, y_pred,
        target_names=EMOTIONS,
        output_dict=True,
        zero_division=0,
    )

    output = {
        "experiment": "exp1_baseline_berturk",
        "model": MODEL_NAME,
        "test_accuracy":    float(accuracy_score(y_true, y_pred)),
        "test_f1_macro":    float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "test_f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {},
        "training_history": history,
    }

    for emo in EMOTIONS:
        if emo in report_dict:
            output["per_class"][emo] = {
                "precision": round(report_dict[emo]["precision"], 4),
                "recall":    round(report_dict[emo]["recall"],    4),
                "f1":        round(report_dict[emo]["f1-score"],  4),
                "support":   int(report_dict[emo]["support"]),
            }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[VIZ] Metrics report: {save_path}")

    # Konsola özet tablo
    print("\n" + "─"*65)
    print(f"  {'Duygu':12s} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Destek':>8}")
    print("─"*65)
    for emo, m in output["per_class"].items():
        print(f"  {emo:12s} {m['precision']:10.4f} {m['recall']:8.4f} {m['f1']:8.4f} {m['support']:8d}")
    print("─"*65)
    print(f"  {'MACRO':12s} {'':>10} {'':>8} {output['test_f1_macro']:8.4f}")
    print(f"  {'WEIGHTED':12s} {'':>10} {'':>8} {output['test_f1_weighted']:8.4f}")
    print(f"  {'ACCURACY':12s} {'':>10} {'':>8} {output['test_accuracy']:8.4f}")
    print("─"*65 + "\n")

    return output


# ─── Ana Eğitim Fonksiyonu ───────────────────────────────────────────────────

def train(epochs: int = 5, max_per_class: int = None, batch_size: int = 64):
    set_seed(SEED)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ── 1. Veri ──────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  EXPERIMENT 1 — BASELINE BERTURK")
    print("="*65)

    df = load_dataframe()
    df_train, df_val, df_test = make_splits(df)

    if max_per_class:
        df_train = (
            df_train.groupby("label_id", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max_per_class), random_state=42))
            .reset_index(drop=True)
        )
        print(f"[FAST] max_per_class={max_per_class} -> egitim: {len(df_train):,} ornek")

    # Sınıf dağılımı
    print("\n[DAĞILIM] Eğitim seti:")
    cnt = Counter(df_train["label_id"].tolist())
    for i, emo in enumerate(EMOTIONS):
        c = cnt.get(i, 0)
        print(f"  {emo:12s}: {c:7,d}  ({c/len(df_train)*100:5.1f}%)")

    # ── 2. Tokenizer ──────────────────────────────────────────────
    print(f"\n[MODEL] Tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = EmotionDataset(df_train, tokenizer, max_length=128)
    val_ds   = EmotionDataset(df_val,   tokenizer, max_length=128)
    test_ds  = EmotionDataset(df_test,  tokenizer, max_length=128)

    # ── 3. Ağırlıklar ─────────────────────────────────────────────
    train_labels = df_train["label_id"].values
    sample_weights = compute_sample_weights(train_labels)
    class_weights  = compute_class_weights(train_labels)

    print("\n[AĞIRLIK] Sınıf ağırlıkları (CrossEntropyLoss):")
    for i, emo in enumerate(EMOTIONS):
        print(f"  {emo:12s}: {class_weights[i]:.4f}")

    # ── 4. Model ──────────────────────────────────────────────────
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

    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Toplam parametre : {total_params:,}")
    print(f"  Eğitilebilir     : {trainable:,}")

    # ── 5. Training Arguments ─────────────────────────────────────
    best_model_dir = RESULTS / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    grad_accum      = 2
    steps_per_epoch = len(df_train) // (batch_size * grad_accum)
    warmup_steps    = int(0.1 * steps_per_epoch * epochs)

    training_args = TrainingArguments(
        output_dir=str(best_model_dir),
        # ─ Donanım (RTX 5070 Ti) ─
        fp16=True,
        bf16=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # ─ Eğitim ─
        num_train_epochs=epochs,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        # ─ Değerlendirme & Kayıt ─
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        # ─ Logging ─
        logging_steps=100,
        logging_first_step=True,
        report_to=[],
        seed=SEED,
    )

    # ── 6. Callbacks ──────────────────────────────────────────────
    epoch_logger = EpochLogger()
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
        callbacks=[
            epoch_logger,
            EarlyStoppingCallback(early_stopping_patience=2),
        ],
    )

    # ── 7. Eğitim ─────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  EĞİTİM BAŞLIYOR")
    print(f"  Efektif batch size : {batch_size * grad_accum} ({batch_size} × grad_accum={grad_accum})")
    print(f"  Adım/epoch         : ~{steps_per_epoch}")
    print(f"  Warmup adımı       : {warmup_steps}")
    print("="*65 + "\n")

    train_result = trainer.train()

    print(f"\n[TAMAMLANDI]")
    print(f"  Toplam adım   : {train_result.global_step}")
    print(f"  Eğitim kaybı  : {train_result.training_loss:.4f}")
    print(f"  Süre (dk)     : {train_result.metrics['train_runtime']/60:.1f}")

    # ── 8. Test Değerlendirmesi ───────────────────────────────────
    print("\n[TEST] Test seti değerlendiriliyor...")
    test_output = trainer.predict(test_ds)
    y_pred = np.argmax(test_output.predictions, axis=-1)
    y_true = test_output.label_ids

    # ── 9. Görseller & Rapor ──────────────────────────────────────
    print("\n[VİZ] Görseller oluşturuluyor...")

    save_confusion_matrix(
        y_true, y_pred,
        save_path=RESULTS / "confusion_matrix.png",
    )
    save_training_curves(
        epoch_logger.history,
        save_path=RESULTS / "training_progress.png",
    )
    save_metrics_report(
        y_true, y_pred,
        history=epoch_logger.history,
        save_path=RESULTS / "metrics_report.json",
    )

    # ── 9b. Kapsamlı Analitik ─────────────────────────────────────
    run_full_analytics(
        y_true=y_true,
        y_pred=y_pred,
        logits_or_probs=test_output.predictions,
        epoch_logger=epoch_logger,
        train_result=train_result,
        out_dir=RESULTS,
        exp_name="Exp1 Baseline BERTurk",
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        tokenizer=tokenizer,
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

    parser = argparse.ArgumentParser(description="Exp1 — Baseline BERTurk")
    parser.add_argument(
        "--rebuild-data", action="store_true",
        help="Veriyi sıfırdan indir ve splits'i yeniden oluştur"
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Epoch sayısı (varsayılan: 5)"
    )
    parser.add_argument(
        "--max-per-class", type=int, default=None,
        help="Sinif basi max ornek (hizli egitim icin, orn: 10000)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Per-device batch size (varsayılan: 64)"
    )
    args = parser.parse_args()

    # Splits yoksa oluştur
    splits_dir = ROOT / "data" / "splits" / "master_splits"
    if args.rebuild_data or not (splits_dir / "train.parquet").exists():
        print("[SETUP] Veri splits oluşturuluyor...")
        df = load_dataframe()
        make_splits(df, force=args.rebuild_data)

    train(epochs=args.epochs, max_per_class=args.max_per_class, batch_size=args.batch_size)
