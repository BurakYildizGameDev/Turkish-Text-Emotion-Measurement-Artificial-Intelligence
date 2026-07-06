"""
Experiment 0 — Gerçek Baseline (WeightedRandomSampler YOK)
============================================================
EXP1 ile aynı ayarlar ANCAK:
  - WeightedRandomSampler KULLANILMIYOR
  - class_weight = None (CrossEntropyLoss ağırlıksız)
  - Sadece 2 epoch (hızlı referans noktası)

Bu deney, EXP1-EXP4'teki WeightedRandomSampler'ın etkisini
ölçmek için referans noktası oluşturur.

Model   : dbmdz/bert-base-turkish-cased
Veri    : Aynı master_splits (karşılaştırılabilirlik)
Sampler : STANDART (SequentialSampler) — AĞIRLIKSIZ
Kayıp   : CrossEntropyLoss (ağırlıksız)
Donanım : RTX 5070 Ti  ->  fp16=True, batch=64, grad_accum=2

Çıktılar (results/exp0/):
  confusion_matrix.png
  training_progress.png
  metrics_report.json
  best_model/
  test_preds.npy / test_labels.npy / test_probs.npy
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
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
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
RESULTS    = ROOT / "results" / "exp0"
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
SEED       = 42

EMOTIONS = [
    "mutluluk", "üzüntü", "öfke", "korku",
    "şaşkınlık", "tiksinti", "sevgi", "nötr", "utanç", "gurur",
]
ASCII_TO_ID = {
    "mutluluk":  0, "uzuntu": 1, "ofke":      2, "korku":   3,
    "saskınlık": 4, "tiksinti": 5, "sevgi":   6, "notr":    7,
    "utanc":     8, "gurur": 9,
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
        if "label_id" not in df.columns:
            df["label_id"] = df["label"].map(LABEL2ID)
    elif legacy.exists():
        print(f"[DATA] combined_clean.parquet yükleniyor ({legacy})")
        df = pd.read_parquet(legacy)
        if "label_id" not in df.columns:
            df["label_id"] = df["label_norm"].map(ASCII_TO_ID)
        if "label" not in df.columns:
            df["label"] = df["label_id"].map(ID2LABEL)
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


def load_splits():
    """Mevcut EXP1 split'lerini kullan (karşılaştırılabilirlik)."""
    splits_dir = ROOT / "data" / "splits" / "master_splits"
    
    train_p = splits_dir / "train.parquet"
    val_p   = splits_dir / "val.parquet"
    test_p  = splits_dir / "test.parquet"
    
    if train_p.exists() and val_p.exists() and test_p.exists():
        print("[SPLIT] Mevcut split'ler kullanılıyor (EXP1 ile aynı)")
        df_train = pd.read_parquet(train_p)
        df_val   = pd.read_parquet(val_p)
        df_test  = pd.read_parquet(test_p)
    else:
        print("[SPLIT] Split dosyaları bulunamadı, yeni oluşturuluyor...")
        df = load_dataframe()
        df_tv, df_test = train_test_split(
            df, test_size=0.15, stratify=df["label_id"], random_state=SEED
        )
        val_r = 0.15 / (1.0 - 0.15)
        df_train, df_val = train_test_split(
            df_tv, test_size=val_r, stratify=df_tv["label_id"], random_state=SEED
        )
        splits_dir.mkdir(parents=True, exist_ok=True)
        for name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
            subset.reset_index(drop=True).to_parquet(splits_dir / f"{name}.parquet", index=False)
    
    # label sütunu yoksa ekle
    for d in [df_train, df_val, df_test]:
        if "label" not in d.columns:
            d["label"] = d["label_id"].map(ID2LABEL)
    
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    print(f"[SPLIT] Train:{len(df_train):,}  Val:{len(df_val):,}  Test:{len(df_test):,}")
    return df_train, df_val, df_test


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


# ─── Vanilla Trainer (AĞIRLIKSIZ) ───────────────────────────────────────────

# NOT: Standart Trainer kullanıyoruz. WeightedRandomSampler YOK.
# NOT: CrossEntropyLoss ağırlıksız. Bu EXP1'den en büyük fark.


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

    fig.suptitle("Experiment 0 — AĞIRLIKSIZ Baseline BERTurk | Confusion Matrix\n"
                 "(WeightedRandomSampler YOK, class_weight YOK)",
                 fontsize=14, fontweight="bold")
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

    fig.suptitle("Experiment 0 — AĞIRLIKSIZ Baseline | Eğitim İlerlemesi",
                 fontsize=13, fontweight="bold")
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
        "experiment": "exp0_unweighted_baseline",
        "model": MODEL_NAME,
        "note": "WeightedRandomSampler ve class_weight kullanılmamıştır. Gerçek baseline.",
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

    # Exp1 ile karşılaştırma
    exp1_json = ROOT / "results" / "exp1" / "metrics_report.json"
    if exp1_json.exists():
        with open(exp1_json, encoding="utf-8") as f:
            exp1 = json.load(f)
        
        output["comparison_vs_exp1"] = {
            "exp0_f1_macro": output["test_f1_macro"],
            "exp1_f1_macro": exp1.get("test_f1_macro", 0),
            "delta_f1_macro": round(exp1.get("test_f1_macro", 0) - output["test_f1_macro"], 4),
            "note": "Pozitif delta = WeightedRandomSampler iyileştirme sağlamış",
            "per_class_delta": {},
        }
        exp1_pc = exp1.get("per_class", {})
        for emo in EMOTIONS:
            f1_e0 = output["per_class"].get(emo, {}).get("f1", 0)
            f1_e1 = exp1_pc.get(emo, {}).get("f1", 0)
            output["comparison_vs_exp1"]["per_class_delta"][emo] = {
                "exp0_f1": round(f1_e0, 4),
                "exp1_f1": round(f1_e1, 4),
                "delta": round(f1_e1 - f1_e0, 4),
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

def train(epochs: int = 2):
    set_seed(SEED)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ── 1. Veri ──────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  EXPERIMENT 0 — AĞIRLIKSIZ BASELINE (WeightedRandomSampler YOK)")
    print("="*65)

    df_train, df_val, df_test = load_splits()

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

    # ── 3. AĞIRLIK YOK ───────────────────────────────────────────
    print("\n[AĞIRLIK] *** AĞIRLIK KULLANILMIYOR ***")
    print("          WeightedRandomSampler: KAPALI")
    print("          CrossEntropyLoss weight: None")

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

    steps_per_epoch = len(df_train) // (64 * 2)
    warmup_steps    = int(0.1 * steps_per_epoch * epochs)

    training_args = TrainingArguments(
        output_dir=str(best_model_dir),
        # ─ Donanım (RTX 5070 Ti) ─
        fp16=True,
        bf16=False,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=2,
        gradient_checkpointing=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # ─ Eğitim ─
        num_train_epochs=epochs,
        max_steps=500,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        # ─ Değerlendirme & Kayıt ─
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        # ─ Logging ─
        logging_steps=50,
        logging_first_step=True,
        report_to=[],
        seed=SEED,
    )

    # ── 6. STANDART Trainer (Ağırlıksız) ─────────────────────────
    epoch_logger = EpochLogger()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # NOT: Standart Trainer kullanıyoruz, WeightedTrainer DEĞİL!
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(),
        callbacks=[
            epoch_logger,
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )

    # ── 7. Eğitim ─────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  EĞİTİM BAŞLIYOR (AĞIRLIKSIZ)")
    print(f"  Efektif batch size : {64 * 2} (64 × grad_accum=2)")
    print(f"  Adım/epoch         : ~{steps_per_epoch}")
    print(f"  Warmup adımı       : {warmup_steps}")
    print(f"  Epoch              : {epochs}")
    print(f"  WeightedSampler    : ✗ KAPALI")
    print(f"  class_weight       : ✗ None")
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
        exp_name="Exp0 Unweighted Baseline",
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

    parser = argparse.ArgumentParser(
        description="Exp0 — Ağırlıksız Baseline BERTurk (WeightedRandomSampler YOK)"
    )
    parser.add_argument(
        "--epochs", type=int, default=2,
        help="Epoch sayısı (varsayılan: 2)"
    )
    args = parser.parse_args()

    train(epochs=args.epochs)
