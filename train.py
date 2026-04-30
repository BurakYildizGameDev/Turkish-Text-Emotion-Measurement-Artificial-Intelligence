# Commit 70: feat(training): add seed argument for deterministic reproducibility
"""
BERTurk Fine-Tuning Pipeline
dbmdz/bert-base-turkish-cased modelini data/splits/tremo_splits ile fine-tune eder.
Not: Klasör adı "tremo_splits" olsa da içinde TREMO verisi yoktur; içerik
src/data/download_dataset.py'nin birleştirdiği 7 kaynaktır (bkz. o dosyanın başı).
RTX 5090 Ti için BF16 + büyük batch size optimize edilmiştir.
"""

import os
import yaml
import json
import numpy as np
from pathlib import Path
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
import evaluate
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── Tokenizasyon ────────────────────────────────────────────

def tokenize_dataset(dataset_dict: DatasetDict, tokenizer, max_length: int) -> DatasetDict:
    """
    Her split'i tokenize eder.
    Dynamic padding kullanıyoruz (DataCollatorWithPadding),
    bu yüzden burada padding=False.
    """
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    # Hangi kolonların kaldırılacağını dinamik belirle
    first_split = list(dataset_dict.keys())[0]
    cols_to_remove = [
        c for c in ["text", "label", "__index_level_0__"]
        if c in dataset_dict[first_split].column_names
    ]
    tokenized = dataset_dict.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        remove_columns=cols_to_remove,
        desc="Tokenizing",
    )
    # HF Trainer "labels" kolonunu bekler
    tokenized = tokenized.rename_column("label_id", "labels")
    tokenized.set_format("torch")
    return tokenized


# ─── Metrikler ───────────────────────────────────────────────

def build_compute_metrics(id2label: dict):
    """
    Macro F1, accuracy ve per-class F1 hesaplar.
    Closure ile id2label erişimi sağlar.
    """
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        f1_macro = f1_metric.compute(
            predictions=preds, references=labels, average="macro"
        )["f1"]
        f1_weighted = f1_metric.compute(
            predictions=preds, references=labels, average="weighted"
        )["f1"]

        # Per-class F1
        f1_per_class = f1_metric.compute(
            predictions=preds, references=labels, average=None
        )["f1"]
        per_class = {
            f"f1_{id2label[i]}": float(f1_per_class[i])
            for i in range(len(f1_per_class))
        }

        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            **per_class,
        }

    return compute_metrics


# ─── Confusion Matrix Görselleştirme ─────────────────────────

def plot_confusion_matrix(labels_true, labels_pred, emotion_names: list, save_path: Path):
    cm = confusion_matrix(labels_true, labels_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # normalize

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (Ham)", "Confusion Matrix (Normalize)"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            xticklabels=emotion_names,
            yticklabels=emotion_names,
            cmap="Blues",
            ax=ax,
        )
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gerçek")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Confusion matrix kaydedildi: {save_path}")


# ─── Ana Eğitim Fonksiyonu ───────────────────────────────────

def train():
    config = load_config()
    emo_cfg     = config["emotions"]
    model_cfg   = config["model"]
    train_cfg   = config["training"]

    if emo_cfg["num_labels"] != 10:
        raise SystemExit(
            "Bu v1 (10 sınıf, tremo_splits) eğitim hattıdır; config.yaml artık v2 (8 sınıf) şemasındadır.\n"
            "Yeni eğitim için: python -m src.training.train_v2 --data gold"
        )
    id2label  = {int(k): v for k, v in emo_cfg["id2label"].items()}
    label2id  = emo_cfg["label2id"]
    num_labels = emo_cfg["num_labels"]
    emotion_names = [id2label[i] for i in range(num_labels)]

    # ── 1. Tokenizer ──────────────────────────────────────────
    print(f"\n[LOAD] Tokenizer yükleniyor: {model_cfg['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])

    # ── 2. Veri Seti ──────────────────────────────────────────
    splits_path = ROOT / config["data"]["splits_dir"] / "tremo_splits"
    print(f"[LOAD] Split'ler yükleniyor: {splits_path}")
    dataset_dict = load_from_disk(str(splits_path))

    tokenized = tokenize_dataset(dataset_dict, tokenizer, model_cfg["max_length"])
    print(f"[INFO] Train: {len(tokenized['train'])} | Val: {len(tokenized['validation'])} | Test: {len(tokenized['test'])}")

    # ── 3. Model ──────────────────────────────────────────────
    print(f"[LOAD] Model yükleniyor: {model_cfg['base_model']}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["base_model"],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        hidden_dropout_prob=model_cfg["dropout"],
        attention_probs_dropout_prob=model_cfg["dropout"],
    )

    # ── 4. Training Arguments ─────────────────────────────────
    output_dir = ROOT / train_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["eval_batch_size"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_steps=int(train_cfg["warmup_ratio"] * (len(tokenized["train"]) // train_cfg["batch_size"]) * train_cfg["num_epochs"]),
        lr_scheduler_type=train_cfg["lr_scheduler"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=True,
        # logging_dir deprecated in v5.2 — tensorboard env var kullanılıyor

        logging_steps=50,
        report_to=["tensorboard"] + (["wandb"] if config["logging"].get("use_wandb", False) else []),
        seed=train_cfg["seed"],
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    # ── 5. Trainer ────────────────────────────────────────────
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── 6. Eğit ───────────────────────────────────────────────
    print("\n[TRAIN] Fine-tuning başlıyor...")
    train_result = trainer.train()

    print(f"\n[TRAIN] Tamamlandı.")
    print(f"  Toplam adım      : {train_result.global_step}")
    print(f"  Eğitim kaybı     : {train_result.training_loss:.4f}")
    print(f"  Süre (sn)        : {train_result.metrics['train_runtime']:.1f}")

    # ── 7. Test Değerlendirmesi ───────────────────────────────
    print("\n[EVAL] Test seti değerlendiriliyor...")
    test_results = trainer.predict(tokenized["test"])
    test_preds = np.argmax(test_results.predictions, axis=-1)
    test_labels = test_results.label_ids

    print("\n[REPORT] Classification Report:")
    print(classification_report(
        test_labels, test_preds,
        target_names=emotion_names,
        digits=4,
    ))

    # ── 8. Confusion Matrix ───────────────────────────────────
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        test_labels, test_preds,
        emotion_names,
        save_path=logs_dir / "confusion_matrix.png",
    )

    # ── 9. Modeli Kaydet ──────────────────────────────────────
    final_dir = ROOT / "models" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n[SAVE] Final model kaydedildi: {final_dir}")

    # Test metrikleri JSON'a yaz
    metrics_path = logs_dir / "test_metrics.json"
    report = classification_report(
        test_labels, test_preds,
        target_names=emotion_names,
        output_dict=True,
    )
    test_metrics = {
        "accuracy": float(report["accuracy"]),
        "macro_avg": {k: float(v) for k, v in report["macro avg"].items()},
        "weighted_avg": {k: float(v) for k, v in report["weighted avg"].items()},
        "per_class": {
            name: {k: float(v) for k, v in report[name].items()}
            for name in emotion_names
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    return trainer, final_dir


if __name__ == "__main__":
    train()
