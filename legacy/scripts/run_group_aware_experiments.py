"""
Group-Aware Split üzerinde EXP2 ve EXP4 Eğitimi
================================================
data/splits/group_aware_splits/ kullanır.

Parametreler:
  max_length     : 64
  batch_size     : 64
  max_per_class  : 30000
  epochs         : 2
  bf16           : True
  lr             : 2e-5
  seed           : 42

Çıktılar:
  results/group_aware_exp2_results.json
  results/group_aware_exp4_results.json
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
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from pathlib import Path
from sklearn.metrics import classification_report, f1_score, accuracy_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)
import evaluate

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

ROOT         = Path(__file__).resolve().parent
GA_SPLITS    = ROOT / "data" / "splits" / "group_aware_splits"
RESULTS_DIR  = ROOT / "results"

OUT_EXP2 = RESULTS_DIR / "group_aware_exp2_results.json"
OUT_EXP4 = RESULTS_DIR / "group_aware_exp4_results.json"

# ── Sabit parametreler ──────────────────────────────────────────────────────

MAX_LENGTH    = 64
BATCH_SIZE    = 64
MAX_PER_CLASS = 30000
EPOCHS        = 2
LR            = 2e-5
SEED          = 42

EMOTIONS = [
    "mutluluk", "üzüntü", "öfke", "korku",
    "şaşkınlık", "tiksinti", "sevgi", "nötr", "utanç", "gurur",
]
LABEL2ID   = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL   = {i: e for i, e in enumerate(EMOTIONS)}
NUM_LABELS = 10
MODEL_NAME = "dbmdz/bert-base-turkish-cased"

OVERSAMPLE_TARGETS = {
    "gurur":     3_500,
    "utanç":     4_000,
    "tiksinti":  4_000,
    "korku":     4_000,
    "şaşkınlık": 6_000,
    "öfke":      6_000,
}
CL_WEIGHT_MULTIPLIER = 2.5
CL_SOURCE            = "goemotions_tr"


# ── Split yükleme ───────────────────────────────────────────────────────────

def load_group_aware_splits():
    df_train = pd.read_parquet(GA_SPLITS / "train.parquet")
    df_val   = pd.read_parquet(GA_SPLITS / "val.parquet")
    df_test  = pd.read_parquet(GA_SPLITS / "test.parquet")

    for df in [df_train, df_val, df_test]:
        if "label" not in df.columns:
            df["label"] = df["label_id"].map(ID2LABEL)
        if "source" not in df.columns:
            df["source"] = "unknown"
        if "label_id" not in df.columns:
            df["label_id"] = df["label"].map(LABEL2ID)
        df["label_id"] = df["label_id"].astype(int)

    print(f"[SPLIT] Train:{len(df_train):,}  Val:{len(df_val):,}  Test:{len(df_test):,}")
    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


# ── Cross-Lingual Enhancer (train_exp2'den aynı mantık) ─────────────────────

def cross_lingual_enhance(df_train: pd.DataFrame, rng=None) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng(SEED)

    print(f"\n[CL-ENHANCE] Cross-Lingual oversample...")
    extra_frames = []
    for emo, target in OVERSAMPLE_TARGETS.items():
        label_id = LABEL2ID[emo]
        cl_mask   = (df_train["label_id"] == label_id) & (df_train["source"] == CL_SOURCE)
        cl_examples = df_train[cl_mask]
        current_total = (df_train["label_id"] == label_id).sum()
        current_cl    = len(cl_examples)

        if current_total >= target:
            print(f"  {emo:12s}: zaten {current_total:,} >= hedef {target:,}, atlandı")
            continue
        if current_cl == 0:
            print(f"  {emo:12s}: cross-lingual örnek yok, atlandı")
            continue

        needed = target - current_total
        idx    = rng.integers(0, current_cl, size=needed)
        extra  = cl_examples.iloc[idx].copy()
        extra["source"] = f"{CL_SOURCE}_aug"
        extra_frames.append(extra)
        print(f"  {emo:12s}: {current_total:,} -> {current_total + needed:,} (+{needed:,})")

    if extra_frames:
        df_aug = pd.concat([df_train] + extra_frames, ignore_index=True)
        df_aug = df_aug.sample(frac=1, random_state=SEED).reset_index(drop=True)
    else:
        df_aug = df_train.copy()

    print(f"  Toplam: {len(df_train):,} -> {len(df_aug):,}\n")
    return df_aug


# ── Ağırlıklar ─────────────────────────────────────────────────────────────

def compute_class_weights(label_ids: np.ndarray) -> torch.Tensor:
    counts = np.bincount(label_ids, minlength=NUM_LABELS).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    return torch.tensor(len(label_ids) / (counts * NUM_LABELS), dtype=torch.float32)


def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    label_ids  = df["label_id"].values
    counts     = np.bincount(label_ids, minlength=NUM_LABELS).astype(float)
    counts     = np.where(counts == 0, 1.0, counts)
    class_w    = len(label_ids) / (counts * NUM_LABELS)
    sample_w   = class_w[label_ids].astype(float)
    minority   = {LABEL2ID[e] for e in ["gurur", "utanç", "tiksinti", "korku"]}
    is_cl      = df["source"].str.startswith(CL_SOURCE).values
    is_min     = np.isin(label_ids, list(minority))
    sample_w[is_cl & is_min] *= CL_WEIGHT_MULTIPLIER
    return sample_w


# ── Metrik fonksiyonu ───────────────────────────────────────────────────────

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


# ── JSON çıktı yardımcısı ───────────────────────────────────────────────────

def build_result_dict(exp_name: str, y_true, y_pred, df_train, df_val, df_test,
                      extra: dict = None) -> dict:
    report = classification_report(
        y_true, y_pred, target_names=EMOTIONS, output_dict=True, zero_division=0,
    )
    per_class = {
        emo: {
            "precision": round(report[emo]["precision"], 4),
            "recall":    round(report[emo]["recall"],    4),
            "f1":        round(report[emo]["f1-score"],  4),
            "support":   int(report[emo]["support"]),
        }
        for emo in EMOTIONS if emo in report
    }
    out = {
        "experiment":       exp_name,
        "splits":           "group_aware_splits",
        "parameters": {
            "max_length":    MAX_LENGTH,
            "batch_size":    BATCH_SIZE,
            "max_per_class": MAX_PER_CLASS,
            "epochs":        EPOCHS,
            "bf16":          True,
            "lr":            LR,
            "seed":          SEED,
        },
        "data": {
            "train": len(df_train),
            "val":   len(df_val),
            "test":  len(df_test),
        },
        "test_accuracy":    round(float(accuracy_score(y_true, y_pred)), 4),
        "test_f1_macro":    round(float(f1_score(y_true, y_pred, average="macro",    zero_division=0)), 4),
        "test_f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "per_class":        per_class,
    }
    if extra:
        out.update(extra)
    return out


def print_per_class(per_class: dict):
    print(f"\n  {'Duygu':12s} {'F1':>8} {'Destek':>8}")
    print("  " + "-"*32)
    for emo in EMOTIONS:
        m = per_class.get(emo, {})
        print(f"  {emo:12s} {m.get('f1', 0.0):8.4f} {m.get('support', 0):8d}")


# ═══════════════════════════════════════════════════════════════════════════
# EXP2 — BERTurk + Cross-Lingual Augmentation
# ═══════════════════════════════════════════════════════════════════════════

from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=MAX_LENGTH):
        self.texts      = df["text"].tolist()
        self.labels     = df["label_id"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc  = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


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


def run_exp2():
    set_seed(SEED)
    out_dir = RESULTS_DIR / "group_aware_exp2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*65)
    print("  EXP2 — BERTURK + CROSS-LINGUAL | GROUP-AWARE SPLITS")
    print("="*65)

    df_train_base, df_val, df_test = load_group_aware_splits()

    # max_per_class cap
    df_train_base = (
        df_train_base.groupby("label_id", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), MAX_PER_CLASS), random_state=42))
        .reset_index(drop=True)
    )
    print(f"[CAP] max_per_class={MAX_PER_CLASS} -> {len(df_train_base):,} örnek")

    df_train = cross_lingual_enhance(df_train_base)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = EmotionDataset(df_train, tokenizer, MAX_LENGTH)
    val_ds   = EmotionDataset(df_val,   tokenizer, MAX_LENGTH)
    test_ds  = EmotionDataset(df_test,  tokenizer, MAX_LENGTH)

    sample_weights = compute_sample_weights(df_train)
    class_weights  = compute_class_weights(df_train["label_id"].values)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        ignore_mismatched_sizes=True,
    )

    steps_per_epoch = len(df_train) // (BATCH_SIZE * 2)
    warmup_steps    = int(0.1 * steps_per_epoch * EPOCHS)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        fp16=False, bf16=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=2,
        gradient_checkpointing=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=100,
        logging_first_step=True,
        report_to=[],
        seed=SEED,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\n[TRAIN] Başlıyor... adım/epoch≈{steps_per_epoch}, warmup={warmup_steps}")
    train_result = trainer.train()
    print(f"[TRAIN] Tamamlandı. Loss={train_result.training_loss:.4f}")

    print("\n[TEST] Test seti değerlendiriliyor...")
    test_out = trainer.predict(test_ds)
    y_pred   = np.argmax(test_out.predictions, axis=-1)
    y_true   = test_out.label_ids

    result = build_result_dict("exp2_group_aware", y_true, y_pred, df_train, df_val, df_test)
    OUT_EXP2.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_EXP2, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVE] {OUT_EXP2}")
    print(f"  F1-Macro  : {result['test_f1_macro']:.4f}")
    print(f"  Accuracy  : {result['test_accuracy']:.4f}")
    print_per_class(result["per_class"])

    # best model kaydet
    best_dir = out_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"[SAVE] Model: {best_dir}")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# EXP4 — BERTurk + Cross-Lingual + Lexicon (NRC categories)
# ═══════════════════════════════════════════════════════════════════════════

def run_exp4():
    # LexiconBERT bileşenlerini import et
    sys.path.insert(0, str(ROOT))
    from train_exp3_lexicon import (
        LexiconExtractor,
        LexiconBERTModel,
        LexiconEmotionDataset,
        LexiconDataCollator,
        LexiconWeightedTrainer,
        build_compute_metrics as build_lex_compute_metrics,
        LEXICON_DIM,
    )

    set_seed(SEED)
    out_dir = RESULTS_DIR / "group_aware_exp4"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*65)
    print("  EXP4 — BERTURK + CROSS-LINGUAL + NRC LEXICON | GROUP-AWARE")
    print("="*65)

    df_train_base, df_val, df_test = load_group_aware_splits()

    # max_per_class cap
    df_train_base = (
        df_train_base.groupby("label_id", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), MAX_PER_CLASS), random_state=42))
        .reset_index(drop=True)
    )
    print(f"[CAP] max_per_class={MAX_PER_CLASS} -> {len(df_train_base):,} örnek")

    df_train = cross_lingual_enhance(df_train_base)

    print("\n[LEXIKON] Extractor başlatılıyor...")
    extractor = LexiconExtractor()
    print(f"  Sözlük: {len(extractor.lexicon)} kelime")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\n[DATASET] Tokenizasyon + lexikon özellik hesaplama...")
    train_ds = LexiconEmotionDataset(df_train, tokenizer, extractor, max_length=MAX_LENGTH)
    val_ds   = LexiconEmotionDataset(df_val,   tokenizer, extractor, max_length=MAX_LENGTH)
    test_ds  = LexiconEmotionDataset(df_test,  tokenizer, extractor, max_length=MAX_LENGTH)

    sample_weights = compute_sample_weights(df_train)
    class_weights  = compute_class_weights(df_train["label_id"].values)

    model = LexiconBERTModel(
        bert_model_name=MODEL_NAME,
        num_labels=NUM_LABELS,
        lexicon_dim=LEXICON_DIM,
        dropout=0.1,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Eğitim parametresi: {trainable:,}")

    steps_per_epoch = len(df_train) // (BATCH_SIZE * 2)
    warmup_steps    = int(0.1 * steps_per_epoch * EPOCHS)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        fp16=False, bf16=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=2,
        gradient_checkpointing=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=100,
        logging_first_step=True,
        report_to=[],
        seed=SEED,
        remove_unused_columns=False,
    )

    data_collator = LexiconDataCollator(tokenizer=tokenizer)
    trainer = LexiconWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_lex_compute_metrics(),
        train_sample_weights=sample_weights,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\n[TRAIN] Başlıyor... adım/epoch≈{steps_per_epoch}, warmup={warmup_steps}")
    train_result = trainer.train()
    print(f"[TRAIN] Tamamlandı. Loss={train_result.training_loss:.4f}")

    print("\n[TEST] Test seti değerlendiriliyor...")
    test_out = trainer.predict(test_ds)
    y_pred   = np.argmax(test_out.predictions, axis=-1)
    y_true   = test_out.label_ids

    result = build_result_dict("exp4_group_aware", y_true, y_pred, df_train, df_val, df_test,
                               extra={"lexicon_dim": LEXICON_DIM})
    OUT_EXP4.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_EXP4, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVE] {OUT_EXP4}")
    print(f"  F1-Macro  : {result['test_f1_macro']:.4f}")
    print(f"  Accuracy  : {result['test_accuracy']:.4f}")
    print_per_class(result["per_class"])

    # model kaydet
    best_dir = out_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), best_dir / "master_hybrid_weights.pt")
    tokenizer.save_pretrained(str(best_dir))
    print(f"[SAVE] Model: {best_dir}")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Karşılaştırma Tablosu
# ═══════════════════════════════════════════════════════════════════════════

def print_comparison_table(exp2_result: dict, exp4_result: dict):
    orig_exp2 = 0.4629
    orig_exp4 = 0.4597

    ga_exp2 = exp2_result["test_f1_macro"]
    ga_exp4 = exp4_result["test_f1_macro"]

    d2 = ga_exp2 - orig_exp2
    d4 = ga_exp4 - orig_exp4

    print("\n" + "="*60)
    print("  KARŞILAŞTIRMA TABLOSU")
    print("="*60)
    print(f"  {'Model':<8} {'Orijinal F1':>13} {'Group-Aware F1':>16} {'Fark':>8}")
    print("  " + "-"*48)
    print(f"  {'EXP2':<8} {orig_exp2:>13.4f} {ga_exp2:>16.4f} {'+' if d2>=0 else ''}{d2:>7.4f}")
    print(f"  {'EXP4':<8} {orig_exp4:>13.4f} {ga_exp4:>16.4f} {'+' if d4>=0 else ''}{d4:>7.4f}")
    print("="*60)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Group-Aware EXP2 + EXP4")
    parser.add_argument("--exp", choices=["exp2", "exp4", "both"], default="both")
    args = parser.parse_args()

    exp2_result = None
    exp4_result = None

    if args.exp in ("exp2", "both"):
        exp2_result = run_exp2()

    if args.exp in ("exp4", "both"):
        exp4_result = run_exp4()

    # Karşılaştırma tablosu
    if exp2_result is None and OUT_EXP2.exists():
        with open(OUT_EXP2, encoding="utf-8") as f:
            exp2_result = json.load(f)
    if exp4_result is None and OUT_EXP4.exists():
        with open(OUT_EXP4, encoding="utf-8") as f:
            exp4_result = json.load(f)

    if exp2_result and exp4_result:
        print_comparison_table(exp2_result, exp4_result)
