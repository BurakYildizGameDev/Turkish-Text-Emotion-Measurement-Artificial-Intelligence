"""
run_deney123.py
===============
Uc deneyi sirayla calistirir:
  DENEY 1 - EXP2 Dogru Repeated Holdout   (seed 42, 123, 7)
  DENEY 2 - EXP4 Robustness               (seed 42, 123)
  DENEY 3 - GoEmotions TR Kaynak Disi Test

GENEL PARAMETRELER:
  max_length=64, batch_size=64, max_per_class=30000
  epochs=2, bf16=True, lr=2e-5, wd=0.01, warmup_ratio=0.1
"""

import sys, json, warnings, logging
import numpy as np
import pandas as pd
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    DataCollatorWithPadding,
    set_seed,
)
import evaluate

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train_exp2_crosslingual import (
    CrossLingualEnhancer,
    EmotionDataset,
    WeightedTrainer,
    compute_class_weights,
    compute_sample_weights,
    EMOTIONS,
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS,
)
from train_exp3_lexicon import (
    LexiconExtractor,
    LexiconBERTModel,
    LexiconEmotionDataset,
    LexiconDataCollator,
    LexiconWeightedTrainer,
    build_compute_metrics as lex_build_metrics,
    LEXICON_DIM,
)

# ── Sabitler ────────────────────────────────────────────────────────────────
MODEL_NAME    = "dbmdz/bert-base-turkish-cased"
MAX_LENGTH    = 64
BATCH_SIZE    = 64
EVAL_BATCH    = 128
MAX_PER_CLASS = 30000
EPOCHS        = 2
LR            = 2e-5
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1
BF16          = True
FP16          = False

DATA_PATH = ROOT / "data" / "processed" / "combined_clean.parquet"
OUT_ROOT  = ROOT / "results"


# ─── Veri ───────────────────────────────────────────────────────────────────

def load_df():
    df = pd.read_parquet(DATA_PATH)
    if "label_id" not in df.columns:
        col = "label_norm" if "label_norm" in df.columns else "label"
        df["label_id"] = df[col].map(LABEL2ID)
    if "label" not in df.columns:
        df["label"] = df["label_id"].map(ID2LABEL)
    if "source" not in df.columns:
        df["source"] = "unknown"
    df = df.dropna(subset=["text", "label_id"])
    df["label_id"] = df["label_id"].astype(int)
    df = df[df["label_id"].between(0, NUM_LABELS - 1)]
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 5].reset_index(drop=True)
    print(f"[DATA] {len(df):,} ornek, {df['source'].nunique()} kaynak")
    return df


def make_fresh_splits(df, seed, test_size=0.15, val_size=0.15):
    df_tv, df_test = train_test_split(
        df, test_size=test_size, stratify=df["label_id"], random_state=seed,
    )
    val_r = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_tv, test_size=val_r, stratify=df_tv["label_id"], random_state=seed,
    )
    print(f"[SPLIT] seed={seed}  Train:{len(df_train):,}  Val:{len(df_val):,}  Test:{len(df_test):,}")
    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


def cap_per_class(df, max_n, seed):
    parts = []
    for lid in range(NUM_LABELS):
        s = df[df["label_id"] == lid]
        if len(s) > max_n:
            s = s.sample(max_n, random_state=seed)
        parts.append(s)
    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


# ─── Epoch Logger ───────────────────────────────────────────────────────────

class SimpleLogger(TrainerCallback):
    def __init__(self):
        self.history = {"epoch": [], "val_loss": [], "val_f1_macro": []}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.history["epoch"].append(round(state.epoch, 1))
            self.history["val_loss"].append(metrics.get("eval_loss"))
            self.history["val_f1_macro"].append(metrics.get("eval_f1_macro"))


# ─── Metrik Fonksiyonu ──────────────────────────────────────────────────────

def make_compute_metrics():
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


# ─── EXP2 Egitim ────────────────────────────────────────────────────────────

def train_exp2_run(df_train, df_val, df_test, seed, output_dir, apply_cl=True):
    """EXP2: BERTurk + Cross-Lingual, sifirdan egit ve test et."""
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    if apply_cl:
        enhancer = CrossLingualEnhancer(seed=seed)
        df_train = enhancer.enhance(df_train)
        print(f"  Train sonrasi CL: {len(df_train):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds  = EmotionDataset(df_train, tokenizer, max_length=MAX_LENGTH)
    val_ds    = EmotionDataset(df_val,   tokenizer, max_length=MAX_LENGTH)
    test_ds   = EmotionDataset(df_test,  tokenizer, max_length=MAX_LENGTH)

    sample_w = compute_sample_weights(df_train)
    class_w  = compute_class_weights(df_train["label_id"].values)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS,
        id2label=ID2LABEL, label2id=LABEL2ID,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        ignore_mismatched_sizes=True,
    )

    steps_ep = max(1, len(df_train) // BATCH_SIZE)
    warmup_s = int(WARMUP_RATIO * steps_ep * EPOCHS)

    args = TrainingArguments(
        output_dir=str(output_dir / "ckpts"),
        fp16=FP16, bf16=BF16,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH,
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_s,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=200,
        report_to=[],
        seed=seed,
    )

    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=make_compute_metrics(),
        train_sample_weights=sample_w,
        class_weights=class_w,
        callbacks=[SimpleLogger(), EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"  Egitim basliyor (steps/epoch={steps_ep}, warmup={warmup_s})...")
    trainer.train()

    out    = trainer.predict(test_ds)
    y_pred = np.argmax(out.predictions, axis=-1)
    y_true = out.label_ids
    pf1    = f1_score(y_true, y_pred, average=None, zero_division=0)

    res = {
        "seed":        seed,
        "f1_macro":    round(float(f1_score(y_true, y_pred, average="macro",    zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "accuracy":    round(float(accuracy_score(y_true, y_pred)), 4),
        "per_class_f1": {EMOTIONS[i]: round(float(pf1[i]), 4) for i in range(NUM_LABELS)},
        "train_size": int(len(df_train)),
        "val_size":   int(len(df_val)),
        "test_size":  int(len(df_test)),
    }
    with open(output_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"  F1-Macro:{res['f1_macro']:.4f}  F1-W:{res['f1_weighted']:.4f}  Acc:{res['accuracy']:.4f}")
    del model, trainer
    torch.cuda.empty_cache()
    return res


# ─── EXP4 Egitim ────────────────────────────────────────────────────────────

def train_exp4_run(df_train, df_val, df_test, seed, output_dir):
    """EXP4: BERTurk + Lexicon (NRC categories) + Cross-Lingual, sifirdan egit ve test et."""
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    enhancer = CrossLingualEnhancer(seed=seed)
    df_train = enhancer.enhance(df_train)
    print(f"  Train sonrasi CL: {len(df_train):,}")

    extractor = LexiconExtractor()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = LexiconEmotionDataset(df_train, tokenizer, extractor, max_length=MAX_LENGTH)
    val_ds   = LexiconEmotionDataset(df_val,   tokenizer, extractor, max_length=MAX_LENGTH)
    test_ds  = LexiconEmotionDataset(df_test,  tokenizer, extractor, max_length=MAX_LENGTH)

    sample_w = compute_sample_weights(df_train)
    class_w  = compute_class_weights(df_train["label_id"].values)

    model = LexiconBERTModel(
        bert_model_name=MODEL_NAME,
        num_labels=NUM_LABELS,
        lexicon_dim=LEXICON_DIM,
        dropout=0.1,
    )

    steps_ep = max(1, len(df_train) // BATCH_SIZE)
    warmup_s = int(WARMUP_RATIO * steps_ep * EPOCHS)

    args = TrainingArguments(
        output_dir=str(output_dir / "ckpts"),
        fp16=FP16, bf16=BF16,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH,
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_s,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=200,
        report_to=[],
        seed=seed,
        remove_unused_columns=False,
    )

    trainer = LexiconWeightedTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=LexiconDataCollator(tokenizer),
        compute_metrics=lex_build_metrics(),
        train_sample_weights=sample_w,
        class_weights=class_w,
        callbacks=[SimpleLogger(), EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"  Egitim basliyor (steps/epoch={steps_ep}, warmup={warmup_s})...")
    trainer.train()

    out    = trainer.predict(test_ds)
    y_pred = np.argmax(out.predictions, axis=-1)
    y_true = out.label_ids
    pf1    = f1_score(y_true, y_pred, average=None, zero_division=0)

    res = {
        "seed":        seed,
        "f1_macro":    round(float(f1_score(y_true, y_pred, average="macro",    zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "accuracy":    round(float(accuracy_score(y_true, y_pred)), 4),
        "per_class_f1": {EMOTIONS[i]: round(float(pf1[i]), 4) for i in range(NUM_LABELS)},
        "train_size": int(len(df_train)),
        "val_size":   int(len(df_val)),
        "test_size":  int(len(df_test)),
    }
    with open(output_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"  F1-Macro:{res['f1_macro']:.4f}  F1-W:{res['f1_weighted']:.4f}  Acc:{res['accuracy']:.4f}")
    del model, trainer
    torch.cuda.empty_cache()
    return res


# ─── Ozet ───────────────────────────────────────────────────────────────────

def compute_summary(per_seed_results, out_file):
    f1m  = [r["f1_macro"]    for r in per_seed_results]
    f1w  = [r["f1_weighted"] for r in per_seed_results]
    accs = [r["accuracy"]    for r in per_seed_results]
    s = {
        "per_seed":         per_seed_results,
        "f1_macro_mean":    round(float(np.mean(f1m)),  4),
        "f1_macro_std":     round(float(np.std(f1m)),   4),
        "f1_weighted_mean": round(float(np.mean(f1w)),  4),
        "f1_weighted_std":  round(float(np.std(f1w)),   4),
        "accuracy_mean":    round(float(np.mean(accs)), 4),
        "accuracy_std":     round(float(np.std(accs)),  4),
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    print(f"[OZET] Kaydedildi: {out_file}")
    return s


# ═══════════════════════════════════════════════════════════════════════════
# DENEY 1
# ═══════════════════════════════════════════════════════════════════════════

def run_deney1(df_full):
    print("\n" + "="*60)
    print("  DENEY 1 - EXP2 Dogru Repeated Holdout  [seeds: 42, 123, 7]")
    print("="*60)

    seeds    = [42, 123, 7]
    out_base = OUT_ROOT / "repeated_holdout" / "exp2_correct"
    results  = []

    for seed in seeds:
        print(f"\n{'─'*50}")
        print(f"  Seed {seed}")
        print(f"{'─'*50}")
        df_train, df_val, df_test = make_fresh_splits(df_full, seed)
        df_train = cap_per_class(df_train, MAX_PER_CLASS,      seed)
        df_val   = cap_per_class(df_val,   MAX_PER_CLASS // 5, seed)
        print(f"  Train (capped):{len(df_train):,}  Val:{len(df_val):,}")
        r = train_exp2_run(df_train, df_val, df_test, seed, out_base / f"seed_{seed}")
        results.append(r)

    s = compute_summary(results, out_base / "summary.json")
    print(f"\n[DENEY 1] F1-Macro:{s['f1_macro_mean']:.4f}+/-{s['f1_macro_std']:.4f}"
          f"  Acc:{s['accuracy_mean']:.4f}+/-{s['accuracy_std']:.4f}")
    return s


# ═══════════════════════════════════════════════════════════════════════════
# DENEY 2
# ═══════════════════════════════════════════════════════════════════════════

def run_deney2(df_full):
    print("\n" + "="*60)
    print("  DENEY 2 - EXP4 Robustness  [seeds: 42, 123]")
    print("="*60)

    seeds    = [42, 123]
    out_base = OUT_ROOT / "repeated_holdout" / "exp4_robustness"
    results  = []

    for seed in seeds:
        print(f"\n{'─'*50}")
        print(f"  Seed {seed}")
        print(f"{'─'*50}")
        df_train, df_val, df_test = make_fresh_splits(df_full, seed)
        df_train = cap_per_class(df_train, MAX_PER_CLASS,      seed)
        df_val   = cap_per_class(df_val,   MAX_PER_CLASS // 5, seed)
        print(f"  Train (capped):{len(df_train):,}  Val:{len(df_val):,}")
        r = train_exp4_run(df_train, df_val, df_test, seed, out_base / f"seed_{seed}")
        results.append(r)

    s = compute_summary(results, out_base / "summary.json")
    print(f"\n[DENEY 2] F1-Macro:{s['f1_macro_mean']:.4f}+/-{s['f1_macro_std']:.4f}"
          f"  Acc:{s['accuracy_mean']:.4f}+/-{s['accuracy_std']:.4f}")
    return s


# ═══════════════════════════════════════════════════════════════════════════
# DENEY 3
# ═══════════════════════════════════════════════════════════════════════════

def run_deney3(df_full):
    print("\n" + "="*60)
    print("  DENEY 3 - GoEmotions TR Kaynak Disi Test")
    print("  Train: winvoker + diger (GoEmotions haric)")
    print("  Test : Sadece GoEmotions TR ornekleri")
    print("="*60)

    out_dir = OUT_ROOT / "source_out_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_go_test = df_full[df_full["source"] == "goemotions_tr"].reset_index(drop=True)
    df_other   = df_full[df_full["source"] != "goemotions_tr"].reset_index(drop=True)

    print(f"\n  GoEmotions TR test seti : {len(df_go_test):,} ornek")
    print(f"  Diger kaynaklar         : {len(df_other):,} ornek")
    print(f"  Kaynak dagilimi         : {df_other['source'].value_counts().to_dict()}")
    print(f"  Test label dagilimi     : {df_go_test['label'].value_counts().to_dict()}")

    # Non-goemotions icin train/val split
    SEED = 42
    df_tv, df_val_raw = train_test_split(
        df_other, test_size=0.15, stratify=df_other["label_id"], random_state=SEED,
    )
    df_train = cap_per_class(df_tv,      MAX_PER_CLASS,      SEED)
    df_val   = cap_per_class(df_val_raw, MAX_PER_CLASS // 5, SEED)
    print(f"\n  Train (capped):{len(df_train):,}  Val:{len(df_val):,}")
    print(f"  Train label dagilimi: {df_train['label'].value_counts().to_dict()}")

    # CrossLingualEnhancer goemotions_tr bulmaz, azinlik siniflari atlar (beklenen davranis)
    r = train_exp2_run(df_train, df_val, df_go_test, SEED, out_dir, apply_cl=True)

    final = {
        "experiment":    "GoEmotions TR Kaynak Disi Test",
        "train_sources": df_other["source"].unique().tolist(),
        "test_source":   "goemotions_tr",
        "train_size":    r["train_size"],
        "test_size":     r["test_size"],
        "f1_macro":      r["f1_macro"],
        "f1_weighted":   r["f1_weighted"],
        "accuracy":      r["accuracy"],
        "per_class_f1":  r["per_class_f1"],
    }
    out_file = out_dir / "goemotions_holdout.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"\n[DENEY 3] F1-Macro:{r['f1_macro']:.4f}  F1-W:{r['f1_weighted']:.4f}  Acc:{r['accuracy']:.4f}")
    print(f"  Kaydedildi: {out_file}")
    return final


# ─── Final Ozet ─────────────────────────────────────────────────────────────

def write_final_summary(d1, d2, d3):
    sd1 = {r["seed"]: r for r in d1["per_seed"]}
    sd2 = {r["seed"]: r for r in d2["per_seed"]}

    lines = [
        "=" * 50,
        "DENEY 1 - EXP2 Dogru Repeated Holdout",
        "=" * 50,
    ]
    for s in [42, 123, 7]:
        r = sd1.get(s, {})
        lines.append(
            f"Seed {s:3d}:  F1-Macro: {r.get('f1_macro', 0):.4f} | "
            f"F1-Weighted: {r.get('f1_weighted', 0):.4f} | "
            f"Accuracy: {r.get('accuracy', 0):.4f}"
        )
    lines.append(
        f"ORTALAMA:  F1-Macro: {d1['f1_macro_mean']:.4f} +/- {d1['f1_macro_std']:.4f} | "
        f"Accuracy: {d1['accuracy_mean']:.4f} +/- {d1['accuracy_std']:.4f}"
    )
    lines += [
        "",
        "=" * 50,
        "DENEY 2 - EXP4 Robustness (2 seed)",
        "=" * 50,
    ]
    for s in [42, 123]:
        r = sd2.get(s, {})
        lines.append(
            f"Seed {s:3d}:  F1-Macro: {r.get('f1_macro', 0):.4f} | "
            f"Accuracy: {r.get('accuracy', 0):.4f}"
        )
    lines.append(
        f"ORTALAMA:  F1-Macro: {d2['f1_macro_mean']:.4f} +/- {d2['f1_macro_std']:.4f}"
    )
    lines += [
        "",
        "=" * 50,
        "DENEY 3 - Kaynak Disi Test",
        "=" * 50,
        f"Train kaynaklari: {d3.get('train_sources', [])}",
        f"Test kaynagi    : GoEmotions TR (hic gorulmemis)",
        f"F1-Macro   : {d3['f1_macro']:.4f}",
        f"F1-Weighted: {d3['f1_weighted']:.4f}",
        f"Accuracy   : {d3['accuracy']:.4f}",
        f"Train ornegi: {d3['train_size']:,} | Test ornegi: {d3['test_size']:,}",
        "=" * 50,
    ]

    text = "\n".join(lines)
    out  = OUT_ROOT / "final_experiment_summary.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + "="*60)
    print("  FINAL OZET")
    print("="*60)
    print(text)
    print(f"\n[KAYIT] {out}")


# ═══════════════════════════════════════════════════════════════════════════
# ANA AKIS
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    t0 = time.time()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("\n" + "="*60)
    print("  MASTER DENEY RUNNER")
    print(f"  Cihaz      : {gpu_name}")
    print(f"  max_length : {MAX_LENGTH}  batch : {BATCH_SIZE}  epochs : {EPOCHS}")
    print(f"  max/class  : {MAX_PER_CLASS}  bf16 : {BF16}  lr : {LR}")
    print("="*60)

    df_full   = load_df()

    d1_sum    = run_deney1(df_full)
    d2_sum    = run_deney2(df_full)
    d3_result = run_deney3(df_full)

    write_final_summary(d1_sum, d2_sum, d3_result)

    total_min = (time.time() - t0) / 60
    print(f"\n[BITTI] Toplam sure: {total_min:.1f} dk")
