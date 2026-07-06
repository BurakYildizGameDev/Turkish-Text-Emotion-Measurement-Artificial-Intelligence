"""
EXP3 Multi-Seed Stability Evaluation  (hizlandirilmis versiyon)
================================================================
3 seed [42, 123, 7] ile EXP3 mimarisi egitilir, group-aware test seti
uzerinde degerlendirilir.

Orijinal EXP3 hiperparametreleri korunur, sadece iki hiz optimizasyonu:
  1. max_per_class=30000  → train 385k -> ~100k  (~4x hiz)
  2. Lexicon features TEK SEFERINDE hesaplanir, 3 seed'e paylastirilir

Diger tum parametreler orijinal EXP3 ile ayni:
  fp16=True, bf16=False, eff_batch=128, lr=2e-5, wd=0.01,
  cosine scheduler, warmup=10%, max_length=128, dropout=0.1,
  EarlyStop(patience=2, metric=f1_macro)

Train/Val : data/splits/master_splits/{train,val}.parquet
Test      : data/splits/group_aware_splits/test.parquet
Cikti     : results/group_aware_eval/exp3_multiseed_results.json
"""

import sys, json, shutil, warnings, logging
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
from torch.utils.data import Dataset, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer, TrainingArguments,
    EarlyStoppingCallback, set_seed,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train_exp3_lexicon import (
    LexiconExtractor,
    LexiconBERTModel,
    LexiconDataCollator,
    LexiconWeightedTrainer,
    build_compute_metrics,
    compute_class_weights,
    compute_sample_weights,
    ID2LABEL,
    NUM_LABELS,
    MODEL_NAME,
    LEXICON_DIM,
)

# ── Sabitler ─────────────────────────────────────────────────────────────────

SEEDS         = [42, 123, 7]
MAX_PER_CLASS = 30_000      # hiz icin: 385k -> ~100k
MAX_LENGTH    = 128          # orijinal EXP3 ile ayni
BATCH_SIZE    = 64
GRAD_ACCUM    = 2
EVAL_BATCH    = 128
NUM_EPOCHS    = 2
LR            = 2e-5
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1
DROPOUT       = 0.1
FP16          = True
BF16          = False

TRAIN_PATH = ROOT / "data" / "splits" / "master_splits" / "train.parquet"
VAL_PATH   = ROOT / "data" / "splits" / "master_splits" / "val.parquet"
TEST_PATH  = ROOT / "data" / "splits" / "group_aware_splits" / "test.parquet"
OUT_DIR    = ROOT / "results" / "group_aware_eval"
OUT_JSON   = OUT_DIR / "exp3_multiseed_results.json"


# ── Dataset (on-hesaplanmis lex feats kabul eder) ────────────────────────────

class CachedLexDataset(Dataset):
    """LexiconEmotionDataset ile ayni, ama lex_feats onceden verilir (hiz)."""

    def __init__(self, df: pd.DataFrame, tokenizer, lex_feats: np.ndarray,
                 max_length: int = 128):
        self.texts     = df["text"].tolist()
        self.labels    = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lex_feats = lex_feats  # (N, 10) — dis'aridan verilir

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc  = self.tokenizer(
            self.texts[idx], max_length=self.max_length,
            truncation=True, padding=False,
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"]           = torch.tensor(self.labels[idx], dtype=torch.long)
        item["lexicon_features"] = torch.tensor(self.lex_feats[idx], dtype=torch.float32)
        return item


# ── Veri & lexikon on-hesaplama ──────────────────────────────────────────────

def cap_per_class(df: pd.DataFrame, max_n: int, seed: int) -> pd.DataFrame:
    parts = []
    for lid in range(NUM_LABELS):
        s = df[df["label_id"] == lid]
        if len(s) > max_n:
            s = s.sample(max_n, random_state=seed)
        parts.append(s)
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def load_and_prepare():
    df_train_full = pd.read_parquet(TRAIN_PATH)
    df_val        = pd.read_parquet(VAL_PATH)
    df_test       = pd.read_parquet(TEST_PATH)

    for df in [df_train_full, df_val, df_test]:
        df["label_id"] = df["label_id"].astype(int)

    # Seed-bagimsiz cap (seed=42 sabit — splitting degil sadece sampling)
    df_train = cap_per_class(df_train_full, MAX_PER_CLASS, seed=42)

    print(f"[DATA] Train: {len(df_train):,}  Val: {len(df_val):,}  "
          f"Test (group-aware): {len(df_test):,}")
    print("[DATA] Train sinif dagilimi:")
    for lid in range(NUM_LABELS):
        n = (df_train["label_id"] == lid).sum()
        print(f"         {ID2LABEL[lid]:12s}: {n:6,d}")

    # Lexikon features TEK SEFERINDE hesapla
    print("\n[LEXICON] Features hesaplaniyor (tek sefer)...")
    extractor = LexiconExtractor()

    print(f"  Train  ({len(df_train):,})...", flush=True)
    lex_train = extractor.batch_extract(df_train["text"].tolist(), verbose=True)
    print(f"  Val    ({len(df_val):,})...",   flush=True)
    lex_val   = extractor.batch_extract(df_val["text"].tolist(),   verbose=True)
    print(f"  Test   ({len(df_test):,})...",  flush=True)
    lex_test  = extractor.batch_extract(df_test["text"].tolist(),  verbose=True)
    print("[LEXICON] Tamamlandi.")

    return df_train, df_val, df_test, lex_train, lex_val, lex_test


# ── Tek seed egitimi ─────────────────────────────────────────────────────────

def train_one_seed(df_train, df_val, df_test,
                   lex_train, lex_val, lex_test,
                   tokenizer, seed):
    print(f"\n{'='*60}")
    print(f"  SEED {seed}")
    print(f"{'='*60}")
    set_seed(seed)

    train_ds = CachedLexDataset(df_train, tokenizer, lex_train, MAX_LENGTH)
    val_ds   = CachedLexDataset(df_val,   tokenizer, lex_val,   MAX_LENGTH)
    test_ds  = CachedLexDataset(df_test,  tokenizer, lex_test,  MAX_LENGTH)

    train_labels   = df_train["label_id"].values
    sample_weights = compute_sample_weights(train_labels)
    class_weights  = compute_class_weights(train_labels)

    model = LexiconBERTModel(
        bert_model_name=MODEL_NAME,
        num_labels=NUM_LABELS,
        lexicon_dim=LEXICON_DIM,
        dropout=DROPOUT,
    )

    steps_per_epoch = len(df_train) // (BATCH_SIZE * GRAD_ACCUM)
    warmup_steps    = int(WARMUP_RATIO * steps_per_epoch * NUM_EPOCHS)
    ckpt_dir        = OUT_DIR / f"tmp_ckpt_seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        fp16=FP16, bf16=BF16,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        logging_first_step=True,
        report_to=[],
        seed=seed,
        remove_unused_columns=False,
    )

    trainer = LexiconWeightedTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=LexiconDataCollator(tokenizer=tokenizer),
        compute_metrics=build_compute_metrics(),
        train_sample_weights=sample_weights,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"[TRAIN] steps/epoch={steps_per_epoch}, warmup={warmup_steps}, "
          f"eff_batch={BATCH_SIZE*GRAD_ACCUM}")
    train_result = trainer.train()
    print(f"[TRAIN] Done — steps={train_result.global_step}, "
          f"loss={train_result.training_loss:.4f}, "
          f"time={train_result.metrics['train_runtime']/60:.1f}min")

    print("[EVAL] Group-aware test seti degerlendiriliyor...")
    out    = trainer.predict(test_ds)
    y_pred = np.argmax(out.predictions, axis=-1)
    y_true = out.label_ids

    f1_mac = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    f1_wgt = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    acc    = float(accuracy_score(y_true, y_pred))
    pf1    = f1_score(y_true, y_pred, average=None, zero_division=0)

    print(f"[RESULT] seed={seed}  F1-Macro={f1_mac:.4f}  "
          f"F1-Weighted={f1_wgt:.4f}  Accuracy={acc:.4f}")

    result = {
        "seed":           seed,
        "f1_macro":       round(f1_mac, 4),
        "f1_weighted":    round(f1_wgt, 4),
        "accuracy":       round(acc,    4),
        "per_class_f1":   {ID2LABEL[i]: round(float(pf1[i]), 4) for i in range(NUM_LABELS)},
        "train_steps":    int(train_result.global_step),
        "train_time_min": round(train_result.metrics["train_runtime"] / 60, 2),
        "train_loss":     round(float(train_result.training_loss), 4),
        "test_size":      int(len(df_test)),
    }

    del model, trainer
    torch.cuda.empty_cache()
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    return result


# ── Ana akis ─────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n{'='*60}")
    print(f"  EXP3 MULTI-SEED GROUP-AWARE EVALUATION")
    print(f"  Device        : {gpu}")
    print(f"  Seeds         : {SEEDS}")
    print(f"  max_per_class : {MAX_PER_CLASS:,}  (hiz optimizasyonu)")
    print(f"  max_length    : {MAX_LENGTH}  |  eff_batch: {BATCH_SIZE*GRAD_ACCUM}")
    print(f"  lr={LR}  wd={WEIGHT_DECAY}  fp16={FP16}  epochs={NUM_EPOCHS}")
    print(f"{'='*60}")

    df_train, df_val, df_test, lex_train, lex_val, lex_test = load_and_prepare()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    seed_results = []
    for seed in SEEDS:
        r = train_one_seed(
            df_train, df_val, df_test,
            lex_train, lex_val, lex_test,
            tokenizer, seed,
        )
        seed_results.append(r)

    f1m  = [r["f1_macro"]    for r in seed_results]
    f1w  = [r["f1_weighted"] for r in seed_results]
    accs = [r["accuracy"]    for r in seed_results]

    summary = {
        "experiment":       "exp3_multiseed_group_aware",
        "model":            MODEL_NAME,
        "architecture":     "BERTurk[CLS](768) ++ Lexikon(10) -> Linear(778,10)",
        "test_set":         "data/splits/group_aware_splits/test.parquet",
        "seeds":            SEEDS,
        "max_per_class":    MAX_PER_CLASS,
        "hyperparameters": {
            "num_epochs": NUM_EPOCHS, "per_device_train_batch": BATCH_SIZE,
            "gradient_accumulation": GRAD_ACCUM, "effective_batch": BATCH_SIZE*GRAD_ACCUM,
            "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO, "lr_scheduler": "cosine",
            "max_grad_norm": 1.0, "max_length": MAX_LENGTH,
            "dropout": DROPOUT, "fp16": FP16, "bf16": BF16,
            "early_stopping_patience": 2, "metric_for_best_model": "f1_macro",
        },
        "per_seed":          seed_results,
        "f1_macro_mean":     round(float(np.mean(f1m)),  4),
        "f1_macro_std":      round(float(np.std(f1m)),   4),
        "f1_weighted_mean":  round(float(np.mean(f1w)),  4),
        "f1_weighted_std":   round(float(np.std(f1w)),   4),
        "accuracy_mean":     round(float(np.mean(accs)), 4),
        "accuracy_std":      round(float(np.std(accs)),  4),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  SONUCLAR — EXP3 Multi-Seed (Group-Aware Test)")
    print(f"{'='*60}")
    print(f"  {'Seed':<8} {'F1-Macro':<12} {'F1-Weighted':<14} {'Accuracy'}")
    print(f"  {'-'*50}")
    for r in seed_results:
        print(f"  {r['seed']:<8} {r['f1_macro']:<12.4f} {r['f1_weighted']:<14.4f} {r['accuracy']:.4f}")
    print(f"  {'-'*50}")
    print(f"  {'Mean':<8} {summary['f1_macro_mean']:<12.4f} "
          f"{summary['f1_weighted_mean']:<14.4f} {summary['accuracy_mean']:.4f}")
    print(f"  {'±Std':<8} {summary['f1_macro_std']:<12.4f} "
          f"{summary['f1_weighted_std']:<14.4f} {summary['accuracy_std']:.4f}")
    print(f"\n[KAYDEDILDI] {OUT_JSON}")


if __name__ == "__main__":
    main()
