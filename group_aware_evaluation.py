"""
Group-Aware Split Evaluation
=============================
7 model — inference only (eğitim yok).
EXP1 için 4-fold group-aware CV (EXP1 ağırlıklarından başlar, 2 epoch fine-tune).

Çıktılar: results/group_aware_eval/
  [model]_results.json   — her model için test sonuçları
  comparison_table.json  — karşılaştırma tablosu
  kfold_results.json     — EXP1 4-fold CV
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
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ─── Sabitler ────────────────────────────────────────────────────────────────

ROOT   = Path(__file__).resolve().parent
SPLITS = ROOT / "data" / "splits" / "group_aware_splits"
OUTPUT = ROOT / "results" / "group_aware_eval"

EMOTIONS = [
    "mutluluk", "üzüntü", "öfke", "korku",
    "şaşkınlık", "tiksinti", "sevgi", "nötr", "utanç", "gurur",
]
LABEL2ID   = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL   = {i: e for i, e in enumerate(EMOTIONS)}
NUM_LABELS = 10

# Her model için yapılandırma
MODELS_CFG = {
    "EXP0": {
        "type":           "standard",
        "path":           "results/exp0/best_model",
        "tokenizer_path": "results/exp0/best_model",
    },
    "EXP1": {
        "type":           "standard",
        "path":           "results/exp1/best_model",
        "tokenizer_path": "results/exp1/best_model",
    },
    "EXP2": {
        "type":           "standard",
        "path":           "results/exp2/best_model",
        "tokenizer_path": "results/exp2/best_model",
    },
    "EXP3": {
        "type":           "lexicon",
        "weights":        "results/exp3/best_model/lexicon_bert_weights.pt",
        "tokenizer_path": "results/exp3/best_model",
        "bert_model":     "dbmdz/bert-base-turkish-cased",
    },
    "EXP4": {
        "type":           "lexicon",
        "weights":        "results/exp4/best_model/master_hybrid_weights.pt",
        "tokenizer_path": "results/exp4/best_model",
        "bert_model":     "dbmdz/bert-base-turkish-cased",
    },
    "mBERT": {
        "type":           "standard",
        "path":           "results/baselines/mbert/checkpoint-12036",
        "tokenizer_path": "google-bert/bert-base-multilingual-cased",
    },
    "XLM-RoBERTa": {
        "type":           "standard",
        "path":           "results/baselines/xlm_roberta/checkpoint-7398",
        "tokenizer_path": "xlm-roberta-base",
    },
}

# ─── EXP3/EXP4 Model Sınıfları (train_exp3_lexicon'dan import) ───────────────

from transformers import BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

# Orijinal lexikonla doğru çıkarım için train_exp3_lexicon'dan import et
try:
    from train_exp3_lexicon import (
        LexiconBERTModel,
        LexiconExtractor,
        LexiconEmotionDataset,
        LexiconDataCollator,
        TURKISH_LEXICON,
        LEXICON_DIM,
    )
    print(f"[OK] train_exp3_lexicon import edildi ({len(TURKISH_LEXICON)} kelime)")
except Exception as _import_err:
    print(f"[UYARI] train_exp3_lexicon import hatası: {_import_err}")
    print("        Yerleşik basit lexikon kullanılacak.")

    import re
    LEXICON_DIM = 10

    class LexiconExtractor:
        def __init__(self, lexicon=None):
            self._lexicon = lexicon or {}
            self._punct_re = re.compile(r"[^\w\s]", re.UNICODE)

        def extract(self, text):
            cleaned = self._punct_re.sub(" ", text.lower()).split()
            vec = np.zeros(LEXICON_DIM, dtype=np.float32)
            hits = 0
            for token in cleaned:
                if token in self._lexicon:
                    vec += self._lexicon[token]
                    hits += 1
            if hits > 0:
                vec /= hits
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
            return vec

        def batch_extract(self, texts, verbose=False):
            return [self.extract(t) for t in texts]

    class LexiconBERTModel(nn.Module):
        def __init__(self, bert_model_name="dbmdz/bert-base-turkish-cased",
                     num_labels=NUM_LABELS, lexicon_dim=LEXICON_DIM, dropout=0.1):
            super().__init__()
            self.num_labels  = num_labels
            self.lexicon_dim = lexicon_dim
            self.bert    = BertModel.from_pretrained(bert_model_name)
            hidden_size  = self.bert.config.hidden_size
            combined_dim = hidden_size + lexicon_dim
            self.dropout    = nn.Dropout(dropout)
            self.classifier = nn.Linear(combined_dim, num_labels)
            self.config = self.bert.config
            self.config.num_labels = num_labels

        def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                    lexicon_features=None, labels=None, **kwargs):
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
            cls_vec = self.dropout(bert_out.last_hidden_state[:, 0, :])
            if lexicon_features is not None:
                lex = lexicon_features.to(cls_vec.device).float()
                combined = torch.cat([cls_vec, lex], dim=-1)
            else:
                zero_lex = torch.zeros(cls_vec.size(0), self.lexicon_dim, device=cls_vec.device)
                combined = torch.cat([cls_vec, zero_lex], dim=-1)
            logits = self.classifier(combined)
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits, labels)
            return SequenceClassifierOutput(loss=loss, logits=logits)

    class LexiconEmotionDataset(Dataset):
        def __init__(self, df, tokenizer, extractor, max_length=128):
            self.texts     = df["text"].tolist()
            self.labels    = df["label_id"].tolist()
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.lex_feats = extractor.batch_extract(self.texts)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(self.texts[idx], max_length=self.max_length,
                                 truncation=True, padding=False)
            item = {k: torch.tensor(v) for k, v in enc.items()}
            item["labels"]           = torch.tensor(self.labels[idx], dtype=torch.long)
            item["lexicon_features"] = torch.tensor(self.lex_feats[idx], dtype=torch.float32)
            return item

    class LexiconDataCollator(DataCollatorWithPadding):
        def __call__(self, features):
            lex_feats = torch.stack([f.pop("lexicon_features") for f in features])
            batch = super().__call__(features)
            batch["lexicon_features"] = lex_feats
            return batch


# ─── Temel Dataset (EXP0/1/2/mBERT/XLM-R) ───────────────────────────────────

class SimpleEmotionDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts  = df["text"].tolist()
        self.labels = df["label_id"].tolist()
        self.tok    = tokenizer
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


# ─── Metrik Hesaplama ─────────────────────────────────────────────────────────

def compute_all_metrics(y_true, y_pred):
    f1_macro    = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    acc         = float(accuracy_score(y_true, y_pred))
    per_class   = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = {EMOTIONS[i]: round(float(per_class[i]), 4) for i in range(NUM_LABELS)}
    return {
        "f1_macro":    round(f1_macro,    4),
        "f1_weighted": round(f1_weighted, 4),
        "accuracy":    round(acc,         4),
        "per_class_f1": per_class_f1,
    }


# ─── Standart Model Değerlendirmesi ──────────────────────────────────────────

def evaluate_standard_model(model_name, cfg, test_df, output_dir):
    print(f"\n{'='*60}")
    print(f"  {model_name} — Standart Model Değerlendirmesi")
    print(f"  Path: {cfg['path']}")
    print(f"{'='*60}")

    t0 = time.time()
    model_path = ROOT / cfg["path"]
    tok_path   = cfg["tokenizer_path"]

    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))

    test_ds = SimpleEmotionDataset(test_df, tokenizer, max_length=128)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    eval_args = TrainingArguments(
        output_dir=str(output_dir / "tmp"),
        per_device_eval_batch_size=128,
        bf16=True,
        fp16=False,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=collator,
    )

    pred_out = trainer.predict(test_ds)
    y_pred   = np.argmax(pred_out.predictions, axis=-1)
    y_true   = pred_out.label_ids

    metrics  = compute_all_metrics(y_true, y_pred)
    elapsed  = time.time() - t0

    result = {
        "model_name": model_name,
        "model_type": "standard",
        "model_path": str(cfg["path"]),
        "test_split": "group_aware_splits/test.parquet",
        "n_test":     int(len(test_df)),
        "duration_s": round(elapsed, 1),
        **metrics,
    }

    out_file = output_dir / f"{model_name.replace('/', '_')}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  F1-Macro:    {result['f1_macro']:.4f}")
    print(f"  F1-Weighted: {result['f1_weighted']:.4f}")
    print(f"  Accuracy:    {result['accuracy']:.4f}")
    print(f"  Süre: {elapsed:.1f}s  |  Sonuç: {out_file.name}")

    # Cleanup tmp
    tmp = output_dir / "tmp"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)

    # VRAM temizle
    del model, trainer
    torch.cuda.empty_cache()

    return result


# ─── Lexicon Model Değerlendirmesi (EXP3/EXP4) ───────────────────────────────

def evaluate_lexicon_model(model_name, cfg, test_df, output_dir):
    print(f"\n{'='*60}")
    print(f"  {model_name} — Lexikon Hibrit Model Değerlendirmesi")
    print(f"  Weights: {cfg['weights']}")
    print(f"{'='*60}")

    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_path"])
    extractor = LexiconExtractor()

    model = LexiconBERTModel(
        bert_model_name=cfg["bert_model"],
        num_labels=NUM_LABELS,
        lexicon_dim=LEXICON_DIM,
    )
    weights_path = ROOT / cfg["weights"]
    state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    test_ds  = LexiconEmotionDataset(test_df, tokenizer, extractor, max_length=128)
    collator = LexiconDataCollator(tokenizer=tokenizer)

    # LexiconBERTModel loss=None döndürür — Trainer.predict() ile uyumsuz.
    # Manuel tahmin döngüsü kullan.
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    loader = DataLoader(test_ds, batch_size=64, collate_fn=collator,
                        num_workers=2, pin_memory=True)

    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            all_logits.append(outputs.logits.cpu().float().numpy())
            all_labels.append(labels.numpy())

    y_pred   = np.argmax(np.concatenate(all_logits, axis=0), axis=-1)
    y_true   = np.concatenate(all_labels, axis=0)

    metrics  = compute_all_metrics(y_true, y_pred)
    elapsed  = time.time() - t0

    result = {
        "model_name": model_name,
        "model_type": "lexicon_hybrid",
        "weights_path": str(cfg["weights"]),
        "test_split": "group_aware_splits/test.parquet",
        "n_test":     int(len(test_df)),
        "duration_s": round(elapsed, 1),
        **metrics,
    }

    out_file = output_dir / f"{model_name}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  F1-Macro:    {result['f1_macro']:.4f}")
    print(f"  F1-Weighted: {result['f1_weighted']:.4f}")
    print(f"  Accuracy:    {result['accuracy']:.4f}")
    print(f"  Süre: {elapsed:.1f}s  |  Sonuç: {out_file.name}")

    del model
    torch.cuda.empty_cache()

    return result


# ─── Weighted Trainer (EXP1 k-fold) ──────────────────────────────────────────

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


def compute_class_weights(label_ids):
    counts = np.bincount(label_ids, minlength=NUM_LABELS).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    return torch.tensor(len(label_ids) / (counts * NUM_LABELS), dtype=torch.float32)


def compute_sample_weights(label_ids):
    counts = np.bincount(label_ids, minlength=NUM_LABELS).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    class_w = len(label_ids) / (counts * NUM_LABELS)
    return class_w[label_ids]


def cap_per_class(df, max_per_class, seed=42):
    groups = []
    for _, grp in df.groupby("label_id"):
        if len(grp) > max_per_class:
            grp = grp.sample(max_per_class, random_state=seed)
        groups.append(grp)
    return pd.concat(groups, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


# ─── EXP1 4-Fold Group-Aware CV ──────────────────────────────────────────────

def run_exp1_4fold_kfold(train_df, output_dir):
    print(f"\n{'='*60}")
    print("  EXP1 — 4-Fold Group-Aware CV")
    print(f"  Train: {len(train_df):,} örnek  |  max_per_class=5000  |  epochs=2")
    print(f"{'='*60}")

    EXP1_PATH  = ROOT / "results" / "exp1" / "best_model"
    N_FOLDS    = 4
    EPOCHS     = 2
    MAX_CLS    = 5000
    BATCH_SIZE = 64
    GRAD_ACCUM = 2
    SEED       = 42

    df_capped = cap_per_class(train_df, MAX_CLS, seed=SEED)
    print(f"\n[CAP] max_per_class={MAX_CLS} sonrası: {len(df_capped):,} örnek")
    cnt = Counter(df_capped["label_id"].tolist())
    for i, emo in enumerate(EMOTIONS):
        print(f"  {emo:12s}: {cnt.get(i, 0):,}")

    tokenizer = AutoTokenizer.from_pretrained(str(EXP1_PATH))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    X   = df_capped.index.values
    y   = df_capped["label_id"].values

    fold_results = []
    t_global = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'─'*50}")
        print(f"  FOLD {fold_idx+1}/{N_FOLDS}  |  "
              f"train={len(train_idx):,}  val={len(val_idx):,}")

        set_seed(SEED + fold_idx)

        df_train_fold = df_capped.iloc[train_idx].reset_index(drop=True)
        df_val_fold   = df_capped.iloc[val_idx].reset_index(drop=True)

        train_ds = SimpleEmotionDataset(df_train_fold, tokenizer, max_length=128)
        val_ds   = SimpleEmotionDataset(df_val_fold,   tokenizer, max_length=128)

        train_labels   = df_train_fold["label_id"].values
        sample_weights = compute_sample_weights(train_labels)
        class_weights  = compute_class_weights(train_labels)

        # EXP1 ağırlıklarından başla (pretrained değil, fine-tuned)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(EXP1_PATH),
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=False,
        )

        fold_dir    = output_dir / f"kfold_fold{fold_idx+1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        tmp_ckpt    = fold_dir / "tmp_ckpt"
        tmp_ckpt.mkdir(parents=True, exist_ok=True)

        steps_per_epoch = max(1, len(df_train_fold) // (BATCH_SIZE * GRAD_ACCUM))
        warmup_steps    = max(1, int(0.1 * steps_per_epoch * EPOCHS))

        train_args = TrainingArguments(
            output_dir=str(tmp_ckpt),
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
            report_to=[],
            seed=SEED + fold_idx,
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
        )

        collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = WeightedTrainer(
            model=model,
            args=train_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=collator,
            train_sample_weights=sample_weights,
            class_weights=class_weights,
        )

        t_fold = time.time()
        trainer.train()
        fold_elapsed = time.time() - t_fold

        pred_out = trainer.predict(val_ds)
        y_pred   = np.argmax(pred_out.predictions, axis=-1)
        y_true   = pred_out.label_ids

        metrics  = compute_all_metrics(y_true, y_pred)

        fold_result = {
            "fold":        fold_idx + 1,
            "train_size":  int(len(df_train_fold)),
            "val_size":    int(len(df_val_fold)),
            "duration_s":  round(fold_elapsed, 1),
            **metrics,
        }
        fold_results.append(fold_result)

        # Fold metriklerini kaydet
        with open(fold_dir / f"fold{fold_idx+1}_metrics.json", "w", encoding="utf-8") as fh:
            json.dump(fold_result, fh, ensure_ascii=False, indent=2)

        print(f"  F1-Macro:    {metrics['f1_macro']:.4f}")
        print(f"  F1-Weighted: {metrics['f1_weighted']:.4f}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Süre: {fold_elapsed/60:.1f} dk")

        del model, trainer
        torch.cuda.empty_cache()
        shutil.rmtree(tmp_ckpt, ignore_errors=True)

    total_time = time.time() - t_global

    # Özet istatistikler
    f1_macros = [r["f1_macro"]    for r in fold_results]
    f1_wgts   = [r["f1_weighted"] for r in fold_results]
    accs      = [r["accuracy"]    for r in fold_results]

    summary = {
        "f1_macro_mean":    round(float(np.mean(f1_macros)), 4),
        "f1_macro_std":     round(float(np.std(f1_macros)),  4),
        "f1_weighted_mean": round(float(np.mean(f1_wgts)),   4),
        "f1_weighted_std":  round(float(np.std(f1_wgts)),    4),
        "accuracy_mean":    round(float(np.mean(accs)),      4),
        "accuracy_std":     round(float(np.std(accs)),       4),
    }

    kfold_out = {
        "experiment":  "exp1_group_aware_4fold_kfold",
        "config": {
            "base_model":      "results/exp1/best_model",
            "train_split":     "group_aware_splits/train.parquet",
            "n_folds":         N_FOLDS,
            "max_per_class":   MAX_CLS,
            "epochs_per_fold": EPOCHS,
            "batch_size":      BATCH_SIZE,
            "grad_accum":      GRAD_ACCUM,
            "total_train_samples_after_cap": int(len(df_capped)),
        },
        "fold_results": fold_results,
        "summary":      summary,
        "total_duration_s": round(total_time, 1),
    }

    kfold_file = output_dir / "kfold_results.json"
    with open(kfold_file, "w", encoding="utf-8") as f:
        json.dump(kfold_out, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("  EXP1 4-Fold CV Özet:")
    for r in fold_results:
        print(f"  Fold {r['fold']}: F1-M={r['f1_macro']:.4f}  "
              f"F1-W={r['f1_weighted']:.4f}  Acc={r['accuracy']:.4f}")
    print(f"  Ortalama — F1-Macro: {summary['f1_macro_mean']:.4f} ± {summary['f1_macro_std']:.4f}")
    print(f"  Toplam süre: {total_time/60:.1f} dk")
    print(f"  Kaydedildi: {kfold_file}")

    return kfold_out


# ─── Karşılaştırma Tablosu ───────────────────────────────────────────────────

def build_comparison_table(all_results, output_dir):
    table = {
        "title":       "Group-Aware Test Set Karşılaştırma Tablosu",
        "test_split":  "data/splits/group_aware_splits/test.parquet",
        "models":      [],
    }

    for r in all_results:
        table["models"].append({
            "model":        r["model_name"],
            "f1_macro":     r["f1_macro"],
            "f1_weighted":  r["f1_weighted"],
            "accuracy":     r["accuracy"],
            "per_class_f1": r["per_class_f1"],
        })

    out_file = output_dir / "comparison_table.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)

    # Konsol tablosu
    print(f"\n{'='*70}")
    print("  GROUP-AWARE TEST SET KARŞILAŞTIRMA TABLOSU")
    print(f"{'='*70}")
    header = f"{'Model':<18} | {'F1-Macro':<10} | {'F1-Weighted':<13} | {'Accuracy'}"
    sep    = f"{'-'*18}-+-{'-'*10}-+-{'-'*13}-+-{'-'*10}"
    print(f"  {header}")
    print(f"  {sep}")
    for r in all_results:
        print(f"  {r['model_name']:<18} | {r['f1_macro']:<10.4f} | "
              f"{r['f1_weighted']:<13.4f} | {r['accuracy']:.4f}")
    print(f"{'='*70}")
    print(f"\n  Kaydedildi: {out_file}")

    return table


# ─── Ana Fonksiyon ────────────────────────────────────────────────────────────

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  GROUP-AWARE SPLIT DEĞERLENDİRMESİ")
    print("  7 Model — Inference Only (Eğitim Yok)")
    print("=" * 70)

    # Test ve train setlerini yükle
    print(f"\n[VERİ] Yükleniyor: {SPLITS}")
    test_df  = pd.read_parquet(SPLITS / "test.parquet")
    train_df = pd.read_parquet(SPLITS / "train.parquet")
    print(f"  Test  : {len(test_df):,} örnek")
    print(f"  Train : {len(train_df):,} örnek (k-fold için)")

    # label_id kontrolü
    if "label_id" not in test_df.columns:
        test_df["label_id"] = test_df["label"].map(LABEL2ID)
    if "label_id" not in train_df.columns:
        train_df["label_id"] = train_df["label"].map(LABEL2ID)

    test_df["label_id"]  = test_df["label_id"].astype(int)
    train_df["label_id"] = train_df["label_id"].astype(int)

    # Test set dağılımı
    print("\n  Test set label dağılımı:")
    for label, cnt in test_df["label"].value_counts().items():
        print(f"    {label}: {cnt}")

    # ── 1. Tüm modelleri değerlendir (inference only) ─────────────────────
    all_results = []

    for model_name, cfg in MODELS_CFG.items():
        try:
            if cfg["type"] == "lexicon":
                result = evaluate_lexicon_model(model_name, cfg, test_df, OUTPUT)
            else:
                result = evaluate_standard_model(model_name, cfg, test_df, OUTPUT)
            all_results.append(result)
        except Exception as e:
            print(f"\n[HATA] {model_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "model_name":  model_name,
                "error":       str(e),
                "f1_macro":    None,
                "f1_weighted": None,
                "accuracy":    None,
                "per_class_f1": {},
            })

    # ── 2. Karşılaştırma tablosu ──────────────────────────────────────────
    valid_results = [r for r in all_results if r.get("f1_macro") is not None]
    if valid_results:
        build_comparison_table(valid_results, OUTPUT)

    # ── 3. EXP1 4-Fold Group-Aware CV ────────────────────────────────────
    print(f"\n{'='*70}")
    print("  EXP1 — 4-FOLD GROUP-AWARE CROSS-VALIDATION")
    print("  (EXP1 ağırlıklarından başlayarak, 2 epoch fine-tune)")
    print(f"{'='*70}")

    kfold_results = run_exp1_4fold_kfold(train_df, OUTPUT)

    # ── 4. Özet rapor ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  ÖZET RAPOR")
    print(f"{'='*70}")

    print("\n  [A] Model Karşılaştırma (Test Set):")
    print(f"  {'Model':<18} | {'F1-Macro':<10} | {'F1-Weighted':<13} | {'Accuracy'}")
    print(f"  {'-'*18}-+-{'-'*10}-+-{'-'*13}-+-{'-'*10}")
    for r in all_results:
        if r.get("f1_macro") is not None:
            print(f"  {r['model_name']:<18} | {r['f1_macro']:<10.4f} | "
                  f"{r['f1_weighted']:<13.4f} | {r['accuracy']:.4f}")
        else:
            print(f"  {r['model_name']:<18} | HATA: {r.get('error', '?')[:30]}")

    kfold_s = kfold_results.get("summary", {})
    print(f"\n  [B] EXP1 4-Fold CV (Group-Aware Train Set):")
    print(f"  F1-Macro  : {kfold_s.get('f1_macro_mean', '?'):.4f} ± {kfold_s.get('f1_macro_std', '?'):.4f}")
    print(f"  F1-Weighted: {kfold_s.get('f1_weighted_mean', '?'):.4f} ± {kfold_s.get('f1_weighted_std', '?'):.4f}")
    print(f"  Accuracy  : {kfold_s.get('accuracy_mean', '?'):.4f} ± {kfold_s.get('accuracy_std', '?'):.4f}")

    print(f"\n  Tüm çıktılar: {OUTPUT}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
