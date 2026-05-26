"""
EXP2 Repeated Holdout Evaluation
=================================
Mevcut EXP2 ağırlıklarını 3 farklı seed ile değerlendirir.
Her seed için 70/15/15 stratified split yapılır, sadece test seti değerlendirilir.
Yeniden eğitim yapılmaz.

Çıktı: results/kfold/exp2_repeated_holdout.json
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

SEEDS = [42, 123, 7]
EXP2_MODEL_DIR = Path("results/exp2/best_model")
DATA_PATH = Path("data/processed/combined_clean.parquet")
OUTPUT_PATH = Path("results/kfold/exp2_repeated_holdout.json")
MAX_LENGTH = 128
BATCH_SIZE = 128

LABEL_NAMES = [
    "mutluluk", "üzüntü", "öfke", "korku",
    "şaşkınlık", "tiksinti", "sevgi", "nötr", "utanç", "gurur"
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_NAMES)}


def evaluate_on_split(test_df, tokenizer, model, device):
    texts = test_df["label_norm"].map(lambda x: x).index  # warmup
    texts = test_df["text"].tolist()
    labels = [LABEL2ID.get(l, 0) for l in test_df["label_norm"].tolist()]

    all_preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i : i + BATCH_SIZE]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())

    labels_arr = np.array(labels)
    preds_arr = np.array(all_preds)

    per_class_arr = f1_score(labels_arr, preds_arr, average=None, zero_division=0)
    per_class_f1 = {LABEL_NAMES[i]: round(float(v), 4) for i, v in enumerate(per_class_arr)}

    return {
        "f1_macro":    round(float(f1_score(labels_arr, preds_arr, average="macro",    zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(labels_arr, preds_arr, average="weighted", zero_division=0)), 4),
        "accuracy":    round(float(accuracy_score(labels_arr, preds_arr)), 4),
        "per_class_f1": per_class_f1,
        "test_size":   int(len(test_df)),
    }


def main():
    print("=" * 60)
    print("EXP2 Repeated Holdout Evaluation")
    print(f"Seeds: {SEEDS}")
    print("=" * 60)

    # Load full dataset
    print(f"\nVeri yükleniyor: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"Toplam örnek: {len(df):,}")
    print(f"Sınıf dağılımı:\n{df['label_norm'].value_counts()}")

    # Load EXP2 model once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nCihaz: {device}")
    print(f"Model yükleniyor: {EXP2_MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(str(EXP2_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(EXP2_MODEL_DIR))
    model = model.to(device)
    model.eval()
    print("Model yüklendi.")

    # Evaluate each seed
    per_seed_results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        # 70/15/15 stratified split
        train_val, test_df = train_test_split(
            df, test_size=0.15, random_state=seed, stratify=df["label_norm"]
        )
        print(f"  Test: {len(test_df):,} örnek")

        result = evaluate_on_split(test_df, tokenizer, model, device)
        result["seed"] = seed
        print(f"  F1-Macro:    {result['f1_macro']}")
        print(f"  F1-Weighted: {result['f1_weighted']}")
        print(f"  Accuracy:    {result['accuracy']}")
        per_seed_results.append(result)

    # Aggregate stats
    f1_macros    = [r["f1_macro"]    for r in per_seed_results]
    f1_weighteds = [r["f1_weighted"] for r in per_seed_results]
    accuracies   = [r["accuracy"]    for r in per_seed_results]

    summary = {
        "experiment":        "EXP2 Repeated Holdout",
        "model":             "BERTurk (dbmdz/bert-base-turkish-cased)",
        "model_dir":         str(EXP2_MODEL_DIR),
        "seeds":             SEEDS,
        "split_ratio":       "70/15/15",
        "per_seed":          per_seed_results,
        "f1_macro_mean":     round(float(np.mean(f1_macros)),    4),
        "f1_macro_std":      round(float(np.std(f1_macros)),     4),
        "f1_weighted_mean":  round(float(np.mean(f1_weighteds)), 4),
        "f1_weighted_std":   round(float(np.std(f1_weighteds)),  4),
        "accuracy_mean":     round(float(np.mean(accuracies)),   4),
        "accuracy_std":      round(float(np.std(accuracies)),    4),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("SONUÇLAR")
    print("=" * 60)
    print(f"F1-Macro:    {summary['f1_macro_mean']:.4f} ± {summary['f1_macro_std']:.4f}")
    print(f"F1-Weighted: {summary['f1_weighted_mean']:.4f} ± {summary['f1_weighted_std']:.4f}")
    print(f"Accuracy:    {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"\nKaydedildi: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
