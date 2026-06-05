"""EXP3 ve EXP4 lexikon modellerini group_aware test setinde değerlendirme."""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import f1_score, accuracy_score

from train_exp3_lexicon import (
    LexiconBERTModel, LexiconExtractor, LexiconEmotionDataset,
    LexiconDataCollator, LEXICON_DIM,
)

ROOT   = Path(__file__).resolve().parent
SPLITS = ROOT / "data" / "splits" / "group_aware_splits"
OUTPUT = ROOT / "results" / "group_aware_eval"
OUTPUT.mkdir(exist_ok=True)

EMOTIONS = [
    "mutluluk", "üzüntü", "öfke", "korku",
    "şaşkınlık", "tiksinti", "sevgi", "nötr", "utanç", "gurur",
]
NUM_LABELS = 10

MODELS = {
    "EXP3": {
        "weights":        "results/exp3/best_model/lexicon_bert_weights.pt",
        "tokenizer_path": "results/exp3/best_model",
        "bert_model":     "dbmdz/bert-base-turkish-cased",
    },
    "EXP4": {
        "weights":        "results/exp4/best_model/master_hybrid_weights.pt",
        "tokenizer_path": "results/exp4/best_model",
        "bert_model":     "dbmdz/bert-base-turkish-cased",
    },
}

def compute_all_metrics(y_true, y_pred):
    f1_macro    = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    acc         = float(accuracy_score(y_true, y_pred))
    per_class   = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "f1_macro":    round(f1_macro,    4),
        "f1_weighted": round(f1_weighted, 4),
        "accuracy":    round(acc,         4),
        "per_class_f1": {EMOTIONS[i]: round(float(per_class[i]), 4) for i in range(NUM_LABELS)},
    }

def evaluate(model_name, cfg, test_df):
    print(f"\n{'='*55}")
    print(f"  {model_name} — Lexikon Hibrit Değerlendirmesi")
    print(f"{'='*55}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_path"])
    extractor = LexiconExtractor()

    model = LexiconBERTModel(
        bert_model_name=cfg["bert_model"],
        num_labels=NUM_LABELS,
        lexicon_dim=LEXICON_DIM,
    )
    sd = torch.load(str(ROOT / cfg["weights"]), map_location="cpu", weights_only=True)
    model.load_state_dict(sd)

    test_ds  = LexiconEmotionDataset(test_df, tokenizer, extractor, max_length=128)
    collator = LexiconDataCollator(tokenizer=tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    model.eval()

    loader = DataLoader(test_ds, batch_size=64, collate_fn=collator,
                        num_workers=2, pin_memory=True)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            labels = batch.pop("labels")
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            all_logits.append(outputs.logits.cpu().float().numpy())
            all_labels.append(labels.numpy())
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(loader)} batch...")

    y_pred  = np.argmax(np.concatenate(all_logits, axis=0), axis=-1)
    y_true  = np.concatenate(all_labels, axis=0)
    metrics = compute_all_metrics(y_true, y_pred)
    elapsed = time.time() - t0

    result = {
        "model_name":   model_name,
        "model_type":   "lexicon_hybrid",
        "weights_path": cfg["weights"],
        "test_split":   "group_aware_splits/test.parquet",
        "n_test":       int(len(test_df)),
        "duration_s":   round(elapsed, 1),
        **metrics,
    }

    out_file = OUTPUT / f"{model_name}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  F1-Macro:    {result['f1_macro']:.4f}")
    print(f"  F1-Weighted: {result['f1_weighted']:.4f}")
    print(f"  Accuracy:    {result['accuracy']:.4f}")
    print(f"  Süre: {elapsed:.1f}s  |  Kaydedildi: {out_file.name}")

    del model
    torch.cuda.empty_cache()
    return result


def update_comparison_table(new_results):
    table_path = OUTPUT / "comparison_table.json"
    if table_path.exists():
        with open(table_path) as f:
            table = json.load(f)
    else:
        table = {"title": "Group-Aware Test Set", "test_split": "group_aware_splits/test.parquet", "models": []}

    existing = {m["model"]: i for i, m in enumerate(table["models"])}
    for r in new_results:
        entry = {
            "model":        r["model_name"],
            "f1_macro":     r["f1_macro"],
            "f1_weighted":  r["f1_weighted"],
            "accuracy":     r["accuracy"],
            "per_class_f1": r["per_class_f1"],
        }
        if r["model_name"] in existing:
            table["models"][existing[r["model_name"]]] = entry
        else:
            table["models"].append(entry)

    with open(table_path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    print(f"\nKarşılaştırma tablosu güncellendi: {table_path}")


if __name__ == "__main__":
    test_df = pd.read_parquet(SPLITS / "test.parquet")
    test_df["label_id"] = test_df["label_id"].astype(int)
    print(f"Test set: {len(test_df):,} örnek")

    results = []
    for name, cfg in MODELS.items():
        try:
            r = evaluate(name, cfg, test_df)
            results.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"HATA [{name}]: {e}")

    if results:
        update_comparison_table(results)
        print("\n=== EXP3/EXP4 SONUÇLARI ===")
        for r in results:
            print(f"  {r['model_name']}: F1-M={r['f1_macro']:.4f}  "
                  f"F1-W={r['f1_weighted']:.4f}  Acc={r['accuracy']:.4f}")
