"""
Fine-Tuned Baseline Karşılaştırma Scripti
==========================================
Bu script mBERT, XLM-RoBERTa modellerini
EXP1 ile aynı train/val/test split üzerinde fine-tune eder.
(ELECTRA-TR ablation design gereği atlandı)

KULLANIM:
  Kuru çalışma (kontrol):  python baseline_finetuned.py
  Gerçek eğitim:           python baseline_finetuned.py --run

NOT: --run olmadan hiçbir GPU işlemi başlatılmaz.
Tahmini süre: ~2-3 saat (RTX 5070 Ti, 2 model x ~1 saat)
"""

import sys
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

MODELS = {
    "mBERT": "google-bert/bert-base-multilingual-cased",
    "XLM-RoBERTa": "xlm-roberta-base",
    # ELECTRA-TR skipped per ablation design
}

TRAIN_CONFIG = {
    "lr": 2e-5,
    "batch_size": 64,
    "epochs": 3,
    "seed": 42,
    "max_length": 64,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "fp16": False,
    "bf16": True,
    "output_dir": "results/baselines"
}

MAX_PER_CLASS = 50000

LABEL_NAMES = [
    "mutluluk", "üzüntü", "öfke", "korku",
    "şaşkınlık", "tiksinti", "sevgi", "nötr", "utanç", "gurur"
]

EXP2_MODEL_PATH = "results/exp2/best_model"
EXP2_REPEATED_HOLDOUT_SEEDS = [42, 123, 7]


def dry_run():
    print("=" * 60)
    print("BASELINE FİNE-TUNE SCRİPTİ — KUR ÇALIŞMA MODU")
    print("=" * 60)
    print("\nEğitilecek modeller:")
    for name, model_id in MODELS.items():
        print(f"  - {name}: {model_id}")
    print(f"\nKonfigurasyon:")
    for k, v in TRAIN_CONFIG.items():
        print(f"  {k}: {v}")
    print(f"  max_per_class: {MAX_PER_CLASS}")
    print(f"\nÇıktı dizini: {TRAIN_CONFIG['output_dir']}")
    print("\nVeri kontrol:")
    splits_path = Path("data/splits/master_splits")
    for split in ["train", "val", "test"]:
        p = splits_path / f"{split}.parquet"
        if p.exists():
            import pandas as pd
            df = pd.read_parquet(p)
            print(f"  {split}.parquet: {len(df)} örnek ✅")
        else:
            print(f"  {split}.parquet: BULUNAMADI ❌")
    print(f"\nEXP2 model path: {EXP2_MODEL_PATH}")
    exp2_ok = Path(EXP2_MODEL_PATH).exists()
    print(f"  EXP2 best_model: {'✅' if exp2_ok else '❌ BULUNAMADI'}")
    print("\nGerçek eğitimi başlatmak için:")
    print("  python baseline_finetuned.py --run")
    print("=" * 60)


def cap_per_class(df, max_per_class):
    """Her sınıftan en fazla max_per_class örnek al."""
    import pandas as pd
    frames = [
        grp.sample(n=min(len(grp), max_per_class), random_state=42)
        for _, grp in df.groupby("label")
    ]
    return pd.concat(frames, ignore_index=True)


def train_single_model(model_name, model_id, train_df, val_df, test_df):
    """Tek bir modeli fine-tune eder ve sonuçları döndürür."""
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                               TrainingArguments, Trainer)
    from datasets import Dataset
    import numpy as np
    from sklearn.metrics import f1_score, accuracy_score

    print(f"\n{'='*50}")
    print(f"Eğitim başlıyor: {model_name}")
    print(f"{'='*50}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    label2id = {l: i for i, l in enumerate(LABEL_NAMES)}

    def prepare_df(df):
        out = df[["text", "label"]].copy().reset_index(drop=True)
        out["label"] = out["label"].map(label2id).astype(int)
        return out

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True,
                         padding="max_length", max_length=TRAIN_CONFIG["max_length"])

    train_ds = Dataset.from_pandas(prepare_df(train_df), preserve_index=False).map(tokenize, batched=True)
    val_ds   = Dataset.from_pandas(prepare_df(val_df),   preserve_index=False).map(tokenize, batched=True)
    test_ds  = Dataset.from_pandas(prepare_df(test_df),  preserve_index=False).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=10,
        id2label={i: l for i, l in enumerate(LABEL_NAMES)},
        label2id={l: i for i, l in enumerate(LABEL_NAMES)}
    )

    output_dir = Path(TRAIN_CONFIG["output_dir"]) / model_name.replace("-", "_").lower()

    total_steps = (len(train_df) // TRAIN_CONFIG["batch_size"]) * TRAIN_CONFIG["epochs"]
    warmup_steps = int(TRAIN_CONFIG["warmup_ratio"] * total_steps)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=TRAIN_CONFIG["epochs"],
        per_device_train_batch_size=TRAIN_CONFIG["batch_size"],
        per_device_eval_batch_size=128,
        learning_rate=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        warmup_steps=warmup_steps,
        bf16=TRAIN_CONFIG["bf16"],
        fp16=TRAIN_CONFIG["fp16"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        seed=TRAIN_CONFIG["seed"],
        report_to="none"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "f1_macro":    f1_score(labels, preds, average="macro",    zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
            "accuracy":    accuracy_score(labels, preds)
        }

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()

    test_results = trainer.predict(test_ds)
    preds  = np.argmax(test_results.predictions, axis=-1)
    labels = test_results.label_ids

    per_class_f1_arr = f1_score(labels, preds, average=None, zero_division=0)
    per_class_f1 = {LABEL_NAMES[i]: round(float(v), 4) for i, v in enumerate(per_class_f1_arr)}

    results = {
        "model":            model_name,
        "model_id":         model_id,
        "type":             "fine-tuned",
        "test_f1_macro":    round(float(f1_score(labels, preds, average="macro",    zero_division=0)), 4),
        "test_f1_weighted": round(float(f1_score(labels, preds, average="weighted", zero_division=0)), 4),
        "test_accuracy":    round(float(accuracy_score(labels, preds)), 4),
        "per_class_f1":     per_class_f1,
        "config":           TRAIN_CONFIG
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{model_name} Test Sonuçları:")
    print(f"  F1-Macro:    {results['test_f1_macro']}")
    print(f"  F1-Weighted: {results['test_f1_weighted']}")
    print(f"  Accuracy:    {results['test_accuracy']}")

    return results


def run_exp2_repeated_holdout(full_df):
    """
    EXP2 ağırlıklarını yükler, 3 farklı seed ile stratified 70/15/15 split yapar,
    sadece test kümesini değerlendirir. Variance tahmini için repeated holdout.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from datasets import Dataset
    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn.metrics import f1_score, accuracy_score

    print(f"\n{'='*50}")
    print(f"EXP2 Repeated Holdout (seeds: {EXP2_REPEATED_HOLDOUT_SEEDS})")
    print(f"{'='*50}")

    label2id = {l: i for i, l in enumerate(LABEL_NAMES)}
    seed_results = []

    tokenizer = AutoTokenizer.from_pretrained(EXP2_MODEL_PATH)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True,
                         padding="max_length", max_length=128)

    for seed in EXP2_REPEATED_HOLDOUT_SEEDS:
        print(f"\n  Seed {seed} çalışıyor...")

        df = full_df[["text", "label"]].copy().reset_index(drop=True)
        df["label_id"] = df["label"].map(label2id).astype(int)

        train_val, test = train_test_split(
            df, test_size=0.15, random_state=seed, stratify=df["label_id"]
        )
        _, test_split = train_test_split(
            train_val, test_size=0.15 / 0.85, random_state=seed, stratify=train_val["label_id"]
        )

        test_prep = test_split[["text", "label_id"]].rename(columns={"label_id": "label"})
        test_ds = Dataset.from_pandas(test_prep, preserve_index=False).map(tokenize, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(EXP2_MODEL_PATH)

        dummy_args = TrainingArguments(
            output_dir="results/baselines/exp2_holdout_tmp",
            per_device_eval_batch_size=128,
            bf16=True,
            report_to="none"
        )

        trainer = Trainer(model=model, args=dummy_args)
        pred_out = trainer.predict(test_ds)
        preds  = np.argmax(pred_out.predictions, axis=-1)
        labels = pred_out.label_ids

        f1_mac  = round(float(f1_score(labels, preds, average="macro",    zero_division=0)), 4)
        f1_wgt  = round(float(f1_score(labels, preds, average="weighted", zero_division=0)), 4)
        acc     = round(float(accuracy_score(labels, preds)), 4)

        print(f"    F1-Macro: {f1_mac} | F1-Weighted: {f1_wgt} | Accuracy: {acc}")
        seed_results.append({"seed": seed, "f1_macro": f1_mac, "f1_weighted": f1_wgt, "accuracy": acc})

    f1_vals = [r["f1_macro"] for r in seed_results]
    mean_f1 = round(float(np.mean(f1_vals)), 4)
    std_f1  = round(float(np.std(f1_vals)), 4)

    print(f"\n  EXP2 Repeated Holdout Özeti:")
    print(f"  F1-Macro: {mean_f1} ± {std_f1}  (seeds: {f1_vals})")

    return {"seed_results": seed_results, "f1_macro_mean": mean_f1, "f1_macro_std": std_f1}


def save_training_summary(baseline_results, exp2_holdout):
    """Tüm sonuçları results/training_summary.txt dosyasına yazar."""
    out_path = Path("results/training_summary.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results_map = {r["model"]: r for r in baseline_results}
    mbert   = results_map.get("mBERT",       {})
    xlmr    = results_map.get("XLM-RoBERTa", {})

    lines = [
        "=" * 65,
        "TÜRKÇE DUYGU ANALİZİ — EĞİTİM ÖZETİ",
        "=" * 65,
        "",
        f"{'Model':<22} | {'F1-Macro':<10} | {'F1-Weighted':<13} | {'Accuracy'}",
        f"{'-'*22}-+-{'-'*10}-+-{'-'*13}-+-{'-'*10}",
        f"{'BERTurk EXP2 (best)':<22} | {0.4629:<10} | {0.8936:<13} | {0.8743}",
        f"{'BERTurk EXP4':<22} | {0.4597:<10} | {0.8956:<13} | {0.8761}",
        f"{'mBERT (fine-tuned)':<22} | {mbert.get('test_f1_macro', '[N/A]'):<10} | {mbert.get('test_f1_weighted', '[N/A]'):<13} | {mbert.get('test_accuracy', '[N/A]')}",
        f"{'XLM-RoBERTa (f-t)':<22} | {xlmr.get('test_f1_macro', '[N/A]'):<10} | {xlmr.get('test_f1_weighted', '[N/A]'):<13} | {xlmr.get('test_accuracy', '[N/A]')}",
        "",
        f"{'='*65}",
        "EXP2 Repeated Holdout (3 seed: 42, 123, 7)",
        f"{'='*65}",
    ]

    if exp2_holdout:
        mean_f1 = exp2_holdout["f1_macro_mean"]
        std_f1  = exp2_holdout["f1_macro_std"]
        lines.append(f"F1-Macro: {mean_f1} ± {std_f1}")
        lines.append("")
        lines.append(f"{'Seed':<8} | {'F1-Macro':<10} | {'F1-Weighted':<13} | {'Accuracy'}")
        lines.append(f"{'-'*8}-+-{'-'*10}-+-{'-'*13}-+-{'-'*10}")
        for sr in exp2_holdout["seed_results"]:
            lines.append(
                f"{sr['seed']:<8} | {sr['f1_macro']:<10} | {sr['f1_weighted']:<13} | {sr['accuracy']}"
            )
    else:
        lines.append("EXP2 Repeated Holdout: ÇALIŞTIRILMADI")

    lines += ["", "=" * 65]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nÖzet dosyası kaydedildi: {out_path}")
    for line in lines:
        print(line)


def run_all():
    import pandas as pd

    print("Veri yükleniyor...")
    splits_path = Path("data/splits/master_splits")
    train_df = pd.read_parquet(splits_path / "train.parquet")
    val_df   = pd.read_parquet(splits_path / "val.parquet")
    test_df  = pd.read_parquet(splits_path / "test.parquet")
    print(f"Ham — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_df = cap_per_class(train_df, MAX_PER_CLASS)
    print(f"Kırpma sonrası — Train: {len(train_df)} (max {MAX_PER_CLASS}/sınıf)")

    all_results = []
    for model_name, model_id in MODELS.items():
        result = train_single_model(model_name, model_id, train_df, val_df, test_df)
        all_results.append(result)

    summary_path = Path(TRAIN_CONFIG["output_dir"]) / "baseline_comparison_finetuned.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n=== BASELINE KARŞILAŞTIRMA TABLOSU ===")
    print(f"{'Model':<20} {'F1-Macro':<12} {'F1-Weighted':<14} {'Accuracy'}")
    print("-" * 58)
    for r in all_results:
        print(f"{r['model']:<20} {r['test_f1_macro']:<12} {r['test_f1_weighted']:<14} {r['test_accuracy']}")
    print(f"\nTam sonuçlar: {summary_path}")

    # EXP2 Repeated Holdout — tüm veriyi birleştir
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    exp2_holdout = run_exp2_repeated_holdout(full_df)

    # Özet dosyasını kaydet
    save_training_summary(all_results, exp2_holdout)


if __name__ == "__main__":
    if "--run" in sys.argv:
        run_all()
    else:
        dry_run()
