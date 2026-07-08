"""
v2 eğitim — 8 sınıflı BERTurk duygu sınıflandırıcı.

Kullanım:
    python -m src.training.train_v2 --data gold --seed 42
    python -m src.training.train_v2 --data gold_weak --seed 42 --save-model

Veri düzenleri (--data):
    gold         data/v2/train.parquet (insan etiketli)
    gold_weak    gold + train_weak (sınıf başına en fazla --weak-cap, sabit alt küme)
    gold_silver  gold + train_silver

Tüm düzenlerde val/test aynıdır ve yalnızca gold veriden oluşur. Model seçimi yalnızca
val macro-F1 ile yapılır; test yalnızca en sonda bir kez değerlendirilir.

v1'den farklar:
  - Dengesizlik için TEK yöntem: --imbalance weighted_loss (varsayılan) veya none.
    Sampler ile ağırlıklı loss birlikte kullanılmaz; ağırlıklar yalnızca train'den hesaplanır.
  - bf16 (RTX 50 serisi), sabit ve tüm düzenlerde aynı eğitim bütçesi.
  - Test metrikleri kaynak bazında da raporlanır; kalibrasyon için val'de temperature scaling.

Çıktılar:
    results/v2/runs/<run>/metrics.json, test_logits.npy, test_labels.npy, val_logits.npy
    models/v2/<run>/ (--save-model ile; model + tokenizer + temperature.json)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data.labels import EMOTIONS, ID2LABEL, LABEL2ID, NUM_LABELS

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "v2"
RUNS_DIR = ROOT / "results" / "v2" / "runs"
MODELS_DIR = ROOT / "models" / "v2"
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
# Karşılaştırma (baseline) modelleri — hepsi aynı veri ve bütçeyle eğitilir
MODELS = {
    "berturk": MODEL_NAME,
    "mbert": "google-bert/bert-base-multilingual-cased",
    "xlmr": "FacebookAI/xlm-roberta-base",
}

# Tüm düzenlerde aynı bütçe — karşılaştırma yalnızca veri/yöntem farkını ölçsün
HPARAMS = {
    "max_length": 128,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_epochs": 4,
    "early_stopping_patience": 2,
    "weak_subset_seed": 42,   # weak alt kümesi seed'den bağımsız sabit — varyans yalnızca eğitimden gelsin
}
NATIVE_SOURCES = ("tremo", "tweet_emotion")   # gold'da nötr/sevgi etiketi olmayan ana-dil kaynaklar


# ─── Veri ────────────────────────────────────────────────────────────────────

def load_train(mix: str, weak_cap: int) -> pd.DataFrame:
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    if mix == "gold":
        return train
    if mix == "gold_weak":
        weak = pd.read_parquet(DATA_DIR / "train_weak.parquet")
        parts = [g.sample(min(len(g), weak_cap), random_state=HPARAMS["weak_subset_seed"])
                 for _, g in weak.groupby("label", sort=True)]
        return pd.concat([train, *parts], ignore_index=True)
    if mix == "gold_silver":
        return pd.concat([train, pd.read_parquet(DATA_DIR / "train_silver.parquet")], ignore_index=True)
    raise ValueError(mix)


class EncodedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.enc = tokenizer(df["text"].tolist(), truncation=True, max_length=max_length)
        self.labels = df["label_id"].to_numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


# ─── Loss ────────────────────────────────────────────────────────────────────

def class_weights(label_ids: np.ndarray) -> torch.Tensor:
    """N / (K * n_c) — yalnızca train etiketlerinden."""
    counts = np.bincount(label_ids, minlength=NUM_LABELS).astype(float)
    counts[counts == 0] = 1.0
    return torch.tensor(len(label_ids) / (NUM_LABELS * counts), dtype=torch.float32)


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, loss_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_weights = loss_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.loss_weights.to(outputs.logits.device) if self.loss_weights is not None else None
        loss = nn.functional.cross_entropy(outputs.logits.float(), labels, weight=w)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {"f1_macro": f1_score(labels, preds, average="macro", labels=range(NUM_LABELS), zero_division=0),
            "accuracy": accuracy_score(labels, preds)}


# ─── Kalibrasyon ─────────────────────────────────────────────────────────────

def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Val NLL'ini en aza indiren tek sıcaklık parametresi (Guo vd., 2017)."""
    lg = torch.tensor(logits, dtype=torch.float32)
    lb = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(lg / log_t.exp(), lb)
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.exp().item()


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


# ─── Değerlendirme ───────────────────────────────────────────────────────────

def evaluate(df: pd.DataFrame, logits: np.ndarray) -> dict:
    y = df["label_id"].to_numpy()
    p = logits.argmax(1)
    rep = classification_report(y, p, labels=range(NUM_LABELS), target_names=EMOTIONS,
                                output_dict=True, zero_division=0)
    out = {
        "f1_macro": round(f1_score(y, p, average="macro", labels=range(NUM_LABELS), zero_division=0), 4),
        "f1_weighted": round(f1_score(y, p, average="weighted", labels=range(NUM_LABELS), zero_division=0), 4),
        "accuracy": round(accuracy_score(y, p), 4),
        "per_class": {e: {k: round(rep[e][k], 4) for k in ("precision", "recall", "f1-score")} | {"support": int(rep[e]["support"])}
                      for e in EMOTIONS},
        "confusion_matrix": confusion_matrix(y, p, labels=range(NUM_LABELS)).tolist(),
    }
    # Kaynak bazlı — her kaynağın yalnızca kendi içerdiği sınıflar üzerinden macro-F1
    out["per_source"] = {}
    subsource = df["subsource"].to_numpy()
    for src in sorted(set(subsource)):
        idx = np.flatnonzero(subsource == src)
        present = sorted(set(y[idx]))
        out["per_source"][src] = {
            "n": int(len(idx)),
            "accuracy": round(accuracy_score(y[idx], p[idx]), 4),
            "f1_macro_own_classes": round(f1_score(y[idx], p[idx], average="macro", labels=present, zero_division=0), 4),
        }
    # Çeviri kısa yolu: ana-dil kaynaklarda gold nötr/sevgi yoktur; bu tahminlerin hepsi hatadır
    native = df["source"].isin(NATIVE_SOURCES).to_numpy()
    shortcut = np.isin(p[native], [LABEL2ID["nötr"], LABEL2ID["sevgi"]])
    out["ceviri_kisa_yolu"] = {
        "aciklama": "TREMO/tweet (ana dil) örneklerinin nötr veya sevgi tahmin edilme oranı; gold'da bu kaynaklarda bu sınıflar yok.",
        "orani": round(float(shortcut.mean()), 4),
        "notr": int((p[native] == LABEL2ID["nötr"]).sum()),
        "sevgi": int((p[native] == LABEL2ID["sevgi"]).sum()),
        "native_n": int(native.sum()),
    }
    return out


# ─── Ana akış ────────────────────────────────────────────────────────────────

def run(mix: str, seed: int, imbalance: str, weak_cap: int, save_model: bool, tag: str | None,
        smoke: bool = False, model_key: str = "berturk", lr: float | None = None) -> dict:
    model_name = MODELS[model_key]
    hp = {**HPARAMS, "learning_rate": lr if lr is not None else HPARAMS["learning_rate"]}
    run_name = tag or f"{mix}{'' if imbalance == 'weighted_loss' else '_' + imbalance}_s{seed}"
    if smoke:
        run_name = "_smoke_" + run_name
    out_dir = RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = ROOT / "models" / "_tmp_ckpt" / run_name
    set_seed(seed)
    t0 = time.time()

    train_df = load_train(mix, weak_cap)
    val_df = pd.read_parquet(DATA_DIR / "val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    for d in (train_df, val_df, test_df):
        assert set(d["label"]) <= set(EMOTIONS)
    assert (val_df["quality"] == "gold").all() and (test_df["quality"] == "gold").all()
    if smoke:   # hat testi: küçük alt küme, tek epoch — sonuçlar anlamsızdır
        train_df = train_df.sample(1024, random_state=seed)
        val_df = val_df.sample(512, random_state=seed).reset_index(drop=True)
        test_df = test_df.sample(512, random_state=seed).reset_index(drop=True)
    print(f"[{run_name}] train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_name)
    ds_train = EncodedDataset(train_df, tok, hp["max_length"])
    ds_val = EncodedDataset(val_df, tok, hp["max_length"])
    ds_test = EncodedDataset(test_df, tok, hp["max_length"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID)
    weights = class_weights(train_df["label_id"].to_numpy()) if imbalance == "weighted_loss" else None

    args = TrainingArguments(
        output_dir=str(tmp_dir),
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=128,
        learning_rate=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
        warmup_ratio=hp["warmup_ratio"],
        num_train_epochs=1 if smoke else hp["max_epochs"],
        lr_scheduler_type="linear",
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=100,
        report_to=[],
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
    )
    trainer = WeightedLossTrainer(
        model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
        processing_class=tok, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics, loss_weights=weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=hp["early_stopping_patience"])],
    )
    train_result = trainer.train()

    val_logits = trainer.predict(ds_val).predictions.astype(np.float32)
    test_logits = trainer.predict(ds_test).predictions.astype(np.float32)
    T = fit_temperature(val_logits, val_df["label_id"].to_numpy())

    test_eval = evaluate(test_df, test_logits)
    y_test = test_df["label_id"].to_numpy()
    history = [h for h in trainer.state.log_history if "eval_f1_macro" in h]
    metrics = {
        "run": run_name, "data": mix, "seed": seed, "imbalance": imbalance,
        "weak_cap": weak_cap if mix == "gold_weak" else None,
        "hparams": hp, "model": model_name, "model_key": model_key,
        "train_size": len(train_df),
        "train_class_counts": {e: int((train_df["label"] == e).sum()) for e in EMOTIONS},
        "train_quality_counts": train_df["quality"].value_counts().to_dict(),
        "loss_weights": [round(float(w), 4) for w in weights] if weights is not None else None,
        "global_steps": train_result.global_step,
        "best_epoch": next((h["epoch"] for h in history
                            if h["eval_f1_macro"] == max(x["eval_f1_macro"] for x in history)), None),
        "val_history": [{k: round(v, 4) for k, v in h.items() if k in ("epoch", "eval_f1_macro", "eval_accuracy", "eval_loss")}
                        for h in history],
        "val_f1_macro": round(max(h["eval_f1_macro"] for h in history), 4),
        "test": test_eval,
        "calibration": {
            "temperature": round(T, 4),
            "test_ece_before": round(ece(softmax(test_logits), y_test), 4),
            "test_ece_after": round(ece(softmax(test_logits / T), y_test), 4),
        },
        "runtime_min": round((time.time() - t0) / 60, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.save(out_dir / "test_logits.npy", test_logits)
    np.save(out_dir / "test_labels.npy", y_test)
    np.save(out_dir / "val_logits.npy", val_logits)

    if save_model:
        mdir = MODELS_DIR / run_name
        trainer.save_model(str(mdir))
        tok.save_pretrained(str(mdir))
        (mdir / "temperature.json").write_text(json.dumps({"temperature": T}) + "\n", encoding="utf-8")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[{run_name}] val F1={metrics['val_f1_macro']} | test F1={test_eval['f1_macro']} "
          f"acc={test_eval['accuracy']} | T={T:.2f} ECE {metrics['calibration']['test_ece_before']}"
          f"→{metrics['calibration']['test_ece_after']} | {metrics['runtime_min']} dk", flush=True)
    return metrics


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="v2 8-sınıf BERTurk eğitimi")
    ap.add_argument("--data", choices=["gold", "gold_weak", "gold_silver"], default="gold")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--imbalance", choices=["weighted_loss", "none"], default="weighted_loss")
    ap.add_argument("--weak-cap", type=int, default=5000, help="gold_weak: sınıf başına en fazla weak örnek")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--tag", default=None, help="Çalıştırma adı (varsayılan: <data>[_<imbalance>]_s<seed>)")
    ap.add_argument("--smoke", action="store_true", help="Hat testi: 1024 örnek, 1 epoch")
    ap.add_argument("--model", choices=list(MODELS), default="berturk")
    ap.add_argument("--lr", type=float, default=None, help=f"Öğrenme hızı (varsayılan {HPARAMS['learning_rate']})")
    a = ap.parse_args()
    run(a.data, a.seed, a.imbalance, a.weak_cap, a.save_model, a.tag, a.smoke, a.model, a.lr)
