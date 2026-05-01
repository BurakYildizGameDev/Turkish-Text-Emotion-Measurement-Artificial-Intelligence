"""
Rakip Model Karşılaştırması — Aynı Test Seti Üzerinde
======================================================
82.532 örneklik test seti üzerinde şu modelleri değerlendirir:
  1. XLM-RoBERTa-base  (zero-shot, fine-tune yok)
  2. mBERT             (zero-shot, fine-tune yok)
  3. BiLSTM baseline   (fastText embeddings ile)

Her model için F1-Macro, Accuracy ve F1-Weighted raporlanır.

Çıktılar (results/baseline_comparison/):
  comparison_results.json  — tüm sonuçlar
  comparison_chart.png     — bar grafiği karşılaştırma
  per_model/               — her modelin detaylı sonuçları

NOT: Zero-shot = modeller hiç fine-tune edilmeden, 
     sadece pre-trained ağırlıklarla çalıştırılır.
     Bu, fair karşılaştırma olması açısından önemlidir.
"""

import sys
import os
import json
import warnings
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
)
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
)

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "baseline_comparison"
SPLITS  = ROOT / "data" / "splits" / "master_splits"

EMOTIONS = [
    "mutluluk", "üzüntü", "öfke", "korku",
    "şaşkınlık", "tiksinti", "sevgi", "nötr", "utanç", "gurur",
]
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL = {i: e for i, e in enumerate(EMOTIONS)}
NUM_LABELS = 10
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Veri Yükleme ───────────────────────────────────────────────────────────

def load_test_data():
    """Mevcut test split'ini yükler."""
    test_path = SPLITS / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Test verisi bulunamadı: {test_path}")
    
    df = pd.read_parquet(test_path)
    if "label_id" not in df.columns:
        df["label_id"] = df["label"].map(LABEL2ID)
    
    df = df.dropna(subset=["text", "label_id"])
    df["label_id"] = df["label_id"].astype(int)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 5].reset_index(drop=True)
    
    print(f"[DATA] Test seti: {len(df):,} örnek")
    cnt = Counter(df["label_id"].tolist())
    for i, emo in enumerate(EMOTIONS):
        print(f"  {emo:12s}: {cnt.get(i, 0):7,d}")
    
    return df


class SimpleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": self.labels[idx],
        }


# ─── Model Değerlendirme ────────────────────────────────────────────────────

def evaluate_transformer_zeroshot(model_name, df_test, batch_size=64, model_label=""):
    """
    Bir transformer modelini zero-shot classification head ile değerlendirir.
    Pre-trained model + rastgele classification head → baseline performansı.
    """
    print(f"\n{'─' * 65}")
    print(f"  [{model_label}] {model_name}")
    print(f"{'─' * 65}")
    
    t0 = time.time()
    
    print(f"  Tokenizer yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"  Model yükleniyor (+ random classification head)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )
    model.to(DEVICE)
    model.eval()
    
    texts = df_test["text"].tolist()
    labels = df_test["label_id"].values
    
    dataset = SimpleDataset(texts, labels, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)
    
    all_preds = []
    all_probs = []
    
    print(f"  Tahminler hesaplanıyor ({len(df_test):,} örnek)...")
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    y_true = labels
    
    elapsed = time.time() - t0
    
    # Metrikler
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    report = classification_report(
        y_true, y_pred, target_names=EMOTIONS, output_dict=True, zero_division=0
    )
    
    per_class = {}
    for emo in EMOTIONS:
        if emo in report:
            per_class[emo] = {
                "precision": round(report[emo]["precision"], 4),
                "recall": round(report[emo]["recall"], 4),
                "f1": round(report[emo]["f1-score"], 4),
                "support": int(report[emo]["support"]),
            }
    
    result = {
        "model": model_name,
        "model_label": model_label,
        "type": "zero-shot (random classification head)",
        "test_samples": len(df_test),
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "per_class": per_class,
        "inference_time_sec": round(elapsed, 2),
    }
    
    # Konsol özet
    print(f"\n  Sonuçlar:")
    print(f"    Accuracy:    {acc:.4f}")
    print(f"    F1-Macro:    {f1_macro:.4f}")
    print(f"    F1-Weighted: {f1_weighted:.4f}")
    print(f"    Süre:        {elapsed:.1f}s")
    
    # Cleanup GPU memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return result, y_pred, y_probs


# ─── BiLSTM Baseline ────────────────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    """Basit BiLSTM sınıflandırıcı, token ID'leri üzerinden çalışır."""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=10, 
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, input_ids, attention_mask=None):
        embeds = self.embedding(input_ids)
        if attention_mask is not None:
            embeds = embeds * attention_mask.unsqueeze(-1)
        
        lstm_out, (hidden, _) = self.lstm(embeds)
        # Son katmanın forward ve backward hidden state'lerini birleştir
        hidden_fwd = hidden[-2]
        hidden_bwd = hidden[-1]
        combined = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        combined = self.dropout(combined)
        logits = self.fc(combined)
        return logits


def evaluate_bilstm_baseline(df_test, batch_size=128):
    """
    BiLSTM baseline: BERTurk tokenizer ile token ID'leri alıp,
    random ağırlıklarla (eğitimsiz) sınıflandırma yapar.
    Bu, random baseline'ı temsil eder.
    """
    print(f"\n{'─' * 65}")
    print(f"  [BiLSTM] Random BiLSTM Baseline")
    print(f"{'─' * 65}")
    
    t0 = time.time()
    
    # BERTurk tokenizer kullanarak token ID'leri oluştur
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    vocab_size = tokenizer.vocab_size
    
    print(f"  Vocab size: {vocab_size}")
    print(f"  Model: BiLSTM (embed=128, hidden=128, layers=2, bidirectional)")
    
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=128,
        num_classes=NUM_LABELS,
        num_layers=2,
        dropout=0.3,
    )
    model.to(DEVICE)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametre: {total_params:,}")
    
    texts = df_test["text"].tolist()
    labels = df_test["label_id"].values
    
    dataset = SimpleDataset(texts, labels, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)
    
    all_preds = []
    all_probs = []
    
    print(f"  Tahminler hesaplanıyor ({len(df_test):,} örnek)...")
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    y_true = labels
    
    elapsed = time.time() - t0
    
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    report = classification_report(
        y_true, y_pred, target_names=EMOTIONS, output_dict=True, zero_division=0
    )
    
    per_class = {}
    for emo in EMOTIONS:
        if emo in report:
            per_class[emo] = {
                "precision": round(report[emo]["precision"], 4),
                "recall": round(report[emo]["recall"], 4),
                "f1": round(report[emo]["f1-score"], 4),
                "support": int(report[emo]["support"]),
            }
    
    result = {
        "model": "BiLSTM (BERTurk tokenizer)",
        "model_label": "BiLSTM",
        "type": "untrained random baseline",
        "architecture": "BiLSTM (embed=128, hidden=128, layers=2, bidirectional)",
        "total_params": total_params,
        "test_samples": len(df_test),
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "per_class": per_class,
        "inference_time_sec": round(elapsed, 2),
    }
    
    print(f"\n  Sonuçlar:")
    print(f"    Accuracy:    {acc:.4f}")
    print(f"    F1-Macro:    {f1_macro:.4f}")
    print(f"    F1-Weighted: {f1_weighted:.4f}")
    print(f"    Süre:        {elapsed:.1f}s")
    
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return result, y_pred, y_probs


# ─── Random Baseline ────────────────────────────────────────────────────────

def random_baseline(df_test):
    """Rastgele tahmin baseline'ı (sınıf dağılımı ile orantılı)."""
    labels = df_test["label_id"].values
    n = len(labels)
    
    # Stratified random
    np.random.seed(SEED)
    class_probs = np.bincount(labels, minlength=NUM_LABELS) / n
    random_preds = np.random.choice(NUM_LABELS, size=n, p=class_probs)
    
    acc = accuracy_score(labels, random_preds)
    f1_macro = f1_score(labels, random_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, random_preds, average="weighted", zero_division=0)
    
    report = classification_report(
        labels, random_preds, target_names=EMOTIONS, output_dict=True, zero_division=0
    )
    
    per_class = {}
    for emo in EMOTIONS:
        if emo in report:
            per_class[emo] = {
                "precision": round(report[emo]["precision"], 4),
                "recall": round(report[emo]["recall"], 4),
                "f1": round(report[emo]["f1-score"], 4),
                "support": int(report[emo]["support"]),
            }
    
    return {
        "model": "Random Baseline (stratified)",
        "model_label": "Random",
        "type": "random baseline",
        "test_samples": n,
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "per_class": per_class,
    }


# ─── Majority Baseline ──────────────────────────────────────────────────────

def majority_baseline(df_test):
    """Çoğunluk sınıfı baseline'ı."""
    labels = df_test["label_id"].values
    n = len(labels)
    
    majority_class = int(np.bincount(labels).argmax())
    majority_preds = np.full(n, majority_class)
    
    acc = accuracy_score(labels, majority_preds)
    f1_macro = f1_score(labels, majority_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, majority_preds, average="weighted", zero_division=0)
    
    return {
        "model": f"Majority Baseline ({EMOTIONS[majority_class]})",
        "model_label": "Majority",
        "type": "majority baseline",
        "majority_class": EMOTIONS[majority_class],
        "test_samples": n,
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
    }


# ─── Görselleştirme ──────────────────────────────────────────────────────────

def save_comparison_chart(results, save_path):
    """Tüm modellerin karşılaştırma bar grafiği."""
    model_names = [r["model_label"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    f1_macros = [r["f1_macro"] for r in results]
    f1_weights = [r["f1_weighted"] for r in results]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bars1 = ax.bar(x - width, accuracies, width, label="Accuracy", color="#3498db", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x, f1_macros, width, label="F1-Macro", color="#e74c3c", alpha=0.85, edgecolor="white")
    bars3 = ax.bar(x + width, f1_weights, width, label="F1-Weighted", color="#2ecc71", alpha=0.85, edgecolor="white")
    
    # Değer etiketleri
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight="bold")
    
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Skor", fontsize=12)
    ax.set_title("Aynı Test Seti Üzerinde Model Karşılaştırması\n"
                 f"(Test Seti: {results[0].get('test_samples', 'N/A'):,} örnek)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    
    # BERTurk (fine-tuned) vurgula
    for i, r in enumerate(results):
        if "berturk" in r.get("model", "").lower() or "exp" in r.get("model_label", "").lower():
            ax.patches[i].set_edgecolor("#2c3e50")
            ax.patches[i].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Comparison chart: {save_path}")


def save_per_class_comparison(results, save_path):
    """Per-class F1 karşılaştırması (sadece per_class verisi olan modeller)."""
    models_with_pc = [r for r in results if "per_class" in r and r["per_class"]]
    if len(models_with_pc) < 2:
        return
    
    n_models = len(models_with_pc)
    x = np.arange(NUM_LABELS)
    width = 0.8 / n_models
    
    fig, ax = plt.subplots(figsize=(16, 7))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#95a5a6"]
    
    for i, r in enumerate(models_with_pc):
        f1_vals = [r["per_class"].get(emo, {}).get("f1", 0) for emo in EMOTIONS]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, f1_vals, width, 
                      label=r["model_label"],
                      color=colors[i % len(colors)],
                      alpha=0.85, edgecolor="white")
    
    ax.set_xticks(x)
    ax.set_xticklabels([e[:9] for e in EMOTIONS], rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Sınıf Bazlı F1 Karşılaştırması — Aynı Test Seti",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Per-class comparison: {save_path}")


# ─── Ana Fonksiyon ──────────────────────────────────────────────────────────

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 65)
    print("  RAKİP MODEL KARŞILAŞTIRMASI — AYNI TEST SETİ")
    print("=" * 65)
    
    df_test = load_test_data()

    MAX_SAMPLES = 5000
    if len(df_test) > MAX_SAMPLES:
        df_test = df_test.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
        print(f"[SAMPLE] Rastgele {MAX_SAMPLES} örnek seçildi")

    all_results = []
    
    # ── 1. Baselines ──────────────────────────────────────────────
    print("\n[1] Baselines...")
    all_results.append(random_baseline(df_test))
    all_results.append(majority_baseline(df_test))
    
    # ── 2. BiLSTM (untrained) ────────────────────────────────────
    print("\n[2] BiLSTM Baseline...")
    try:
        bilstm_result, _, _ = evaluate_bilstm_baseline(df_test, batch_size=128)
        all_results.append(bilstm_result)
    except Exception as e:
        print(f"  [HATA] BiLSTM: {e}")
    
    # ── 3. mBERT (zero-shot) ─────────────────────────────────────
    print("\n[3] mBERT Zero-Shot...")
    try:
        mbert_result, _, _ = evaluate_transformer_zeroshot(
            "bert-base-multilingual-cased",
            df_test, batch_size=64,
            model_label="mBERT"
        )
        all_results.append(mbert_result)
    except Exception as e:
        print(f"  [HATA] mBERT: {e}")
    
    # ── 4. XLM-RoBERTa (zero-shot) ──────────────────────────────
    print("\n[4] XLM-RoBERTa Zero-Shot...")
    try:
        xlmr_result, _, _ = evaluate_transformer_zeroshot(
            "xlm-roberta-base",
            df_test, batch_size=64,
            model_label="XLM-RoBERTa"
        )
        all_results.append(xlmr_result)
    except Exception as e:
        print(f"  [HATA] XLM-RoBERTa: {e}")
    
    # ── 5. Mevcut Deneylerimiz ───────────────────────────────────
    print("\n[5] Mevcut Deneyler (fine-tuned BERTurk)...")
    for exp_name in ["exp0", "exp1", "exp2", "exp3", "exp4"]:
        metrics_path = ROOT / "results" / exp_name / "metrics_report.json"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                exp_data = json.load(f)
            all_results.append({
                "model": exp_data.get("model", "BERTurk"),
                "model_label": f"BERTurk ({exp_name.upper()})",
                "type": "fine-tuned",
                "test_samples": sum(
                    exp_data.get("per_class", {}).get(e, {}).get("support", 0)
                    for e in EMOTIONS
                ),
                "accuracy": exp_data.get("test_accuracy", 0),
                "f1_macro": exp_data.get("test_f1_macro", 0),
                "f1_weighted": exp_data.get("test_f1_weighted", 0),
                "per_class": exp_data.get("per_class", {}),
            })
            print(f"  {exp_name.upper()}: F1-Macro={exp_data.get('test_f1_macro', 0):.4f}")
    
    # ── Sonuçları Kaydet ─────────────────────────────────────────
    output_path = RESULTS / "comparison_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[KAYIT] Sonuçlar: {output_path}")
    
    # ── Görseller ────────────────────────────────────────────────
    save_comparison_chart(all_results, RESULTS / "comparison_chart.png")
    save_per_class_comparison(all_results, RESULTS / "per_class_comparison.png")
    
    # ── Özet Tablo ───────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("  KARŞILAŞTIRMA ÖZETİ")
    print("=" * 85)
    print(f"  {'Model':25s} {'Tür':>15s} {'Acc':>8s} {'F1-Mac':>8s} {'F1-Wt':>8s}")
    print("─" * 85)
    for r in sorted(all_results, key=lambda x: x["f1_macro"], reverse=True):
        print(f"  {r['model_label']:25s} {r['type'][:15]:>15s} "
              f"{r['accuracy']:8.4f} {r['f1_macro']:8.4f} {r['f1_weighted']:8.4f}")
    print("=" * 85)
    
    print(f"\n  NOT: Zero-shot modeller hiç fine-tune edilmemiştir.")
    print(f"       BiLSTM rastgele ağırlıklarla çalıştırılmıştır.")
    print(f"       BERTurk (EXP1-4) fine-tune edilmiştir.")
    print(f"       Bu karşılaştırma, fine-tuning'in etkisini gösterir.\n")
    

if __name__ == "__main__":
    main()
