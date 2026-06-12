"""
EXP2 / EXP3 / EXP4 modellerini test_augmented.parquet üzerinde değerlendirir.
Yeniden eğitim yapılmaz — sadece inference + metrik hesaplama.

Kullanım:
    python evaluate_augmented.py --run --model exp2
    python evaluate_augmented.py --run --model exp3
    python evaluate_augmented.py --run --model exp4
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

ROOT      = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "splits" / "test_augmented.parquet"   # varsayılan

MODEL_CONFIGS = {
    "exp2": {
        "model_dir":    ROOT / "results" / "exp2" / "best_model",
        "weights_file": None,
        "model_type":   "hf",
        "out_path":     ROOT / "results" / "augmented_exp2_results.json",
    },
    "exp3": {
        "model_dir":    ROOT / "results" / "exp3" / "best_model",
        "weights_file": ROOT / "results" / "exp3" / "best_model" / "lexicon_bert_weights.pt",
        "model_type":   "lexicon",
        "out_path":     ROOT / "results" / "augmented_exp3_results.json",
    },
    "exp4": {
        "model_dir":    ROOT / "results" / "exp4" / "best_model",
        "weights_file": ROOT / "results" / "exp4" / "best_model" / "master_hybrid_weights.pt",
        "model_type":   "lexicon",
        "out_path":     ROOT / "results" / "augmented_exp4_results.json",
    },
}

# Exp3/exp4 için sabit label sırası (train_exp3_lexicon.py ile aynı)
LEXICON_EMOTIONS = [
    "mutluluk", "üzüntü", "öfke", "korku",
    "şaşkınlık", "tiksinti", "sevgi", "nötr", "utanç", "gurur",
]
LEXICON_LABEL2ID = {e: i for i, e in enumerate(LEXICON_EMOTIONS)}
LEXICON_ID2LABEL = {i: e for i, e in enumerate(LEXICON_EMOTIONS)}


# ─── Ortak metrik hesaplama ──────────────────────────────────────────────────

def _compute_and_save(
    true_ids: list,
    all_preds: list,
    all_probs: list,
    id2label: dict,
    df,
    exp_name: str,
    model_path: str,
    data_path: str,
    device: str,
    out_path: Path,
) -> dict:
    import numpy as np
    from sklearn.metrics import classification_report, f1_score, accuracy_score

    labels_list  = list(range(len(id2label)))
    label_names  = [id2label[i] for i in labels_list]

    acc         = accuracy_score(true_ids, all_preds)
    f1_macro    = f1_score(true_ids, all_preds, average="macro",    labels=labels_list, zero_division=0)
    f1_weighted = f1_score(true_ids, all_preds, average="weighted", labels=labels_list, zero_division=0)

    report = classification_report(
        true_ids, all_preds,
        labels=labels_list,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}
    for name in label_names:
        row = report.get(name, {})
        per_class[name] = {
            "precision": round(row.get("precision", 0.0), 4),
            "recall":    round(row.get("recall",    0.0), 4),
            "f1":        round(row.get("f1-score",  0.0), 4),
            "support":   int(row.get("support",     0)),
        }

    source_support = (
        df.groupby(["label", "source"]).size()
        .unstack(fill_value=0)
        .to_dict()
    )

    result = {
        "experiment":       f"{exp_name}_augmented_eval",
        "model":            str(model_path),
        "data":             str(data_path),
        "total_samples":    len(df),
        "device":           device,
        "test_accuracy":    round(acc, 4),
        "test_f1_macro":    round(f1_macro, 4),
        "test_f1_weighted": round(f1_weighted, 4),
        "per_class":        per_class,
        "source_support": {
            str(k): {str(kk): int(vv) for kk, vv in v.items()}
            for k, v in source_support.items()
        },
        "highlighted": {
            "gurur_f1":      per_class.get("gurur", {}).get("f1", None),
            "utanc_f1":      per_class.get("utanç", {}).get("f1", None),
            "gurur_support": per_class.get("gurur", {}).get("support", None),
            "utanc_support": per_class.get("utanç", {}).get("support", None),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[OK] Sonuçlar kaydedildi: {out_path}")

    print("\n" + "=" * 58)
    print(f"SONUÇLAR — {exp_name.upper()} / test_augmented")
    print("=" * 58)
    print(f"Accuracy   : {acc:.4f}")
    print(f"F1 Macro   : {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print()
    print(f"{'Sınıf':<14} {'Destek':>8} {'F1':>8}")
    print("-" * 34)
    for name, vals in per_class.items():
        print(f"{name:<14} {vals['support']:>8} {vals['f1']:>8.4f}")
    print()
    print(">>> ODAK SINIFLAR <<<")
    g = per_class.get("gurur", {})
    u = per_class.get("utanç", {})
    print(f"  gurur — destek: {g.get('support','?'):>4}  F1: {g.get('f1','?')}")
    print(f"  utanç — destek: {u.get('support','?'):>4}  F1: {u.get('f1','?')}")

    return result


# ─── EXP2 (standart HuggingFace modeli) ─────────────────────────────────────

def run_evaluation_exp2(
    model_dir: Path,
    data_path: Path,
    out_path:  Path,
    batch_size: int = 64,
    device: str | None = None,
) -> dict:
    import torch
    import pandas as pd
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Cihaz   : {device}")
    print(f"[INFO] Model   : {model_dir}")
    print(f"[INFO] Veri    : {data_path}")

    print("[INFO] Model yükleniyor (eğitim yok)...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model     = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    print(f"[INFO] Sınıf sayısı: {model.config.num_labels}")

    df = pd.read_parquet(str(data_path))
    print(f"[INFO] Test verisi: {len(df)} satır")

    df["true_id"] = df["label"].map(label2id)
    missing = df["true_id"].isna().sum()
    if missing > 0:
        print(f"[WARN] {missing} satır için label eşleştirme başarısız, atlanıyor.")
        df = df[df["true_id"].notna()].copy()
    df["true_id"] = df["true_id"].astype(int)

    texts    = df["text"].tolist()
    true_ids = df["true_id"].tolist()
    max_len  = min(getattr(tokenizer, "model_max_length", 256), 256)

    print(f"[INFO] Inference başlıyor — batch_size={batch_size}...")
    all_preds, all_probs = [], []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc   = tokenizer(batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc   = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(probs.argmax(axis=-1).tolist())
        all_probs.extend(probs.tolist())
        if (i // batch_size) % 20 == 0:
            print(f"  {i}/{len(texts)} işlendi...")

    print(f"[INFO] Inference tamamlandı: {len(all_preds)} tahmin")

    return _compute_and_save(
        true_ids, all_preds, all_probs, id2label, df,
        "exp2", str(model_dir), str(data_path), device, out_path,
    )


# ─── EXP3 / EXP4 (LexiconBERT custom modeli) ────────────────────────────────

def run_evaluation_lexicon(
    model_dir:    Path,
    weights_file: Path,
    data_path:    Path,
    out_path:     Path,
    exp_name:     str,
    batch_size:   int = 64,
    device: str | None = None,
) -> dict:
    import torch
    import pandas as pd
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    # train_exp3_lexicon.py'den gerekli sınıfları import et
    sys.path.insert(0, str(ROOT))
    from train_exp3_lexicon import LexiconBERTModel, LexiconExtractor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Cihaz   : {device}")
    print(f"[INFO] Model   : {model_dir}")
    print(f"[INFO] Ağırlık : {weights_file}")
    print(f"[INFO] Veri    : {data_path}")

    print("[INFO] Tokenizer yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    print("[INFO] LexiconBERTModel yükleniyor (eğitim yok)...")
    model = LexiconBERTModel(
        bert_model_name="dbmdz/bert-base-turkish-cased",
        num_labels=10,
        lexicon_dim=10,
    )
    state = torch.load(str(weights_file), map_location=device)
    # state_dict doğrudan ya da 'model_state_dict' anahtarı altında olabilir
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[INFO] Ağırlıklar yüklendi: {weights_file.name}")

    id2label = LEXICON_ID2LABEL
    label2id = LEXICON_LABEL2ID

    df = pd.read_parquet(str(data_path))
    print(f"[INFO] Test verisi: {len(df)} satır")

    df["true_id"] = df["label"].map(label2id)
    missing = df["true_id"].isna().sum()
    if missing > 0:
        print(f"[WARN] {missing} satır için label eşleştirme başarısız, atlanıyor.")
        df = df[df["true_id"].notna()].copy()
    df["true_id"] = df["true_id"].astype(int)

    texts    = df["text"].tolist()
    true_ids = df["true_id"].tolist()
    max_len  = 128

    # Lexikon özelliklerini tüm veri için önceden hesapla (CPU)
    print("[INFO] Lexikon özellikleri hesaplanıyor...")
    extractor = LexiconExtractor()
    lex_feats_all = extractor.batch_extract(texts, verbose=True)  # (N, 10)
    print()

    print(f"[INFO] Inference başlıyor — batch_size={batch_size}...")
    all_preds, all_probs = [], []

    for i in range(0, len(texts), batch_size):
        batch      = texts[i: i + batch_size]
        batch_lex  = torch.tensor(lex_feats_all[i: i + batch_size], dtype=torch.float32).to(device)
        enc        = tokenizer(batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc        = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            output = model(lexicon_features=batch_lex, **enc)

        probs = F.softmax(output.logits, dim=-1).cpu().numpy()
        all_preds.extend(probs.argmax(axis=-1).tolist())
        all_probs.extend(probs.tolist())
        if (i // batch_size) % 20 == 0:
            print(f"  {i}/{len(texts)} işlendi...")

    print(f"[INFO] Inference tamamlandı: {len(all_preds)} tahmin")

    return _compute_and_save(
        true_ids, all_preds, all_probs, id2label, df,
        exp_name, str(model_dir), str(data_path), device, out_path,
    )


# ─── Dispatch ────────────────────────────────────────────────────────────────

def run_evaluation(
    exp: str,
    batch_size: int = 64,
    device: str | None = None,
    data_path: Path | None = None,
    out_path: Path | None = None,
) -> dict:
    if exp not in MODEL_CONFIGS:
        raise ValueError(f"Geçersiz model: {exp}. Seçenekler: {list(MODEL_CONFIGS)}")

    cfg = MODEL_CONFIGS[exp]
    _data = Path(data_path) if data_path else DATA_PATH
    _out  = Path(out_path)  if out_path  else cfg["out_path"]

    if cfg["model_type"] == "hf":
        return run_evaluation_exp2(
            model_dir=cfg["model_dir"],
            data_path=_data,
            out_path=_out,
            batch_size=batch_size,
            device=device,
        )
    else:
        return run_evaluation_lexicon(
            model_dir=cfg["model_dir"],
            weights_file=cfg["weights_file"],
            data_path=_data,
            out_path=_out,
            exp_name=exp,
            batch_size=batch_size,
            device=device,
        )


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--run",        action="store_true", help="Gerçek inference başlat")
    parser.add_argument("--model",      type=str, default="exp2", choices=["exp2", "exp3", "exp4"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device",     type=str, default=None)
    parser.add_argument("--data",       type=str, default=None,   help="Özel veri dosyası (varsayılan: test_augmented.parquet)")
    parser.add_argument("--out",        type=str, default=None,   help="Özel çıktı JSON yolu")
    args = parser.parse_args()

    if args.run:
        run_evaluation(
            exp=args.model,
            batch_size=args.batch_size,
            device=args.device,
            data_path=args.data,
            out_path=args.out,
        )
    else:
        print("HAZIR — Çalıştırmak için: python evaluate_augmented.py --run --model exp2")
        print("GPU işlemi başlatılmadı.")
