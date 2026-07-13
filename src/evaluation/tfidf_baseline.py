"""
Klasik baseline: TF-IDF (kelime 1-2 gram + karakter 2-5 gram) + Logistic Regression.

    python -m src.evaluation.tfidf_baseline

Transformer'larla aynı veri (gold train), aynı val/test ve aynı metrik/çıktı biçimi
(results/v2/runs/tfidf_lr_s42/). C parametresi yalnızca val macro-F1 ile seçilir.
Deterministiktir; bu yüzden tek çalıştırma yeterlidir.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.data.labels import EMOTIONS, NUM_LABELS, tr_lower
from src.training.train_v2 import DATA_DIR, RUNS_DIR, ece, evaluate, fit_temperature, softmax

C_GRID = [0.5, 1.0, 2.0, 4.0, 8.0]


def main() -> dict:
    t0 = time.time()
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    word = TfidfVectorizer(preprocessor=tr_lower, ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    char = TfidfVectorizer(preprocessor=tr_lower, analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                           sublinear_tf=True, max_features=300_000)
    Xtr = hstack([word.fit_transform(train["text"]), char.fit_transform(train["text"])]).tocsr()
    Xva = hstack([word.transform(val["text"]), char.transform(val["text"])]).tocsr()
    Xte = hstack([word.transform(test["text"]), char.transform(test["text"])]).tocsr()
    ytr, yva, yte = (d["label_id"].to_numpy() for d in (train, val, test))

    grid = {}
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(Xtr, ytr)
        grid[C] = f1_score(yva, clf.predict(Xva), average="macro", labels=range(NUM_LABELS))
        print(f"  C={C}: val F1={grid[C]:.4f}", flush=True)
    best_C = max(grid, key=grid.get)
    clf = LogisticRegression(C=best_C, max_iter=2000).fit(Xtr, ytr)

    # Transformer çıktılarıyla aynı biçim: logit yerine log-olasılık
    val_logits = clf.predict_log_proba(Xva).astype(np.float32)
    test_logits = clf.predict_log_proba(Xte).astype(np.float32)
    T = fit_temperature(val_logits, yva)

    out_dir = RUNS_DIR / "tfidf_lr_s42"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "run": "tfidf_lr_s42", "data": "gold", "seed": 42, "imbalance": "none",
        "model": "TF-IDF (word 1-2 + char_wb 2-5) + LogisticRegression", "model_key": "tfidf_lr",
        "hparams": {"C_grid": C_GRID, "best_C": best_C, "val_f1_by_C": {str(k): round(v, 4) for k, v in grid.items()}},
        "train_size": len(train),
        "best_epoch": None,
        "val_f1_macro": round(grid[best_C], 4),
        "test": evaluate(test, test_logits),
        "calibration": {"temperature": round(T, 4),
                        "test_ece_before": round(ece(softmax(test_logits), yte), 4),
                        "test_ece_after": round(ece(softmax(test_logits / T), yte), 4)},
        "runtime_min": round((time.time() - t0) / 60, 2),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.save(out_dir / "test_logits.npy", test_logits)
    np.save(out_dir / "test_labels.npy", yte)
    np.save(out_dir / "val_logits.npy", val_logits)
    print(f"[tfidf_lr] C={best_C} val F1={metrics['val_f1_macro']} test F1={metrics['test']['f1_macro']} "
          f"acc={metrics['test']['accuracy']}", flush=True)
    return metrics


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
