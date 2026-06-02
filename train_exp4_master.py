"""
Experiment 4 — Master Hybrid (BERT + Lexicon + Cross-Lingual)
=============================================================
Tüm sistemlerin birleştiği Final Mimari:
  ┌─────────────────────────────────────────────────────┐
  │  VERİ    : Cross-Lingual oversample (Exp2 logic)    │
  │  MODEL   : BERTurk [CLS](768) ++ Lexikon(10) = 778  │
  │  KAYIP   : WeightedCrossEntropyLoss                  │
  │  SAMPLER : WeightedRandomSampler + CL boost          │
  │  DONANIM : RTX 5070 Ti  fp16=True batch=64 accum=2   │
  └─────────────────────────────────────────────────────┘

Exp3'ten Farklar:
  - CrossLingualEnhancer ile azınlık sınıflar hedef sayıya çoğaltılır
  - CL kaynaklı minority örneklere ek sample-weight katsayısı uygulanır
  - Eğitim sonrası CLS embedding'leri t-SNE için kaydedilir

Çıktılar (results/exp4/):
  confusion_matrix.png
  training_progress.png
  radar_chart.png          ← Exp1-4 karşılaştırması
  lexicon_importance.png
  metrics_report.json      ← Exp1/2/3 delta karşılaştırması
  test_cls_embeddings.npy  ← t-SNE için (N, 768)
  test_labels.npy          ← (N,)
  best_model/
"""

import os, sys, json, warnings, logging

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import pi
from pathlib import Path
from typing import Optional, Dict, List
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from transformers import (
    AutoTokenizer, TrainingArguments,
    TrainerCallback, EarlyStoppingCallback, set_seed,
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score,
)
import evaluate

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ─── Paylaşılan bileşenleri Exp2 ve Exp3'ten import et ──────────────────────

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train_exp2_crosslingual import (
    CrossLingualEnhancer,
    OVERSAMPLE_TARGETS,
    CL_WEIGHT_MULTIPLIER,
    load_dataframe,
    make_splits,
)
from train_exp3_lexicon import (
    LexiconExtractor,
    LexiconBERTModel,
    LexiconEmotionDataset,
    LexiconDataCollator,
    LexiconWeightedTrainer,
    EpochLogger,
    build_compute_metrics,
    save_confusion_matrix,
    save_training_curves,
    save_lexicon_importance,
    NRC_CATS,
    LEXICON_DIM,
    TURKISH_LEXICON,
)
from training_analytics import run_full_analytics

# ─── Sabitler ───────────────────────────────────────────────────────────────

RESULTS    = ROOT / "results" / "exp4"
EXP1_JSON  = ROOT / "results" / "exp1" / "metrics_report.json"
EXP2_JSON  = ROOT / "results" / "exp2" / "metrics_report.json"
EXP3_JSON  = ROOT / "results" / "exp3" / "metrics_report.json"
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
SEED       = 42

EMOTIONS   = [
    "mutluluk", "üzüntü", "öfke", "korku",
    "şaşkınlık", "tiksinti", "sevgi", "nötr", "utanç", "gurur",
]
LABEL2ID   = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL   = {i: e for i, e in enumerate(EMOTIONS)}
NUM_LABELS = 10


# ─── Ağırlık Hesabı ─────────────────────────────────────────────────────────

def compute_class_weights(label_ids: np.ndarray) -> torch.Tensor:
    counts = np.bincount(label_ids, minlength=NUM_LABELS).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    return torch.tensor(len(label_ids) / (counts * NUM_LABELS), dtype=torch.float32)


def compute_sample_weights(
    df: pd.DataFrame,
    cl_source: str = "goemotions_tr",
    cl_multiplier: Optional[float] = None,
) -> np.ndarray:
    # Varsayılan tanım anında bağlanmasın diye çağrı anında okunur (--cl-multiplier için)
    if cl_multiplier is None:
        cl_multiplier = CL_WEIGHT_MULTIPLIER
    label_ids = df["label_id"].values
    counts    = np.bincount(label_ids, minlength=NUM_LABELS).astype(float)
    counts    = np.where(counts == 0, 1.0, counts)
    class_w   = len(label_ids) / (counts * NUM_LABELS)
    sample_w  = class_w[label_ids].astype(float)

    # CL minority boost
    minority_ids = {LABEL2ID[e] for e in ["gurur", "utanç", "tiksinti", "korku"]}
    is_cl        = df["source"].str.startswith(cl_source).values
    is_minority  = np.isin(label_ids, list(minority_ids))
    sample_w[is_cl & is_minority] *= cl_multiplier
    return sample_w


# ─── CLS Embedding Çıkarıcı (t-SNE için) ────────────────────────────────────

def extract_cls_embeddings(
    model: LexiconBERTModel,
    dataset: LexiconEmotionDataset,
    tokenizer,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """
    Test seti için BERT [CLS] vektörlerini çıkarır.
    Döndürür: (N, 768) ndarray
    """
    model.eval()
    collator = LexiconDataCollator(tokenizer=tokenizer)
    loader   = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collator, pin_memory=True,
    )
    all_cls = []
    print(f"  CLS embedding çıkarılıyor ({len(dataset)} örnek)...")
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels", None)
            batch.pop("lexicon_features", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            bert_out = model.bert(**batch)
            cls = bert_out.last_hidden_state[:, 0, :].cpu().numpy()
            all_cls.append(cls)
    embeddings = np.concatenate(all_cls, axis=0)
    print(f"  Embeddings: {embeddings.shape}")
    return embeddings


# ─── Görselleştirme ──────────────────────────────────────────────────────────

def save_radar_chart(exp4_per_class: dict, save_path: Path):
    """Exp1-4 için 10 sınıf F1 radar (örümcek) grafiği."""

    def _load(p: Path) -> dict:
        if not p.exists():
            return {}
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return {e: d.get("per_class", {}).get(e, {}).get("f1", 0.0) for e in EMOTIONS}

    data = {
        "Exp1 Baseline":       (_load(EXP1_JSON), "#95a5a6", "--"),
        "Exp2 Cross-Lingual":  (_load(EXP2_JSON), "#3498db", "-"),
        "Exp3 Lexicon":        (_load(EXP3_JSON), "#8e44ad", "-"),
        "Exp4 Master Hybrid":  (exp4_per_class,   "#e74c3c", "-"),
    }

    labels = [e[:9] for e in EMOTIONS]
    N      = len(labels)
    angles = [n / float(N) * 2 * pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(11, 11), subplot_kw={"polar": True})
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=8, color="grey")
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.4)

    for label, (scores, color, ls) in data.items():
        if not scores:
            continue
        vals  = [scores.get(e, 0.0) for e in EMOTIONS] + [scores.get(EMOTIONS[0], 0.0)]
        lw    = 3.0 if "Master" in label else 1.8
        alpha = 0.15 if "Master" in label else 0.05
        ax.plot(angles, vals, linewidth=lw, linestyle=ls, color=color, label=label)
        ax.fill(angles, vals, alpha=alpha, color=color)

    ax.set_title(
        "Experiment 1 / 2 / 3 / 4 — 10 Sınıf F1 Karşılaştırması\n(Radar Chart)",
        size=14, fontweight="bold", y=1.1,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.2), fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Radar chart: {save_path}")


def save_metrics_report(
    y_true, y_pred, history: dict, lex_stats: dict, save_path: Path,
) -> dict:
    report_dict = classification_report(
        y_true, y_pred, target_names=EMOTIONS, output_dict=True, zero_division=0,
    )
    per_class = {
        emo: {
            "precision": round(report_dict[emo]["precision"], 4),
            "recall":    round(report_dict[emo]["recall"],    4),
            "f1":        round(report_dict[emo]["f1-score"],  4),
            "support":   int(report_dict[emo]["support"]),
        }
        for emo in EMOTIONS if emo in report_dict
    }

    output = {
        "experiment":       "exp4_master_hybrid",
        "model":            f"{MODEL_NAME} + Lexicon (NRC categories) + Cross-Lingual Data",
        "architecture":     "BERTurk[CLS](768) ++ Lexikon(10) -> Linear(778,10)",
        "data_enhancement": {
            "cross_lingual_source": "goemotions_tr",
            "oversample_targets":   OVERSAMPLE_TARGETS,
            "cl_weight_multiplier": CL_WEIGHT_MULTIPLIER,
        },
        "test_accuracy":    round(float(accuracy_score(y_true, y_pred)), 4),
        "test_f1_macro":    round(float(f1_score(y_true, y_pred, average="macro",    zero_division=0)), 4),
        "test_f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "per_class":        per_class,
        "lexicon_analysis": lex_stats,
        "training_history": history,
        "comparison":       {},
    }

    for exp_name, exp_path in [("exp1",EXP1_JSON),("exp2",EXP2_JSON),("exp3",EXP3_JSON)]:
        if exp_path.exists():
            with open(exp_path, encoding="utf-8") as f:
                ref = json.load(f)
            delta = output["test_f1_macro"] - ref.get("test_f1_macro", 0)
            output["comparison"][exp_name] = {
                "f1_macro_ref":   ref.get("test_f1_macro"),
                "f1_macro_exp4":  output["test_f1_macro"],
                "f1_macro_delta": round(delta, 4),
                "per_class_delta": {
                    emo: round(
                        per_class.get(emo, {}).get("f1", 0.0) -
                        ref.get("per_class", {}).get(emo, {}).get("f1", 0.0),
                        4
                    ) for emo in EMOTIONS
                },
            }
        else:
            output["comparison"][exp_name] = {"note": f"{exp_path.name} bulunamadi"}

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[VIZ] Metrics report: {save_path}")

    # Konsol özet
    print("\n" + "─"*75)
    print(f"  {'Duygu':12s} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Destek':>8}  "
          f"{'vsExp1':>7} {'vsExp2':>7} {'vsExp3':>7}")
    print("─"*75)
    for emo in EMOTIONS:
        m  = per_class.get(emo, {})
        ds = [output["comparison"].get(f"exp{i}",{}).get("per_class_delta",{}).get(emo,0.0)
              for i in [1,2,3]]
        row = (f"  {emo:12s} {m.get('precision',0):8.4f} {m.get('recall',0):8.4f} "
               f"{m.get('f1',0):8.4f} {m.get('support',0):8d}")
        for d in ds:
            s = ("+" if d>=0 else "") + f"{d:.3f}"
            row += f"  {s:>7}"
        print(row)
    print("─"*75)
    macro_row = (f"  {'MACRO':12s} {'':>8} {'':>8} {output['test_f1_macro']:8.4f} {'':>8}")
    for exp_n in ["exp1","exp2","exp3"]:
        dm = output["comparison"].get(exp_n, {}).get("f1_macro_delta", None)
        s  = (("+" if dm>=0 else "")+f"{dm:.4f}") if isinstance(dm, float) else "  N/A"
        macro_row += f"  {s:>7}"
    print(macro_row)
    print(f"  {'ACCURACY':12s} {'':>8} {'':>8} {output['test_accuracy']:8.4f}")
    b = lex_stats.get('bert_contribution', 0.0)
    print(f"\n  [LEXIKON] Yalnizca Lexikon F1-Macro : {lex_stats.get('lexicon_only_f1_macro',0.0):.4f}")
    print(f"  [LEXIKON] Hibrit F1-Macro            : {lex_stats.get('hybrid_f1_macro',0.0):.4f}")
    print(f"  [LEXIKON] BERT katkisi               : {'+' if b>=0 else ''}{b:.4f}")
    print("─"*75 + "\n")

    return output


# ─── Ana Eğitim Fonksiyonu ───────────────────────────────────────────────────

def train(epochs: int = 5, max_per_class: int = None):
    set_seed(SEED)
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*65)
    print("  EXPERIMENT 4 — MASTER HYBRID (BERT + LEXICON + CROSS-LINGUAL)")
    print("  Model    : " + MODEL_NAME)
    print("  Mimari   : BERTurk[CLS](768) ++ Lexikon(10) -> Linear(778,10)")
    print("  Veri     : Cross-Lingual oversample + WeightedRandomSampler")
    print("="*65)

    # ── 1. Veri ──────────────────────────────────────────────────
    df = load_dataframe()
    df_train_base, df_val, df_test = make_splits(df)

    # Hızlı eğitim: çoğunluk sınıfları cap'le, azınlıklar CL ile büyüyecek
    if max_per_class:
        df_train_base = (
            df_train_base.groupby("label_id", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max_per_class), random_state=42))
            .reset_index(drop=True)
        )
        print(f"[FAST] max_per_class={max_per_class} -> base: {len(df_train_base):,} ornek")

    # ── 2. Cross-Lingual Enhancement ─────────────────────────────
    enhancer = CrossLingualEnhancer()
    df_train = enhancer.enhance(df_train_base)
    enhancer.log_distribution(df_train, "EGİTİM SETİ (CROSS-LINGUAL SONRASI)")

    # ── 3. Lexikon & Tokenizer ────────────────────────────────────
    print("\n[LEXIKON] Extractor baslatiliyor...")
    extractor = LexiconExtractor()
    print(f"  Sozluk: {len(extractor.lexicon)} kelime | Kok indeksi: {len(extractor._stem_index)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── 4. Dataset ────────────────────────────────────────────────
    print("\n[DATASET] Tokenizasyon + Lexikon ozellik hesaplama...")
    train_ds = LexiconEmotionDataset(df_train, tokenizer, extractor, max_length=128)
    val_ds   = LexiconEmotionDataset(df_val,   tokenizer, extractor, max_length=128)
    test_ds  = LexiconEmotionDataset(df_test,  tokenizer, extractor, max_length=128)

    # ── 5. Ağırlıklar ─────────────────────────────────────────────
    sample_weights = compute_sample_weights(df_train)
    class_weights  = compute_class_weights(df_train["label_id"].values)

    print("\n[AGIRLIK] Sinif agirlikları:")
    for i, emo in enumerate(EMOTIONS):
        print(f"  {emo:12s}: {float(class_weights[i]):.4f}")

    # ── 6. Model ──────────────────────────────────────────────────
    print(f"\n[MODEL] Hibrit model yukleniyor...")
    model = LexiconBERTModel(
        bert_model_name=MODEL_NAME,
        num_labels=NUM_LABELS,
        lexicon_dim=LEXICON_DIM,
        dropout=0.1,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Egitim parametresi: {trainable:,}")
    print(f"  Classifier boyutu : {768 + LEXICON_DIM} -> {NUM_LABELS}  (BERT+Lex concat)")

    # ── 7. Training Arguments ─────────────────────────────────────
    best_model_dir  = RESULTS / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = len(df_train) // (64 * 2)
    warmup_steps    = int(0.1 * steps_per_epoch * epochs)

    training_args = TrainingArguments(
        output_dir=str(best_model_dir),
        # ── RTX 5070 Ti ──
        fp16=True, bf16=False,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=2,
        gradient_checkpointing=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # ── Eğitim ──
        num_train_epochs=epochs,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        # ── Değerlendirme ──
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=100,
        logging_first_step=True,
        report_to=[],
        seed=SEED,
        remove_unused_columns=False,
    )

    epoch_logger  = EpochLogger()
    data_collator = LexiconDataCollator(tokenizer=tokenizer)

    trainer = LexiconWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(),
        train_sample_weights=sample_weights,
        class_weights=class_weights,
        callbacks=[epoch_logger, EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── 8. Eğitim ─────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  EGİTİM BASLIYOR")
    print(f"  Efektif batch : {64*2} | Adım/epoch : ~{steps_per_epoch}")
    print(f"  Warmup adımı  : {warmup_steps}")
    print(f"  CL Boost      : {CL_WEIGHT_MULTIPLIER}x (minority cross-lingual)")
    print(f"  Lexikon boyutu: {LEXICON_DIM} NRC kategorisi")
    print("="*65 + "\n")

    train_result = trainer.train()
    print(f"\n[TAMAMLANDI] Adim:{train_result.global_step} | "
          f"Kayıp:{train_result.training_loss:.4f} | "
          f"Sure:{train_result.metrics['train_runtime']/60:.1f}dk")

    # ── 9. Test Değerlendirmesi ───────────────────────────────────
    print("\n[TEST] Test seti degerlendiriliyor...")
    test_output = trainer.predict(test_ds)
    y_pred = np.argmax(test_output.predictions, axis=-1)
    y_true = test_output.label_ids
    test_lex = test_ds.lex_feats

    # ── 10. CLS Embedding Çıkar (t-SNE için) ──────────────────────
    print("\n[EMBED] CLS embedding'ler cıkarılıyor (t-SNE icin)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    try:
        cls_embeddings = extract_cls_embeddings(model, test_ds, tokenizer, device)
        np.save(RESULTS / "test_cls_embeddings.npy", cls_embeddings)
        np.save(RESULTS / "test_labels.npy",         y_true)
        np.save(RESULTS / "test_preds.npy",          y_pred)
        print(f"  Kaydedildi: {RESULTS}/test_cls_embeddings.npy  shape={cls_embeddings.shape}")
    except Exception as e:
        print(f"  [WARN] Embedding cıkarımı basarisiz: {e}")

    # ── 11. Görseller ─────────────────────────────────────────────
    print("\n[VIZ] Gorseller olusturuluyor...")
    save_confusion_matrix(y_true, y_pred, RESULTS / "confusion_matrix.png")
    save_training_curves(epoch_logger.history, RESULTS / "training_progress.png")
    lex_stats = save_lexicon_importance(test_lex, y_true, y_pred, RESULTS / "lexicon_importance.png")
    report    = save_metrics_report(y_true, y_pred, epoch_logger.history, lex_stats,
                                    RESULTS / "metrics_report.json")
    exp4_f1s = {e: (v["f1"] if isinstance(v, dict) else v) for e, v in report["per_class"].items()}
    save_radar_chart(exp4_f1s, RESULTS / "radar_chart.png")

    # ── 11b. Kapsamlı Analitik ────────────────────────────────────
    run_full_analytics(
        y_true=y_true,
        y_pred=y_pred,
        logits_or_probs=test_output.predictions,
        epoch_logger=epoch_logger,
        train_result=train_result,
        out_dir=RESULTS,
        exp_name="Exp4 Master Hybrid",
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        tokenizer=tokenizer,
        extra_info={
            "lexicon_dim": LEXICON_DIM,
            "cl_weight_multiplier": CL_WEIGHT_MULTIPLIER,
            "oversample_targets": OVERSAMPLE_TARGETS,
            "lexicon_analysis": lex_stats,
        },
    )

    # ── 12. Model Kaydet ──────────────────────────────────────────
    torch.save(model.state_dict(), best_model_dir / "master_hybrid_weights.pt")
    tokenizer.save_pretrained(str(best_model_dir))
    with open(best_model_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment":    "exp4_master_hybrid",
            "bert_model":    MODEL_NAME,
            "num_labels":    NUM_LABELS,
            "lexicon_dim":   LEXICON_DIM,
            "nrc_categories": NRC_CATS,
            "architecture":  "BERTurk[CLS](768) ++ Lexikon(10) -> Linear(778,10)",
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[KAYIT] Model : {best_model_dir}/master_hybrid_weights.pt")
    print(f"[KAYIT] Sonuc : {RESULTS}")
    return trainer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exp4 — Master Hybrid")
    parser.add_argument("--rebuild-data",  action="store_true")
    parser.add_argument("--epochs",        type=int,   default=5)
    parser.add_argument("--max-per-class", type=int,   default=None,
                        help="Sinif basi max ornek (hizli egitim, orn: 10000)")
    parser.add_argument("--cl-multiplier", type=float, default=CL_WEIGHT_MULTIPLIER)
    args = parser.parse_args()

    if args.cl_multiplier != CL_WEIGHT_MULTIPLIER:
        CL_WEIGHT_MULTIPLIER = args.cl_multiplier
        print(f"[CONFIG] CL multiplier: {CL_WEIGHT_MULTIPLIER}")

    splits_dir = ROOT / "data" / "splits" / "master_splits"
    if args.rebuild_data or not (splits_dir / "train.parquet").exists():
        df = load_dataframe()
        make_splits(df)

    train(epochs=args.epochs, max_per_class=args.max_per_class)
