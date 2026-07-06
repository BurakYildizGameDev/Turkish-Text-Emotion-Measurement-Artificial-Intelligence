"""
training_analytics.py
=====================
Kapsamlı eğitim analitik, izleme ve görselleştirme modülü.
Tüm deney dosyaları (exp1–exp4) tarafından import edilir.

Her eğitim çalışmasında otomatik kaydedilen çıktılar:
  step_loss_curve.png          — adım bazlı loss + LR schedule (twin axis)
  per_class_f1_evolution.png   — epoch × sınıf F1 ısı haritası + çizgi
  roc_curves.png               — OvR ROC eğrileri (10 sınıf + micro/macro AUC)
  pr_curves.png                — Precision-Recall eğrileri + Average Precision
  calibration_curves.png       — güvenilirlik diyagramları + ECE
  confidence_analysis.png      — güven skoru dağılımı (doğru vs yanlış)
  data_distribution.png        — train/val/test sınıf dengesizliği
  token_length_dist.png        — metin token uzunluğu dağılımı + CDF
  test_probs.npy               — softmax çıkışları [N, 10]
  test_preds.npy               — tahmin sınıfları [N]
  test_labels.npy              — gerçek etiketler [N]
  step_history.json            — adım bazlı loss/LR log
  epoch_history.json           — epoch bazlı tam metrik geçmişi
  training_summary.json        — kapsamlı özet (ROC AUC, AP, ECE, timing, ...)
"""

import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any

from scipy.special import softmax as _scipy_softmax
from transformers import TrainerCallback

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score,
    f1_score, accuracy_score, classification_report,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize

# ─── Paylaşılan Sabitler ────────────────────────────────────────────────────

EMOTIONS = [
    "mutluluk", "üzüntü", "öfke", "korku",
    "şaşkınlık", "tiksinti", "sevgi", "nötr", "utanç", "gurur",
]
NUM_LABELS = 10
EMOTION_COLORS = [
    "#e74c3c", "#3498db", "#e67e22", "#9b59b6", "#1abc9c",
    "#e91e63", "#f39c12", "#95a5a6", "#d35400", "#27ae60",
]


# ─── EnhancedEpochLogger ────────────────────────────────────────────────────

class EnhancedEpochLogger(TrainerCallback):
    """
    Kapsamlı eğitim metrik kaydedici. Tüm exp dosyaları tarafından kullanılır.

    Epoch bazlı (history):
      epoch, train_loss, val_loss, val_acc, val_f1_macro, val_f1_weighted,
      val_f1_<her_duygu> (10 sınıf)

    Adım bazlı (step_history):
      step, loss, learning_rate, epoch_frac
      (her logging_steps = 100 adımda bir, ~kayıt)
    """

    def __init__(self):
        self.history: Dict[str, List] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1_macro": [],
            "val_f1_weighted": [],
            **{f"val_f1_{emo}": [] for emo in EMOTIONS},
        }
        self.step_history: Dict[str, List] = {
            "step": [],
            "loss": [],
            "learning_rate": [],
            "epoch_frac": [],
        }
        self.epoch_times: List[float] = []
        self._last_train_loss: Optional[float] = None
        self._epoch_start: float = 0.0
        self._train_start: float = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self._train_start = time.time()
        self._epoch_start = time.time()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_start = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_times.append(round(time.time() - self._epoch_start, 2))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        loss = logs.get("loss")
        if loss is not None:
            self._last_train_loss = loss
            self.step_history["step"].append(state.global_step)
            self.step_history["loss"].append(loss)
            self.step_history["learning_rate"].append(logs.get("learning_rate", 0.0))
            self.step_history["epoch_frac"].append(state.epoch or 0.0)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        epoch = round(state.epoch or 0, 1)
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(self._last_train_loss or float("nan"))
        self.history["val_loss"].append(metrics.get("eval_loss", float("nan")))
        self.history["val_acc"].append(metrics.get("eval_accuracy", float("nan")))
        self.history["val_f1_macro"].append(metrics.get("eval_f1_macro", float("nan")))
        self.history["val_f1_weighted"].append(metrics.get("eval_f1_weighted", float("nan")))
        for emo in EMOTIONS:
            v = metrics.get(f"eval_f1_{emo}", float("nan"))
            self.history[f"val_f1_{emo}"].append(v)

        print(
            f"  Epoch {epoch:4.1f} | "
            f"train_loss={self._last_train_loss or 0:.4f} | "
            f"val_loss={metrics.get('eval_loss', 0):.4f} | "
            f"val_acc={metrics.get('eval_accuracy', 0):.4f} | "
            f"f1_macro={metrics.get('eval_f1_macro', 0):.4f}"
        )


# ─── 1. Adım Bazlı Loss + LR Eğrisi ────────────────────────────────────────

def save_step_loss_curve(step_history: dict, save_path: Path, exp_name: str = ""):
    steps = step_history["step"]
    losses = step_history["loss"]
    lrs = step_history["learning_rate"]
    if not steps:
        print("[VIZ] step_history boş, adım loss eğrisi atlandı.")
        return

    losses_arr = np.array(losses, dtype=float)
    window = min(10, len(losses_arr))
    kernel = np.ones(window) / window
    smoothed = np.convolve(losses_arr, kernel, mode="valid")
    smooth_steps = steps[window - 1:]

    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax1.plot(steps, losses, alpha=0.25, color="#e74c3c", linewidth=0.7, label="Loss (ham)")
    ax1.plot(smooth_steps, smoothed, color="#e74c3c", linewidth=2.2,
             label=f"Loss (hareketli ort. n={window})")
    ax1.set_xlabel("Adım (Global Step)", fontsize=12)
    ax1.set_ylabel("Training Loss", color="#c0392b", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax1.grid(True, alpha=0.25)
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.plot(steps, lrs, color="#3498db", linewidth=1.6, linestyle="--",
             alpha=0.85, label="Learning Rate")
    ax2.set_ylabel("Learning Rate", color="#2980b9", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#2980b9")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1e}"))
    ax2.set_ylim(bottom=0)

    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbl1 + lbl2, loc="upper right", fontsize=9, framealpha=0.9)

    title = (f"[{exp_name}] " if exp_name else "") + "Adım Bazlı Loss & Learning Rate Schedule"
    plt.title(title, fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Step loss curve: {save_path}")


# ─── 2. Per-Class F1 Evrimi ─────────────────────────────────────────────────

def save_per_class_f1_evolution(history: dict, save_path: Path, exp_name: str = ""):
    epochs = history["epoch"]
    if not epochs:
        print("[VIZ] Epoch history boş, F1 evrimi atlandı.")
        return

    # matrix: (n_epochs, 10)
    matrix = np.array([
        [history.get(f"val_f1_{emo}", [float("nan")] * len(epochs))[i]
         for emo in EMOTIONS]
        for i in range(len(epochs))
    ], dtype=float)

    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.1, 1], hspace=0.45)

    # ── Üst: ısı haritası ──
    ax1 = fig.add_subplot(gs[0])
    filled = np.where(np.isnan(matrix), 0, matrix)
    im = ax1.imshow(filled.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax1.set_xticks(range(len(epochs)))
    ax1.set_xticklabels([f"E{e:.0f}" for e in epochs], fontsize=10)
    ax1.set_yticks(range(NUM_LABELS))
    ax1.set_yticklabels(EMOTIONS, fontsize=10)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_title("Per-Class F1 Evrimi — Isı Haritası", fontweight="bold", fontsize=12)
    for i in range(len(epochs)):
        for j in range(NUM_LABELS):
            v = filled[i, j]
            if not np.isnan(matrix[i, j]):
                color = "black" if 0.3 < v < 0.75 else "white"
                ax1.text(i, j, f"{v:.2f}", ha="center", va="center",
                         color=color, fontsize=8, fontweight="bold")
    plt.colorbar(im, ax=ax1, shrink=0.85, label="F1 Score")

    # ── Alt: çizgi grafik ──
    ax2 = fig.add_subplot(gs[1])
    for j, (emo, color) in enumerate(zip(EMOTIONS, EMOTION_COLORS)):
        vals = matrix[:, j]
        valid = ~np.isnan(vals)
        if valid.any():
            ax2.plot(np.array(epochs)[valid], vals[valid],
                     "o-", color=color, linewidth=1.9, markersize=5,
                     label=emo[:8], alpha=0.88)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("F1 Score", fontsize=11)
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_title("Per-Class F1 Evrimi — Çizgi Grafik", fontweight="bold", fontsize=12)
    ax2.legend(loc="lower right", ncol=5, fontsize=8, framealpha=0.92)
    ax2.grid(True, alpha=0.3)

    suptitle = (f"[{exp_name}] " if exp_name else "") + "Per-Class F1 Eğitim Evrimi"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Per-class F1 evolution: {save_path}")


# ─── 3. ROC Eğrileri ────────────────────────────────────────────────────────

def save_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: Path,
    exp_name: str = "",
) -> dict:
    y_bin = label_binarize(y_true, classes=list(range(NUM_LABELS)))

    fpr_d, tpr_d, auc_d = {}, {}, {}
    for i in range(NUM_LABELS):
        fpr_d[i], tpr_d[i], _ = roc_curve(y_bin[:, i], y_probs[:, i])
        auc_d[i] = auc(fpr_d[i], tpr_d[i])

    fpr_d["micro"], tpr_d["micro"], _ = roc_curve(y_bin.ravel(), y_probs.ravel())
    auc_d["micro"] = auc(fpr_d["micro"], tpr_d["micro"])

    all_fpr = np.unique(np.concatenate([fpr_d[i] for i in range(NUM_LABELS)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_LABELS):
        mean_tpr += np.interp(all_fpr, fpr_d[i], tpr_d[i])
    mean_tpr /= NUM_LABELS
    auc_d["macro"] = auc(all_fpr, mean_tpr)

    fig, axes = plt.subplots(2, 5, figsize=(23, 9))
    axes_flat = axes.flatten()

    for i, (emo, color) in enumerate(zip(EMOTIONS, EMOTION_COLORS)):
        ax = axes_flat[i]
        ax.plot(fpr_d[i], tpr_d[i], color=color, linewidth=2.2,
                label=f"AUC = {auc_d[i]:.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.45)
        ax.fill_between(fpr_d[i], tpr_d[i], alpha=0.10, color=color)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.05])
        ax.set_xlabel("FPR", fontsize=9)
        ax.set_ylabel("TPR", fontsize=9)
        ax.set_title(emo, fontweight="bold", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.22)

    fig.suptitle(
        (f"[{exp_name}] " if exp_name else "") +
        f"ROC Eğrileri (One-vs-Rest) | "
        f"Micro-AUC={auc_d['micro']:.4f}  Macro-AUC={auc_d['macro']:.4f}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] ROC curves: {save_path}")

    return {
        **{emo: round(auc_d[i], 4) for i, emo in enumerate(EMOTIONS)},
        "micro_avg": round(auc_d["micro"], 4),
        "macro_avg": round(auc_d["macro"], 4),
    }


# ─── 4. Precision-Recall Eğrileri ───────────────────────────────────────────

def save_pr_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: Path,
    exp_name: str = "",
) -> dict:
    y_bin = label_binarize(y_true, classes=list(range(NUM_LABELS)))

    prec_d, rec_d, ap_d = {}, {}, {}
    for i in range(NUM_LABELS):
        prec_d[i], rec_d[i], _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        ap_d[i] = average_precision_score(y_bin[:, i], y_probs[:, i])

    prec_d["micro"], rec_d["micro"], _ = precision_recall_curve(
        y_bin.ravel(), y_probs.ravel()
    )
    ap_d["micro"] = average_precision_score(y_bin, y_probs, average="micro")
    mean_ap = float(np.mean([ap_d[i] for i in range(NUM_LABELS)]))

    fig, axes = plt.subplots(2, 5, figsize=(23, 9))
    axes_flat = axes.flatten()

    for i, (emo, color) in enumerate(zip(EMOTIONS, EMOTION_COLORS)):
        ax = axes_flat[i]
        ax.step(rec_d[i], prec_d[i], where="post", color=color, linewidth=2.2,
                label=f"AP = {ap_d[i]:.3f}")
        ax.fill_between(rec_d[i], prec_d[i], step="post", alpha=0.10, color=color)
        baseline = y_bin[:, i].sum() / len(y_true)
        ax.axhline(baseline, color="gray", linestyle="--", linewidth=0.85,
                   alpha=0.65, label=f"Baseline={baseline:.2f}")
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.05])
        ax.set_xlabel("Recall", fontsize=9)
        ax.set_ylabel("Precision", fontsize=9)
        ax.set_title(emo, fontweight="bold", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.22)

    fig.suptitle(
        (f"[{exp_name}] " if exp_name else "") +
        f"Precision-Recall Eğrileri | "
        f"Micro-AP={ap_d['micro']:.4f}  Mean-AP={mean_ap:.4f}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] PR curves: {save_path}")

    return {
        **{emo: round(ap_d[i], 4) for i, emo in enumerate(EMOTIONS)},
        "micro_avg": round(ap_d["micro"], 4),
        "mean_ap": round(mean_ap, 4),
    }


# ─── 5. Kalibrasyon Eğrileri ────────────────────────────────────────────────

def save_calibration_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: Path,
    exp_name: str = "",
    n_bins: int = 10,
) -> dict:
    y_bin = label_binarize(y_true, classes=list(range(NUM_LABELS)))

    fig, axes = plt.subplots(2, 5, figsize=(23, 9))
    axes_flat = axes.flatten()
    ece_scores = {}

    for i, (emo, color) in enumerate(zip(EMOTIONS, EMOTION_COLORS)):
        ax = axes_flat[i]
        try:
            prob_true, prob_pred = calibration_curve(
                y_bin[:, i], y_probs[:, i], n_bins=n_bins, strategy="uniform"
            )
        except ValueError:
            prob_true, prob_pred = np.array([0.0, 1.0]), np.array([0.0, 1.0])

        ece = float(np.mean(np.abs(prob_true - prob_pred)))
        ece_scores[emo] = round(ece, 4)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.55, label="Mükemmel")
        ax.plot(prob_pred, prob_true, "o-", color=color, linewidth=2.0,
                markersize=6, label=f"ECE={ece:.3f}")
        ax.fill_between(prob_pred, prob_pred, prob_true,
                        alpha=0.13, color=color)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.05])
        ax.set_xlabel("Tahmin Güveni", fontsize=9)
        ax.set_ylabel("Gerçek Oran", fontsize=9)
        ax.set_title(emo, fontweight="bold", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.22)

    mean_ece = float(np.mean(list(ece_scores.values())))
    fig.suptitle(
        (f"[{exp_name}] " if exp_name else "") +
        f"Kalibrasyon Eğrileri (Güvenilirlik Diyagramı) | Ort. ECE={mean_ece:.4f}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Calibration curves: {save_path}")

    ece_scores["mean_ece"] = round(mean_ece, 4)
    return ece_scores


# ─── 6. Güven Skoru Analizi ─────────────────────────────────────────────────

def save_confidence_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    save_path: Path,
    exp_name: str = "",
) -> dict:
    max_probs = y_probs.max(axis=1)
    correct = (y_pred == y_true)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    # ── Doğru vs Yanlış güven histogram ──
    ax = axes[0]
    bins = np.linspace(0, 1, 26)
    ax.hist(max_probs[correct],  bins=bins, alpha=0.72, color="#27ae60",
            label=f"Doğru  n={correct.sum():,}", density=True)
    ax.hist(max_probs[~correct], bins=bins, alpha=0.72, color="#e74c3c",
            label=f"Yanlış n={(~correct).sum():,}", density=True)
    ax.axvline(max_probs[correct].mean(), color="#1e8449", linestyle="--",
               linewidth=1.8, label=f"Doğru ort={max_probs[correct].mean():.3f}")
    ax.axvline(max_probs[~correct].mean(), color="#922b21", linestyle="--",
               linewidth=1.8, label=f"Yanlış ort={max_probs[~correct].mean():.3f}")
    ax.set_xlabel("Max Softmax Güveni", fontsize=11)
    ax.set_ylabel("Yoğunluk", fontsize=11)
    ax.set_title("Doğru vs Yanlış Tahmin Güvenleri", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.28)

    # ── Sınıf başına ortalama doğru tahmin güveni ──
    ax = axes[1]
    class_conf = []
    for i in range(NUM_LABELS):
        mask = (y_true == i) & correct
        class_conf.append(max_probs[mask].mean() if mask.any() else 0.0)
    bars = ax.bar(range(NUM_LABELS), class_conf,
                  color=EMOTION_COLORS, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, class_conf):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(NUM_LABELS))
    ax.set_xticklabels([e[:7] for e in EMOTIONS], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Ort. Güven (Doğru Tahminler)", fontsize=10)
    ax.set_title("Sınıf Başına Doğru Tahmin Güveni", fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.28, axis="y")

    # ── Güven persentilleri + overconfidence oranı ──
    ax = axes[2]
    pcts = [10, 25, 50, 75, 90, 95, 99]
    vals = [np.percentile(max_probs, p) for p in pcts]
    bars2 = ax.bar(range(len(pcts)), vals,
                   color=EMOTION_COLORS[:len(pcts)], alpha=0.85, edgecolor="white")
    for bar, val in zip(bars2, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(pcts)))
    ax.set_xticklabels([f"P{p}" for p in pcts], fontsize=10)
    ax.set_ylabel("Güven Eşiği", fontsize=11)
    ax.set_title("Güven Skoru Persentilleri", fontweight="bold")
    ax.set_ylim(0, 1.18)
    ax.grid(True, alpha=0.28, axis="y")
    overconf_rate = float((max_probs[~correct] > 0.8).mean())
    ax.text(0.5, 0.98, f"Yanlış & güven>0.8: %{overconf_rate*100:.1f}",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=10, color="#c0392b", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#e74c3c", alpha=0.9))

    suptitle = (f"[{exp_name}] " if exp_name else "") + "Güven Skoru Analizi"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Confidence analysis: {save_path}")

    return {
        "mean_confidence_correct":  round(float(max_probs[correct].mean()), 4),
        "mean_confidence_wrong":    round(float(max_probs[~correct].mean()), 4),
        "median_confidence":        round(float(np.median(max_probs)), 4),
        "p90_confidence":           round(float(np.percentile(max_probs, 90)), 4),
        "overconfidence_rate_0.8":  round(overconf_rate, 4),
    }


# ─── 7. Veri Sınıf Dağılımı ─────────────────────────────────────────────────

def save_data_distribution(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    save_path: Path,
    label_col: str = "label_id",
):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    splits = [
        ("Train", df_train, "#3498db"),
        ("Val",   df_val,   "#2ecc71"),
        ("Test",  df_test,  "#e74c3c"),
    ]
    for ax, (name, df, color) in zip(axes, splits):
        counts = [int((df[label_col] == i).sum()) for i in range(NUM_LABELS)]
        total = sum(counts)
        bars = ax.barh(EMOTIONS, counts, color=color, alpha=0.82, edgecolor="white")
        max_c = max(counts) if counts else 1
        for bar, cnt in zip(bars, counts):
            pct = cnt / total * 100 if total else 0
            ax.text(
                bar.get_width() + max_c * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{cnt:,}  ({pct:.1f}%)",
                va="center", fontsize=9,
            )
        ax.set_title(f"{name} Seti  (Toplam: {total:,})", fontweight="bold", fontsize=12)
        ax.set_xlabel("Örnek Sayısı", fontsize=10)
        ax.set_xlim(0, max_c * 1.35)
        ax.grid(True, alpha=0.28, axis="x")

    plt.suptitle("Veri Seti Sınıf Dağılımı", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Data distribution: {save_path}")


# ─── 8. Token Uzunluğu Dağılımı ─────────────────────────────────────────────

def save_token_length_distribution(
    df: pd.DataFrame,
    tokenizer,
    save_path: Path,
    text_col: str = "text",
    max_length: int = 128,
) -> dict:
    print("[VIZ] Token uzunlukları hesaplanıyor (maks 5000 örnek)...")
    sample = df.sample(min(5000, len(df)), random_state=42).copy()

    lengths = []
    for text in sample[text_col].tolist():
        try:
            ids = tokenizer.encode(str(text), add_special_tokens=True)
            lengths.append(len(ids))
        except Exception:
            lengths.append(0)
    lengths = np.array(lengths, dtype=int)
    sample["__tok_len"] = lengths

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    # Histogram
    ax = axes[0]
    ax.hist(lengths, bins=50, color="#3498db", alpha=0.82, edgecolor="white")
    ax.axvline(max_length, color="#e74c3c", linestyle="--", linewidth=2.0,
               label=f"max_length={max_length}")
    ax.axvline(lengths.mean(), color="#f39c12", linestyle="-.", linewidth=1.6,
               label=f"Ort.={lengths.mean():.1f}")
    ax.axvline(np.median(lengths), color="#2ecc71", linestyle=":", linewidth=1.6,
               label=f"Medyan={np.median(lengths):.0f}")
    ax.set_xlabel("Token Sayısı", fontsize=11)
    ax.set_ylabel("Frekans", fontsize=11)
    ax.set_title("Token Uzunluğu Dağılımı", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.28)

    # Sınıf başına medyan
    ax = axes[1]
    class_medians = []
    for i in range(NUM_LABELS):
        sub = sample[sample["label_id"] == i]["__tok_len"]
        class_medians.append(sub.median() if len(sub) > 0 else 0)
    bars = ax.bar(range(NUM_LABELS), class_medians,
                  color=EMOTION_COLORS, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, class_medians):
        if val:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(NUM_LABELS))
    ax.set_xticklabels([e[:7] for e in EMOTIONS], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Medyan Token Uzunluğu", fontsize=11)
    ax.set_title("Sınıf Başına Medyan Token Uzunluğu", fontweight="bold")
    ax.axhline(max_length, color="#e74c3c", linestyle="--", alpha=0.75,
               label=f"max_length={max_length}")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.28, axis="y")

    # CDF
    ax = axes[2]
    sorted_len = np.sort(lengths)
    cdf = np.arange(1, len(sorted_len) + 1) / len(sorted_len)
    ax.plot(sorted_len, cdf, color="#9b59b6", linewidth=2.2)
    ax.axvline(max_length, color="#e74c3c", linestyle="--", linewidth=2.0,
               label=f"max_length={max_length}")
    pct_trunc = float((lengths > max_length).mean() * 100)
    ax.axhline(1 - pct_trunc / 100, color="#f39c12", linestyle=":",
               linewidth=1.6, alpha=0.85,
               label=f"{100 - pct_trunc:.1f}% ≤ {max_length} token")
    ax.set_xlabel("Token Uzunluğu", fontsize=11)
    ax.set_ylabel("Kümülatif Oran", fontsize=11)
    ax.set_title(f"CDF — Truncate Oranı: %{pct_trunc:.1f}", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.28)

    plt.suptitle("Metin Token Uzunluğu Analizi", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Token length dist: {save_path}")

    return {
        "mean":         round(float(lengths.mean()), 2),
        "median":       round(float(np.median(lengths)), 2),
        "p95":          round(float(np.percentile(lengths, 95)), 2),
        "p99":          round(float(np.percentile(lengths, 99)), 2),
        "max":          int(lengths.max()),
        "truncated_pct": round(pct_trunc, 2),
    }


# ─── 9. Kapsamlı Training Summary JSON ─────────────────────────────────────

def save_training_summary_json(
    exp_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    epoch_logger: EnhancedEpochLogger,
    train_result,
    auc_summary: dict,
    ap_summary: dict,
    ece_summary: dict,
    conf_summary: dict,
    save_path: Path,
    extra_info: Optional[dict] = None,
) -> dict:
    report = classification_report(
        y_true, y_pred,
        target_names=EMOTIONS,
        output_dict=True,
        zero_division=0,
    )
    per_class_summary = {}
    for i, emo in enumerate(EMOTIONS):
        if emo in report:
            per_class_summary[emo] = {
                "precision": round(report[emo]["precision"], 4),
                "recall":    round(report[emo]["recall"],    4),
                "f1":        round(report[emo]["f1-score"],  4),
                "support":   int(report[emo]["support"]),
                "roc_auc":   auc_summary.get(emo, 0.0),
                "avg_prec":  ap_summary.get(emo, 0.0),
                "ece":       ece_summary.get(emo, 0.0),
            }

    summary = {
        "experiment":   exp_name,
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_metrics": {
            "accuracy":    round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_macro":    round(float(f1_score(y_true, y_pred, average="macro",    zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "roc_auc_macro":  auc_summary.get("macro_avg", 0.0),
            "roc_auc_micro":  auc_summary.get("micro_avg", 0.0),
            "mean_avg_prec":  ap_summary.get("mean_ap", 0.0),
            "mean_ece":       ece_summary.get("mean_ece", 0.0),
        },
        "per_class":            per_class_summary,
        "roc_auc":              auc_summary,
        "average_precision":    ap_summary,
        "calibration_ece":      ece_summary,
        "confidence_analysis":  conf_summary,
        "training_history":     epoch_logger.history,
        "step_history_length":  len(epoch_logger.step_history["step"]),
        "epoch_times_sec":      epoch_logger.epoch_times,
        "total_steps":          int(train_result.global_step),
        "final_train_loss":     round(float(train_result.training_loss), 4),
        "total_runtime_sec":    round(float(train_result.metrics.get("train_runtime", 0)), 2),
        "total_runtime_min":    round(float(train_result.metrics.get("train_runtime", 0)) / 60, 2),
        "samples_per_second":   round(float(train_result.metrics.get("train_samples_per_second", 0)), 2),
    }
    if extra_info:
        summary["extra"] = extra_info

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[VIZ] Training summary: {save_path}")
    return summary


# ─── Ana Giriş Noktası ───────────────────────────────────────────────────────

def run_full_analytics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    logits_or_probs: np.ndarray,
    epoch_logger: EnhancedEpochLogger,
    train_result,
    out_dir: Path,
    exp_name: str,
    df_train: Optional[pd.DataFrame] = None,
    df_val:   Optional[pd.DataFrame] = None,
    df_test:  Optional[pd.DataFrame] = None,
    tokenizer=None,
    extra_info: Optional[dict] = None,
):
    """
    Tüm post-training analitiği çalıştırır ve out_dir altına kaydeder.

    Parametreler:
      logits_or_probs — ham logits veya softmax probs (otomatik algılanır)
      df_train/val/test — veri dağılımı ve token analizi için (opsiyonel)
      tokenizer — token uzunluğu dağılımı için (opsiyonel)
      extra_info — JSON özetine eklenecek ek sözlük (opsiyonel)
    """
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*65}")
    print(f"  [ANALİTİK] Kapsamlı analiz başlatıldı")
    print(f"  Deney : {exp_name}")
    print(f"  Dizin : {out_dir}")
    print(f"{'='*65}")

    # Logits → softmax probs
    arr = np.array(logits_or_probs, dtype=float)
    row_min, row_max = arr.min(axis=1), arr.max(axis=1)
    row_sum = arr.sum(axis=1)
    is_prob = (row_min >= 0).all() and (row_max <= 1.001).all() and \
              np.abs(row_sum - 1.0).mean() < 0.01
    y_probs = arr if is_prob else _scipy_softmax(arr, axis=1)

    # ── .npy kayıt ──
    np.save(out_dir / "test_probs.npy",  y_probs.astype(np.float32))
    np.save(out_dir / "test_preds.npy",  y_pred.astype(np.int32))
    np.save(out_dir / "test_labels.npy", y_true.astype(np.int32))
    print(f"[VIZ] Numpy arrays: test_probs/preds/labels.npy  (probs shape={y_probs.shape})")

    # ── Step + epoch history JSON ──
    with open(out_dir / "step_history.json", "w", encoding="utf-8") as f:
        json.dump(epoch_logger.step_history, f, indent=2)
    with open(out_dir / "epoch_history.json", "w", encoding="utf-8") as f:
        json.dump(epoch_logger.history, f, ensure_ascii=False, indent=2)
    print(f"[VIZ] Histories: step_history.json  epoch_history.json")

    # ── Grafikler ──
    save_step_loss_curve(
        epoch_logger.step_history,
        out_dir / "step_loss_curve.png",
        exp_name,
    )
    save_per_class_f1_evolution(
        epoch_logger.history,
        out_dir / "per_class_f1_evolution.png",
        exp_name,
    )
    auc_summary  = save_roc_curves(y_true, y_probs, out_dir / "roc_curves.png", exp_name)
    ap_summary   = save_pr_curves(y_true, y_probs, out_dir / "pr_curves.png", exp_name)
    ece_summary  = save_calibration_curves(y_true, y_probs, out_dir / "calibration_curves.png", exp_name)
    conf_summary = save_confidence_analysis(y_true, y_pred, y_probs, out_dir / "confidence_analysis.png", exp_name)

    if df_train is not None and df_val is not None and df_test is not None:
        save_data_distribution(df_train, df_val, df_test, out_dir / "data_distribution.png")

    if tokenizer is not None and df_train is not None:
        tok_stats = save_token_length_distribution(
            df_train, tokenizer, out_dir / "token_length_dist.png"
        )
        extra_info = extra_info or {}
        extra_info["token_length_stats"] = tok_stats

    # ── Kapsamlı özet ──
    save_training_summary_json(
        exp_name, y_true, y_pred, y_probs,
        epoch_logger, train_result,
        auc_summary, ap_summary, ece_summary, conf_summary,
        out_dir / "training_summary.json",
        extra_info=extra_info,
    )

    elapsed = time.time() - t0
    print(f"\n[ANALİTİK] Tamamlandı! ({elapsed:.1f}s)")
    print(f"  Kaydedilen dosyalar ({out_dir.name}/):")
    all_files = (
        sorted(out_dir.glob("*.npy")) +
        sorted(out_dir.glob("*.json")) +
        sorted(out_dir.glob("step_loss_curve.png")) +
        sorted(out_dir.glob("per_class_f1_evolution.png")) +
        sorted(out_dir.glob("roc_curves.png")) +
        sorted(out_dir.glob("pr_curves.png")) +
        sorted(out_dir.glob("calibration_curves.png")) +
        sorted(out_dir.glob("confidence_analysis.png")) +
        sorted(out_dir.glob("data_distribution.png")) +
        sorted(out_dir.glob("token_length_dist.png"))
    )
    for fp in all_files:
        if fp.exists():
            kb = fp.stat().st_size / 1024
            print(f"    {fp.name:<45} {kb:>7.1f} KB")
    print(f"{'='*65}\n")
