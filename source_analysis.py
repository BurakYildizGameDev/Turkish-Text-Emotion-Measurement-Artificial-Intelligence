"""
Kaynak Bazlı Hata Analizi — EXP4 Test Seti
===========================================
Mevcut test_preds.npy + test_labels.npy + test.parquet['source'] kullanılır.
Yeniden eğitim yapılmaz.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix
)

# ─── Sabitler ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent

EMOTIONS = [
    "mutluluk", "üzüntü", "öfke", "korku",
    "şaşkınlık", "tiksinti", "sevgi", "nötr", "utanç", "gurur",
]
ID2LABEL = {i: e for i, e in enumerate(EMOTIONS)}

# Kaynak hakkında makale metadatası
SOURCE_META = {
    "winvoker":      {"tur": "Ürün Yorumu",       "etiketleme": "Otomatik (sözlük)", "gurultu": "Orta"},
    "goemotions_tr": {"tur": "Sosyal Medya (TR)",  "etiketleme": "İnsan (EN→TR çeviri)", "gurultu": "Yüksek"},
    "ttc4900":       {"tur": "Haber",              "etiketleme": "Kategori→Nötr",   "gurultu": "Düşük"},
    "mteb_movie":    {"tur": "Film Yorumu",         "etiketleme": "Otomatik (sözlük)", "gurultu": "Orta"},
    "mteb_product":  {"tur": "Ürün Yorumu",        "etiketleme": "Otomatik (sözlük)", "gurultu": "Orta"},
    "fthbrmnby":     {"tur": "Ürün Yorumu",        "etiketleme": "Otomatik (sözlük)", "gurultu": "Orta"},
}

OUT_DIR = ROOT / "results" / "source_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXP_DIR = ROOT / "results" / "exp4"


# ─── ADIM 1-2: Veri yükle ────────────────────────────────────────────────────

def load_data():
    print("[1/6] Test verisi ve tahminler yükleniyor...")

    df_test = pd.read_parquet(ROOT / "data" / "splits" / "master_splits" / "test.parquet")
    print(f"  Test satır sayısı : {len(df_test)}")
    print(f"  Sütunlar          : {df_test.columns.tolist()}")

    preds  = np.load(EXP_DIR / "test_preds.npy")
    labels = np.load(EXP_DIR / "test_labels.npy")
    print(f"  test_preds.npy    : {preds.shape}")
    print(f"  test_labels.npy   : {labels.shape}")

    assert len(df_test) == len(preds) == len(labels), (
        f"Boyut uyumsuzluğu: df={len(df_test)}, preds={len(preds)}, labels={len(labels)}"
    )

    # master_dataset_with_source.parquet — combined_clean zaten source sütunu içeriyor
    df_combined = pd.read_parquet(ROOT / "data" / "processed" / "combined_clean.parquet")
    df_combined.to_parquet(ROOT / "data" / "processed" / "master_dataset_with_source.parquet", index=False)
    print(f"  master_dataset_with_source.parquet kaydedildi ({len(df_combined)} satır)")

    # test_source_labels.npy
    sources = df_test["source"].values
    np.save(OUT_DIR / "test_source_labels.npy", sources)
    print(f"  test_source_labels.npy kaydedildi")

    return df_test, preds, labels, sources


# ─── ADIM 3: Kaynak bazlı metrik hesapla ─────────────────────────────────────

def compute_source_metrics(df_test, preds, labels, sources):
    print("\n[2/6] Kaynak bazlı metrikler hesaplanıyor...")

    unique_sources = sorted(set(sources))
    rows = []

    for src in unique_sources:
        mask = sources == src
        y_true = labels[mask]
        y_pred = preds[mask]

        n_total   = int(mask.sum())
        n_correct = int((y_true == y_pred).sum())
        n_wrong   = n_total - n_correct
        acc       = n_correct / n_total if n_total > 0 else 0.0
        err_rate  = n_wrong  / n_total if n_total > 0 else 0.0

        f1_macro    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        per_class_f1 = f1_score(
            y_true, y_pred,
            labels=list(range(10)), average=None, zero_division=0
        )

        row = {
            "Kaynak":          src,
            "Tür":             SOURCE_META.get(src, {}).get("tur", "Bilinmiyor"),
            "Örnek Sayısı":    n_total,
            "Doğru":           n_correct,
            "Doğru Oranı":     round(acc, 4),
            "Yanlış":          n_wrong,
            "Hata Oranı":      round(err_rate, 4),
            "F1-Macro":        round(f1_macro, 4),
            "F1-Weighted":     round(f1_weighted, 4),
        }
        for i, emo in enumerate(EMOTIONS):
            row[f"F1_{emo}"] = round(float(per_class_f1[i]), 4)

        rows.append(row)
        print(f"  {src:20s}: n={n_total:6d}  hata={err_rate:.3f}  F1-macro={f1_macro:.4f}")

    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(OUT_DIR / "source_metrics_table.csv", index=False, encoding="utf-8-sig")
    print(f"\n  source_metrics_table.csv kaydedildi")
    return df_metrics, unique_sources


# ─── ADIM 4: Hata analizi çaprazlama tablosu ─────────────────────────────────

def compute_error_crosslap(df_test, preds, labels, sources, unique_sources):
    print("\n[3/6] Hata çaprazlama tablosu oluşturuluyor...")

    error_rows = []

    for src in unique_sources:
        mask = sources == src
        y_true = labels[mask]
        y_pred = preds[mask]
        n_total = int(mask.sum())

        wrong_mask = y_true != y_pred
        n_wrong = int(wrong_mask.sum())

        for cls_id, cls_name in enumerate(EMOTIONS):
            cls_mask = y_true == cls_id
            n_cls = int(cls_mask.sum())
            if n_cls == 0:
                continue
            n_cls_wrong = int((cls_mask & wrong_mask).sum())
            cls_err_rate = n_cls_wrong / n_cls

            # En sık karışılan sınıf
            wrong_cls = y_pred[cls_mask & wrong_mask]
            if len(wrong_cls) > 0:
                confused_id   = int(np.bincount(wrong_cls, minlength=10).argmax())
                confused_name = EMOTIONS[confused_id]
                confused_count = int(np.bincount(wrong_cls, minlength=10).max())
            else:
                confused_name  = "-"
                confused_count = 0

            error_rows.append({
                "Kaynak":           src,
                "Sınıf":            cls_name,
                "Sınıf_n":          n_cls,
                "Yanlış_n":         n_cls_wrong,
                "Hata_Oranı":       round(cls_err_rate, 4),
                "En_Çok_Karışılan": confused_name,
                "Karışma_Sayısı":   confused_count,
            })

    df_err = pd.DataFrame(error_rows)
    df_err.to_csv(OUT_DIR / "source_error_analysis.csv", index=False, encoding="utf-8-sig")
    print(f"  source_error_analysis.csv kaydedildi")
    return df_err


# ─── ADIM 5a: F1 Bar Chart ───────────────────────────────────────────────────

def plot_f1_bar(df_metrics):
    print("\n[4/6] Kaynak bazlı F1 bar chart çiziliyor...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Kaynak Bazlı Model Performansı (EXP4)", fontsize=14, fontweight="bold")

    colors = plt.cm.Set2(np.linspace(0, 1, len(df_metrics)))

    # F1-Macro
    ax = axes[0]
    bars = ax.barh(df_metrics["Kaynak"], df_metrics["F1-Macro"], color=colors)
    ax.set_xlabel("F1-Macro")
    ax.set_title("F1-Macro (Kaynak Bazlı)")
    ax.set_xlim(0, 1.0)
    for bar, val in zip(bars, df_metrics["F1-Macro"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    # Hata Oranı
    ax2 = axes[1]
    bars2 = ax2.barh(df_metrics["Kaynak"], df_metrics["Hata Oranı"], color=colors)
    ax2.set_xlabel("Hata Oranı")
    ax2.set_title("Hata Oranı (Kaynak Bazlı)")
    ax2.set_xlim(0, 1.0)
    for bar, val in zip(bars2, df_metrics["Hata Oranı"]):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "source_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  source_comparison.png kaydedildi")


# ─── ADIM 5b: Hata Isı Haritası ──────────────────────────────────────────────

def plot_error_heatmap(df_err, unique_sources):
    print("\n[5/6] Kaynak x sınıf hata ısı haritası çiziliyor...")

    pivot = df_err.pivot_table(
        index="Kaynak", columns="Sınıf", values="Hata_Oranı", fill_value=0
    )
    # Sınıfları sabit sırada tut
    pivot = pivot.reindex(columns=[e for e in EMOTIONS if e in pivot.columns])

    fig, ax = plt.subplots(figsize=(16, max(4, len(unique_sources) * 1.0)))
    sns.heatmap(
        pivot,
        annot=True, fmt=".2f",
        cmap="RdYlGn_r",
        vmin=0, vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Hata Oranı"},
    )
    ax.set_title("Kaynak × Sınıf Hata Oranı Isı Haritası (EXP4)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Duygu Sınıfı")
    ax.set_ylabel("Kaynak")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "source_error_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  source_error_heatmap.png kaydedildi")


# ─── ADIM 6: Özet bulgular ────────────────────────────────────────────────────

def write_summary(df_metrics, df_err, preds, labels, sources):
    print("\n[6/6] Özet bulgular yazılıyor...")

    lines = []
    lines.append("=" * 70)
    lines.append("KAYNAK BAZLI HATA ANALİZİ — EXP4 ÖZET BULGULAR")
    lines.append("=" * 70)
    lines.append(f"Analiz tarihi : 2026-05-01")
    lines.append(f"Test seti boyutu: {len(preds):,} örnek")
    lines.append("")

    # ── 1. Genel kaynak tablosu
    lines.append("─" * 70)
    lines.append("1. KAYNAK BAZLI GENEL METRİKLER")
    lines.append("─" * 70)
    for _, row in df_metrics.iterrows():
        lines.append(
            f"  {row['Kaynak']:20s}: n={row['Örnek Sayısı']:6,}  "
            f"Hata={row['Hata Oranı']:.3f}  "
            f"F1-Macro={row['F1-Macro']:.4f}  "
            f"F1-Weighted={row['F1-Weighted']:.4f}"
        )

    # ── 2. GoEmotions vs winvoker karşılaştırması
    lines.append("")
    lines.append("─" * 70)
    lines.append("2. GOEMOTİONS_TR vs WİNVOKER KARŞILAŞTIRMASI")
    lines.append("─" * 70)

    go_row  = df_metrics[df_metrics["Kaynak"] == "goemotions_tr"]
    win_row = df_metrics[df_metrics["Kaynak"] == "winvoker"]

    if len(go_row) > 0 and len(win_row) > 0:
        go_err  = float(go_row["Hata Oranı"].values[0])
        win_err = float(win_row["Hata Oranı"].values[0])
        ratio = go_err / win_err if win_err > 0 else float("inf")
        lines.append(f"  GoEmotions_TR hata oranı : {go_err:.3f}")
        lines.append(f"  winvoker      hata oranı : {win_err:.3f}")
        lines.append(f"  GoEmotions_TR hata oranı winvoker'a göre {ratio:.2f} kat yüksek.")

        go_f1  = float(go_row["F1-Macro"].values[0])
        win_f1 = float(win_row["F1-Macro"].values[0])
        lines.append(f"  GoEmotions_TR F1-Macro   : {go_f1:.4f}")
        lines.append(f"  winvoker      F1-Macro   : {win_f1:.4f}")
        lines.append(f"  F1-Macro farkı (win - go): {win_f1 - go_f1:+.4f}")

    # ── 3. En çok karışan kaynak x sınıf çiftleri
    lines.append("")
    lines.append("─" * 70)
    lines.append("3. EN ÇOK KARIŞTIRMA OLAN KAYNAK × SINIF ÇİFTLERİ (top 10)")
    lines.append("─" * 70)
    top_err = df_err[df_err["Sınıf_n"] >= 5].nlargest(10, "Hata_Oranı")
    for _, row in top_err.iterrows():
        lines.append(
            f"  {row['Kaynak']:20s} | {row['Sınıf']:12s} | "
            f"n={row['Sınıf_n']:5d} | hata={row['Hata_Oranı']:.3f} | "
            f"→ {row['En_Çok_Karışılan']} ({row['Karışma_Sayısı']}x)"
        )

    # ── 4. winvoker baskın sınıflar (mutluluk, nötr)
    lines.append("")
    lines.append("─" * 70)
    lines.append("4. WİNVOKER BASKÍN SINIFLAR (mutluluk, nötr) — KAYNAK KARŞILAŞTIRMASI")
    lines.append("─" * 70)
    for cls in ["mutluluk", "nötr"]:
        cls_df = df_err[df_err["Sınıf"] == cls].copy()
        if len(cls_df) == 0:
            continue
        cls_df = cls_df.sort_values("Hata_Oranı")
        lines.append(f"\n  Sınıf: {cls}")
        for _, row in cls_df.iterrows():
            lines.append(
                f"    {row['Kaynak']:20s}: n={row['Sınıf_n']:6,}  hata={row['Hata_Oranı']:.3f}"
            )

    # ── 5. Makale tablosu (Section 3.1)
    lines.append("")
    lines.append("─" * 70)
    lines.append("5. MAKALE İÇİN GENIŞLETILMIŞ KAYNAK TABLOSU (Bölüm 3.1)")
    lines.append("─" * 70)
    header = (
        f"{'Kaynak':20s} | {'Tür':20s} | {'Örnek Sayısı':14s} | "
        f"{'Etiketleme':26s} | {'Tahmini Gürültü':15s} | "
        f"{'Test Hata Oranı':16s} | {'F1-Macro':10s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for _, row in df_metrics.iterrows():
        src = row["Kaynak"]
        meta = SOURCE_META.get(src, {})
        lines.append(
            f"{src:20s} | {meta.get('tur','?'):20s} | "
            f"{row['Örnek Sayısı']:14,} | "
            f"{meta.get('etiketleme','?'):26s} | "
            f"{meta.get('gurultu','?'):15s} | "
            f"{row['Hata Oranı']:16.3f} | "
            f"{row['F1-Macro']:10.4f}"
        )

    # ── 6. Sınıf bazlı F1 per kaynak (özet)
    lines.append("")
    lines.append("─" * 70)
    lines.append("6. SINIF BAZLI F1 (KAYNAK × SINIF)")
    lines.append("─" * 70)
    f1_cols = [f"F1_{e}" for e in EMOTIONS]
    header2 = f"{'Kaynak':20s} | " + " | ".join(f"{e[:6]:8s}" for e in EMOTIONS)
    lines.append(header2)
    lines.append("-" * len(header2))
    for _, row in df_metrics.iterrows():
        vals = " | ".join(f"{row[c]:8.4f}" for c in f1_cols)
        lines.append(f"{row['Kaynak']:20s} | {vals}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("Tüm çıktılar: results/source_analysis/")
    lines.append("=" * 70)

    summary_text = "\n".join(lines)
    out_path = OUT_DIR / "source_analysis_summary.txt"
    out_path.write_text(summary_text, encoding="utf-8")
    print(f"  source_analysis_summary.txt kaydedildi")
    return summary_text


# ─── ANA AKIŞ ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  KAYNAK BAZLI HATA ANALİZİ — EXP4")
    print("=" * 70)

    df_test, preds, labels, sources = load_data()

    print("\nKaynak dağılımı (test seti):")
    for src, cnt in pd.Series(sources).value_counts().items():
        print(f"  {src:20s}: {cnt:6,}")

    df_metrics, unique_sources = compute_source_metrics(df_test, preds, labels, sources)
    df_err = compute_error_crosslap(df_test, preds, labels, sources, unique_sources)
    plot_f1_bar(df_metrics)
    plot_error_heatmap(df_err, unique_sources)
    summary = write_summary(df_metrics, df_err, preds, labels, sources)

    print("\n" + "=" * 70)
    print("  ANALİZ TAMAMLANDI")
    print("=" * 70)
    print(f"\nÇıktı klasörü: {OUT_DIR}")
    for f in sorted(OUT_DIR.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name:45s} {size/1024:8.1f} KB")

    print("\n" + "─" * 70)
    print("ÖZET METİN (source_analysis_summary.txt):")
    print("─" * 70)
    print(summary)


if __name__ == "__main__":
    main()
