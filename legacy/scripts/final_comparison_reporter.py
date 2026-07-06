"""
Final Comparison Reporter
=========================
4 deneyin sonuçlarını okur ve akademik sunum için kapsamlı çıktılar üretir.

Çıktılar (results/final/):
  comparison_table.txt      ← ASCII konsol tablosu (kopyala-yapıştır)
  comparison_table.csv      ← Excel / pandas için
  comparison_table.tex      ← LaTeX \booktabs tablosu
  grouped_bar_chart.png     ← 4 model × 10 duygu F1 bar grafiği
  tsne_exp4.png             ← En iyi model CLS embedding t-SNE
  final_summary.json        ← Makine okunabilir tüm metrikler

Kullanım:
  python final_comparison_reporter.py
  python final_comparison_reporter.py --skip-tsne   # t-SNE'yi atla (yavaş)
  python final_comparison_reporter.py --tsne-samples 2000
"""

import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent
FINAL   = ROOT / "results" / "final"
RESULTS = ROOT / "results"

EMOTIONS = [
    "mutluluk", "üzüntü", "öfke", "korku",
    "şaşkınlık", "tiksinti", "sevgi", "nötr", "utanç", "gurur",
]
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}
NUM_LABELS = 10

EXP_META = {
    1: {"name": "Exp1 — Baseline",       "short": "Baseline",     "color": "#95a5a6"},
    2: {"name": "Exp2 — Cross-Lingual",  "short": "Cross-Ling.",  "color": "#3498db"},
    3: {"name": "Exp3 — Lexicon",        "short": "Lexicon",      "color": "#8e44ad"},
    4: {"name": "Exp4 — Master Hybrid",  "short": "Master",       "color": "#e74c3c"},
}


# ─── JSON Yükleyici ──────────────────────────────────────────────────────────

def load_results() -> Dict[int, dict]:
    loaded = {}
    for eid in [1, 2, 3, 4]:
        path = RESULTS / f"exp{eid}" / "metrics_report.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                loaded[eid] = json.load(f)
            print(f"[YUKLE] Exp{eid} — F1-Macro={loaded[eid].get('test_f1_macro','N/A')}")
        else:
            print(f"[EKSIK] Exp{eid} sonucu bulunamadi: {path}")
    return loaded


# ─── Karşılaştırma Tablosu ───────────────────────────────────────────────────

def build_comparison_df(results: Dict[int, dict]) -> pd.DataFrame:
    rows = []
    for eid, data in results.items():
        row = {
            "Deney":       EXP_META[eid]["name"],
            "Mimari":      data.get("architecture", data.get("model", "—"))[:45],
            "Accuracy":    data.get("test_accuracy",    None),
            "F1-Macro":    data.get("test_f1_macro",    None),
            "F1-Weighted": data.get("test_f1_weighted", None),
        }
        for emo in EMOTIONS:
            f1 = data.get("per_class", {}).get(emo, {}).get("f1", None)
            row[emo[:8]] = f1
        rows.append(row)
    return pd.DataFrame(rows)


def print_comparison_table(df: pd.DataFrame) -> str:
    """ASCII tablosunu oluştur ve döndür."""
    lines = []
    sep = "═" * 105
    lines.append(sep)
    lines.append("  TÜRKÇE DUYGU ANALİZİ ABLATION STUDY — KARŞILAŞTIRMA TABLOSU")
    lines.append(f"  Oluşturulma: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(sep)

    # Başlık satırı
    header = f"  {'Deney':<28} {'Acc':>6} {'F1-Mac':>7} {'F1-Wt':>7}"
    for emo in EMOTIONS:
        header += f" {emo[:6]:>7}"
    lines.append(header)
    lines.append("─" * 105)

    for _, row in df.iterrows():
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else "  N/A "
        line = f"  {row['Deney']:<28} {fmt(row['Accuracy']):>7} {fmt(row['F1-Macro']):>7} {fmt(row['F1-Weighted']):>7}"
        for emo in EMOTIONS:
            line += f" {fmt(row.get(emo[:8])): >7}"
        lines.append(line)

    lines.append("─" * 105)

    # En iyi değerleri vurgula
    numeric_cols = ["Accuracy", "F1-Macro", "F1-Weighted"] + [e[:8] for e in EMOTIONS]
    lines.append("  En Yuksek Deger:")
    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        best_val = vals.max()
        best_exp = df.loc[vals.idxmax(), "Deney"]
        lines.append(f"    {col:12s}: {best_val:.4f}  ({best_exp})")

    lines.append(sep)
    table_str = "\n".join(lines)
    print(table_str)
    return table_str


def save_latex_table(df: pd.DataFrame, save_path: Path):
    """LaTeX booktabs formatında tablo üret."""
    main_cols = ["Deney", "Accuracy", "F1-Macro", "F1-Weighted"]
    minority  = ["utanç", "gurur", "tiksinti", "korku"]
    minority_cols = [e[:8] for e in minority]

    df_lat = df[main_cols + minority_cols].copy()
    df_lat = df_lat.rename(columns={
        "Deney": "Model",
        "utanç":   "Utanç",
        "gurur":   "Gurur",
        "tiksin":  "Tiksin.",
        "korku":   "Korku",
    })

    def fmt_cell(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v) if v is not None else "—"

    lines = [
        "% Otomatik üretildi: final_comparison_reporter.py",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Türkçe Duygu Analizi Ablation Study Sonuçları}",
        r"  \label{tab:ablation}",
        r"  \begin{tabular}{lcccccccc}",
        r"  \toprule",
    ]
    header_row = "  " + " & ".join(df_lat.columns) + r" \\"
    lines.append(header_row)
    lines.append(r"  \midrule")
    for _, row in df_lat.iterrows():
        data_row = "  " + " & ".join(fmt_cell(row[c]) for c in df_lat.columns) + r" \\"
        lines.append(data_row)
    lines.append(r"  \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[KAYIT] LaTeX tablosu: {save_path}")


# ─── Grouped Bar Chart ───────────────────────────────────────────────────────

def save_grouped_bar_chart(results: Dict[int, dict], save_path: Path):
    """4 model × 10 duygu F1 grouped bar grafiği."""
    available = {eid: d for eid, d in results.items()}
    if not available:
        print("[WARN] Grafik icin veri yok")
        return

    x     = np.arange(NUM_LABELS)
    n_exp = len(available)
    total_width = 0.75
    bar_w = total_width / n_exp
    offsets = np.linspace(-total_width/2 + bar_w/2, total_width/2 - bar_w/2, n_exp)

    fig, (ax_main, ax_summary) = plt.subplots(
        2, 1, figsize=(18, 12),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # ── Üst: Per-class F1 ────────────────────────────────────────
    for offset, (eid, data) in zip(offsets, sorted(available.items())):
        meta  = EXP_META[eid]
        f1s   = [data.get("per_class", {}).get(emo, {}).get("f1", 0.0) for emo in EMOTIONS]
        bars  = ax_main.bar(
            x + offset, f1s, bar_w,
            label=meta["short"], color=meta["color"],
            alpha=0.85, edgecolor="white", linewidth=0.5,
        )
        # Değer etiketi (sadece > 0.05 olanlar)
        for bar, val in zip(bars, f1s):
            if val > 0.05:
                ax_main.text(
                    bar.get_x() + bar.get_width()/2, val + 0.008,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=6.5, rotation=90, color=meta["color"],
                )

    # Azınlık sınıfları vurgula
    minority_ids = [LABEL2ID[e] for e in ["gurur", "utanç", "tiksinti", "korku"]]
    for mid in minority_ids:
        ax_main.axvspan(mid - 0.45, mid + 0.45, alpha=0.06, color="#e74c3c", zorder=0)

    ax_main.set_xticks(x)
    ax_main.set_xticklabels([e[:10] for e in EMOTIONS], rotation=25, ha="right", fontsize=10)
    ax_main.set_ylim(0, 1.18)
    ax_main.set_ylabel("F1 Skoru", fontsize=12)
    ax_main.set_title(
        "Ablation Study — 4 Model × 10 Duygu Sınıfı F1 Karşılaştırması\n"
        "(Kırmızı arka plan: azınlık sınıflar)",
        fontsize=14, fontweight="bold",
    )
    ax_main.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax_main.grid(axis="y", alpha=0.3, linestyle="--")
    ax_main.axhline(y=0, color="black", linewidth=0.5)

    # Azınlık etiketi
    ax_main.text(
        0.99, 0.99, "Azınlık sınıflar ←",
        transform=ax_main.transAxes, ha="right", va="top",
        fontsize=9, color="#e74c3c", style="italic",
    )

    # ── Alt: Macro/Accuracy özet bar ─────────────────────────────
    summary_metrics = ["F1-Macro", "F1-Weighted", "Accuracy"]
    s_x   = np.arange(len(summary_metrics))
    s_bar = 0.65 / n_exp
    s_off = np.linspace(-0.65/2 + s_bar/2, 0.65/2 - s_bar/2, n_exp)

    for soff, (eid, data) in zip(s_off, sorted(available.items())):
        meta = EXP_META[eid]
        vals = [
            data.get("test_f1_macro",    0.0),
            data.get("test_f1_weighted", 0.0),
            data.get("test_accuracy",    0.0),
        ]
        bars = ax_summary.bar(
            s_x + soff, vals, s_bar,
            color=meta["color"], alpha=0.9,
            edgecolor="white", label=meta["short"],
        )
        for bar, val in zip(bars, vals):
            ax_summary.text(
                bar.get_x() + bar.get_width()/2, val + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax_summary.set_xticks(s_x)
    ax_summary.set_xticklabels(summary_metrics, fontsize=11)
    ax_summary.set_ylim(0, 1.12)
    ax_summary.set_ylabel("Skor", fontsize=11)
    ax_summary.set_title("Özet Metrikler", fontsize=12, fontweight="bold")
    ax_summary.grid(axis="y", alpha=0.3, linestyle="--")
    ax_summary.legend(loc="lower right", fontsize=9)

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Grouped bar chart: {save_path}")


# ─── t-SNE Görselleştirme ────────────────────────────────────────────────────

def save_tsne(
    emb_path:    Path,
    labels_path: Path,
    preds_path:  Path,
    save_path:   Path,
    n_samples:   int  = 3000,
):
    """
    Exp4 CLS embedding'lerinden t-SNE oluşturur.
    emb_path   : (N, 768) ndarray
    labels_path: (N,)     int ndarray
    """
    if not emb_path.exists():
        print(f"[WARN] t-SNE icin embedding bulunamadi: {emb_path}")
        print("       Exp4'u calistirarak embeddingleri uretebilirsiniz.")
        return

    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    print(f"[tSNE] Embedding yukluyor: {emb_path}")
    emb    = np.load(emb_path)
    labels = np.load(labels_path)
    preds  = np.load(preds_path) if preds_path.exists() else labels

    # Örneklem sınırla
    if len(emb) > n_samples:
        rng = np.random.default_rng(42)
        # Stratified örnekle
        idx_list = []
        per_cls  = max(1, n_samples // NUM_LABELS)
        for cls_id in range(NUM_LABELS):
            cls_idx = np.where(labels == cls_id)[0]
            if len(cls_idx) == 0:
                continue
            chosen = rng.choice(cls_idx, min(per_cls, len(cls_idx)), replace=False)
            idx_list.extend(chosen.tolist())
        idx_list = idx_list[:n_samples]
        emb    = emb[idx_list]
        labels = labels[idx_list]
        preds  = preds[idx_list]

    print(f"[tSNE] PCA → 50 dim uygulaniyor ({len(emb)} örnek)...")
    pca    = PCA(n_components=50, random_state=42)
    emb_pca = pca.fit_transform(emb)

    print("[tSNE] t-SNE hesaplaniyor (bu birkaç dakika surebilir)...")
    tsne   = TSNE(
        n_components=2, perplexity=40, learning_rate="auto",
        max_iter=1000, random_state=42, n_jobs=-1,
    )
    emb_2d = tsne.fit_transform(emb_pca)

    # Renk paleti
    palette = [
        "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
        "#1abc9c","#e91e63","#607d8b","#ff5722","#795548",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    for ax, color_by, title_suffix in [
        (axes[0], labels, "Gerçek Etiket"),
        (axes[1], preds,  "Tahmin Edilen"),
    ]:
        for cls_id in range(NUM_LABELS):
            mask = (color_by == cls_id)
            if not mask.any():
                continue
            ax.scatter(
                emb_2d[mask, 0], emb_2d[mask, 1],
                c=palette[cls_id % len(palette)],
                label=EMOTIONS[cls_id][:10],
                alpha=0.6, s=12, edgecolors="none",
            )
        ax.set_title(f"t-SNE — {title_suffix}", fontsize=13, fontweight="bold")
        ax.set_xlabel("t-SNE 1", fontsize=10); ax.set_ylabel("t-SNE 2", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, markerscale=2,
                  ncol=2, framealpha=0.8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        "Experiment 4 (Master Hybrid) — CLS Embedding t-SNE Görselleştirmesi\n"
        f"({len(emb)} örnek, PCA(50) → t-SNE(2D))",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] t-SNE: {save_path}")


# ─── Improvement Heatmap ─────────────────────────────────────────────────────

def save_improvement_heatmap(results: Dict[int, dict], save_path: Path):
    """Her deneyin Exp1'e göre F1 iyileşmesini gösteren ısı haritası."""
    if 1 not in results:
        print("[WARN] Improvement heatmap icin Exp1 sonucu gerekli")
        return

    base = results[1].get("per_class", {})
    data = []
    row_labels = []

    for eid in [2, 3, 4]:
        if eid not in results:
            continue
        pc = results[eid].get("per_class", {})
        deltas = [
            pc.get(emo, {}).get("f1", 0.0) - base.get(emo, {}).get("f1", 0.0)
            for emo in EMOTIONS
        ]
        data.append(deltas)
        row_labels.append(EXP_META[eid]["name"])

    if not data:
        return

    mat    = np.array(data)
    vmax   = max(abs(mat).max(), 0.01)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(NUM_LABELS))
    ax.set_xticklabels([e[:9] for e in EMOTIONS], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)

    # Değer etiketleri
    for i in range(len(row_labels)):
        for j in range(NUM_LABELS):
            val  = mat[i, j]
            sign = "+" if val >= 0 else ""
            clr  = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{sign}{val:.3f}", ha="center", va="center",
                    fontsize=8, color=clr, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8, label="F1 Delta (Exp1'e göre)")
    ax.set_title("Exp1 Baseline'a Göre F1 İyileşmesi (Yeşil=iyileşme, Kırmızı=kötüleşme)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Improvement heatmap: {save_path}")


# ─── Final JSON Özeti ────────────────────────────────────────────────────────

def save_final_summary(results: Dict[int, dict], df: pd.DataFrame, save_path: Path):
    summary = {
        "generated_at": datetime.now().isoformat(),
        "experiments":  {},
        "best_overall": {},
        "per_emotion_winners": {},
    }

    for eid, data in results.items():
        summary["experiments"][f"exp{eid}"] = {
            "name":        EXP_META[eid]["name"],
            "accuracy":    data.get("test_accuracy"),
            "f1_macro":    data.get("test_f1_macro"),
            "f1_weighted": data.get("test_f1_weighted"),
            "per_class":   {
                emo: data.get("per_class", {}).get(emo, {}).get("f1")
                for emo in EMOTIONS
            },
        }

    # Genel en iyi
    best_eid  = max(results.keys(), key=lambda e: results[e].get("test_f1_macro", 0.0))
    summary["best_overall"] = {
        "exp_id":   best_eid,
        "name":     EXP_META[best_eid]["name"],
        "f1_macro": results[best_eid].get("test_f1_macro"),
    }

    # Sınıf bazlı en iyi
    for emo in EMOTIONS:
        best_e = max(
            results.keys(),
            key=lambda e: results[e].get("per_class", {}).get(emo, {}).get("f1", 0.0),
        )
        best_f1 = results[best_e].get("per_class", {}).get(emo, {}).get("f1", 0.0)
        summary["per_emotion_winners"][emo] = {
            "exp_id": best_e,
            "name":   EXP_META[best_e]["name"],
            "f1":     best_f1,
        }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[KAYIT] Final ozet: {save_path}")

    # Konsola sınıf kazananları
    print("\n  PER-EMOTION KAZANANLARI:")
    for emo in EMOTIONS:
        w = summary["per_emotion_winners"][emo]
        print(f"    {emo:12s}: {w['name']:25s}  F1={w['f1']:.4f}")

    print(f"\n  GENEL EN IYI MODEL: {summary['best_overall']['name']}")
    print(f"  F1-Macro           : {summary['best_overall']['f1_macro']:.4f}")

    return summary


# ─── Ana Fonksiyon ───────────────────────────────────────────────────────────

def main(skip_tsne: bool = False, tsne_samples: int = 3000):
    FINAL.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*65)
    print("  FINAL COMPARISON REPORTER")
    print(f"  Cikti: {FINAL}")
    print("="*65 + "\n")

    # ── 1. Sonuçları Yükle ────────────────────────────────────────
    results = load_results()
    if not results:
        print("[HATA] Hicbir deney sonucu bulunamadi.")
        print("       Once: python run_experiments.py")
        sys.exit(1)

    # ── 2. DataFrame & Tablolar ───────────────────────────────────
    df = build_comparison_df(results)

    table_str = print_comparison_table(df)
    with open(FINAL / "comparison_table.txt", "w", encoding="utf-8") as f:
        f.write(table_str)
    print(f"\n[KAYIT] ASCII tablo: {FINAL}/comparison_table.txt")

    df.to_csv(FINAL / "comparison_table.csv", index=False, encoding="utf-8-sig")
    print(f"[KAYIT] CSV tablo  : {FINAL}/comparison_table.csv")

    save_latex_table(df, FINAL / "comparison_table.tex")

    # ── 3. Grouped Bar Chart ──────────────────────────────────────
    save_grouped_bar_chart(results, FINAL / "grouped_bar_chart.png")

    # ── 4. Improvement Heatmap ────────────────────────────────────
    save_improvement_heatmap(results, FINAL / "improvement_heatmap.png")

    # ── 5. t-SNE (Exp4 embeddings) ────────────────────────────────
    if not skip_tsne:
        exp4_dir  = RESULTS / "exp4"
        save_tsne(
            emb_path    = exp4_dir / "test_cls_embeddings.npy",
            labels_path = exp4_dir / "test_labels.npy",
            preds_path  = exp4_dir / "test_preds.npy",
            save_path   = FINAL / "tsne_exp4.png",
            n_samples   = tsne_samples,
        )
    else:
        print("[ATLA] t-SNE atlandı (--skip-tsne)")

    # ── 6. Final JSON Özeti ───────────────────────────────────────
    save_final_summary(results, df, FINAL / "final_summary.json")

    print("\n" + "="*65)
    print("  RAPOR TAMAMLANDI")
    print(f"  Klasor: {FINAL}")
    print("="*65)
    print("\n  Uretilen dosyalar:")
    for f in sorted(FINAL.iterdir()):
        size = f.stat().st_size
        unit = "KB" if size >= 1024 else "B"
        sz   = size // 1024 if size >= 1024 else size
        print(f"    {f.name:35s} {sz:6d} {unit}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Final Comparison Reporter — 4 Deney Karsilastirmasi"
    )
    parser.add_argument("--skip-tsne",     action="store_true",
                        help="t-SNE hesaplamayı atla (yavaş, GPU/RAM gerektirir)")
    parser.add_argument("--tsne-samples",  type=int, default=3000,
                        help="t-SNE için örneklem sayısı (varsayılan: 3000)")
    args = parser.parse_args()

    main(skip_tsne=args.skip_tsne, tsne_samples=args.tsne_samples)
