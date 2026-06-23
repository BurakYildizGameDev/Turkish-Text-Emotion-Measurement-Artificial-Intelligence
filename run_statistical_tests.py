"""
İstatistiksel Anlamlılık Testleri
===================================
1. McNemar testi  (χ² ile p-değeri)
2. Bootstrap %95 güven aralığı (F1-Macro farkı)

Çıktılar (results/statistical_tests/):
  statistical_tests.json  — tüm sonuçlar
  bootstrap_ci.png        — bootstrap dağılım + CI görselleştirmesi
"""

import sys
import json
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2
from sklearn.metrics import f1_score, accuracy_score

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "statistical_tests"

EMOTIONS = [
    "mutluluk", "uzuntu", "ofke", "korku",
    "saskinlik", "tiksinti", "sevgi", "notr", "utanc", "gurur",
]

# ─── McNemar Testi ──────────────────────────────────────────────────────────

def mcnemar_test(preds_a, preds_b, labels, name_a="ModelA", name_b="ModelB"):
    a_correct = (preds_a == labels)
    b_correct = (preds_b == labels)

    both_correct = np.sum(a_correct & b_correct)
    both_wrong   = np.sum(~a_correct & ~b_correct)
    b_only       = np.sum(~a_correct & b_correct)
    a_only       = np.sum(a_correct & ~b_correct)

    if (b_only + a_only) == 0:
        stat = 0.0
        p_value = 1.0
    else:
        stat = (abs(b_only - a_only) - 1) ** 2 / (b_only + a_only)
        p_value = 1 - chi2.cdf(stat, df=1)

    result = {
        "comparison": f"{name_a} vs {name_b}",
        "contingency_table": {
            "both_correct": int(both_correct),
            "both_wrong": int(both_wrong),
            f"{name_b}_only_correct (b)": int(b_only),
            f"{name_a}_only_correct (c)": int(a_only),
        },
        "mcnemar_statistic": round(float(stat), 4),
        "p_value": float(p_value),
        "significant_at_0.05": p_value < 0.05,
        "significant_at_0.01": p_value < 0.01,
        "significant_at_0.001": p_value < 0.001,
    }

    if p_value < 0.001:
        result["interpretation"] = f"{name_b}, {name_a}'e gore p < 0.001 duzeyinde anlamli bicimde farklidir."
    elif p_value < 0.01:
        result["interpretation"] = f"{name_b}, {name_a}'e gore p < 0.01 duzeyinde anlamli bicimde farklidir."
    elif p_value < 0.05:
        result["interpretation"] = f"{name_b}, {name_a}'e gore p < 0.05 duzeyinde anlamli bicimde farklidir."
    else:
        result["interpretation"] = f"Iki model arasinda istatistiksel olarak anlamli fark bulunamamistir (p = {p_value:.4f})."

    return result


# ─── Bootstrap Güven Aralığı ────────────────────────────────────────────────

def bootstrap_ci(labels, preds_a, preds_b, n_bootstrap=1000, seed=42):
    np.random.seed(seed)
    all_classes = np.arange(10)
    n = len(labels)
    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        l, pa, pb = labels[idx], preds_a[idx], preds_b[idx]
        fa = f1_score(l, pa, average='macro', zero_division=0, labels=all_classes)
        fb = f1_score(l, pb, average='macro', zero_division=0, labels=all_classes)
        diffs.append(fb - fa)
    diffs = np.array(diffs)
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    mean_diff = np.mean(diffs)
    return {'mean': float(mean_diff), 'ci_low': float(ci_low), 'ci_high': float(ci_high),
            'significant': bool(ci_low > 0)}, diffs


# ─── Görselleştirme ──────────────────────────────────────────────────────────

def save_bootstrap_plot(comparisons, save_path):
    n_plots = len(comparisons)
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    for ax, (name, result, diffs) in zip(axes, comparisons):
        ci_low   = result["ci_low"]
        ci_high  = result["ci_high"]
        mean_d   = result["mean"]
        sig      = result["significant"]

        color = "#27ae60" if sig and mean_d > 0 else (
            "#e74c3c" if sig and mean_d < 0 else "#95a5a6"
        )

        ax.hist(diffs, bins=80, color=color, alpha=0.7, edgecolor="white", density=True)
        ax.axvline(0, color="black", linestyle="-", linewidth=2, alpha=0.8, label="Fark = 0")
        ax.axvline(mean_d,  color="#2c3e50", linestyle="--", linewidth=2,
                   label=f"Ort = {mean_d:.4f}")
        ax.axvline(ci_low,  color="#e74c3c", linestyle=":", linewidth=2,
                   label=f"%2.5 = {ci_low:.4f}")
        ax.axvline(ci_high, color="#e74c3c", linestyle=":", linewidth=2,
                   label=f"%97.5 = {ci_high:.4f}")
        ax.axvspan(ci_low, ci_high, alpha=0.15, color=color)

        sig_text = "Anlamli" if sig else "Anlamli degil"
        ax.set_title(f"{name}\n{sig_text}", fontweight="bold", fontsize=12,
                     color="#27ae60" if sig else "#e74c3c")
        ax.set_xlabel("F1-Macro Farki (dF1)", fontsize=11)
        ax.set_ylabel("Yogunluk", fontsize=11)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Bootstrap %95 Guven Araligi — F1-Macro Fark Dagilimi (n={1000})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Bootstrap CI: {save_path}")


# ─── Ana Fonksiyon ──────────────────────────────────────────────────────────

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  ISTATISTIKSEL ANLAMLILIK TESTLERI")
    print("=" * 65)

    exp_dirs = {}
    for exp_name in ["exp0", "exp1", "exp2", "exp3", "exp4"]:
        exp_dir = ROOT / "results" / exp_name
        preds_path  = exp_dir / "test_preds.npy"
        labels_path = exp_dir / "test_labels.npy"
        if preds_path.exists() and labels_path.exists():
            exp_dirs[exp_name] = {
                "preds":  np.load(preds_path),
                "labels": np.load(labels_path),
            }
            print(f"[YUKLENDI] {exp_name}: {len(exp_dirs[exp_name]['preds']):,} ornek")

    if len(exp_dirs) < 2:
        print("[HATA] En az 2 deney sonucu gerekiyor.")
        return

    label_set = None
    for data in exp_dirs.values():
        if label_set is None:
            label_set = data["labels"]
        elif not np.array_equal(data["labels"], label_set):
            print("[UYARI] Farkli test etiketleri!")
    labels = label_set

    pairs = []
    if "exp2" in exp_dirs and "exp3" in exp_dirs:
        pairs.append(("exp2", "exp3", "EXP2 vs EXP3"))
    if "exp1" in exp_dirs and "exp3" in exp_dirs:
        pairs.append(("exp1", "exp3", "EXP1 vs EXP3"))
    if "exp1" in exp_dirs and "exp4" in exp_dirs:
        pairs.append(("exp1", "exp4", "EXP1 vs EXP4"))
    if "exp2" in exp_dirs and "exp4" in exp_dirs:
        pairs.append(("exp2", "exp4", "EXP2 vs EXP4"))
    if "exp3" in exp_dirs and "exp4" in exp_dirs:
        pairs.append(("exp3", "exp4", "EXP3 vs EXP4"))

    all_results = {}
    comparisons_for_plot = []

    for name_a, name_b, pair_name in pairs:
        preds_a = exp_dirs[name_a]["preds"]
        preds_b = exp_dirs[name_b]["preds"]

        print(f"\n{'─'*65}")
        print(f"  {pair_name}")
        print(f"{'─'*65}")

        f1_a  = f1_score(labels, preds_a, average="macro", zero_division=0)
        f1_b  = f1_score(labels, preds_b, average="macro", zero_division=0)
        acc_a = accuracy_score(labels, preds_a)
        acc_b = accuracy_score(labels, preds_b)

        print(f"  {name_a.upper()} F1-Macro: {f1_a:.4f}  |  Accuracy: {acc_a:.4f}")
        print(f"  {name_b.upper()} F1-Macro: {f1_b:.4f}  |  Accuracy: {acc_b:.4f}")
        print(f"  dF1-Macro: {f1_b - f1_a:+.4f}")

        print("\n  [1] McNemar Testi:")
        mc = mcnemar_test(preds_a, preds_b, labels, name_a.upper(), name_b.upper())
        ct = mc["contingency_table"]
        print(f"      b = {ct[f'{name_b.upper()}_only_correct (b)']}")
        print(f"      c = {ct[f'{name_a.upper()}_only_correct (c)']}")
        print(f"      x2 = {mc['mcnemar_statistic']:.4f}  p = {mc['p_value']:.6f}")
        print(f"      -> {mc['interpretation']}")

        print(f"\n  [2] Bootstrap %95 Guven Araligi (n=1000):")
        br, diffs = bootstrap_ci(labels, preds_a, preds_b, n_bootstrap=1000, seed=42)
        print(f"      dF1 ortalamasi: {br['mean']:+.4f}")
        print(f"      %95 GA: [{br['ci_low']:.4f}, {br['ci_high']:.4f}]")
        print(f"      Anlamli: {'Evet' if br['significant'] else 'Hayir'}")

        all_results[pair_name] = {
            "original_metrics": {
                f"{name_a}_f1_macro": round(f1_a, 4),
                f"{name_b}_f1_macro": round(f1_b, 4),
                "delta_f1_macro":     round(f1_b - f1_a, 4),
                f"{name_a}_accuracy": round(acc_a, 4),
                f"{name_b}_accuracy": round(acc_b, 4),
            },
            "mcnemar":      mc,
            "bootstrap_ci": br,
        }
        comparisons_for_plot.append((pair_name, br, diffs))

    # Kaydet
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):  return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.bool_):    return bool(obj)
            if isinstance(obj, np.ndarray):  return obj.tolist()
            return super().default(obj)

    output_path = RESULTS / "statistical_tests.json"
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing.update(all_results)
            all_results = existing
        except (json.JSONDecodeError, ValueError):
            pass
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, cls=NpEncoder)
    print(f"\n[KAYIT] {output_path}")

    if comparisons_for_plot:
        save_bootstrap_plot(comparisons_for_plot, RESULTS / "bootstrap_ci.png")

    print("\n" + "=" * 75)
    print("  OZET TABLO")
    print("=" * 75)
    print(f"  {'Karsilastirma':25s} {'dF1':>8s} {'McNemar p':>12s} {'%95 GA':>22s} {'Anlamli':>8s}")
    print("─" * 75)
    for pname, data in all_results.items():
        if "original_metrics" not in data:
            continue
        delta = data["original_metrics"]["delta_f1_macro"]
        pval  = data["mcnemar"]["p_value"]
        ci    = data["bootstrap_ci"]
        sig   = "Evet" if ci["significant"] else "Hayir"
        print(f"  {pname:25s} {delta:+8.4f} {pval:12.6f} "
              f"[{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}] {sig:>8s}")
    print("=" * 75)


if __name__ == "__main__":
    main()
