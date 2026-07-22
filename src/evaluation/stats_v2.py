"""
Faz 5 — istatistiksel karşılaştırma ve hata analizi.

    python -m src.evaluation.stats_v2

Girdi : results/v2/runs/<düzen>_s<seed>/test_logits.npy (train_v2 / tfidf_baseline çıktıları)
Çıktı : results/v2/stats/significance.json, stats_report.md
        results/v2/stats/error_analysis.json, confusion_matrix.png, errors_sample.csv

Yöntem
  * Referans: val F1 ile seçilen düzen (ablation_summary.json → secilen_duzen).
  * Paired bootstrap (B=10.000): her yeniden örneklemede test örnekleri ortak seçilir; her düzenin
    macro-F1'i seed'ler üzerinden ortalanır, fark alınır. %95 güven aralığı + iki yönlü p-değeri.
    Böylece hem test örneklemesi belirsizliği hem seed ortalaması hesaba katılır.
  * McNemar (tam binom testi): her seed çifti için ayrı (aynı seed'ler eşleştirilir).
  * Çoklu karşılaştırma: bootstrap p-değerlerine Holm düzeltmesi.
  * Kontrollü karşılaştırmalar: tek etkenin değiştiği çiftler (CONTROLLED), ör. weak veri etkisi = gold_weak − gold.
  * Kaynak bazlı %95 GA: referans düzen için her kaynakta ayrı bootstrap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.data.labels import EMOTIONS, NUM_LABELS

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "results" / "v2" / "runs"
OUT_DIR = ROOT / "results" / "v2" / "stats"
TEST_PATH = ROOT / "data" / "v2" / "test.parquet"
SUMMARY = ROOT / "results" / "v2" / "ablation_summary.json"
SEEDS = [42, 43, 44]
B = 10_000
RNG_SEED = 0
# Tek bir etkenin değiştiği çiftler (A − B). Referansla karşılaştırma iki etkeni birden değiştirebilir
# (ör. gold_none − gold_weak: hem weak veri hem loss ağırlığı), bu yüzden etkiler ayrıca ölçülür.
CONTROLLED = [
    ("gold_weak", "gold", "weak veri eklemenin etkisi (ikisi de ağırlıklı loss)"),
    ("gold_silver", "gold", "silver veri eklemenin etkisi (ikisi de ağırlıklı loss)"),
    ("gold_none", "gold", "loss ağırlığını kaldırmanın etkisi (ikisi de yalnızca gold)"),
]
# TREMO lisansı gereği TREMO metinleri repoya konan hata örneklerine yazılmaz
PUBLIC_TEXT_SOURCES = {"goemotions_tr", "tweet_emotion"}


def load_preds(config: str) -> np.ndarray:
    """(n_seed, n_test) tahmin matrisi."""
    preds = [np.load(RUNS_DIR / f"{config}_s{s}" / "test_logits.npy").argmax(1)
             for s in SEEDS if (RUNS_DIR / f"{config}_s{s}" / "test_logits.npy").exists()]
    return np.stack(preds)


def macro_f1_batch(y: np.ndarray, p: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """idx: (B, n) bootstrap indeksleri → (B,) macro-F1 (test'te olmayan sınıf 0 sayılmaz, labels=all)."""
    yt, yp = y[idx], p[idx]
    f1 = np.zeros(idx.shape[0])
    for c in range(NUM_LABELS):
        tp = ((yt == c) & (yp == c)).sum(1)
        fp = ((yt != c) & (yp == c)).sum(1)
        fn = ((yt == c) & (yp != c)).sum(1)
        denom = 2 * tp + fp + fn
        f1 += np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return f1 / NUM_LABELS


CHUNK = 500


def boot_chunks(n: int):
    """Bootstrap indekslerini parça parça, sabit tohumlardan üretir (hepsini bellekte tutmadan).
    Aynı parça numarası her düzen için aynı indeksleri verir → karşılaştırma eşleştirilmiş kalır."""
    for i in range(B // CHUNK):
        yield np.random.default_rng(RNG_SEED + i).integers(0, n, size=(CHUNK, n))


def bootstrap_seed_mean_f1(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """(B,) — her yeniden örneklemede seed'ler üzerinden ortalama macro-F1."""
    return np.concatenate([np.mean([macro_f1_batch(y, P[s], idx) for s in range(P.shape[0])], axis=0)
                           for idx in boot_chunks(len(y))])


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def mcnemar_exact(y, pa, pb) -> dict:
    a_ok, b_ok = pa == y, pb == y
    b = int((a_ok & ~b_ok).sum())   # yalnız A doğru
    c = int((~a_ok & b_ok).sum())   # yalnız B doğru
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"yalniz_ref_dogru": b, "yalniz_diger_dogru": c, "p": float(p)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    ref = summary["secilen_duzen"]
    configs = [c for c in summary["configs"] if c != ref]
    test = pd.read_parquet(TEST_PATH)
    y = test["label_id"].to_numpy()
    n = len(y)

    rng = np.random.default_rng(RNG_SEED)
    P_ref = load_preds(ref)
    f_ref = bootstrap_seed_mean_f1(y, P_ref)
    point_ref = float(np.mean([macro_f1_batch(y, P_ref[s], np.arange(n)[None])[0] for s in range(len(P_ref))]))

    results, pvals = {}, {}
    for cfg in configs:
        P = load_preds(cfg)
        f = bootstrap_seed_mean_f1(y, P)
        diff = f_ref - f
        point = point_ref - float(np.mean([macro_f1_batch(y, P[s], np.arange(n)[None])[0] for s in range(len(P))]))
        p_boot = float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))
        pvals[cfg] = p_boot
        mc = [mcnemar_exact(y, P_ref[s], P[s]) for s in range(min(len(P_ref), len(P)))]
        results[cfg] = {
            "n_seed": int(len(P)),
            "f1_fark_ref_eksi_diger": round(point, 4),
            "ci95": [round(float(np.percentile(diff, 2.5)), 4), round(float(np.percentile(diff, 97.5)), 4)],
            "p_bootstrap": round(p_boot, 5),
            "mcnemar_seed_bazli": mc,
        }
        print(f"  {ref} − {cfg}: {point:+.4f}  GA95 {results[cfg]['ci95']}  p={p_boot:.4g}", flush=True)
    for cfg, p_adj in holm(pvals).items():
        results[cfg]["p_holm"] = round(p_adj, 5)
        results[cfg]["anlamli_0.05"] = p_adj < 0.05

    controlled = []
    for a, b, desc in CONTROLLED:
        if a not in summary["configs"] or b not in summary["configs"]:
            continue
        Pa, Pb = load_preds(a), load_preds(b)
        diff = bootstrap_seed_mean_f1(y, Pa) - bootstrap_seed_mean_f1(y, Pb)
        point = float(np.mean([macro_f1_batch(y, Pa[s], np.arange(n)[None])[0] for s in range(len(Pa))])
                      - np.mean([macro_f1_batch(y, Pb[s], np.arange(n)[None])[0] for s in range(len(Pb))]))
        p_boot = float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))
        controlled.append({"a": a, "b": b, "aciklama": desc, "f1_fark_a_eksi_b": round(point, 4),
                           "ci95": [round(float(np.percentile(diff, 2.5)), 4), round(float(np.percentile(diff, 97.5)), 4)],
                           "p_bootstrap": round(p_boot, 5),
                           "mcnemar_p_seed_bazli": [float(f"{mcnemar_exact(y, Pb[s], Pa[s])['p']:.3g}")
                                                    for s in range(min(len(Pa), len(Pb)))]})
        print(f"  {a} − {b}: {point:+.4f}  GA95 {controlled[-1]['ci95']}  p={p_boot:.4g}", flush=True)

    # Referans düzen için kaynak bazlı güven aralıkları (seed ortalaması, kaynağın kendi sınıfları)
    per_source = {}
    for src in sorted(test["subsource"].unique()):
        sel = np.flatnonzero(test["subsource"].to_numpy() == src)
        ys = y[sel]
        present = np.unique(ys)
        idx = rng.integers(0, len(sel), size=(2000, len(sel)))
        vals = []
        for s in range(len(P_ref)):
            ps = P_ref[s][sel]
            f1 = np.zeros(2000)
            for c in present:
                yt, yp = ys[idx], ps[idx]
                tp = ((yt == c) & (yp == c)).sum(1); fp = ((yt != c) & (yp == c)).sum(1); fn = ((yt == c) & (yp != c)).sum(1)
                f1 += 2 * tp / np.maximum(2 * tp + fp + fn, 1)
            vals.append(f1 / len(present))
        v = np.mean(vals, axis=0)
        per_source[src] = {"n": int(len(sel)), "f1_mean": round(float(v.mean()), 4),
                           "ci95": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]}

    sig = {"referans": ref, "B": B, "seeds": SEEDS, "referans_test_f1": round(point_ref, 4),
           "karsilastirmalar": results, "kontrollu_karsilastirmalar": controlled, "referans_kaynak_bazli": per_source}
    (OUT_DIR / "significance.json").write_text(json.dumps(sig, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    err = error_analysis(ref, test)
    (OUT_DIR / "stats_report.md").write_text(report_md(sig, err, summary), encoding="utf-8")
    print((OUT_DIR / "stats_report.md").read_text(encoding="utf-8"))


# ─── Hata analizi ────────────────────────────────────────────────────────────

def error_analysis(ref: str, test: pd.DataFrame) -> dict:
    run = RUNS_DIR / f"{ref}_s{SEEDS[0]}"
    logits = np.load(run / "test_logits.npy")
    T = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["calibration"]["temperature"]
    probs = np.exp(logits / T - (logits / T).max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    y, p = test["label_id"].to_numpy(), probs.argmax(1)
    conf = probs.max(1)
    wrong = p != y

    cm = np.zeros((NUM_LABELS, NUM_LABELS), dtype=int)
    np.add.at(cm, (y, p), 1)
    cm_norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_LABELS), EMOTIONS, rotation=45, ha="right")
    ax.set_yticks(range(NUM_LABELS), EMOTIONS)
    for i in range(NUM_LABELS):
        for j in range(NUM_LABELS):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
    ax.set_title(f"{ref} (seed {SEEDS[0]}) — satır normalize karışıklık matrisi")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)

    pairs = [(EMOTIONS[i], EMOTIONS[j], int(cm[i, j])) for i in range(NUM_LABELS) for j in range(NUM_LABELS) if i != j]
    pairs.sort(key=lambda x: -x[2])
    by_source = {}
    for src, g in test.assign(wrong=wrong, conf=conf).groupby("subsource"):
        w = g[g["wrong"]]
        by_source[src] = {"n": len(g), "hata": int(len(w)), "hata_orani": round(len(w) / len(g), 4),
                          "yuksek_guvenli_hata_(>0.9)": int((w["conf"] > 0.9).sum())}

    df = test.assign(tahmin=[EMOTIONS[i] for i in p], guven=conf.round(3), dogru=~wrong)
    sample = (df[wrong & df["source"].isin(PUBLIC_TEXT_SOURCES).to_numpy()]
              .sort_values("guven", ascending=False)
              .groupby(["label", "tahmin"], sort=False).head(3)
              .head(60)[["subsource", "label", "tahmin", "guven", "text"]])
    sample.to_csv(OUT_DIR / "errors_sample.csv", index=False, encoding="utf-8-sig")

    out = {"referans_calistirma": run.name, "temperature": T,
           "en_sik_karisiklik": [{"gercek": a, "tahmin": b, "n": c} for a, b, c in pairs[:12]],
           "kaynak_bazli_hata": by_source,
           "confusion_matrix": cm.tolist()}
    (OUT_DIR / "error_analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def _p(p: float) -> str:
    """Bootstrap p=0 → B örneklemin hiçbirinde fark işaret değiştirmedi, yani p < 1/B."""
    return f"<{1 / B:.0e}".replace("e-0", "e-") if p == 0 else f"{p:.4g}"


def report_md(sig: dict, err: dict, summary: dict) -> str:
    ref = sig["referans"]
    L = [f"# Faz 5 — İstatistiksel Karşılaştırma ve Hata Analizi", "",
         f"Referans (val F1 ile seçildi): **`{ref}`** — test macro-F1 (seed ortalaması) **{sig['referans_test_f1']}**",
         f"Paired bootstrap B={sig['B']:,} (test örnekleri ortak yeniden örneklenir, seed'ler ortalanır) · "
         "McNemar tam binom testi seed bazında · Holm düzeltmesi", "",
         "## Referans ile karşılaştırmalar", "",
         "| Düzen | Seed | Test F1 | Fark (ref − düzen) | %95 GA | p (Holm) | Anlamlı | McNemar p (seed başına) |",
         "|---|---|---|---|---|---|---|---|"]
    order = sorted(sig["karsilastirmalar"].items(), key=lambda kv: kv[1]["f1_fark_ref_eksi_diger"])
    for cfg, r in order:
        tf1 = summary["configs"][cfg]["test_f1_macro"]["mean"]
        mc = ", ".join(f"{m['p']:.2g}" for m in r["mcnemar_seed_bazli"])
        L.append(f"| `{cfg}` | {r['n_seed']} | {tf1:.4f} | {r['f1_fark_ref_eksi_diger']:+.4f} | "
                 f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | {_p(r['p_holm'])} | "
                 f"{'evet' if r['anlamli_0.05'] else 'hayır'} | {mc} |")
    L += ["", "Pozitif fark: referans daha iyi. GA sıfırı içeriyorsa fark istatistiksel olarak ayırt edilemez.", "",
          "## Kontrollü karşılaştırmalar (tek etken değişir)", "",
          "Referansla karşılaştırma birden fazla etkeni aynı anda değiştirebilir (ör. `gold_none` − `gold_weak`: "
          "hem weak veri hem loss ağırlığı). Tek bir etkenin katkısı aşağıdaki çiftlerle ölçülür "
          "(düzeltilmemiş p; planlı karşılaştırmalar).", "",
          "| A − B | Ölçülen etki | Fark | %95 GA | p (bootstrap) | McNemar p (seed başına) |",
          "|---|---|---|---|---|---|"]
    for c in sig.get("kontrollu_karsilastirmalar", []):
        L.append(f"| `{c['a']}` − `{c['b']}` | {c['aciklama']} | {c['f1_fark_a_eksi_b']:+.4f} | "
                 f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | {_p(c['p_bootstrap'])} | "
                 + ", ".join(f"{m:.2g}" for m in c["mcnemar_p_seed_bazli"]) + " |")
    L += ["",
          f"## `{ref}` — kaynak bazlı test F1 (%95 GA)", "",
          "| Kaynak | n | F1 (kendi sınıfları) | %95 GA |", "|---|---|---|---|"]
    for s, v in sig["referans_kaynak_bazli"].items():
        L.append(f"| {s} | {v['n']:,} | {v['f1_mean']:.4f} | [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] |")
    L += ["", f"## Hata analizi (`{err['referans_calistirma']}`, kalibre T={err['temperature']})", "",
          "![karışıklık matrisi](confusion_matrix.png)", "", "En sık karışan çiftler:", "",
          "| Gerçek | Tahmin | n |", "|---|---|---|"]
    for r in err["en_sik_karisiklik"]:
        L.append(f"| {r['gercek']} | {r['tahmin']} | {r['n']} |")
    L += ["", "| Kaynak | n | Hata | Hata oranı | Yüksek güvenli hata (>0.9) |", "|---|---|---|---|---|"]
    for s, v in err["kaynak_bazli_hata"].items():
        L.append(f"| {s} | {v['n']:,} | {v['hata']} | {v['hata_orani']:.3f} | {v['yuksek_guvenli_hata_(>0.9)']} |")
    L += ["", "Örnek hatalar: `errors_sample.csv` (TREMO lisansı gereği yalnızca GoEmotions/tweet metinleri).", ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
