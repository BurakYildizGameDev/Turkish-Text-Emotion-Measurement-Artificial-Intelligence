"""
v2 ablation — tüm düzenleri aynı bütçe ve aynı seed'lerle koşar, sonra özetler.

    python -m src.training.run_ablation_v2            # eksik çalıştırmaları koşar + özet
    python -m src.training.run_ablation_v2 --summary  # yalnızca özet

Her çalıştırma ayrı süreçte koşar (GPU belleği her seferinde tamamen boşalsın).
metrics.json'u zaten olan çalıştırmalar atlanır → yarıda kesilirse kaldığı yerden devam eder.

Düzen seçimi YALNIZCA val macro-F1 ortalamasına göre yapılır; test skorları raporlanır ama seçimde kullanılmaz.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.data.labels import EMOTIONS

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "results" / "v2" / "runs"
SUMMARY_JSON = ROOT / "results" / "v2" / "ablation_summary.json"
SUMMARY_MD = ROOT / "results" / "v2" / "ablation_summary.md"

SEEDS = [42, 43, 44]
# grup → { düzen adı: train_v2 argümanları }
GROUPS = {
    "classic": {    # Faz 5 — klasik baseline (deterministik, tek çalıştırma; None = ayrı script)
        "tfidf_lr": None,
    },
    "ablation": {   # Faz 4 — veri ve dengesizlik yöntemi
        "gold":        ["--data", "gold", "--imbalance", "weighted_loss"],
        "gold_none":   ["--data", "gold", "--imbalance", "none"],
        "gold_weak":   ["--data", "gold_weak", "--imbalance", "weighted_loss"],
        "gold_silver": ["--data", "gold_silver", "--imbalance", "weighted_loss"],
    },
    "baselines": {  # Faz 5 — aynı veri/bütçe ile çok dilli modeller
        "mbert_none":  ["--data", "gold", "--imbalance", "none", "--model", "mbert"],
        "xlmr_none":   ["--data", "gold", "--imbalance", "none", "--model", "xlmr"],
    },
    "lr": {         # Faz 5 — seçilen düzen için öğrenme hızı (varsayılan 2e-5 = gold_none)
        "gold_none_lr1e-5": ["--data", "gold", "--imbalance", "none", "--lr", "1e-5"],
        "gold_none_lr3e-5": ["--data", "gold", "--imbalance", "none", "--lr", "3e-5"],
        "gold_none_lr5e-5": ["--data", "gold", "--imbalance", "none", "--lr", "5e-5"],
    },
}
CONFIGS = {name: args for group in GROUPS.values() for name, args in group.items()}
DESCRIPTIONS = {
    "tfidf_lr": "TF-IDF (kelime+karakter n-gram) + Logistic Regression, yalnızca gold",
    "gold":"BERTurk, yalnızca gold, ağırlıklı loss",
    "gold_none": "BERTurk, yalnızca gold, ağırlıksız loss",
    "gold_weak": "BERTurk, gold + weak (sınıf başına ≤5000), ağırlıklı loss",
    "gold_silver": "BERTurk, gold + silver (nihal_8class), ağırlıklı loss",
    "mbert_none": "mBERT (çok dilli), yalnızca gold, ağırlıksız",
    "xlmr_none": "XLM-RoBERTa-base, yalnızca gold, ağırlıksız",
    "gold_none_lr1e-5": "BERTurk gold_none, lr=1e-5",
    "gold_none_lr3e-5": "BERTurk gold_none, lr=3e-5",
    "gold_none_lr5e-5": "BERTurk gold_none, lr=5e-5",
}


def run_all(groups: list[str]) -> None:
    for group in groups:
        for name, args in GROUPS[group].items():
            if args is None:   # deterministik klasik baseline — tek çalıştırma
                if not (RUNS_DIR / f"{name}_s{SEEDS[0]}" / "metrics.json").exists():
                    print(f"[başla] {name}", flush=True)
                    subprocess.run([sys.executable, "-m", "src.evaluation.tfidf_baseline"], cwd=ROOT, check=True)
                continue
            for seed in SEEDS:
                tag = f"{name}_s{seed}"
                if (RUNS_DIR / tag / "metrics.json").exists():
                    print(f"[atla] {tag}", flush=True)
                    continue
                cmd = [sys.executable, "-m", "src.training.train_v2", *args,
                       "--seed", str(seed), "--tag", tag]
                if seed == SEEDS[0]:
                    cmd.append("--save-model")
                print(f"[başla] {tag}: {' '.join(cmd[2:])}", flush=True)
                subprocess.run(cmd, cwd=ROOT, check=True)


def _ms(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": round(float(a.mean()), 4), "std": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 4),
            "values": [round(float(v), 4) for v in a]}


def summarize() -> dict:
    summary = {"seeds": SEEDS, "secim_olcutu": "val_f1_macro ortalaması (test seçimde kullanılmaz)", "configs": {}}
    for name in CONFIGS:
        runs = [json.loads((RUNS_DIR / f"{name}_s{s}" / "metrics.json").read_text(encoding="utf-8"))
                for s in SEEDS if (RUNS_DIR / f"{name}_s{s}" / "metrics.json").exists()]
        if not runs:
            continue
        sources = sorted(runs[0]["test"]["per_source"])
        summary["configs"][name] = {
            "aciklama": DESCRIPTIONS[name],
            "n_runs": len(runs),
            "train_size": runs[0]["train_size"],
            "val_f1_macro": _ms([r["val_f1_macro"] for r in runs]),
            "test_f1_macro": _ms([r["test"]["f1_macro"] for r in runs]),
            "test_accuracy": _ms([r["test"]["accuracy"] for r in runs]),
            "test_per_class_f1": {e: _ms([r["test"]["per_class"][e]["f1-score"] for r in runs]) for e in EMOTIONS},
            "test_per_source_f1": {s: _ms([r["test"]["per_source"][s]["f1_macro_own_classes"] for r in runs]) for s in sources},
            "ceviri_kisa_yolu_orani": _ms([r["test"]["ceviri_kisa_yolu"]["orani"] for r in runs]),
            "ece_before": _ms([r["calibration"]["test_ece_before"] for r in runs]),
            "ece_after": _ms([r["calibration"]["test_ece_after"] for r in runs]),
            "best_epoch": [r["best_epoch"] for r in runs],
            "runtime_min": _ms([r["runtime_min"] for r in runs]),
        }
    cfg = summary["configs"]
    if cfg:
        summary["secilen_duzen"] = max(cfg, key=lambda n: cfg[n]["val_f1_macro"]["mean"])
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SUMMARY_MD.write_text(_markdown(summary), encoding="utf-8")
    print(SUMMARY_MD.read_text(encoding="utf-8"))
    return summary


def _fmt(d: dict) -> str:
    return f"{d['mean']:.4f} ± {d['std']:.4f}"


def _markdown(s: dict) -> str:
    cfg = s["configs"]
    lines = [
        "# v2 Deney Özeti (ablation + baseline + öğrenme hızı)",
        "",
        f"Seed'ler: {s['seeds']} · ortalama ± standart sapma · düzen seçimi: **{s['secim_olcutu']}**",
        f"Seçilen düzen: **{s.get('secilen_duzen')}**",
        "",
        "| Düzen | Train | Val F1-macro | Test F1-macro | Test Acc | Çeviri kısa yolu | ECE önce → sonra |",
        "|---|---|---|---|---|---|---|",
    ]
    for n, c in cfg.items():
        lines.append(f"| `{n}` — {c['aciklama']} | {c['train_size']:,} | {_fmt(c['val_f1_macro'])} | "
                     f"{_fmt(c['test_f1_macro'])} | {_fmt(c['test_accuracy'])} | "
                     f"{c['ceviri_kisa_yolu_orani']['mean']:.3f} | "
                     f"{c['ece_before']['mean']:.3f} → {c['ece_after']['mean']:.3f} |")
    lines += ["", "## Sınıf bazlı test F1 (ortalama)", "",
              "| Sınıf | " + " | ".join(f"`{n}`" for n in cfg) + " |",
              "|---|" + "---|" * len(cfg)]
    for e in EMOTIONS:
        lines.append(f"| {e} | " + " | ".join(f"{cfg[n]['test_per_class_f1'][e]['mean']:.3f}" for n in cfg) + " |")
    srcs = sorted({x for c in cfg.values() for x in c["test_per_source_f1"]})
    lines += ["", "## Kaynak bazlı test F1 (kaynağın kendi sınıfları üzerinden, ortalama)", "",
              "| Kaynak | " + " | ".join(f"`{n}`" for n in cfg) + " |",
              "|---|" + "---|" * len(cfg)]
    for src in srcs:
        lines.append(f"| {src} | " + " | ".join(
            f"{cfg[n]['test_per_source_f1'][src]['mean']:.3f}" if src in cfg[n]["test_per_source_f1"] else "—"
            for n in cfg) + " |")
    lines += ["", "Çeviri kısa yolu: TREMO/tweet (ana dil) test örneklerinin nötr veya sevgi tahmin edilme oranı. "
              "Bu kaynaklarda gold nötr/sevgi olmadığı için her biri hatadır.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true", help="Yalnızca özet üret")
    ap.add_argument("--groups", nargs="+", choices=list(GROUPS), default=list(GROUPS),
                    help="Koşulacak gruplar (varsayılan: hepsi; tamamlananlar atlanır)")
    a = ap.parse_args()
    if not a.summary:
        run_all(a.groups)
    summarize()
