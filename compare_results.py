# Commit 92: feat(eval): compare exp0 vs exp1 in compare_results reporter
"""
EXP2 baseline sonuclari ile test_augmented sonuclari karsilastirir.

Calistirmak icin: python compare_results.py
(evaluate_augmented.py'yi once calistirmaniz gerekir)
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

BASELINE_PATH  = ROOT / "results" / "exp2" / "training_summary.json"
AUGMENTED_PATH = ROOT / "results" / "augmented_test_results.json"
OUT_CSV_PATH   = ROOT / "results" / "augmented_comparison_table.csv"

# Baseline veride Turkce karakter kullanan anahtarlar
BASELINE_LABEL_MAP = {
    "gurur":     "gurur",
    "utanc":     "utanc",
    "mutluluk":  "mutluluk",
    "uzuntu":    "uzuntu",
    "ofke":      "ofke",
    "korku":     "korku",
    "saskınlık": "saskınlık",
    "tiksinti":  "tiksinti",
    "sevgi":     "sevgi",
    "notr":      "notr",
}

# Baseline JSON'da Turkce karakterli anahtarlar
BASELINE_TR_KEYS = {
    "gurur":     "gurur",
    "utanc":     "utanç",
    "mutluluk":  "mutluluk",
    "uzuntu":    "üzüntü",  # üzüntü
    "ofke":      "öfke",         # öfke
    "korku":     "korku",
    "saskınlık": "şaşkınlık",  # şaşkınlık
    "tiksinti":  "tiksinti",
    "sevgi":     "sevgi",
    "notr":      "nötr",         # nötr
}

# Onceki test setindeki destek sayilari (exp2 metrics_report.json'dan)
BASELINE_SUPPORT = {
    "mutluluk":  41960,
    "uzuntu":    9947,
    "ofke":      802,
    "korku":     95,
    "saskınlık": 660,
    "tiksinti":  87,
    "sevgi":     753,
    "notr":      28183,
    "utanc":     37,
    "gurur":     8,
}


def load_baseline() -> dict:
    with open(str(BASELINE_PATH), "r", encoding="utf-8") as f:
        data = json.load(f)
    per_class_raw = data.get("per_class", {})

    result = {}
    for norm_key, tr_key in BASELINE_TR_KEYS.items():
        row = per_class_raw.get(tr_key, per_class_raw.get(norm_key, {}))
        result[norm_key] = {
            "f1":      row.get("f1", None),
            "support": BASELINE_SUPPORT.get(norm_key, row.get("support", None)),
        }
    return result


# augmented JSON'da model config'in Turkce anahtarlari kullanilir
AUG_TR_KEYS = {
    "gurur":     "gurur",
    "utanc":     "utanç",
    "mutluluk":  "mutluluk",
    "uzuntu":    "üzüntü",
    "ofke":      "öfke",
    "korku":     "korku",
    "saskınlık": "şaşkınlık",
    "tiksinti":  "tiksinti",
    "sevgi":     "sevgi",
    "notr":      "nötr",
}


def load_augmented() -> dict:
    if not AUGMENTED_PATH.exists():
        return {}
    with open(str(AUGMENTED_PATH), "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("per_class", {})
    # Normalize: Turkce anahtarlari (model ciktisi) -> translitere anahtarlara cevir
    norm = {}
    for norm_key, tr_key in AUG_TR_KEYS.items():
        if tr_key in raw:
            norm[norm_key] = raw[tr_key]
        elif norm_key in raw:
            norm[norm_key] = raw[norm_key]
    return norm


def build_table(baseline: dict, augmented: dict) -> list[dict]:
    rows = []
    all_keys = list(BASELINE_SUPPORT.keys())

    for key in all_keys:
        b = baseline.get(key, {})
        a = augmented.get(key, {})

        old_support = b.get("support")
        new_support = a.get("support")
        old_f1      = b.get("f1")
        new_f1      = a.get("f1")

        if old_f1 is not None and new_f1 is not None:
            delta = round(new_f1 - old_f1, 4)
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        else:
            delta_str = "[?]"

        rows.append({
            "sinif":       key,
            "eski_support": old_support if old_support is not None else "?",
            "yeni_support": new_support if new_support is not None else "?",
            "eski_f1":      f"{old_f1:.3f}" if old_f1 is not None else "?",
            "yeni_f1":      f"{new_f1:.3f}" if new_f1 is not None else "[calistir]",
            "degisim":      delta_str,
        })
    return rows


def print_table(rows: list[dict]) -> None:
    header = f"{'Sinif':<14} | {'Eski ornek':>10} | {'Yeni ornek':>10} | {'Eski F1':>8} | {'Yeni F1':>10} | {'Degisim':>10}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        line = (
            f"{r['sinif']:<14} | {str(r['eski_support']):>10} | "
            f"{str(r['yeni_support']):>10} | {r['eski_f1']:>8} | "
            f"{r['yeni_f1']:>10} | {r['degisim']:>10}"
        )
        # Odak siniflar icin isaretci
        if r["sinif"] in ("gurur", "utanc"):
            line += "  <<<"
        print(line)
    print(sep)


def save_csv(rows: list[dict]) -> None:
    import csv
    OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(str(OUT_CSV_PATH), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sinif", "eski_support", "yeni_support", "eski_f1", "yeni_f1", "degisim"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Karsilastirma tablosu kaydedildi: {OUT_CSV_PATH}")


def run_comparison() -> None:
    print("=" * 60)
    print("KARSILASTIRMA: EXP2 Baseline  vs  test_augmented")
    print("=" * 60)

    if not BASELINE_PATH.exists():
        print(f"[HATA] Baseline bulunamadi: {BASELINE_PATH}")
        sys.exit(1)

    if not AUGMENTED_PATH.exists():
        print(f"[UYARI] Augmented sonuclar henuz yok: {AUGMENTED_PATH}")
        print("        Once evaluate_augmented.py calistirin.")
        print()
        print("Mevcut baseline ile on izleme tablosu:")
        baseline = load_baseline()
        augmented = {}
    else:
        baseline  = load_baseline()
        augmented = load_augmented()

    rows = build_table(baseline, augmented)
    print()
    print_table(rows)
    print()

    # Ozet — gurur / utanc
    gurur_row = next((r for r in rows if r["sinif"] == "gurur"), {})
    utanc_row = next((r for r in rows if r["sinif"] == "utanc"), {})

    print("ODAK SINIF OZETI:")
    print(f"  gurur : {gurur_row.get('eski_support','?')} -> {gurur_row.get('yeni_support','?')} ornek | "
          f"F1: {gurur_row.get('eski_f1','?')} -> {gurur_row.get('yeni_f1','?')}  ({gurur_row.get('degisim','?')})")
    print(f"  utanc : {utanc_row.get('eski_support','?')} -> {utanc_row.get('yeni_support','?')} ornek | "
          f"F1: {utanc_row.get('eski_f1','?')} -> {utanc_row.get('yeni_f1','?')}  ({utanc_row.get('degisim','?')})")
    print()

    if augmented:
        save_csv(rows)


if __name__ == "__main__":
    run_comparison()
