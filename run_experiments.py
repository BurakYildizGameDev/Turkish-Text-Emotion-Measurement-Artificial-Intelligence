"""
Ablation Study Orchestrator
===========================
Dört deneyi sırayla çalıştırır.
Her deney kendi Python sürecinde çalışır (VRAM temizlenir).
Checkpoint varsa ilgili deney atlanır (--force ile zorla yeniden çalıştır).

Kullanım:
  python run_experiments.py               # tümünü çalıştır
  python run_experiments.py --only 1 4    # sadece Exp1 ve Exp4
  python run_experiments.py --force       # mevcut sonuçları sil, yeniden çalıştır
  python run_experiments.py --dry-run     # neyin çalışacağını göster, çalıştırma
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent

# --- Deney Tanımları --------------------------------------------------------

EXPERIMENTS = [
    {
        "id":          1,
        "name":        "Baseline BERTurk",
        "script":      ROOT / "train_exp1_baseline.py",
        "result_json": ROOT / "results" / "exp1" / "metrics_report.json",
        "results_dir": ROOT / "results" / "exp1",
        "description": "Sadece BERTurk + WeightedRandomSampler",
    },
    {
        "id":          2,
        "name":        "Cross-Lingual Enhanced",
        "script":      ROOT / "train_exp2_crosslingual.py",
        "result_json": ROOT / "results" / "exp2" / "metrics_report.json",
        "results_dir": ROOT / "results" / "exp2",
        "description": "BERTurk + GoEmotions TR oversample (azinlik sinif boost)",
    },
    {
        "id":          3,
        "name":        "Lexicon Informed",
        "script":      ROOT / "train_exp3_lexicon.py",
        "result_json": ROOT / "results" / "exp3" / "metrics_report.json",
        "results_dir": ROOT / "results" / "exp3",
        "description": "BERTurk[CLS](768) ++ Sözlük (NRC kategorili)(10) -> Linear(778,10)",
    },
    {
        "id":          4,
        "name":        "Master Hybrid",
        "script":      ROOT / "train_exp4_master.py",
        "result_json": ROOT / "results" / "exp4" / "metrics_report.json",
        "results_dir": ROOT / "results" / "exp4",
        "description": "BERT + Lexikon + Cross-Lingual (tum sistemler birlesik)",
    },
]


# --- Yardımcı Fonksiyonlar --------------------------------------------------

def fmt_duration(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds, 3600)
    m, s   = divmod(rem, 60)
    if td.days > 0:
        return f"{td.days}g {h:02d}:{m:02d}:{s:02d}"
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def read_f1_macro(json_path: Path) -> Optional[float]:
    try:
        with open(json_path, encoding="utf-8") as f:
            return json.load(f).get("test_f1_macro")
    except Exception:
        return None


def print_banner(text: str, width: int = 65):
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_status_table(experiments, completed_times: dict):
    """Tüm deneylerin anlık durumunu göster."""
    print("\n" + "-" * 72)
    print(f"  {'#':>2}  {'Ad':22s}  {'Durum':12s}  {'F1-Macro':>9}  {'Süre':>8}")
    print("-" * 72)
    for exp in experiments:
        eid   = exp["id"]
        name  = exp["name"]
        jpath = exp["result_json"]

        if jpath.exists():
            f1   = read_f1_macro(jpath)
            f1s  = f"{f1:.4f}" if f1 is not None else "  N/A"
            dur  = fmt_duration(completed_times.get(eid, 0))
            stat = "TAMAMLANDI"
        else:
            f1s  = "  ---"
            dur  = "  ---"
            stat = "bekliyor"

        print(f"  {eid:>2}  {name:22s}  {stat:12s}  {f1s:>9}  {dur:>8}")
    print("-" * 72 + "\n")


# --- Ana Çalışma Döngüsü ----------------------------------------------------

def run_all(only_ids=None, force=False, dry_run=False, extra_args=None):
    start_global = time.time()
    completed_times = {}
    results = {}

    print_banner("ABLATION STUDY ORCHESTRATOR — 4 DENEY")
    print(f"  Baslangic : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Force     : {force}")
    print(f"  Dry-Run   : {dry_run}")
    if only_ids:
        print(f"  Secilen   : Exp {only_ids}")

    # Ilk durum tablosu
    print_status_table(EXPERIMENTS, completed_times)

    for exp in EXPERIMENTS:
        eid  = exp["id"]
        name = exp["name"]

        # Filtre
        if only_ids and eid not in only_ids:
            print(f"[ATLA] Exp{eid} ({name}) — secilmedi")
            continue

        # Force: mevcut sonuclari sil
        if force and exp["result_json"].exists():
            print(f"[FORCE] Exp{eid} sonuclari siliniyor: {exp['results_dir']}")
            if not dry_run:
                for f in exp["results_dir"].glob("*.json"):
                    f.unlink(missing_ok=True)
                for f in exp["results_dir"].glob("*.png"):
                    f.unlink(missing_ok=True)
                best = exp["results_dir"] / "best_model"
                if best.exists():
                    shutil.rmtree(best, ignore_errors=True)

        # Zaten tamamlandı mı?
        if exp["result_json"].exists() and not force:
            f1 = read_f1_macro(exp["result_json"])
            _f1s = f"{f1:.4f}" if f1 is not None else "N/A"
            print(f"[ATLA] Exp{eid} zaten tamamlandi (F1={_f1s})  -> {exp['result_json']}")
            results[eid] = {"status": "skipped", "f1_macro": f1}
            continue

        # Script var mı?
        if not exp["script"].exists():
            print(f"[HATA] Script bulunamadi: {exp['script']}")
            results[eid] = {"status": "error", "reason": "script not found"}
            continue

        print_banner(f"EXP{eid} BASLIYOR — {name}")
        print(f"  Script     : {exp['script'].name}")
        print(f"  Aciklama   : {exp['description']}")
        print(f"  Cikti      : {exp['results_dir']}")

        if dry_run:
            print(f"  [DRY-RUN] Calistirılmadi.")
            results[eid] = {"status": "dry-run"}
            continue

        # Results dizini hazırla
        exp["results_dir"].mkdir(parents=True, exist_ok=True)

        # Süreci başlat
        cmd = [sys.executable, str(exp["script"])]
        if extra_args:
            cmd += extra_args

        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                check=True,
                cwd=str(ROOT),
                env=dict(os.environ, PYTHONIOENCODING="utf-8"),
            )
            elapsed = time.time() - t0
            completed_times[eid] = elapsed

            f1 = read_f1_macro(exp["result_json"])
            print_banner(f"EXP{eid} TAMAMLANDI — {fmt_duration(elapsed)}")
            _f1s = f"{f1:.4f}" if f1 is not None else "N/A"
            print(f"  F1-Macro : {_f1s}")
            results[eid] = {"status": "success", "duration_s": elapsed, "f1_macro": f1}

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - t0
            print_banner(f"EXP{eid} BASARISIZ — {fmt_duration(elapsed)}")
            print(f"  Return code : {e.returncode}")
            results[eid] = {"status": "failed", "return_code": e.returncode, "duration_s": elapsed}

        # Güncel durum tablosu
        print_status_table(EXPERIMENTS, completed_times)

    # -- Final Özet ------------------------------------------------
    total_elapsed = time.time() - start_global
    print_banner(f"TUM DENEYLER TAMAMLANDI — Toplam: {fmt_duration(total_elapsed)}")

    ok      = [eid for eid, r in results.items() if r.get("status") == "success"]
    skipped = [eid for eid, r in results.items() if r.get("status") == "skipped"]
    failed  = [eid for eid, r in results.items() if r.get("status") == "failed"]

    print(f"  Basarili  : Exp {ok}") if ok else None
    print(f"  Atlanan   : Exp {skipped}") if skipped else None
    print(f"  Basarisiz : Exp {failed}") if failed else None

    # Tum tamamlananların F1 sıralaması
    all_f1 = []
    for exp in EXPERIMENTS:
        f1 = read_f1_macro(exp["result_json"])
        if f1 is not None:
            all_f1.append((exp["id"], exp["name"], f1))

    if all_f1:
        all_f1.sort(key=lambda x: -x[2])
        print("\n  F1-Macro Siralamasi (yuksekten dusuge):")
        for rank, (eid, name, f1) in enumerate(all_f1, 1):
            medal = ["🥇","🥈","🥉","  "][min(rank-1, 3)]
            print(f"    {rank}. Exp{eid} {name:25s}  F1={f1:.4f}")

    # Toplam özeti JSON'a kaydet
    summary_path = ROOT / "results" / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp":    datetime.now().isoformat(),
            "total_seconds": total_elapsed,
            "results":      {str(k): v for k, v in results.items()},
            "f1_ranking":   [{"exp": e, "name": n, "f1": f} for e, n, f in all_f1],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Ozet kaydedildi: {summary_path}")

    # Rapor oluşturmayı öner
    reporter = ROOT / "final_comparison_reporter.py"
    if reporter.exists():
        print(f"\n  Karsilastirma raporu icin:")
        print(f"    python final_comparison_reporter.py")

    return results


# --- CLI ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation Study Orchestrator — 4 Deney",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  python run_experiments.py                    # tumunu calistir
  python run_experiments.py --only 1 2         # sadece Exp1 ve Exp2
  python run_experiments.py --force --only 4   # Exp4'u zorla yeniden calistir
  python run_experiments.py --dry-run          # neyin calisacagini goster
        """,
    )
    parser.add_argument(
        "--only", type=int, nargs="+", choices=[1,2,3,4],
        metavar="N", help="Sadece belirtilen deney(ler)i calistir",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Mevcut sonuclari yoksay, tum deneyleri yeniden calistir",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Plani goster, hicbir sey calistirma",
    )
    parser.add_argument(
        "--rebuild-data", action="store_true",
        help="Her deney scriptine --rebuild-data ilet",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Epoch sayisi (orn: 2 = hizli egitim)",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=None,
        help="Sinif basi max ornek (orn: 10000 = hizli egitim)",
    )
    args = parser.parse_args()

    extra = []
    if args.rebuild_data:
        extra += ["--rebuild-data"]
    if args.epochs:
        extra += ["--epochs", str(args.epochs)]
    if args.max_per_class:
        extra += ["--max-per-class", str(args.max_per_class)]

    run_all(
        only_ids=args.only,
        force=args.force,
        dry_run=args.dry_run,
        extra_args=extra if extra else None,
    )
