"""
Ana Pipeline Yürütücüsü
Sırasıyla: veri hazırlama → eğitim → değerlendirme → demo
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def step_prepare():
    print("\n" + "="*60)
    print("ADIM 1: Veri Seti Hazırlama")
    print("="*60)
    from src.data.download_dataset import prepare_dataset
    prepare_dataset()


def step_train():
    print("\n" + "="*60)
    print("ADIM 2: BERTurk Fine-Tuning")
    print("="*60)
    from src.training.train import train
    train()


def step_demo():
    print("\n" + "="*60)
    print("ADIM 3: Gradio Demo")
    print("="*60)
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "demo" / "app.py")])


def main():
    parser = argparse.ArgumentParser(description="Türkçe Duygu Analizi Pipeline")
    parser.add_argument(
        "--step",
        choices=["prepare", "train", "demo", "all"],
        default="all",
        help="Çalıştırılacak adım (varsayılan: all)",
    )
    args = parser.parse_args()

    if args.step in ("prepare", "all"):
        step_prepare()

    if args.step in ("train", "all"):
        step_train()

    if args.step in ("demo", "all"):
        step_demo()


if __name__ == "__main__":
    main()
