"""
Duygu Sınıflandırma Modülü
Fine-tune edilmiş BERTurk modeliyle Türkçe metinden duygu tespiti yapar.
Batch inference ve güven skoru döndürme desteği içerir.
"""

from __future__ import annotations
import json
import torch
import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── Veri Yapıları ───────────────────────────────────────────

@dataclass
class EmotionResult:
    """Tek bir metin için duygu analizi sonucu."""
    text: str
    emotion: str                          # En yüksek olasılıklı duygu
    confidence: float                     # [0, 1] arası güven skoru
    probabilities: dict[str, float] = field(default_factory=dict)  # Tüm sınıf olasılıkları
    label_id: int = -1

    def __repr__(self) -> str:
        prob_str = " | ".join(
            f"{k}: {v:.3f}" for k, v in
            sorted(self.probabilities.items(), key=lambda x: -x[1])
        )
        return (
            f"EmotionResult(\n"
            f"  metin      : {self.text[:60]}{'...' if len(self.text) > 60 else ''}\n"
            f"  duygu      : {self.emotion} ({self.confidence:.1%})\n"
            f"  olasılıklar: {prob_str}\n"
            f")"
        )


# ─── Sınıflandırıcı ──────────────────────────────────────────

class EmotionClassifier:
    """
    BERTurk tabanlı Türkçe duygu sınıflandırıcı.

    Kullanım:
        clf = EmotionClassifier()
        result = clf.predict("Bugün çok mutluyum!")
        results = clf.predict_batch(["Harika!", "Berbat bir gün."])
    """

    def __init__(self, model_path: str | Path | None = None, device: str | None = None):
        config = load_config()
        self.config = config
        self.id2label = {int(k): v for k, v in config["emotions"]["id2label"].items()}
        self.label2id = config["emotions"]["label2id"]
        self.num_labels = config["emotions"]["num_labels"]
        self.max_length = config["model"]["max_length"]

        # Cihaz seçimi
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Model yolu: verilmezse config.yaml → inference.model_dir, sonra models/final
        # (büyük/küçük harf fark etmeksizin), bulunamazsa base model (classifier eğitilmemiş).
        self.is_finetuned = True
        if model_path is None:
            cfg_dir = config.get("inference", {}).get("model_dir")
            if cfg_dir and (ROOT / cfg_dir / "config.json").exists():
                model_path = ROOT / cfg_dir
            else:
                model_path = self._find_final_model_dir()
            if model_path is None:
                print(f"[WARN] Fine-tune model bulunamadı. Base model yükleniyor: {config['model']['base_model']}")
                print("[WARN] Base model'in sınıflandırma katmanı rastgele — tahminler anlamsızdır.")
                model_path = config["model"]["base_model"]
                self.is_finetuned = False

        expected_labels = self.num_labels
        self._load_model(str(model_path))
        if self.num_labels != expected_labels:
            print(
                f"[WARN] Yüklenen model {self.num_labels} sınıflı "
                f"({list(self.id2label.values())}), config.yaml ise {expected_labels} sınıf bekliyor. "
                "Yanlış model yüklenmiş olabilir."
            )
        print(f"[OK] EmotionClassifier hazır | cihaz: {self.device} | model: {model_path}")

    @staticmethod
    def _find_final_model_dir() -> Path | None:
        """models/ altında adı 'final' olan (harf büyüklüğünden bağımsız) dolu klasörü bulur."""
        models_dir = ROOT / "models"
        if not models_dir.is_dir():
            return None
        for child in models_dir.iterdir():
            if child.is_dir() and child.name.lower() == "final" and (child / "config.json").exists():
                return child
        return None

    def _load_model(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # num_labels'ı dışarıdan geçirmiyoruz; kaydedilmiş model config'ini kullan.
        # Bu sayede config.yaml ile model ağırlıkları arasındaki sınıf sayısı
        # uyuşmazlığından kaynaklanan load_state_dict hatası ortadan kalkar.
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        # Çalışma zamanı etiket haritasını yüklenen modelin kendi config'iyle eşitle
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.num_labels = self.model.config.num_labels
        self.model.to(self.device)
        self.model.eval()
        # Kalibrasyon: eğitimde val setinde öğrenilen sıcaklık (train_v2 → temperature.json).
        # Yoksa T=1 (kalibre edilmemiş; "güven skoru" gerçek doğruluğu yansıtmayabilir).
        t_file = Path(model_path) / "temperature.json"
        self.temperature = json.loads(t_file.read_text())["temperature"] if t_file.exists() else 1.0
        self.is_calibrated = t_file.exists()

    # ── Tek Tahmin ────────────────────────────────────────────

    def predict(self, text: str) -> EmotionResult:
        """Tek bir Türkçe metin için duygu tahmini yapar."""
        results = self.predict_batch([text])
        return results[0]

    # ── Batch Tahmin ──────────────────────────────────────────

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[EmotionResult]:
        """
        Birden fazla metin için batch inference.
        Büyük listeler otomatik olarak mini-batch'lere bölünür.
        """
        all_results: list[EmotionResult] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_results = self._infer_batch(batch_texts)
            all_results.extend(batch_results)

        return all_results

    def _infer_batch(self, texts: list[str]) -> list[EmotionResult]:
        encoding = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)

        probs = torch.softmax(outputs.logits.float() / self.temperature, dim=-1).cpu().numpy()  # (batch, num_labels)
        pred_ids = np.argmax(probs, axis=-1)

        results = []
        for text, prob_vec, pred_id in zip(texts, probs, pred_ids):
            prob_dict = {self.id2label[j]: float(prob_vec[j]) for j in range(self.num_labels)}
            results.append(EmotionResult(
                text=text,
                emotion=self.id2label[int(pred_id)],
                confidence=float(prob_vec[pred_id]),
                probabilities=prob_dict,
                label_id=int(pred_id),
            ))
        return results

    # ── Top-K Tahmin ──────────────────────────────────────────

    def predict_topk(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        """En yüksek k duygu ve olasılıklarını döndürür."""
        result = self.predict(text)
        sorted_probs = sorted(result.probabilities.items(), key=lambda x: -x[1])
        return sorted_probs[:k]

    # ── GPU Bilgisi ───────────────────────────────────────────

    def device_info(self) -> str:
        if self.device == "cuda":
            name = torch.cuda.get_device_name(0)
            mem_alloc = torch.cuda.memory_allocated(0) / 1e9
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"GPU: {name} | VRAM: {mem_alloc:.1f}/{mem_total:.1f} GB"
        return "CPU modu"


# ─── Hızlı Test ──────────────────────────────────────────────

if __name__ == "__main__":
    clf = EmotionClassifier()
    print(clf.device_info())

    test_texts = [
        "Bugün inanılmaz güzel bir gün, çok mutluyum!",
        "Bu haberi duyunca içim sıkıştı, çok üzüldüm.",
        "Sana çok kızgınım, bu davranışın kabul edilemez!",
        "Karanlık odada yalnız kalmaktan çok korkuyorum.",
        "Bu yemek gerçekten iğrenç, tadına bile bakamadım.",
        "Bunu hiç beklemiyordum, tamamen şaşırdım!",
    ]

    for text in test_texts:
        result = clf.predict(text)
        print(result)
        print()
