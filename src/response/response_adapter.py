"""
Duygu Farkındalıklı Yanıt Adaptasyon Modülü
Tespit edilen duyguya göre yanıt tonu, empati düzeyi ve
içerik stratejisini dinamik olarak ayarlar.
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from src.inference.emotion_classifier import EmotionResult


# ─── Veri Yapıları ───────────────────────────────────────────

@dataclass
class AdaptedResponse:
    """Duyguya göre uyarlanmış yanıt."""
    original_text: str
    detected_emotion: str
    confidence: float
    tone: str
    response_text: str
    suggestions: list[str]
    emoji: str

    def __repr__(self) -> str:
        sugg = "\n    ".join(f"• {s}" for s in self.suggestions)
        return (
            f"\n{'='*60}\n"
            f"  Tespit Edilen Duygu : {self.emoji} {self.detected_emotion.upper()} "
            f"({self.confidence:.1%})\n"
            f"  Yanıt Tonu          : {self.tone}\n"
            f"{'─'*60}\n"
            f"  Yanıt:\n  {self.response_text}\n"
            f"{'─'*60}\n"
            f"  Öneriler:\n    {sugg}\n"
            f"{'='*60}\n"
        )


# ─── Duygu Profilleri ─────────────────────────────────────────
# Her duygu için ton, empati stratejisi ve örnek yanıt şablonları.

EMOTION_PROFILES: dict[str, dict] = {
    "mutluluk": {
        "tone": "Coşkulu & Destekleyici",
        "emoji": "😊",
        "empathy_level": "yüksek_pozitif",
        "response_templates": [
            "Bu harika! {text} duyunca ben de çok mutlu oldum.",
            "Ne güzel bir haber! Sevincinizi paylaşmak harika.",
            "Müthiş! Bu mutluluğun tadını çıkarın.",
        ],
        "tone_instructions": (
            "Enerjik ve coşkulu bir ton kullan. "
            "Kullanıcının sevincini pekiştir ve olumlu duyguyu yansıt. "
            "Kısa, dinamik cümleler tercih et."
        ),
        "suggestions": [
            "Bu mutlu anı bir günlüğe yazın — güzel anlar kaybolup gidiyor.",
            "Bu sevinci sevdiklerinizle paylaşın.",
            "Kendinizi ödüllendirmeyi unutmayın!",
        ],
    },

    "üzüntü": {
        "tone": "Empatik & Sakinleştirici",
        "emoji": "💙",
        "empathy_level": "derin_empati",
        "response_templates": [
            "Sizi duyuyorum. Bu dönemden geçmek gerçekten zor.",
            "Üzüldüğünüzü hissedebiliyorum. Yanınızdayım.",
            "Bu duyguyla baş başa kalmak ağır olabilir. Kendinize karşı nazik olun.",
        ],
        "tone_instructions": (
            "Sakin, sıcak ve anlayışlı bir ton kullan. "
            "Yargılamadan dinle. Aceleye getirme, çözüm dayatma. "
            "Kısa ve içten cümleler yaz."
        ),
        "suggestions": [
            "Duygularınızı bir yakınınızla paylaşmayı deneyin.",
            "Küçük adımlar atmak yeterli — bugün sadece bir şey yapın.",
            "Profesyonel destek almaktan çekinmeyin, bu bir güç işaretidir.",
        ],
    },

    "öfke": {
        "tone": "Sakin & Çözüm Odaklı",
        "emoji": "🔥",
        "empathy_level": "validasyon_ve_yönlendirme",
        "response_templates": [
            "Neden bu kadar sinirlendiğinizi anlıyorum. Bu his geçerli.",
            "Öfkeniz çok anlaşılır. Şimdi birlikte düşünelim.",
            "Bu durum gerçekten sinir bozucu. Derin bir nefes alalım.",
        ],
        "tone_instructions": (
            "Sakin ama kararlı bir ton kullan. "
            "Duyguyu yargılamadan onayla, ardından yavaşça çözüme yönlendir. "
            "Ateşi körükleyecek ifadelerden kaçın."
        ),
        "suggestions": [
            "Tepki vermeden önce 10 derin nefes almayı deneyin.",
            "Öfkenizin kaynağını yazıya dökmek mesafelenmenizi sağlar.",
            "Fiziksel aktivite (yürüyüş, spor) öfkeyi hızla düşürür.",
        ],
    },

    "korku": {
        "tone": "Güven Verici & Yapılandırıcı",
        "emoji": "🛡️",
        "empathy_level": "güven_inşası",
        "response_templates": [
            "Korkularınız çok gerçek ve anlaşılır. Yalnız değilsiniz.",
            "Bu his geçici. Birlikte adım adım ilerleyebiliriz.",
            "Korku bazen bizi korur. Onu dinleyelim ama kontrol ettirmeyelim.",
        ],
        "tone_instructions": (
            "Sakin, güven verici ve net bir ton kullan. "
            "Somut, küçük adımlar öner. Belirsizliği azalt. "
            "Abartılı iyimserlikten kaçın — gerçekçi ol."
        ),
        "suggestions": [
            "Korkunuzu 5–4–3–2–1 duyusal farkındalık tekniğiyle yönetin.",
            "En kötü senaryoyu yazın, ardından gerçekçi senaryoyu yazın.",
            "Güvendiğiniz biriyle bu endişeleri paylaşın.",
        ],
    },

    "tiksinti": {
        "tone": "Anlayışlı & Sınır Koyan",
        "emoji": "😖",
        "empathy_level": "sinir_yönetimi",
        "response_templates": [
            "Bu kadar rahatsız edici bulmak çok anlaşılır.",
            "Sınırlarınızı koruma hakkınız var. Bu his sizi dinliyor.",
            "Bu durumdan uzak durmak istemeniz çok doğal.",
        ],
        "tone_instructions": (
            "Anlayışlı ama pratik bir ton kullan. "
            "Sınır koyma ve uzaklaşmayı normalize et. "
            "Aşırı tepkileri yatıştırmadan onayla."
        ),
        "suggestions": [
            "Bu durumdan/kişiden mesafe koymayı değerlendirin.",
            "Sınır koymanız gerekiyorsa bunu açık ve sakin bir şekilde ifade edin.",
            "Tetikleyiciyi not alın — gelecekte önlem alabilirsiniz.",
        ],
    },

    "şaşkınlık": {
        "tone": "Meraklı & Keşifsel",
        "emoji": "✨",
        "empathy_level": "merak_yönetimi",
        "response_templates": [
            "Vay! Bu gerçekten beklenmedik bir gelişme.",
            "Şaşkınlık yeni bir şeyin kapısını aralamış olabilir.",
            "Bu sürpriz sizi nereye götürüyor?",
        ],
        "tone_instructions": (
            "Enerjik ve meraklı bir ton kullan. "
            "Şaşkınlığı olumlu bir keşif fırsatı olarak çerçevele. "
            "Açık uçlu sorularla kullanıcıyı düşünmeye teşvik et."
        ),
        "suggestions": [
            "İlk tepkinizi not edin — sürprizler size kendiniz hakkında çok şey söyler.",
            "Bu beklenmedik durumu fırsata dönüştürebilir misiniz?",
            "Zaman ayırın — ani kararlardan kaçının.",
        ],
    },

    "sevgi": {
        "tone": "Sıcak & Bağlantı Kurucu",
        "emoji": "❤️",
        "empathy_level": "yüksek_pozitif",
        "response_templates": [
            "Ne güzel bir his! Sevgi, hayatımızı anlamlı kılan en değerli duygudur.",
            "Bu bağı hissedebiliyorum. Sevdiğiniz kişiyi ne kadar şanslısınız!",
            "Sevginin gücü inanılmaz. Bu duyguyu yaşıyor olmanız çok kıymetli.",
        ],
        "tone_instructions": (
            "Sıcak, samimi ve bağlantı kurucu bir ton kullan. "
            "Sevginin olumlu etkisini pekiştir. "
            "Duyguyu kutla ve destekle."
        ),
        "suggestions": [
            "Sevdiğiniz kişiye bunu açıkça ifade etmekten çekinmeyin.",
            "Bu duyguyu beslemek için zaman ve ilgi ayırın.",
            "Sevginin karşılıklı büyüdüğünü unutmayın — paylaştıkça çoğalır.",
        ],
    },

    "nötr": {
        "tone": "Nötr & Bilgilendirici",
        "emoji": "😐",
        "empathy_level": "tarafsız",
        "response_templates": [
            "Anlıyorum. Daha fazla bilgi vermek ister misiniz?",
            "Size nasıl yardımcı olabileceğimi anlatabilirsiniz.",
            "Durumu değerlendiriyorum. Devam edebilirsiniz.",
        ],
        "tone_instructions": (
            "Tarafsız, saygılı ve dinleyici bir ton kullan. "
            "Yargılamadan bil, yönlendirmeden dinle. "
            "Kullanıcının konuşmayı yönlendirmesine izin ver."
        ),
        "suggestions": [
            "Düşüncelerinizi netleştirmek için biraz zaman ayırabilirsiniz.",
            "Ne hissettiğinizi daha açık ifade etmek ister misiniz?",
            "İhtiyacınız olursa buradayım.",
        ],
    },

    "utanç": {
        "tone": "Anlayışlı & Normalleştirici",
        "emoji": "🙈",
        "empathy_level": "derin_empati",
        "response_templates": [
            "Bu his çok insani. Herkes zaman zaman utanç duyar.",
            "Kendinize karşı nazik olun — bu duygu sizi tanımlamaz.",
            "Utanç, değer verdiğinizin bir işareti. Kendinizi affetmeyi deneyin.",
        ],
        "tone_instructions": (
            "Nazik, yargılamayan ve normalleştirici bir ton kullan. "
            "Utancı küçümseme ama yükünü hafiflet. "
            "Öz-şefkati teşvik et."
        ),
        "suggestions": [
            "Kendinize, iyi bir arkadaşınıza söyler gibi konuşun.",
            "Bu duygunun geçici olduğunu hatırlayın — tanımlamaz, bilgilendirir.",
            "Güvendiğiniz biriyle paylaşmak yükü hafifletebilir.",
        ],
    },

    "gurur": {
        "tone": "Kutlayıcı & Onaylayıcı",
        "emoji": "🏆",
        "empathy_level": "yüksek_pozitif",
        "response_templates": [
            "Bravo! Bu başarının tadını çıkarın — hak ettiniz.",
            "Gurur duymak ne güzel. Bu anı içinizde saklayın.",
            "Harika! Emekleriniz karşılığını bulmuş.",
        ],
        "tone_instructions": (
            "Coşkulu ve onaylayıcı bir ton kullan. "
            "Başarıyı somutlaştır ve kutla. "
            "Kullanıcının öz-değerini pekiştir."
        ),
        "suggestions": [
            "Bu başarıyı sevdiklerinizle paylaşın.",
            "Buraya nasıl geldiğinizi not edin — zorlu günlerde ilham verir.",
            "Kendinizi ödüllendirin. Bu anı hak ettiniz!",
        ],
    },
}


# ─── Adaptör Sınıfı ──────────────────────────────────────────

class ResponseAdapter:
    """
    EmotionResult alır, duyguya özel yanıt tonu ve içerik üretir.

    Kullanım:
        adapter = ResponseAdapter()
        adapted = adapter.adapt(emotion_result)
        print(adapted)
    """

    def __init__(self, confidence_threshold: float = 0.4):
        """
        confidence_threshold: Bu değerin altındaki tahminler için
        belirsiz duygu yolu kullanılır.
        """
        self.confidence_threshold = confidence_threshold

    def adapt(self, result: EmotionResult) -> AdaptedResponse:
        """EmotionResult'u alır, duyguya uyarlanmış yanıt üretir."""
        emotion = result.emotion
        confidence = result.confidence

        # Düşük güven durumunda daha nötr yaklaş
        if confidence < self.confidence_threshold:
            return self._low_confidence_response(result)

        # Tanımsız bir etiket gelirse coşkulu "mutluluk" yerine tarafsız profil kullanılır
        profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["nötr"])
        template = random.choice(profile["response_templates"])
        response_text = template.format(text=result.text[:40])

        return AdaptedResponse(
            original_text=result.text,
            detected_emotion=emotion,
            confidence=confidence,
            tone=profile["tone"],
            response_text=response_text,
            suggestions=profile["suggestions"],
            emoji=profile["emoji"],
        )

    def adapt_with_context(
        self,
        result: EmotionResult,
        user_message: str | None = None,
    ) -> AdaptedResponse:
        """
        Kullanıcı mesajını bağlam olarak kullanarak daha kişiselleştirilmiş yanıt üretir.
        LLM entegrasyonu için genişletilebilir zemin.
        """
        adapted = self.adapt(result)

        # Bağlamsal iyileştirme: şiddetli duygular için ek not
        if result.confidence > 0.85 and result.emotion in ("öfke", "korku", "üzüntü"):
            adapted.response_text += (
                " Eğer bu his sürekli devam ediyorsa, "
                "bir uzmana danışmak faydalı olabilir."
            )

        return adapted

    def get_tone_instructions(self, emotion: str) -> str:
        """LLM prompt'larında kullanılmak üzere ton talimatlarını döndürür."""
        profile = EMOTION_PROFILES.get(emotion, {})
        return profile.get("tone_instructions", "Nötr ve saygılı bir ton kullan.")

    def _low_confidence_response(self, result: EmotionResult) -> AdaptedResponse:
        top2 = sorted(result.probabilities.items(), key=lambda x: -x[1])[:2]
        emotions_str = " ve ".join(f"{e} ({p:.1%})" for e, p in top2)

        return AdaptedResponse(
            original_text=result.text,
            detected_emotion="belirsiz",
            confidence=result.confidence,
            tone="Nötr & Dikkatli",
            response_text=(
                f"Duygunuzu tam anlayabilmek için biraz daha bilgiye ihtiyacım var. "
                f"({emotions_str} hissedildi, ancak emin olamadım.)"
            ),
            suggestions=[
                "Hissettiklerinizi biraz daha açıklamak ister misiniz?",
                "Şu an en çok ne hissediyorsunuz?",
            ],
            emoji="🤔",
        )


# ─── Hızlı Test ──────────────────────────────────────────────

if __name__ == "__main__":
    from src.inference.emotion_classifier import EmotionResult

    adapter = ResponseAdapter()

    test_cases = [
        EmotionResult("Bugün harika bir gün!", "mutluluk", 0.92,
                      {"mutluluk": 0.92, "şaşkınlık": 0.05}, 0),
        EmotionResult("Çok üzgünüm, bir şey yapamıyorum.", "üzüntü", 0.87,
                      {"üzüntü": 0.87, "korku": 0.08}, 1),
        EmotionResult("Bu kadar sinir bozucu!", "öfke", 0.78,
                      {"öfke": 0.78, "tiksinti": 0.15}, 2),
    ]

    for er in test_cases:
        adapted = adapter.adapt(er)
        print(adapted)
