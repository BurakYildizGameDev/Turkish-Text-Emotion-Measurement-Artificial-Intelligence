"""
Streamlit Demo — Türkçe Duygu Analizi + Duygu Farkındalıklı Yanıt Sistemi
"""

import asyncio
import sys
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Python 3.11+ Windows'ta ProactorEventLoop varsayılan oldu; Streamlit shutdown
# sırasında "Event loop is closed" hatasına yol açar. Selector policy düzeltir.
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import pandas as pd
import streamlit as st
from src.inference.emotion_classifier import EmotionClassifier
from src.response.response_adapter import ResponseAdapter, EMOTION_PROFILES

# ── Sayfa Ayarı ───────────────────────────────────────────────
st.set_page_config(
    page_title="Türkçe Duygu Analizi",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource(show_spinner="Model yükleniyor...")
def load_models():
    return EmotionClassifier(), ResponseAdapter()


clf, adapter = load_models()

EXAMPLES = [
    "Bugün iş görüşmesini kazandım, inanılmaz mutluyum!",
    "Annem hasta, çok endişeleniyorum ve üzgünüm.",
    "Bu kadar saçmalık görmedim, gerçekten çok sinir bozucu!",
    "Karanlık odada yalnız kalmak beni korkutuyor.",
    "Bu yemek gerçekten iğrenç, yiyemiyorum bile.",
    "Bunu hiç beklemiyordum, tamamen şaşırdım!",
]

# ── Başlık ────────────────────────────────────────────────────
st.title("🧠 Türkçe Duygu Analizi & Farkındalıklı Yanıt Sistemi")
_label_display = " · ".join(
    f"{EMOTION_PROFILES[e]['emoji']} {e.capitalize()}"
    if e in EMOTION_PROFILES
    else e.capitalize()
    for e in clf.id2label.values()
)
st.caption(
    f"**BERTurk** (bert-base-turkish-cased) ile {clf.num_labels} duygu sınıfı: {_label_display}"
)
if not clf.is_finetuned:
    st.error(
        "Fine-tune edilmiş model bulunamadı; eğitilmemiş base model yüklendi. "
        "Tahminler rastgeledir."
    )
elif clf.num_labels != clf.config["emotions"]["num_labels"]:
    st.warning(
        f"Yüklenen model {clf.num_labels} sınıflı, config.yaml "
        f"{clf.config['emotions']['num_labels']} sınıf bekliyor. Yanlış model yüklenmiş olabilir."
    )

tab1, tab2, tab3 = st.tabs(["Tekli Analiz", "Toplu Analiz", "Hakkında"])


# ── Tekli Analiz ──────────────────────────────────────────────
with tab1:
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        text_input = st.text_area(
            "Türkçe Metin",
            value=st.session_state.input_text,
            placeholder="Hissettiklerinizi buraya yazın...",
            height=130,
        )

        analyze_btn = st.button("Analiz Et", type="primary", use_container_width=True)

        st.markdown("**Örnek metinler:**")
        for i, example in enumerate(EXAMPLES):
            label = example[:55] + ("…" if len(example) > 55 else "")
            if st.button(label, key=f"ex_{i}", use_container_width=True):
                st.session_state.input_text = example
                st.rerun()

    with col_right:
        if analyze_btn:
            if not text_input.strip():
                st.warning("Lütfen bir metin girin.")
            else:
                with st.spinner("Analiz yapılıyor..."):
                    result = clf.predict(text_input)
                    adapted = adapter.adapt_with_context(result)

                # Duygu başlığı + metrikler
                st.markdown(f"### {adapted.emoji} {adapted.detected_emotion.upper()}")
                m1, m2 = st.columns(2)
                m1.metric("Güven Skoru", f"{adapted.confidence:.1%}")
                m2.metric("Ton", adapted.tone)

                st.markdown("**Uyarlanmış Yanıt**")
                st.info(adapted.response_text)

                # Olasılık grafiği
                st.markdown("**Duygu Olasılıkları**")
                prob_data = {
                    f"{EMOTION_PROFILES.get(k, {}).get('emoji', '•')} {k}": float(v)
                    for k, v in sorted(
                        result.probabilities.items(), key=lambda x: -x[1]
                    )
                }
                df = pd.DataFrame.from_dict(
                    prob_data, orient="index", columns=["Olasılık"]
                )
                st.bar_chart(df, horizontal=True)

                # Öneriler
                st.markdown("**Öneriler**")
                for s in adapted.suggestions:
                    st.markdown(f"• {s}")

                with st.expander("Ton Talimatları (LLM Entegrasyonu için)"):
                    st.code(adapter.get_tone_instructions(result.emotion))


# ── Toplu Analiz ──────────────────────────────────────────────
with tab2:
    st.markdown("Her satıra bir metin girin.")
    batch_input = st.text_area(
        "Metinler",
        placeholder="Bugün çok mutluyum!\nBu durum beni korkutuyor.\nİnanılmaz bir haber!",
        height=200,
        key="batch_text",
    )
    batch_btn = st.button("Toplu Analiz Et", type="primary", key="batch_btn")

    if batch_btn:
        lines = [ln.strip() for ln in batch_input.strip().splitlines() if ln.strip()]
        if not lines:
            st.warning("Metin giriniz.")
        else:
            with st.spinner(f"{len(lines)} metin analiz ediliyor..."):
                results = clf.predict_batch(lines)

            for r in results:
                adapted = adapter.adapt(r)
                st.markdown(
                    f"**{adapted.emoji} {r.emotion.upper()}** — {r.confidence:.0%} güven  \n"
                    f"_{r.text[:80]}{'...' if len(r.text) > 80 else ''}_  \n"
                    f"→ {adapted.response_text}"
                )
                st.divider()


# ── Hakkında ──────────────────────────────────────────────────
with tab3:
    st.markdown(
        f"""
## Sistem Bilgisi

| Alan | Değer |
|------|-------|
| **Model** | dbmdz/bert-base-turkish-cased ({"fine-tuned" if clf.is_finetuned else "eğitilmemiş base model"}) |
| **Veri (v2)** | Gold: TREMO, Türkçe tweet duygu, GoEmotions TR (çeviri) · Weak: ürün/film yorumlarından kural tabanlı etiket — bkz. data/v2/README.md |
| **Güven skoru** | {"kalibre (temperature scaling, T=" + format(clf.temperature, ".2f") + ")" if clf.is_calibrated else "kalibre edilmemiş"} |
| **Duygu Sınıfları** | {", ".join(clf.id2label.values())} |
| **Donanım** | RTX 5070 Ti + i9 |

## Mimari

```
Kullanıcı Metni
     ↓
BERTurk Tokenizer
     ↓
BERTurk Fine-Tuned ({clf.num_labels}-sınıf)
     ↓
EmotionResult (duygu + güven skoru + olasılıklar)
     ↓
ResponseAdapter (ton + empati + öneriler)
     ↓
Uyarlanmış Yanıt
```
"""
    )
