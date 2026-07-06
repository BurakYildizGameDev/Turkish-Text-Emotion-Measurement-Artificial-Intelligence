"""
Türkçe Duygu Analizi — Streamlit Demo
======================================
Master Hybrid Model (Exp4): BERTurk [CLS](768) ++ Sözlük (NRC kategorili)(10) → Linear(778,10)

ESKİ (v1): Bu demo 10 sınıflı v1 sözlük-hibrit modelini (results/exp4/best_model) bekler; o model
repoda yoktur ve v1 verisi Faz 3'te terk edilmiştir. v2 (8 sınıf) demosu: streamlit run demo/app.py

Başlatmak için:
    streamlit run app.py
"""

import sys
import json
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# SHAP yorumlanabilirlik araçları (isteğe bağlı)
try:
    from shap_interpretability import (
        compute_gradient_saliency,
        top_keywords_per_emotion,
        ModelLoader as _SHAPModelLoader,
    )
    _SHAP_MODULE_OK = True
except Exception:
    _SHAP_MODULE_OK = False

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ─── Konfigürasyon ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Türkçe Duygu Analizi | Master Hybrid",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sabitler ────────────────────────────────────────────────────────────────

EMOTIONS = [
    "mutluluk", "üzüntü", "öfke", "korku",
    "şaşkınlık", "tiksinti", "sevgi", "nötr", "utanç", "gurur",
]
NRC_CATS = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "negative", "positive", "sadness", "surprise", "trust",
]

EMOTION_META: Dict[str, dict] = {
    "mutluluk":  {"emoji": "😊", "color": "#f1c40f", "en": "Joy",
                  "desc": "Neşe, sevinç, başarı ve tatmin hisleri. Pozitif deneyimlerden kaynaklanan iyi hissetme hali."},
    "üzüntü":    {"emoji": "😢", "color": "#3498db", "en": "Sadness",
                  "desc": "Kayıp, hayal kırıklığı ve keder duyguları. Bir şeyin yolunda gitmediği hissi."},
    "öfke":      {"emoji": "😠", "color": "#e74c3c", "en": "Anger",
                  "desc": "Sinirlilik, engellenme ve adalet duygusu. Haksızlığa verilen güçlü tepki."},
    "korku":     {"emoji": "😨", "color": "#8e44ad", "en": "Fear",
                  "desc": "Tehlike, endişe ve kaygı duyguları. Tehdit algısına verilen koruyucu tepki."},
    "şaşkınlık": {"emoji": "😲", "color": "#1abc9c", "en": "Surprise",
                  "desc": "Beklenmedik olaylar karşısında duyulan hayret ve şok. Hızlı bilişsel yeniden değerlendirme."},
    "tiksinti":  {"emoji": "🤢", "color": "#27ae60", "en": "Disgust",
                  "desc": "Hoşnutsuzluk, iğrenme ve reddetme duygusu. Zararlı şeylerden kaçınma içgüdüsü."},
    "sevgi":     {"emoji": "❤️",  "color": "#e91e63", "en": "Love/Trust",
                  "desc": "Bağlılık, şefkat ve güven duyguları. Derin bir ilgi ve olumlu sosyal bağ hissi."},
    "nötr":      {"emoji": "😐", "color": "#95a5a6", "en": "Neutral",
                  "desc": "Belirgin bir duygusal yük içermeyen, tarafsız ve bilgilendirici ifadeler."},
    "utanç":     {"emoji": "😳", "color": "#e67e22", "en": "Shame",
                  "desc": "Mahcubiyet, utanma ve pişmanlık hisleri. Sosyal normları ihlal etmenin bilinci."},
    "gurur":     {"emoji": "🏆", "color": "#f39c12", "en": "Pride",
                  "desc": "Başarı, onur ve kendine saygı duyguları. Olumlu öz-değerlendirme ve sosyal statü."},
}

DEMO_TEXTS = [
    ("😊 Mutluluk",   "Bugün harika bir gün! Sınav sonuçlarım çıktı ve çok başarılıydım. Bunu kutlamak istiyorum!"),
    ("😢 Üzüntü",     "Artık dayanamıyorum. Her şey çok zor, kendimi çok yalnız ve değersiz hissediyorum."),
    ("😠 Öfke",       "Bu kadar saçmalığa tahammül edemiyorum! Tamamen haksız bir karar, kızgınlıktan ne yapacağımı bilemiyorum."),
    ("😨 Korku",      "Geç saatte yalnız yürürken arkamdan birinin geldiğini duydum. Kalp atışlarım hızlandı, dehşete düştüm."),
    ("😲 Şaşkınlık",  "İnanamıyorum! Hiç beklemiyordum bunu, gerçekten şoke oldum. Bu nasıl mümkün olabilir?"),
    ("🤢 Tiksinti",   "Bu yüzden her gün göbek deliğimi temizliyorum. Bu iğrenç. Onun YouTube'daki işlerini izliyorum. İğrenç!"),
    ("❤️ Sevgi",      "Seni çok seviyorum, hayatıma girdiğin için minnettarım. Seninle her şey daha güzel."),
    ("😐 Nötr",       "Türkiye'nin nüfusu 85 milyon civarındadır. Ülke 81 ile ayrılmıştır ve başkenti Ankara'dır."),
    ("😳 Utanç",      "Ne kadar utanç verici! Kimse onların hakkı olmadığını söylemedi. Hâlâ utanıyorum bu durumdan."),
    ("🏆 Gurur",      "Çünkü biz dahiyiz. Bu projeyi başarıyla tamamladık, takımımızla gurur duyuyorum!"),
]

# ─── CSS ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Genel */
.main { padding-top: 1rem; }
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

/* Başlık */
.app-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.app-subtitle {
    color: #7f8c8d;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}

/* Duygu kartı */
.emotion-card {
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    border-left: 5px solid;
    background: rgba(255,255,255,0.05);
}

/* Sistem bilgisi */
.sys-badge {
    display: inline-block;
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: white;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0.15rem;
}
.sys-badge-gpu {
    background: linear-gradient(135deg, #6a11cb, #2575fc);
}

/* Stat kutusu */
.stat-box {
    background: rgba(102,126,234,0.1);
    border: 1px solid rgba(102,126,234,0.3);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
    margin: 0.3rem 0;
}
.stat-value { font-size: 1.4rem; font-weight: 700; color: #667eea; }
.stat-label { font-size: 0.75rem; color: #95a5a6; margin-top: 0.1rem; }

/* Lexikon kelime etiketi */
.lex-tag {
    display: inline-block;
    border-radius: 6px;
    padding: 0.15rem 0.5rem;
    margin: 0.15rem;
    font-size: 0.82rem;
    font-weight: 500;
}

/* Demo örnek butonu */
.demo-header { font-weight: 600; color: #bdc3c7; font-size: 0.85rem; margin-bottom: 0.4rem; }

/* Büyük Sentiment Card */
.sentiment-card {
    border-radius: 18px;
    padding: 2rem 2.5rem;
    margin: 0.8rem 0 1.2rem 0;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.18);
}
.sentiment-card::before {
    content: '';
    position: absolute;
    top: -40%; right: -15%;
    width: 320px; height: 320px;
    border-radius: 50%;
    background: rgba(255,255,255,0.07);
    pointer-events: none;
}
.sc-emoji   { font-size: 4rem; line-height: 1; display: block; margin-bottom: 0.3rem; }
.sc-label   { font-size: 2.6rem; font-weight: 900; letter-spacing: 0.04em;
              color: white; text-shadow: 0 2px 8px rgba(0,0,0,0.25); }
.sc-sub     { font-size: 1rem; color: rgba(255,255,255,0.75); margin-top: 0.25rem; }
.sc-conf    { font-size: 1.4rem; font-weight: 700; color: white; margin-top: 0.7rem;
              display: inline-block; background: rgba(0,0,0,0.2);
              padding: 0.2rem 0.9rem; border-radius: 20px; }
.sc-ms      { font-size: 0.82rem; color: rgba(255,255,255,0.55); margin-left: 0.6rem; }

/* Hardware badge (fixed footer) */
.hw-footer {
    position: fixed; bottom: 12px; right: 14px; z-index: 999;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(102,126,234,0.35);
    border-radius: 10px;
    padding: 6px 12px;
    font-size: 0.7rem;
    color: #a0aec0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    line-height: 1.55;
    text-align: right;
}
.hw-footer strong { color: #667eea; }
</style>
""", unsafe_allow_html=True)


# ─── Model Yükleme ───────────────────────────────────────────────────────────

MODEL_SEARCH_PATHS = [
    ROOT / "results" / "exp4" / "best_model",
    ROOT / "results" / "exp3" / "best_model",
]
WEIGHT_NAMES = ["master_hybrid_weights.pt", "lexicon_bert_weights.pt"]


@st.cache_resource(show_spinner=False)
def load_model_and_tools() -> Tuple[Optional[object], Optional[object], Optional[object], str, bool]:
    """
    Model, tokenizer ve LexiconExtractor'ü yükler.
    Döndürür: (model, tokenizer, extractor, model_path_str, is_trained)
    """
    from transformers import AutoTokenizer
    from train_exp3_lexicon import (
        LexiconBERTModel, LexiconExtractor, LEXICON_DIM, load_lexicon_bert_weights,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Ağırlık dosyasını bul ──────────────────────────────────────
    found_path = None
    found_dir  = None
    for model_dir in MODEL_SEARCH_PATHS:
        for weight_name in WEIGHT_NAMES:
            w = model_dir / weight_name
            if w.exists():
                found_path = w
                found_dir  = model_dir
                break
        if found_path:
            break

    # ── Extractor ─────────────────────────────────────────────────
    extractor = LexiconExtractor()

    # ── Tokenizer ─────────────────────────────────────────────────
    tokenizer_path = str(found_dir) if found_dir and found_dir.exists() else "dbmdz/bert-base-turkish-cased"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

    # ── Model ─────────────────────────────────────────────────────
    model = LexiconBERTModel(
        bert_model_name="dbmdz/bert-base-turkish-cased",
        num_labels=10,
        lexicon_dim=LEXICON_DIM,
        dropout=0.1,
    )

    is_trained = False
    model_label = "Taban Model (Eğitilmemiş)"

    if found_path:
        try:
            state = torch.load(found_path, map_location=device, weights_only=True)
            load_lexicon_bert_weights(model, state)
            is_trained = True
            exp_num = "4" if "exp4" in str(found_path) else "3"
            model_label = f"Exp{exp_num} {'Master Hybrid' if exp_num=='4' else 'Lexicon'} (Eğitilmiş)"
        except Exception as e:
            model_label = f"Yükleme hatası: {e}"

    model.to(device)
    model.eval()
    return model, tokenizer, extractor, model_label, is_trained, device


@torch.no_grad()
def predict(
    text: str,
    model,
    tokenizer,
    extractor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Tek metin için tahmin yapar.
    Döndürür: (probabilities[10], lexicon_features[10], inference_ms)
    """
    from train_exp3_lexicon import LEXICON_DIM

    t0 = time.perf_counter()

    # Lexikon özelliği
    lex_feat = extractor.extract(text)
    lex_tensor = torch.tensor(lex_feat, dtype=torch.float32).unsqueeze(0).to(device)

    # Tokenize
    enc = tokenizer(
        text,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    # Forward
    out = model(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        token_type_ids=enc.get("token_type_ids"),
        lexicon_features=lex_tensor,
    )

    probs = F.softmax(out.logits, dim=-1).squeeze().cpu().numpy()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return probs, lex_feat, elapsed_ms


# ─── Grafik Fonksiyonları ────────────────────────────────────────────────────

def make_probability_chart(probs: np.ndarray, dominant_idx: int) -> go.Figure:
    """Olasılık dağılımı — yatay bar grafiği."""
    sorted_idx = np.argsort(probs)
    emos  = [EMOTIONS[i] for i in sorted_idx]
    vals  = [float(probs[i] * 100) for i in sorted_idx]
    emojis = [EMOTION_META[e]["emoji"] for e in emos]
    colors = [
        EMOTION_META[EMOTIONS[dominant_idx]]["color"]
        if i == dominant_idx else "#ecf0f1"
        for i in sorted_idx
    ]

    labels = [f"{em} {e}" for em, e in zip(emojis, emos)]

    fig = go.Figure(go.Bar(
        x=vals,
        y=labels,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="rgba(0,0,0,0.1)", width=0.5),
        ),
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(size=12, family="monospace"),
        hovertemplate="<b>%{y}</b><br>Olasılık: %{x:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(
            range=[0, max(vals) * 1.2],
            title="Olasılık (%)",
            gridcolor="rgba(128,128,128,0.15)",
            showgrid=True,
        ),
        yaxis=dict(tickfont=dict(size=13)),
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif"),
        showlegend=False,
    )
    return fig


def make_lexicon_radar(lex_feat: np.ndarray) -> go.Figure:
    """sözlük (NRC kategorili) özellikleri — radar grafik."""
    cats_tr = [
        "Öfke", "Beklenti", "Tiksinti", "Korku", "Sevinç",
        "Negatif", "Pozitif", "Üzüntü", "Şaşkınlık", "Güven",
    ]
    vals = list(lex_feat) + [lex_feat[0]]
    cats = cats_tr + [cats_tr[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals,
        theta=cats,
        fill="toself",
        fillcolor="rgba(102,126,234,0.2)",
        line=dict(color="#667eea", width=2),
        hovertemplate="<b>%{theta}</b>: %{r:.3f}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(lex_feat), 0.05) * 1.2],
                gridcolor="rgba(128,128,128,0.2)",
                tickfont=dict(size=9),
            ),
            angularaxis=dict(tickfont=dict(size=11)),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=30, r=30, t=30, b=30),
        height=340,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def make_nrc_bar(lex_feat: np.ndarray) -> go.Figure:
    """NRC kategorileri — dikey bar (sıfır olmayan kanallar)."""
    cats_tr = [
        "Öfke", "Beklenti", "Tiksinti", "Korku", "Sevinç",
        "Negatif", "Pozitif", "Üzüntü", "Şaşkınlık", "Güven",
    ]
    nz_idx = [i for i, v in enumerate(lex_feat) if v > 0.001]
    if not nz_idx:
        nz_idx = list(range(10))

    colors_nrc = ["#e74c3c","#f39c12","#27ae60","#8e44ad","#f1c40f",
                  "#95a5a6","#2ecc71","#3498db","#1abc9c","#e91e63"]

    fig = go.Figure(go.Bar(
        x=[cats_tr[i] for i in nz_idx],
        y=[float(lex_feat[i]) for i in nz_idx],
        marker=dict(
            color=[colors_nrc[i] for i in nz_idx],
            line=dict(color="rgba(0,0,0,0.1)", width=0.5),
        ),
        text=[f"{lex_feat[i]:.3f}" for i in nz_idx],
        textposition="outside",
        hovertemplate="<b>%{x}</b>: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(
            title="NRC Skoru",
            gridcolor="rgba(128,128,128,0.15)",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


# ─── Sistem Bilgisi ──────────────────────────────────────────────────────────

def get_system_info() -> dict:
    info = {
        "cuda":      torch.cuda.is_available(),
        "gpu_name":  "CPU only",
        "vram_gb":   0.0,
        "vram_used": 0.0,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"]  = props.name
        info["vram_gb"]   = props.total_memory / 1e9
        info["vram_used"] = torch.cuda.memory_allocated(0) / 1e9
    return info


def get_exp4_metrics() -> Optional[dict]:
    for exp_id in ["exp4", "exp3", "exp2", "exp1"]:
        p = ROOT / "results" / exp_id / "metrics_report.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return None


# ─── Lexikon Kelime Analizi ──────────────────────────────────────────────────

def analyze_lexicon_words(text: str, extractor) -> list:
    """
    Metindeki lexikon eşleşmelerini döndürür.
    [{word, categories, score}, ...]
    """
    import re
    punct_re = re.compile(r"[^\w\s]", re.UNICODE)
    tr_upper = str.maketrans("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ",
                              "abcçdefgğhıijklmnoöprsştuüvyz")
    clean = text.translate(tr_upper)
    clean = punct_re.sub(" ", clean)
    tokens = [t for t in clean.split() if len(t) >= 2]

    NRC_TR = ["Öfke","Beklenti","Tiksinti","Korku","Sevinç",
               "Negatif","Pozitif","Üzüntü","Şaşkınlık","Güven"]
    NRC_COLORS = ["#e74c3c","#f39c12","#27ae60","#8e44ad","#f1c40f",
                   "#7f8c8d","#2ecc71","#3498db","#1abc9c","#e91e63"]

    hits = []
    for token in tokens:
        vec = None
        if token in extractor.lexicon:
            vec = extractor.lexicon[token]
        else:
            ascii_tok = token.translate(str.maketrans("çğışöüîâû","cgisouiau"))
            if ascii_tok in extractor.lexicon:
                vec = extractor.lexicon[ascii_tok]
            elif len(token) >= 4:
                for sl in (7, 6, 5, 4):
                    stem = token[:sl] if len(token) >= sl else token
                    if stem in extractor._stem_index:
                        vec = extractor._stem_index[stem]
                        break
        if vec is not None and vec.sum() > 0:
            cats = [(NRC_TR[i], NRC_COLORS[i]) for i in range(10) if vec[i] > 0]
            hits.append({"word": token, "categories": cats, "score": float(vec.sum())})

    return sorted(hits, key=lambda x: -x["score"])


# ─── SHAP / Gradient Görselleştirme ─────────────────────────────────────────

def make_word_contribution_chart(
    words:  list,
    scores: np.ndarray,
    emotion: str,
    top_n:  int = 15,
) -> go.Figure:
    """
    Gradient × Giriş saliency değerlerini yatay Plotly bar olarak çizer.
    Pozitif (duyguyu destekleyen) → turuncu-kırmızı
    Negatif (duyguyu köreltir)   → mavi
    """
    n_show   = min(top_n, len(words))
    order    = np.argsort(np.abs(scores))[::-1][:n_show]
    words_s  = [words[j]  for j in order]
    scores_s = scores[order]

    # Görüntü için küçükten büyüğe
    sidx     = np.argsort(scores_s)
    words_s  = [words_s[j]  for j in sidx]
    scores_s = scores_s[sidx]

    colors = [
        EMOTION_META.get(emotion, {}).get("color", "#e74c3c") if s > 0 else "#3498db"
        for s in scores_s
    ]

    fig = go.Figure(go.Bar(
        x=scores_s.tolist(),
        y=words_s,
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.06)", width=0.5)),
        text=[f"{s:+.3f}" for s in scores_s],
        textposition="outside",
        textfont=dict(size=10, family="monospace"),
        hovertemplate="<b>%{y}</b><br>Katkı: %{x:+.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"Kelime Katkıları — <b>{emotion.upper()}</b> "
                 f"<span style='font-size:11px;color:#95a5a6'>(Gradient × Giriş Saliency)</span>",
            font=dict(size=13),
            x=0,
        ),
        xaxis=dict(
            title="Katkı Skoru",
            zeroline=True,
            zerolinecolor="#2c3e50",
            zerolinewidth=1.5,
            gridcolor="rgba(128,128,128,0.15)",
        ),
        yaxis=dict(tickfont=dict(size=12)),
        height=max(300, n_show * 35 + 80),
        margin=dict(l=10, r=80, t=50, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        annotations=[
            dict(
                x=0.01, y=1.06, xref="paper", yref="paper",
                text="🔴 Destekler  🔵 Köreltir",
                showarrow=False, font=dict(size=10, color="#7f8c8d"),
                align="left",
            )
        ],
    )
    return fig


def make_top_keywords_chart(keywords: list, emotion: str) -> go.Figure:
    """
    Lexikon × sınıflandırıcı ağırlık analizinden gelen en etkili kelimeleri gösterir.
    keywords: [{"word":..., "score":..., "nrc_channels":[...]}, ...]
    """
    if not keywords:
        return go.Figure()

    words  = [d["word"]  for d in keywords]
    scores = [d["score"] for d in keywords]
    nrc    = [", ".join(d.get("nrc_channels", [])[:2]) for d in keywords]
    color  = EMOTION_META.get(emotion, {}).get("color", "#667eea")

    fig = go.Figure(go.Bar(
        x=scores,
        y=words,
        orientation="h",
        marker=dict(
            color=scores,
            colorscale=[[0, "#ecf0f1"], [1, color]],
            line=dict(color="rgba(0,0,0,0.06)", width=0.5),
        ),
        text=nrc,
        textposition="outside",
        textfont=dict(size=9.5, color="#7f8c8d"),
        customdata=scores,
        hovertemplate="<b>%{y}</b><br>Skor: %{customdata:.3f}<br>NRC: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"En Etkili Kelimeler — <b>{emotion.upper()}</b> "
                 f"<span style='font-size:11px;color:#95a5a6'>(Lexikon × Ağırlık Analizi)</span>",
            font=dict(size=13), x=0,
        ),
        xaxis=dict(title="Lexikon × Sınıflandırıcı Skoru",
                   gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(tickfont=dict(size=12)),
        height=max(260, len(keywords) * 42 + 80),
        margin=dict(l=10, r=120, t=50, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


@st.cache_data(show_spinner=False)
def cached_top_keywords(_model, emotion: str, n: int = 5) -> list:
    """Model ağırlıklarından lexikon anahtar kelimelerini hesaplar (önbellekli)."""
    if not _SHAP_MODULE_OK:
        return []
    try:
        all_kw = top_keywords_per_emotion(_model, n=n)
        return all_kw.get(emotion, [])
    except Exception:
        return []


def gradient_saliency_for_text(
    text: str,
    target_idx: int,
    model,
    tokenizer,
    extractor,
    device: torch.device,
) -> Tuple[list, np.ndarray]:
    """
    Gradient × Giriş saliency — app içi wrapper.
    Modeli geçici olarak eğitim moduna alır (grad için), sonra eval'a döner.
    """
    if not _SHAP_MODULE_OK:
        return [], np.array([])

    # ModelLoader gerektirmeden doğrudan hesapla
    import re

    model.train()  # gradients için
    try:
        enc = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=128, return_offsets_mapping=True,
        )
        offset_mapping = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(device) for k, v in enc.items()}

        lex_vec  = extractor.extract(text)
        lex_tens = torch.tensor(lex_vec, dtype=torch.float32).unsqueeze(0).to(device)

        embeddings = model.bert.embeddings.word_embeddings(enc["input_ids"])
        embeddings = embeddings.detach().requires_grad_(True)

        bert_out  = model.bert(
            inputs_embeds=embeddings,
            attention_mask=enc["attention_mask"],
            token_type_ids=enc.get("token_type_ids"),
        )
        cls_vec   = model.dropout(bert_out.last_hidden_state[:, 0, :])
        combined  = torch.cat([cls_vec, lex_tens], dim=-1)
        logits    = model.classifier(combined)

        logits[0, target_idx].backward()

        grads    = embeddings.grad[0].cpu().detach().numpy()
        saliency = np.abs(grads).sum(axis=-1)

        from shap_interpretability import _aggregate_subwords
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].cpu().tolist())
        words, word_scores = _aggregate_subwords(tokens, saliency, offset_mapping, text)

        norm = np.linalg.norm(word_scores)
        if norm > 1e-8:
            word_scores = word_scores / norm

        return words, word_scores

    except Exception as e:
        st.warning(f"Gradient hesaplaması başarısız: {e}", icon="⚠️")
        return [], np.array([])
    finally:
        model.eval()


# ─── Ana Uygulama ────────────────────────────────────────────────────────────

def main():
    # ── Safety: exp4 best_model kontrolü ─────────────────────────
    _exp4_path = ROOT / "results" / "exp4" / "best_model"
    _exp3_path = ROOT / "results" / "exp3" / "best_model"
    if not _exp4_path.exists() and not _exp3_path.exists():
        st.error(
            "### Eğitilmiş model bulunamadı\n\n"
            "**results/exp4/best_model** klasörü mevcut değil.\n\n"
            "Lütfen önce modeli eğitin:\n"
            "```bash\npython run_experiments.py --only 4\n```\n"
            "Yalnızca demo modu aktif — tahminler rastgele olacaktır.",
            icon="🚫",
        )
        st.info(
            "Eğer sadece arayüzü keşfetmek istiyorsanız "
            "**Analiz Et** butonunu kullanabilirsiniz — "
            "model ağırlıkları yüklenene kadar tahminler sembolik kalır.",
            icon="💡",
        )

    # ── Model yükle ──────────────────────────────────────────────
    with st.spinner("Model yükleniyor..."):
        model, tokenizer, extractor, model_label, is_trained, device = load_model_and_tools()

    sys_info = get_system_info()
    metrics  = get_exp4_metrics()

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔧 Sistem Bilgisi")

        # GPU badge
        if sys_info["cuda"]:
            st.markdown(
                f'<span class="sys-badge sys-badge-gpu">🖥️ {sys_info["gpu_name"]}</span>'
                f'<span class="sys-badge">💾 {sys_info["vram_gb"]:.1f} GB VRAM</span>'
                f'<span class="sys-badge">⚡ fp16 Aktif</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<span class="sys-badge">💻 CPU Mode</span>', unsafe_allow_html=True)

        st.markdown("---")

        # Model bilgisi
        st.markdown("## 🧠 Model Bilgisi")
        status_color = "#2ecc71" if is_trained else "#e74c3c"
        status_icon  = "✅" if is_trained else "⚠️"
        st.markdown(
            f"<div style='padding:0.6rem;background:rgba(0,0,0,0.1);"
            f"border-radius:8px;border-left:3px solid {status_color}'>"
            f"<strong>{status_icon} {model_label}</strong></div>",
            unsafe_allow_html=True,
        )

        # Mimari özeti
        st.markdown("""
        **Mimari:**
        ```
        BERTurk [CLS]  → 768-dim
        Sözlük (NRC kategorili)    →  10-dim
                    ──────────
        Concatenate    → 778-dim
        Linear(778→10) →  10 sınıf
        ```
        """)

        # Eğitim metrikleri (varsa)
        if metrics:
            st.markdown("**Test Sonuçları:**")
            col1, col2 = st.columns(2)
            col1.metric("F1-Macro",  f"{metrics.get('test_f1_macro', 0):.4f}")
            col2.metric("Accuracy",  f"{metrics.get('test_accuracy', 0):.4f}")

        if not is_trained:
            st.warning(
                "⚠️ Eğitilmiş ağırlık bulunamadı.\n\n"
                "Tahminler anlamsız olabilir.\n\n"
                "Eğitmek için:\n```\npython run_experiments.py --only 4\n```",
                icon="⚠️",
            )

        st.markdown("---")

        # Demo örnekler
        st.markdown("## 💡 Demo Örnekler")
        st.markdown('<p class="demo-header">Hızlı test için tıkla:</p>', unsafe_allow_html=True)

        for label, text in DEMO_TEXTS:
            if st.button(label, key=f"demo_{label}", use_container_width=True):
                st.session_state["input_text"] = text

        st.markdown("---")

        # ── Proje Hakkında ────────────────────────────────────────
        with st.expander("📚 Proje Hakkında", expanded=False):
            st.markdown("**TR-EmotionNet** — 10-Sınıflı Türkçe Duygu Analizi")
            st.markdown(
                "<div style='font-size:0.82rem;line-height:1.9'>"
                + "".join(
                    f"<div style='display:flex;align-items:center;gap:6px;padding:1px 0'>"
                    f"<span style='font-size:1.1rem'>{m}</span>"
                    f"<span style='width:6px;height:6px;border-radius:50%;"
                    f"background:{c};display:inline-block;flex-shrink:0'></span>"
                    f"<span style='color:#bdc3c7'>{e}</span></div>"
                    for e, m, c in [
                        ("mutluluk",  "😊", "#f1c40f"),
                        ("üzüntü",    "😢", "#3498db"),
                        ("öfke",      "😠", "#e74c3c"),
                        ("korku",     "😨", "#8e44ad"),
                        ("şaşkınlık","😲", "#1abc9c"),
                        ("tiksinti",  "🤢", "#27ae60"),
                        ("sevgi",     "❤️", "#e91e63"),
                        ("nötr",      "😐", "#95a5a6"),
                        ("utanç",     "😳", "#e67e22"),
                        ("gurur",     "🏆", "#f39c12"),
                    ]
                )
                + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("---")
            st.markdown(
                "<div style='font-size:0.8rem;color:#95a5a6'>"
                "<b style='color:#667eea'>Hibrit Mimari</b><br>"
                "BERTurk <code>[CLS]</code> → <b>768</b>-dim<br>"
                "Sözlük (NRC kategorili) → <b>10</b>-dim<br>"
                "<span style='color:#667eea'>──────────────</span><br>"
                "Concat → <b>778</b>-dim<br>"
                "Linear(778 → 10) → 10 sınıf<br><br>"
                "<b style='color:#667eea'>Ablasyon</b><br>"
                "Exp1 Baseline &nbsp;→ Exp2 Cross-Lingual<br>"
                "Exp3 Lexikon &nbsp;&nbsp;→ Exp4 Master Hybrid"
                "</div>",
                unsafe_allow_html=True,
            )

    # ── Ana İçerik ───────────────────────────────────────────────
    st.markdown('<div class="app-title">🧠 Türkçe Duygu Analizi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Master Hybrid Model — BERTurk + Sözlük (NRC kategorili) | 10 Duygu Sınıfı</div>',
        unsafe_allow_html=True,
    )

    # Giriş alanı
    default_text = st.session_state.get("input_text", "")
    input_text = st.text_area(
        "📝 Analiz edilecek metni girin:",
        value=default_text,
        height=130,
        placeholder=(
            "Türkçe bir metin yazın veya sol taraftaki demo örneklerden birini seçin...\n\n"
            "Örnek: 'Bugün harika bir gün! Çok mutluyum.'"
        ),
        key="main_textarea",
    )

    # Buton
    col_btn, col_len, col_info = st.columns([2, 2, 4])
    with col_btn:
        analyze_clicked = st.button(
            "🔍 Analiz Et",
            type="primary",
            use_container_width=True,
        )
    with col_len:
        if input_text:
            words = len(input_text.split())
            chars = len(input_text)
            st.caption(f"📊 {words} kelime · {chars} karakter")

    st.markdown("---")

    # ── Analiz & Sonuçlar ────────────────────────────────────────
    if analyze_clicked and input_text.strip():

        with st.spinner("Analiz ediliyor..."):
            probs, lex_feat, ms = predict(input_text, model, tokenizer, extractor, device)

        dominant_idx  = int(np.argmax(probs))
        dominant_emo  = EMOTIONS[dominant_idx]
        dominant_meta = EMOTION_META[dominant_emo]

        # ── Sentiment Card ───────────────────────────────────────
        conf      = float(probs[dominant_idx]) * 100
        dom_color = dominant_meta["color"]
        # İkincil renk: tonu koyulaştır (gradient için)
        hex_c  = dom_color.lstrip("#")
        r, g, b = int(hex_c[0:2],16), int(hex_c[2:4],16), int(hex_c[4:6],16)
        dark   = f"#{max(r-45,0):02x}{max(g-45,0):02x}{max(b-45,0):02x}"

        # Güven seviyesi etiketi
        if conf >= 80:   conf_label = "Yüksek Güven"
        elif conf >= 50: conf_label = "Orta Güven"
        else:            conf_label = "Düşük Güven"

        # 2. ve 3. en yüksek duygu
        top3 = np.argsort(probs)[::-1][:3]
        runner_ups = " · ".join(
            f"{EMOTION_META[EMOTIONS[i]]['emoji']} {EMOTIONS[i]} {probs[i]*100:.0f}%"
            for i in top3[1:]
        )

        st.markdown(
            f"<div class='sentiment-card' "
            f"style='background:linear-gradient(135deg, {dom_color} 0%, {dark} 100%)'>"
            f"<span class='sc-emoji'>{dominant_meta['emoji']}</span>"
            f"<span class='sc-label'>{dominant_emo.upper()}</span>"
            f"<div class='sc-sub'>{dominant_meta['en']} &nbsp;·&nbsp; {dominant_meta['desc'][:60]}…</div>"
            f"<div style='margin-top:0.9rem'>"
            f"<span class='sc-conf'>{conf:.1f}% — {conf_label}</span>"
            f"<span class='sc-ms'>{ms:.1f} ms</span>"
            f"</div>"
            f"<div style='margin-top:0.6rem;font-size:0.8rem;color:rgba(255,255,255,0.5)'>"
            f"Diğer: {runner_ups}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Sekmeler ─────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Duygu Analizi",
            "📖 Lexikon İstatistikleri",
            "🔍 SHAP Yorumlanabilirlik",
            "ℹ️ Duygu Bilgisi",
        ])

        # TAB 1: Olasılık grafiği
        with tab1:
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown("#### Olasılık Dağılımı")
                fig_bar = make_probability_chart(probs, dominant_idx)
                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

            with c2:
                st.markdown("#### Skor Tablosu")
                rows = sorted(
                    [(EMOTIONS[i], float(probs[i])) for i in range(10)],
                    key=lambda x: -x[1],
                )
                for emo, p in rows:
                    meta   = EMOTION_META[emo]
                    width  = int(p * 100)
                    is_dom = (emo == dominant_emo)
                    color  = meta["color"]
                    fw     = 700 if is_dom else 400
                    tc     = color if is_dom else "#7f8c8d"
                    st.markdown(
                        f"<div style='display:flex;align-items:center;margin:3px 0;'>"
                        f"<span style='width:1.5rem;text-align:center'>{meta['emoji']}</span>"
                        f"<span style='width:5.5rem;font-size:0.85rem;padding:0 6px'>{emo}</span>"
                        f"<div style='flex:1;background:#ecf0f1;border-radius:4px;height:18px'>"
                        f"<div style='width:{width}%;background:{color};"
                        f"border-radius:4px;height:100%;'></div></div>"
                        f"<span style='width:3.8rem;text-align:right;font-size:0.82rem;"
                        f"font-weight:{fw};color:{tc}'>"
                        f"{p*100:.1f}%</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # TAB 2: Lexikon istatistikleri
        with tab2:
            has_lex = lex_feat.sum() > 0.001
            if not has_lex:
                st.info(
                    "Bu metin için lexikon eşleşmesi bulunamadı. "
                    "NRC sözlüğündeki Türkçe duygu kelimelerini içeren bir metin deneyin.",
                    icon="ℹ️",
                )
            else:
                c_radar, c_bar = st.columns([1, 1])
                with c_radar:
                    st.markdown("#### NRC Boyut Profili")
                    st.plotly_chart(
                        make_lexicon_radar(lex_feat),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                with c_bar:
                    st.markdown("#### Aktif NRC Kanalları")
                    st.plotly_chart(
                        make_nrc_bar(lex_feat),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

            # Eşleşen kelimeler
            st.markdown("#### 🔍 Metindeki Duygu Kelimeleri")
            word_hits = analyze_lexicon_words(input_text, extractor)
            if word_hits:
                html_parts = []
                for hit in word_hits:
                    cats_html = "".join(
                        f"<span class='lex-tag' style='background:{c}22;color:{c};border:1px solid {c}44'>"
                        f"{cat}</span>"
                        for cat, c in hit["categories"][:3]
                    )
                    html_parts.append(
                        f"<div style='display:flex;align-items:center;gap:8px;"
                        f"padding:4px 8px;border-radius:6px;margin:2px 0;"
                        f"background:rgba(255,255,255,0.03)'>"
                        f"<strong style='min-width:100px;font-size:0.9rem'>{hit['word']}</strong>"
                        f"{cats_html}"
                        f"<span style='color:#7f8c8d;font-size:0.8rem;margin-left:auto'>"
                        f"toplam: {hit['score']:.2f}</span>"
                        f"</div>"
                    )
                st.markdown("".join(html_parts), unsafe_allow_html=True)
                st.caption(f"Toplam {len(word_hits)} duygu kelimesi tespit edildi.")
            else:
                st.caption("Metinde lexikon sözlüğünde bulunan kelime tespit edilemedi.")

            # Lexikon istatistikleri
            with st.expander("📋 Ham NRC Vektör Değerleri"):
                NRC_TR = ["Öfke","Beklenti","Tiksinti","Korku","Sevinç",
                           "Negatif","Pozitif","Üzüntü","Şaşkınlık","Güven"]
                df_lex = {
                    "NRC Kategorisi": NRC_TR,
                    "İngilizce": ["Anger","Anticip.","Disgust","Fear","Joy",
                                   "Negative","Positive","Sadness","Surprise","Trust"],
                    "Skor": [f"{v:.4f}" for v in lex_feat],
                }
                import pandas as pd
                st.dataframe(pd.DataFrame(df_lex), hide_index=True, use_container_width=True)

        # TAB 3: SHAP Yorumlanabilirlik
        with tab3:
            if not _SHAP_MODULE_OK:
                st.error(
                    "shap_interpretability.py modülü yüklenemedi. "
                    "train_exp3_lexicon.py'nin proje kökünde bulunduğundan emin olun.",
                    icon="❌",
                )
            else:
                st.markdown(
                    "#### Modelin Kararı Nasıl Veriyor?\n"
                    "Aşağıdaki grafik, modelin tahminini etkileyen kelimeleri gösterir. "
                    "**Turuncu/kırmızı** kelimeler duyguyu destekler, "
                    "**mavi** kelimeler duyguyu köreltir.",
                )

                shap_col1, shap_col2 = st.columns([1, 1])

                # ── Sol: Gradient Saliency ─────────────────────────────────
                with shap_col1:
                    st.markdown("##### Gradient × Giriş Saliency")
                    st.caption(
                        "Her kelimenin tahmine yaptığı doğrudan etki "
                        "(∂tahmin/∂embedding × embedding normu)."
                    )

                    with st.spinner("Gradient hesaplanıyor..."):
                        words_g, scores_g = gradient_saliency_for_text(
                            text       = input_text,
                            target_idx = dominant_idx,
                            model      = model,
                            tokenizer  = tokenizer,
                            extractor  = extractor,
                            device     = device,
                        )

                    if len(words_g) > 0:
                        fig_grad = make_word_contribution_chart(
                            words_g, scores_g, dominant_emo, top_n=12
                        )
                        st.plotly_chart(fig_grad, use_container_width=True,
                                        config={"displayModeBar": False})

                        # Isı haritası (token renklendirme)
                        st.markdown("**Token Saliency Isı Haritası:**")
                        max_s = scores_g.max() if scores_g.max() > 1e-8 else 1.0
                        norm_s = scores_g / max_s
                        dom_color_hex = dominant_meta["color"].lstrip("#")
                        r_d = int(dom_color_hex[0:2], 16)
                        g_d = int(dom_color_hex[2:4], 16)
                        b_d = int(dom_color_hex[4:6], 16)
                        tags = []
                        for word, ns in zip(words_g, norm_s):
                            alpha = max(0.15, float(ns))
                            tags.append(
                                f"<span style='background:rgba({r_d},{g_d},{b_d},{alpha:.2f});"
                                f"color:{'white' if alpha > 0.5 else '#2c3e50'};"
                                f"padding:2px 6px;border-radius:4px;"
                                f"margin:2px;display:inline-block;font-size:0.9rem;"
                                f"font-weight:{'600' if alpha > 0.4 else '400'}'>"
                                f"{word}</span>"
                            )
                        st.markdown(
                            "<div style='line-height:2.2;padding:8px'>"
                            + " ".join(tags) + "</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("Gradient saliency hesaplanamadı.", icon="ℹ️")

                # ── Sağ: Lexikon × Ağırlık Anahtar Kelimeleri ─────────────
                with shap_col2:
                    st.markdown("##### En Etkili Lexikon Kelimeleri")
                    st.caption(
                        f"**{dominant_emo.upper()}** sınıfı için "
                        "sınıflandırıcı ağırlıkları × NRC vektörü skoru.\n"
                        "Eğitime gerek duymaz — doğrudan model ağırlıklarından türetilir."
                    )

                    kw = cached_top_keywords(model, dominant_emo, n=7)
                    if kw:
                        fig_kw = make_top_keywords_chart(kw, dominant_emo)
                        st.plotly_chart(fig_kw, use_container_width=True,
                                        config={"displayModeBar": False})

                        # Tablo
                        st.markdown("**Detay Tablosu:**")
                        for rank, d in enumerate(kw, 1):
                            nrc_str  = ", ".join(d.get("nrc_channels", []))
                            score_pct = min(100, abs(d["score"]) * 50)
                            bar_color = dominant_meta["color"]
                            st.markdown(
                                f"<div style='display:flex;align-items:center;"
                                f"padding:4px 6px;margin:2px 0;border-radius:6px;"
                                f"background:rgba(255,255,255,0.03)'>"
                                f"<span style='width:1.4rem;color:#7f8c8d;font-size:0.8rem'>"
                                f"{rank}.</span>"
                                f"<strong style='min-width:90px;font-size:0.9rem'>{d['word']}</strong>"
                                f"<div style='flex:1;background:#ecf0f1;border-radius:3px;height:10px;margin:0 8px'>"
                                f"<div style='width:{score_pct:.0f}%;background:{bar_color};"
                                f"border-radius:3px;height:100%'></div></div>"
                                f"<span style='font-size:0.75rem;color:#95a5a6;min-width:80px'>{nrc_str}</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info(
                            "Lexikon anahtar kelime analizi için model ağırlıkları gerekli.",
                            icon="ℹ️",
                        )

                # ── Alt: Tüm Duygular için Özet ────────────────────────────
                with st.expander("📋 Tüm Duygular için Anahtar Kelimeler (Hoca Sunumu)", expanded=False):
                    st.caption(
                        "Her duygu sınıfı için en yüksek skorlu 5 lexikon kelimesi. "
                        "Kaydetmek için: `python shap_interpretability.py --top-words`"
                    )
                    all_kw = cached_top_keywords(model, "mutluluk", n=1)  # ısınma
                    try:
                        all_emotions_kw = top_keywords_per_emotion(model, n=5)
                    except Exception:
                        all_emotions_kw = {}

                    if all_emotions_kw:
                        cols_kw = st.columns(2)
                        for i, emo in enumerate(EMOTIONS):
                            kw_emo = all_emotions_kw.get(emo, [])
                            with cols_kw[i % 2]:
                                color_emo = EMOTION_META[emo]["color"]
                                emoji_emo = EMOTION_META[emo]["emoji"]
                                words_txt = " | ".join(
                                    f'**{d["word"]}** ({d["score"]:+.2f})'
                                    for d in kw_emo
                                )
                                st.markdown(
                                    f"<div style='border-left:3px solid {color_emo};"
                                    f"padding:4px 10px;margin:4px 0;border-radius:0 6px 6px 0;"
                                    f"background:rgba(255,255,255,0.02)'>"
                                    f"<strong>{emoji_emo} {emo.upper()}</strong><br>"
                                    f"<span style='font-size:0.82rem;color:#bdc3c7'>"
                                    f"{words_txt}</span></div>",
                                    unsafe_allow_html=True,
                                )

                # ── Raporu Kaydet ──────────────────────────────────────────
                st.markdown("---")
                st.markdown(
                    "**Tam raporu kaydetmek için terminal'de:**\n"
                    "```bash\n"
                    "python shap_interpretability.py --full-report\n"
                    "```\n"
                    "Çıktı: `results/interpretability/` — force_plot, summary_plot, "
                    "saliency haritaları, top_keywords_all.json"
                )

        # TAB 4: Duygu bilgisi
        with tab4:
            st.markdown(f"### {dominant_meta['emoji']} {dominant_emo.capitalize()} Hakkında")
            st.markdown(
                f"<div class='emotion-card' style='border-color:{dom_color}'>"
                f"<p style='font-size:1.05rem'>{dominant_meta['desc']}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.markdown("### Tüm Duygu Sınıfları")
            cols = st.columns(2)
            for i, emo in enumerate(EMOTIONS):
                meta = EMOTION_META[emo]
                p    = float(probs[i])
                is_dom = (emo == dominant_emo)
                with cols[i % 2]:
                    border = f"2px solid {meta['color']}" if is_dom else "1px solid rgba(128,128,128,0.2)"
                    bg     = f"rgba({','.join(str(int(meta['color'].lstrip('#')[j*2:j*2+2],16)) for j in range(3))},0.08)" if is_dom else "rgba(255,255,255,0.02)"
                    st.markdown(
                        f"<div style='border:{border};background:{bg};"
                        f"border-radius:8px;padding:8px 12px;margin:4px 0'>"
                        f"<span style='font-size:1.2rem'>{meta['emoji']}</span>"
                        f" <strong>{emo}</strong>"
                        f"<span style='float:right;color:{meta['color']};font-weight:700'>{p*100:.1f}%</span><br>"
                        f"<span style='font-size:0.78rem;color:#95a5a6'>{meta['desc'][:70]}...</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    elif analyze_clicked and not input_text.strip():
        st.warning("Lütfen analiz edilecek bir metin girin.", icon="⚠️")
    else:
        # Karşılama ekranı
        st.markdown("""
        <div style='text-align:center;padding:3rem 2rem;opacity:0.7'>
            <div style='font-size:4rem'>🧠</div>
            <h3>Türkçe metninizdeki duyguyu keşfedin</h3>
            <p style='color:#95a5a6'>
                Yukarıdaki metin alanına yazın veya sol taraftaki demo örneklerden birini seçin,<br>
                ardından <strong>Analiz Et</strong> butonuna tıklayın.
            </p>
            <div style='margin-top:1.5rem;display:flex;justify-content:center;flex-wrap:wrap;gap:8px'>
        """, unsafe_allow_html=True)

        welcome_emojis = [m["emoji"] + " " + e for e, m in EMOTION_META.items()]
        cols = st.columns(5)
        for i, (emo, meta) in enumerate(EMOTION_META.items()):
            with cols[i % 5]:
                st.markdown(
                    f"<div style='text-align:center;padding:8px;border-radius:8px;"
                    f"border:1px solid {meta['color']}33;margin:3px'>"
                    f"<div style='font-size:1.6rem'>{meta['emoji']}</div>"
                    f"<div style='font-size:0.8rem;color:{meta['color']}'>{emo}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("</div></div>", unsafe_allow_html=True)

    # ── Hardware Flex Badge (sabit sağ alt köşe) ─────────────────
    gpu_line = (
        f"🖥️ <strong>{sys_info['gpu_name']}</strong>"
        if sys_info["cuda"]
        else "💻 CPU Mode"
    )
    st.markdown(
        f"<div class='hw-footer'>"
        f"{gpu_line}<br>"
        f"<strong>Intel Core i9</strong> &nbsp;·&nbsp; "
        f"<strong style='color:#2ecc71'>fp16</strong> aktif<br>"
        f"<span style='opacity:0.45;font-size:0.65rem'>TR-EmotionNet · Exp4 Master Hybrid</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
