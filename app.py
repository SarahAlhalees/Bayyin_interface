import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification
import torch
import numpy as np
import re
from collections import Counter
import joblib

# -----------------------------------------
# Streamlit Page Settings
# -----------------------------------------
st.set_page_config(
    page_title="بَيِّنْ - مصنف قراءة النصوص العربية",
    page_icon="📚",
    layout="centered"
)

# -----------------------------------------
# Arabic text normalization
# -----------------------------------------
ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")

def normalize_ar(text):
    text = str(text)
    text = ARABIC_DIACRITICS.sub("", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[ؤئ]", "ء", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------------------
# Load Models
# -----------------------------------------
@st.cache_resource
def load_models():
    models = {}
    
    # Original Model: Arabertv2_D3Tok
    try:
        orig_repo = "SarahAlhalees/Arabertv2_D3Tok"
        orig_subfolder = "Arabertv2_D3Tok"
        models['orig_tokenizer'] = AutoTokenizer.from_pretrained(orig_repo, subfolder=orig_subfolder)
        models['orig_model'] = AutoModelForSequenceClassification.from_pretrained(orig_repo, subfolder=orig_subfolder)
    except Exception as e:
        st.error(f"خطأ في تحميل النموذج الأصلي: {str(e)}")
        models['orig_tokenizer'] = None
        models['orig_model'] = None
    
    # Model 1: CAMeLBERTmix_D3Tok
    try:
        mix_repo = "SarahAlhalees/CAMeLBERTmix_D3Tok"
        models['mix_tokenizer'] = AutoTokenizer.from_pretrained(mix_repo)
        models['mix_model'] = BertForSequenceClassification.from_pretrained(mix_repo)
    except Exception as e:
        st.error(f"خطأ في تحميل نموذج CAMeLBERTmix: {str(e)}")
        models['mix_tokenizer'] = None
        models['mix_model'] = None
    
    # Model 2: CAMeLBERTmsa_D3Tok
    try:
        msa_repo = "SarahAlhalees/CAMeLBERTmsa_D3Tok"
        models['msa_tokenizer'] = AutoTokenizer.from_pretrained(msa_repo)
        models['msa_model'] = BertForSequenceClassification.from_pretrained(msa_repo)
    except Exception as e:
        st.error(f"خطأ في تحميل نموذج CAMeLBERTmsa: {str(e)}")
        models['msa_tokenizer'] = None
        models['msa_model'] = None

    # Model 3: BiLSTM + AraBERT (joblib)
    try:
        bilstm_path = "Raya-y/Bayyin_models/bilstm_arabert_bayyin.joblib"
        models['bilstm_model'] = joblib.load(bilstm_path)
    except Exception as e:
        st.error(f"خطأ في تحميل نموذج BiLSTM-AraBERT: {str(e)}")
        models['bilstm_model'] = None
    
    return models

models_dict = load_models()
orig_tokenizer = models_dict.get('orig_tokenizer')
orig_model = models_dict.get('orig_model')
mix_tokenizer = models_dict.get('mix_tokenizer')
mix_model = models_dict.get('mix_model')
msa_tokenizer = models_dict.get('msa_tokenizer')
msa_model = models_dict.get('msa_model')
bilstm_model = models_dict.get('bilstm_model')

# -----------------------------------------
# UI Styling
# -----------------------------------------
st.markdown("""
    <style>
    textarea {
        direction: rtl;
        text-align: right;
        font-size: 16px;
    }
    .gradient-title {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: bold;
        text-align: center;
    }
    .final-verdict-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        border: 2px solid #fff;
    }

    /* Model card styling */
    .model-card {
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    .model-card-orig { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); }
    .model-card-mix { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .model-card-msa { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }

    /* NEW custom color card */
    .model-card-bilstm { background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%); }

    .level-badge {
        padding: 8px 20px;
        border-radius: 25px;
        font-size: 1.2em;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .level-1, .level-2 { background: #2ecc71; }
    .level-3, .level-4 { background: #f39c12; }
    .level-5, .level-6 { background: #e74c3c; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='gradient-title'>بَيِّنْ</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; direction: rtl; color: #667eea;'>مصنف مستوى قراءة النصوص العربية</h3>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------------------
# User Input
# -----------------------------------------
text = st.text_area(
    label="",
    height=200,
    placeholder="اكتب أو الصق النص هنا...",
    key="arabic_input"
)

# -----------------------------------------
# Prediction Logic
# -----------------------------------------
if st.button("تصنيف النص", use_container_width=True):

    if not text.strip():
        st.error("الرجاء إدخال نص قبل الضغط على زر التصنيف.")

    elif not re.search(r'[\u0600-\u06ff]', text):
        st.error("عذراً، النص المدخل لا يبدو أنه باللغة العربية.")

    else:
        cleaned = normalize_ar(text)

        def predict_level(model, tokenizer, text_input):
            if model and tokenizer:
                try:
                    inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=256)
                    with torch.no_grad():
                        logits = model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).numpy()[0]
                    return np.argmax(probs) + 1
                except:
                    return None
            return None

        def predict_bilstm(model, text_input):
            if model:
                try:
                    pred = model.predict([text_input])[0]
                    return int(pred)
                except:
                    return None
            return None

        # Predictions
        orig_level = predict_level(orig_model, orig_tokenizer, cleaned)
        mix_level  = predict_level(mix_model, mix_tokenizer, cleaned)
        msa_level  = predict_level(msa_model, msa_tokenizer, cleaned)
        bilstm_level = predict_bilstm(bilstm_model, cleaned)

        # Hard voting
        predictions = [l for l in [orig_level, mix_level, msa_level, bilstm_level] if l is not None]

        final_level = None
        if predictions:
            final_level = Counter(predictions).most_common(1)[0][0]

        # -----------------------------------------
        # Display Results
        # -----------------------------------------
        level_names = {
            1: "سهل جداً", 2: "سهل", 3: "متوسط",
            4: "صعب قليلاً", 5: "صعب", 6: "صعب جداً"
        }

        if final_level:
            st.markdown("---")
            st.markdown(f"""
            <div class='final-verdict-card'>
                <h2>النتيجة النهائية</h2>
                <h1 style='font-size: 3.5em;'>المستوى {final_level}</h1>
                <h3>{level_names.get(final_level)}</h3>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<h4 style='text-align: right; direction: rtl; color: #555;'>تفاصيل النماذج:</h4>", unsafe_allow_html=True)

        # Model Cards
        c1, c2, c3, c4 = st.columns(4)

        def display_mini_card(col, title, level, css_class):
            with col:
                if level:
                    st.markdown(f"""
                    <div class='model-card {css_class}'>
                        <h4>{title}</h4>
                        <div class='level-badge level-{level}'>المستوى {level}</div>
                        <p>{level_names.get(level)}</p>
                    </div>
                    """, unsafe_allow_html=True)

        display_mini_card(c1, "Arabertv2", orig_level, "model-card-orig")
        display_mini_card(c2, "CAMeLBERT Mix", mix_level, "model-card-mix")
        display_mini_card(c3, "CAMeLBERT MSA", msa_level, "model-card-msa")
        display_mini_card(c4, "BiLSTM + AraBERT", bilstm_level, "model-card-bilstm")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #667eea;'>© 2025 — مشروع بَيِّنْ</p>", unsafe_allow_html=True)
