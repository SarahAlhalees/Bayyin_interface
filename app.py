import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import re

# -----------------------------------------
# Streamlit Page Settings
# -----------------------------------------
st.set_page_config(
    page_title="بَيِّنْ - مصنف قراءة النصوص العربية",
    page_icon="",
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
    # Original Model: Arabertv2_D3Tok
    orig_repo = "SarahAlhalees/Arabertv2_D3Tok"
    orig_subfolder = "Arabertv2_D3Tok"
    orig_tokenizer = AutoTokenizer.from_pretrained(orig_repo, subfolder=orig_subfolder)
    orig_model = AutoModelForSequenceClassification.from_pretrained(orig_repo, subfolder=orig_subfolder)
    
    # Model 1: Machine Learning
    ml_repo = "SarahAlhalees/MachineLearning"
    ml_tokenizer = AutoTokenizer.from_pretrained(ml_repo)
    ml_model = AutoModelForSequenceClassification.from_pretrained(ml_repo)
    
    # Model 2: Deep Learning
    dl_repo = "SarahAlhalees/Deeplearning"
    dl_tokenizer = AutoTokenizer.from_pretrained(dl_repo)
    dl_model = AutoModelForSequenceClassification.from_pretrained(dl_repo)
    
    return orig_tokenizer, orig_model, ml_tokenizer, ml_model, dl_tokenizer, dl_model

orig_tokenizer, orig_model, ml_tokenizer, ml_model, dl_tokenizer, dl_model = load_models()

# -----------------------------------------
# UI Layout with Colorful Styling
# -----------------------------------------
st.markdown("""
    <style>
    textarea {
        direction: rtl;
        text-align: right;
        font-size: 16px;
    }
    .rtl-text {
        direction: rtl;
        text-align: right;
    }
    .gradient-title {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3em;
        font-weight: bold;
        text-align: center;
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .model-card-orig {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
    }
    .model-card-ml {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .model-card-dl {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .level-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 25px;
        font-size: 1.2em;
        font-weight: bold;
        margin: 10px 0;
    }
    .level-1, .level-2 { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; }
    .level-3, .level-4 { background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%); color: white; }
    .level-5, .level-6 { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); color: white; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='gradient-title'>بَيِّنْ</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; direction: rtl; color: #667eea;'>مصنف مستوى قراءة النصوص العربية</h3>", unsafe_allow_html=True)

st.markdown("---")

text = st.text_area(
    label="",
    height=200,
    placeholder="اكتب أو الصق النص هنا...",
    key="arabic_input"
)

# -----------------------------------------
# Prediction
# -----------------------------------------
if st.button("تصنيف النص", use_container_width=True):
    if not text.strip():
        st.warning("الرجاء إدخال نص.")
    else:
        cleaned = normalize_ar(text)
        
        # Predict with Original Model (Arabertv2_D3Tok)
        orig_inputs = orig_tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        
        with torch.no_grad():
            orig_logits = orig_model(**orig_inputs).logits
        
        orig_probs = torch.softmax(orig_logits, dim=-1).numpy()[0]
        orig_pred_idx = np.argmax(orig_probs)
        orig_level = orig_pred_idx + 1
        
        # Predict with Model 1 (Machine Learning)
        ml_inputs = ml_tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        
        with torch.no_grad():
            ml_logits = ml_model(**ml_inputs).logits
        
        ml_probs = torch.softmax(ml_logits, dim=-1).numpy()[0]
        ml_pred_idx = np.argmax(ml_probs)
        ml_level = ml_pred_idx + 1
        
        # Predict with Model 2 (Deep Learning)
        dl_inputs = dl_tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        
        with torch.no_grad():
            dl_logits = dl_model(**dl_inputs).logits
        
        dl_probs = torch.softmax(dl_logits, dim=-1).numpy()[0]
        dl_pred_idx = np.argmax(dl_probs)
        dl_level = dl_pred_idx + 1

        # -----------------------------------------
        # Results Section
        # -----------------------------------------
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; direction: rtl; color: #667eea;'>نتائج التصنيف</h2>", unsafe_allow_html=True)
        
        level_colors = {1: "", 2: "", 3: "", 4: "", 5: "", 6: ""}
        level_names = {
            1: "سهل جداً", 2: "سهل", 3: "متوسط", 
            4: "صعب قليلاً", 5: "صعب", 6: "صعب جداً"
        }
        
        # Original Model Results
        st.markdown("<div class='model-card model-card-orig'>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: right; margin: 0; color: #333;'>النموذج الأصلي (Arabertv2_D3Tok)</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='level-badge level-{orig_level}'>{level_colors.get(orig_level, '')} المستوى {orig_level}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<p style='font-size: 1.3em; margin-top: 15px; color: #333;'><strong>{level_names.get(orig_level, 'غير معروف')}</strong></p>", unsafe_allow_html=True)
        
        st.progress(int(orig_probs[orig_pred_idx] * 100))
        st.markdown(f"<p style='text-align: right; color: #333;'><strong>نسبة الثقة:</strong> {orig_probs[orig_pred_idx]:.2%}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Machine Learning Model Results
        st.markdown("<div class='model-card model-card-ml'>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: right; margin: 0;'>نموذج التعلم الآلي (Machine Learning)</h3>", unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f"<div class='level-badge level-{ml_level}'>{level_colors.get(ml_level, '')} المستوى {ml_level}</div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<p style='font-size: 1.3em; margin-top: 15px;'><strong>{level_names.get(ml_level, 'غير معروف')}</strong></p>", unsafe_allow_html=True)
        
        st.progress(int(ml_probs[ml_pred_idx] * 100))
        st.markdown(f"<p style='text-align: right;'><strong>نسبة الثقة:</strong> {ml_probs[ml_pred_idx]:.2%}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Deep Learning Model Results
        st.markdown("<div class='model-card model-card-dl'>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: right; margin: 0;'>نموذج التعلم العميق (Deep Learning)</h3>", unsafe_allow_html=True)
        
        col5, col6 = st.columns(2)
        with col5:
            st.markdown(f"<div class='level-badge level-{dl_level}'>{level_colors.get(dl_level, '')} المستوى {dl_level}</div>", unsafe_allow_html=True)
        with col6:
            st.markdown(f"<p style='font-size: 1.3em; margin-top: 15px;'><strong>{level_names.get(dl_level, 'غير معروف')}</strong></p>", unsafe_allow_html=True)
        
        st.progress(int(dl_probs[dl_pred_idx] * 100))
        st.markdown(f"<p style='text-align: right;'><strong>نسبة الثقة:</strong> {dl_probs[dl_pred_idx]:.2%}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # -----------------------------------------
        # Processed Text Section
        # -----------------------------------------
        st.markdown("---")
        st.markdown("<h3 style='text-align: right; direction: rtl; color: #667eea;'>النص بعد المعالجة</h3>", unsafe_allow_html=True)
        st.markdown(f'<div class="rtl-text" style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 20px; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">{cleaned}</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #667eea;'>© 2025 — مشروع بَيِّنْ</p>", unsafe_allow_html=True)
