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
        mix_subfolder = "CAMeLBERTmix_D3Tok"
        models['mix_tokenizer'] = AutoTokenizer.from_pretrained(mix_repo, subfolder=mix_subfolder)
        models['mix_model'] = AutoModelForSequenceClassification.from_pretrained(mix_repo, subfolder=mix_subfolder)
    except Exception as e:
        st.error(f"خطأ في تحميل نموذج CAMeLBERTmix: {str(e)}")
        models['mix_tokenizer'] = None
        models['mix_model'] = None
    
    # Model 2: CAMeLBERTmsa_D3Tok
    try:
        msa_repo = "SarahAlhalees/CAMeLBERTmsa_D3Tok"
        msa_subfolder = "CAMeLBERTmsa_D3Tok"
        models['msa_tokenizer'] = AutoTokenizer.from_pretrained(msa_repo, subfolder=msa_subfolder)
        models['msa_model'] = AutoModelForSequenceClassification.from_pretrained(msa_repo, subfolder=msa_subfolder)
    except Exception as e:
        st.error(f"خطأ في تحميل نموذج CAMeLBERTmsa: {str(e)}")
        models['msa_tokenizer'] = None
        models['msa_model'] = None
    
    return models

models_dict = load_models()
orig_tokenizer = models_dict.get('orig_tokenizer')
orig_model = models_dict.get('orig_model')
mix_tokenizer = models_dict.get('mix_tokenizer')
mix_model = models_dict.get('mix_model')
msa_tokenizer = models_dict.get('msa_tokenizer')
msa_model = models_dict.get('msa_model')

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
    .model-card-mix {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .model-card-msa {
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
        # Check if at least one model is loaded
        if not any([orig_model, mix_model, msa_model]):
            st.error("لم يتم تحميل أي نموذج. الرجاء التحقق من الاتصال بالإنترنت والمحاولة مرة أخرى.")
        else:
            cleaned = normalize_ar(text)
        
            # Predict with Original Model (Arabertv2_D3Tok)
            orig_level = None
            orig_probs = None
            orig_pred_idx = None
            
            if orig_model and orig_tokenizer:
                try:
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
                except Exception as e:
                    st.warning(f"تعذر التنبؤ بالنموذج الأصلي: {str(e)}")
            
            # Predict with Model 1 (CAMeLBERTmix)
            mix_level = None
            mix_probs = None
            mix_pred_idx = None
            
            if mix_model and mix_tokenizer:
                try:
                    mix_inputs = mix_tokenizer(
                        cleaned,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=256
                    )
                    
                    with torch.no_grad():
                        mix_logits = mix_model(**mix_inputs).logits
                    
                    mix_probs = torch.softmax(mix_logits, dim=-1).numpy()[0]
                    mix_pred_idx = np.argmax(mix_probs)
                    mix_level = mix_pred_idx + 1
                except Exception as e:
                    st.warning(f"تعذر التنبؤ بنموذج CAMeLBERTmix: {str(e)}")
            
            # Predict with Model 2 (CAMeLBERTmsa)
            msa_level = None
            msa_probs = None
            msa_pred_idx = None
            
            if msa_model and msa_tokenizer:
                try:
                    msa_inputs = msa_tokenizer(
                        cleaned,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=256
                    )
                    
                    with torch.no_grad():
                        msa_logits = msa_model(**msa_inputs).logits
                    
                    msa_probs = torch.softmax(msa_logits, dim=-1).numpy()[0]
                    msa_pred_idx = np.argmax(msa_probs)
                    msa_level = msa_pred_idx + 1
                except Exception as e:
                    st.warning(f"تعذر التنبؤ بنموذج CAMeLBERTmsa: {str(e)}")

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
            if orig_level is not None:
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
            
            # CAMeLBERTmix Model Results
            if mix_level is not None:
                st.markdown("<div class='model-card model-card-mix'>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: right; margin: 0;'>نموذج CAMeLBERTmix</h3>", unsafe_allow_html=True)
                
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown(f"<div class='level-badge level-{mix_level}'>{level_colors.get(mix_level, '')} المستوى {mix_level}</div>", unsafe_allow_html=True)
                with col4:
                    st.markdown(f"<p style='font-size: 1.3em; margin-top: 15px;'><strong>{level_names.get(mix_level, 'غير معروف')}</strong></p>", unsafe_allow_html=True)
                
                st.progress(int(mix_probs[mix_pred_idx] * 100))
                st.markdown(f"<p style='text-align: right;'><strong>نسبة الثقة:</strong> {mix_probs[mix_pred_idx]:.2%}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # CAMeLBERTmsa Model Results
            if msa_level is not None:
                st.markdown("<div class='model-card model-card-msa'>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: right; margin: 0;'>نموذج CAMeLBERTmsa</h3>", unsafe_allow_html=True)
                
                col5, col6 = st.columns(2)
                with col5:
                    st.markdown(f"<div class='level-badge level-{msa_level}'>{level_colors.get(msa_level, '')} المستوى {msa_level}</div>", unsafe_allow_html=True)
                with col6:
                    st.markdown(f"<p style='font-size: 1.3em; margin-top: 15px;'><strong>{level_names.get(msa_level, 'غير معروف')}</strong></p>", unsafe_allow_html=True)
                
                st.progress(int(msa_probs[msa_pred_idx] * 100))
                st.markdown(f"<p style='text-align: right;'><strong>نسبة الثقة:</strong> {msa_probs[msa_pred_idx]:.2%}</p>", unsafe_allow_html=True)
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
