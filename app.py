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
    page_icon="📖",
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
# Load Model
# -----------------------------------------
@st.cache_resource
def load_model():
    repo_id = "SarahAlhalees/Arabertv2_D3Tok"
    subfolder = "Arabertv2_D3Tok"
    tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, subfolder=subfolder)
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------------------
# UI Layout
# -----------------------------------------
st.markdown("""
    <h1 style='text-align: center; direction: rtl;'>بَيِّنْ</h1>
    <h3 style='text-align: center; direction: rtl;'>مصنف مستوى قراءة النصوص العربية</h3>
""", unsafe_allow_html=True)

st.markdown("---")

text = st.text_area(
    label="",
    height=200,
    placeholder="اكتب أو الصق النص هنا...",
    key="arabic_input"
)

# Add RTL styling for the text area
st.markdown("""
    <style>
    textarea {
        direction: rtl;
        text-align: right;
        font-size: 16px;
    }
    .token-box {
        display: inline-block;
        background-color: #e3f2fd;
        border: 1px solid #90caf9;
        border-radius: 4px;
        padding: 4px 8px;
        margin: 2px;
        font-family: monospace;
        direction: rtl;
    }
    .rtl-text {
        direction: rtl;
        text-align: right;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------
# Prediction
# -----------------------------------------
if st.button("🔍 تصنيف النص", use_container_width=True):
    if not text.strip():
        st.warning("⚠️ الرجاء إدخال نص.")
    else:
        cleaned = normalize_ar(text)
        
        inputs = tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        
        with torch.no_grad():
            logits = model(**inputs).logits
        
        probs = torch.softmax(logits, dim=-1).numpy()[0]
        pred_idx = np.argmax(probs)
        level = pred_idx + 1

        # -----------------------------------------
        # Results Section
        # -----------------------------------------
        st.markdown("---")
        st.subheader("📊 نتيجة التصنيف")
        
        # Level display with color coding
        level_colors = {1: "🟢", 2: "🟢", 3: "🟡", 4: "🟡", 5: "🔴", 6: "🔴"}
        level_names = {
            1: "سهل جداً", 2: "سهل", 3: "متوسط", 
            4: "صعب قليلاً", 5: "صعب", 6: "صعب جداً"
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="المستوى", value=f"{level_colors.get(level, '⚪')} {level}")
        with col2:
            st.metric(label="الوصف", value=level_names.get(level, "غير معروف"))
        
        st.progress(int(probs[pred_idx] * 100))
        st.write(f"**نسبة الثقة:** {probs[pred_idx]:.2%}")

        # -----------------------------------------
        # Tokenization Section
        # -----------------------------------------
        st.markdown("---")
        st.subheader("🔤 تحليل التوكنات (Tokenization)")
        
        # Get tokens
        token_ids = inputs["input_ids"][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        
        # Filter out special tokens for display
        special_tokens = [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]
        display_tokens = [(tok, tid) for tok, tid in zip(tokens, token_ids) if tok not in special_tokens]
        
        # Token statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("عدد التوكنات", len(display_tokens))
        with col2:
            st.metric("عدد التوكنات (مع الخاصة)", len(tokens))
        with col3:
            word_count = len(cleaned.split())
            st.metric("عدد الكلمات", word_count)
        
        # Display tokens visually
        st.write("**التوكنات:**")
        token_html = '<div class="rtl-text" style="line-height: 2.5;">'
        for tok, tid in display_tokens:
            # Clean token display (remove ## prefix for subwords)
            display_tok = tok.replace("##", "")
            token_html += f'<span class="token-box" title="ID: {tid}">{display_tok}</span>'
        token_html += '</div>'
        st.markdown(token_html, unsafe_allow_html=True)
        
        # Expandable section for detailed token info
        with st.expander("📋 عرض تفاصيل التوكنات"):
            import pandas as pd
            token_data = {
                "التوكن": [tok for tok, _ in display_tokens],
                "Token ID": [tid for _, tid in display_tokens],
                "نوع": ["جزء من كلمة" if tok.startswith("##") else "كلمة/بداية" for tok, _ in display_tokens]
            }
            df = pd.DataFrame(token_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # -----------------------------------------
        # Processed Text Section
        # -----------------------------------------
        st.markdown("---")
        st.subheader("🔧 النص بعد المعالجة")
        st.markdown(f'<div class="rtl-text" style="background-color: #f5f5f5; padding: 15px; border-radius: 8px;">{cleaned}</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("© 2025 — مشروع بَيِّنْ")
