import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import re

# -----------------------------------------
# Streamlit Page Settings
# -----------------------------------------
st.set_page_config(
    page_title="بَيِّنْ",
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
    repo_id = "SarahAlhalees/Arabertv2_D3Tok"  # Just username/repo_name
    subfolder = "Arabertv2_D3Tok"  # The folder inside the repo
    tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, subfolder=subfolder)
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------------------
# UI Layout
# -----------------------------------------
st.title("بَيِّنْ")
st.markdown("""
### ✨ أدخل النص ليتم تحديد مستوى سهولة قراءته 
""")
text = st.text_area(
    "أدخل النص هنا:",
    height=200,
    placeholder="اكتب هنا النص المراد تقييم سهولة قراءته..."
)

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
        # Map to levels 1-6
        level = pred_idx + 1  # 0 → 1, 5 → 6

        st.success(f"🔹 مستوى سهولة القراءة: **المستوى {level}**")
        st.progress(int(probs[pred_idx] * 100))
        st.write(f"نسبة الثقة: {probs[pred_idx]:.2%}")

        st.subheader("🔧 النص بعد المعالجة:")
        st.write(cleaned)

# Footer
st.caption("© 2025 — مشروع بَيِّنْ ")










