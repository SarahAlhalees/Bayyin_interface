import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import torch
import re

# -----------------------------------------
# Page Config
# -----------------------------------------
st.set_page_config(
    page_title="بَيِّنْ - تصنيف وتبسيط النصوص العربية",
    page_icon="📖",
    layout="centered"
)

# -----------------------------------------
# Arabic Text Normalization
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
def load_classification_model():
    try:
        repo_id = "SarahAlhalees/AraBERTv2_RefinedBayyin"
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForSequenceClassification.from_pretrained(repo_id)
        return tokenizer, model
    except Exception as e:
        st.error(f"خطأ في تحميل نموذج التصنيف: {str(e)}")
        return None, None

@st.cache_resource
def load_simplification_model():
    try:
        repo_id = "SarahAlhalees/bassit-simplifier"
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
        return tokenizer, model
    except Exception as e:
        st.warning(f"نموذج التبسيط غير متوفر: {str(e)}")
        return None, None

clf_tokenizer, clf_model = load_classification_model()
simplifier_tokenizer, simplifier_model = load_simplification_model()

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
.simplified-box {
    background-color: #1e3a2e;
    padding: 20px;
    border-radius: 10px;
    direction: rtl;
    text-align: right;
    color: #ffffff;
    border-right: 4px solid #4ade80;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align: center; direction: rtl;'>بَيِّنْ</h1>
<h3 style='text-align: center; direction: rtl;'>مصنف وتبسيط النصوص العربية</h3>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------
# Input
# -----------------------------------------
text = st.text_area("", height=200, placeholder="اكتب أو الصق النص هنا...")

# Session State
if 'done' not in st.session_state:
    st.session_state.done = False

# -----------------------------------------
# Classification
# -----------------------------------------
if st.button("📊 بَيِّنْ", use_container_width=True):
    if not text.strip():
        st.warning("⚠️ الرجاء إدخال نص.")
    elif not clf_model:
        st.error("⚠️ نموذج التصنيف غير متوفر.")
    else:
        with st.spinner("جاري التحليل..."):
            cleaned = normalize_ar(text)

            inputs = clf_tokenizer(
                cleaned,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256
            )

            with torch.no_grad():
                outputs = clf_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            prediction = torch.argmax(probs, dim=1).item() + 1
            confidence = probs[0][prediction - 1].item()

            st.session_state.done = True
            st.session_state.level = prediction
            st.session_state.conf = confidence
            st.session_state.original = text

# -----------------------------------------
# Results
# -----------------------------------------
if st.session_state.done:
    st.markdown("---")
    st.subheader("📊 نتيجة التصنيف")

    level = st.session_state.level

    level_colors = {1: "🟢", 2: "🟢", 3: "🟡", 4: "🟡", 5: "🔴", 6: "🔴"}
    level_names = {
        1: "سهل جداً",
        2: "سهل",
        3: "متوسط",
        4: "صعب قليلاً",
        5: "صعب",
        6: "صعب جداً"
    }

    col1, col2 = st.columns(2)
    col1.metric("المستوى", f"{level_colors.get(level)} {level}")
    col2.metric("الوصف", level_names.get(level))

    st.progress(int(st.session_state.conf * 100))
    st.write(f"نسبة الثقة: {st.session_state.conf:.2%}")

    # -----------------------------------------
    # Simplification
    # -----------------------------------------
    if level >= 4:
        st.markdown("---")
        st.info("💡 النص صعب، يمكنك تبسيطه.")

        if st.button("✨ بَسِّطْ", use_container_width=True):
            if simplifier_model and simplifier_tokenizer:
                with st.spinner("جاري التبسيط..."):
                    cleaned = normalize_ar(st.session_state.original)

                    inputs = simplifier_tokenizer(
                        cleaned,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=512
                    )

                    with torch.no_grad():
                        outputs = simplifier_model.generate(
                            **inputs,
                            max_length=512,
                            num_beams=4,
                            no_repeat_ngram_size=3
                        )

                    simplified = simplifier_tokenizer.decode(outputs[0], skip_special_tokens=True)

                    st.subheader("✨ النص المبسط")
                    st.markdown(f'<div class="simplified-box">{simplified}</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ نموذج التبسيط غير متوفر.")

st.markdown("---")
st.caption("© 2025 — مشروع بَيِّنْ")
