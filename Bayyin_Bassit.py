import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import joblib
import torch
import numpy as np
import re

# -----------------------------------------
# Streamlit Page Settings
# -----------------------------------------
st.set_page_config(
    page_title="بَيِّنْ - مصنف وتبسيط النصوص العربية",
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
# Load Models
# -----------------------------------------
@st.cache_resource
def load_classification_model():
    """Load the classification model (joblib)"""
    # Option 1: If using joblib model from HuggingFace
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(
        repo_id="SarahAlhalees/ensemble",  # Your classifier repo
        filename="meta_svm_tuned.joblib"  # Your joblib model file
    )
    classifier = joblib.load(model_path)
    return classifier

@st.cache_resource
def load_simplification_model():
    """Load the text simplification model (generative)"""
    repo_id = "SarahAlhalees/bassit-simplifier"  # Your simplification model repo
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
    return tokenizer, model

# Load models
try:
    classifier = load_classification_model()
    simplifier_tokenizer, simplifier_model = load_simplification_model()
    models_loaded = True
except Exception as e:
    st.error(f"خطأ في تحميل النماذج: {str(e)}")
    models_loaded = False

# -----------------------------------------
# UI Layout
# -----------------------------------------
st.markdown("""
    <h1 style='text-align: center; direction: rtl;'>بَيِّنْ</h1>
    <h3 style='text-align: center; direction: rtl;'>مصنف وتبسيط النصوص العربية</h3>
""", unsafe_allow_html=True)

st.markdown("---")

# Input text
text = st.text_area(
    label="",
    height=200,
    placeholder="اكتب أو الصق النص هنا...",
    key="arabic_input"
)

# Add RTL styling
st.markdown("""
    <style>
    textarea {
        direction: rtl;
        text-align: right;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for storing results
if 'classification_done' not in st.session_state:
    st.session_state.classification_done = False
if 'readability_level' not in st.session_state:
    st.session_state.readability_level = 0
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.0
if 'original_text' not in st.session_state:
    st.session_state.original_text = ""

# -----------------------------------------
# Classification Button: "بَيِّنْ"
# -----------------------------------------
if st.button("📊 بَيِّنْ", use_container_width=True, type="primary"):
    if not text.strip():
        st.warning("⚠️ الرجاء إدخال نص.")
    elif not models_loaded:
        st.error("⚠️ لم يتم تحميل النماذج بشكل صحيح.")
    else:
        with st.spinner("جاري التحليل..."):
            # Normalize text
            cleaned = normalize_ar(text)
            
            # Classify using joblib model
            # Adjust this based on your model's input format
            prediction = classifier.predict([cleaned])[0]
            
            # If your model outputs probabilities
            try:
                probs = classifier.predict_proba([cleaned])[0]
                confidence = np.max(probs)
            except:
                confidence = 1.0
            
            # Store in session state
            st.session_state.classification_done = True
            st.session_state.readability_level = prediction
            st.session_state.confidence = confidence
            st.session_state.original_text = text

# -----------------------------------------
# Display Classification Results
# -----------------------------------------
if st.session_state.classification_done:
    st.markdown("---")
    st.subheader("📊 نتيجة التصنيف")
    
    level = st.session_state.readability_level
    
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
    
    st.progress(int(st.session_state.confidence * 100))
    st.write(f"**نسبة الثقة:** {st.session_state.confidence:.2%}")
    
    # -----------------------------------------
    # Simplification Button: "بَسِّطْ" (only for levels 4-6)
    # -----------------------------------------
    if level >= 4:
        st.markdown("---")
        st.info("💡 هذا النص صعب القراءة. يمكنك تبسيطه بالضغط على الزر أدناه.")
        
        if st.button("✨ بَسِّطْ", use_container_width=True, type="secondary"):
            with st.spinner("جاري التبسيط..."):
                # Normalize text for simplification
                cleaned = normalize_ar(st.session_state.original_text)
                
                # Tokenize for simplification model
                inputs = simplifier_tokenizer(
                    cleaned,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                
                # Generate simplified text
                with torch.no_grad():
                    outputs = simplifier_model.generate(
                        **inputs,
                        max_length=512,
                        num_beams=5,
                        early_stopping=True
                    )
                
                simplified_text = simplifier_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Display simplified text
                st.markdown("---")
                st.subheader("✨ النص المبسط")
                st.markdown(f"""
                    <div style='background-color: #1e3a2e; padding: 20px; border-radius: 10px; 
                                direction: rtl; text-align: right; color: #ffffff; 
                                border-right: 4px solid #4ade80;'>
                        {simplified_text}
                    </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("© 2025 — مشروع بَيِّنْ")
