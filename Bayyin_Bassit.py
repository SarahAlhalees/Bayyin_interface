import streamlit as st
import torch
import numpy as np
import joblib
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from huggingface_hub import hf_hub_download

# -----------------------------------------
# Page config
# -----------------------------------------
st.set_page_config(
    page_title="بَيِّنْ - مصنف وتبسيط النصوص العربية",
    page_icon="📖",
    layout="centered"
)

# -----------------------------------------
# Arabic normalization
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
    return re.sub(r"\s+", " ", text).strip()

# -----------------------------------------
# Load models
# -----------------------------------------
@st.cache_resource
def load_models():
    models = {}

    # AraBERT
    models['arabert_tokenizer'] = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
    models['arabert_model'] = AutoModelForSequenceClassification.from_pretrained("SarahAlhalees/Arabertv2_D3Tok")

    # CAMeLBERT MSA
    models['camel_tokenizer'] = AutoTokenizer.from_pretrained("SarahAlhalees/CAMeLBERTmsa_D3Tok")
    models['camel_model'] = AutoModelForSequenceClassification.from_pretrained("SarahAlhalees/CAMeLBERTmsa_D3Tok")

    # BiLSTM (expects embeddings)
    bilstm_path = hf_hub_download(
        repo_id="Raya-y/Bayyin_models",
        filename="bilstm_arabert_bayyin.joblib"
    )
    models['bilstm'] = joblib.load(bilstm_path)

    models['bilstm_tokenizer'] = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
    models['bilstm_bert'] = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv2")

    # Meta learner
    meta_path = hf_hub_download(
        repo_id="SarahAlhalees/ensemble",
        filename="meta_svm_tuned.joblib"
    )
    models['meta'] = joblib.load(meta_path)

    # OPTIONAL scaler (only if you saved it separately)
    try:
        scaler_path = hf_hub_download(
            repo_id="SarahAlhalees/ensemble",
            filename="scaler.joblib"
        )
        models['scaler'] = joblib.load(scaler_path)
    except:
        models['scaler'] = None

    return models

models = load_models()

# -----------------------------------------
# Helper: get probabilities from transformer
# -----------------------------------------
def get_probs(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return probs  # shape (6,)

# -----------------------------------------
# Helper: BiLSTM probabilities
# -----------------------------------------
def get_bilstm_probs(model, tokenizer, bert_model, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = bert_model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1].mean(dim=1)

    probs = model.predict_proba(embeddings.numpy())[0]  # (6,)
    return probs

# -----------------------------------------
# UI
# -----------------------------------------
st.title("بَيِّنْ")
st.write("تصنيف مستوى قراءة النصوص العربية باستخدام نموذج تجميعي (Ensemble)")

text = st.text_area("أدخل النص", height=200)

# -----------------------------------------
# Prediction
# -----------------------------------------
if st.button("📊 بَيِّنْ"):

    if not text.strip():
        st.warning("⚠️ أدخل نص أولاً")
    else:
        cleaned = normalize_ar(text)

        with st.spinner("جاري التحليل..."):

            # 1) Base model probabilities
            p_arabert = get_probs(models['arabert_model'], models['arabert_tokenizer'], cleaned)
            p_camel = get_probs(models['camel_model'], models['camel_tokenizer'], cleaned)
            p_bilstm = get_bilstm_probs(
                models['bilstm'],
                models['bilstm_tokenizer'],
                models['bilstm_bert'],
                cleaned
            )

            # 2) Build 18-dim vector
            meta_input = np.concatenate([p_arabert, p_camel, p_bilstm]).reshape(1, -1)

            # 3) Scale if needed
            if models['scaler'] is not None:
                meta_input = models['scaler'].transform(meta_input)

            # 4) Final prediction
            prediction = models['meta'].predict(meta_input)[0]

            try:
                probs = models['meta'].predict_proba(meta_input)[0]
                confidence = np.max(probs)
            except:
                confidence = None

        # -----------------------------------------
        # Display
        # -----------------------------------------
        level_names = {
            1: "سهل جداً", 2: "سهل", 3: "متوسط",
            4: "صعب قليلاً", 5: "صعب", 6: "صعب جداً"
        }

        st.success(f"📊 المستوى: {prediction}")
        st.write(f"الوصف: {level_names.get(prediction, '')}")

        if confidence:
            st.write(f"نسبة الثقة: {confidence:.2%}")
