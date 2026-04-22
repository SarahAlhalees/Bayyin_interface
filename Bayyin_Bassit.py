import os
import warnings
import logging
import sys
import types
# Suppress transformers __path__ scanning noise and deprecation warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModel
import torch
import torch.nn as nn
import numpy as np
import re
import joblib
from huggingface_hub import hf_hub_download


# -----------------------------------------
# BiLSTM Model Class Definition (MUST BE BEFORE LOADING)
# -----------------------------------------
class BiLSTMWithMeta(nn.Module):
    def __init__(self, input_dim, categorical_cardinalities, num_numeric,
                 lstm_hidden=256, meta_proj_dim=128, num_classes=6, dropout=0.3,
                 use_bert=False, bert_model_name=None):
        super().__init__()
        self.use_bert = use_bert
        if use_bert and bert_model_name:
            from transformers import AutoModel
            self.bert = AutoModel.from_pretrained(bert_model_name)
            input_dim = self.bert.config.hidden_size
        else:
            self.bert = None

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden,
                            num_layers=1, batch_first=True, bidirectional=True)

        self.cat_names = list(categorical_cardinalities.keys())
        self.cat_embeddings = nn.ModuleDict()
        total_cat_emb_dim = 0
        for name, card in categorical_cardinalities.items():
            emb_dim = min(50, max(4, int(card**0.5)))
            self.cat_embeddings[name] = nn.Embedding(card, emb_dim)
            total_cat_emb_dim += emb_dim

        self.meta_proj = nn.Sequential(
            nn.Linear(total_cat_emb_dim + num_numeric, meta_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2 + meta_proj_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, numeric_meta, categorical_meta, attention_mask=None):
        if self.use_bert and self.bert is not None:
            bert_out = self.bert(input_ids=x, attention_mask=attention_mask)
            x = bert_out.last_hidden_state
        else:
            if len(x.shape) == 2:
                x = x.unsqueeze(1)

        lstm_out, _ = self.lstm(x)
        pooled = lstm_out.mean(dim=1) if self.use_bert else lstm_out.squeeze(1)

        cat_embs = [self.cat_embeddings[name](categorical_meta[:, i])
                    for i, name in enumerate(self.cat_names)]
        cat_concat = torch.cat(cat_embs, dim=1) if cat_embs else \
                     torch.zeros(numeric_meta.size(0), 0, device=numeric_meta.device)

        meta_concat = torch.cat([numeric_meta, cat_concat], dim=1)
        meta_vec = self.meta_proj(meta_concat)

        fused = torch.cat([pooled, meta_vec], dim=1)
        fused = self.dropout(fused)
        return self.classifier(fused)


class BiLSTMWrapper:
    def __init__(self, model, cat_cardinalities, num_numeric, num_classes=6, device='cpu'):
        self.device = device
        self.num_classes = num_classes
        self.cat_cardinalities = cat_cardinalities
        self.num_numeric = num_numeric
        self.model = model
        self.default_numeric = np.zeros(num_numeric, dtype=np.float32)
        self.default_categorical = np.zeros(len(cat_cardinalities), dtype=np.int64)

    def predict(self, X):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
            if len(X.shape) == 1:
                X = X.unsqueeze(0)

            batch_size = X.shape[0]
            numeric = torch.tensor(np.tile(self.default_numeric, (batch_size, 1)), dtype=torch.float32).to(self.device)
            categorical = torch.tensor(np.tile(self.default_categorical, (batch_size, 1)), dtype=torch.long).to(self.device)

            logits = self.model(X.to(self.device), numeric, categorical)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            return predictions + 1

    def predict_proba(self, X):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
            if len(X.shape) == 1:
                X = X.unsqueeze(0)

            batch_size = X.shape[0]
            numeric = torch.tensor(np.tile(self.default_numeric, (batch_size, 1)), dtype=torch.float32).to(self.device)
            categorical = torch.tensor(np.tile(self.default_categorical, (batch_size, 1)), dtype=torch.long).to(self.device)

            logits = self.model(X.to(self.device), numeric, categorical)
            return torch.softmax(logits, dim=1).cpu().numpy()


# Register classes for joblib unpickling
current_module = sys.modules[__name__]
current_module.BiLSTMWrapper = BiLSTMWrapper
current_module.BiLSTMWithMeta = BiLSTMWithMeta

if '__main__' not in sys.modules:
    sys.modules['__main__'] = types.ModuleType('__main__')
sys.modules['__main__'].BiLSTMWrapper = BiLSTMWrapper
sys.modules['__main__'].BiLSTMWithMeta = BiLSTMWithMeta

if 'main' not in sys.modules:
    sys.modules['main'] = current_module
else:
    sys.modules['main'].BiLSTMWrapper = BiLSTMWrapper
    sys.modules['main'].BiLSTMWithMeta = BiLSTMWithMeta

# -----------------------------------------
# Streamlit Page Settings
# -----------------------------------------
st.set_page_config(
    page_title="بَيِّنْ - تصنيف وتبسيط النصوص العربية",
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
def load_base_models():
    """
    Load the 3 base models for stacking.

    Stage 1 architecture (per paper):
      Each base model = Transformer + 2-layer BiLSTM + attention pooling
                        + linguistic features → classification head (6 classes)

    Stage 2 meta-learner input (per paper):
      18-dim vector = concat of 6 softmax probs from each of the 3 base models.
      x = [p_AraBERT | p_CamelBERT-MSA | p_BiLSTM]  ∈ R^18

    The BiLSTM base model uses CamelBERT-MSA as its transformer backbone.
    We load:
      • best_bilstm_camelbert.joblib  — BiLSTMWrapper (predict_proba returns 6 probs)
      • meta_encoders.joblib          — sklearn scaler/encoder fitted on the CLS
                                        embeddings; transforms raw CLS → input for
                                        the BiLSTMWrapper
      • CamelBERT-MSA base model      — extracts CLS token representations
    """
    models = {}

    # --- AraBERT v2 (fine-tuned, has classification head) ---
    # AraBERT uses SentencePiece; use_fast=False avoids the fast-tokenizer
    # conversion error when the sentencepiece wheel is not pre-installed.
    try:
        # Try flat repo layout first, then subfolder layout as fallback.
        try:
            models['arabert_tokenizer'] = AutoTokenizer.from_pretrained(
                "SarahAlhalees/AraBERTv2_RefinedBayyin",
                use_fast=False
            )
            models['arabert_model'] = AutoModelForSequenceClassification.from_pretrained(
                "SarahAlhalees/AraBERTv2_RefinedBayyin"
            )
        except Exception:
            models['arabert_tokenizer'] = AutoTokenizer.from_pretrained(
                "SarahAlhalees/AraBERTv2_RefinedBayyin",
                subfolder="Arabertv2_D3Tok",
                use_fast=False
            )
            models['arabert_model'] = AutoModelForSequenceClassification.from_pretrained(
                "SarahAlhalees/AraBERTv2_RefinedBayyin",
                subfolder="Arabertv2_D3Tok"
            )
        models['arabert_model'].eval()
    except Exception as e:
        st.warning(f"خطأ في تحميل AraBERT: {str(e)}")
        models['arabert_tokenizer'] = None
        models['arabert_model'] = None

    # --- CamelBERT-MSA (fine-tuned, has classification head) ---
    # CamelBERT also uses WordPiece / SentencePiece; use_fast=False is safe here.
    try:
        models['camelbert_tokenizer'] = AutoTokenizer.from_pretrained(
            "SarahAlhalees/CamelBERTmsa_RefinedBayyin",
            use_fast=False
        )
        models['camelbert_model'] = AutoModelForSequenceClassification.from_pretrained(
            "SarahAlhalees/CamelBERTmsa_RefinedBayyin"
        )
        models['camelbert_model'].eval()
    except Exception as e:
        st.warning(f"خطأ في تحميل CamelBERT: {str(e)}")
        models['camelbert_tokenizer'] = None
        models['camelbert_model'] = None

    # --- BiLSTM + CamelBERT-MSA backbone ---
    try:
        bilstm_path = hf_hub_download(
            repo_id="SarahAlhalees/BiLSTM_RefinedBayyin",
            filename="best_bilstm_camelbert.joblib"
        )
        meta_enc_path = hf_hub_download(
            repo_id="SarahAlhalees/BiLSTM_RefinedBayyin",
            filename="meta_encoders.joblib"
        )

        # best_bilstm_camelbert.joblib may be a dict containing the wrapper
        # and/or encoders. Unwrap gracefully.
        bilstm_raw = joblib.load(bilstm_path)
        if isinstance(bilstm_raw, dict):
            # Common key names the training script might have used
            bilstm_wrapper = (
                bilstm_raw.get('model') or
                bilstm_raw.get('bilstm') or
                bilstm_raw.get('wrapper') or
                bilstm_raw.get('classifier') or
                next((v for v in bilstm_raw.values()
                      if hasattr(v, 'predict_proba')), None)
            )
            models['bilstm_model'] = bilstm_wrapper
            # Log available keys to help debug if still None
            if bilstm_wrapper is None:
                st.warning(f"BiLSTM joblib keys: {list(bilstm_raw.keys())}")
        else:
            models['bilstm_model'] = bilstm_raw

        # meta_encoders.joblib may also be a dict
        meta_raw = joblib.load(meta_enc_path)
        if isinstance(meta_raw, dict):
            # Try to find an encoder/scaler/pipeline inside the dict
            # Also check if the BiLSTM model is stored here instead
            if models['bilstm_model'] is None:
                models['bilstm_model'] = (
                    meta_raw.get('model') or
                    meta_raw.get('bilstm') or
                    meta_raw.get('wrapper') or
                    meta_raw.get('classifier') or
                    next((v for v in meta_raw.values()
                          if hasattr(v, 'predict_proba')), None)
                )
            encoder = (
                meta_raw.get('encoder') or
                meta_raw.get('scaler') or
                meta_raw.get('label_encoder') or
                meta_raw.get('pca') or
                meta_raw.get('pipeline') or
                next((v for v in meta_raw.values()
                      if hasattr(v, 'transform')), None)
            )
            models['meta_encoders'] = encoder
            models['meta_encoders_dict'] = meta_raw  # keep full dict for debug
        else:
            models['meta_encoders'] = meta_raw
            models['meta_encoders_dict'] = None

        # CamelBERT-MSA base (no classification head) — supplies CLS embeddings
        # that meta_encoders projects before passing to BiLSTMWrapper.
        models['bilstm_tokenizer'] = AutoTokenizer.from_pretrained(
            "CAMeL-Lab/bert-base-arabic-camelbert-msa",
            use_fast=False
        )
        models['bilstm_bert'] = AutoModel.from_pretrained(
            "CAMeL-Lab/bert-base-arabic-camelbert-msa"
        )
        models['bilstm_bert'].eval()
    except Exception as e:
        st.warning(f"خطأ في تحميل BiLSTM: {str(e)}")
        models['bilstm_model']        = None
        models['bilstm_tokenizer']    = None
        models['bilstm_bert']         = None
        models['meta_encoders']       = None
        models['meta_encoders_dict']  = None

    return models


@st.cache_resource
def load_meta_learner():
    """Load stacking meta-learner (SVM / LR / MLP trained on Stage-2 features)"""
    try:
        meta_path = hf_hub_download(
            repo_id="SarahAlhalees/ensemble",
            filename="meta_svm_tuned.joblib"
        )
        return joblib.load(meta_path)
    except Exception as e:
        st.error(f"خطأ في تحميل Meta-Learner: {str(e)}")
        return None


@st.cache_resource
def load_simplification_model():
    """Load AraBART text simplification model"""
    try:
        repo_id = "SarahAlhalees/bassit-simplifier"
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
        return tokenizer, model
    except Exception as e:
        st.warning(f"نموذج التبسيط غير متوفر: {str(e)}")
        return None, None


# Load all models at startup
base_models = load_base_models()
meta_learner = load_meta_learner()
simplifier_tokenizer, simplifier_model = load_simplification_model()

# --- Debug expander: shows joblib dict keys to help diagnose loading issues ---
with st.expander("🔧 تشخيص تحميل النماذج", expanded=False):
    bm = base_models.get('bilstm_model')
    me = base_models.get('meta_encoders')
    md = base_models.get('meta_encoders_dict')
    st.write(f"**bilstm_model type:** `{type(bm)}`")
    if isinstance(bm, dict):
        st.write(f"**bilstm_model keys:** `{list(bm.keys())}`")
    st.write(f"**meta_encoders type:** `{type(me)}`")
    if isinstance(md, dict):
        st.write(f"**meta_encoders_dict keys:** `{list(md.keys())}`")
    st.write(f"**arabert loaded:** `{base_models.get('arabert_model') is not None}`")
    st.write(f"**camelbert loaded:** `{base_models.get('camelbert_model') is not None}`")
    st.write(f"**meta_learner loaded:** `{meta_learner is not None}`")

# -----------------------------------------
# Feature Extraction
# -----------------------------------------
def get_softmax_probs_from_classifier(text, tokenizer, clf_model, max_length=256):
    """
    Return 6-class softmax probabilities from a fine-tuned classifier model.
    Shape: (1, 6)
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    )
    with torch.no_grad():
        logits = clf_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).numpy()   # (1, 6)
    return probs


def get_bilstm_probs(text, models, max_length=256):
    """
    Get 6-class softmax probabilities from the BiLSTM+CamelBERT-MSA base model.

    Pipeline (matches paper Stage 1 BiLSTM branch):
      1. Tokenise with CamelBERT-MSA tokenizer.
      2. Extract CLS token embedding from the CamelBERT-MSA base model.
      3. Apply meta_encoders (sklearn scaler / PCA / pipeline) to project the
         CLS embedding into the space the BiLSTMWrapper was trained on.
      4. Call BiLSTMWrapper.predict_proba() → (1, 6) softmax probs.

    Shape returned: (1, 6)
    """
    tokenizer  = models['bilstm_tokenizer']
    bert_model = models['bilstm_bert']
    bilstm     = models['bilstm_model']
    meta_enc   = models['meta_encoders']

    # Safety check — if loading produced a dict instead of a wrapper, try to
    # extract the predict_proba-capable object one more time at inference time.
    if isinstance(bilstm, dict):
        bilstm = (
            bilstm.get('model') or
            bilstm.get('bilstm') or
            bilstm.get('wrapper') or
            bilstm.get('classifier') or
            next((v for v in bilstm.values()
                  if hasattr(v, 'predict_proba')), None)
        )
    if bilstm is None or not hasattr(bilstm, 'predict_proba'):
        # Last resort: check meta_encoders_dict
        meta_dict = models.get('meta_encoders_dict') or {}
        bilstm = next(
            (v for v in meta_dict.values() if hasattr(v, 'predict_proba')), None
        )
    if bilstm is None:
        raise ValueError(
            "BiLSTM wrapper with predict_proba not found in loaded joblib files. "
            f"bilstm_model type={type(models['bilstm_model'])}, "
            f"meta_encoders_dict keys={list((models.get('meta_encoders_dict') or {}).keys())}"
        )

    # Step 1 & 2 — CLS embedding
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_emb = outputs.last_hidden_state[:, 0, :].numpy()   # (1, hidden_size)

    # Step 3 — encode / project via meta_encoders if available
    if meta_enc is not None:
        if hasattr(meta_enc, 'transform'):
            cls_emb = meta_enc.transform(cls_emb)
        elif isinstance(meta_enc, (list, tuple)):
            for enc in meta_enc:
                if hasattr(enc, 'transform'):
                    cls_emb = enc.transform(cls_emb)

    # Step 4 — BiLSTM softmax probs
    probs = bilstm.predict_proba(cls_emb)   # (1, 6)
    return probs


def get_model_probabilities(text, models):
    """
    Build the 18-dim Stage-2 feature vector for the SVM meta-learner.

    Per paper:
        x = [p_AraBERT | p_CamelBERT-MSA | p_BiLSTM]  ∈ R^18
        (6 softmax probs per base model × 3 models)

    No CLS embeddings are included here — that is specific to the *hybrid*
    BiLSTM Stage 1 pipeline, not the SVM meta-learner's input.
    """
    feature_parts = []

    # ---- 1. AraBERT v2 (6 probs) ----
    if models.get('arabert_model') and models.get('arabert_tokenizer'):
        try:
            probs = get_softmax_probs_from_classifier(
                text, models['arabert_tokenizer'], models['arabert_model']
            )   # (1, 6)
        except Exception as e:
            st.warning(f"AraBERT inference error: {e}")
            probs = np.zeros((1, 6))
    else:
        probs = np.zeros((1, 6))
    feature_parts.append(probs)

    # ---- 2. CamelBERT-MSA fine-tuned classifier (6 probs) ----
    if models.get('camelbert_model') and models.get('camelbert_tokenizer'):
        try:
            probs = get_softmax_probs_from_classifier(
                text, models['camelbert_tokenizer'], models['camelbert_model']
            )   # (1, 6)
        except Exception as e:
            st.warning(f"CamelBERT inference error: {e}")
            probs = np.zeros((1, 6))
    else:
        probs = np.zeros((1, 6))
    feature_parts.append(probs)

    # ---- 3. BiLSTM + CamelBERT-MSA backbone (6 probs) ----
    if (models.get('bilstm_model') and models.get('bilstm_tokenizer')
            and models.get('bilstm_bert')):
        try:
            probs = get_bilstm_probs(text, models)   # (1, 6)
        except Exception as e:
            st.warning(f"BiLSTM inference error: {e}")
            probs = np.zeros((1, 6))
    else:
        probs = np.zeros((1, 6))
    feature_parts.append(probs)

    # Concatenate → (1, 18)
    feature_vector = np.concatenate(feature_parts, axis=1)
    return feature_vector


# -----------------------------------------
# UI Layout
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

text = st.text_area(
    label="",
    height=200,
    placeholder="اكتب أو الصق النص هنا...",
    key="arabic_input"
)

# Session state
if 'classification_done' not in st.session_state:
    st.session_state.classification_done = False
if 'readability_level' not in st.session_state:
    st.session_state.readability_level = 0
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.0
if 'original_text' not in st.session_state:
    st.session_state.original_text = ""

# -----------------------------------------
# بَيِّنْ Button
# -----------------------------------------
if st.button("📊 بَيِّنْ", use_container_width=True, type="primary"):
    if not text.strip():
        st.warning("⚠️ الرجاء إدخال نص.")
    elif not meta_learner:
        st.error("⚠️ لم يتم تحميل النموذج.")
    else:
        with st.spinner("جاري التحليل..."):
            cleaned = normalize_ar(text)

            # Build Stage-2 feature vector (softmax probs + CLS embeddings)
            features = get_model_probabilities(cleaned, base_models)

            # Predict with meta-learner
            prediction = meta_learner.predict(features)[0]

            try:
                probs = meta_learner.predict_proba(features)[0]
                confidence = probs[prediction - 1] if prediction <= len(probs) else np.max(probs)
            except Exception:
                confidence = 1.0

            st.session_state.classification_done = True
            st.session_state.readability_level = prediction
            st.session_state.confidence = confidence
            st.session_state.original_text = text

# -----------------------------------------
# Display Results
# -----------------------------------------
if st.session_state.classification_done:
    st.markdown("---")
    st.subheader("📊 نتيجة التصنيف")

    level = st.session_state.readability_level
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
    with col1:
        st.metric(label="المستوى", value=f"{level_colors.get(level, '⚪')} {level}")
    with col2:
        st.metric(label="الوصف", value=level_names.get(level, "غير معروف"))

    st.progress(int(st.session_state.confidence * 100))
    st.write(f"**نسبة الثقة:** {st.session_state.confidence:.2%}")

    # -----------------------------------------
    # بَسِّطْ Button (for levels 4–6)
    # -----------------------------------------
    if level >= 4:
        st.markdown("---")
        st.info("💡 هذا النص صعب القراءة. يمكنك تبسيطه بالضغط على الزر أدناه.")

        if st.button("✨ بَسِّطْ", use_container_width=True, type="secondary"):
            if simplifier_model and simplifier_tokenizer:
                with st.spinner("جاري التبسيط..."):
                    cleaned = normalize_ar(st.session_state.original_text)

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
                            length_penalty=1.0,
                            early_stopping=True,
                            no_repeat_ngram_size=3
                        )

                    simplified_text = simplifier_tokenizer.decode(
                        outputs[0], skip_special_tokens=True
                    )

                    st.markdown("---")
                    st.subheader("✨ النص المبسط")
                    st.markdown(
                        f'<div class="simplified-box">{simplified_text}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.warning("⚠️ نموذج التبسيط غير متوفر حالياً.")

st.markdown("---")
st.caption("© 2025 — مشروع بَيِّنْ")
