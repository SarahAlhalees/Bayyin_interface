import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModel
import torch
import torch.nn as nn
import numpy as np
import re
import joblib
from huggingface_hub import hf_hub_download
import sys
import types

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


# Register classes for joblib
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
    """Load the 3 base models for stacking"""
    models = {}
    
    # AraBERT v2
    try:
        models['arabert_tokenizer'] = AutoTokenizer.from_pretrained("SarahAlhalees/Arabertv2_D3Tok", subfolder="Arabertv2_D3Tok")
        models['arabert_model'] = AutoModelForSequenceClassification.from_pretrained("SarahAlhalees/Arabertv2_D3Tok", subfolder="Arabertv2_D3Tok")
    except Exception as e:
        st.warning(f"خطأ في تحميل AraBERT: {str(e)}")
        models['arabert_tokenizer'] = None
        models['arabert_model'] = None
    
    # CamelBERT MSA
    try:
        models['camelbert_tokenizer'] = AutoTokenizer.from_pretrained("SarahAlhalees/CAMeLBERTmsa_D3Tok")
        models['camelbert_model'] = AutoModelForSequenceClassification.from_pretrained("SarahAlhalees/CAMeLBERTmsa_D3Tok")
    except Exception as e:
        st.warning(f"خطأ في تحميل CamelBERT: {str(e)}")
        models['camelbert_tokenizer'] = None
        models['camelbert_model'] = None
    
    # BiLSTM
    try:
        bilstm_path = hf_hub_download(repo_id="Raya-y/Bayyin_models", filename="bilstm_arabert_bayyin.joblib")
        models['bilstm_model'] = joblib.load(bilstm_path)
        models['bilstm_tokenizer'] = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
        models['bilstm_bert'] = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv2")
    except Exception as e:
        st.warning(f"خطأ في تحميل BiLSTM: {str(e)}")
        models['bilstm_model'] = None
        models['bilstm_tokenizer'] = None
        models['bilstm_bert'] = None
    
    return models

@st.cache_resource
def load_meta_learner():
    """Load stacking meta-learner"""
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

# Load all models
base_models = load_base_models()
meta_learner = load_meta_learner()
simplifier_tokenizer, simplifier_model = load_simplification_model()

# -----------------------------------------
# Helper Functions
# -----------------------------------------
def get_model_probabilities(text, models):
    """Extract 18-dim feature vector from base models"""
    probabilities = []
    
    # AraBERT
    if models['arabert_model'] and models['arabert_tokenizer']:
        inputs = models['arabert_tokenizer'](text, return_tensors="pt", truncation=True, max_length=256, padding=True)
        with torch.no_grad():
            logits = models['arabert_model'](**inputs).logits
            probs = torch.softmax(logits, dim=-1).numpy()[0]
        probabilities.extend(probs)
    else:
        probabilities.extend([0.0] * 6)
    
    # CamelBERT
    if models['camelbert_model'] and models['camelbert_tokenizer']:
        inputs = models['camelbert_tokenizer'](text, return_tensors="pt", truncation=True, max_length=256, padding=True)
        with torch.no_grad():
            logits = models['camelbert_model'](**inputs).logits
            probs = torch.softmax(logits, dim=-1).numpy()[0]
        probabilities.extend(probs)
    else:
        probabilities.extend([0.0] * 6)
    
    # BiLSTM
    if models['bilstm_model'] and models['bilstm_tokenizer'] and models['bilstm_bert']:
        inputs = models['bilstm_tokenizer'](text, return_tensors="pt", truncation=True, max_length=256, padding=True)
        with torch.no_grad():
            outputs = models['bilstm_bert'](**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1].mean(dim=1)
        probs = models['bilstm_model'].predict_proba(embeddings.numpy())[0]
        probabilities.extend(probs)
    else:
        probabilities.extend([0.0] * 6)
    
    return np.array(probabilities).reshape(1, -1)

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
            
            # Get 18-dim features
            features = get_model_probabilities(cleaned, base_models)
            
            # Predict with meta-learner
            prediction = meta_learner.predict(features)[0]
            
            try:
                probs = meta_learner.predict_proba(features)[0]
                confidence = probs[prediction - 1] if prediction <= len(probs) else np.max(probs)
            except:
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
    level_names = {1: "سهل جداً", 2: "سهل", 3: "متوسط", 4: "صعب قليلاً", 5: "صعب", 6: "صعب جداً"}
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="المستوى", value=f"{level_colors.get(level, '⚪')} {level}")
    with col2:
        st.metric(label="الوصف", value=level_names.get(level, "غير معروف"))
    
    st.progress(int(st.session_state.confidence * 100))
    st.write(f"**نسبة الثقة:** {st.session_state.confidence:.2%}")
    
    # -----------------------------------------
    # بَسِّطْ Button (for levels 4-6)
    # -----------------------------------------
    if level >= 4:
        st.markdown("---")
        st.info("💡 هذا النص صعب القراءة. يمكنك تبسيطه بالضغط على الزر أدناه.")
        
        if st.button("✨ بَسِّطْ", use_container_width=True, type="secondary"):
            if simplifier_model and simplifier_tokenizer:
                with st.spinner("جاري التبسيط..."):
                    cleaned = normalize_ar(st.session_state.original_text)
                    
                    # Tokenize for AraBART
                    inputs = simplifier_tokenizer(
                        cleaned,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=512
                    )
                    
                    # Generate simplified text with AraBART
                    with torch.no_grad():
                        outputs = simplifier_model.generate(
                            **inputs,
                            max_length=512,
                            num_beams=4,
                            length_penalty=1.0,
                            early_stopping=True,
                            no_repeat_ngram_size=3
                        )
                    
                    simplified_text = simplifier_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    st.markdown("---")
                    st.subheader("✨ النص المبسط")
                    st.markdown(f'<div class="simplified-box">{simplified_text}</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ نموذج التبسيط غير متوفر حالياً.")

st.markdown("---")
st.caption("© 2025 — مشروع بَيِّنْ")
