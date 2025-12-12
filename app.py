import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification, AutoModel
import torch
import torch.nn as nn
import numpy as np
import re
from collections import Counter
import joblib
from huggingface_hub import hf_hub_download
import sys
import types

# -----------------------------------------
# BiLSTM Model Class Definition (MUST BE BEFORE LOADING)
# -----------------------------------------
class BiLSTMWithMeta(nn.Module):
    """
    BiLSTM with metadata support.
    Works with any input embedding (static or contextual).
    """
    def __init__(self, input_dim, categorical_cardinalities, num_numeric,
                 lstm_hidden=256, meta_proj_dim=128, num_classes=6, dropout=0.3,
                 use_bert=False, bert_model_name=None):
        super().__init__()

        self.use_bert = use_bert

        # Optional BERT encoder (not used for embedding-based models)
        if use_bert and bert_model_name:
            from transformers import AutoModel
            self.bert = AutoModel.from_pretrained(bert_model_name)
            input_dim = self.bert.config.hidden_size
        else:
            self.bert = None

        # BiLSTM layer
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden,
                           num_layers=1, batch_first=True, bidirectional=True)

        # Metadata embeddings for categorical features
        self.cat_names = list(categorical_cardinalities.keys())
        self.cat_embeddings = nn.ModuleDict()
        total_cat_emb_dim = 0
        for name, card in categorical_cardinalities.items():
            emb_dim = min(50, max(4, int(card**0.5)))
            self.cat_embeddings[name] = nn.Embedding(card, emb_dim)
            total_cat_emb_dim += emb_dim

        # Metadata projection
        self.meta_proj = nn.Sequential(
            nn.Linear(total_cat_emb_dim + num_numeric, meta_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2 + meta_proj_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, numeric_meta, categorical_meta, attention_mask=None):
        # Process text through BERT if available
        if self.use_bert and self.bert is not None:
            bert_out = self.bert(input_ids=x, attention_mask=attention_mask)
            x = bert_out.last_hidden_state
        else:
            # For static embeddings, add sequence dimension
            if len(x.shape) == 2:
                x = x.unsqueeze(1)

        # BiLSTM processing
        lstm_out, _ = self.lstm(x)
        pooled = lstm_out.mean(dim=1) if self.use_bert else lstm_out.squeeze(1)

        # Process metadata
        cat_embs = [self.cat_embeddings[name](categorical_meta[:, i])
                   for i, name in enumerate(self.cat_names)]
        cat_concat = torch.cat(cat_embs, dim=1) if cat_embs else \
                     torch.zeros(numeric_meta.size(0), 0, device=numeric_meta.device)

        meta_concat = torch.cat([numeric_meta, cat_concat], dim=1)
        meta_vec = self.meta_proj(meta_concat)

        # Combine and classify
        fused = torch.cat([pooled, meta_vec], dim=1)
        fused = self.dropout(fused)
        return self.classifier(fused)


class BiLSTMWrapper:
    """
    Wrapper class for BiLSTM model that provides sklearn-compatible interface.
    Works with AraBERT embeddings (768-dim vectors).
    """
    def __init__(self, model, cat_cardinalities, num_numeric, num_classes=6, device='cpu'):
        self.device = device
        self.num_classes = num_classes
        self.cat_cardinalities = cat_cardinalities
        self.num_numeric = num_numeric
        self.model = model
        self.default_numeric = np.zeros(num_numeric, dtype=np.float32)
        self.default_categorical = np.zeros(len(cat_cardinalities), dtype=np.int64)

    def predict(self, X):
        """Predict class labels (1-6) for AraBERT embeddings."""
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
            return predictions + 1  # Convert 0-5 to 1-6

    def predict_proba(self, X):
        """Predict class probabilities for AraBERT embeddings."""
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


# -----------------------------------------
# Fix for joblib loading - IMPROVED METHOD
# -----------------------------------------
# Get the current module
current_module = sys.modules[__name__]

# Register classes in current module
current_module.BiLSTMWrapper = BiLSTMWrapper
current_module.BiLSTMWithMeta = BiLSTMWithMeta

# Also register in __main__ for compatibility
if '__main__' not in sys.modules:
    sys.modules['__main__'] = types.ModuleType('__main__')
sys.modules['__main__'].BiLSTMWrapper = BiLSTMWrapper
sys.modules['__main__'].BiLSTMWithMeta = BiLSTMWithMeta

# Register in 'main' namespace (Streamlit specific)
if 'main' not in sys.modules:
    sys.modules['main'] = current_module
else:
    sys.modules['main'].BiLSTMWrapper = BiLSTMWrapper
    sys.modules['main'].BiLSTMWithMeta = BiLSTMWithMeta


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
    
    # Model 3: BiLSTM AraBERT
    try:
        bilstm_repo = "Raya-y/Bayyin_models"
        bilstm_file = "bilstm_arabert_bayyin.joblib"
        bilstm_path = hf_hub_download(repo_id=bilstm_repo, filename=bilstm_file)
        
        # Load with explicit module mapping
        models['bilstm_model'] = joblib.load(bilstm_path)
        
        # Load AraBERT model for embeddings
        models['bilstm_tokenizer'] = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
        models['bilstm_bert'] = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv2")
    except Exception as e:
        st.error(f"خطأ في تحميل نموذج BiLSTM: {str(e)}")
        models['bilstm_model'] = None
        models['bilstm_tokenizer'] = None
        models['bilstm_bert'] = None
    
    return models

models_dict = load_models()
orig_tokenizer = models_dict.get('orig_tokenizer')
orig_model = models_dict.get('orig_model')
mix_tokenizer = models_dict.get('mix_tokenizer')
mix_model = models_dict.get('mix_model')
msa_tokenizer = models_dict.get('msa_tokenizer')
msa_model = models_dict.get('msa_model')
bilstm_model = models_dict.get('bilstm_model')
bilstm_tokenizer = models_dict.get('bilstm_tokenizer')
bilstm_bert = models_dict.get('bilstm_bert')

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
    
    .model-card-orig { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: white; }
    .model-card-mix { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
    .model-card-msa { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; }
    .model-card-bilstm { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; }
    
    .level-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 25px;
        font-size: 1.2em;
        font-weight: bold;
        margin: 10px 0;
    }
    .level-1, .level-2 { background: #2ecc71; color: white; }
    .level-3, .level-4 { background: #f39c12; color: white; }
    .level-5, .level-6 { background: #e74c3c; color: white; }
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
# Prediction Logic
# -----------------------------------------
if st.button("تصنيف النص", use_container_width=True):
    
    if not text.strip():
        st.error("الرجاء إدخال نص قبل الضغط على زر التصنيف.")
    
    elif not re.search(r'[\u0600-\u06ff]', text):
        st.error("عذراً، النص المدخل لا يبدو أنه باللغة العربية. الرجاء إدخال نص عربي صحيح.")
        
    else:
        if not any([orig_model, mix_model, msa_model, bilstm_model]):
            st.error("لم يتم تحميل أي نموذج.")
        else:
            cleaned = normalize_ar(text)
            
            def predict_level(model, tokenizer, text_input):
                if model and tokenizer:
                    try:
                        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=256)
                        with torch.no_grad():
                            logits = model(**inputs).logits
                        probs = torch.softmax(logits, dim=-1).numpy()[0]
                        pred_idx = np.argmax(probs)
                        level = pred_idx + 1
                        return level
                    except Exception as e:
                        return None
                return None
            
            def predict_bilstm(model, tokenizer, bert_model, text_input):
                if model and tokenizer and bert_model:
                    try:
                        # Get BERT embeddings
                        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=256)
                        with torch.no_grad():
                            outputs = bert_model(**inputs, output_hidden_states=True)
                            embeddings = outputs.hidden_states[-1].mean(dim=1)  # Average pooling
                        
                        # Use wrapper's predict method
                        level = model.predict(embeddings.numpy())[0]
                        return level
                    except Exception as e:
                        st.warning(f"خطأ في التنبؤ باستخدام BiLSTM: {str(e)}")
                        return None
                return None

            # Get predictions from all models
            orig_level = predict_level(orig_model, orig_tokenizer, cleaned)
            mix_level = predict_level(mix_model, mix_tokenizer, cleaned)
            msa_level = predict_level(msa_model, msa_tokenizer, cleaned)
            bilstm_level = predict_bilstm(bilstm_model, bilstm_tokenizer, bilstm_bert, cleaned)

            # Hard Voting Implementation
            predictions = [l for l in [orig_level, mix_level, msa_level, bilstm_level] if l is not None]
            
            final_level = None
            if predictions:
                counts = Counter(predictions)
                final_level = counts.most_common(1)[0][0]

            # Results Display
            level_names = {
                1: "سهل جداً", 2: "سهل", 3: "متوسط", 
                4: "صعب قليلاً", 5: "صعب", 6: "صعب جداً"
            }

            if final_level:
                st.markdown("---")
                st.markdown(f"""
                <div class='final-verdict-card'>
                    <h2 style='margin:0;'>النتيجة النهائية</h2>
                    <h1 style='font-size: 3.5em; margin: 10px 0;'>المستوى {final_level}</h1>
                    <h3 style='opacity: 0.9;'>{level_names.get(final_level, '')}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<h4 style='text-align: right; direction: rtl; color: #555;'>تفاصيل النماذج:</h4>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)

            def display_mini_card(column, title, level, css_class):
                with column:
                    if level:
                        st.markdown(f"""
                        <div class='model-card {css_class}'>
                            <h4 style='margin-bottom:5px;'>{title}</h4>
                            <div class='level-badge level-{level}'>المستوى {level}</div>
                            <p style='margin-top:5px; font-weight:bold;'>{level_names.get(level)}</p>
                        </div>
                        """, unsafe_allow_html=True)

            display_mini_card(c1, "Arabertv2", orig_level, "model-card-orig")
            display_mini_card(c2, "CAMeLBERT Mix", mix_level, "model-card-mix")
            display_mini_card(c3, "CAMeLBERT MSA", msa_level, "model-card-msa")
            display_mini_card(c4, "BiLSTM AraBERT", bilstm_level, "model-card-bilstm")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #667eea;'>© 2025 — مشروع بَيِّنْ</p>", unsafe_allow_html=True)
