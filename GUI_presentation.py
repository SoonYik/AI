import os
import joblib
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@[^\s]+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)


import __main__
__main__.preprocess_text = preprocess_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


st.set_page_config(page_title="Satirical Sentiment Analysis", page_icon="📊", layout="wide")
st.title("📊 Satirical Sentiment Analysis on Social Media")
st.markdown("""
This dashboard demonstrates the performance of three different Machine Learning architectures 
in predicting general **Sentiment** and detecting **Sarcasm** in unfiltered social media text.
""")

st.sidebar.header("Configuration")
selected_model = st.sidebar.radio(
    "Select Model Architecture:",
    ("Support Vector Machine (SVM)", "Naive Bayes", "BERT Transformer")
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Group Members:**")
st.sidebar.markdown("- Soon Yik (SVM)\n- Ong Zi Jian (Naive Bayes)\n- Lim Yek Sean (BERT)")

@st.cache_resource
def load_traditional_model(model_name):
    """Loads your saved .pkl files using absolute paths."""
    try:
        if model_name == "SVM":
            sent_path = os.path.join(BASE_DIR, 'svm_sentiment_model.pkl')
            sarc_path = os.path.join(BASE_DIR, 'svm_sarcasm_model.pkl')
            
            sent_model = joblib.load(sent_path)
            sarc_model = joblib.load(sarc_path)
        else: # Naive Bayes
            sent_path = os.path.join(BASE_DIR, 'nb_sentiment_model.pkl')
            sarc_path = os.path.join(BASE_DIR, 'nb_sarcasm_model.pkl')
            
            sent_model = joblib.load(sent_path)
            sarc_model = joblib.load(sarc_path)
            
        return sent_model, sarc_model
        
    except FileNotFoundError as e:
        st.error(f"❌ Cannot find file: {e.filename}")
        return None, None

@st.cache_resource
def load_bert_model():
    """Loads the fine-tuned BERT models."""
    try:
        sent_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        sent_model = AutoModelForSequenceClassification.from_pretrained("./checkpoints/sentiment")
        
        sarc_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        sarc_model = AutoModelForSequenceClassification.from_pretrained("./checkpoints/sarcasm")
        return (sent_tokenizer, sent_model), (sarc_tokenizer, sarc_model)
    except Exception as e:
        print(f"Error loading BERT: {e}")
        return None, None


st.subheader("📝 Live Text Analysis")
user_input = st.text_area("Enter a Reddit comment or social media post to analyze:", 
                          height=150, 
                          placeholder="e.g., Oh great, another update that breaks everything! Exactly what I wanted.")

if st.button("Analyze Text", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner(f"Analyzing using {selected_model}..."):
            time.sleep(0.5) 
            
            sentiment_map = {-1: "Negative 🔴", 0: "Neutral ⚪", 1: "Positive 🟢"}
            sarcasm_map = {0: "Not Sarcastic 🗣️", 1: "Sarcastic 🎭"}
            
            sent_pred_label = "N/A"
            sarc_pred_label = "N/A"

            if selected_model in ["Support Vector Machine (SVM)", "Naive Bayes"]:
                model_key = "SVM" if "SVM" in selected_model else "NB"
                sent_model, sarc_model = load_traditional_model(model_key)
                
                if sent_model and sarc_model:
                    try:
                        if model_key == "SVM":
                            sent_pred = sent_model.predict([user_input])[0]
                            sarc_pred = sarc_model.predict([user_input])[0]
                            
                            sent_pred_label = sentiment_map.get(sent_pred, "Unknown")
                            sarc_pred_label = sarcasm_map.get(sarc_pred, "Unknown")
                            
                        else:
                            from scipy.sparse import hstack, csr_matrix
                            
                            sarc_vec = sarc_model['hv'].transform([user_input])
                            sarc_pred = sarc_model['clf'].predict(sarc_vec)[0]
                            sarc_prob = sarc_model['clf'].predict_proba(sarc_vec)[:, 1]
                            
                            sent_vec = sent_model['tfidf'].transform([user_input])
                            stacked_vec = hstack([sent_vec, csr_matrix(sarc_prob).T])
                            raw_sent_pred = sent_model['clf'].predict(stacked_vec)[0]
                            
                            FLIP_MAP = {'Positive': 'Negative', 'Negative': 'Positive', 'Neutral': 'Neutral'}
                            final_sent = FLIP_MAP.get(raw_sent_pred, raw_sent_pred) if sarc_pred == 1 else raw_sent_pred
                            
                            emoji_map = {'Positive': 'Positive 🟢', 'Neutral': 'Neutral ⚪', 'Negative': 'Negative 🔴'}
                            sent_pred_label = emoji_map.get(final_sent, "Unknown")
                            sarc_pred_label = sarcasm_map.get(sarc_pred, "Unknown")
                            
                    except Exception as e:
                        st.error(f"Prediction Error: {e}. Ensure you saved your Pipeline correctly.")
                else:
                    st.error(f"⚠️ Model files for {selected_model} not found in this folder.")

            elif selected_model == "BERT Transformer":
                sent_bundle, sarc_bundle = load_bert_model()
                
                if sent_bundle and sarc_bundle:
                    sent_tok, sent_mod = sent_bundle
                    sarc_tok, sarc_mod = sarc_bundle

                    inputs = sent_tok(user_input, return_tensors="pt", truncation=True, padding=True)
                    with torch.no_grad():
                        logits = sent_mod(**inputs).logits
                        sent_pred = torch.argmax(logits, dim=1).item()
                    
                    inputs = sarc_tok(user_input, return_tensors="pt", truncation=True, padding=True)
                    with torch.no_grad():
                        logits = sarc_mod(**inputs).logits
                        sarc_pred = torch.argmax(logits, dim=1).item()
                    
                    sent_pred_label = f"Class {sent_pred}" 
                    sarc_pred_label = sarcasm_map.get(sarc_pred, "Unknown")
                else:
                    st.error("⚠️ BERT model folders not found in this directory.")

            
            st.markdown("### 🔍 Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("**General Sentiment**")
                st.subheader(sent_pred_label)
                
            with col2:
                if "Sarcastic 🎭" in sarc_pred_label:
                    st.error("**Sarcasm Detection**")
                else:
                    st.success("**Sarcasm Detection**")
                st.subheader(sarc_pred_label)