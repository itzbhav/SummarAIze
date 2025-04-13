import streamlit as st
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# 🔧 Must be first Streamlit command
st.set_page_config(page_title="Keyword & Summary Bot", page_icon="🧠")

# 📦 Load models only once
@st.cache_resource
def load_models():
    kw_model = KeyBERT(SentenceTransformer('all-MiniLM-L6-v2'))
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return kw_model, summarizer

kw_model, summarizer = load_models()

# 🧠 UI
st.title("🤖 NLP Assistant: Keyword Extractor & Summarizer")
st.write("Welcome! Select a task below and enter your text to get smart results.")

# 🧭 Task Selection
task = st.selectbox("Choose your task:", ["Select task", "Keyword Extraction", "Text Summarization"])

# ✏️ User Input
user_input = st.text_area("Enter your text here:")

# 🚀 Submit Button
if st.button("Submit") and user_input.strip():

    # 🔑 Keyword Extraction
    if task == "Keyword Extraction":
        keywords = kw_model.extract_keywords(
            user_input,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=5
        )
        keyword_list = [kw[0] for kw in keywords]
        st.success(f"🔑 Keywords: {', '.join(keyword_list)}")

    # 📃 Text Summarization
    elif task == "Text Summarization":
        if len(user_input.split()) < 50:
            st.warning("⚠️ Enter a longer paragraph (at least 50 words) for better summarization.")
        elif len(user_input.split()) > 500:
            st.warning("⚠️ Your input is too long. Try to shorten it below 500 words.")
        else:
            summary = summarizer(
                user_input,
                max_length=100,
                min_length=30,
                do_sample=False
            )
            st.success(f"📃 Summary: {summary[0]['summary_text']}")

    else:
        st.warning("⚠️ Please select a task to perform.")
