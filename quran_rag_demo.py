import fitz  # PyMuPDF
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import streamlit as st

# ----------- Verse extraction and normalization -----------

def extract_verses(text):
    """
    Extract Arabic verses based on Arabic digits pattern.
    Returns list of (verse_num, verse_text)
    """
    pattern = r'([٠-٩]{1,3})\s+(.*?)(?=[٠-٩]{1,3}\s+|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def normalize_verses(verses):
    """
    Placeholder normalization: strip whitespace, could add diacritics normalization etc.
    """
    normalized = []
    for num, verse in verses:
        norm_verse = verse.strip()
        normalized.append((num, norm_verse))
    return normalized

# ----------- PDF Loading & Verse Extraction -----------

def load_pdf_and_extract_verses(pdf_path, max_pages=10):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        st.error(f"Failed to open PDF: {e}")
        return []

    all_verses = []
    try:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            text = page.get_text()
            verses = extract_verses(text)
            if verses:
                verses = normalize_verses(verses)
                all_verses.extend(verses)
    except Exception as e:
        st.error(f"Error extracting text from PDF pages: {e}")
    finally:
        doc.close()
    return all_verses

# ----------- Embedding & FAISS Index -----------

class QuranRetriever:
    def __init__(self, verses, model):
        self.verses = verses
        self.texts = [v[1] for v in verses]
        self.model = model
        self.embeddings = self.model.encode(self.texts, show_progress_bar=True, convert_to_numpy=True)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])  # Inner product similarity
        faiss.normalize_L2(self.embeddings)  # Normalize embeddings for cosine similarity
        self.index.add(self.embeddings)

    def search(self, query, top_k=5):
        query_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        D, I = self.index.search(query_emb, top_k)
        results = []
        for idx in I[0]:
            results.append(self.verses[idx])
        return results

# ----------- Model loading (cached) -----------

@st.cache_resource
def load_models():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    generator = pipeline("summarization", model="facebook/bart-large-cnn")
    return model, generator

model, generator = load_models()

# ----------- Generation -----------

def generate_answer(retrieved_verses, question):
    """
    Generate an answer by concatenating retrieved verses and summarizing with question.
    This is a simplified demo using summarization.
    """
    context = " ".join([v[1] for v in retrieved_verses])
    input_text = f"Question: {question}\nContext: {context}"
    output = generator(input_text, max_length=100, min_length=20, do_sample=False)
    return output[0]['summary_text']

# ----------- Evaluation -----------

def evaluate_answer(answer, retrieved_verses):
    answer_len = len(answer.split())
    answer_words = set(answer.split())
    verses_text = " ".join([v[1] for v in retrieved_verses])
    verse_words = set(verses_text.split())
    overlap = len(answer_words.intersection(verse_words)) / max(len(answer_words), 1)
    return {
        "answer_length_words": answer_len,
        "word_overlap_with_retrieved": overlap
    }

# ----------- Streamlit UI -----------

def main():
    st.title("Quran RAG Demo — Arabic Verse Q&A")

    uploaded_file = st.file_uploader("Upload Quran PDF", type=["pdf"])

    if uploaded_file:
        with open("temp_quran.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
            f.flush()
        st.success("PDF uploaded successfully!")

        if st.button("Process PDF and build index"):
            with st.spinner("Extracting verses and building index..."):
                verses = load_pdf_and_extract_verses("temp_quran.pdf", max_pages=20)
                if not verses:
                    st.error("No verses extracted. Please check the PDF content or extraction logic.")
                else:
                    st.write(f"Extracted {len(verses)} verses.")
                    st.session_state['retriever'] = QuranRetriever(verses, model)
                    st.success("Index built successfully!")

    if 'retriever' in st.session_state:
        retriever = st.session_state['retriever']
        question = st.text_input("Enter your question about Quran verses:")
        if question:
            with st.spinner("Searching and generating answer..."):
                results = retriever.search(question, top_k=5)
                answer = generate_answer(results, question)
                eval_metrics = evaluate_answer(answer, results)
                st.subheader("Answer:")
                st.write(answer)
                st.subheader("Retrieved Verses:")
                for num, verse in results:
                    st.write(f"{num}: {verse}")
                st.subheader("Evaluation Metrics:")
                st.write(eval_metrics)

if __name__ == "__main__":
    main()
