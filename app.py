import re
import joblib
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document

# Load model
model = joblib.load("resume_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(
    page_title="Resume Classification System",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Resume Classification System")
st.write("AI-based app to classify resumes into job categories using NLP and Machine Learning.")

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Extract PDF
def extract_pdf_text(file):
    text = ""
    pdf = PdfReader(file)
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text

# Extract DOCX
def extract_docx_text(file):
    text = ""
    doc = Document(file)
    for para in doc.paragraphs:
        text += para.text + " "
    return text

# Upload
uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx", "txt"]
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".pdf"):
        resume_text = extract_pdf_text(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        resume_text = extract_docx_text(uploaded_file)
    else:
        resume_text = uploaded_file.read().decode("utf-8", errors="ignore")

    st.subheader("Extracted Resume Text")
    st.text_area("Preview", resume_text[:3000], height=250)

    # Prediction
    cleaned_text = clean_text(resume_text)
    vectorized_text = vectorizer.transform([cleaned_text])

    probs = model.predict_proba(vectorized_text)[0]
    classes = model.classes_

    # Top 3 predictions
    top_indices = probs.argsort()[-3:][::-1]

    st.subheader("🎯 Top Predictions")

    for i in top_indices:
        st.write(f"👉 {classes[i]} → {probs[i]*100:.2f}%")

    # Final prediction
    final_prediction = classes[top_indices[0]]
    final_confidence = probs[top_indices[0]] * 100

    st.subheader("🏆 Best Match")
    st.success(f"{final_prediction}")

    # Confidence UI
    st.subheader("Confidence Score")
    st.progress(int(final_confidence))
    st.info(f"{final_confidence:.2f}% Confidence")