# Resume Classification System

An AI/ML-based web application that automatically classifies resumes into suitable job categories using Natural Language Processing and Machine Learning.

## 🚀 Live Demo

🔗 https://resume-classification-system-7h7rkn5ldzmmjjbcc2dpml.streamlit.app/

## Description

The Resume Classification System helps recruiters filter and shortlist resumes efficiently. It extracts resume text, processes it using NLP techniques, and predicts the most suitable job category.

## Features

- Upload resumes in PDF, DOCX, or TXT format
- Extract resume text automatically
- Clean and preprocess text
- TF-IDF feature extraction
- Machine Learning based classification
- Confidence score display
- Simple Streamlit web interface

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- NLP
- TF-IDF
- Logistic Regression
- PyPDF2
- python-docx

## Project Structure

```text
Resume-Classification-System/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── resume_classifier.pkl
├── tfidf_vectorizer.pkl
└── dataset/
    └── resume_dataset.csv
