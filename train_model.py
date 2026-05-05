import re
import joblib
import pandas as pd
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report

nltk.download("stopwords")

df = pd.read_csv("dataset/resume_dataset.csv")

df = df[["Category", "Resume_str"]].dropna()
df = df.rename(columns={"Resume_str": "Resume"})

#  KEEP ONLY RELEVANT TECH CATEGORIES
allowed_categories = [
    "INFORMATION-TECHNOLOGY",
    "ENGINEERING",
    "DESIGNER",
    "FINANCE",
    "HR",
    "SALES"
]

df = df[df["Category"].isin(allowed_categories)]

#  OPTIONAL: MERGE SIMILAR INTO IT
df["Category"] = df["Category"].replace({
    "ENGINEERING": "INFORMATION-TECHNOLOGY",
    "DESIGNER": "INFORMATION-TECHNOLOGY"
})

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

df["cleaned_resume"] = df["Resume"].apply(clean_text)

X = df["cleaned_resume"]
y = df["Category"]

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

base_model = LinearSVC(class_weight="balanced")
model = CalibratedClassifierCV(base_model, cv=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

joblib.dump(model, "resume_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print(" Cleaned model trained and saved!")