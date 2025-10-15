import os
import re
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    classification_report
)

from data_manipulation import MERGE_MAP, US_STATE_ABBRS, DROP_LABELS

PATH = os.path.join("data", "books.csv")
MIN_PER_GENRE = 50
TOP_N_TERMS = 10
NGRAM_RANGE = (1, 2)
MAX_FEATURES = 50_000
THRESH = 0.40
RANDOM_STATE = 42

df = pd.read_csv(PATH, low_memory=False)
assert "Description" in df.columns, "Column 'Description' not found"
assert "Category" in df.columns, "Column 'Category' not found"
df = df[["Description", "Category"]].dropna(how="all").copy()

def clean_text(s: str) -> str:
    s = str(s).strip()
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

SEP_RE = re.compile(r"[,\|;/]")
PARENS_RE = re.compile(r"\([^)]*\)")

def _rule_based_merge(lbl: str) -> str:
    lbl = lbl.replace("&", "and")
    lbl = re.sub(r"\b(stories)\b", "story", lbl)
    lbl = re.sub(r"\b(mysteries)\b", "mystery", lbl)
    return lbl.strip()

def normalize_label(raw: str) -> str | None:
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower()
    s = PARENS_RE.sub("", s)
    s = s.replace("&", "and")
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    if not s or s == "general":
        return None
    if s in DROP_LABELS or s in US_STATE_ABBRS:
        return None

    s = _rule_based_merge(s)
    s = MERGE_MAP.get(s, s)

    if s in DROP_LABELS or s in US_STATE_ABBRS or len(s) <= 2:
        return None
    return s

def extract_all_categories(cat: str) -> list[str]:
    if not isinstance(cat, str) or not cat.strip():
        return []
    parts = [p for p in SEP_RE.split(cat) if p.strip()]
    out, seen = [], set()
    for p in parts:
        norm = normalize_label(p)
        if norm and norm not in seen:
            out.append(norm)
            seen.add(norm)
    return out

df["desc_clean"] = df["Description"].astype(str).apply(clean_text)
df["labels_raw"] = df["Category"].apply(extract_all_categories)

df = df[(df["desc_clean"].str.len() >= 25)]
df = df[df["labels_raw"].map(len) > 0].copy()

label_counter = Counter(lbl for labels in df["labels_raw"] for lbl in labels)
keep_labels = {lbl for lbl, cnt in label_counter.items() if cnt >= MIN_PER_GENRE}

df["labels"] = df["labels_raw"].apply(lambda L: [l for l in L if l in keep_labels])
df = df[df["labels"].map(len) > 0].copy()

X = df["desc_clean"].values
Y_labels = df["labels"].values

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(Y_labels)
label_names = mlb.classes_

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
)

def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE
        )),
        ("clf", OneVsRestClassifier(
            LogisticRegression(max_iter=300, solver="liblinear")
        ))
    ])

def print_top_terms_per_label(pipeline: Pipeline, label_names_arr, top_n=10):
    tfidf = pipeline.named_steps["tfidf"]
    ovr = pipeline.named_steps["clf"]
    feature_names = np.array(tfidf.get_feature_names_out())
    estimators = ovr.estimators_
    print(f"\nTop {top_n} indicative terms per genre:")
    for label, est in zip(label_names_arr, estimators):
        coefs = est.coef_.ravel()
        top_idx = np.argsort(coefs)[::-1][:top_n]
        terms = feature_names[top_idx]
        weights = coefs[top_idx]
        pretty = ", ".join([f"{t} ({w:.2f})" for t, w in zip(terms, weights)])
        print(f"\n[{label}]")
        print(pretty)

def evaluate_model(pipe: Pipeline):
    Y_prob = pipe.predict_proba(X_test)
    Y_pred = (Y_prob >= THRESH).astype(int)

    for i in range(Y_pred.shape[0]):
        if Y_pred[i].sum() == 0:
            j = int(np.argmax(Y_prob[i]))
            Y_pred[i, j] = 1

    sample_accuracy = (Y_test == Y_pred).all(axis=1).mean()

    print(f"\nEvaluation")
    print(f"  Subset accuracy: {sample_accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(Y_test, Y_pred, target_names=mlb.classes_, zero_division=0))

if __name__ == "__main__":
    pipe = build_pipeline()
    print("Training model...")
    pipe.fit(X_train, Y_train)
    print("Done.\n")
    evaluate_model(pipe)
    print_top_terms_per_label(pipe, label_names, top_n=TOP_N_TERMS)