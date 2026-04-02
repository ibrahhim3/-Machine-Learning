"""
 Intelligent SMS Spam Detection for Mobile Communication Networks
 CSE 476/575 Term Project - Spring 2026
 Student: Ibrahim Al Said | Instructor: Prof. Dr. Hasari Celebi
"""

import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
STOP_WORDS = set(ENGLISH_STOP_WORDS)


# ── Step 1: Load Dataset ────────────────────────────────────────────────────
def load_dataset():
    print("=" * 60)
    print("  STEP 1: Loading Dataset")
    print("=" * 60)
    df = pd.read_csv("SMSSpamCollection", sep="\t", header=None,
                      names=["label", "message"], encoding="latin-1")
    print(f"  Loaded {df.shape[0]} messages\n")
    return df


# ── Step 2: Exploratory Data Analysis ───────────────────────────────────────
def exploratory_analysis(df):
    print("=" * 60)
    print("  STEP 2: Exploratory Data Analysis")
    print("=" * 60)
    counts = df["label"].value_counts()
    print(f"  Ham: {counts.get('ham', 0)}  |  Spam: {counts.get('spam', 0)}")
    print(f"  Spam ratio: {counts.get('spam', 0) / len(df) * 100:.1f}%")

    df["msg_length"] = df["message"].apply(len)
    print(f"  Avg length - Ham: {df[df['label']=='ham']['msg_length'].mean():.0f}  "
          f"Spam: {df[df['label']=='spam']['msg_length'].mean():.0f}\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts.plot(kind="bar", ax=axes[0], color=["#2ecc71", "#e74c3c"], edgecolor="black")
    axes[0].set_title("Class Distribution", fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)

    for label, color in [("ham", "#2ecc71"), ("spam", "#e74c3c")]:
        axes[1].hist(df[df["label"] == label]["msg_length"], bins=30,
                     alpha=0.7, label=label.title(), color=color, edgecolor="black")
    axes[1].set_title("Message Length Distribution", fontweight="bold")
    axes[1].set_xlabel("Characters")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_eda_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    return df


# ── Step 3: Text Preprocessing ──────────────────────────────────────────────
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)   # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)            # remove punctuation/numbers
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def preprocess_dataset(df):
    print("=" * 60)
    print("  STEP 3: Text Preprocessing")
    print("=" * 60)
    df["cleaned"] = df["message"].apply(preprocess_text)
    df["label_enc"] = df["label"].map({"ham": 0, "spam": 1})

    for i in range(3):
        print(f"  Original: {df.iloc[i]['message'][:70]}")
        print(f"  Cleaned:  {df.iloc[i]['cleaned'][:70]}\n")
    return df


# ── Step 4: TF-IDF Vectorization ────────────────────────────────────────────
def vectorize_data(df):
    print("=" * 60)
    print("  STEP 4: TF-IDF Vectorization")
    print("=" * 60)
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df["cleaned"])
    y = df["label_enc"]
    print(f"  Vocabulary: {len(tfidf.vocabulary_)} terms  |  Matrix: {X.shape}\n")
    return X, y, tfidf


# ── Step 5: Train/Test Split ────────────────────────────────────────────────
def split_data(X, y):
    print("=" * 60)
    print("  STEP 5: Train/Test Split (80/20)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"  Training: {X_train.shape[0]}  |  Testing: {X_test.shape[0]}\n")
    return X_train, X_test, y_train, y_test


# ── Step 6: Train Naive Bayes ───────────────────────────────────────────────
def train_model(X_train, y_train):
    print("=" * 60)
    print("  STEP 6: Training Multinomial Naive Bayes")
    print("=" * 60)
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)
    print(f"  Training accuracy: {model.score(X_train, y_train) * 100:.2f}%\n")
    return model


# ── Step 7: Evaluation ──────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    print("=" * 60)
    print("  STEP 7: Model Evaluation")
    print("=" * 60)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"],
                linewidths=1, linecolor="black",
                annot_kws={"size": 18, "weight": "bold"}, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    return acc


# ── Step 8: Live Demo ───────────────────────────────────────────────────────
def live_demo(model, tfidf):
    print("=" * 60)
    print("  STEP 8: Live Prediction Demo")
    print("=" * 60)
    test_messages = [
        "Hey, want to grab lunch together at noon?",
        "CONGRATULATIONS! You won a $1000 Visa gift card! Claim NOW at www.scam.com",
        "Can you send me the homework for tomorrow's class?",
        "URGENT: Your bank account is locked. Click here to verify your identity.",
        "I'll be home by 7. Do you want pizza for dinner?",
        "FREE FREE FREE! Get unlimited data for just $1/month! Text YES to 55555",
    ]
    print()
    for msg in test_messages:
        cleaned = preprocess_text(msg)
        vec = tfidf.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        label = "HAM" if pred == 0 else "SPAM"
        print(f"  [{label:4s} {prob[pred]*100:5.1f}%]  {msg[:65]}")
    print()


# ── Step 9: Interactive Mode ────────────────────────────────────────────────
def interactive_mode(model, tfidf):
    print("=" * 60)
    print("  INTERACTIVE MODE - Type any SMS to classify it!")
    print("  Type 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            msg = input("  >> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in ("quit", "exit", "q"):
            print("  Goodbye!\n")
            break
        if not msg:
            continue

        cleaned = preprocess_text(msg)
        vec = tfidf.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        label = "HAM (Legitimate)" if pred == 0 else "SPAM (Blocked)"
        print(f"\n  Result: {label}")
        print(f"  Ham: {prob[0]*100:.1f}%  |  Spam: {prob[1]*100:.1f}%")
        print(f"  Cleaned: {cleaned[:50]}\n")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  SMS SPAM DETECTION - Ibrahim Al Said")
    print("  CSE 476/575 Term Project")
    print("=" * 60 + "\n")

    df = load_dataset()
    df = exploratory_analysis(df)
    df = preprocess_dataset(df)
    X, y, tfidf = vectorize_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    acc = evaluate_model(model, X_test, y_test)
    live_demo(model, tfidf)

    print(f"  DONE! Accuracy: {acc*100:.2f}% | Figures in ./{OUTPUT_DIR}/\n")
    interactive_mode(model, tfidf)


if __name__ == "__main__":
    main()
