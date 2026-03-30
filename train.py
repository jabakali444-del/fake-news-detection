import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils import clean_text


def main():
    # Load datasets
    fake = pd.read_csv("data/Fake.csv")
    true = pd.read_csv("data/True.csv")

    # Add labels
    fake["label"] = "FAKE"
    true["label"] = "REAL"

    # Keep useful columns only
    fake = fake[["title", "text", "label"]]
    true = true[["title", "text", "label"]]

    # Merge and shuffle
    data = pd.concat([fake, true], axis=0)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Fill missing values
    data["title"] = data["title"].fillna("")
    data["text"] = data["text"].fillna("")

    # Combine title + text
    data["content"] = data["title"] + " " + data["text"]

    # Clean text
    data["content"] = data["content"].apply(clean_text)

    # Features and labels
    X = data["content"]
    y = data["label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)

    print("Dataset shape:", data.shape)
    print("\nLabel counts:")
    print(data["label"].value_counts())

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    # Save model artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fake_news_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("\nModel and vectorizer saved successfully.")

    # Quick test samples
    true_sample = clean_text(str(true.iloc[0]["title"]) + " " + str(true.iloc[0]["text"]))
    fake_sample = clean_text(str(fake.iloc[0]["title"]) + " " + str(fake.iloc[0]["text"]))

    true_vector = vectorizer.transform([true_sample])
    fake_vector = vectorizer.transform([fake_sample])

    print("\nTest sample from True.csv:")
    print("Predicted:", model.predict(true_vector)[0])
    print("Expected: REAL")

    print("\nTest sample from Fake.csv:")
    print("Predicted:", model.predict(fake_vector)[0])
    print("Expected: FAKE")


if __name__ == "__main__":
    main()