import pandas as pd
import re
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1) Load datasets
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# 2) Add labels
fake["label"] = "FAKE"
true["label"] = "REAL"

# 3) Keep useful columns only
fake = fake[["title", "text", "label"]]
true = true[["title", "text", "label"]]

# 4) Merge datasets
data = pd.concat([fake, true], axis=0)

# 5) Shuffle dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# 6) Combine title and text
data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")
data["content"] = data["title"] + " " + data["text"]

# 7) Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 8) Apply cleaning
data["content"] = data["content"].apply(clean_text)

# 9) Quick checks
print("Dataset shape:", data.shape)
print("\nLabel counts:")
print(data["label"].value_counts())

print("\nFake sample:")
print(fake[["title", "label"]].head())

print("\nTrue sample:")
print(true[["title", "label"]].head())

# 10) Features and labels
X = data["content"]
y = data["label"]

# 11) Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 12) TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 13) Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 14) Prediction
y_pred = model.predict(X_test_vec)

# 15) Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# 16) Save model and vectorizer
joblib.dump(model, "models/fake_news_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModel saved successfully.")

# 17) Test on one sample from True.csv
true_sample = clean_text(str(true.iloc[0]["title"]) + " " + str(true.iloc[0]["text"]))
true_vector = vectorizer.transform([true_sample])
true_prediction = model.predict(true_vector)[0]

print("\nTest sample from True.csv:")
print("Predicted:", true_prediction)
print("Expected: REAL")

# 18) Test on one sample from Fake.csv
fake_sample = clean_text(str(fake.iloc[0]["title"]) + " " + str(fake.iloc[0]["text"]))
fake_vector = vectorizer.transform([fake_sample])
fake_prediction = model.predict(fake_vector)[0]

print("\nTest sample from Fake.csv:")
print("Predicted:", fake_prediction)
print("Expected: FAKE")