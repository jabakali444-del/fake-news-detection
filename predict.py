import joblib
import re
import string

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

title = input("Enter news title: ")
text = input("Enter news text: ")

content = title + " " + text
content = clean_text(content)

vector = vectorizer.transform([content])

prediction = model.predict(vector)[0]
probabilities = model.predict_proba(vector)[0]

classes = model.classes_
confidence = max(probabilities) * 100

print("\nPrediction:", prediction)
print("Confidence: {:.2f}%".format(confidence))

print("\nClass probabilities:")
for label, prob in zip(classes, probabilities):
    print(f"{label}: {prob * 100:.2f}%")