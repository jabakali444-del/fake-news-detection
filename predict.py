import joblib
from utils import clean_text


model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


def main():
    title = input("Enter news title: ")
    text = input("Enter news text: ")

    content = f"{title} {text}".strip()

    if not content:
        print("Please enter a title or text.")
        return

    cleaned = clean_text(content)
    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    confidence = max(probabilities) * 100

    print("\nPrediction:", prediction)
    print("Confidence: {:.2f}%".format(confidence))

    print("\nClass probabilities:")
    for label, prob in zip(model.classes_, probabilities):
        print(f"{label}: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()