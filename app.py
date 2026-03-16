import streamlit as st
import joblib
import re
import string

# Load model and vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection App")
st.write("Enter a news title and article text, then click Predict.")

title = st.text_input("News Title")
text = st.text_area("News Article Text", height=250)

if st.button("Predict"):
    if title.strip() == "" and text.strip() == "":
        st.warning("Please enter a news title or article text.")
    else:
        content = title + " " + text
        cleaned = clean_text(content)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]
        confidence = max(probabilities) * 100

        if prediction == "FAKE":
            st.error(f"Prediction: {prediction}")
        else:
            st.success(f"Prediction: {prediction}")

        st.info(f"Confidence: {confidence:.2f}%")

        st.subheader("Class Probabilities")
        for label, prob in zip(model.classes_, probabilities):
            st.write(f"**{label}:** {prob * 100:.2f}%")