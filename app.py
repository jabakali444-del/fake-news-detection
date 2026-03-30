import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# نفس الستايل تبعك (ما تغيره)
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .result-box {
            padding: 18px;
            border-radius: 12px;
            margin-top: 20px;
            margin-bottom: 15px;
            font-size: 20px;
            font-weight: bold;
            text-align: center;
        }
        .fake-box {
            background-color: #ffe6e6;
            color: #b30000;
            border: 1px solid #ff9999;
        }
        .real-box {
            background-color: #e8f8ec;
            color: #1b7f3b;
            border: 1px solid #9ad8ab;
        }
        .confidence-box {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #ddd;
            margin-top: 10px;
            margin-bottom: 20px;
            text-align: center;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📰 Fake News Detection App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">This app uses a FastAPI backend.</div>',
    unsafe_allow_html=True
)

title = st.text_input("News Title")
text = st.text_area("News Article Text", height=250)

col1, col2 = st.columns(2)

with col1:
    predict_clicked = st.button("Predict", use_container_width=True)

with col2:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

if predict_clicked:
    content = f"{title} {text}".strip()

    if not content:
        st.warning("Please enter a news title or article text.")
    else:
        payload = {
            "title": title,
            "text": text
        }

        try:
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                result = response.json()

                prediction = result["prediction"]
                confidence = result["confidence"]
                probabilities = result["probabilities"]

                if prediction == "FAKE":
                    st.markdown(
                        f'<div class="result-box fake-box">Prediction: {prediction}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="result-box real-box">Prediction: {prediction}</div>',
                        unsafe_allow_html=True
                    )

                st.markdown(
                    f'<div class="confidence-box">Confidence: {confidence:.2f}%</div>',
                    unsafe_allow_html=True
                )

                st.subheader("Class Probabilities")
                for label, prob in probabilities.items():
                    st.write(f"**{label}:** {prob:.2f}%")
                    st.progress(prob / 100)

            else:
                st.error("API Error")

        except:
            st.error("Cannot connect to API. Make sure FastAPI is running.")