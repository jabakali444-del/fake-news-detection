# Fake News Detection (Full Stack ML Project)

A full-stack machine learning project that classifies news articles as **FAKE** or **REAL** using NLP techniques, a FastAPI backend, and a Streamlit web application.

---

## 🚀 Features

* Fake vs Real news classification
* Text preprocessing (cleaning, normalization)
* TF-IDF vectorization
* Logistic Regression model
* FastAPI backend (REST API)
* Streamlit interactive UI
* Confidence score + probabilities

---

## 🧠 Technologies Used

* Python
* Pandas
* Scikit-learn
* FastAPI
* Streamlit
* Joblib
* Requests

---

## 🧱 Project Structure

fake-news-detection/

├── app.py
├── api.py
├── train.py
├── predict.py
├── utils.py
├── requirements.txt
├── README.md

├── data/
├── models/

---

## ⚙️ How to Run

### 1. Install dependencies

pip install -r requirements.txt

---

### 2. Run API

uvicorn api:app --reload

Open:
http://127.0.0.1:8000/docs

---

### 3. Run Streamlit app

streamlit run app.py

---

## 📡 API Example

POST /predict

```json
{
  "title": "Government announces new plan",
  "text": "Officials announced a new economic plan."
}
```

---

## 🧪 Example Output

```json
{
  "prediction": "REAL",
  "confidence": 94.26,
  "probabilities": {
    "FAKE": 5.74,
    "REAL": 94.26
  }
}
```

---

## 📸 Screenshots

(Add screenshots here)

---

## 👨‍💻 Author

Ali Jabak
Computer Science Student | AI & Machine Learning
