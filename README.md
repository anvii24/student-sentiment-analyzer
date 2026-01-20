# Student Feedback Sentiment Analyzer

An ML-powered sentiment analysis system that classifies student course feedback as **Positive**, **Neutral**, or **Negative**.

---

## Features
- Real-time sentiment prediction
- Trained on real RateMyProfessors reviews
- Logistic Regression text classification model
- TF-IDF feature extraction
- Clean Streamlit UI
- Keyword override for obvious negative cases

---

## Model Details
- **Algorithm:** Logistic Regression
- **Vectorization:** TF-IDF (5000 features)
- **Dataset:** Kaggle (RateMyProfessors reviews)
- **Labels:** Positive / Neutral / Negative

---

## Project Structure
student-sentiment-analyzer/
├── app/
│   └── app.py                     # Streamlit UI
│
├── data/                          # Dataset files
│   ├── all_reviews.json
│   ├── cleaned_reviews.csv
│   ├── flat_reviews.csv
│   └── raw_feedback.csv
│
├── models/                        # Saved ML artifacts
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── src/                           # Processing & training scripts
│   ├── flatten_data.py
│   ├── inspect_data.py
│   ├── predict.py
│   ├── preprocess_data.py
│   ├── scrape_rmp.py
│   └── train_model.py
│
├── venv/                          # Virtual environment 
├── .gitignore
├── README.md
└── requirements.txt

