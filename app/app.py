import streamlit as st
import joblib
import os

# Loading the model & vectorizer
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "..", "models", "sentiment_model.pkl")
vectorizer_path = os.path.join(base_path, "..", "models", "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# UI styling
st.set_page_config(page_title="Student Sentiment Analyzer", layout="centered")

st.markdown(
    """
    <h1 style='text-align:center; font-size:42px;'>
    ⭐ Student Feedback Sentiment Analyzer
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("")

# Input box
st.markdown("#### Enter student feedback:")
text_input = st.text_area("", height=150, placeholder="Type feedback here...")

# Analyzing button
if st.button("Analyze Sentiment"):
    if text_input.strip() == "":
        st.warning("Please enter some feedback before analyzing.")
    else:
        # Transforming the input & predicting
        transformed = vectorizer.transform([text_input])
        pred = model.predict(transformed)[0]

        # Displaying the result
        color_map = {
            "Positive": "#1A7F37",
            "Neutral": "#6E7781",
            "Negative": "#D1242F"
        }

        st.markdown(
            f"""
            <div style='margin-top:20px; padding:15px; border-radius:8px; 
            text-align:center; font-size:22px; font-weight:600; 
            background-color:{color_map.get(pred, "#333")}; color:white;'>
            Sentiment: {pred}
            </div>
            """,
            unsafe_allow_html=True
        )
