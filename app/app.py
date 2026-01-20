import streamlit as st
import joblib

# Loading the model & vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.markdown("""
    <style>
    .centered-title {
        text-align: center;
        font-size: 32px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Centered title
st.markdown('<div class="centered-title">⭐ Student Feedback Sentiment Analyzer</div>', unsafe_allow_html=True)

# Input section text area
st.markdown("**Enter student feedback:**")
text = st.text_area("", height=150, placeholder="Type your review here...")

# Analyzing button
if st.button("Analyze Sentiment"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        text_lower = text.lower()

        # Quick keyword-based override for obvious negatives
        neg_words = ["bad", "badly", "terrible", "awful", "worst", "poor", "horrible", "sad", "disappointing"]
        neg_phrases = ["not good", "not great", "not helpful", "very bad", "waste of time", "not worth", "not recommended"]

        if any(word in text_lower for word in neg_words) or any(phrase in text_lower for phrase in neg_phrases):
            pred = "Negative"
        else:
            # ML model prediction
            X = vectorizer.transform([text])
            pred = model.predict(X)[0]

        # Displaying the result
        if pred == "Positive":
            st.success(f"🎉 Sentiment: **{pred}**")
        elif pred == "Neutral":
            st.info(f"😐 Sentiment: **{pred}**")
        else:
            st.error(f"⚠️ Sentiment: **{pred}**")



