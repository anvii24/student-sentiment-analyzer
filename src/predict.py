import joblib

# Loading the model & vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

def predict_sentiment(text):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return pred

if __name__ == "__main__":
    while True:
        user_input = input("\nEnter review (or 'quit'): ")
        if user_input.lower() == "quit":
            break
        sentiment = predict_sentiment(user_input)
        print("Predicted Sentiment:", sentiment)
