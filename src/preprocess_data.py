import pandas as pd

# Loading the flattened dataset
df = pd.read_csv("data/flat_reviews.csv")

# Keeping only the columns needed
df = df[['Comment', 'Quality']]

# Dropping rows without text or rating
df = df.dropna(subset=['Comment', 'Quality'])

# Converting Quality to float
df['Quality'] = df['Quality'].astype(float)

# Creating sentiment labels
def label_sentiment(q):
    if q >= 4.0:
        return "Positive"
    elif q == 3.0:
        return "Neutral"
    else:
        return "Negative"

df['Sentiment'] = df['Quality'].apply(label_sentiment)

print(df.head())
print(df['Sentiment'].value_counts())

# Saving the cleaned dataset
df.to_csv("data/cleaned_reviews.csv", index=False)
print("Saved -> data/cleaned_reviews.csv")
