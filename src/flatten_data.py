import json
import pandas as pd

with open('data/all_reviews.json', 'r', encoding='utf-8') as f:
    raw = json.load(f)

# raw is a list of lists of dictionaries
all_reviews = []

for prof_reviews in raw:
    if isinstance(prof_reviews, list):
        for review in prof_reviews:
            all_reviews.append(review)

df = pd.DataFrame(all_reviews)

print("Flattened shape:", df.shape)
print(df.head())
print(df.columns)

df.to_csv("data/flat_reviews.csv", index=False)
print("Saved -> data/flat_reviews.csv")
