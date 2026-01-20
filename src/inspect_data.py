import json
import pandas as pd

with open('data/all_reviews.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

print("\n=== HEAD ===")
print(df.head())

print("\n=== COLUMNS ===")
print(df.columns)

print("\n=== SAMPLE ROW ===")
print(df.iloc[0])
