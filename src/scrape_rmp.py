import requests
import pandas as pd
import time

# Professor IDs to scrape
professor_ids = [
    "622173",
    "2857198",
    "2992181",
    "1736941",
    "1393534",
    "108405",
    "2817314",
    "361101"
]

def fetch_reviews(prof_id, max_pages=5):
    url = "https://www.ratemyprofessors.com/graphql"
    headers = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.ratemyprofessors.com/"
}

    
    reviews = []
    
    for page in range(max_pages):
        payload = {
            "query": """
            query RatingsListQuery($id: ID!, $page: Int!) {
              node(id: $id) {
                ... on Professor {
                  ratings(page: $page) {
                    edges {
                      node {
                        id
                        comment
                        rating
                      }
                    }
                  }
                }
              }
            }
            """,
            "variables": {"id": prof_id, "page": page}
        }
        
        res = requests.post(url, json=payload, headers=headers)
        data = res.json()
        
        try:
            edges = data["data"]["node"]["ratings"]["edges"]
        except:
            break
        
        if not edges:
            break
        
        for edge in edges:
            node = edge["node"]
            if node["comment"]:
                reviews.append({
                    "comment": node["comment"],
                    "rating": node["rating"]
                })
        
        time.sleep(1)
    
    return reviews

def label_sentiment(rating):
    rating = int(rating)
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

all_reviews = []

for pid in professor_ids:
    print(f"Fetching reviews for professor {pid}...")
    reviews = fetch_reviews(pid)
    
    for r in reviews:
        all_reviews.append({
            "text": r["comment"],
            "rating": r["rating"],
            "sentiment": label_sentiment(r["rating"])
        })

df = pd.DataFrame(all_reviews)
df.to_csv("data/raw_feedback.csv", index=False)
print("Saved data/raw_feedback.csv with", len(df), "rows!")
