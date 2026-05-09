"""Quick test to confirm NewsAPI key works and articles are fetchable."""

import os
from dotenv import load_dotenv
import requests

load_dotenv()

API_KEY = os.getenv("NEWSAPI_KEY")
URL = "https://newsapi.org/v2/everything"

params = {
    "q": "stock market OR interest rates OR inflation OR GDP OR earnings",
    "domains": "reuters.com,bloomberg.com,wsj.com,ft.com,cnbc.com,marketwatch.com,forbes.com,economist.com",
    "language": "en",
    "pageSize": 5,
    "sortBy": "publishedAt",
    "apiKey": API_KEY,
}

response = requests.get(URL, params=params)
data = response.json()

if data["status"] != "ok":
    print(f"ERROR: {data}")
else:
    print(f"Success! Total results available: {data['totalResults']}\n")
    for i, article in enumerate(data["articles"], 1):
        print(f"[{i}] {article['title']}")
        print(f"    URL: {article['url']}")
        print(f"    Source: {article['source']['name']}")
        print(f"    Published: {article['publishedAt']}")
        print()
