"""News article collector — fetches financial articles from NewsAPI and appends
new rows to the master CSV, skipping duplicates by URL.

Run:
    python -m src.data_pipeline.collector
    python -m src.data_pipeline.collector --pages 3 --page-size 100
"""

from __future__ import annotations

import argparse
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = _PROJECT_ROOT / "data" / "silver_dataset_master_with_text.csv"

NEWSAPI_URL = "https://newsapi.org/v2/everything"
FINANCIAL_DOMAINS = (
    "reuters.com,bloomberg.com,wsj.com,ft.com,cnbc.com,"
    "marketwatch.com,forbes.com,economist.com"
)
SEARCH_QUERY = "stock market OR interest rates OR inflation OR GDP OR earnings"


def fetch_articles(api_key: str, page: int = 1, page_size: int = 100) -> list[dict]:
    params = {
        "q": SEARCH_QUERY,
        "domains": FINANCIAL_DOMAINS,
        "language": "en",
        "pageSize": page_size,
        "page": page,
        "sortBy": "publishedAt",
        "apiKey": api_key,
    }
    resp = requests.get(NEWSAPI_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "ok":
        raise RuntimeError(f"NewsAPI error: {data.get('message')}")
    return data["articles"]


def articles_to_rows(articles: list[dict]) -> list[dict]:
    rows = []
    for a in articles:
        title = a.get("title") or ""
        text = a.get("content") or a.get("description") or ""
        url = a.get("url") or ""
        if not url or not title:
            continue
        rows.append({
            "article_id": str(uuid.uuid4()),
            "source": "newsapi",
            "title": title.strip(),
            "text": text.strip(),
            "url": url.strip(),
            "published_at": a.get("publishedAt") or datetime.now(timezone.utc).isoformat(),
            "source_site": (a.get("source") or {}).get("name") or "",
            "is_financial": None,
            "macro": None,
            "industry": None,
            "entity": None,
            "label_version": None,
        })
    return rows


def collect(
    csv_path: Path = DEFAULT_CSV,
    pages: int = 1,
    page_size: int = 100,
) -> None:
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise RuntimeError("NEWSAPI_KEY not set in .env")

    # Load existing CSV to get known URLs for deduplication.
    if csv_path.exists():
        existing = pd.read_csv(csv_path, low_memory=False)
        known_urls: set[str] = set(existing["url"].dropna().tolist())
        print(f"[collector] loaded {len(existing):,} existing rows, {len(known_urls):,} known URLs")
    else:
        existing = pd.DataFrame()
        known_urls = set()
        print("[collector] no existing CSV found — will create one")

    new_rows = []
    for page in range(1, pages + 1):
        print(f"[collector] fetching page {page}/{pages} ...")
        articles = fetch_articles(api_key, page=page, page_size=page_size)
        rows = articles_to_rows(articles)

        for row in rows:
            if row["url"] in known_urls:
                continue  # duplicate — skip
            new_rows.append(row)
            known_urls.add(row["url"])

    if not new_rows:
        print("[collector] no new articles found — CSV unchanged")
        return

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    combined.to_csv(csv_path, index=False)
    print(f"[collector] appended {len(new_rows):,} new articles → {csv_path.name} now has {len(combined):,} rows")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to master CSV")
    p.add_argument("--pages", type=int, default=1, help="Number of API pages to fetch")
    p.add_argument("--page-size", type=int, default=100, help="Articles per page (max 100)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect(csv_path=Path(args.csv), pages=args.pages, page_size=args.page_size)
