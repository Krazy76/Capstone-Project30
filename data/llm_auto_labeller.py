"""
LLM auto-labeller for the Tri-Level Architecture (Macro / Industry / Entity).

Sources:
  1. data/database 1.db                      (table: articles, 87 rows)
  2. data/silver_dataset_final(in).csv
  3. data/Datasets/*.zip                     (raw JSON articles, English only)

Backend: Ollama at http://192.168.0.19:11434, model gemma4:latest.

Macro taxonomy: FRED-MD 8 (McCracken & Ng, 2016).
Industry taxonomy: GICS 11 sectors.
Entity types: open-set {ORG, PER, LOC, PRODUCT, TICKER}.

Mode: overwrite. Re-labels every row unconditionally.
"""

import json
import sqlite3
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


# Config
OLLAMA_URL = "http://192.168.0.19:11434/api/generate"
MODEL = "gemma4:latest"
REQUEST_TIMEOUT = 300  # seconds; long articles + cold model can be slow
MAX_RETRIES = 3
HARD_CHAR_CAP = 8_000   # ~2K tokens; covers the vast majority of articles
NUM_CTX = 4096          # Ollama context window
MAX_CSV_ROWS = 8000     # cap the (corrupted, 61K row) CSV pass

BASE = Path(r"E:\Coding\Python\Capstone-Project30\data")
DB_PATH = BASE / "database 1.db"
CSV_IN = BASE / "silver_dataset_final(in).csv"
CSV_OUT = BASE / "silver_dataset_final_labelled.csv"
ZIP_DIR = BASE / "Datasets"
ZIP_OUT = BASE / "silver_dataset_zips_labelled.csv"
MASTER_OUT = BASE / "silver_dataset_master.csv"
MASTER_DB = BASE / "silver_dataset_master.db"


# Controlled vocabularies
MACRO_RULES = {
    "Output and Income": [
        "GDP report", "gross domestic product", "industrial production",
        "personal income", "economic growth", "recession", "expansion",
        "capacity utilization",
    ],
    "Labor Market": [
        "unemployment rate", "non-farm payrolls", "jobs report",
        "jobless claims", "wage growth", "job openings", "JOLTS",
        "labor force participation", "layoffs", "hiring",
    ],
    "Housing": [
        "housing starts", "building permits", "home sales",
        "mortgage rates", "housing market", "real estate prices",
        "case-shiller", "new home sales", "existing home sales",
    ],
    "Consumption, Orders, and Inventories": [
        "retail sales", "consumer spending", "durable goods orders",
        "factory orders", "business inventories", "manufacturing orders",
        "consumer confidence",
    ],
    "Money and Credit": [
        "money supply", "M1", "M2", "consumer credit", "bank reserves",
        "credit conditions", "loan growth", "bank lending",
    ],
    "Interest and Exchange Rates": [
        "interest rate decision", "federal reserve", "FOMC", "rate hike",
        "rate cut", "jerome powell", "treasury yield", "yield curve",
        "bond yields", "dollar index", "forex", "currency",
        "quantitative tightening",
    ],
    "Prices": [
        "CPI", "consumer price index", "PPI", "producer price index",
        "inflation rate", "core inflation", "deflation", "stagflation",
        "cost of living", "price pressures",
    ],
    "Stock Market": [
        "S&P 500", "Dow Jones", "Nasdaq", "stock market rally",
        "market selloff", "equity markets", "bull market", "bear market",
        "market volatility", "VIX",
    ],
}

INDUSTRY_RULES = {
    "Energy": [
        "oil prices", "energy sector", "crude oil",
        "natural gas stocks", "renewable energy",
    ],
    "Materials": [
        "mining stocks", "chemical industry", "steel prices",
        "gold mining", "raw materials",
    ],
    "Industrials": [
        "aerospace defense", "airline stocks", "machinery",
        "construction industry", "freight logistics",
    ],
    "Consumer Discretionary": [
        "automotive stocks", "luxury goods", "retail trends",
        "travel leisure", "ev market",
    ],
    "Consumer Staples": [
        "food beverage stocks", "household products",
        "supermarket chains", "tobacco industry",
    ],
    "Health Care": [
        "biotech news", "pharmaceutical industry", "medical devices",
        "healthcare insurance", "drug approval",
    ],
    "Financials": [
        "banking sector", "fintech news", "insurance companies",
        "investment banking", "interest income",
    ],
    "Information Technology": [
        "tech stocks", "semiconductor shortage", "software cloud",
        "artificial intelligence", "cybersecurity",
    ],
    "Communication Services": [
        "social media stocks", "telecom industry",
        "streaming services", "advertising revenue",
    ],
    "Utilities": [
        "electric utilities", "water infrastructure",
        "gas utilities", "power grid",
    ],
    "Real Estate": [
        "housing market", "REITs", "commercial real estate",
        "property development",
    ],
}

ENTITY_TYPES = {"ORG", "PER", "LOC", "PRODUCT", "TICKER"}

WEAK_LABEL_RULES = {"Macro": MACRO_RULES, "Industry": INDUSTRY_RULES}


# Prompt construction
def _format_vocab(rules: dict) -> str:
    lines = []
    for label, kws in rules.items():
        lines.append(f"- {label}: {', '.join(kws)}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are a financial news classifier. Read the article and \
return STRICT JSON only, no prose. You must apply three labelling tasks at once:

1. is_financial (boolean): true if the article discusses financial markets, \
the economy, companies, or investing; false otherwise (sports, lifestyle, etc.).

2. macro (list of strings): zero or more labels from the FRED-MD 8 taxonomy \
below. Pick ONLY labels that the article meaningfully discusses. Empty list \
is correct when no macro indicator applies.

{_format_vocab(MACRO_RULES)}

3. industry (list of strings): zero or more GICS sector labels from the list \
below. Pick ONLY sectors the article meaningfully discusses. Empty list is \
correct for pure macro news with no sector angle.

{_format_vocab(INDUSTRY_RULES)}

4. entity (list of objects): named entities mentioned in the article, each as \
{{"name": "...", "type": "..."}} where type is one of: ORG, PER, LOC, PRODUCT, \
TICKER. Deduplicate. Use the canonical English name.

Return JSON in exactly this shape:
{{"is_financial": bool, "macro": [str], "industry": [str], \
"entity": [{{"name": str, "type": str}}]}}

Use ONLY label strings from the vocabularies above. Do not invent new labels."""


def build_user_prompt(title: str, text: str) -> str:
    title = (title or "").strip()
    text = (text or "").strip()
    if len(text) > HARD_CHAR_CAP:
        text = text[:HARD_CHAR_CAP]
    return f"TITLE: {title}\n\nARTICLE:\n{text}"


# Ollama client
def call_ollama(title: str, text: str) -> dict | None:
    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": build_user_prompt(title, text),
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": NUM_CTX},
    }
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            raw = r.json().get("response", "")
            return json.loads(raw)
        except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
            last_err = e
            time.sleep(2 * attempt)
    print(f"  [warn] Ollama failed after {MAX_RETRIES} attempts: {last_err}")
    return None


# Validation
def validate(obj: dict | None) -> dict:
    if not isinstance(obj, dict):
        return {"is_financial": None, "macro": [], "industry": [], "entity": []}

    def _coerce_labels(raw, vocab):
        out = []
        if not isinstance(raw, list):
            return out
        for item in raw:
            if isinstance(item, str):
                cand = item
            elif isinstance(item, dict):
                cand = item.get("name") or item.get("label") or item.get("category") or ""
            else:
                continue
            cand = cand.strip()
            if cand in vocab and cand not in out:
                out.append(cand)
        return out

    macro = _coerce_labels(obj.get("macro", []), MACRO_RULES)
    industry = _coerce_labels(obj.get("industry", []), INDUSTRY_RULES)

    entity = []
    seen = set()
    for e in obj.get("entity", []):
        if not isinstance(e, dict):
            continue
        name = (e.get("name") or "").strip()
        etype = (e.get("type") or "").strip().upper()
        if not name or etype not in ENTITY_TYPES:
            continue
        key = (name.lower(), etype)
        if key in seen:
            continue
        seen.add(key)
        entity.append({"name": name, "type": etype})

    is_fin = obj.get("is_financial")
    if not isinstance(is_fin, bool):
        is_fin = None

    return {
        "is_financial": is_fin,
        "macro": macro,
        "industry": industry,
        "entity": entity,
    }


# Source loaders / writers
def label_database():
    print(f"\n=== Labelling SQLite: {DB_PATH.name} ===")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    rows = cur.execute("SELECT id, title, text FROM articles").fetchall()
    print(f"  {len(rows)} rows to label.")

    for row_id, title, text in tqdm(rows, desc="DB"):
        if not (text or title):
            continue
        result = validate(call_ollama(title, text))
        cur.execute(
            """UPDATE articles
               SET is_financial = ?, macro = ?, industry = ?, entity = ?
               WHERE id = ?""",
            (
                int(result["is_financial"]) if result["is_financial"] is not None else None,
                json.dumps(result["macro"]),
                json.dumps(result["industry"]),
                json.dumps(result["entity"]),
                row_id,
            ),
        )
        conn.commit()  # incremental, resumable
    conn.close()
    print("  DB done.")


def label_csv():
    print(f"\n=== Labelling CSV: {CSV_IN.name} ===")
    df = pd.read_csv(CSV_IN, low_memory=False)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df[df["url"].astype(str).str.match(r"^https://", na=False)].reset_index(drop=True)
    if MAX_CSV_ROWS and len(df) > MAX_CSV_ROWS:
        print(f"  Capping CSV pass: {len(df)} -> {MAX_CSV_ROWS} rows.")
        df = df.head(MAX_CSV_ROWS)
    print(f"  {len(df)} rows to label.")

    # Resume support: if output file exists, skip already-done article_ids.
    done_ids = set()
    if CSV_OUT.exists():
        try:
            done_df = pd.read_csv(CSV_OUT)
            done_ids = set(done_df["article_id"].astype(str).tolist())
            print(f"  Resume: {len(done_ids)} rows already labelled in {CSV_OUT.name}.")
        except Exception:
            pass

    out_cols = [
        "article_id", "title", "url", "published_at", "source_site",
        "is_financial_llm", "macro_llm", "industry_llm", "entity_llm",
    ]
    write_header = not CSV_OUT.exists()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="CSV"):
        aid = str(row.get("article_id", ""))
        if aid in done_ids:
            continue
        title = row.get("title", "")
        text = row.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue

        result = validate(call_ollama(title, text))
        out_row = {
            "article_id": aid,
            "title": title,
            "url": row.get("url", ""),
            "published_at": row.get("published_at", ""),
            "source_site": row.get("source_site", ""),
            "is_financial_llm": result["is_financial"],
            "macro_llm": json.dumps(result["macro"]),
            "industry_llm": json.dumps(result["industry"]),
            "entity_llm": json.dumps(result["entity"]),
        }
        pd.DataFrame([out_row], columns=out_cols).to_csv(
            CSV_OUT, mode="a", header=write_header, index=False
        )
        write_header = False

    print(f"  CSV done. Output: {CSV_OUT}")


def parse_article_json(data: dict) -> dict | None:
    """Extract English articles from a Webhose-style JSON blob."""
    if data.get("language") != "english":
        return None
    return {
        "article_id": data.get("uuid"),
        "title": data.get("title"),
        "text": data.get("text"),
        "published_at": data.get("published"),
        "url": data.get("url"),
        "source_site": (data.get("thread") or {}).get("site"),
        "source_country": (data.get("thread") or {}).get("country"),
    }


def label_zips():
    print(f"\n=== Labelling ZIPs in: {ZIP_DIR} ===")
    if not ZIP_DIR.exists():
        print(f"  [skip] {ZIP_DIR} does not exist.")
        return
    zip_files = sorted(ZIP_DIR.glob("*.zip"))
    print(f"  Found {len(zip_files)} zip files.")

    done_ids = set()
    if ZIP_OUT.exists():
        try:
            done_df = pd.read_csv(ZIP_OUT, usecols=["article_id"])
            done_ids = set(done_df["article_id"].astype(str).tolist())
            print(f"  Resume: {len(done_ids)} articles already labelled.")
        except Exception:
            pass

    out_cols = [
        "article_id", "title", "url", "published_at",
        "source_site", "source_country",
        "is_financial_llm", "macro_llm", "industry_llm", "entity_llm",
    ]
    write_header = not ZIP_OUT.exists()

    for zip_path in tqdm(zip_files, desc="ZIPs"):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                json_names = [n for n in zf.namelist() if n.endswith(".json")]
                for jn in tqdm(json_names, desc=zip_path.name, leave=False):
                    try:
                        with zf.open(jn) as fh:
                            payload = json.loads(fh.read().decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue

                    art = parse_article_json(payload)
                    if not art or not (art.get("text") or "").strip():
                        continue
                    aid = str(art.get("article_id") or "")
                    if not aid or aid in done_ids:
                        continue

                    result = validate(call_ollama(art["title"], art["text"]))
                    out_row = {
                        "article_id": aid,
                        "title": art.get("title", ""),
                        "url": art.get("url", ""),
                        "published_at": art.get("published_at", ""),
                        "source_site": art.get("source_site", ""),
                        "source_country": art.get("source_country", ""),
                        "is_financial_llm": result["is_financial"],
                        "macro_llm": json.dumps(result["macro"]),
                        "industry_llm": json.dumps(result["industry"]),
                        "entity_llm": json.dumps(result["entity"]),
                    }
                    pd.DataFrame([out_row], columns=out_cols).to_csv(
                        ZIP_OUT, mode="a", header=write_header, index=False
                    )
                    write_header = False
                    done_ids.add(aid)
        except zipfile.BadZipFile:
            print(f"  [error] {zip_path.name} is corrupted, skipping.")

    print(f"  ZIPs done. Output: {ZIP_OUT}")


# Entry
def health_check():
    try:
        r = requests.get("http://192.168.0.19:11434/api/tags", timeout=10)
        r.raise_for_status()
        tags = [m["name"] for m in r.json().get("models", [])]
        print(f"Ollama reachable. Models available: {tags}")
        if MODEL not in tags:
            print(f"[warn] {MODEL} not in tag list — request may fail.")
    except Exception as e:
        print(f"[fatal] Cannot reach Ollama at {OLLAMA_URL}: {e}")
        raise SystemExit(1)


def main():
    health_check()
    label_database()
    label_csv()
    label_zips()
    merge_outputs()
    print("\nAll done.")


def merge_outputs():
    """Merge DB + CSV + ZIP labelled outputs into one master CSV.

    Schema: article_id, source, title, url, published_at, source_site,
            is_financial, macro, industry, entity
    Dedupe key: article_id (CSV/ZIP win over DB on collision; ZIP wins over CSV).
    """
    print("\n=== Merging outputs into master CSV ===")
    frames = []

    # 1. DB rows
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        db_df = pd.read_sql_query(
            "SELECT id, title, source, publishedAt, is_financial, "
            "macro, industry, entity FROM articles", conn,
        )
        conn.close()
        db_df = db_df.rename(columns={
            "id": "article_id",
            "publishedAt": "published_at",
            "source": "source_site",
        })
        db_df["article_id"] = "db_" + db_df["article_id"].astype(str)
        db_df["url"] = ""
        db_df["source"] = "db"
        # Normalize is_financial to bool/None
        db_df["is_financial"] = db_df["is_financial"].apply(
            lambda v: bool(v) if pd.notna(v) else None
        )
        frames.append(db_df[[
            "article_id", "source", "title", "url", "published_at",
            "source_site", "is_financial", "macro", "industry", "entity",
        ]])
        print(f"  DB rows: {len(db_df)}")

    # 2. CSV pass
    if CSV_OUT.exists():
        csv_df = pd.read_csv(CSV_OUT)
        csv_df = csv_df.rename(columns={
            "is_financial_llm": "is_financial",
            "macro_llm": "macro",
            "industry_llm": "industry",
            "entity_llm": "entity",
        })
        csv_df["source"] = "csv"
        for col in ("url", "source_site"):
            if col not in csv_df.columns:
                csv_df[col] = ""
        frames.append(csv_df[[
            "article_id", "source", "title", "url", "published_at",
            "source_site", "is_financial", "macro", "industry", "entity",
        ]])
        print(f"  CSV rows: {len(csv_df)}")

    # 3. ZIP pass
    if ZIP_OUT.exists():
        zip_df = pd.read_csv(ZIP_OUT)
        zip_df = zip_df.rename(columns={
            "is_financial_llm": "is_financial",
            "macro_llm": "macro",
            "industry_llm": "industry",
            "entity_llm": "entity",
        })
        zip_df["source"] = "zip"
        frames.append(zip_df[[
            "article_id", "source", "title", "url", "published_at",
            "source_site", "is_financial", "macro", "industry", "entity",
        ]])
        print(f"  ZIP rows: {len(zip_df)}")

    if not frames:
        print("  [skip] Nothing to merge.")
        return

    # Concatenate; later sources overwrite earlier on article_id collision.
    # Order: db, csv, zip → ZIP wins, then CSV, then DB.
    master = pd.concat(frames, ignore_index=True)
    before = len(master)
    master = master.drop_duplicates(subset="article_id", keep="last").reset_index(drop=True)
    print(f"  Merged: {before} rows → {len(master)} after dedupe.")

    master.to_csv(MASTER_OUT, index=False)
    print(f"  Master CSV written: {MASTER_OUT}")

    # Also write to SQLite (table: articles, replaced on each run).
    conn = sqlite3.connect(MASTER_DB)
    master.to_sql("articles", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_article_id ON articles(article_id)")
    conn.commit()
    conn.close()
    print(f"  Master DB written:  {MASTER_DB}")


if __name__ == "__main__":
    main()
