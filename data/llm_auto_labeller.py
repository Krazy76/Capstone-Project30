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

Current pass: enriched-MACRO_RULES re-label (FRED-MD appendix derived, ~180
keywords across 8 groups, up from ~75). Caps lifted: NUM_CTX=8192,
HARD_CHAR_CAP=24K, MAX_CSV_ROWS=None, resume guards removed so every row
gets re-scored by the richer prompt.
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
OLLAMA_URL = "http://192.168.0.27:11434/api/generate"
MODEL = "gemma4:latest"
REQUEST_TIMEOUT = 300  # seconds; long articles + cold model can be slow
MAX_RETRIES = 3
# Caps lifted for the enriched-MACRO_RULES re-label pass. Bumping NUM_CTX to
# 8192 and the char cap to 24K (~6K tokens) lets the model see the full body
# of essentially every article in the corpus. MAX_CSV_ROWS=None removes the
# 8K cap on the CSV pass so the full ~61K-row CSV is processed.
HARD_CHAR_CAP = 24_000  # ~6K tokens; covers >99% of articles end-to-end
NUM_CTX = 8192          # Ollama context window (raised from 4096)
MAX_CSV_ROWS = None     # no cap; process entire CSV

BASE = Path(r"E:\Coding\Python\Capstone-Project30\data")
DB_PATH = BASE / "database 1.db"
CSV_IN = BASE / "silver_dataset_final(in).csv"
CSV_OUT = BASE / "silver_dataset_final_labelled.csv"
ZIP_DIR = BASE / "Datasets"
ZIP_OUT = BASE / "silver_dataset_zips_labelled.csv"
MASTER_OUT = BASE / "silver_dataset_master.csv"
MASTER_DB = BASE / "silver_dataset_master.db"
MASTER_WITH_TEXT_OUT = BASE / "silver_dataset_master_with_text.csv"


# Controlled vocabularies
# Macro vocabulary derived from the FRED-MD appendix (8 groups, ~134
# underlying monthly indicators). Keywords cover the high-level concept,
# the indicator names that appear in financial news, and the sector- or
# region-specific breakouts that articles commonly report on.
# Doubling the keyword density vs the original short list is intentional:
# the original 25% macro coverage ceiling on the silver corpus traced to
# under-firing on company-level news that should have mapped to Labor
# Market (sector payrolls), Prices (oil/PPI), Money and Credit (real-estate
# / vehicle / consumer loans), and Interest and Exchange Rates (corporate
# bond yields, bilateral FX).
MACRO_RULES = {
    "Output and Income": [
        "GDP", "gross domestic product", "industrial production", "IP index",
        "personal income", "real personal income", "manufacturing output",
        "factory output", "capacity utilization", "industrial capacity",
        "consumer goods production", "durable goods production",
        "nondurable goods production", "business equipment production",
        "industrial materials", "manufacturing index", "economic growth",
        "recession", "expansion", "GDP growth", "real GDP",
    ],
    "Labor Market": [
        "unemployment rate", "non-farm payrolls", "nonfarm payrolls",
        "jobs report", "jobless claims", "initial claims", "continuing claims",
        "wage growth", "average hourly earnings", "average weekly hours",
        "overtime hours", "JOLTS", "job openings", "labor force participation",
        "layoffs", "hiring", "job cuts", "job losses", "workforce reduction",
        "manufacturing employment", "manufacturing jobs", "construction jobs",
        "retail employment", "wholesale employment", "government payrolls",
        "service sector jobs", "mining employment", "help wanted",
        "civilian labor force", "civilian employment", "unemployment duration",
    ],
    "Housing": [
        "housing starts", "building permits", "home sales", "new home sales",
        "existing home sales", "mortgage rates", "mortgage applications",
        "housing market", "real estate prices", "case-shiller", "home prices",
        "home builder", "homebuilder confidence", "single-family", "multi-family",
        "regional housing", "housing inventory", "private housing permits",
        "construction permits",
    ],
    "Consumption, Orders, and Inventories": [
        "retail sales", "consumer spending", "personal consumption", "PCE",
        "real personal consumption", "durable goods orders", "factory orders",
        "new orders", "unfilled orders", "business inventories",
        "inventory to sales", "wholesale sales", "wholesale inventories",
        "consumer sentiment", "consumer confidence", "Michigan sentiment",
        "Conference Board confidence", "manufacturing orders",
        "capital goods orders", "consumer goods orders", "trade sales",
    ],
    "Money and Credit": [
        "money supply", "M1", "M2", "monetary base", "consumer credit",
        "bank reserves", "credit conditions", "loan growth", "bank lending",
        "commercial loans", "industrial loans", "C&I loans",
        "real estate loans", "mortgage credit", "auto loans", "vehicle loans",
        "credit card debt", "consumer debt", "nonrevolving credit",
        "revolving credit", "bank credit", "securities holdings",
        "commercial paper outstanding", "money stock",
    ],
    "Interest and Exchange Rates": [
        "interest rate decision", "federal reserve", "FOMC", "rate hike",
        "rate cut", "jerome powell", "fed funds rate", "fed funds",
        "treasury yield", "yield curve", "bond yields", "T-bill",
        "3-month treasury", "10-year treasury", "30-year treasury",
        "AAA bond yield", "BAA bond yield", "corporate bond yield",
        "credit spread", "yield spread", "commercial paper rate",
        "dollar index", "DXY", "trade weighted dollar", "forex", "currency",
        "yen", "japanese yen", "euro", "british pound", "pound sterling",
        "yuan", "renminbi", "swiss franc", "canadian dollar", "exchange rate",
        "bilateral exchange rate", "quantitative tightening", "quantitative easing",
    ],
    "Prices": [
        "CPI", "consumer price index", "PPI", "producer price index",
        "inflation rate", "core inflation", "headline inflation", "deflation",
        "disinflation", "stagflation", "cost of living", "price pressures",
        "PCE deflator", "personal consumption deflator", "core PCE",
        "oil prices", "crude oil prices", "WTI", "brent crude", "gasoline prices",
        "metals prices", "commodity prices", "import prices", "export prices",
        "wholesale prices", "intermediate materials prices", "crude materials prices",
        "apparel prices", "medical care prices", "transportation prices",
        "services inflation", "goods inflation", "food prices", "energy prices",
    ],
    "Stock Market": [
        "S&P 500", "Dow Jones", "Nasdaq", "stock market rally", "market selloff",
        "equity markets", "bull market", "bear market", "market volatility", "VIX",
        "P/E ratio", "price-earnings ratio", "dividend yield", "stock index",
        "market correction", "all-time high", "record close", "index futures",
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


_MACRO_LABELS = list(MACRO_RULES.keys())
_INDUSTRY_LABELS = list(INDUSTRY_RULES.keys())

# Build keyword hints: "- LabelName (keywords: kw1, kw2, ...)"
def _format_vocab_with_hints(rules: dict) -> str:
    lines = []
    for label, kws in rules.items():
        lines.append(f"  - {label} (e.g. {', '.join(kws[:6])}{'...' if len(kws)>6 else ''})")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are a financial news classifier. Read the article and \
return STRICT JSON only, no prose.

1. is_financial (boolean): true if the article discusses financial markets, \
the economy, companies, or investing; false otherwise.

2. macro (list of strings): RETURN ONLY strings from this exact list:
{json.dumps(_MACRO_LABELS)}
Apply a label only when the macro indicator is the article's PRIMARY topic or a \
MAJOR driver of the story - not a passing mention or distant side effect.
Label definitions and boundaries:
  - Stock Market: aggregate index movements ONLY (S&P 500, Dow Jones, Nasdaq, VIX, \
market-wide sentiment, index futures). Do NOT apply to individual company stock price \
changes, earnings beats/misses, or single-stock news.
  - Prices: economy-wide price levels (CPI, PPI, inflation rate, oil/commodity prices, \
import/export prices). Not individual asset valuations or company pricing.
  - Output and Income: national/sectoral production and income aggregates (GDP, \
industrial production, capacity utilization). Not individual company revenue.
  - All other labels: apply when the article substantively reports the indicator itself \
or its direct economy-wide impact.
Empty list when no macro indicator is a primary or major topic.
Reference keywords (do NOT return these - return the label name above):
{_format_vocab_with_hints(MACRO_RULES)}

3. industry (list of strings): RETURN ONLY strings from this exact list:
{json.dumps(_INDUSTRY_LABELS)}
Empty list for pure macro news with no sector angle.
Reference keywords:
{_format_vocab_with_hints(INDUSTRY_RULES)}

4. entity (list of objects): named entities, each as \
{{"name": "...", "type": "..."}} where type is one of: ORG, PER, LOC, PRODUCT, \
TICKER. Deduplicate. Use canonical English name.

Return JSON in exactly this shape (no other keys, no prose):
{{"is_financial": bool, "macro": [str], "industry": [str], \
"entity": [{{"name": str, "type": str}}]}}"""


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


# Keyword → group-name reverse index for fallback recovery.
# If Gemma returns a keyword ("inflation") instead of the group name ("Prices"),
# we can still recover the correct label rather than silently dropping it.
_MACRO_KW_TO_GROUP: dict[str, str] = {
    kw.lower(): group
    for group, kws in MACRO_RULES.items()
    for kw in kws
}
_INDUSTRY_KW_TO_GROUP: dict[str, str] = {
    kw.lower(): group
    for group, kws in INDUSTRY_RULES.items()
    for kw in kws
}


# Validation
def validate(obj: dict | None) -> dict:
    if not isinstance(obj, dict):
        return {"is_financial": None, "macro": [], "industry": [], "entity": []}

    def _coerce_labels(raw, vocab, kw_index):
        """Accept group names directly; fall back to keyword reverse-lookup."""
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
            # Primary: exact group-name match.
            if cand in vocab:
                if cand not in out:
                    out.append(cand)
                continue
            # Fallback: Gemma returned a keyword → map to its parent group.
            group = kw_index.get(cand.lower())
            if group and group not in out:
                out.append(group)
        return out

    macro = _coerce_labels(obj.get("macro", []), MACRO_RULES, _MACRO_KW_TO_GROUP)
    industry = _coerce_labels(obj.get("industry", []), INDUSTRY_RULES, _INDUSTRY_KW_TO_GROUP)

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

    # Fresh pass: delete any prior labelled output so existing rows are
    # re-labelled with the enriched MACRO_RULES rather than skipped.
    if CSV_OUT.exists():
        print(f"  Overwrite: removing prior {CSV_OUT.name}")
        CSV_OUT.unlink()

    out_cols = [
        "article_id", "title", "url", "published_at", "source_site",
        "is_financial_llm", "macro_llm", "industry_llm", "entity_llm",
    ]
    write_header = True

    for _, row in tqdm(df.iterrows(), total=len(df), desc="CSV"):
        aid = str(row.get("article_id", ""))
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

    # Fresh pass: delete prior labelled output so enriched MACRO_RULES apply.
    if ZIP_OUT.exists():
        print(f"  Overwrite: removing prior {ZIP_OUT.name}")
        ZIP_OUT.unlink()
    done_ids: set = set()  # only used for in-run dedupe across zips

    out_cols = [
        "article_id", "title", "text", "url", "published_at",
        "source_site", "source_country",
        "is_financial_llm", "macro_llm", "industry_llm", "entity_llm",
    ]
    write_header = True

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
                        "text": art.get("text", ""),   # full body — included for master
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
        base = OLLAMA_URL.rsplit("/", 2)[0]  # strip /api/generate
        r = requests.get(f"{base}/api/tags", timeout=10)
        r.raise_for_status()
        tags = [m["name"] for m in r.json().get("models", [])]
        print(f"Ollama reachable. Models available: {tags}")
        if MODEL not in tags:
            print(f"[warn] {MODEL} not in tag list — request may fail.")
    except Exception as e:
        print(f"[fatal] Cannot reach Ollama at {OLLAMA_URL}: {e}")
        raise SystemExit(1)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--merge-only", action="store_true",
        help="Skip labelling; merge whatever labelled files exist and produce "
             "silver_dataset_master_with_text.csv.",
    )
    p.add_argument(
        "--relabel-master", action="store_true",
        help="Re-label rows in silver_dataset_master_with_text.csv with the "
             "current prompt. Marks each processed row with label_version=v2.",
    )
    p.add_argument(
        "--max-rows", type=int, default=None,
        help="Stop after processing this many rows (for staged runs, e.g. 20000).",
    )
    args = p.parse_args()

    if args.merge_only:
        merge_outputs_with_text()
    elif args.relabel_master:
        health_check()
        relabel_master(max_rows=args.max_rows)
    else:
        health_check()
        label_zips()
        merge_outputs_with_text()
    print("\nAll done.")


LABEL_VERSION = "v2"  # bump this when the prompt changes meaningfully


def relabel_master(
    path: Path = MASTER_WITH_TEXT_OUT,
    checkpoint: Path | None = None,
    max_rows: int | None = None,
):
    """Re-label silver_dataset_master_with_text.csv with the current prompt.

    Columns updated per row:
      is_financial, macro, industry, entity  -- new Gemma labels
      label_version                          -- set to LABEL_VERSION ("v2")

    Rows NOT yet processed keep label_version=NaN (or prior value).
    Training pipeline should filter: df[df["label_version"] == "v2"].

    Resume: a .ckpt sidecar tracks the last completed row index.
    max_rows: stop after this many rows have been processed (for staged runs).
    """
    import time as _time

    if checkpoint is None:
        checkpoint = path.with_suffix(".relabel_ckpt")

    print(f"\n=== Re-labelling master: {path.name} ===")
    df = pd.read_csv(path, low_memory=False)
    n = len(df)

    # Ensure label_version column exists.
    if "label_version" not in df.columns:
        df["label_version"] = None

    # Resume: find last completed index.
    start = 0
    if checkpoint.exists():
        try:
            start = int(checkpoint.read_text().strip()) + 1
            print(f"  Resuming from row {start}.")
        except Exception:
            pass

    stop = min(n, start + max_rows) if max_rows else n
    already_done = int(df["label_version"].eq(LABEL_VERSION).sum())
    print(f"  {n} rows total | {already_done} already v2 | "
          f"processing rows {start}-{stop-1} ({stop - start} rows)")

    t0 = _time.time()
    macro_hits = 0
    pbar = tqdm(range(start, stop), total=stop - start, desc="relabel",
                unit="row", dynamic_ncols=True)
    for i in pbar:
        row = df.iloc[i]
        title = str(row.get("title") or "")
        text  = str(row.get("text")  or "")
        if not text.strip() and not title.strip():
            df.at[i, "label_version"] = LABEL_VERSION
            continue

        result = validate(call_ollama(title, text))
        df.at[i, "is_financial"]  = result["is_financial"]
        df.at[i, "macro"]         = json.dumps(result["macro"])
        df.at[i, "industry"]      = json.dumps(result["industry"])
        df.at[i, "entity"]        = json.dumps(result["entity"])
        df.at[i, "label_version"] = LABEL_VERSION

        if result["macro"]:
            macro_hits += 1
        pbar.set_postfix(macro=macro_hits, v2=int(df["label_version"].eq(LABEL_VERSION).sum()), row=i)

        # Flush every 100 rows.
        if (i + 1) % 100 == 0:
            df.to_csv(path, index=False)
            checkpoint.write_text(str(i))

    # Final write.
    df.to_csv(path, index=False)
    checkpoint.unlink(missing_ok=True)
    elapsed = _time.time() - t0
    v2_total = int(df["label_version"].eq(LABEL_VERSION).sum())
    print(f"  Done. processed={stop - start}  v2_total={v2_total}  "
          f"macro_hits={macro_hits}  time={elapsed:.0f}s")


def merge_outputs_with_text():
    """Merge DB + CSV + ZIP labelled outputs into one master CSV **with text**.

    Text sourcing:
      - DB   : SELECT text FROM articles in SQLite
      - CSV  : join CSV_OUT (labels) with CSV_IN (source, has text) on article_id
      - ZIP  : ZIP_OUT already contains text column (written by label_zips)

    Output schema:
      article_id, source, title, text, url, published_at, source_site,
      is_financial, macro, industry, entity

    Dedupe key: article_id. Priority: ZIP > CSV > DB on collision.
    Written to: silver_dataset_master_with_text.csv
    """
    print("\n=== Merging outputs into master-with-text CSV ===")
    FINAL_COLS = [
        "article_id", "source", "title", "text", "url", "published_at",
        "source_site", "is_financial", "macro", "industry", "entity",
    ]
    frames = []

    # 1. DB rows (87 rows, already labelled, text lives in SQLite)
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        db_df = pd.read_sql_query(
            "SELECT id, title, text, source, publishedAt, is_financial, "
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
        db_df["is_financial"] = db_df["is_financial"].apply(
            lambda v: bool(v) if pd.notna(v) else None
        )
        db_df["text"] = db_df["text"].fillna("")
        frames.append(db_df[FINAL_COLS])
        print(f"  DB rows: {len(db_df)}")

    # 2. CSV-labelled rows (~29K in CSV_OUT) — join with CSV_IN to get text.
    if CSV_OUT.exists():
        print(f"  Loading labels from {CSV_OUT.name} ...")
        csv_lbl = pd.read_csv(CSV_OUT, low_memory=False)
        csv_lbl = csv_lbl.rename(columns={
            "is_financial_llm": "is_financial",
            "macro_llm": "macro",
            "industry_llm": "industry",
            "entity_llm": "entity",
        })
        csv_lbl["source"] = "csv"
        for col in ("url", "source_site"):
            if col not in csv_lbl.columns:
                csv_lbl[col] = ""

        # Join text from original source CSV.
        print(f"  Loading text from {CSV_IN.name} (may take a moment) ...")
        csv_src = pd.read_csv(CSV_IN, low_memory=False, usecols=["article_id", "text"])
        csv_src["article_id"] = csv_src["article_id"].astype(str)
        csv_lbl["article_id"] = csv_lbl["article_id"].astype(str)
        csv_lbl = csv_lbl.merge(csv_src, on="article_id", how="left")
        csv_lbl["text"] = csv_lbl["text"].fillna("")
        frames.append(csv_lbl[FINAL_COLS])
        print(f"  CSV rows: {len(csv_lbl)}  (text attached: {(csv_lbl['text'] != '').sum()})")

    # 3. ZIP rows — ZIP_OUT now contains text column written by label_zips().
    if ZIP_OUT.exists():
        zip_df = pd.read_csv(ZIP_OUT, low_memory=False)
        zip_df = zip_df.rename(columns={
            "is_financial_llm": "is_financial",
            "macro_llm": "macro",
            "industry_llm": "industry",
            "entity_llm": "entity",
        })
        zip_df["source"] = "zip"
        zip_df["text"] = zip_df.get("text", pd.Series(dtype=str)).fillna("")
        for col in ("url", "source_site", "source_country"):
            if col not in zip_df.columns:
                zip_df[col] = ""
        frames.append(zip_df[FINAL_COLS])
        print(f"  ZIP rows: {len(zip_df)}  (text attached: {(zip_df['text'] != '').sum()})")

    if not frames:
        print("  [skip] Nothing to merge.")
        return

    # Concatenate — ZIP wins over CSV wins over DB on article_id collision.
    master = pd.concat(frames, ignore_index=True)
    before = len(master)
    master = master.drop_duplicates(subset="article_id", keep="last").reset_index(drop=True)
    print(f"  Merged: {before} rows -> {len(master)} after dedupe.")

    # Drop rows with empty text — these can't be used for training.
    has_text = master["text"].str.strip().str.len() > 0
    master = master[has_text].reset_index(drop=True)
    print(f"  After dropping empty-text rows: {len(master)}")

    master.to_csv(MASTER_WITH_TEXT_OUT, index=False)
    print(f"  Master-with-text CSV written: {MASTER_WITH_TEXT_OUT}")

    # Summary stats.
    fin = master["is_financial"].apply(lambda v: v is True or v == 1 or str(v).lower() == "true")
    print(f"  is_financial=True: {fin.sum()} / {len(master)}")


def merge_outputs():
    """Legacy merge without text. Kept for reference — not called by main()."""
    pass


if __name__ == "__main__":
    main()
