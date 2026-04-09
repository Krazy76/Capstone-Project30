"""
Pre-labelling EDA: char / word / token statistics across all three data sources.

Purpose: inform num_ctx and HARD_CHAR_CAP choices in llm_auto_labeller.py
without waiting for labelling to finish. Tokens are counted with the
roberta-base tokenizer because RoBERTa is the strongest baseline backbone in
our comparison set — the distribution it sees is what matters for max_length
decisions in the downstream model.

Outputs:
  - data/eda.csv          per-article rows: source, article_id, chars, words, tokens
  - data/eda_summary.csv  per-source summary statistics

Safe to run while llm_auto_labeller.py is labelling — only reads source files.
"""

import json
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

BASE = Path(r"E:\Coding\Python\Capstone-Project30\data")
DB_PATH = BASE / "database 1.db"
CSV_IN = BASE / "silver_dataset_final(in).csv"
ZIP_DIR = BASE / "Datasets"
EDA_OUT = BASE / "eda.csv"
SUMMARY_OUT = BASE / "eda_summary.csv"

# Sampling: full DB (small), capped CSV/ZIP for speed.
CSV_SAMPLE = 5000
ZIP_SAMPLE_PER_FILE = 50  # 77 zips × 50 ≈ 3850 articles

TOKENIZER_NAME = "roberta-base"


def measure(text: str, tokenizer) -> tuple[int, int, int]:
    text = text or ""
    chars = len(text)
    words = len(text.split())
    # add_special_tokens=False so the count reflects content, not framing
    tokens = len(tokenizer.encode(text, add_special_tokens=False, truncation=False))
    return chars, words, tokens


def collect_db(tokenizer) -> list[dict]:
    if not DB_PATH.exists():
        print(f"  [skip] {DB_PATH} missing")
        return []
    print(f"\n=== DB: {DB_PATH.name} ===")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = conn.execute("SELECT id, title, text FROM articles").fetchall()
    conn.close()
    out = []
    for row_id, title, text in tqdm(rows, desc="DB"):
        full = f"{title or ''}\n\n{text or ''}"
        chars, words, tokens = measure(full, tokenizer)
        out.append({
            "source": "db",
            "article_id": f"db_{row_id}",
            "chars": chars,
            "words": words,
            "tokens": tokens,
        })
    return out


def collect_csv(tokenizer) -> list[dict]:
    if not CSV_IN.exists():
        print(f"  [skip] {CSV_IN} missing")
        return []
    print(f"\n=== CSV: {CSV_IN.name} ===")
    df = pd.read_csv(CSV_IN, low_memory=False)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df[df["url"].astype(str).str.match(r"^https://", na=False)].reset_index(drop=True)
    print(f"  Total cleaned rows: {len(df)}; sampling {min(CSV_SAMPLE, len(df))}")
    df = df.head(CSV_SAMPLE)
    out = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="CSV"):
        title = row.get("title", "") if isinstance(row.get("title", ""), str) else ""
        text = row.get("text", "") if isinstance(row.get("text", ""), str) else ""
        full = f"{title}\n\n{text}"
        chars, words, tokens = measure(full, tokenizer)
        out.append({
            "source": "csv",
            "article_id": str(row.get("article_id", "")),
            "chars": chars,
            "words": words,
            "tokens": tokens,
        })
    return out


def collect_zips(tokenizer) -> list[dict]:
    if not ZIP_DIR.exists():
        print(f"  [skip] {ZIP_DIR} missing")
        return []
    print(f"\n=== ZIPs in {ZIP_DIR} ===")
    zip_files = sorted(ZIP_DIR.glob("*.zip"))
    print(f"  {len(zip_files)} zips, sampling up to {ZIP_SAMPLE_PER_FILE} English articles per zip")
    out = []
    for zp in tqdm(zip_files, desc="ZIPs"):
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                names = [n for n in zf.namelist() if n.endswith(".json")]
                count = 0
                for n in names:
                    if count >= ZIP_SAMPLE_PER_FILE:
                        break
                    try:
                        with zf.open(n) as fh:
                            payload = json.loads(fh.read().decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    if payload.get("language") != "english":
                        continue
                    title = payload.get("title") or ""
                    text = payload.get("text") or ""
                    full = f"{title}\n\n{text}"
                    chars, words, tokens = measure(full, tokenizer)
                    out.append({
                        "source": "zip",
                        "article_id": str(payload.get("uuid") or ""),
                        "chars": chars,
                        "words": words,
                        "tokens": tokens,
                    })
                    count += 1
        except zipfile.BadZipFile:
            print(f"  [warn] {zp.name} corrupted, skipping")
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, g in df.groupby("source"):
        for col in ("chars", "words", "tokens"):
            rows.append({
                "source": source,
                "metric": col,
                "n": len(g),
                "mean": round(g[col].mean(), 1),
                "median": int(g[col].median()),
                "p90": int(g[col].quantile(0.90)),
                "p95": int(g[col].quantile(0.95)),
                "p99": int(g[col].quantile(0.99)),
                "max": int(g[col].max()),
            })
    return pd.DataFrame(rows)


def main():
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    records = []
    records.extend(collect_db(tokenizer))
    records.extend(collect_csv(tokenizer))
    records.extend(collect_zips(tokenizer))

    if not records:
        print("No records collected. Exiting.")
        return

    df = pd.DataFrame(records)
    df.to_csv(EDA_OUT, index=False)
    print(f"\nPer-article EDA written: {EDA_OUT}  ({len(df)} rows)")

    summary = summarize(df)
    summary.to_csv(SUMMARY_OUT, index=False)
    print(f"Summary written:         {SUMMARY_OUT}\n")
    print(summary.to_string(index=False))

    # Quick % over 4096 / 8192 token thresholds — directly answers num_ctx question
    print("\n=== num_ctx coverage ===")
    for src, g in df.groupby("source"):
        n = len(g)
        for cap in (2048, 4096, 8192):
            pct = (g["tokens"] <= cap).mean() * 100
            print(f"  {src:>4}  ≤{cap:>5} tokens: {pct:5.1f}%  ({int(pct/100*n)}/{n})")


if __name__ == "__main__":
    main()
