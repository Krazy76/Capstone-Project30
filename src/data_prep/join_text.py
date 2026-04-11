"""Rehydrate the article body onto silver_dataset_master.csv.

The master CSV only carries labels + metadata; bodies live in three
source stores (SQLite DB, raw CSV, 77 Webhose zips). This joins text
back in by article_id so downstream code never touches the sources.
Writes data/silver_dataset_master_with_text.csv.
"""

from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BASE = Path(r"E:\Coding\Python\Capstone-Project30\data")
MASTER_IN = BASE / "silver_dataset_master.csv"
MASTER_OUT = BASE / "silver_dataset_master_with_text.csv"
DB_PATH = BASE / "database 1.db"
CSV_IN = BASE / "silver_dataset_final(in).csv"
ZIP_DIR = BASE / "Datasets"


def load_db_texts(needed: set[str]) -> dict[str, str]:
    """Look up body text from SQLite for article_ids of form 'db_<row_id>'."""
    if not DB_PATH.exists():
        return {}
    wanted_raw_ids = {aid[3:] for aid in needed if aid.startswith("db_")}
    if not wanted_raw_ids:
        return {}
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = conn.execute("SELECT id, text FROM articles").fetchall()
    conn.close()
    out = {f"db_{rid}": (text or "") for rid, text in rows if f"db_{rid}" in needed}
    print(f"  DB matched: {len(out)}")
    return out


def load_csv_texts(needed: set[str]) -> dict[str, str]:
    """Look up body text from silver_dataset_final(in).csv by article_id."""
    if not CSV_IN.exists():
        return {}
    df = pd.read_csv(CSV_IN, low_memory=False)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if "article_id" not in df.columns or "text" not in df.columns:
        return {}
    df["article_id"] = df["article_id"].astype(str)
    df = df[df["article_id"].isin(needed)]
    out = dict(zip(df["article_id"], df["text"].fillna("").astype(str)))
    print(f"  CSV matched: {len(out)}")
    return out


def load_zip_texts(needed: set[str]) -> dict[str, str]:
    """Scan 77 Webhose zips for article_ids (uuids), stopping once all found."""
    if not ZIP_DIR.exists():
        return {}
    zip_files = sorted(ZIP_DIR.glob("*.zip"))
    out: dict[str, str] = {}
    remaining = set(needed)
    for zp in tqdm(zip_files, desc="ZIPs"):
        if not remaining:
            break
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                for n in zf.namelist():
                    if not n.endswith(".json"):
                        continue
                    try:
                        with zf.open(n) as fh:
                            payload = json.loads(fh.read().decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    uid = str(payload.get("uuid") or "")
                    if uid and uid in remaining:
                        out[uid] = payload.get("text") or ""
                        remaining.discard(uid)
                        if not remaining:
                            break
        except zipfile.BadZipFile:
            print(f"  [warn] {zp.name} corrupted")
    print(f"  ZIPs matched: {len(out)}")
    return out


def main():
    print(f"Reading {MASTER_IN}")
    master = pd.read_csv(MASTER_IN, low_memory=False)
    master["article_id"] = master["article_id"].astype(str)
    print(f"  rows: {len(master)}")

    by_source = {str(src): set(g["article_id"].tolist())
                 for src, g in master.groupby("source")}
    for k, v in by_source.items():
        print(f"  source={k}: {len(v)} ids")

    text_map: dict[str, str] = {}
    if "db" in by_source:
        text_map.update(load_db_texts(by_source["db"]))
    if "csv" in by_source:
        text_map.update(load_csv_texts(by_source["csv"]))
    if "zip" in by_source:
        text_map.update(load_zip_texts(by_source["zip"]))

    master["text"] = master["article_id"].map(text_map).fillna("")
    missing = int((master["text"].str.len() == 0).sum())
    print(f"\nTotal rows: {len(master)}   rows with empty body: {missing}")
    master.to_csv(MASTER_OUT, index=False)
    print(f"Wrote: {MASTER_OUT}")


if __name__ == "__main__":
    main()
