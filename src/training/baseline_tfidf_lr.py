from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_list_cell(cell) -> list:
    """Parse list cells safely."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    if isinstance(cell, list):
        return cell

    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return []

    try:
        v = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        try:
            v = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []

    return v if isinstance(v, list) else []


def is_financial_true(x) -> bool:
    """Match the repo's financial-row filtering logic."""
    if isinstance(x, bool):
        return x
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return False
    return str(x).strip().lower() in {"true", "1", "yes"}


def clean_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.replace("â\x80\x99", "'")
    text = text.replace("\u2019", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_baseline_dataframe(csv_path: Path) -> pd.DataFrame:
    """
    Load the repo's master-with-text CSV and apply the same baseline filtering idea:
    - keep only is_financial == True
    - parse macro / industry / entity into lists
    - drop rows where all three are empty
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # Keep only financial rows
    df = df[df["is_financial"].apply(is_financial_true)].copy()

    # Parse label columns
    df["macro_list"] = df["macro"].apply(parse_list_cell)
    df["industry_list"] = df["industry"].apply(parse_list_cell)
    df["entity_list"] = df["entity"].apply(parse_list_cell)

    # Drop rows where all three label groups are empty
    all_empty = (
        df["macro_list"].map(len).eq(0)
        & df["industry_list"].map(len).eq(0)
        & df["entity_list"].map(len).eq(0)
    )
    df = df[~all_empty].copy()

    # Build text field using: title + text
    if "title" not in df.columns:
        df["title"] = ""
    if "text" not in df.columns:
        df["text"] = ""

    df["model_text"] = (
        df["title"].fillna("").map(clean_text)
        + " [SEP] "
        + df["text"].fillna("").map(clean_text)
    ).str.strip()

    # Drop very short rows
    df = df[df["model_text"].str.len() > 50].copy()

    # Build top-level binary targets
    df["macro_binary"] = df["macro_list"].apply(lambda x: int(len(x) > 0))
    df["industry_binary"] = df["industry_list"].apply(lambda x: int(len(x) > 0))
    df["entity_binary"] = df["entity_list"].apply(lambda x: int(len(x) > 0))

    return df.reset_index(drop=True)


def main():
    root = project_root()
    csv_path = root / "data" / "silver_dataset_master_with_text.csv"
    output_dir = root / "reports" / "baseline_tfidf_lr"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. Make sure silver_dataset_master_with_text.csv exists first."
        )

    df = load_baseline_dataframe(csv_path)

    X = df["model_text"]
    y = df[["macro_binary", "industry_binary", "entity_binary"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                    )
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    report_lines = []
    report_lines.append("=== Project 30 baseline: TF-IDF + Logistic Regression ===\n")
    report_lines.append(f"Input file: {csv_path}\n")
    report_lines.append(f"Rows used after filtering: {len(df)}\n\n")

    report_lines.append("Label distribution (full filtered dataset):\n")
    report_lines.append(str(y.sum()) + "\n\n")

    label_names = ["macro_binary", "industry_binary", "entity_binary"]
    pretty_names = ["MACRO", "INDUSTRY", "ENTITY"]

    for i, pretty in enumerate(pretty_names):
        report_lines.append(f"--- {pretty} ---\n")
        report_lines.append(
            classification_report(
                y_test.iloc[:, i],
                y_pred[:, i],
                digits=4,
                zero_division=0,
            )
        )
        report_lines.append("\n")

    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    report_lines.append("Overall multilabel F1 scores:\n")
    report_lines.append(f"micro_f1={micro_f1:.4f}\n")
    report_lines.append(f"macro_f1={macro_f1:.4f}\n")
    report_lines.append(f"weighted_f1={weighted_f1:.4f}\n")

    report_path = output_dir / "baseline_report.txt"
    report_path.write_text("".join(report_lines), encoding="utf-8")

    sample_df = pd.DataFrame(
        {
            "text": X_test.reset_index(drop=True),
            "true_macro": y_test["macro_binary"].reset_index(drop=True),
            "true_industry": y_test["industry_binary"].reset_index(drop=True),
            "true_entity": y_test["entity_binary"].reset_index(drop=True),
            "pred_macro": y_pred[:, 0],
            "pred_industry": y_pred[:, 1],
            "pred_entity": y_pred[:, 2],
        }
    ).head(50)

    sample_path = output_dir / "baseline_predictions_sample.csv"
    sample_df.to_csv(sample_path, index=False)

    print(f"Saved report to: {report_path}")
    print(f"Saved sample predictions to: {sample_path}")
    print(f"Rows used: {len(df)}")
    print(f"micro_f1={micro_f1:.4f}")
    print(f"macro_f1={macro_f1:.4f}")
    print(f"weighted_f1={weighted_f1:.4f}")


if __name__ == "__main__":
    main()