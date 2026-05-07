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
    Load the master CSV using the SAME filter the transformer pipeline uses
    (load_and_filter_master with default label_version='v2'). Otherwise the
    baseline's row count diverges from the transformer's, and the locked
    splits file points to wrong rows.
    """
    import sys
    sys.path.insert(0, str(project_root()))
    from src.data_prep.filters import load_and_filter_master

    # Same filter as transformer: is_financial=True AND not all-empty AND
    # label_version=='v2'. Returns df with macro_list/industry_list/entity_list
    # already parsed and reset_index applied.
    df = load_and_filter_master(csv_path).copy()

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


def _build_per_class_targets(df: pd.DataFrame):
    """Per-class multi-label targets matching the transformer pipeline.

    Returns:
        macro_y: (n, 8) one-hot for FRED-MD macro labels
        industry_y: (n, 11) one-hot for GICS industry labels
        macro_labels, industry_labels: column orderings
    """
    import sys, numpy as np
    sys.path.insert(0, str(project_root()))
    from src.data_prep.filters import MACRO_LABELS, INDUSTRY_LABELS

    n = len(df)
    macro_y = np.zeros((n, len(MACRO_LABELS)), dtype=int)
    industry_y = np.zeros((n, len(INDUSTRY_LABELS)), dtype=int)
    macro_to_idx = {lbl: i for i, lbl in enumerate(MACRO_LABELS)}
    industry_to_idx = {lbl: i for i, lbl in enumerate(INDUSTRY_LABELS)}

    for i, (mlist, ilist) in enumerate(zip(df["macro_list"], df["industry_list"])):
        for m in mlist:
            if m in macro_to_idx:
                macro_y[i, macro_to_idx[m]] = 1
        for ind in ilist:
            if ind in industry_to_idx:
                industry_y[i, industry_to_idx[ind]] = 1
    return macro_y, industry_y, MACRO_LABELS, INDUSTRY_LABELS


def main():
    import argparse
    root = project_root()
    p = argparse.ArgumentParser()
    p.add_argument("--master-csv", default=str(root / "data" / "silver_dataset_master_with_text.csv"))
    p.add_argument("--out-name", default="baseline_tfidf_lr",
                   help="Subdir name under reports/ for outputs")
    p.add_argument("--use-splits", action="store_true", default=True,
                   help="Use the locked splits from data/splits/ to match transformer pipeline")
    p.add_argument("--no-splits", dest="use_splits", action="store_false")
    args = p.parse_args()

    csv_path = Path(args.master_csv)
    output_dir = root / "reports" / args.out_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}.")

    df = load_baseline_dataframe(csv_path)

    X = df["model_text"]
    y = df[["macro_binary", "industry_binary", "entity_binary"]]

    # ---- Split strategy ----
    if args.use_splits:
        # Use the same train/val/test indices as the transformer pipeline so
        # numbers compare apples-to-apples in section 4.4.3.
        import sys, json
        sys.path.insert(0, str(root))
        from src.training.splits import load_or_create_splits
        splits = load_or_create_splits(n_rows=len(df))
        train_idx = splits["train"]
        test_idx = splits["test"]    # Use TEST split (matches transformer eval)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        print(f"[baseline] using locked splits: train={len(train_idx)} test={len(test_idx)}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
        )
        train_idx = X_train.index.tolist()
        test_idx = X_test.index.tolist()

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

    report_lines.append("Overall binary multilabel F1 scores:\n")
    report_lines.append(f"micro_f1={micro_f1:.4f}\n")
    report_lines.append(f"macro_f1={macro_f1:.4f}\n")
    report_lines.append(f"weighted_f1={weighted_f1:.4f}\n\n")

    # -----------------------------------------------------------------------
    # Per-class multi-label baseline (matches transformer head structure).
    # 8 macro classes + 11 industry classes -> directly comparable F1 numbers
    # for section 4.4.3 Model Comparison.
    # -----------------------------------------------------------------------
    print("\n=== Per-class multi-label baseline (matches transformer heads) ===")
    macro_y, industry_y, macro_labels, industry_labels = _build_per_class_targets(df)
    macro_y_train, macro_y_test = macro_y[train_idx], macro_y[test_idx]
    industry_y_train, industry_y_test = industry_y[train_idx], industry_y[test_idx]

    perclass_metrics = {}

    for head_name, y_tr, y_te, label_names in [
        ("macro", macro_y_train, macro_y_test, macro_labels),
        ("industry", industry_y_train, industry_y_test, industry_labels),
    ]:
        clf_perclass = Pipeline(
            [
                ("tfidf", TfidfVectorizer(
                    max_features=20000, ngram_range=(1, 2),
                    min_df=2, max_df=0.95, sublinear_tf=True,
                )),
                ("clf", OneVsRestClassifier(
                    LogisticRegression(max_iter=2000, class_weight="balanced"),
                )),
            ]
        )
        clf_perclass.fit(X_train, y_tr)
        y_pred_pc = clf_perclass.predict(X_test)

        micro = f1_score(y_te, y_pred_pc, average="micro", zero_division=0)
        macro_avg = f1_score(y_te, y_pred_pc, average="macro", zero_division=0)
        weighted = f1_score(y_te, y_pred_pc, average="weighted", zero_division=0)
        per_cls = f1_score(y_te, y_pred_pc, average=None, zero_division=0)

        perclass_metrics[head_name] = {
            "micro_f1": float(micro),
            "macro_f1": float(macro_avg),
            "weighted_f1": float(weighted),
            "per_class_f1": {lbl: float(f) for lbl, f in zip(label_names, per_cls)},
        }

        report_lines.append(f"\n=== {head_name.upper()} per-class F1 (test set) ===\n")
        report_lines.append(f"micro_f1={micro:.4f}  macro_f1={macro_avg:.4f}  weighted_f1={weighted:.4f}\n")
        for lbl, f in zip(label_names, per_cls):
            report_lines.append(f"  {lbl:<40} {f:.4f}\n")
        print(f"  {head_name}: micro_f1={micro:.4f}  macro_f1={macro_avg:.4f}")

    # JSON dump for the report table.
    metrics_path = output_dir / "baseline_metrics.json"
    metrics_path.write_text(
        json.dumps({
            "binary": {"micro_f1": micro_f1, "macro_f1": macro_f1, "weighted_f1": weighted_f1},
            "per_class": perclass_metrics,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "csv_path": str(csv_path),
        }, indent=2),
        encoding="utf-8",
    )

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