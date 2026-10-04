"""
Evaluates a trained model against a held-out labeled CSV and, if the CSV
also has a 'category' column (e.g. the FlakyCat dataset), checks whether
the model's SHAP-based explanations tend to point at the SAME cause
category as the test's documented, real root cause.

Usage:
    python -m src.evaluate --model models/flaky_rf.joblib --data data/flakycat_tests.csv
"""

import argparse
from collections import Counter

import joblib
import pandas as pd
from sklearn.metrics import classification_report

from src.data_loader import load_labeled_csv
from src.explain import explain_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--code-col", default="test_code")
    parser.add_argument("--category-col", default="category")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    clf = bundle["model"]

    df = pd.read_csv(args.data)
    has_categories = args.category_col in df.columns

    match_count = 0
    total_checked = 0

    for _, row in df.iterrows():
        code = str(row[args.code_col])
        result = explain_prediction(code, clf, bundle["feature_columns"], bundle["n_handcrafted_features"])
        print("-" * 60)
        print(f"Prediction: {result['label']} (confidence {result['confidence']:.2f})")
        print("Top reasons:")
        for reason in result["top_reasons"]:
            print(f"  - {reason}")

        if has_categories:
            true_category = str(row[args.category_col])
            predicted_categories = {r["category"] for r in result["top_reasons_structured"]}
            total_checked += 1
            if true_category in predicted_categories:
                match_count += 1
            print(f"  (documented real cause: {true_category})")

    if has_categories and total_checked > 0:
        print("\n=== Explanation alignment check ===")
        print(
            f"{match_count}/{total_checked} "
            f"({100 * match_count / total_checked:.1f}%) of explanations named "
            "the same category as the documented real cause."
        )


if __name__ == "__main__":
    main()
