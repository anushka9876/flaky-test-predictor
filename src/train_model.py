"""
Trains the flaky-test classifier.

Recipe follows FlakyQ (Rahman & Misailovic, ICST 2024): extract code
features (there via a quantized fine-tuned CodeBERT; here via pretrained
CodeBERT embeddings + handcrafted features), then train a traditional
classifier (Random Forest) on top. FlakyQ reports F1 scores of 93.3-94.4
using this general approach — treat that as your benchmark to match.

Usage:
    python -m src.train_model --data data/idoft_tests.csv --output models/flaky_rf.joblib
"""

import argparse
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from src.data_loader import load_labeled_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to labeled CSV (test_code,label)")
    parser.add_argument("--output", default="models/flaky_rf.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading and featurizing {args.data} ...")
    X, y, feature_columns = load_labeled_csv(args.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    print(f"Training RandomForestClassifier (n_estimators={args.n_estimators}) ...")
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        class_weight="balanced",  # datasets like IDoFT skew heavily toward flaky examples
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["not_flaky", "flaky"])
    f1 = f1_score(y_test, y_pred)

    print("\n=== Held-out test set results ===")
    print(report)
    print(f"F1 score: {f1:.4f}  (FlakyQ's reported benchmark: 0.933-0.944)")

    joblib.dump(
        {
            "model": clf,
            "feature_columns": feature_columns,
            "n_handcrafted_features": 18,  # keep in sync with handcrafted_features.py
        },
        args.output,
    )
    print(f"\nSaved trained model to {args.output}")

    metrics_path = args.output.replace(".joblib", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"f1": f1, "report": report}, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
