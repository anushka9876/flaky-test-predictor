"""
Command-line tool: point it at a single test file and get a flakiness
prediction plus a plain-English explanation.

Usage:
    python -m src.predict --model models/flaky_rf.joblib --file path/to/SomeTest.java
"""

import argparse

import joblib

from src.explain import explain_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--file", required=True, help="Path to a test source file")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()

    result = explain_prediction(
        code, bundle["model"], bundle["feature_columns"], bundle["n_handcrafted_features"]
    )

    print(f"\nFile: {args.file}")
    print(f"Prediction: {result['label'].upper()}  (confidence: {result['confidence']:.1%})")
    print("\nWhy:")
    for reason in result["top_reasons"]:
        print(f"  - {reason}")
    print(
        f"\n(The model's deeper code understanding also contributed, "
        f"with a combined signal strength of {result['embedding_contribution']:.3f}.)"
    )


if __name__ == "__main__":
    main()
