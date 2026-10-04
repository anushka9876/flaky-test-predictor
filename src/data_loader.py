"""
Loads a labeled CSV of test code and builds the combined feature matrix
(handcrafted features + CodeBERT embedding) used by every other script.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.handcrafted_features import extract_handcrafted_features, feature_names
from src.feature_extraction import get_codebert_embedding

HANDCRAFTED_NAMES = feature_names()
EMBEDDING_DIM = 768


def build_feature_matrix(df: pd.DataFrame, code_col: str = "test_code"):
    """
    df must have a column with raw test source code (default: 'test_code').
    Returns:
        X: numpy array of shape (n_samples, n_handcrafted + 768)
        feature_columns: list of column names, in the same order as X's columns
                          (the first n_handcrafted are the interpretable ones;
                          the rest are "embedding_0".."embedding_767")
    """
    handcrafted_rows = []
    embedding_rows = []

    for code in tqdm(df[code_col].astype(str).tolist(), desc="Extracting features"):
        handcrafted_rows.append(extract_handcrafted_features(code))
        embedding_rows.append(get_codebert_embedding(code))

    handcrafted_df = pd.DataFrame(handcrafted_rows)[HANDCRAFTED_NAMES]
    embedding_arr = np.vstack(embedding_rows)
    embedding_cols = [f"embedding_{i}" for i in range(embedding_arr.shape[1])]

    X = np.hstack([handcrafted_df.values, embedding_arr])
    feature_columns = HANDCRAFTED_NAMES + embedding_cols
    return X, feature_columns


def load_labeled_csv(path: str, code_col: str = "test_code", label_col: str = "label"):
    """
    Loads a CSV with at least [code_col, label_col] and returns
    (X, y, feature_columns). Labels are mapped to 1 ("flaky") / 0 (anything else).
    """
    df = pd.read_csv(path)
    X, feature_columns = build_feature_matrix(df, code_col=code_col)
    y = (df[label_col].astype(str).str.lower() == "flaky").astype(int).values
    return X, y, feature_columns


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.data_loader path/to/idoft_tests.csv")
        sys.exit(1)
    X, y, cols = load_labeled_csv(sys.argv[1])
    print("X shape:", X.shape)
    print("y distribution:", np.bincount(y))
