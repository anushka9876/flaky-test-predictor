"""
The explainability layer — this is the project's actual research contribution.

FlakyQ (the closest prior paper) explicitly lists explainability as future
work it did NOT do. This file is what fills that gap: it takes a trained
RandomForest's prediction and uses SHAP (SHapley Additive exPlanations) to
say WHY the model thinks a test is flaky, in terms a developer can act on.

Design choice worth stating plainly in your paper: CodeBERT embedding
dimensions are not individually interpretable (dimension 482 being "high"
means nothing to a human), so we only surface the HANDCRAFTED, named
features (see handcrafted_features.py) as human-readable reasons, and we
summarize the embedding's total contribution as a single supporting signal
rather than pretending every number is meaningful. This is an honest,
defensible way to combine "accurate but opaque" (the embedding) with
"transparent but narrower" (the handcrafted features).
"""

import numpy as np
import shap

from src.handcrafted_features import extract_handcrafted_features, feature_names, category_for_feature
from src.feature_extraction import get_codebert_embedding

HANDCRAFTED_NAMES = feature_names()


def _featurize_single(code: str):
    handcrafted = extract_handcrafted_features(code)
    handcrafted_vec = np.array([handcrafted[name] for name in HANDCRAFTED_NAMES])
    embedding_vec = get_codebert_embedding(code)
    x = np.concatenate([handcrafted_vec, embedding_vec]).reshape(1, -1)
    return x, handcrafted


def explain_prediction(code: str, clf, feature_columns, n_handcrafted_features: int, top_k: int = 3):
    """
    Returns a dict:
        {
          "label": "flaky" | "not_flaky",
          "confidence": float,
          "top_reasons": [human-readable strings],
          "top_reasons_structured": [{"feature": ..., "category": ..., "shap_value": ...}],
          "embedding_contribution": float,  # combined |SHAP| of all embedding dims
        }
    """
    x, handcrafted = _featurize_single(code)

    proba = clf.predict_proba(x)[0]
    pred_class = int(np.argmax(proba))
    label = "flaky" if pred_class == 1 else "not_flaky"
    confidence = float(proba[pred_class])

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(x)

    # Handle both possible SHAP return shapes across versions:
    # a list of per-class arrays, or a single array for the positive class.
    if isinstance(shap_values, list):
        class_shap = shap_values[pred_class][0]
    else:
        class_shap = shap_values[0]
        if class_shap.ndim > 1:
            class_shap = class_shap[:, pred_class]

    handcrafted_shap = class_shap[:n_handcrafted_features]
    embedding_shap = class_shap[n_handcrafted_features:]

    # Only rank handcrafted features that are actually PRESENT in this test
    # (a feature with value 0 contributing little tells you nothing useful).
    present_features = [
        (name, handcrafted_shap[i])
        for i, name in enumerate(HANDCRAFTED_NAMES)
        if handcrafted[name] == 1
    ]
    present_features.sort(key=lambda pair: abs(pair[1]), reverse=True)
    top_present = present_features[:top_k]

    top_reasons = []
    top_reasons_structured = []
    for name, shap_val in top_present:
        category = category_for_feature(name)
        direction = "increased" if shap_val > 0 else "decreased"
        top_reasons.append(
            f"Pattern '{name}' (category: {category}) was detected in the test code and {direction} the flakiness score."
        )
        top_reasons_structured.append({"feature": name, "category": category, "shap_value": float(shap_val)})

    embedding_contribution = float(np.sum(np.abs(embedding_shap)))
    if not top_reasons:
        top_reasons.append(
            "No known keyword-based risk pattern was detected; the prediction is driven mainly by "
            "the model's deeper understanding of the code (not individually explainable)."
        )

    return {
        "label": label,
        "confidence": confidence,
        "top_reasons": top_reasons,
        "top_reasons_structured": top_reasons_structured,
        "embedding_contribution": embedding_contribution,
    }


if __name__ == "__main__":
    import sys
    import joblib

    if len(sys.argv) < 2:
        print("Usage: python -m src.explain models/flaky_rf.joblib")
        sys.exit(1)

    bundle = joblib.load(sys.argv[1])
    sample_code = """
    @Test
    public void testFetchData() throws Exception {
        Thread.sleep(1000);
        HttpClient client = new HttpClient();
        String result = client.get("http://example.com/api");
        assertNotNull(result);
    }
    """
    result = explain_prediction(
        sample_code, bundle["model"], bundle["feature_columns"], bundle["n_handcrafted_features"]
    )
    print(result)
