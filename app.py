"""
Streamlit front end for the flaky-test predictor.

Run with:
    streamlit run app.py

This gives you a browser-based demo: paste or upload a test's source code,
click Predict, and see the flaky/not-flaky call plus the plain-English
SHAP-based explanation -- instead of reading terminal output.
"""

import json
import os

import joblib
import streamlit as st

from src.explain import explain_prediction
from src.handcrafted_features import FEATURE_SPECS

MODEL_PATH = "models/flaky_rf.joblib"
METRICS_PATH = "models/flaky_rf_metrics.json"

st.set_page_config(page_title="Flaky Test Predictor", page_icon="🧪", layout="wide")


@st.cache_resource
def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


st.title("🧪 LLM-Based Flaky Test Prediction & Explainable Risk Analysis")
st.caption("Predicts whether a CI/CD test is likely flaky, and explains why in plain English.")

tab_demo, tab_overview, tab_performance = st.tabs(["Try It", "Project Overview", "Model Performance"])

# --------------------------------------------------------------------------
# TAB 1: Live demo
# --------------------------------------------------------------------------
with tab_demo:
    bundle = load_model_bundle()

    if bundle is None:
        st.warning(
            "No trained model found yet. Train one first from a terminal:\n\n"
            "`python -m src.train_model --data data/your_data.csv --output models/flaky_rf.joblib`\n\n"
            "Once that finishes, refresh this page."
        )
    else:
        st.write("Paste a test's source code below, or upload a file, then click **Predict**.")

        col_input, col_result = st.columns([1, 1])

        with col_input:
            uploaded = st.file_uploader("Upload a test file (.java / .py / .txt)", type=["java", "py", "txt"])
            default_code = (
                "@Test\n"
                "public void testFetchData() throws Exception {\n"
                "    Thread.sleep(1000);\n"
                "    HttpClient client = new HttpClient();\n"
                "    String result = client.get(\"http://example.com/api\");\n"
                "    assertNotNull(result);\n"
                "}"
            )
            if uploaded is not None:
                code_text = uploaded.read().decode("utf-8", errors="ignore")
            else:
                code_text = default_code

            code = st.text_area("Test source code", value=code_text, height=300)
            predict_clicked = st.button("Predict", type="primary")

        with col_result:
            if predict_clicked and code.strip():
                with st.spinner("Running CodeBERT + classifier + SHAP..."):
                    result = explain_prediction(
                        code, bundle["model"], bundle["feature_columns"], bundle["n_handcrafted_features"]
                    )

                label = result["label"]
                confidence = result["confidence"]

                if label == "flaky":
                    st.error(f"### Prediction: FLAKY")
                else:
                    st.success(f"### Prediction: NOT FLAKY")

                st.metric("Confidence", f"{confidence:.1%}")
                st.progress(confidence)

                st.markdown("#### Why the model thinks this")
                for reason in result["top_reasons"]:
                    st.markdown(f"- {reason}")

                st.caption(
                    f"The model's deeper code understanding (CodeBERT embedding) also contributed, "
                    f"with a combined signal strength of {result['embedding_contribution']:.3f}."
                )

                if result["top_reasons_structured"]:
                    st.markdown("#### Contributing patterns (SHAP values)")
                    chart_data = {
                        r["feature"]: r["shap_value"] for r in result["top_reasons_structured"]
                    }
                    st.bar_chart(chart_data)
            else:
                st.info("Paste or upload code on the left, then click Predict.")

# --------------------------------------------------------------------------
# TAB 2: Project overview (for showing your guide/panel the full picture)
# --------------------------------------------------------------------------
with tab_overview:
    st.markdown(
        """
### The problem
Flaky tests pass or fail nondeterministically with no real bug behind them.
Once developers see enough of these false alarms, they stop trusting CI
failures altogether -- which is how real bugs slip through disguised as
"probably just flaky."

### The approach
1. **Handcrafted features** -- 18 keyword/pattern detectors tied to known
   flaky-test root-cause categories (concurrency, timing, randomness,
   network calls, order dependency, platform dependency, unordered
   collections).
2. **CodeBERT embedding** -- a pretrained code-understanding model
   (`microsoft/codebert-base`) turns the test's source into a 768-number
   representation capturing deeper code patterns.
3. **RandomForestClassifier** -- trained on the combined feature set to
   predict flaky vs. not_flaky.
4. **SHAP explainability layer** -- explains each prediction by surfacing
   which handcrafted patterns (and their known category) drove the result.

### The research gap this fills
FlakyQ (Rahman & Misailovic, ICST 2024) -- the closest prior work --
predicts flakiness with a quantized CodeBERT model but explicitly lists
**explainability** as future work it did not attempt. This project adds
exactly that layer, and evaluates whether its explanations line up with
the real, documented root causes in the FlakyCat dataset.

### Tech stack
- Python, `transformers` + `microsoft/codebert-base` (free, open-weight)
- `scikit-learn` (RandomForest)
- `shap` (explainability)
- `streamlit` (this front end)
- All free and open-source -- no paid APIs.
        """
    )

# --------------------------------------------------------------------------
# TAB 3: Model performance (real numbers once you've trained on real data)
# --------------------------------------------------------------------------
with tab_performance:
    metrics = load_metrics()
    if metrics is None:
        st.info(
            "No metrics yet. They're generated automatically the first time you run "
            "`python -m src.train_model ...` -- train a model, then refresh this page."
        )
    else:
        st.metric("F1 score", f"{metrics['f1']:.3f}")
        st.caption("FlakyQ's published benchmark for comparison: 0.933 - 0.944")
        st.markdown("#### Full classification report")
        st.code(metrics["report"])

    st.markdown("#### The 18 interpretable features the explanation layer draws on")
    import pandas as pd

    feature_table = pd.DataFrame(
        [{"Feature": spec.name, "Category": spec.category} for spec in FEATURE_SPECS]
    )
    st.dataframe(feature_table, use_container_width=True, hide_index=True)