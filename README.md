# LLM-Based Flaky Test Prediction and Explainable Risk Analysis for CI/CD Pipelines

A starter implementation: predicts whether a test is likely to be "flaky"
(passes/fails nondeterministically with no real bug behind it) and explains
*why*, using SHAP to tie the prediction back to known, human-readable
root-cause patterns (concurrency, timing, randomness, network calls, order
dependency, etc.).

## Why this project (the research gap)

FlakyQ (Rahman & Misailovic, ICST 2024) — the closest prior work — predicts
flakiness using a quantized CodeBERT model plus a traditional classifier,
reaching ~93-94% F1, but explicitly lists **explainability** as future work
it did not attempt. This project fills that gap: a prediction pipeline in
the same spirit, with a SHAP-based explanation layer on top that maps back
to the root-cause categories identified in the flaky-test survey literature
(Parry et al.; Fatima & Nagappan).

## Architecture

```
  test source code (.java / .py file)
            |
            v
  +-----------------------+        +--------------------------+
  | handcrafted_features  |        |   feature_extraction     |
  | (keyword/regex rules, |        |   (CodeBERT embedding,   |
  |  18 interpretable     |        |    768 numbers, NOT      |
  |  0/1 features, each   |        |    individually          |
  |  tagged to a known    |        |    interpretable)        |
  |  flaky-cause category)|        |                          |
  +-----------------------+        +--------------------------+
            \\                              /
             \\                            /
              v                          v
          combined feature vector (18 + 768 = 786 numbers)
                            |
                            v
                RandomForestClassifier
                  (flaky / not_flaky)
                            |
                            v
                    SHAP TreeExplainer
                            |
                            v
        "flaky, 87% confidence -- because:
         pattern 'uses_sleep' (Timing) and
         'uses_network' (Network/External I-O)
         were detected in this test"
```

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Language | Python 3.10+ | Standard for ML/data work; all libraries below are Python-first |
| Code understanding | `transformers` + `microsoft/codebert-base` | Pretrained model that understands source code, used to generate the embedding half of the feature vector |
| Deep learning runtime | `torch` | Required to run CodeBERT |
| Classifier | `scikit-learn` (RandomForestClassifier) | Simple, fast, and pairs natively with SHAP's TreeExplainer |
| Explainability | `shap` | Computes per-prediction feature attributions |
| Data handling | `pandas`, `numpy` | Loading CSVs, building feature matrices |
| Model persistence | `joblib` | Saving/loading the trained classifier |
| Version control | Git + GitHub | Source control and submission |
| (Optional, stretch goal) | Flask or FastAPI | Wrap `predict.py` in a small web API / GitHub Action so it can comment on real pull requests |

You do not need a GPU for this starter pipeline — CodeBERT in "feature
extraction" mode (no fine-tuning) runs acceptably on CPU for a project-sized
dataset (hundreds to a few thousand tests). It will just be slower per test
(a second or two) than with a GPU.

## Setup

```bash
# 1. Clone your repo (see GitHub section below) and cd into it
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data

See `data/README.md` for exact instructions and links. In short:
download/derive `data/idoft_tests.csv` (flaky vs. not_flaky labels) from
https://github.com/TestingResearchIllinois/idoft, and
`data/flakycat_tests.csv` (root-cause category labels) from
https://github.com/Amal-AK/FLAKYCAT.

**If you want to sanity-check the pipeline before the real data is ready**,
create a tiny CSV by hand with 10-20 rows (a few obviously flaky-looking
tests using `Thread.sleep`/`Random`, a few plain ones) — this lets you run
every command below end-to-end today, then swap in the real dataset later.

## Run it

```bash
# 1. Train the model (first run downloads CodeBERT automatically, ~500MB)
python -m src.train_model --data data/idoft_tests.csv --output models/flaky_rf.joblib

# 2. Evaluate against FlakyCat and check explanation alignment with real
#    documented root causes
python -m src.evaluate --model models/flaky_rf.joblib --data data/flakycat_tests.csv

# 3. Predict + explain for a single test file
python -m src.predict --model models/flaky_rf.joblib --file path/to/SomeTest.java
```

## Project layout

```
flaky-test-predictor/
  README.md
  requirements.txt
  data/
    README.md          <- dataset download instructions
  models/               <- trained models land here (gitignored)
  src/
    handcrafted_features.py   <- the 18 interpretable, named features
    feature_extraction.py     <- CodeBERT embedding
    data_loader.py             <- combines both into one feature matrix
    train_model.py             <- trains + saves the RandomForest
    explain.py                 <- SHAP explanation layer (your contribution)
    evaluate.py                <- metrics + explanation-alignment check
    predict.py                 <- CLI for a single file
```

## Suggested semester timeline

- **Weeks 1-2**: Get both datasets downloaded and converted to the expected CSV format. This is the least glamorous step and also the one most worth doing early.
- **Weeks 3-4**: Get `train_model.py` running end to end on a small subset; confirm accuracy is in a reasonable range before scaling up.
- **Weeks 5-6**: Scale to the full dataset; tune `n_estimators` and compare against the FlakyQ benchmark (93.3-94.4% F1).
- **Weeks 7-9**: Build out `explain.py`'s evaluation against FlakyCat's documented categories — this is your core research result.
- **Weeks 10-11**: Write up the paper (introduction, related work from the reading list, methodology, results, discussion of explanation-alignment findings).
- **Weeks 12+**: Buffer for the inevitable — debugging, re-running experiments, polishing the report/slides.

## Known limitations (good to state honestly in your paper)

- The handcrafted features are regex/keyword-based, not a true static-analysis/AST-based detector — a false positive is possible if a keyword appears in a comment or unrelated context.
- CodeBERT is used in frozen/pretrained mode, not fine-tuned on flaky-test data specifically — fine-tuning (as in Flakify, uOttawa-Nanda-Lab/Flakify) would likely raise accuracy further and is a natural next step.
- The explanation-alignment check in `evaluate.py` is a reasonable proxy for "is this explanation good," but isn't a substitute for an actual human/developer study, which is worth mentioning as future work.

## Setting up a shared GitHub repo for both partners

1. One of you creates the repository on GitHub (Settings decide public vs. private — most courses are fine with private as long as the instructor/TA is added as a collaborator too).
2. Go to **Settings → Collaborators** on the repo and add your partner's GitHub username. They'll get an email invite; once accepted, the repo shows under both of your GitHub accounts.
3. Both of you should clone with your own GitHub credentials and commit under your own name/email (`git config user.name` / `git config user.email`) — this keeps an honest, accurate commit history, which your instructor may check.
4. List both names as the project team in the README (standard practice for a group submission) regardless of how the actual coding work is split — that's a normal and honest thing to do; it's just not the same as crediting commits that didn't happen.
