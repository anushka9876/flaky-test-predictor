# Datasets

This project does not bundle the datasets — they belong to their original authors.
Download them yourself and place the CSVs here as described below.

## 1. IDoFT (primary flaky/non-flaky labels)

Repository: https://github.com/TestingResearchIllinois/idoft

This is a large, continuously-updated catalog of flaky tests mined from real
open-source Java projects. It is distributed as a set of spreadsheets/CSVs
per project rather than one single file. For this project:

1. Clone or download the repository.
2. Pick a handful of projects (start with 3-5 to keep training time reasonable).
3. For each flaky test listed, pull the actual test source code from that
   project's GitHub repo at the commit referenced.
4. Build a single CSV named `idoft_tests.csv` with columns:
   - `test_code` (the full source of the test method, as a string)
   - `label` (`flaky` or `not_flaky`)

A non-flaky comparison set can be built by sampling other test methods from
the same files/projects that are NOT listed as flaky.

## 2. FlakyCat (root-cause categories)

Repository: https://github.com/Amal-AK/FLAKYCAT

This dataset already provides flaky tests labeled by root-cause category
(e.g. Concurrency, Async Wait, Unordered Collection, Time, Order Dependency).
Export it to a CSV named `flakycat_tests.csv` with columns:
   - `test_code`
   - `category` (one of FlakyCat's five root-cause labels)

This is the file `src/explain.py` uses to check whether SHAP's explanations
line up with the real documented cause of each flaky test.

## 3. (Optional, for extra reference) Flakify replication package

https://github.com/uOttawa-Nanda-Lab/Flakify

Not required, but worth reading — it's another CodeBERT-based flaky-test
predictor and its replication package shows a working example of how to
structure this kind of pipeline end to end.

## Expected folder layout once you're done

```
data/
  idoft_tests.csv
  flakycat_tests.csv
```
