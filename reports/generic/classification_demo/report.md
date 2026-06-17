# Generic Baseline Report

## Run Config

- **dataset_path**: /home/ai4mater/003-Scitime-agent/scitime-agent/data/demo/generic_classification_demo.csv
- **target_col**: fast_fail
- **task_type**: classification
- **n_features**: 5

## Feature Types

- **numeric_cols**: temperature, cycles, throughput
- **categorical_cols**: sample_id, material_type

## Metrics

model  accuracy  f1  roc_auc  n_train  n_test
    LogisticRegression       1.0 1.0      1.0        9       3
RandomForestClassifier       1.0 1.0      1.0        9       3

## Artifacts

- **metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/generic/classification_demo/metrics.csv
- **predictions**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/generic/classification_demo/predictions.csv
- **manifest**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/generic/classification_demo/manifest.json
