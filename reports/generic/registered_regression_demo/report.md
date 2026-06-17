# Generic Baseline Report

## Run Config

- **dataset_path**: /home/ai4mater/003-Scitime-agent/scitime-agent/data/demo/generic_regression_demo.csv
- **target_col**: target_value
- **task_type**: regression
- **n_features**: 5

## Feature Types

- **numeric_cols**: temperature, cycles, throughput
- **categorical_cols**: sample_id, material_type

## Metrics

model      mae     rmse       r2  n_train  n_test
     LinearRegression 0.008227 0.009076 0.913790        9       3
RandomForestRegressor 0.020333 0.022127 0.487644        9       3

## Artifacts

- **metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/generic/registered_regression_demo/metrics.csv
- **predictions**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/generic/registered_regression_demo/predictions.csv
- **manifest**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/generic/registered_regression_demo/manifest.json
