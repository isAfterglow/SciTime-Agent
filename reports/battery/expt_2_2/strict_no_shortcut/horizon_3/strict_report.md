# Strict No-Shortcut - horizon_3

## Run Config

- **task**: horizon_3
- **dataset**: data/processed/battery/expt_2_2/summary_horizon_3_dataset.csv
- **target**: target_soh_h3
- **feature_set**: process_plus_resistance
- **models**: GradientBoosting

## Dataset Shape

- **n_rows**: 60
- **n_cols**: 138

## Average Metrics

model      mae     rmse     mape      r2
GradientBoosting 0.007169 0.007952 0.779025 0.72515

## Artifacts

- **metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_no_shortcut/horizon_3/strict_metrics.csv
- **average_metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_no_shortcut/horizon_3/strict_average_metrics.csv
- **predictions**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_no_shortcut/horizon_3/strict_predictions.csv
- **manifest**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_no_shortcut/horizon_3/strict_manifest.json
- **figures_dir**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_no_shortcut/horizon_3/figures
