# Strict No-Shortcut - next_step

## Run Config

- **task**: next_step
- **dataset**: data/processed/battery/expt_2_2/summary_next_soh_dataset.csv
- **target**: target_soh_next
- **feature_set**: process_plus_resistance
- **models**: GradientBoosting

## Dataset Shape

- **n_rows**: 72
- **n_cols**: 138

## Average Metrics

model      mae     rmse     mape       r2
GradientBoosting 0.005978 0.006949 0.647884 0.885237

## Artifacts

- **metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_no_shortcut/next_step/strict_metrics.csv
- **average_metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_no_shortcut/next_step/strict_average_metrics.csv
- **predictions**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_no_shortcut/next_step/strict_predictions.csv
- **manifest**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_no_shortcut/next_step/strict_manifest.json
- **figures_dir**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_no_shortcut/next_step/figures
