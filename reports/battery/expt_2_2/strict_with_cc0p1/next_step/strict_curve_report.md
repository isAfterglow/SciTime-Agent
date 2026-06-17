# Strict + 0.1C - next_step

## Run Config

- **task**: next_step
- **dataset**: data/processed/battery/expt_2_2/summary_next_soh_dataset.csv
- **target**: target_soh_next
- **feature_set**: process_plus_resistance_plus_cc0p1
- **models**: GradientBoosting

## Dataset Shape

- **n_rows**: 72
- **n_cols**: 138

## Average Metrics

model      mae     rmse     mape       r2
GradientBoosting 0.002443 0.002944 0.263641 0.983411

## Artifacts

- **metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1/next_step/strict_curve_metrics.csv
- **average_metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1/next_step/strict_curve_average_metrics.csv
- **predictions**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1/next_step/strict_curve_predictions.csv
- **manifest**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1/next_step/strict_curve_manifest.json
- **figures_dir**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1/next_step/figures
