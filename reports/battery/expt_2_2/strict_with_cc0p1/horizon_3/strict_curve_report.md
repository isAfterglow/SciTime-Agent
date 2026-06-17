# Strict + 0.1C - horizon_3

## Run Config

- **task**: horizon_3
- **dataset**: data/processed/battery/expt_2_2/summary_horizon_3_dataset.csv
- **target**: target_soh_h3
- **feature_set**: process_plus_resistance_plus_cc0p1
- **models**: GradientBoosting

## Dataset Shape

- **n_rows**: 60
- **n_cols**: 138

## Average Metrics

model      mae     rmse     mape       r2
GradientBoosting 0.004852 0.005688 0.528719 0.876738

## Artifacts

- **metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1/horizon_3/strict_curve_metrics.csv
- **average_metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1/horizon_3/strict_curve_average_metrics.csv
- **predictions**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1/horizon_3/strict_curve_predictions.csv
- **manifest**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1/horizon_3/strict_curve_manifest.json
- **figures_dir**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1/horizon_3/figures
