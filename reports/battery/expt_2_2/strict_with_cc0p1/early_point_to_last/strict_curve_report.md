# Strict + 0.1C - early_point_to_last

## Run Config

- **task**: early_point_to_last
- **dataset**: data/processed/battery/expt_2_2/summary_early_point_to_last_dataset.csv
- **target**: target_soh_last
- **feature_set**: process_plus_resistance_plus_cc0p1
- **models**: GradientBoosting

## Dataset Shape

- **n_rows**: 24
- **n_cols**: 139

## Average Metrics

model      mae    rmse     mape  r2
GradientBoosting 0.010521 0.01102 1.159344 0.0

## Artifacts

- **metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1/early_point_to_last/strict_curve_metrics.csv
- **average_metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1/early_point_to_last/strict_curve_average_metrics.csv
- **predictions**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1/early_point_to_last/strict_curve_predictions.csv
- **manifest**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1/early_point_to_last/strict_curve_manifest.json
- **figures_dir**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1/early_point_to_last/figures
