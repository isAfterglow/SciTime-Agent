# strict_with_cc0p1_handcrafted_pca - early_point_to_last

## Run Config

- **task**: early_point_to_last
- **dataset**: data/processed/battery/expt_2_2/summary_early_point_to_last_dataset.csv
- **target**: target_soh_last
- **variant**: strict_with_cc0p1_handcrafted_pca
- **include_handcrafted**: True
- **models**: GradientBoosting

## Dataset Shape

- **n_rows**: 24
- **n_cols**: 44

## Average Metrics

model      mae   rmse     mape  r2
GradientBoosting 0.008597 0.0091 0.949557 0.0

## Artifacts

- **metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1_handcrafted_pca/early_point_to_last/metrics.csv
- **average_metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1_handcrafted_pca/early_point_to_last/average_metrics.csv
- **predictions**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1_handcrafted_pca/early_point_to_last/predictions.csv
- **manifest**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1_handcrafted_pca/early_point_to_last/manifest.json
- **figures_dir**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1_handcrafted_pca/early_point_to_last/figures
