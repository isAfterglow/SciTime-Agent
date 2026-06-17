# strict_with_cc0p1_pca - horizon_3

## Run Config

- **task**: horizon_3
- **dataset**: data/processed/battery/expt_2_2/summary_horizon_3_dataset.csv
- **target**: target_soh_h3
- **variant**: strict_with_cc0p1_pca
- **include_handcrafted**: False
- **models**: GradientBoosting

## Dataset Shape

- **n_rows**: 60
- **n_cols**: 43

## Average Metrics

model      mae     rmse     mape       r2
GradientBoosting 0.005491 0.006376 0.596837 0.837239

## Artifacts

- **metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1_pca/horizon_3/metrics.csv
- **average_metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1_pca/horizon_3/average_metrics.csv
- **predictions**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1_pca/horizon_3/predictions.csv
- **manifest**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1_pca/horizon_3/manifest.json
- **figures_dir**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1_pca/horizon_3/figures
