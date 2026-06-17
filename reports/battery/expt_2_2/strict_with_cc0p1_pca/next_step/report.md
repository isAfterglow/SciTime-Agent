# strict_with_cc0p1_pca - next_step

## Run Config

- **task**: next_step
- **dataset**: data/processed/battery/expt_2_2/summary_next_soh_dataset.csv
- **target**: target_soh_next
- **variant**: strict_with_cc0p1_pca
- **include_handcrafted**: False
- **models**: GradientBoosting

## Dataset Shape

- **n_rows**: 72
- **n_cols**: 43

## Average Metrics

model      mae     rmse     mape       r2
GradientBoosting 0.004297 0.005147 0.463725 0.941742

## Artifacts

- **metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1_pca/next_step/metrics.csv
- **average_metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1_pca/next_step/average_metrics.csv
- **predictions**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1_pca/next_step/predictions.csv
- **manifest**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1_pca/next_step/manifest.json
- **figures_dir**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1_pca/next_step/figures
