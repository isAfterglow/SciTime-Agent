# strict_with_cc0p1_handcrafted_pca - next_step

## Run Config

- **task**: next_step
- **dataset**: data/processed/battery/expt_2_2/summary_next_soh_dataset.csv
- **target**: target_soh_next
- **variant**: strict_with_cc0p1_handcrafted_pca
- **include_handcrafted**: True
- **models**: GradientBoosting

## Dataset Shape

- **n_rows**: 72
- **n_cols**: 43

## Average Metrics

model      mae     rmse     mape       r2
GradientBoosting 0.002436 0.002864 0.262967 0.984352

## Artifacts

- **metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1_handcrafted_pca/next_step/metrics.csv
- **average_metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1_handcrafted_pca/next_step/average_metrics.csv
- **predictions**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1_handcrafted_pca/next_step/predictions.csv
- **manifest**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1_handcrafted_pca/next_step/manifest.json
- **figures_dir**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/strict_with_cc0p1_handcrafted_pca/next_step/figures
