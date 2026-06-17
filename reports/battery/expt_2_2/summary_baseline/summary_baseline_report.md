# Battery Summary Baseline

## Run Config

- **dataset**: /home/ai4mater/003-Scitime-agent/scitime-agent/data/processed/battery/expt_2_2/summary_next_soh_dataset.csv
- **rpt_table**: /home/ai4mater/003-Scitime-agent/scitime-agent/data/processed/battery/expt_2_2/rpt_summary_table.csv
- **target**: target_soh_next
- **output_dir**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/summary_baseline

## Dataset Shape

- **n_rows**: 72
- **n_cols**: 138

## Average Metrics

model      mae     rmse     mape       r2
GradientBoosting 0.002268 0.002619 0.243613 0.985980
LinearRegression 0.002586 0.003136 0.279586 0.981609
    RandomForest 0.002826 0.003314 0.303039 0.975906
           Ridge 0.002855 0.003417 0.306771 0.973861

## Artifacts

- **metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/summary_baseline/summary_baseline_metrics.csv
- **average_metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/summary_baseline/summary_baseline_average_metrics.csv
- **predictions**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/summary_baseline/summary_baseline_predictions.csv
- **figures_dir**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/summary_baseline/figures
