# Battery Feature Ablation

## Run Config

- **dataset**: /home/ai4mater/003-Scitime-agent/scitime-agent/data/processed/battery/expt_2_2/summary_next_soh_dataset.csv
- **target**: target_soh_next
- **models**: GradientBoosting
- **output_dir**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/feature_ablation

## Dataset Shape

- **n_rows**: 72
- **n_cols**: 138

## Feature Sets

- persistence: soh -> y_pred
- process_only: temperature, rpt_index, charge_throughput, energy_throughput, ageing_cycles, days_of_degradation, delta_charge_throughput, soc_range
- process_plus_resistance: temperature, rpt_index, charge_throughput, energy_throughput, ageing_cycles, days_of_degradation, delta_charge_throughput, soc_range, resistance_0p1s, resistance_delta_prev
- capacity_only: capacity_c10
- capacity_with_delta: capacity_c10, capacity_delta_prev
- soh_only: soh
- soh_with_delta: soh, soh_delta_prev
- state_only: capacity_c10, soh, capacity_delta_prev, soh_delta_prev
- state_aware_full: temperature, rpt_index, charge_throughput, energy_throughput, ageing_cycles, days_of_degradation, delta_charge_throughput, soc_range, resistance_0p1s, resistance_delta_prev, capacity_c10, soh, capacity_delta_prev, soh_delta_prev

## Average Metrics

feature_set            model      mae     rmse     mape       r2
          capacity_only GradientBoosting 0.003718 0.004309 0.398884 0.962121
    capacity_with_delta GradientBoosting 0.003575 0.004301 0.383408 0.961114
            persistence      Persistence 0.008047 0.009813 0.851974 0.818590
           process_only GradientBoosting 0.005855 0.006776 0.635329 0.895893
process_plus_resistance GradientBoosting 0.005978 0.006949 0.647884 0.885237
               soh_only GradientBoosting 0.002623 0.002988 0.282103 0.982916
         soh_with_delta GradientBoosting 0.002420 0.002788 0.260333 0.984804
       state_aware_full GradientBoosting 0.002320 0.002689 0.249129 0.985281
             state_only GradientBoosting 0.002711 0.003211 0.290965 0.979509

## Artifacts

- **metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/feature_ablation/ablation_metrics.csv
- **average_metrics**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/feature_ablation/ablation_average_metrics.csv
- **predictions**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/feature_ablation/ablation_predictions.csv
- **manifest**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/feature_ablation/ablation_manifest.json
- **figures_dir**: /home/ai4mater/003-Scitime-agent/scitime-agent/reports/battery/expt_2_2/feature_ablation/figures
