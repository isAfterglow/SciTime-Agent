# Generic Data Profile

## Dataset

- **dataset_path**: /home/ai4mater/003-Scitime-agent/scitime-agent/data/demo/generic_regression_demo.csv
- **target_col**: target_value

## Profile

- **n_rows**: 12
- **n_cols**: 6
- **columns**: ['sample_id', 'temperature', 'material_type', 'cycles', 'throughput', 'target_value']
- **numeric_cols**: ['temperature', 'cycles', 'throughput', 'target_value']
- **categorical_cols**: ['sample_id', 'material_type']
- **datetime_cols**: []
- **missing_ratio**: {'sample_id': 0.0, 'temperature': 0.0, 'material_type': 0.0, 'cycles': 0.0, 'throughput': 0.0, 'target_value': 0.0}
- **duplicate_rows**: 0
- **target_col**: material_type
- **numeric_summary**: {'temperature': {'mean': 35.0, 'std': 8.528028654224418, 'min': 25.0, 'max': 45.0}, 'cycles': {'mean': 159.16666666666666, 'std': 50.53501636279422, 'min': 90.0, 'max': 240.0}, 'throughput': {'mean': 185.6916666666667, 'std': 60.00908201466195, 'min': 100.0, 'max': 280.7}, 'target_value': {'mean': 0.85, 'std': 0.057682516651692016, 'min': 0.76, 'max': 0.94}}
