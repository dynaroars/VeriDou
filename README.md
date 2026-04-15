# Verifying Neural Network Robustness with Dual Perturbations

## 1. Installation

1. Clone the repository:
```bash
git clone https://github.com/dynaroars/VeriDou
cd VeriDou
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install verifiers:
```bash
# Venus
https://github.com/vas-group-imperial/venus2.git

# NeuralSAT
https://github.com/dynaroars/neuralsat.git

# alpha-beta-CROWN
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
```

## 2. Generating Benchmarks

### 2.1. Independent Benchmarks

```bash
python -m spec.generate_independent_benchmarks --output_dir "output/benchmark_independent"
```

**Parameters:**
- `--perturbation_types`: Motion blur (0°, 45°, 90°, 135°), box blur, sharpen
- `--strengths`: [0.2, 0.4, 0.6, 0.8, 1.0]
- `--kernel_sizes`: [3, 5, 7, 9]
- `--benchmark_names`: ["mnist_fc", "oval21", "sri_resnet_a", "cifar100", "tinyimagenet"]

### 2.2 Dual Benchmarks

```bash
python -m spec.generate_veridou_benchmarks --output_dir "output/benchmark_veridou"
```

**Parameters:**
- `--kernel_types`: ["veridou", "general"] - Type of kernel generation (default: ["general"])
- `--kernel_sizes`: [5, 7, 9] - Size of convolution kernels (default: [5, 7, 9])
- `--kernels`: ["0-30", "30-60", "60-90"] - Angle ranges for fixed kernels (default: ["0-30", "30-60", "60-90"])
- `--strength`: [lower, upper] - Lower and upper bounds for kernel strength (default: [-0.001, 0.001])
- `--perturb_ratios`: [0.0, 0.5, 1.0] - Ratio of pixels to perturb (default: [0.0, 0.5, 1.0])
- `--robust_intervals`: [0.005] - Robustness interval size (default: [0.005])
- `--benchmark_names`: ["mnist_fc", "oval21", "sri_resnet_a", "cifar100", "tinyimagenet"] - Benchmarks to generate
- `--mask_probability`: 0.5 - Probability threshold for mask generation in general kernels (default: 0.5)
- `--num_instances`: 20 - Number of instances to process from each CSV (default: 20)
- `--timeout`: 30.0 - Timeout for verification (default: 30.0)

## 3. Running Verifiers


```bash
python run_verifier.py --benchmark_folder "output/benchmark_independent" --verifier_path /absolute/path/to/[venus2|neuralsat|alpha-beta-CROWN]/ --result_dir "output/result_independent"
python run_verifier.py --benchmark_folder "output/benchmark_veridou" --verifier_path /absolute/path/to/[venus2|neuralsat|alpha-beta-CROWN]/ --result_dir "output/result_veridou"
```

<!-- 
### Alpha-Beta-CROWN Configuration

To run benchmarks with Alpha-Beta-CROWN, you may need to modify YAML configuration files. For example:

**Independent benchmarks** for `mnist_fc`:
- Configuration file: `alpha-beta-CROWN/complete_verifier/exp_configs/vnncomp22/mnistfc.yaml`


```yaml
...
model:
  input_shape: [-1, 1]
...
```
**veridou benchmarks** for `mnist_fc`:
- Configuration file: `alpha-beta-CROWN/complete_verifier/exp_configs/vnncomp22/mnistfc_independent.yaml`

```yaml
...
model:
  input_shape: [-1, 793]
...
```
**veridou benchmarks** for `oval21`:
- Configuration file: `alpha-beta-CROWN/complete_verifier/exp_configs/vnncomp22/oval22.yaml`
- Use the same configuration as `acasxu` benchmark:

```yaml
# Configuration file for running the ACASXu benchmark (all properties).
general:
  root_path: ../../vnncomp2023_benchmarks/benchmarks/acasxu  # Please clone the vnncomp2023 repo first as it contains the benchmark.
  csv_name: instances.csv
  enable_incomplete_verification: False
solver:
  batch_size: 1000  # Number of parallel domains to compute on GPU.
  bound_prop_method: crown
bab:
  branching:
    method: naive  # Split on input space.
    input_split:
      enable: True
      enable_clip_domains: True
      reorder_bab: True
attack:
  pgd_order: after
  pgd_restarts: 10000
  pgd_restart_when_stuck: True
```

### Visualizing Kernels

Visualize kernels extracted from verification logs:

```bash
python visualize_log_kernels.py --log_dir <log_directory> --benchmark_dir <benchmark_directory> --output_dir <output_directory>
```

### Analysis

Run unified analysis on verification results:

```bash
python unified_analysis.py
``` -->


## 4. Results

### Summarizing Results

```bash
python summarize_results.py --result_dir "output/result_independent" --benchmark [independent|veridou] --verifier [venus|crown|neuralsat]
python summarize_results.py --result_dir "output/result_veridou" --benchmark [independent|veridou] --verifier [venus|crown|neuralsat]
```
