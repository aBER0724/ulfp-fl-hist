# Experiment Runner

This script automates running multiple experiments with various parameter combinations for federated learning experiments.

## Overview

The experiment runner script (`run_experiments.py`) allows you to:

1. Run experiments with different parameter combinations
2. Filter experiments by specific parameters
3. Run experiments in parallel
4. Monitor progress with a progress bar

## Parameters

The script supports all the parameters from the original FedAvg script, with these default values:

### General Parameters
- **dataset**: mnist, creditcard
- **seed**: 114514  
- **distribute_mode**: uniform, zipf, dirichlet  
- **n_silos**: 5
- **user_alpha**: 0.5  
- **silo_alpha**: 2.0  
- **is_secure**: ndp, uldp  
- **is_histogram_secure**: ndp, dp, nf_dp, sf_dp  
- **histogram_epsilon**: 1.0, 3.0, 5.0   
- **sigma**: 5.0    
- **max_grad_norm**: 1.0
- **local_learning_rate**: 0.01  
- **global_learning_rate**: 10.0  

### Dataset-Specific Parameters

#### CreditCard 
- **n_users**: 100, 1000  
- **iters**: 100  
- **local_epoch**: 30  

#### MNIST
- **n_users**: 100, 10000  
- **iters**: 200  
- **local_epoch**: 20  

## Usage

### Basic Usage

Run all experiment combinations:

```bash
python run_experiments.py
```

### Filtering Experiments

Filter by specific parameters:

```bash
# Run only MNIST experiments
python run_experiments.py --dataset mnist

# Run only experiments with uniform distribution
python run_experiments.py --distribute_mode uniform

# Run only experiments with specific epsilon value
python run_experiments.py --histogram_epsilon 1.0
```

### Parallel Execution

Run experiments in parallel:

```bash
# Run with 4 parallel workers
python run_experiments.py --parallel 4
```

### Limiting Experiments

Limit the number of experiments:

```bash
# Run only the first 10 experiments
python run_experiments.py --limit 10
```

### Dry Run

Print commands without executing them:

```bash
# Show what would be executed without running experiments
python run_experiments.py --dry_run
```

## Example Commands

```bash
# Run MNIST experiments with uniform distribution in parallel
python run_experiments.py --dataset mnist --distribute_mode uniform --parallel 4

# Run only 5 CreditCard experiments with specific number of users
python run_experiments.py --dataset creditcard --n_users 100 --limit 5

# Print all experiment commands without running them
python run_experiments.py --dry_run
```

## Output

The script provides detailed logging of:
- Experiment start time
- Commands being executed
- Progress (via tqdm progress bar)
- Success/failure status of each experiment
- Final statistics summary 