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

```bash
python run_experiments.py

# Run only MNIST experiments
python run_experiments.py --dataset mnist

# Run only experiments with uniform distribution
python run_experiments.py --distribute_mode uniform

# Run only experiments with specific epsilon value
python run_experiments.py --histogram_epsilon 1.0

# Run with 4 parallel workers
python run_experiments.py --parallel 4

# Run only the first 10 experiments
python run_experiments.py --limit 10

# Show what would be executed without running experiments
python run_experiments.py --dry_run
```