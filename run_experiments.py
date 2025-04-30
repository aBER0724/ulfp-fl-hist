#!/usr/bin/env python
import argparse
import itertools
import subprocess
import os
import time
import concurrent.futures
from tqdm import tqdm
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# General Parameters
DATASETS = ["mnist", "creditcard"]
SEEDS = [114514]
DISTRIBUTE_MODES = ["uniform", "zipf"]
N_SILOS = [5]
USER_ALPHAS = [0.5]
SILO_ALPHAS = [2.0]
IS_SECURE_OPTIONS = ["ndp", "uldp"]
IS_HISTOGRAM_SECURE_OPTIONS = ["dp", "nf_dp", "sf_dp"]
HISTOGRAM_EPSILONS = [1.0, 3.0]
SIGMAS = [5.0]
MAX_GRAD_NORMS = [1.0]
LOCAL_LEARNING_RATES = [0.01]
GLOBAL_LEARNING_RATES = [10.0]

# Dataset specific parameters
DATASET_PARAMS = {
    "creditcard": {
        "n_users": [100, 1000],
        "iters": [100],
        "local_epoch": [30]
    },
    "mnist": {
        "n_users": [100, 10000],
        "iters": [200],
        "local_epoch": [20]
    }
}

def run_experiment(params, experiment_idx=None, total=None):
    """Run a single experiment with the given parameters."""
    cmd = ["python", "FL/FedAvg.py"]
    
    # Add all parameters to command
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])
    
    # Log the command
    if experiment_idx is not None:
        logger.info(f"Experiment {experiment_idx+1}/{total}")
    logger.info(f"Running: {' '.join(cmd)}")
    
    # Run the command
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        logger.error(f"Experiment failed with exit code {process.returncode}")
        logger.error(f"stderr: {stderr.decode('utf-8')}")
    else:
        logger.info(f"Experiment completed successfully")
    
    return process.returncode == 0

def generate_experiment_combinations(args):
    """Generate all combinations of experiment parameters."""
    all_experiments = []
    
    # Use filtered parameters based on command line arguments
    datasets = [args.dataset] if args.dataset else DATASETS
    distribute_modes = [args.distribute_mode] if args.distribute_mode else DISTRIBUTE_MODES
    is_secure_options = [args.is_secure] if args.is_secure else IS_SECURE_OPTIONS
    is_histogram_secure_options = [args.is_histogram_secure] if args.is_histogram_secure else IS_HISTOGRAM_SECURE_OPTIONS
    histogram_epsilons = [float(args.histogram_epsilon)] if args.histogram_epsilon else HISTOGRAM_EPSILONS
    n_users_filter = args.n_users if args.n_users else None
    
    for dataset in datasets:
        # Use dataset-specific parameters
        dataset_specific = DATASET_PARAMS[dataset]
        n_users_list = [n_users_filter] if n_users_filter else dataset_specific["n_users"]
        iters_list = dataset_specific["iters"]
        local_epoch_list = dataset_specific["local_epoch"]
        
        # Generate parameter combinations
        combinations = list(itertools.product(
            [dataset],
            SEEDS,
            distribute_modes,
            N_SILOS,
            n_users_list,
            USER_ALPHAS,
            SILO_ALPHAS,
            is_secure_options,
            is_histogram_secure_options,
            histogram_epsilons,
            SIGMAS,
            MAX_GRAD_NORMS,
            LOCAL_LEARNING_RATES,
            GLOBAL_LEARNING_RATES,
            iters_list,
            local_epoch_list
        ))
        
        # Convert combinations to parameter dictionaries
        for combo in combinations:
            params = {
                "dataset": combo[0],
                "seed": combo[1],
                "distribute_mode": combo[2],
                "n_silos": combo[3],
                "n_users": combo[4],
                "user_alpha": combo[5],
                "silo_alpha": combo[6],
                "is_secure": combo[7],
                "is_histogram_secure": combo[8],
                "histogram_epsilon": combo[9],
                "sigma": combo[10],
                "max_grad_norm": combo[11],
                "local_learning_rate": combo[12],
                "global_learning_rate": combo[13],
                "iters": combo[14],
                "local_epoch": combo[15]
            }
            all_experiments.append(params)
    
    return all_experiments

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run multiple FL experiments with different parameter combinations")
    
    # Filters for experiment parameters
    parser.add_argument("--dataset", type=str, choices=DATASETS, help="Filter by dataset")
    parser.add_argument("--distribute_mode", type=str, choices=DISTRIBUTE_MODES, help="Filter by distribution mode")
    parser.add_argument("--is_secure", type=str, choices=IS_SECURE_OPTIONS, help="Filter by security option")
    parser.add_argument("--is_histogram_secure", type=str, choices=IS_HISTOGRAM_SECURE_OPTIONS, help="Filter by histogram security option")
    parser.add_argument("--histogram_epsilon", type=str, help="Filter by histogram epsilon")
    parser.add_argument("--n_users", type=int, help="Filter by number of users")
    
    # Run options
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--parallel", type=int, default=1, help="Number of experiments to run in parallel")
    parser.add_argument("--limit", type=int, help="Limit the number of experiments to run")
    
    return parser.parse_args()

def run_experiments_sequentially(experiments):
    """Run experiments one after another."""
    total_experiments = len(experiments)
    successful = 0
    failed = 0
    
    for i, params in enumerate(tqdm(experiments, desc="Running experiments")):
        # Show current experiment details
        success = run_experiment(params, i, total_experiments)
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Brief pause between experiments
        time.sleep(1)
    
    return successful, failed

def run_experiments_parallel(experiments, max_workers):
    """Run experiments in parallel."""
    total_experiments = len(experiments)
    successful = 0
    failed = 0
    
    # Create a progress bar
    pbar = tqdm(total=total_experiments, desc="Running experiments")
    
    # Run experiments in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Prepare the futures
        future_to_idx = {}
        for i, params in enumerate(experiments):
            future = executor.submit(run_experiment, params)
            future_to_idx[future] = i
        
        # Process the results as they complete
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                success = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as exc:
                logger.error(f"Experiment {idx+1} generated an exception: {exc}")
                failed += 1
            
            # Update the progress bar
            pbar.update(1)
    
    pbar.close()
    return successful, failed

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create a timestamp for this batch of experiments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting experiment batch at {timestamp}")
    
    # Generate all experiment combinations
    all_experiments = generate_experiment_combinations(args)
    total_experiments = len(all_experiments)
    
    if args.limit and args.limit < total_experiments:
        all_experiments = all_experiments[:args.limit]
        logger.info(f"Limiting to {args.limit} experiments")
        total_experiments = len(all_experiments)
    
    logger.info(f"Generated {total_experiments} experiment combinations")
    
    if args.dry_run:
        logger.info("Dry run mode: commands will be printed but not executed")
        for i, params in enumerate(all_experiments):
            cmd = ["python", "FL/FedAvg.py"]
            for key, value in params.items():
                cmd.extend([f"--{key}", str(value)])
            logger.info(f"Experiment {i+1}/{total_experiments}: {' '.join(cmd)}")
        return
    
    # Run experiments
    if args.parallel > 1:
        logger.info(f"Running experiments in parallel with {args.parallel} workers")
        successful, failed = run_experiments_parallel(all_experiments, args.parallel)
    else:
        logger.info("Running experiments sequentially")
        successful, failed = run_experiments_sequentially(all_experiments)
    
    # Log final statistics
    logger.info(f"Experiment batch completed")
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")

if __name__ == "__main__":
    main() 