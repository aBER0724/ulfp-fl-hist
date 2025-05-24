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
IS_SECURE_OPTIONS = ["uldp"]
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

def create_log_filename(params, timestamp):
    """
    Creates a structured log filename based on experiment parameters and a timestamp.

    Args:
        params (dict): A dictionary containing the parameters for the experiment.
                       Expected keys include 'dataset', 'distribute_mode', 'n_silos',
                       'n_users', 'is_secure', 'is_histogram_secure', 'histogram_epsilon'.
        timestamp (str): A timestamp string (e.g., "YYYYMMDD_HHMMSS") to prefix the log file.

    Returns:
        str: A log filename string (e.g., "YYYYMMDD_HHMMSS_dataset_mnist_dist_uniform_....log").
    """
    parts = [
        timestamp,
        f"dataset_{params['dataset']}",
        f"dist_{params['distribute_mode']}",
        f"silos_{params['n_silos']}",
        f"users_{params['n_users']}",
        f"secure_{params['is_secure']}",
        f"hist_{params['is_histogram_secure']}",
        f"eps_{params['histogram_epsilon']}"
    ]
    return "_".join(parts) + ".log"

def run_experiment(params, experiment_idx=None, total=None, log_dir=None, timestamp=None):
    """
    Runs a single federated learning experiment using FL/FedAvg.py with the given parameters.

    This function constructs a command, executes it as a subprocess, and handles logging
    of the experiment's output.

    Args:
        params (dict): A dictionary of parameters to be passed as command-line arguments
                       to FL/FedAvg.py.
        experiment_idx (int, optional): The index of the current experiment (for logging).
        total (int, optional): The total number of experiments being run (for logging).
        log_dir (str, optional): The directory where log files should be stored.
                                 If None, output goes to console.
        timestamp (str, optional): A timestamp string used for creating unique log filenames
                                   if log_dir is specified.

    Returns:
        bool: True if the experiment subprocess completed with a return code of 0, False otherwise.

    Side Effects:
        - Executes `FL/FedAvg.py` as a subprocess.
        - If `log_dir` and `timestamp` are provided, creates a log file in `log_dir`
          and writes the experiment's stdout and stderr to it.
        - Logs information about the experiment execution using the `logger`.
    """
    cmd = ["python", "FL/FedAvg.py"]
    
    # Add all parameters to command
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])
    
    # Log the command
    if experiment_idx is not None:
        logger.info(f"Experiment {experiment_idx+1}/{total}")
    logger.info(f"Running: {' '.join(cmd)}")
    
    # Create log file if log_dir is specified
    if log_dir and timestamp:
        os.makedirs(log_dir, exist_ok=True)
        log_filename = create_log_filename(params, timestamp)
        log_path = os.path.join(log_dir, log_filename)
        logger.info(f"Logging output to: {log_path}")
        
        with open(log_path, 'w') as log_file:
            # Write experiment info to log file
            log_file.write(f"Command: {' '.join(cmd)}\n")
            log_file.write(f"Parameters: {params}\n")
            log_file.write(f"Timestamp: {timestamp}\n")
            log_file.write("="*80 + "\n\n")
            log_file.flush()
            
            # Run the command with output redirected to log file
            process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            process.wait()
    else:
        # Run the command with output to console
        process = subprocess.Popen(cmd)
        process.wait()
    
    if process.returncode != 0:
        logger.error(f"Experiment failed with exit code {process.returncode}")
    else:
        logger.info(f"Experiment completed successfully")
    
    return process.returncode == 0

def generate_experiment_combinations(args):
    """
    Generates a list of all possible experiment parameter combinations.

    It considers base parameters and dataset-specific parameters. Filters can be applied
    via the `args` object to narrow down the combinations.

    Args:
        args (argparse.Namespace): Parsed command-line arguments that may contain
                                   filters for datasets, distribution modes, security options, etc.

    Returns:
        list[dict]: A list where each element is a dictionary representing
                    a unique set of parameters for an experiment.
    """
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
    """
    Parses command-line arguments for the experiment runner script.

    Defines arguments for filtering experiments (e.g., by dataset, number of users)
    and for controlling execution (e.g., dry run, parallelism, logging).

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
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
    parser.add_argument("--log_dir", type=str, default="experiment_logs", help="Directory to store experiment logs")
    parser.add_argument("--console_output", action="store_true", help="Show experiment output in console instead of log files")
    
    return parser.parse_args()

def run_experiments_sequentially(experiments, log_dir, timestamp, console_output):
    """
    Runs a list of experiments sequentially, one after another.

    Args:
        experiments (list[dict]): A list of parameter dictionaries, each defining an experiment.
        log_dir (str, optional): Directory to store log files. If None, output to console.
        timestamp (str): Timestamp string for log file naming.
        console_output (bool): If True, directs experiment output to console instead of log files.

    Returns:
        tuple[int, int]: A tuple containing the number of successful experiments
                         and the number of failed experiments.
    """
    total_experiments = len(experiments)
    successful = 0
    failed = 0
    
    for i, params in enumerate(tqdm(experiments, desc="Running experiments")):
        # Show current experiment details
        success = run_experiment(
            params, 
            i, 
            total_experiments, 
            None if console_output else log_dir, 
            timestamp
        )
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Brief pause between experiments
        time.sleep(1)
    
    return successful, failed

def run_experiments_parallel(experiments, max_workers, log_dir, timestamp, console_output):
    """
    Runs a list of experiments in parallel using a process pool.

    Args:
        experiments (list[dict]): A list of parameter dictionaries, each defining an experiment.
        max_workers (int): The maximum number of parallel processes to use.
        log_dir (str, optional): Directory to store log files. If None, output to console.
        timestamp (str): Timestamp string for log file naming.
        console_output (bool): If True, directs experiment output to console. Note that
                               output may be interleaved when running in parallel with console output.

    Returns:
        tuple[int, int]: A tuple containing the number of successful experiments
                         and the number of failed experiments.
    """
    total_experiments = len(experiments)
    successful = 0
    failed = 0
    
    if console_output:
        logger.warning("Note: When running in parallel mode with console output, output from individual experiments may be mixed together.")
        logger.warning("Consider using log files (remove --console_output) for cleaner output.")
    
    # Create a progress bar
    pbar = tqdm(total=total_experiments, desc="Running experiments")
    
    # Run experiments in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Prepare the futures
        future_to_idx = {}
        for i, params in enumerate(experiments):
            future = executor.submit(
                run_experiment, 
                params, 
                i, 
                total_experiments, 
                None if console_output else log_dir, 
                timestamp
            )
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
    """
    Main function to set up and run federated learning experiments.

    Parses command-line arguments, generates experiment parameter combinations,
    and then runs these experiments either sequentially or in parallel based on
    the provided arguments. Logs results and outputs.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Create a timestamp for this batch of experiments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting experiment batch at {timestamp}")
    
    # Create log directory if not using console output
    if not args.console_output:
        log_dir = f"{args.log_dir}_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"Experiment logs will be stored in: {log_dir}")
    else:
        log_dir = None
    
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
        successful, failed = run_experiments_parallel(
            all_experiments, 
            args.parallel, 
            log_dir, 
            timestamp, 
            args.console_output
        )
    else:
        logger.info("Running experiments sequentially")
        successful, failed = run_experiments_sequentially(
            all_experiments, 
            log_dir, 
            timestamp, 
            args.console_output
        )
    
    # Log final statistics
    logger.info(f"Experiment batch completed")
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    if not args.console_output:
        logger.info(f"Experiment logs are stored in: {log_dir}")

if __name__ == "__main__":
    main() 