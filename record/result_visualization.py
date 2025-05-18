import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import os
import re
import numpy as np

# Change to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get all .pth files from all output_* directories
pth_files = []
output_dirs = glob.glob(os.path.join(script_dir, "output_*"))
for output_dir in output_dirs:
    pth_files.extend(glob.glob(os.path.join(output_dir, "*.pth")))

# Create a figure
plt.figure(figsize=(30, 10))

# Set global font size
plt.rcParams.update({'font.size': 14})

# Define colorblind-friendly colors and line styles
colors = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]
line_width = 2.5  # Increased line width for better visibility

# Function to parse the filename and extract parameters
def parse_filename(filename):
    # Pattern from FedAvg.py: 
    # fedavg_{dataset_name}_seed{seed}_{mode}_s{n_silos}_ua{user_alpha}_sa{silo_alpha}_{is_secure}_{is_histogram_secure}_he{histogram_epsilon}_sigma{sigma}_mgn{max_grad_norm}_llr{local_learning_rate}_glr{global_learning_rate}_u{n_users}_i{iters}_e{local_epochs}.pth
    basename = os.path.basename(filename)
    
    # Extract parameters using regex
    dataset_match = re.search(r'fedavg_([^_]+)_', basename)
    seed_match = re.search(r'seed(\d+)', basename)
    mode_match = re.search(r'seed\d+_([^_]+)_s\d+', basename)
    silos_match = re.search(r's(\d+)', basename)
    user_alpha_match = re.search(r'ua([\d\.]+)', basename)
    silo_alpha_match = re.search(r'sa([\d\.]+)', basename)
    is_secure_match = re.search(r'sa[\d\.]+_([^_]+)_', basename)
    is_histogram_secure_match = re.search(r'sa[\d\.]+_[^_]+_([^_]+)_', basename)
    histogram_epsilon_match = re.search(r'he([\d\.]+)', basename)
    sigma_match = re.search(r'sigma([\d\.]+)', basename)
    max_grad_norm_match = re.search(r'mgn([\d\.]+)', basename)
    local_lr_match = re.search(r'llr([\d\.]+)', basename)
    global_lr_match = re.search(r'glr([\d\.]+)', basename)
    users_match = re.search(r'u(\d+)', basename)
    iters_match = re.search(r'i(\d+)', basename)
    epochs_match = re.search(r'e(\d+)\.pth', basename)
    
    # Default values
    params = {
        'dataset': dataset_match.group(1) if dataset_match else "",
        'seed': int(seed_match.group(1)) if seed_match else 0,
        'mode': mode_match.group(1) if mode_match else "",
        'n_silos': int(silos_match.group(1)) if silos_match else 0,
        'user_alpha': float(user_alpha_match.group(1)) if user_alpha_match else 0.0,
        'silo_alpha': float(silo_alpha_match.group(1)) if silo_alpha_match else 0.0,
        'is_secure': is_secure_match.group(1) if is_secure_match else "",
        'is_histogram_secure': is_histogram_secure_match.group(1) if is_histogram_secure_match else "",
        'histogram_epsilon': float(histogram_epsilon_match.group(1)) if histogram_epsilon_match else 0.0,
        'sigma': float(sigma_match.group(1)) if sigma_match else 0.0,
        'max_grad_norm': float(max_grad_norm_match.group(1)) if max_grad_norm_match else 0.0,
        'local_learning_rate': float(local_lr_match.group(1)) if local_lr_match else 0.0,
        'global_learning_rate': float(global_lr_match.group(1)) if global_lr_match else 0.0,
        'n_users': int(users_match.group(1)) if users_match else 0,
        'iters': int(iters_match.group(1)) if iters_match else 0,
        'local_epochs': int(epochs_match.group(1)) if epochs_match else 0
    }
    
    return params

# Function to generate legend label based on filename
def generate_label(filename, params):
    basename = os.path.basename(filename)
    
    # Check for the special cases
    if "uldp_dp" in basename:
        model_type = "ULDP-AVG-w-Hist-DP"
    elif "uldp_nf_dp" in basename:
        model_type = "ULDP-AVG-w-Hist-NF-DP"
    elif "uldp_sf_dp" in basename:
        model_type = "ULDP-AVG-w-Hist-SF-DP"
    else:
        model_type = params["is_secure"]
    
    # Add mode in parentheses if it's uniform or zipf
    mode_str = ""
    if params["mode"] == "uniform":
        mode_str = " (uniform)"
    elif params["mode"] == "zipf":
        mode_str = " (zipf)"
    
    return f'{model_type}{mode_str}'

# Group files by specific parameters
# For example, group by local_epochs to compare the effect of different epoch settings
grouped_by_epochs = {}

for pth_file in pth_files:
    params = parse_filename(pth_file)
    local_epochs = params['local_epochs']
    
    if local_epochs not in grouped_by_epochs:
        grouped_by_epochs[local_epochs] = []
    
    grouped_by_epochs[local_epochs].append((pth_file, params))

# Loss curve
plt.subplot(1, 3, 1)
for idx, (local_epochs, files_params) in enumerate(grouped_by_epochs.items()):
    for inner_idx, (pth_file, params) in enumerate(files_params):
        # Load record
        record = torch.load(pth_file, map_location=torch.device('cpu'), weights_only=False)
        
        # Create a label with the new format
        label = generate_label(pth_file, params)
        
        # Plot loss curve with color-blind friendly colors and different line styles
        color_idx = (idx * len(files_params) + inner_idx) % len(colors)
        style_idx = (idx * len(files_params) + inner_idx) % len(line_styles)
        plt.plot(record['training_history']['losses'], 
                label=label,
                color=colors[color_idx],
                linestyle=line_styles[style_idx],
                linewidth=line_width)

plt.title('Training Loss Comparison', fontsize=24)
plt.xlabel('Iteration', fontsize=22)
plt.ylabel('Loss', fontsize=22)
plt.legend(fontsize=18, loc='upper right')
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Accuracy curve
plt.subplot(1, 3, 2)
for idx, (local_epochs, files_params) in enumerate(grouped_by_epochs.items()):
    for inner_idx, (pth_file, params) in enumerate(files_params):
        # Load record
        record = torch.load(pth_file, map_location=torch.device('cpu'), weights_only=False)
        
        # Create a label with the new format
        label = generate_label(pth_file, params)
        
        # Plot accuracy curve with color-blind friendly colors and different line styles
        color_idx = (idx * len(files_params) + inner_idx) % len(colors)
        style_idx = (idx * len(files_params) + inner_idx) % len(line_styles)
        plt.plot(record['training_history']['accuracies'], 
                label=label,
                color=colors[color_idx],
                linestyle=line_styles[style_idx],
                linewidth=line_width)

plt.title('Training Accuracy Comparison', fontsize=24)
plt.xlabel('Iteration', fontsize=22)
plt.ylabel('Accuracy (%)', fontsize=22)
plt.legend(fontsize=18, loc='lower right')
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Epsuilon curve
plt.subplot(1, 3, 3)
for idx, (local_epochs, files_params) in enumerate(grouped_by_epochs.items()):
    for inner_idx, (pth_file, params) in enumerate(files_params):
        # Load record
        record = torch.load(pth_file, map_location=torch.device('cpu'), weights_only=False)
        
        # Create a label with the new format
        label = generate_label(pth_file, params)
        
        # Plot accuracy curve with color-blind friendly colors and different line styles
        color_idx = (idx * len(files_params) + inner_idx) % len(colors)
        style_idx = (idx * len(files_params) + inner_idx) % len(line_styles)
        plt.plot(record['training_history']['consumed_epsilons'], 
                label=label,
                color=colors[color_idx],
                linestyle=line_styles[style_idx],
                linewidth=line_width)

plt.title('Training Epsilon Comparison', fontsize=24)
plt.xlabel('Iteration', fontsize=22)
plt.ylabel('Epsilon', fontsize=22)
plt.yscale('log')  # Keep logarithmic scale for epsilon
# Format y-axis to show powers of 10 only for epsilon plot
formatter = ticker.FuncFormatter(lambda y, _: f'$10^{{{int(round(np.log10(y)))}}}$' if y > 0 else '0')
plt.gca().yaxis.set_major_formatter(formatter)
plt.legend(fontsize=18, loc='lower right')
plt.grid(True, alpha=0.3, which='both')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Adjust layout to prevent legend cutoff
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'fedavg_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()