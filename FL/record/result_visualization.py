import torch
import matplotlib.pyplot as plt
import glob
import os
import re

# Change to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get all .pth files from record_output directory
pth_files = glob.glob(os.path.join(script_dir, "record_output", "fedavg_*.pth"))

# Create a figure
plt.figure(figsize=(30, 10))

# Define colorblind-friendly colors and line styles
colors = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]
line_width = 2.5  # Increased line width for better visibility

# Function to parse the filename and extract parameters
def parse_filename(filename):
    # Pattern: fedavg_s{n_silos}_u{n_users}_e{local_epochs}_b{batch_size}_lr{learning_rate}_a{alpha}_{mode}_{train_mode}_i{iters}.pth
    basename = os.path.basename(filename)
    
    # Extract parameters using regex
    silos_match = re.search(r's(\d+)', basename)
    users_match = re.search(r'u(\d+)', basename)
    epochs_match = re.search(r'e(\d+)', basename)
    batch_match = re.search(r'b(\d+)', basename)
    lr_match = re.search(r'lr([\d\.]+)', basename)
    alpha_match = re.search(r'a([\d\.]+)', basename)
    iters_match = re.search(r'i(\d+)', basename)
    
    # Get mode and train_mode
    parts = basename.split('_')
    mode_idx = -3  # Third from the end
    train_mode_idx = -2  # Second from the end
    
    # Default values
    params = {
        'n_silos': int(silos_match.group(1)) if silos_match else 0,
        'n_users': int(users_match.group(1)) if users_match else 0,
        'local_epochs': int(epochs_match.group(1)) if epochs_match else 0,
        'batch_size': int(batch_match.group(1)) if batch_match else 0,
        'learning_rate': float(lr_match.group(1)) if lr_match else 0.0,
        'alpha': float(alpha_match.group(1)) if alpha_match else 0.0,
        'mode': parts[mode_idx] if len(parts) > abs(mode_idx) else "",
        'train_mode': parts[train_mode_idx] if len(parts) > abs(train_mode_idx) else "",
        'iters': int(iters_match.group(1)) if iters_match else 0
    }
    
    return params

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
plt.subplot(1, 2, 1)
for idx, (local_epochs, files_params) in enumerate(grouped_by_epochs.items()):
    for inner_idx, (pth_file, params) in enumerate(files_params):
        # Load record
        record = torch.load(pth_file)
        
        # Create a label with relevant parameters
        label = f'E={local_epochs}, Mode={params["mode"]}'
        if params["train_mode"] != '':
            label += f'_{params["train_mode"]}'
        
        # Plot loss curve with color-blind friendly colors and different line styles
        color_idx = (idx * len(files_params) + inner_idx) % len(colors)
        style_idx = (idx * len(files_params) + inner_idx) % len(line_styles)
        plt.plot(record['training_history']['losses'], 
                label=label,
                color=colors[color_idx],
                linestyle=line_styles[style_idx],
                linewidth=line_width)

plt.title('Training Loss Comparison', fontsize=14)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Accuracy curve
plt.subplot(1, 2, 2)
for idx, (local_epochs, files_params) in enumerate(grouped_by_epochs.items()):
    for inner_idx, (pth_file, params) in enumerate(files_params):
        # Load record
        record = torch.load(pth_file)
        
        # Create a label with relevant parameters
        label = f'E={local_epochs}, Mode={params["mode"]}'
        if params["train_mode"] != '':
            label += f'_{params["train_mode"]}'
        
        # Plot accuracy curve with color-blind friendly colors and different line styles
        color_idx = (idx * len(files_params) + inner_idx) % len(colors)
        style_idx = (idx * len(files_params) + inner_idx) % len(line_styles)
        plt.plot(record['training_history']['accuracies'], 
                label=label,
                color=colors[color_idx],
                linestyle=line_styles[style_idx],
                linewidth=line_width)

plt.title('Training Accuracy Comparison', fontsize=14)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Adjust layout to prevent legend cutoff
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'fedavg_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()