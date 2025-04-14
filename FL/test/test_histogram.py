from FL.utilis.silo import Silo
from FL.utilis.server import Server
from FL.utilis.dataset import Dataset
from model.CNN import CNNMnist
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_histograms(silos, output_dir='./test_output'):
	"""
	Visualize user data histograms for each silo, comparing with and without noise.
	
	Args:
		silos: List of Silo objects
		output_dir: Directory to save visualization outputs
	"""
	# Create output directory if it doesn't exist
	os.makedirs(output_dir, exist_ok=True)
	
	# Process each silo
	for i, silo in enumerate(silos):
		no_noise_hist = silo.get_user_histogram(is_secure=False)
		with_noise_hist = silo.get_user_histogram(is_secure=True)
		
		print(f"Silo {i} - Without Noise", no_noise_hist)
		print(f"Silo {i} - With Noise", with_noise_hist)
		
		# Create comparison visualization
		plt.figure(figsize=(12, 6))
		
		# Extract user IDs and data counts
		user_ids = list(range(len(no_noise_hist)))
		no_noise_values = no_noise_hist
		with_noise_values = with_noise_hist
		
		# Set bar positions (centered, not offset)
		x = np.arange(len(user_ids))
		width = 0.5  # wider bars
		
		# Draw overlapping bars with transparency
		bars1 = plt.bar(x, no_noise_values, width, label='Without Noise', alpha=0.7, color='#2B75BC')  # blue
		bars2 = plt.bar(x, with_noise_values, width, label='With Noise', alpha=0.6, color='#FF8C00')  # dark orange
		
		# Add data labels on top of each bar
		for bar in bars1:
			height = bar.get_height()
			plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
					f'{int(height)}', ha='center', va='bottom', fontsize=8, color='#2B75BC')
		
		for bar in bars2:
			height = bar.get_height()
			plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
					f'{int(height)}', ha='center', va='bottom', fontsize=8, color='#FF8C00')
		
		# Add labels and title
		plt.xlabel('User ID')
		plt.ylabel('Data Count')
		plt.title(f'Silo {i} User Data Distribution Histogram Comparison')
		plt.xticks(x, user_ids)
		plt.legend()
		
		# Save individual silo image
		plt.savefig(f'{output_dir}/silo_{i}_histogram_comparison.png')
		plt.close()
	
	# Create summary visualization with all silos
	n_silos = len(silos)
	# Change layout to have fewer columns and more rows to make each subplot wider
	cols = 2  # Only 2 columns to make each subplot wider
	rows = (n_silos + cols - 1) // cols  # Calculate rows needed
	
	# Increase figure width and adjust height based on number of rows
	plt.figure(figsize=(20, 8 * rows))  # Wider figure with 20 inches width
	
	for i, silo in enumerate(silos):
		plt.subplot(rows, cols, i+1)
		
		no_noise_hist = silo.get_user_histogram(is_secure=False)
		with_noise_hist = silo.get_user_histogram(is_secure=True)
		
		user_ids = list(range(len(no_noise_hist)))
		no_noise_values = no_noise_hist
		with_noise_values = with_noise_hist
		
		x = np.arange(len(user_ids))
		width = 0.5  # wider bars
		
		bars1 = plt.bar(x, no_noise_values, width, label='Without Noise', alpha=0.7, color='#2B75BC')  # blue
		bars2 = plt.bar(x, with_noise_values, width, label='With Noise', alpha=0.6, color='#FF8C00')  # dark orange
		
		# Add data labels on top of each bar (smaller font for summary plot)
		for bar in bars1:
			height = bar.get_height()
			plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
					f'{int(height)}', ha='center', va='bottom', fontsize=6, color='#2B75BC')
		
		for bar in bars2:
			height = bar.get_height()
			plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
					f'{int(height)}', ha='center', va='bottom', fontsize=6, color='#FF8C00')
		
		plt.xlabel('User ID')
		plt.ylabel('Data Count')
		plt.title(f'Silo {i}')
		plt.xticks(x, user_ids)
		if i == 0:
			plt.legend()
	
	plt.suptitle('All Silos Histogram Comparison', fontsize=16)
	plt.tight_layout(pad=3.0)  # Add more padding between subplots
	plt.savefig(f'{output_dir}/all_silos_histogram_comparison.png', dpi=300)
	plt.close()
	
	return f"Histogram visualizations saved to {output_dir}"

if __name__ == "__main__":
	dataset_name = "mnist"
	max_samples = 1000
	batch_size = 256
	iters = 20
	local_epoch = 5
	distribute_mode = "zipf"
	is_secure = "ndp"
	learning_rate = 0.1
	n_silos = 5
	n_users = 100
	seed = 1
	alpha = 0.05
	q_for_batch_size = 0.1
	is_histogram_secure = True
	delta = 1e-5
	sigma = 0.5
	max_grad_norm = 1.2
	model = CNNMnist()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	dataset = Dataset(dataset_name, n_silos, n_users, alpha, seed, distribute_mode, max_samples=max_samples)

	server = Server(dataset, model, learning_rate, is_secure)
	silos = [Silo(i, server, dataset, model, learning_rate, q_for_batch_size, local_epoch, is_secure, False, is_histogram_secure, delta, sigma, max_grad_norm) for i in range(n_silos)]
    
	# Call the visualization function
	visualize_histograms(silos)
