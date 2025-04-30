from FL.utilis.silo import Silo
from FL.utilis.server import Server
from FL.utilis.dataset import Dataset
from model.CNN import CNNMnist
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(message)s', datefmt='%Y-%m-%d %H:%M')
logger = logging.getLogger(__name__)

def compare_noise_methods(histogram, epsilon=1.0, sigma=5.0, k=10, F=100.0, output_dir='./noise_method_comparison'):
	"""
	Compares different noise addition methods from hist_noise.py on the same histogram.
	
	Args:
		histogram: Original histogram data to add noise to 
		epsilon: Privacy budget for DP methods
		sigma: Noise parameter for Gaussian noise
		k: Number of bins for structure_first_dp
		F: Sensitivity parameter for structure_first_dp
		output_dir: Directory to save visualization outputs
	
	Returns:
		String indicating where visualizations were saved
	"""
	from FL.utilis.hist_noise import histogram_dp, histogram_structure_first_dp, histogram_noise_first_dp
	
	# Create output directory if it doesn't exist
	os.makedirs(output_dir, exist_ok=True)
	
	# Apply different noise methods
	dp_hist = histogram_dp(histogram, sigma)
	sf_dp_hist = histogram_structure_first_dp(histogram, epsilon, k, F)
	nf_dp_hist = histogram_noise_first_dp(histogram, epsilon)
	
	# Log results
	logger.info(f"Original histogram: {histogram}")
	logger.info(f"Gaussian DP (sigma={sigma}): {dp_hist}")
	logger.info(f"Structure-First DP (epsilon={epsilon}): {sf_dp_hist}")
	logger.info(f"Noise-First DP (epsilon={epsilon}): {nf_dp_hist}")
	
	# Create visualization for single histogram comparison
	plt.figure(figsize=(14, 8))
	
	# Extract indices and values
	indices = np.arange(len(histogram))
	width = 0.2  # narrower bars for side-by-side comparison
	
	# Draw bars side by side
	plt.bar(indices - 1.5*width, histogram, width, label='Original', color='blue', alpha=0.7)
	plt.bar(indices - 0.5*width, dp_hist, width, label=f'Gaussian DP (σ={sigma})', color='green', alpha=0.7)
	plt.bar(indices + 0.5*width, sf_dp_hist, width, label=f'Structure-First DP (ε={epsilon})', color='red', alpha=0.7)
	plt.bar(indices + 1.5*width, nf_dp_hist, width, label=f'Noise-First DP (ε={epsilon})', color='purple', alpha=0.7)
	
	# Add labels and title
	plt.xlabel('Bin Index')
	plt.ylabel('Count')
	plt.title('Comparison of Different Noise Methods')
	plt.xticks(indices)
	plt.legend()
	plt.grid(axis='y', linestyle='--', alpha=0.7)
	
	# Save image
	plt.savefig(f'{output_dir}/noise_methods_comparison.png', dpi=300)
	plt.close()
	
	# Create a detailed visualization with multiple subplots for better analysis
	plt.figure(figsize=(20, 15))
	
	# Original vs each method individually
	plt.subplot(2, 2, 1)
	plt.bar(indices, histogram, alpha=0.7, label='Original', color='blue')
	plt.bar(indices, dp_hist, alpha=0.5, label=f'Gaussian DP (σ={sigma})', color='green')
	plt.title('Original vs Gaussian DP')
	plt.xlabel('Bin Index')
	plt.ylabel('Count')
	plt.legend()
	plt.grid(axis='y', linestyle='--', alpha=0.7)
	
	plt.subplot(2, 2, 2)
	plt.bar(indices, histogram, alpha=0.7, label='Original', color='blue')
	plt.bar(indices, sf_dp_hist, alpha=0.5, label=f'Structure-First DP (ε={epsilon})', color='red')
	plt.title('Original vs Structure-First DP')
	plt.xlabel('Bin Index')
	plt.ylabel('Count')
	plt.legend()
	plt.grid(axis='y', linestyle='--', alpha=0.7)
	
	plt.subplot(2, 2, 3)
	plt.bar(indices, histogram, alpha=0.7, label='Original', color='blue')
	plt.bar(indices, nf_dp_hist, alpha=0.5, label=f'Noise-First DP (ε={epsilon})', color='purple')
	plt.title('Original vs Noise-First DP')
	plt.xlabel('Bin Index')
	plt.ylabel('Count')
	plt.legend()
	plt.grid(axis='y', linestyle='--', alpha=0.7)
	
	# Difference plots
	plt.subplot(2, 2, 4)
	diff_dp = np.array(dp_hist) - np.array(histogram)
	diff_sf = np.array(sf_dp_hist) - np.array(histogram)
	diff_nf = np.array(nf_dp_hist) - np.array(histogram)
	
	plt.bar(indices - width, diff_dp, width, label='Gaussian DP Difference', color='green', alpha=0.7)
	plt.bar(indices, diff_sf, width, label='Structure-First DP Difference', color='red', alpha=0.7)
	plt.bar(indices + width, diff_nf, width, label='Noise-First DP Difference', color='purple', alpha=0.7)
	plt.title('Difference from Original Histogram')
	plt.xlabel('Bin Index')
	plt.ylabel('Difference')
	plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
	plt.legend()
	plt.grid(axis='y', linestyle='--', alpha=0.7)
	
	plt.suptitle('Detailed Comparison of Noise Methods', fontsize=16)
	plt.tight_layout(pad=3.0)
	plt.savefig(f'{output_dir}/noise_methods_detailed_comparison.png', dpi=300)
	plt.close()
	
	# Compute and display statistical metrics for each method
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
	
	# Mean absolute error
	mae_dp = np.mean(np.abs(diff_dp))
	mae_sf = np.mean(np.abs(diff_sf))
	mae_nf = np.mean(np.abs(diff_nf))
	
	# Mean squared error
	mse_dp = np.mean(np.square(diff_dp))
	mse_sf = np.mean(np.square(diff_sf))
	mse_nf = np.mean(np.square(diff_nf))
	
	methods = ['Gaussian DP', 'Structure-First DP', 'Noise-First DP']
	mae_values = [mae_dp, mae_sf, mae_nf]
	mse_values = [mse_dp, mse_sf, mse_nf]
	
	ax1.bar(methods, mae_values, color=['green', 'red', 'purple'], alpha=0.7)
	ax1.set_title('Mean Absolute Error')
	ax1.set_ylabel('MAE')
	ax1.grid(axis='y', linestyle='--', alpha=0.7)
	
	ax2.bar(methods, mse_values, color=['green', 'red', 'purple'], alpha=0.7)
	ax2.set_title('Mean Squared Error')
	ax2.set_ylabel('MSE')
	ax2.grid(axis='y', linestyle='--', alpha=0.7)
	
	for i, v in enumerate(mae_values):
		ax1.text(i, v + 0.1, f'{v:.2f}', ha='center')
	
	for i, v in enumerate(mse_values):
		ax2.text(i, v + 0.1, f'{v:.2f}', ha='center')
	
	plt.suptitle('Error Metrics Comparison', fontsize=16)
	plt.tight_layout(pad=3.0)
	plt.savefig(f'{output_dir}/noise_methods_error_metrics.png', dpi=300)
	plt.close()
	
	return f"Noise method comparisons saved to {output_dir}"

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
	q_for_batch_size = 0.1
	is_histogram_secure = "dp"
	delta = 1e-5
	sigma = 5.0
	epsilon = 5.0
	max_grad_norm = 1.2
	F = 100.0
	model = CNNMnist()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	dataset = Dataset(
		dataset_name=dataset_name,
		n_silos=n_silos,
		n_users=n_users,
		seed=seed,
		mode=distribute_mode,
		max_samples=max_samples
	)

	server = Server(dataset, model, learning_rate, is_secure)
	silos = [Silo(i, server, dataset, model, learning_rate, q_for_batch_size, local_epoch, is_secure, False, is_histogram_secure, delta, sigma, max_grad_norm) for i in range(n_silos)]
	
	sample_histogram = silos[0].user_histogram
	compare_noise_methods(
		histogram=sample_histogram,
		epsilon=epsilon,
		sigma=sigma,
		k=int(len(sample_histogram)/10),
		F=F,
		output_dir="FL/test/test_output/noise_method_comparison"
	)
