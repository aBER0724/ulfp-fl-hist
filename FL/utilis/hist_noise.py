import numpy as np
import math

def calculate_sse(arr):
    """Calculates the Sum of Squared Errors for a single bin (array segment)."""
    if len(arr) == 0:
        return 0
    mean = np.mean(arr)
    return np.sum((arr - mean)**2)

def find_optimal_histogram_dp(histogram_data, max_k):
    """
    Finds the optimal histogram structures for all k up to max_k using
    dynamic programming to minimize Sum of Squared Errors (SSE).
    """
    n = len(histogram_data)
    k = max_k
    dp = np.full((n + 1, k + 1), np.inf)  # DP table
    ptr = np.zeros((n + 1, k + 1), dtype=int)  # Point table
    dp[0][0] = 0

    # Precompute SSE for all possible single bins to speed up DP
    sse_cache = np.full((n, n), np.inf)
    for i in range(n):
        current_sum = 0
        current_sum_sq = 0
        for j in range(i, n):
            val = histogram_data[j]
            current_sum += val
            current_sum_sq += val**2
            count = j - i + 1
            mean = current_sum / count
            # SSE = sum(x^2) - n * mean^2
            sse = current_sum_sq - count * (mean**2) if count > 0 else 0
            sse_cache[i][j] = sse


    for j in range(1, k + 1): # Number of bins
        for i in range(j, n + 1): # End index (exclusive) of data considered
            min_sse_for_i_j = np.inf
            best_l_for_i_j = -1
            for l in range(j - 1, i): # Start index (inclusive) of the last bin
                # Cost of the last bin (data from index l to i-1)
                last_bin_sse = sse_cache[l][i-1] if l < i else 0

                # Ensure dp[l][j-1] is finite before adding
                if np.isfinite(dp[l][j - 1]):
                    current_sse = dp[l][j - 1] + last_bin_sse
                    if current_sse < min_sse_for_i_j:
                        min_sse_for_i_j = current_sse
                        best_l_for_i_j = l

            dp[i][j] = min_sse_for_i_j
            ptr[i][j] = best_l_for_i_j # Store the start index of the last bin

    return dp, ptr

def reconstruct_boundaries(ptr_table, n, k):
    """Reconstructs the optimal boundaries from the backpointer table."""
    boundaries = []
    current_k = k
    current_idx = n
    while current_k > 0 and current_idx > 0:
        start_index = ptr_table[current_idx][current_k]
        if start_index < 0 or start_index >= current_idx:
             # Handle invalid pointer, maybe raise error or use fallback
             print(f"Warning: Invalid pointer found for k={current_k}, idx={current_idx}. Pointer={start_index}")
             # Fallback: Add the current_idx and break or try previous k
             boundaries.append(current_idx) # Add boundary heuristically
             break # Or implement more sophisticated recovery


        boundaries.append(start_index)
        current_idx = start_index
        current_k -= 1

    final_boundaries = sorted(list(set(boundaries)))
    if 0 in final_boundaries:
         final_boundaries.remove(0)
    if n not in final_boundaries:
        final_boundaries.append(n)

    if len(final_boundaries) != k:
         print(f"Warning: Reconstructed boundaries count {len(final_boundaries)} != k={k}. Boundaries: {final_boundaries}")

    return final_boundaries

def histogram_noise_first_dp(user_histogram, epsilon):
    """
    Implements the NoiseFirst algorithm with automatic k selection
    """
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive.")

    n = len(user_histogram)
    if n == 0:
        return np.array([])

    noisy_histogram = np.zeros_like(user_histogram, dtype=float)

    # Add Laplace noise to each bin count
    scale = 1.0 / epsilon
    noise = np.random.laplace(0, scale, n)
    noisy_histogram = np.maximum(0, np.round(user_histogram + noise)).astype(int)

    # Run DP to find min SSE for all possible k (1 to n) on noisy data
    # max_k is n because we need to evaluate all possibilities
    dp_table, ptr_table = find_optimal_histogram_dp(noisy_histogram, max_k=n)

    # Select optimal k* based on Eq. 4 from Xu et al.
    best_k = -1
    min_objective_value = np.inf
    epsilon_sq = epsilon**2

    for k_candidate in range(1, n + 1):
        # SSE(H_k*, D_hat) is the min SSE for k bins on noisy data
        sse_noisy = dp_table[n][k_candidate]

        if np.isfinite(sse_noisy): # Only consider valid k values from DP
            # Objective: Minimize SSE_noisy - (2n - 4k) / epsilon^2
            # Equivalent to minimizing SSE_noisy * eps^2 - 2n + 4k
            # Using the form from Eq. 4 directly:
            objective = sse_noisy - (2.0 * n - 4.0 * k_candidate) / epsilon_sq

            if objective < min_objective_value:
                min_objective_value = objective
                best_k = k_candidate

    if best_k == -1:
        # Fallback if no k is found (e.g., if all SSEs were inf)
        print("Warning: Could not find optimal k automatically. Falling back to k=1.")
        best_k = 1 # Or perhaps n, or raise an error

    print(f"Automatically selected k = {best_k}")

    # Reconstruct boundaries for the chosen k*
    boundaries = reconstruct_boundaries(ptr_table, n, best_k)

    # Construct the output histogram using noisy means for k* bins
    output_histogram = np.zeros_like(noisy_histogram)
    start_idx = 0
    for end_idx in boundaries:
        # Ensure start_idx is less than end_idx and both are within bounds
        if start_idx < end_idx and start_idx < n and end_idx <= n:
            bin_data = noisy_histogram[start_idx:end_idx]
            if len(bin_data) > 0: # Avoid division by zero for empty bins
                 noisy_mean = np.mean(bin_data)
                 output_histogram[start_idx:end_idx] = noisy_mean
            # else: handle empty bin case if necessary, e.g., fill with 0 or NaN
        elif start_idx >= end_idx:
             print(f"Warning: Invalid boundary segment found: start={start_idx}, end={end_idx}. Skipping.")
        # else: boundary indices out of range, indicates error in reconstruction

        start_idx = end_idx
        if start_idx >= n: # Break if we've covered the whole histogram
             break

    # Post-processing: Round and ensure non-negativity
    final_output = np.maximum(0, np.round(output_histogram)).astype(int)

    return final_output, best_k

def histogram_structure_first_dp(user_histogram, epsilon, k, F):
	"""
	Implements the Mean-StructureFirst algorithm for differentially private histograms.
	"""
	n = len(user_histogram)

	# Split epsilon (simple 50/50 split for demonstration)
	epsilon1 = epsilon / 2.0  # For structure perturbation
	epsilon2 = epsilon / 2.0  # For noise addition to means

	# Find optimal k-bin structure on original data
	# We need the full DP table for error calculation during perturbation
	dp_table, ptr_table = find_optimal_histogram_dp(
		user_histogram, k)
	# Precompute SSE cache on original data if not already done by dp function
	sse_cache_orig = np.full((n, n), np.inf)
	for i in range(n):
		current_sum = 0
		current_sum_sq = 0
		for j in range(i, n):
			val = user_histogram[j]
			current_sum += val
			current_sum_sq += val**2
			count = j - i + 1
			mean = current_sum / count
			sse = current_sum_sq - count * (mean**2) if count > 0 else 0
			sse_cache_orig[i][j] = sse

	# Perturb Boundaries using Exponential Mechanism
	perturbed_boundaries = [0] * k  # Store start indices initially
	perturbed_boundaries[-1] = n  # The end of the last bin is fixed at n

	# Sensitivity for SSE change (as per Lemma 3 in Xu et al. 2013)
	# Note: This sensitivity depends on the definition (neighboring DB adds/removes 1 count vs changes 1 count)
	# Using the definition from Lemma 3: |Error' - Error| <= 2F + 1
	structure_sensitivity = 2.0 * F + 1.0
	if structure_sensitivity <= 0:
		structure_sensitivity = 1.0 # Avoid division by zero/negative scale

	# Iterate backwards to determine boundaries
	current_end_idx = n
	for j in range(k - 1, 0, -1):  # Determine boundary between bin j-1 and j
		possible_starts_last_bin = list(range(j, current_end_idx))  # Possible start indices for bin j
		if not possible_starts_last_bin:  # Cannot form j bins ending at current_end_idx
			print(
				f"Warning: Could not find valid split for bin {j} ending at {current_end_idx}")

			perturbed_boundaries[j-1] = perturbed_boundaries[j] if j < k-1 else j-1  # Heuristic fallback
			current_end_idx = perturbed_boundaries[j-1]
			continue

		costs = []
		for q in possible_starts_last_bin:  # q is the potential start index of the last bin (bin j)
			# Cost = SSE(optimal j-1 bins ending at q-1) + SSE(bin from q to current_end_idx-1)
			sse_last_bin = sse_cache_orig[q][current_end_idx- \
				1] if q < current_end_idx else 0
			# Ensure dp_table index is valid
			sse_prev_bins = dp_table[q][j] if q >= j else np.inf
			cost = sse_prev_bins + sse_last_bin
			costs.append(cost)

		costs = np.array(costs)
		# Probabilities based on exponential mechanism
		# Scale for exponent sensitivity / (2 * budget_fraction)
		# Budget fraction per boundary decision is epsilon1 / (k-1)
		prob_scale = structure_sensitivity / \
			(2.0 * (epsilon1 / (k - 1))) if k > 1 else structure_sensitivity
		if prob_scale <= 1e-9:
			prob_scale = 1e-9 # Avoid division by zero or extremely large exponents

		# Subtract max cost for numerical stability before exponentiation
		max_cost = np.max(costs) if len(costs) > 0 else 0
		log_probs = - (costs - max_cost) / prob_scale  # Negative cost, higher prob for lower cost

		# Handle potential inf values in costs before exponentiation
		log_probs[np.isinf(costs)] = -np.inf

		probs = np.exp(log_probs)

		# Normalize probabilities
		sum_probs = np.sum(probs)
		if sum_probs > 0:
			probabilities = probs / sum_probs
		else:
			# If all probabilities are zero (e.g., due to inf costs or numerical issues), assign uniform probability
			print("Warning: Uniform probability fallback in exponential mechanism.")
			probabilities = np.ones_like(
				probs) / len(probs) if len(probs) > 0 else []

		if len(possible_starts_last_bin) > 0 and len(probabilities) == len(possible_starts_last_bin):
			# Sample the start index for the last bin (bin j)
			chosen_start_idx = np.random.choice(
				possible_starts_last_bin, p=probabilities)
			perturbed_boundaries[j-1] = chosen_start_idx  # This is the start index of bin j
			current_end_idx = chosen_start_idx  # Update the end for the next iteration (j-1)
		else:
			# Handle empty possible_starts or probability mismatch (should not happen with proper checks)
			print(f"Error: Boundary perturbation failed for bin {j}.")
			# Fallback or raise error - using previous boundary as fallback
			perturbed_boundaries[j-1] = perturbed_boundaries[j] if j < k-1 else j-1  # Heuristic fallback
			current_end_idx = perturbed_boundaries[j-1]

	# Convert start indices to exclusive end indices: [b1, b2, ..., n]
	perturbed_end_boundaries = sorted(perturbed_boundaries)[1:]  # Get end indices b1 to b_{k-1}
	perturbed_end_boundaries.append(n)  # Add the final end index

	# Calculate means from original data and add Laplace noise
	output_histogram = np.zeros_like(user_histogram, dtype=float)
	start_idx = 0
	for end_idx in perturbed_end_boundaries:
		if start_idx < end_idx:  # Ensure the slice is not empty
			bin_data_orig = user_histogram[start_idx:end_idx]
			mean_orig = np.mean(bin_data_orig)
			bin_width = end_idx - start_idx

			# Laplace noise scale depends on bin width for Mean-StructureFirst
			noise_scale = (bin_width * epsilon2)
			if noise_scale <= 1e-9:  # Avoid division by zero or very small scale
				# Handle case of zero/small scale - perhaps add no noise or a default minimum noise
				laplace_noise = 0
				print(
					f"Warning: Skipping noise for bin [{start_idx}, {end_idx}) due to zero/small scale.")
			else:
				scale_param = 1.0 / noise_scale
				laplace_noise = np.random.laplace(0, scale_param)

			noisy_mean = mean_orig + laplace_noise
			output_histogram[start_idx:end_idx] = noisy_mean
		start_idx = end_idx

	# Post-processing: Round and ensure non-negativity
	final_output = np.maximum(0, np.round(output_histogram)).astype(int)

	return final_output

def histogram_dp(user_histogram, sigma):
	"""
	Add noise to the user's histogram.
	"""
	# Using sigma directly instead of epsilon
	noise = np.random.normal(0, sigma, size=len(user_histogram))

	# Use np.maximum to ensure values are non-negative
	return np.maximum(0, np.round(user_histogram + noise)).astype(int)
