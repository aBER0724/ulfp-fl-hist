import numpy as np
import math

# --- histogram_dp (高斯噪声) ---
def histogram_dp(user_histogram, sigma):
    """
    Adds Gaussian noise to a user histogram for differential privacy.

    Args:
        user_histogram (np.ndarray): The original histogram data (counts per bin).
        sigma (float): The standard deviation of the Gaussian noise to be added.
                       This is related to the sensitivity of the query and the privacy budget.

    Returns:
        np.ndarray: The noisy histogram, with counts rounded to the nearest integer
                    and floored at 0.
    """
    noise = np.random.normal(0, sigma, size=len(user_histogram))
    return np.maximum(0, np.round(user_histogram + noise)).astype(int)

# --- 辅助错误计算函数 ---
def calculate_sse_mean_repr(arr):
    """
    Calculates the Sum of Squared Errors (SSE) for a given array segment,
    using the mean of the segment as the representative value for all elements.

    This is often used in histogram optimization problems to quantify the error
    introduced by representing a range of bins with a single value (the mean).

    Args:
        arr (np.ndarray): A 1D NumPy array representing a segment of a histogram.

    Returns:
        float: The Sum of Squared Errors (SSE) for the segment. Returns 0 if the array is empty.
    """
    if len(arr) == 0:
        return 0
    representative_value = np.mean(arr)
    return np.sum((arr - representative_value)**2)

def calculate_sae_median_repr(arr):
    """
    Calculates the Sum of Absolute Errors (SAE) for a given array segment,
    using the median of the segment as the representative value for all elements.

    Similar to SSE, SAE is used in histogram optimization, particularly when
    median-based representation is preferred (e.g., for robustness to outliers).

    Args:
        arr (np.ndarray): A 1D NumPy array representing a segment of a histogram.

    Returns:
        float: The Sum of Absolute Errors (SAE) for the segment. Returns 0 if the array is empty.
    """
    if len(arr) == 0:
        return 0
    representative_value = np.median(arr)
    return np.sum(np.abs(arr - representative_value))

# --- 通用动态规划函数 ---
def _find_optimal_histogram_dp_generic(histogram_data, max_k, error_cache):
    """
    Finds the optimal histogram structure (minimizing total error) using dynamic programming.
    This is a generic version that relies on a precomputed `error_cache`.

    This function is an internal helper for `find_optimal_sse_histogram_dp` and
    `find_optimal_sae_histogram_dp`.

    Args:
        histogram_data (np.ndarray): The input histogram data (1D array of counts).
        max_k (int): The maximum number of bins (partitions) allowed in the output histogram.
        error_cache (np.ndarray): A precomputed n x n matrix where `error_cache[i][j]`
                                  stores the error of representing `histogram_data[i...j]`
                                  as a single segment. The type of error (SSE, SAE)
                                  depends on how this cache was populated.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - dp_table: A (n+1) x (max_k+1) DP table where `dp_table[i][j]` stores the
                        minimum total error for partitioning the first `i` data points
                        of `histogram_data` into `j` bins.
            - ptr_table: A (n+1) x (max_k+1) pointer table used to reconstruct the
                         optimal bin boundaries. `ptr_table[i][j]` stores the starting
                         index of the last bin when partitioning the first `i` points
                         into `j` bins.
    """
    n = len(histogram_data)
    k_val = max_k # 使用 k_val 避免与外层 k 冲突
    dp_table = np.full((n + 1, k_val + 1), np.inf)  # DP 表
    ptr_table = np.zeros((n + 1, k_val + 1), dtype=int)  # 指针表
    dp_table[0][0] = 0

    for j_bins in range(1, k_val + 1):  # 区间数量
        for i_data_end_idx in range(j_bins, n + 1):  # 考虑的数据结束索引 (不包含)
            min_error_for_i_j = np.inf
            best_l_for_i_j = -1
            # l_last_bin_start_idx 是最后一个区间的起始索引 (包含)
            # 最后一个区间是 histogram_data[l_last_bin_start_idx ... i_data_end_idx-1]
            for l_last_bin_start_idx in range(j_bins - 1, i_data_end_idx):
                # 最后一个区间的误差 (数据从索引 l_last_bin_start_idx 到 i_data_end_idx-1)
                # error_cache 的索引对应原始数据的索引
                last_bin_error = error_cache[l_last_bin_start_idx][i_data_end_idx-1] if l_last_bin_start_idx < i_data_end_idx else 0

                if np.isfinite(dp_table[l_last_bin_start_idx][j_bins - 1]):
                    current_total_error = dp_table[l_last_bin_start_idx][j_bins - 1] + last_bin_error
                    if current_total_error < min_error_for_i_j:
                        min_error_for_i_j = current_total_error
                        best_l_for_i_j = l_last_bin_start_idx
            
            dp_table[i_data_end_idx][j_bins] = min_error_for_i_j
            ptr_table[i_data_end_idx][j_bins] = best_l_for_i_j # 存储最后一个区间的起始索引

    return dp_table, ptr_table

def find_optimal_sse_histogram_dp(histogram_data, max_k):
    """
    Finds the optimal histogram structure that minimizes the total Sum of Squared Errors (SSE),
    using dynamic programming. Each bin is represented by the mean of its original values.

    Args:
        histogram_data (np.ndarray): The input histogram data (1D array of counts).
        max_k (int): The maximum number of bins allowed in the output histogram.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - dp_table: The DP table storing minimum SSEs.
            - ptr_table: The pointer table for reconstructing boundaries.
            (See `_find_optimal_histogram_dp_generic` for details on tables).
    """
    n = len(histogram_data)
    sse_cache = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i, n):
            sub_array = histogram_data[i:j+1]
            sse_cache[i][j] = calculate_sse_mean_repr(sub_array)
    return _find_optimal_histogram_dp_generic(histogram_data, max_k, sse_cache)

def find_optimal_sae_histogram_dp(histogram_data, max_k):
    """
    Finds the optimal histogram structure that minimizes the total Sum of Absolute Errors (SAE),
    using dynamic programming. Each bin is represented by the median of its original values.

    Args:
        histogram_data (np.ndarray): The input histogram data (1D array of counts).
        max_k (int): The maximum number of bins allowed in the output histogram.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - dp_table: The DP table storing minimum SAEs.
            - ptr_table: The pointer table for reconstructing boundaries.
            (See `_find_optimal_histogram_dp_generic` for details on tables).
    """
    n = len(histogram_data)
    sae_cache = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i, n):
            sub_array = histogram_data[i:j+1]
            sae_cache[i][j] = calculate_sae_median_repr(sub_array)
    return _find_optimal_histogram_dp_generic(histogram_data, max_k, sae_cache)


# --- 重建边界 ---
def reconstruct_boundaries(ptr_table, n, k):
    """
    Reconstructs the optimal bin boundaries from the pointer table generated by
    the dynamic programming histogram optimization.

    Args:
        ptr_table (np.ndarray): The pointer table from `_find_optimal_histogram_dp_generic`.
                                `ptr_table[i][j]` stores the starting index of the
                                last bin when partitioning the first `i` points into `j` bins.
        n (int): The total number of data points in the original histogram.
        k (int): The number of bins for which to reconstruct boundaries.

    Returns:
        list[int]: A list of integers representing the end points (exclusive) of the
                   reconstructed bins. For `k` bins, this list will contain `k` values
                   `[b_1, b_2, ..., b_k]`, where the bins are `[0, b_1)`, `[b_1, b_2)`, ...,
                   `[b_{k-1}, b_k=n)`. Returns an empty list if k=0.
    """
    boundaries = []
    current_k = k
    current_idx = n
    while current_k > 0 and current_idx > 0:
        start_index = ptr_table[current_idx][current_k]
        if start_index < 0 or start_index > current_idx : # 修正: start_index 可以等于 current_idx 吗？ 通常是 <
             print(f"警告: 发现无效指针 k={current_k}, idx={current_idx}. 指针={start_index}")
             boundaries.append(current_idx) 
             break
        
        # 边界是区间的起始点，所以我们添加的是 start_index
        # 最后一个区间的起始点是 ptr_table[n][k]
        # 它的前一个区间的结束点（也是这个区间的起始点）
        boundaries.append(start_index) 
        current_idx = start_index
        current_k -= 1

    # 边界应该是排序的，并且包含0和n
    # ptr_table 返回的是上一个区间的结束点（即当前区间的起始点，但不包括0）
    # 例如，如果k=1, ptr[n][1] = 0 boundaries = [0]
    # 如果k=2, ptr[n][2] = b1, ptr[b1][1] = 0  boundaries = [b1, 0]
    
    final_boundaries = sorted(list(set(boundaries))) # 去重并排序
    
    # 确保边界从0开始，到n结束，用于定义区间
    # reconstruct_boundaries 的目的是返回分割点 
    # 如果 boundaries 是 [s_k, s_{k-1}, ..., s_1] 其中 s_1=0
    # 那么区间是 [s_1, s_2-1], [s_2, s_3-1], ..., [s_k, n-1]
    # 或者更常见的定义是返回区间的结束点 (exclusive)
    # [0, b1), [b1, b2), ..., [b_{k-1}, n)
    # 这里的 boundaries 存储的是每个区间的起始索引（除了第一个区间的0）
    
    # 我们需要返回 k-1 个分割点，加上最后的 n
    # 例如，如果最优结构是 [0, idx1), [idx1, idx2), [idx2, n)
    # reconstruct_boundaries 会从 ptr[n][k] 开始回溯
    # ptr[n][k] = idx2
    # ptr[idx2][k-1] = idx1
    # ptr[idx1][k-2] = 0 (假设k=3)
    # boundaries 将会是 [idx2, idx1, 0] (顺序可能不同，取决于添加方式)
    # 排序后是 [0, idx1, idx2]
    # 我们需要的是 [idx1, idx2, n] 作为结束边界点
    
    # 论文中的边界通常指区间的结束点（不包含）或者起始点（包含）
    # 如果 ptr_table[current_idx][current_k] 存储的是最后一个区间的起始索引 l
    # 那么这个区间是 [l, current_idx-1]
    # boundaries 列表收集了这些 l 
    # 排序后，例如 [0, l2, l3] for k=3 bins, where bins are [0,l2-1], [l2,l3-1], [l3,n-1]
    # 我们需要的是分割点：l2, l3, n
    
    processed_boundaries = []
    if 0 in final_boundaries: # 移除 DP 基准情况的 0
        final_boundaries.remove(0)
    
    processed_boundaries = sorted(list(set(final_boundaries))) # 确保唯一和有序
                                                     # 这些是 k-1 个内部边界点
    
    # 添加最终的边界 n
    if n not in processed_boundaries:
        processed_boundaries.append(n)

    if len(processed_boundaries) != k and k != 0: # 如果k=0, processed_boundaries 应该是空的
         # This warning might be too strict if k is small or n is small
         # print(f"警告: 重建的边界数量 {len(processed_boundaries)} != k={k}. 边界: {processed_boundaries}")
         pass


    return processed_boundaries # 返回的是区间的结束点（不包含起始的0）

# --- NoiseFirst 算法 ---
def Mean_NoiseFirst(user_histogram, epsilon):
    """
    Implements the Mean-NoiseFirst algorithm for differentially private histogram publication.
    This corresponds to Algorithms 1 and 2 in the referenced literature (e.g., Qardaji et al.).
    It automatically selects the optimal number of bins `k` by minimizing an objective function.

    The algorithm first adds Laplace noise to the original histogram, then finds the
    optimal k-bin histogram structure on this noisy data that minimizes SSE, and finally
    selects the `k` that minimizes `SSE_noisy - (2n - 4k) / epsilon^2`.

    Args:
        user_histogram (np.ndarray): The original non-private histogram counts.
        epsilon (float): The differential privacy budget (must be positive).

    Returns:
        tuple[np.ndarray, int]:
            - final_output (np.ndarray): The differentially private histogram with `best_k` bins,
                                         where each bin's value is the (noisy) mean of the
                                         original noisy counts in that bin. Counts are rounded
                                         and non-negative.
            - best_k (int): The automatically selected optimal number of bins.
    Raises:
        ValueError: If epsilon is not positive.
    """
    if epsilon <= 0:
        raise ValueError("Epsilon 必须为正 ")

    n = len(user_histogram)
    if n == 0:
        return np.array([]), 0

    # 步骤 1: 添加拉普拉斯噪声 (Algorithm 1, line 1)
    scale = 1.0 / epsilon
    noise = np.random.laplace(0, scale, n)
    noisy_histogram = np.maximum(0, np.round(user_histogram + noise)).astype(int)

    # 步骤 2: 在噪声数据上运行 DP 找到所有 k 的最小 SSE (Algorithm 1, line 2 preparation)
    dp_table, ptr_table = find_optimal_sse_histogram_dp(noisy_histogram, max_k=n)

    # 步骤 3: 根据论文 Eq. 4 (Algorithm 2) 选择最优 k*
    best_k = -1
    min_objective_value = np.inf
    epsilon_sq = epsilon**2

    for k_candidate in range(1, n + 1):
        sse_noisy = dp_table[n][k_candidate] # SSE(H_k*, D_hat)

        if np.isfinite(sse_noisy):
            objective = sse_noisy - (2.0 * n - 4.0 * k_candidate) / epsilon_sq
            if objective < min_objective_value:
                min_objective_value = objective
                best_k = k_candidate
    
    if best_k == -1: # 如果没有找到有效的k (例如所有SSE都是inf)
        print("警告: 无法自动找到最优k 回退到 k=1 ")
        best_k = 1 if n > 0 else 0


    # 为选定的 k* 重建边界
    # reconstruct_boundaries 返回的是区间的结束点列表 [b1, b2, ..., n]
    boundaries = reconstruct_boundaries(ptr_table, n, best_k)
    
    # 构建输出直方图 (Algorithm 1, line 3)
    output_histogram = np.zeros_like(noisy_histogram, dtype=float)
    start_idx = 0
    for end_idx in boundaries:
        if start_idx < end_idx and start_idx < n and end_idx <= n:
            bin_data_noisy = noisy_histogram[start_idx:end_idx]
            if len(bin_data_noisy) > 0:
                representative_value = np.mean(bin_data_noisy)
                output_histogram[start_idx:end_idx] = representative_value
        elif start_idx >= end_idx and best_k > 0 : # 仅当k>0时，这是一个问题
             print(f"警告 (Mean_NoiseFirst): 无效边界段: start={start_idx}, end={end_idx} for k={best_k}")
        start_idx = end_idx
        if start_idx >= n:
            break
            
    final_output = np.maximum(0, np.round(output_histogram)).astype(int)
    return final_output, best_k

def Median_NoiseFirst(user_histogram, epsilon):
    """
    Implements the Median-NoiseFirst algorithm for differentially private histogram publication.
    This corresponds to Algorithm 5 and the k-selection strategy from Section 5.1
    in the referenced literature (e.g., Qardaji et al.).
    It automatically selects the optimal number of bins `k`.

    The algorithm first adds Laplace noise, then finds the optimal k-bin structure on
    this noisy data that minimizes SAE, and finally selects `k` that minimizes
    `SAE_noisy - 3(n - k) / epsilon`.

    Args:
        user_histogram (np.ndarray): The original non-private histogram counts.
        epsilon (float): The differential privacy budget (must be positive).

    Returns:
        tuple[np.ndarray, int]:
            - final_output (np.ndarray): The differentially private histogram with `best_k` bins,
                                         where each bin's value is the (noisy) median of the
                                         original noisy counts in that bin. Counts are rounded
                                         and non-negative.
            - best_k (int): The automatically selected optimal number of bins.
    Raises:
        ValueError: If epsilon is not positive.
    """
    if epsilon <= 0:
        raise ValueError("Epsilon 必须为正 ")

    n = len(user_histogram)
    if n == 0:
        return np.array([]), 0

    # 步骤 1: 添加拉普拉斯噪声 (Algorithm 5, line 1)
    scale = 1.0 / epsilon
    noise = np.random.laplace(0, scale, n)
    noisy_histogram = np.maximum(0, np.round(user_histogram + noise)).astype(int)

    # 步骤 2: 在噪声数据上运行 DP 找到所有 k 的最小 SAE (Algorithm 5, line 2 preparation)
    dp_table, ptr_table = find_optimal_sae_histogram_dp(noisy_histogram, max_k=n)

    # 步骤 3: 根据论文 Eq. 15 (Section 5.1) 选择最优 k*
    best_k = -1
    min_objective_value = np.inf

    for k_candidate in range(1, n + 1):
        sae_noisy = dp_table[n][k_candidate] # SAE(H_k*, D_hat)

        if np.isfinite(sae_noisy):
            # 目标: Minimize SAE_noisy - 3(n-k)/epsilon
            objective = sae_noisy - (3.0 * (n - k_candidate)) / epsilon
            if objective < min_objective_value:
                min_objective_value = objective
                best_k = k_candidate
    
    if best_k == -1:
        print("警告: 无法自动找到最优k 回退到 k=1 ")
        best_k = 1 if n > 0 else 0

    boundaries = reconstruct_boundaries(ptr_table, n, best_k)
    
    # 构建输出直方图 (Algorithm 5, line 3)
    output_histogram = np.zeros_like(noisy_histogram, dtype=float)
    start_idx = 0
    for end_idx in boundaries:
        if start_idx < end_idx and start_idx < n and end_idx <= n:
            bin_data_noisy = noisy_histogram[start_idx:end_idx]
            if len(bin_data_noisy) > 0:
                representative_value = np.median(bin_data_noisy)
                output_histogram[start_idx:end_idx] = representative_value
        elif start_idx >= end_idx and best_k > 0:
             print(f"警告 (Median_NoiseFirst): 无效边界段: start={start_idx}, end={end_idx} for k={best_k}")
        start_idx = end_idx
        if start_idx >= n:
            break
            
    final_output = np.maximum(0, np.round(output_histogram)).astype(int)
    return final_output, best_k

# --- StructureFirst 算法的辅助函数：边界扰动 ---
def _perturb_boundaries_exponential_mechanism(
    dp_table_orig,      # 在原始数据上计算的DP表 (Error(H*(D_i,j), D_i))
    error_cache_orig,   # 在原始数据上计算的各独立区间的错误缓存 (SSE或SAE)
    n,                  # 数据点数量
    k,                  # 期望的区间数量
    epsilon1,           # 用于结构扰动的隐私预算
    metric_sensitivity  # 对应错误度量的敏感度 (例如，对SSE是2F+1，对SAE是1)
):
    """
    Iteratively selects and perturbs histogram bin boundaries using the exponential mechanism.
    This is a core component of the StructureFirst algorithms.

    The function determines `k-1` internal split points `r_1, ..., r_{k-1}` from right to left.
    The quality score for choosing a split point `q` for `r_j` (given `r_{j+1}`) is based on
    minimizing `Error_DP(H*(D_q, j), D_q) + Error_Segment(D, q, r_{j+1}-1)`, where `Error_DP`
    is the error of the optimal `j`-bin histogram on `D[0...q-1]` and `Error_Segment` is the
    error of representing `D[q...r_{j+1}-1]` as one bin.

    Args:
        dp_table_orig (np.ndarray): The DP table computed on the original (non-noisy) data,
                                    where `dp_table_orig[i][j]` is the min error for
                                    partitioning the first `i` points into `j` bins.
        error_cache_orig (np.ndarray): Precomputed error for any segment `[i...j]` on original data.
        n (int): Total number of data points (length of the original histogram).
        k (int): The desired number of bins in the final histogram.
        epsilon1 (float): The privacy budget allocated for perturbing all `k-1` boundaries.
        metric_sensitivity (float): The sensitivity of the error metric used (e.g., `2F+1` for SSE
                                    with mean, `1` for SAE with median).

    Returns:
        list[int]: A list of `k` perturbed end boundaries `[r_1, r_2, ..., r_k=n]`.
                   These define `k` bins: `[0, r_1)`, `[r_1, r_2)`, ..., `[r_{k-1}, r_k=n)`.
                   Returns an empty list if k=0.

    Raises:
        ValueError: If k is negative.
    """
    if k < 0: # k 应该是非负的
        raise ValueError("区间数量 k 不能为负 ")
    if k == 0:
        return []
    
    perturbed_end_boundaries = [0] * k # 初始化列表以存储区间的结束点
                                     # perturbed_end_boundaries[i] 是第 (i+1) 个区间的结束点(exclusive)
                                     # 即 r_{i+1}

    # 对于 k=1 的情况，只有一个区间，其结束点是 n 没有内部边界需要选择 
    if k == 1:
        if n > 0:
            perturbed_end_boundaries[0] = n
            return perturbed_end_boundaries
        else: # n=0, k=1 (或 k=0 已处理)
            return [] # 或者 [0] 如果约定 k=1 且 n=0 意味着一个空区间在0处结束

    perturbed_end_boundaries[k - 1] = n  # 最后一个区间的结束点 r_k 固定为 n

    current_fixed_end_idx = n  # 这是 r_{j+1} 在循环中的角色
                             # (即当前正在考虑的、已固定的右侧大区间的结束点)

    # 从右到左，依次确定 k-1 个内部边界点 (r_{k-1}, r_{k-2}, ..., r_1)
    # j_boundary_idx 对应论文中的 j，表示正在确定第 j 个区间的右边界 r_j
    for j_boundary_idx in range(k - 1, 0, -1):
        # 我们正在确定 perturbed_end_boundaries[j_boundary_idx-1]，即 r_j
        # q_val 是 r_j 的候选值 
        # 前 j_boundary_idx 个区间覆盖 [0...q_val-1]
        # 第 (j_boundary_idx+1)-th 个区间是 [q_val...current_fixed_end_idx-1]
        
        min_q_val = j_boundary_idx  # r_j 必须至少是 j_boundary_idx (确保前j个区间至少各1个元素)
        max_q_val = current_fixed_end_idx - 1 # r_j 必须小于 r_{j+1} (即 current_fixed_end_idx)

        if min_q_val > max_q_val:
            # 这种情况理论上不应该频繁发生，除非k非常接近n或数据分布极端
            # 如果发生，意味着无法为当前j_boundary_idx找到有效的分割点q_val
            # 例如，如果 current_fixed_end_idx = j_boundary_idx，则 max_q_val < min_q_val
            print(f"警告 (_perturb_boundaries): 无法为边界索引 {j_boundary_idx-1} (r_{j_boundary_idx}) 在 ({min_q_val}, {max_q_val}] 找到有效分割点 "
                  f"r_{{j+1}}={current_fixed_end_idx}. 使用回退策略 ")
            # 回退策略：将此边界设置为紧邻下一个边界或一个合理的值
            # chosen_split_point = max_q_val if max_q_val >= min_q_val else min_q_val
            # 更安全的做法可能是让这个区间为空，或者与前一个合并，但这会改变k
            # 论文中没有明确指定此情况的回退，通常假设总能找到分割
            # 一个简单的回退是尽可能地向左推，但不小于min_q_val
            chosen_split_point = min_q_val 
            if j_boundary_idx < k -1 and perturbed_end_boundaries[j_boundary_idx] <= chosen_split_point: # 避免 r_j >= r_{j+1}
                chosen_split_point = perturbed_end_boundaries[j_boundary_idx] -1
            if chosen_split_point < j_boundary_idx: chosen_split_point = j_boundary_idx


            perturbed_end_boundaries[j_boundary_idx - 1] = chosen_split_point
            current_fixed_end_idx = chosen_split_point
            continue

        candidate_split_points_q = list(range(min_q_val, max_q_val + 1))
        costs = []

        for q_val in candidate_split_points_q:
            # E_metric(q, j, r_{j+1}) = Error_DP(H*(D_q, j), D_q) + Error_Segment(D, q, r_{j+1}-1)
            # dp_table_orig[q_val][j_boundary_idx] 是 Error_DP(H*(D_q, j), D_q)
            error_prev_bins = dp_table_orig[q_val][j_boundary_idx]
            
            # 第 (j_boundary_idx+1)-th 个区间的数据是 user_histogram[q_val ... current_fixed_end_idx-1]
            # error_cache_orig 的索引是相对于原始数据的
            error_current_segment = error_cache_orig[q_val][current_fixed_end_idx - 1] if q_val < current_fixed_end_idx else 0
            
            cost = error_prev_bins + error_current_segment
            costs.append(cost)

        costs = np.array(costs)
        
        # 指数机制的概率计算
        # 尺度参数: (2 * (迭代次数 k-1) * 敏感度) / epsilon1
        # 注意：这里的 (k-1) 是指总共进行 k-1 次边界选择 
        # 论文中分母的 (k-1) 是指总的边界扰动次数 
        # 如果每次迭代都用 epsilon1 / (k-1) 的预算，那么单次迭代的尺度是 (2 * sensitivity) / (epsilon1 / (k-1))
        # 这里 epsilon1 是已经分配给 *所有* 结构扰动的总预算 
        # 论文 Algorithm 3, line 7: exp{ -eps1 * E_SSE / (2(k-1)(2F+1)) }
        #  => 尺度 = (2 * (k-1) * (2F+1)) / eps1
        # 论文 Algorithm 6, line 7: exp{ -eps1 * E_SAE / (2(k-1)) }
        #  => 尺度 = (2 * (k-1) * 1) / eps1
        
        if k == 1: # 已在函数开始处处理，但作为防御
             prob_scale_exp = np.inf # 避免在k=1时除以0，虽然此循环不会执行
        elif epsilon1 <= 1e-9: # epsilon1 为0或极小
            prob_scale_exp = np.inf # 导致均匀选择（如果成本有限）
        else:
            prob_scale_exp = (2.0 * (k - 1) * metric_sensitivity) / epsilon1
        
        if prob_scale_exp <= 1e-9: # 避免尺度过小导致数值问题
             prob_scale_exp = 1e-9


        min_c = np.min(costs[np.isfinite(costs)]) if np.any(np.isfinite(costs)) else 0
        log_probs = -(costs - min_c) / prob_scale_exp
        log_probs[np.isinf(costs)] = -np.inf 

        log_probs = np.clip(log_probs, -700, 700) 
        probabilities = np.exp(log_probs)
        probabilities[~np.isfinite(probabilities)] = 0.0 
        
        sum_probs = np.sum(probabilities)
        if sum_probs > 0:
            probabilities /= sum_probs
        else: 
            # print(f"警告 (_perturb_boundaries): 指数机制中所有概率为零，使用均匀分布 j_boundary_idx={j_boundary_idx}")
            probabilities = np.ones_like(costs) / len(costs) if len(costs) > 0 else []

        if len(candidate_split_points_q) > 0 and len(probabilities) == len(candidate_split_points_q) and np.sum(probabilities) > 0:
            chosen_split_point = np.random.choice(candidate_split_points_q, p=probabilities)
        elif len(candidate_split_points_q) > 0: 
            # print(f"警告 (_perturb_boundaries): 概率计算故障或候选点不足，从候选点中均匀随机选择 j_boundary_idx={j_boundary_idx}")
            chosen_split_point = np.random.choice(candidate_split_points_q)
        else: 
            # print(f"警告 (_perturb_boundaries): 没有候选分割点 j_boundary_idx={j_boundary_idx}")
            # 这是一个严重错误，意味着 min_q_val > max_q_val 之前的检查可能不够
            chosen_split_point = current_fixed_end_idx - 1 # 紧急回退
            if chosen_split_point < j_boundary_idx : chosen_split_point = j_boundary_idx


        perturbed_end_boundaries[j_boundary_idx - 1] = chosen_split_point 
        current_fixed_end_idx = chosen_split_point
        
    return perturbed_end_boundaries

# --- StructureFirst 算法 ---
def Mean_StructureFirst(user_histogram, epsilon, k, F):
    """
    Implements the Mean-StructureFirst algorithm for differentially private histogram publication.
    This corresponds to Algorithm 3 in the referenced literature (e.g., Qardaji et al.).

    The algorithm first perturbs the bin boundaries using the exponential mechanism
    (allocating `epsilon/2` budget), then adds Laplace noise to the mean count of each
    resulting bin (allocating `epsilon/2` budget).

    Args:
        user_histogram (np.ndarray): The original non-private histogram counts.
        epsilon (float): The total differential privacy budget (must be positive).
        k (int): The desired number of bins in the output histogram (must be non-negative).
                 If k=0, an empty array is returned. If k > n, k is capped at n.
        F (float): An upper bound on the counts in the original `user_histogram`. This is used
                   to determine the sensitivity for boundary perturbation with SSE.

    Returns:
        np.ndarray: The differentially private histogram. Counts are rounded and non-negative.
                    Returns an empty array if k=0 or n=0.

    Raises:
        ValueError: If epsilon is not positive or k is negative.
    """
    if epsilon <= 0:
        raise ValueError("Epsilon 必须为正 ")
    if k < 0: # 允许 k=0 (返回空)
        raise ValueError("区间数量 k 不能为负 ")
    n = len(user_histogram)
    
    if k == 0:
        return np.array([])
    if n == 0: # 如果n=0, 那么k必须是0才能有意义
        return np.array([]) if k==0 else np.array([0]*0) # 或者根据约定返回

    if k > n :
        # print(f"警告 (Mean_StructureFirst): k ({k}) 大于数据点数量 ({n}) 将 k 设为 {n} ")
        k = n

    epsilon1 = epsilon / 2.0
    epsilon2 = epsilon / 2.0

    sse_cache_orig = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i, n):
            sse_cache_orig[i][j] = calculate_sse_mean_repr(user_histogram[i:j+1])
    
    # DP表需要计算到k，即使k=n
    dp_table_orig, _ = _find_optimal_histogram_dp_generic(user_histogram, k, sse_cache_orig)

    structure_sensitivity = 2.0 * F + 1.0
    if structure_sensitivity <= 0: 
        structure_sensitivity = 1.0 
    
    perturbed_end_boundaries = _perturb_boundaries_exponential_mechanism(
        dp_table_orig, sse_cache_orig, n, k, epsilon1, structure_sensitivity
    )

    output_histogram = np.zeros_like(user_histogram, dtype=float)
    start_idx = 0
    for end_idx in perturbed_end_boundaries:
        if start_idx < end_idx and start_idx < n and end_idx <= n:
            bin_data_orig = user_histogram[start_idx:end_idx]
            if len(bin_data_orig) > 0:
                mean_orig = np.mean(bin_data_orig)
                bin_width = end_idx - start_idx
                
                laplace_param_for_mean_noise = (epsilon2 * bin_width)
                if laplace_param_for_mean_noise <= 1e-9 or epsilon2 == 0: 
                    laplace_noise_val = 0
                else:
                    noise_b_for_laplace = 1.0 / laplace_param_for_mean_noise
                    laplace_noise_val = np.random.laplace(0, noise_b_for_laplace)
                
                noisy_mean = mean_orig + laplace_noise_val
                output_histogram[start_idx:end_idx] = noisy_mean
        elif start_idx >= end_idx and k > 0:
             # This might indicate an issue in boundary perturbation if k > 0 and we get such segments
             # print(f"警告 (Mean_StructureFirst): 输出构建时无效边界段: start={start_idx}, end={end_idx}")
             pass


        start_idx = end_idx
        if start_idx >= n and n > 0 : # 只有当n>0时，start_idx>=n才意味着结束
            break
            
    final_output = np.maximum(0, np.round(output_histogram)).astype(int)
    return final_output

def Median_StructureFirst(user_histogram, epsilon, k):
    """
    Implements the Median-StructureFirst algorithm for differentially private histogram publication.
    This corresponds to Algorithm 6 in the referenced literature (e.g., Qardaji et al.).

    The algorithm first perturbs the bin boundaries using the exponential mechanism
    (allocating `epsilon/2` budget), then adds Laplace noise to the median count of each
    resulting bin (allocating `epsilon/2` budget).

    Args:
        user_histogram (np.ndarray): The original non-private histogram counts.
        epsilon (float): The total differential privacy budget (must be positive).
        k (int): The desired number of bins in the output histogram (must be non-negative).
                 If k=0, an empty array is returned. If k > n, k is capped at n.

    Returns:
        np.ndarray: The differentially private histogram. Counts are rounded and non-negative.
                    Returns an empty array if k=0 or n=0.

    Raises:
        ValueError: If epsilon is not positive or k is negative.
    """
    if epsilon <= 0:
        raise ValueError("Epsilon 必须为正 ")
    if k < 0:
        raise ValueError("区间数量 k 不能为负 ")
    n = len(user_histogram)

    if k == 0:
        return np.array([])
    if n == 0:
        return np.array([]) if k==0 else np.array([0]*0)


    if k > n :
        # print(f"警告 (Median_StructureFirst): k ({k}) 大于数据点数量 ({n}) 将 k 设为 {n} ")
        k = n

    epsilon1 = epsilon / 2.0
    epsilon2 = epsilon / 2.0

    sae_cache_orig = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i, n):
            sae_cache_orig[i][j] = calculate_sae_median_repr(user_histogram[i:j+1])
            
    dp_table_orig, _ = _find_optimal_histogram_dp_generic(user_histogram, k, sae_cache_orig)
    
    structure_sensitivity_sae = 1.0 # Lemma 9
    
    perturbed_end_boundaries = _perturb_boundaries_exponential_mechanism(
        dp_table_orig, sae_cache_orig, n, k, epsilon1, structure_sensitivity_sae
    )
    
    output_histogram = np.zeros_like(user_histogram, dtype=float)
    start_idx = 0
    for end_idx in perturbed_end_boundaries:
        if start_idx < end_idx and start_idx < n and end_idx <= n:
            bin_data_orig = user_histogram[start_idx:end_idx]
            if len(bin_data_orig) > 0:
                median_orig = np.median(bin_data_orig)
                
                if epsilon2 <= 1e-9: 
                    laplace_noise_val = 0
                else:
                    # Algorithm 6, line 10: Add Lap(1/epsilon2) to median counts
                    noise_b_for_laplace = 1.0 / epsilon2 
                    laplace_noise_val = np.random.laplace(0, noise_b_for_laplace)
                
                noisy_median = median_orig + laplace_noise_val
                output_histogram[start_idx:end_idx] = noisy_median
        elif start_idx >= end_idx and k > 0:
             # print(f"警告 (Median_StructureFirst): 输出构建时无效边界段: start={start_idx}, end={end_idx}")
             pass
        start_idx = end_idx
        if start_idx >= n and n > 0:
            break
            
    final_output = np.maximum(0, np.round(output_histogram)).astype(int)
    return final_output