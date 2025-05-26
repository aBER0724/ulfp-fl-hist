import numpy as np
import math

# histogram_dp 高斯噪声
def histogram_dp(user_histogram, sigma):
    """
    为用户直方图添加高斯噪声以实现差分隐私. 

    参数:
        - user_histogram (np.ndarray): 原始直方图数据 
        - sigma (float): 要添加的高斯噪声的标准差

    返回:
        - np.ndarray: 带噪声的直方图, 计数四舍五入到最近的整数并向下取整到0. 
    """
    noise = np.random.normal(0, sigma, size=len(user_histogram)) # 生成高斯噪声
    return np.maximum(0, np.round(user_histogram + noise)).astype(int) # 添加噪声, 四舍五入并确保非负

# 计算平方误差和 SSE
def calculate_sse_mean_repr(arr):
    """
    计算给定数组的平方误差和(SSE), 使用平均值作为代表值. 

    这通常用于直方图优化问题中, 以量化
    通过使用单个值(平均值)表示一定范围的区间所引入的误差. 

    参数:
        - arr (np.ndarray): 表示直方图片段的一维 NumPy 数组. 

    返回:
        - float: 片段的平方误差和 (SSE). 如果数组为空, 则返回 0. 
    """
    if len(arr) == 0:
        return 0
    representative_value = np.mean(arr) # 计算平均值
    return np.sum((arr - representative_value)**2) # 计算 SSE

# 计算绝对误差和 SAE
def calculate_sae_median_repr(arr):
    """
    计算给定数组的绝对误差和(SAE), 使用中位数作为代表值. 

    参数:
        - arr (np.ndarray): 表示直方图片段的一维 NumPy 数组. 

    返回:
        - float: 片段的绝对误差和 (SAE). 如果数组为空, 则返回 0. 
    """
    if len(arr) == 0:
        return 0
    representative_value = np.median(arr) # 计算中位数
    return np.sum(np.abs(arr - representative_value)) # 计算 SAE

# 最优直方图结构-通用动态规划函数
def _find_optimal_histogram_dp_generic(histogram_data, max_k, error_cache):
    """
    `find_optimal_sse_histogram_dp` 和 `find_optimal_sae_histogram_dp` 的动态规划的辅助函数

    参数:
        - histogram_data (np.ndarray): 直方图数据
        - max_k (int): 直方图中允许的最大分桶数
        - error_cache (np.ndarray): 一个预计算的 n x n 矩阵, 其中 `error_cache[i][j]`为`histogram_data[i...j]`片段的误差.

    返回:
        - tuple[np.ndarray, np.ndarray]:
            - dp_table: 一个 (n+1) x (max_k+1) 的 DP 表, 其中 `dp_table[i][j]` 存储
                        将 `histogram_data` 的前 `i` 个数据点
                        划分到 `j` 个区间的最小总误差. 
            - ptr_table: 一个 (n+1) x (max_k+1) 的指针表, 用于重建
                            最优区间边界. `ptr_table[i][j]` 存储将前 `i` 个点
                            划分到 `j` 个区间时, 最后一个区间的起始索引. 
    """
    n = len(histogram_data)
    dp_table = np.full((n + 1, max_k + 1), np.inf)  # DP 表
    ptr_table = np.zeros((n + 1, max_k + 1), dtype=int)  # 指针表
    dp_table[0][0] = 0

    for j_bins in range(1, max_k + 1): # 遍历所有桶数, 直到最大 max_k
        for i_data_end_idx in range(j_bins, n + 1):
            min_error_i_j = np.inf
            best_l_i_j = -1
            for l_last_bin_start_idx in range(j_bins - 1, i_data_end_idx):
                # 找到最后一个桶的最佳起始位置
                last_bin_error = error_cache[l_last_bin_start_idx][i_data_end_idx-1] if l_last_bin_start_idx < i_data_end_idx else 0

                if np.isfinite(dp_table[l_last_bin_start_idx][j_bins - 1]):
                    current_total_error = dp_table[l_last_bin_start_idx][j_bins - 1] + last_bin_error
                    if current_total_error < min_error_i_j:
                        # 更新最小误差值的起始点
                        min_error_i_j = current_total_error
                        best_l_i_j = l_last_bin_start_idx
            
            dp_table[i_data_end_idx][j_bins] = min_error_i_j # 将前 i 个数据点分成 j 个桶的最小总误差
            ptr_table[i_data_end_idx][j_bins] = best_l_i_j # 最后一个分桶的起始索引

    return dp_table, ptr_table

# SSE 最优直方图结构
def find_optimal_sse_histogram_dp(histogram_data, max_k):
    """
    使用动态规划找到最小化总平方误差和 (SSE) 的最优直方图结构. 
    每个区间由其原始值的平均值表示. 

    参数:
        - histogram_data (np.ndarray): 直方图数据
        - max_k (int): 直方图中允许的最大分桶数

    返回:
        - tuple[np.ndarray, np.ndarray]:
            - dp_table: 存储最小 SSE 的 DP 表
            - ptr_table: 用于重建边界的指针表
    """
    n = len(histogram_data)
    sse_cache = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i, n):
            sub_array = histogram_data[i:j+1]
            sse_cache[i][j] = calculate_sse_mean_repr(sub_array)
    return _find_optimal_histogram_dp_generic(histogram_data, max_k, sse_cache)

# SAE 最优直方图结构
def find_optimal_sae_histogram_dp(histogram_data, max_k):
    """
    使用动态规划找到最小化总绝对误差和 (SAE) 的最优直方图结构. 
    每个区间由其原始值的中位数表示. 

    参数:
        - histogram_data (np.ndarray): 直方图数据
        - max_k (int): 直方图中允许的最大分桶数

    返回:
        - tuple[np.ndarray, np.ndarray]:
            - dp_table: 存储最小 SAE 的 DP 表
            - ptr_table: 用于重建边界的指针表
    """
    n = len(histogram_data)
    sae_cache = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i, n):
            sub_array = histogram_data[i:j+1]
            sae_cache[i][j] = calculate_sae_median_repr(sub_array)
    return _find_optimal_histogram_dp_generic(histogram_data, max_k, sae_cache)

# 重建边界
def reconstruct_boundaries(ptr_table, n, k):
    """
    从动态规划直方图优化生成的指针表中重建最优区间边界. 

    参数:
        - ptr_table (np.ndarray): 来自 `_find_optimal_histogram_dp_generic` 的指针表. 
                                `ptr_table[i][j]` 存储将前 `i` 个点划分到 `j` 个区间时, 
                                最后一个区间的起始索引. 
        - n (int): 原始直方图中的总数据点数
        - k (int): 要为其重建边界的桶数

    返回:
        - list[int]: 一个整数列表, 表示重建区间的结束点(不包含). 
                    对于 `k` 个区间, 此列表将包含 `k` 个值
                    `[b_1, b_2, ..., b_k]`, 其中区间为 `[0, b_1)`、`[b_1, b_2)`、...、
                    `[b_{k-1}, b_k=n)`. 如果 k=0, 则返回空列表. 
    """
    boundaries = []
    current_k = k
    current_idx = n
    # 从最后一个桶开始, 向前逐个确定每个分桶的起始位置
    while current_k > 0 and current_idx > 0:
        start_index = ptr_table[current_idx][current_k]
        if start_index < 0 or start_index > current_idx :
            print(f"警告: 发现无效指针 k={current_k}, idx={current_idx}. 指针={start_index}")
            boundaries.append(current_idx) 
            break
        
        boundaries.append(start_index) 
        current_idx = start_index
        current_k -= 1
    
    final_boundaries = sorted(list(set(boundaries))) # 去重并排序
    
    processed_boundaries = []
    if 0 in final_boundaries: # 移除用于起始位置的 0
        final_boundaries.remove(0)
    
    processed_boundaries = sorted(list(set(final_boundaries))) # 去重并排序
    
    # 添加最终的边界 n (最后一个桶的结束位置)
    if n not in processed_boundaries:
        processed_boundaries.append(n)

    # k 不应为 0
    if len(processed_boundaries) != k and k != 0:
         pass
     
    # 返回的是分桶的结束点(不包含起始的 0)
    return processed_boundaries 

# Mean-NoiseFirst 算法
def Mean_NoiseFirst(user_histogram, epsilon):
    """
    Mean-NoiseFirst 算法, 用于差分隐私直方图发布
    通过最小化目标函数自动选择最优区间数 `k`

    首先向原始直方图添加拉普拉斯噪声, 
    然后在该噪声数据上找到最小化 SSE 的最优 k 区间直方图结构, 
    最后选择最小化 `SSE_noisy - (2n - 4k) / epsilon^2` 的 `k`. 

    参数:
        - user_histogram (np.ndarray): 原始直方图
        - epsilon (float): 差分隐私预算

    返回:
        - tuple[np.ndarray, int]:
            - final_output (np.ndarray): 具有 `best_k` 个区间的差分隐私直方图 
            - best_k (int): 自动选择的最优区间数
    引发:
        ValueError: 如果 epsilon 不是正数. 
    """
    if epsilon <= 0:
        raise ValueError("Epsilon 必须为正 ")

    n = len(user_histogram)
    if n == 0:
        return np.array([]), 0

    # 添加拉普拉斯噪声
    scale = 1.0 / epsilon
    noise = np.random.laplace(0, scale, n)
    noisy_histogram = np.maximum(0, np.round(user_histogram + noise)).astype(int)

    # 在噪声数据上找到所有 k 的最小 SSE
    dp_table, ptr_table = find_optimal_sse_histogram_dp(noisy_histogram, max_k=n)

    # 选择最优 k*
    best_k = -1
    min_objective_value = np.inf
    epsilon_sq = epsilon**2

    for k_candidate in range(1, n + 1):
        sse_noisy = dp_table[n][k_candidate] # SSE(H_k*, D_hat)

        if np.isfinite(sse_noisy):
            # 目标：最小化 SAE_noisy - (2n - 4k)/epsilon
            objective = sse_noisy - (2.0 * n - 4.0 * k_candidate) / epsilon_sq
            if objective < min_objective_value:
                min_objective_value = objective
                best_k = k_candidate
                
    # 如果没有找到有效的 k, 回退到 k = 1
    if best_k == -1: 
        print("警告: 无法自动找到最优k 回退到 k=1 ")
        best_k = 1 if n > 0 else 0

    # 根据 k* 重建边界
    boundaries = reconstruct_boundaries(ptr_table, n, best_k)
    
    # 构建输出直方图
    output_histogram = np.zeros_like(noisy_histogram, dtype=float)
    start_idx = 0
    for end_idx in boundaries:
        if start_idx < end_idx and start_idx < n and end_idx <= n:
            bin_data_noisy = noisy_histogram[start_idx:end_idx]
            if len(bin_data_noisy) > 0:
                representative_value = np.mean(bin_data_noisy)
                output_histogram[start_idx:end_idx] = representative_value
        elif start_idx >= end_idx and best_k > 0:
             print(f"警告 (Mean_NoiseFirst): 无效边界段: start={start_idx}, end={end_idx} for k={best_k}")
        start_idx = end_idx
        if start_idx >= n:
            break
    
    final_output = np.maximum(0, np.round(output_histogram)).astype(int)
    return final_output, best_k

# Median-NoiseFirst 算法
def Median_NoiseFirst(user_histogram, epsilon):
    """
    Median-NoiseFirst 算法, 用于差分隐私直方图发布
    通过最小化目标函数自动选择最优区间数 `k`

    该算法首先添加拉普拉斯噪声, 
    然后在该噪声数据上找到最小化 SAE 的最优 k 区间结构, 
    最后选择最小化 `SAE_noisy - 3(n - k) / epsilon` 的 `k`. 

    参数:
        - user_histogram (np.ndarray): 原始非私有直方图计数. 
        - epsilon (float): 差分隐私预算(必须为正). 

    返回:
        - tuple[np.ndarray, int]:
            - final_output (np.ndarray): 具有 `best_k` 个区间的差分隐私直方图
            - best_k (int): 自动选择的最优区间数
    引发:
        ValueError: 如果 epsilon 不是正数. 
    """
    if epsilon <= 0:
        raise ValueError("Epsilon 必须为正 ")

    n = len(user_histogram)
    if n == 0:
        return np.array([]), 0

    # 添加拉普拉斯噪声
    scale = 1.0 / epsilon
    noise = np.random.laplace(0, scale, n)
    noisy_histogram = np.maximum(0, np.round(user_histogram + noise)).astype(int)

    # 在噪声数据上找到所有 k 的最小 SAE
    dp_table, ptr_table = find_optimal_sae_histogram_dp(noisy_histogram, max_k=n)

    # 选择最优 k*
    best_k = -1
    min_objective_value = np.inf

    for k_candidate in range(1, n + 1):
        sae_noisy = dp_table[n][k_candidate] # SAE(H_k*, D_hat)

        if np.isfinite(sae_noisy):
            # 目标：最小化 SAE_noisy - 3(n - k)/epsilon
            objective = sae_noisy - (3.0 * (n - k_candidate)) / epsilon
            if objective < min_objective_value:
                min_objective_value = objective
                best_k = k_candidate
                
    # 如果没有找到有效的 k, 回退到 k = 1
    if best_k == -1:
        print("警告: 无法自动找到最优k 回退到 k=1 ")
        best_k = 1 if n > 0 else 0

    # 根据 k* 重建边界
    boundaries = reconstruct_boundaries(ptr_table, n, best_k)
    
    # 构建输出直方图
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

# StructureFirst 算法-边界扰动
def _perturb_boundaries_exponential_mechanism(dp_table_orig,error_cache_orig,n,k,epsilon1,metric_sensitivity):
    """
    使用指数机制迭代选择和扰动直方图区间边界. 

    该函数从右到左确定 `k-1` 个内部分割点 `r_1, ..., r_{k-1}`. 
    为 `r_j`(给定 `r_{j+1}`)选择分割点 `q` 的质量分数基于
    最小化 `Error_DP(H*(D_q, j), D_q) + Error_Segment(D, q, r_{j+1}-1)`, 其中 `Error_DP`
    是最优 `j` 区间直方图在 `D[0...q-1]` 上的误差, `Error_Segment` 是
    将 `D[q...r_{j+1}-1]` 表示为一个区间的误差. 

    参数:
        - dp_table_orig (np.ndarray): 在原始数据上计算的 DP 表, 其中 `dp_table_orig[i][j]` 是将前 `i` 个点划分到 `j` 个区间的最小误差. 
        - error_cache_orig (np.ndarray): 在原始数据上为任何片段 `[i...j]` 预计算的误差
        - n (int): 总数据点数(原始直方图的长度)
        - k (int): 最终直方图中期望的区间数
        - epsilon1 (float): 分配给扰动所有 `k-1` 个边界的隐私预算. 
        - metric_sensitivity (float): 使用的误差度量的敏感度, 均值的 SSE 是 `2F+1`, 中位数的 SAE 是 `1` 

    返回:
        - list[int]: `k` 个扰动后的结束边界列表 `[r_1, r_2, ..., r_k=n]`. 
                   它们定义了 `k` 个区间：`[0, r_1)`、`[r_1, r_2)`、...、`[r_{k-1}, r_k=n)`. 
                   如果 k=0, 则返回空列表. 

    引发:
        ValueError: 如果 k 为负数. 
    """
    if k < 0:
        raise ValueError("区间数量 k 不能为负 ")
    if k == 0:
        return []
    
    perturbed_end_boundaries = [0] * k

    if k == 1:
        if n > 0:
            perturbed_end_boundaries[0] = n
            return perturbed_end_boundaries
        else: 
            return []
        
    # 最后一个分桶的结束位置固定为 n
    perturbed_end_boundaries[k - 1] = n  

    current_fixed_end_idx = n

    # 从最后一个分桶开始
    for j_boundary_idx in range(k - 1, 0, -1):
        # 分桶范围为 [min_q_val, max_q_val]
        min_q_val = j_boundary_idx
        max_q_val = current_fixed_end_idx - 1

        # 候选边界点
        candidate_split_points_q = list(range(min_q_val, max_q_val + 1))
        costs = []

        for q_val in candidate_split_points_q:
            # q_val 左侧误差
            error_prev_bins = dp_table_orig[q_val][j_boundary_idx]
            
            # q_val 右侧侧误差
            error_current_segment = error_cache_orig[q_val][current_fixed_end_idx - 1] if q_val < current_fixed_end_idx else 0
            
            # 以 q_val 划分造成的总误差
            cost = error_prev_bins + error_current_segment
            costs.append(cost)

        costs = np.array(costs)
        
        # 使用指数机制选择边界
        if k == 1:
            prob_scale_exp = np.inf
        elif epsilon1 <= 1e-9:
            prob_scale_exp = np.inf
        else:
            prob_scale_exp = (2.0 * (k - 1) * metric_sensitivity) / epsilon1
        
        if prob_scale_exp <= 1e-9:
             prob_scale_exp = 1e-9

        min_c = np.min(costs[np.isfinite(costs)]) if np.any(np.isfinite(costs)) else 0
        log_probs = -(costs - min_c) / prob_scale_exp
        log_probs[np.isinf(costs)] = -np.inf 
        log_probs = np.clip(log_probs, -700, 700) # 防止 np.exp 的溢出
        probabilities = np.exp(log_probs)
        probabilities[~np.isfinite(probabilities)] = 0.0 # 清理 NaN 或 Inf
        
        # 归一化
        sum_probs = np.sum(probabilities)
        if sum_probs > 0:
            probabilities /= sum_probs
        else: 
            # 概率都是 0, 回退使用均匀分布
            probabilities = np.ones_like(costs) / len(costs) if len(costs) > 0 else []

        # 根据概率选择边界点
        if len(candidate_split_points_q) > 0 and len(probabilities) == len(candidate_split_points_q) and np.sum(probabilities) > 0:
            chosen_split_point = np.random.choice(candidate_split_points_q, p=probabilities)
        elif len(candidate_split_points_q) > 0: 
            chosen_split_point = np.random.choice(candidate_split_points_q)

        # 更新固定边界
        perturbed_end_boundaries[j_boundary_idx - 1] = chosen_split_point 
        current_fixed_end_idx = chosen_split_point
        
    return perturbed_end_boundaries

# Mean-StructureFirst 算法
def Mean_StructureFirst(user_histogram, epsilon, k, F):
    """
    Mean-StructureFirst 算法, 用于差分隐私直方图发布.

    首先使用指数机制扰动区间边界(分配 `epsilon/2` 预算),
    然后向每个分桶的平均计数添加拉普拉斯噪声(分配 `epsilon/2` 预算). 

    参数:
        - user_histogram (np.ndarray): 原始直方图
        - epsilon (float): 总差分隐私预算
        - k (int): 输出直方图中期望的分桶数
        - F (float): 原始 `user_histogram` 中计数的上限. 进行边界扰动的敏感度. 

    返回:
        - np.ndarray: 差分隐私直方图

    引发:
        ValueError: 如果 epsilon 不是正数或 k 是负数. 
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
        k = n

    epsilon1 = epsilon / 2.0 # 结构扰动
    epsilon2 = epsilon / 2.0 # 数值扰动

    # 计算 SSE 和 DP 表 
    sse_cache_orig = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i, n):
            sse_cache_orig[i][j] = calculate_sse_mean_repr(user_histogram[i:j+1])
    
    dp_table_orig, _ = _find_optimal_histogram_dp_generic(user_histogram, k, sse_cache_orig)

    # 敏感度为 2F + 1
    structure_sensitivity = 2.0 * F + 1.0
    if structure_sensitivity <= 0: 
        structure_sensitivity = 1.0 
    
    # 扰动边界
    perturbed_end_boundaries = _perturb_boundaries_exponential_mechanism(
        dp_table_orig, sse_cache_orig, n, k, epsilon1, structure_sensitivity
    )

    # 构建输出直方图
    output_histogram = np.zeros_like(user_histogram, dtype=float)
    start_idx = 0
    for end_idx in perturbed_end_boundaries:
        if start_idx < end_idx and start_idx < n and end_idx <= n:
            bin_data_orig = user_histogram[start_idx:end_idx]
            if len(bin_data_orig) > 0:
                mean_orig = np.mean(bin_data_orig)
                bin_width = end_idx - start_idx # 桶宽度
                laplace_param_for_mean_noise = (epsilon2 * bin_width)
                
                if laplace_param_for_mean_noise <= 1e-9 or epsilon2 == 0: 
                    laplace_noise_val = 0
                else:
                    noise_b_for_laplace = 1.0 / laplace_param_for_mean_noise
                    laplace_noise_val = np.random.laplace(0, noise_b_for_laplace)
                
                # 添加拉普拉斯噪声
                noisy_mean = mean_orig + laplace_noise_val
                output_histogram[start_idx:end_idx] = noisy_mean
        elif start_idx >= end_idx and k > 0:
            pass

        start_idx = end_idx
        if start_idx >= n and n > 0:
            break
            
    final_output = np.maximum(0, np.round(output_histogram)).astype(int)
    return final_output

# Median-StructureFirst 算法
def Median_StructureFirst(user_histogram, epsilon, k):
    """
    Median-StructureFirst 算法, 用于差分隐私直方图发布.

    该算法首先使用指数机制扰动区间边界(分配 `epsilon/2` 预算), 
    然后向每个分桶的平均计数添加拉普拉斯噪声(分配 `epsilon/2` 预算). 

    参数:
        - user_histogram (np.ndarray): 原始直方图 
        - epsilon (float): 总差分隐私预算
        - k (int): 输出直方图中期望的分桶数

    返回:
        - np.ndarray: 差分隐私直方图

    引发:
        ValueError: 如果 epsilon 不是正数或 k 是负数. 
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
        k = n

    epsilon1 = epsilon / 2.0 # 结构扰动
    epsilon2 = epsilon / 2.0 # 数值扰动

    # 计算 SAE 和 DP 表 
    sae_cache_orig = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i, n):
            sae_cache_orig[i][j] = calculate_sae_median_repr(user_histogram[i:j+1])
            
    dp_table_orig, _ = _find_optimal_histogram_dp_generic(user_histogram, k, sae_cache_orig)
    
    # 敏感度为 1
    structure_sensitivity_sae = 1.0
    
    # 扰动边界
    perturbed_end_boundaries = _perturb_boundaries_exponential_mechanism(
        dp_table_orig, sae_cache_orig, n, k, epsilon1, structure_sensitivity_sae
    )
    
    # 构建输出直方图
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
                    noise_b_for_laplace = 1.0 / epsilon2 
                    laplace_noise_val = np.random.laplace(0, noise_b_for_laplace)
                
                # 添加拉普拉斯噪声
                noisy_median = median_orig + laplace_noise_val
                output_histogram[start_idx:end_idx] = noisy_median
        elif start_idx >= end_idx and k > 0:
             pass
        start_idx = end_idx
        if start_idx >= n and n > 0:
            break
            
    final_output = np.maximum(0, np.round(output_histogram)).astype(int)
    return final_output