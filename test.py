current_fixed_end_idx = 30
k = 10

for j_boundary_idx in range(k - 1, 0, -1):
        # 分桶范围为 [min_q_val, max_q_val]
        min_q_val = j_boundary_idx
        max_q_val = current_fixed_end_idx - 1

        candidate_split_points_q = list(range(min_q_val, max_q_val + 1))
        print("j_boundary_idx=", j_boundary_idx)
        print(candidate_split_points_q)