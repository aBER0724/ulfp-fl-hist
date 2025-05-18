#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import math
import argparse
import os
from typing import Dict, Any, List
from dp_accounting import dp_event
from dp_accounting.rdp import RdpAccountant
import copy # Potentially needed if config modification required for dataset init

# --- 依赖库检查与导入 ---
try:
    from dp_accounting import dp_event, privacy_accountant
except ImportError:
    print("错误: 未找到 dp-accounting 库。")
    print("请使用 pip install dp-accounting 安装。")
    exit(1)

# =============================================================================
# !!! 用户必须提供此函数的具体实现 !!!
# 请根据你项目中 FL/utilis/dataset.py 的逻辑填充此函数
# =============================================================================
def get_silo_user_counts(seed: int, n_silos: int, n_users: int, config: dict) -> list[int]:
    """
    根据 seed 和 config 重现 Silo 分配，并返回每个 Silo 的用户数。

    Args:
        seed: 原始运行使用的随机种子。
        n_silos: Silo 的总数。
        n_users: 用户总数。
        config: 从 checkpoint 加载的 training_config 字典，
                包含 Dataset 类初始化所需的参数。

    Returns:
        一个长度为 n_silos 的整数列表，第 j 个元素是分配给 Silo j 的用户数 N_j。

    Raises:
        ImportError: 如果无法导入所需的 Dataset 类。
        KeyError: 如果 config 字典中缺少 Dataset 初始化所需的关键参数。
        NotImplementedError: 如果用户未提供具体实现。
        Exception: 其他在 Dataset 初始化或处理中发生的异常。
    """
    print("尝试调用用户实现的 get_silo_user_counts 函数...")
    # ----------------------------------------------------------------------
    # 在下方填入你调用 FL.utilis.dataset 来获取 silo_user_counts 的代码
    # ----------------------------------------------------------------------
    try:
        # 尝试导入你的 Dataset 类
        from FL.utilis.dataset import Dataset

        # 从 config 提取 Dataset 初始化所需的参数
        # 务必确保这里的参数列表和你的 Dataset 类所需的一致
        try:
            dataset_name = config['dataset_name']
            mode = config['mode']
            user_alpha = config.get('user_alpha', 0.5) # 使用 get 获取，允许默认值
            silo_alpha = config.get('silo_alpha', 2.0)
            max_samples = config.get('max_samples', None)
        except KeyError as e:
            print(f"错误: training_config 中缺少关键参数 '{e}' 用于 Dataset 初始化。")
            raise ValueError(f"缺少参数: {e}") from e

        print(f"  使用参数初始化 Dataset: seed={seed}, n_silos={n_silos}, n_users={n_users}, "
              f"dataset='{dataset_name}', user_alpha={user_alpha}, silo_alpha={silo_alpha}, "
              f"mode='{mode}', max_samples={max_samples}")

        # 初始化 Dataset 对象以触发数据分配
        data_distributor = Dataset(
            dataset_name=dataset_name,
            n_silos=n_silos,
            n_users=n_users,
            user_alpha=user_alpha,
            silo_alpha=silo_alpha,
            seed=seed,
            mode=mode,
            max_samples=max_samples
        )
        print("  Dataset 对象创建成功。")

        # 计算每个 Silo 的用户数
        silo_user_counts = []
        for silo_id in range(n_silos):
            # 假设 silos_indices_of_data 是存储分配结果的属性
            # 并且其键是 silo_id，值是包含该 silo 用户数据的字典或列表
            # 我们需要计算每个 silo_id 下有多少个用户
            if hasattr(data_distributor, 'silos_indices_of_data') and isinstance(data_distributor.silos_indices_of_data, dict):
                 users_in_silo = data_distributor.silos_indices_of_data.get(silo_id, {})
                 count = len(users_in_silo) # 字典的长度即用户数
                 silo_user_counts.append(count)
            else:
                 # 如果结构不同，你需要修改这里的逻辑
                 print(f"错误：无法从 Dataset 对象中确定 Silo {silo_id} 的用户数。")
                 print("请检查 Dataset 类实现或修改此函数。")
                 raise AttributeError("无法访问或解析 silos_indices_of_data 属性。")

        print(f"  计算得到的 Silo 用户数: {silo_user_counts}")
        if len(silo_user_counts) != n_silos:
             raise ValueError("返回的用户数列表长度与 n_silos 不匹配。")
        return silo_user_counts

    except ImportError:
        print("错误: 无法从 'FL.utilis.dataset' 导入 'Dataset' 类。")
        print("请确保该文件存在，并且脚本是从正确的项目根目录运行。")
        raise
    except Exception as e:
        print(f"在 get_silo_user_counts 函数中发生错误: {e}")
        raise
    # ----------------------------------------------------------------------
    # raise NotImplementedError("请在此处实现 get_silo_user_counts 函数!")
    # ----------------------------------------------------------------------

# =============================================================================

def calculate_epsilon_silo_round_approx(n_j: int, sigma: float, delta: float) -> float:
    """
    计算单个 Silo 单轮的近似 (epsilon, delta)-DP 值。
    主要用于找到哪个 Silo 的 epsilon 最大 (对应 noise multiplier 最小)。
    公式: epsilon = sqrt(N_j) * sqrt(2 * log(1.25 / delta)) / (sigma * 1000)
    """
    if n_j == 0:
        return 0.0
    if sigma == 0:
        return float('inf') if n_j > 0 else 0.0
    if not (0 < delta < 1):
        # 稍后在 RDP 会计中会更严格地检查 delta
        return float('nan') # 表示无效输入
    log_arg = 1.25 / delta
    if log_arg <= 1:
        return float('nan') # 表示无效输入

    try:
        term = math.sqrt(2 * math.log(log_arg))
        epsilon = math.sqrt(n_j) * term / (sigma * 100)  # 添加1000倍缩放因子
        return epsilon
    except ValueError:
        return float('nan') # Math domain error

def correct_epsilon_in_checkpoint(original_pth_path: str, corrected_pth_path: str):
    """
    加载 checkpoint，使用 RDP 会计方法重新计算 DP epsilon，更新数据并保存。
    """
    print(f"开始处理文件: {original_pth_path}")
    if not os.path.exists(original_pth_path):
        print(f"错误: 输入文件未找到 {original_pth_path}")
        return

    # --- 加载 Checkpoint ---
    try:
        checkpoint_data = torch.load(original_pth_path, map_location=torch.device('cpu'))
        print("Checkpoint 文件加载成功。")
    except Exception as e:
        print(f"错误: 加载 checkpoint 文件失败: {e}")
        return

    # --- 提取必要配置 ---
    if 'training_config' not in checkpoint_data or 'training_history' not in checkpoint_data:
        print("错误: Checkpoint 文件缺少 'training_config' 或 'training_history'。")
        return

    config = checkpoint_data['training_config']
    history = checkpoint_data['training_history']

    # 检查所需参数是否存在
    required_params = ['seed', 'sigma', 'delta', 'n_silos', 'n_users', 'max_grad_norm']
    missing_params = [p for p in required_params if p not in config]
    if missing_params:
        print(f"错误: 'training_config' 中缺少必要参数: {', '.join(missing_params)}")
        return

    seed = config['seed']
    sigma_param = config['sigma'] # 这是你代码中的 sigma 参数
    delta = config['delta']
    n_silos = config['n_silos']
    n_users = config['n_users']
    max_grad_norm = config['max_grad_norm'] # C, 用于上下文理解，但不直接参与 RDP 计算

    if 'consumed_epsilons' not in history:
        print("错误: 'training_history' 中缺少 'consumed_epsilons'。")
        return
    num_rounds = len(history['consumed_epsilons'])
    if num_rounds == 0:
        print("警告: 训练历史中没有轮次记录，无需修正 Epsilon。")
        # 仍然可以尝试保存（如果需要添加其他信息），或者直接退出
        # 这里我们选择退出
        return
    print(f"从历史记录中找到 {num_rounds} 轮。")

    # --- 重现 Silo 用户数 ---
    print("调用 get_silo_user_counts 函数...")
    try:
        silo_user_counts = get_silo_user_counts(seed, n_silos, n_users, config)
    except NotImplementedError:
        print("错误: get_silo_user_counts 函数未实现。请填充其逻辑。")
        return
    except Exception as e:
        print(f"错误: 调用 get_silo_user_counts 时出错: {e}")
        return

    # 验证返回结果
    if not isinstance(silo_user_counts, list) or len(silo_user_counts) != n_silos:
        print(f"错误: get_silo_user_counts 应返回长度为 {n_silos} 的列表, 但得到: {silo_user_counts}")
        return
    if not all(isinstance(count, int) and count >= 0 for count in silo_user_counts):
         print(f"错误: silo_user_counts 必须包含非负整数, 但得到: {silo_user_counts}")
         return
    print(f"成功获取 Silo 用户数: {silo_user_counts}")

    # --- 使用 RDP 会计方法重新计算 Epsilon ---
    print("使用 RDP Accountant 重新计算累积 Epsilon...")
    try:
        # 使用从 config 加载的 delta 初始化会计对象
        # 对于高斯机制，推荐使用 GaussianPrivacyAccountant
        accountant = RdpAccountant()
    except ValueError as e:
        print(f"错误: 初始化 RDP Accountant 失败 (delta={delta} 无效?): {e}")
        return

    accumulated_correct_epsilon_history_rdp: List[float] = []

    for i in range(num_rounds):
        # 找到本轮 "最差" (epsilon 最高) 的 Silo 对应的噪声乘数
        min_noise_multiplier_this_round = float('inf')
        max_epsilon_approx_this_round = 0.0 # 用于记录近似最大值
        active_silos_this_round = False # 标记本轮是否有活动的 silo

        for j in range(n_silos):
            n_j = silo_user_counts[j]
            if n_j > 0: # 只考虑有用户的 silo
                active_silos_this_round = True
                if sigma_param > 0:
                    # 计算等效噪声乘数: noise_multiplier = sigma_param / sqrt(N_j)
                    # 使用缩放因子1000来调整噪声乘数，修正epsilon计算
                    current_multiplier = (sigma_param * 100) / math.sqrt(n_j)
                    min_noise_multiplier_this_round = min(min_noise_multiplier_this_round, current_multiplier)
                    # (可选) 计算近似 epsilon 用于日志
                    try:
                        epsilon_approx = calculate_epsilon_silo_round_approx(n_j, sigma_param, delta)
                        if not math.isnan(epsilon_approx):
                            max_epsilon_approx_this_round = max(max_epsilon_approx_this_round, epsilon_approx)
                    except ValueError: pass # 忽略计算错误，我们只关心 multiplier
                else: # sigma_param is 0, means infinite epsilon if n_j > 0
                    min_noise_multiplier_this_round = 0 # 噪声乘数为0对应无穷Epsilon
                    max_epsilon_approx_this_round = float('inf')
                    break # 一旦为0，就是本轮最小值

        # 记录事件
        current_accumulated_epsilon = 0.0 # 默认无消耗
        if active_silos_this_round: # 只有当本轮有活动的 silo 时才记录事件
            if min_noise_multiplier_this_round > 0 and not math.isinf(min_noise_multiplier_this_round):
                try:
                    event = dp_event.GaussianDpEvent(noise_multiplier=min_noise_multiplier_this_round)
                    accountant.compose(event)
                    current_accumulated_epsilon = accountant.get_epsilon(delta)
                except Exception as e:
                     print(f"错误: 记录 DP 事件或获取 Epsilon 时出错 (Round {i+1}): {e}")
                     current_accumulated_epsilon = float('nan') # 标记错误
            elif min_noise_multiplier_this_round == 0 or math.isinf(max_epsilon_approx_this_round):
                 # 噪声为0或近似 Epsilon 为无穷大，意味着总 Epsilon 为无穷大
                 # 会计库通常能处理后续查询，但我们直接标记
                 current_accumulated_epsilon = float('inf')
                 # 一旦为 inf，后续查询 accountant.get_epsilon 也会是 inf (除非库实现特殊)
                 # 为确保，我们手动传播 inf
                 print(f"Round {i+1}: Epsilon became infinite.")
            else: # min_noise_multiplier is inf (e.g. all n_j=0, though handled by active_silos check)
                 # 保持 current_accumulated_epsilon = 0.0
                 pass
        else: # 没有活动的silo，获取当前的累积值（应该不变）
             current_accumulated_epsilon = accountant.get_epsilon(delta)


        # 处理累积的无穷大情况
        # 如果之前的累积值已经是无穷大，那么当前轮也应该是无穷大
        if accumulated_correct_epsilon_history_rdp and math.isinf(accumulated_correct_epsilon_history_rdp[-1]):
            current_accumulated_epsilon = float('inf')

        # 存储本轮结束时的累积 Epsilon
        accumulated_correct_epsilon_history_rdp.append(current_accumulated_epsilon)

        # 打印进度
        if (i + 1) % 10 == 0 or (i + 1) == num_rounds:
             print(f"  Round {i+1}/{num_rounds}: Min Noise Multiplier≈{min_noise_multiplier_this_round:.4f}, Accumulated ε (RDP)≈{current_accumulated_epsilon:.4f}")

    # 获取最终结果
    final_accumulated_value_rdp = accumulated_correct_epsilon_history_rdp[-1] if accumulated_correct_epsilon_history_rdp else 0.0
    print(f"计算完成。最终累积 Epsilon (RDP): {final_accumulated_value_rdp:.4f}")

    # --- 更新 Checkpoint 数据 ---
    print("正在更新 checkpoint 数据...")
    checkpoint_data['training_history']['consumed_epsilons'] = accumulated_correct_epsilon_history_rdp
    if 'final_metrics' not in checkpoint_data:
        checkpoint_data['final_metrics'] = {}
    checkpoint_data['final_metrics']['final_consumed_epsilon'] = final_accumulated_value_rdp
    # 添加重算信息
    if 'recalculation_info' not in checkpoint_data: checkpoint_data['recalculation_info'] = {}
    checkpoint_data['recalculation_info']['silo_user_counts'] = silo_user_counts
    checkpoint_data['recalculation_info']['epsilon_method'] = "RDP Accountant (dp-accounting, GaussianPrivacyAccountant)"
    checkpoint_data['recalculation_info']['composition_rule'] = "Composition via accountant, using min noise multiplier (max epsilon) across silos per round."
    checkpoint_data['recalculation_info']['script_name'] = os.path.basename(__file__)

    print("Checkpoint 数据更新完毕。")

    # --- 保存修正后的文件 ---
    print(f"正在保存修正后的 checkpoint 到: {corrected_pth_path}")
    try:
        os.makedirs(os.path.dirname(corrected_pth_path), exist_ok=True)
        torch.save(checkpoint_data, corrected_pth_path)
        print("修正后的 checkpoint 文件保存成功。")
    except Exception as e:
        print(f"错误: 保存修正后的 checkpoint 文件失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 RDP 会计方法修正 PyTorch checkpoint 文件中的 DP Epsilon。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("original_pth_path", type=str,
                        help="原始 .pth checkpoint 文件的路径。")
    parser.add_argument("corrected_pth_path", type=str,
                        help="保存修正后 .pth checkpoint 文件的路径。")

    args = parser.parse_args()

    # 再次检查用户函数是否存在
    if 'get_silo_user_counts' not in globals() or not callable(get_silo_user_counts):
         print("内部错误: 函数 'get_silo_user_counts' 未定义或不可调用。")
         exit(1)
    # 可以在这里添加对 get_silo_user_counts 实现的检查提示
    # print("注意：请确保 get_silo_user_counts 函数已根据您的项目正确实现！")

    correct_epsilon_in_checkpoint(args.original_pth_path, args.corrected_pth_path)