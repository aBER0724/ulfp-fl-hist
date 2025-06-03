import numpy as np
import os
import logging
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import argparse
from FL.utilis.dataset import Dataset

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(message)s', datefmt='%Y-%m-%d %H:%M')
logger = logging.getLogger(__name__)

def analyze_distributions(data_indices_of_user, data_indices_of_silo, dataset_train, silo_datasets, mode, dataset_object=None):
    """
    分析并准备数据, 用于可视化跨用户和数据机构的数据样本分布, 以及数据机构内的标签分布. 

    参数:
        - data_indices_of_user (dict): 将用户 ID 映射到分配给该用户的数据索引列表. 
                                     示例: {user_id: [idx1, idx2, ...]}
        - data_indices_of_silo (dict): 将数据机构 ID 映射到用户 ID, 然后再映射到数据索引列表. 
                                     示例: {silo_id: {user_id: [idx_a, idx_b, ...]}}
                                     或者, 如果提供了 `dataset_object`, 如果首选 `get_user_histogram`, 则可能不直接使用此参数. 
        - dataset_train (torch.utils.data.Dataset): 完整的训练数据集, 用于获取标签. 
        - silo_datasets (dict): 将数据机构 ID 映射到其各自的
                              torch.utils.data.Subset 数据集. 
                              示例: {silo_id: <Subset object>}
        - mode (str): 数据分布模式的字符串标识符. 
        - dataset_object (FL.utilis.dataset.Dataset, optional): Dataset 类的实例 

    返回:
        PlotFunctions: 内部 PlotFunctions 类的实例, 其中包含生成和保存分布图的方法. 
    """

    # 创建一个将数据索引映射到用户 ID 的字典
    data_to_user = {}
    for user_id, indices in data_indices_of_user.items():
        for idx in indices:
            data_to_user[idx] = user_id

    # 1. 计算每个数据机构的用户分布
    silo_user_distributions = defaultdict(dict)
    
    if dataset_object is not None:
        # 如果提供了 dataset_object, 则使用新的 get_user_histogram 方法
        for silo_id in silo_datasets.keys():
            silo_user_distributions[silo_id] = dataset_object.get_user_histogram(silo_id)
    else:
        # 如果未提供 dataset_object, 则回退到手动计算
        for silo_id, silo_indices in data_indices_of_silo.items():
            user_counts = defaultdict(int)
            for idx in silo_indices:
                user_id = data_to_user[idx]
                user_counts[user_id] += 1
            silo_user_distributions[silo_id] = user_counts

    # 2. 标签分布
    silo_label_distributions = {}
    for silo_id, dataset in silo_datasets.items():
        labels = [dataset_train.targets[i].item() for i in dataset.indices]
        unique, counts = np.unique(labels, return_counts=True)
        silo_label_distributions[silo_id] = dict(zip(unique, counts))

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), './test_output'))
    os.makedirs(output_dir, exist_ok=True)

    class PlotFunctions:
        @staticmethod
        def plot_silos_distribution():
            """
            生成并保存一个图表, 显示每个数据机构的前 N 个用户分布（样本计数）和标签分布. 

            该图表将显示多个子图, 每个数据机构有两个子图：
            一个用于用户样本计数, 一个用于标签计数. 
            """
            n_silos = len(silo_datasets)
            silos_per_row = 2  # 每行的数据机构数量 (目标: 一行最多4个子图, 每个数据机构2个子图)
            rows = (n_silos + silos_per_row - 1) // silos_per_row
            plt.figure(figsize=(24, 6 * rows))  # 调整了每行的高度

            for silo_idx, silo_id in enumerate(silo_datasets.keys()):
                row = silo_idx // silos_per_row
                col_base = (silo_idx % silos_per_row) * 2

                # 用户分布子图
                plt.subplot(rows, silos_per_row * 2, row * silos_per_row * 2 + col_base + 1)
                user_counts = silo_user_distributions[silo_id]
                
                # 处理 user_counts 可能是列表或字典的情况
                if isinstance(user_counts, dict):
                    top_users = sorted(user_counts.items(),
                                key=lambda x: x[1], reverse=True)[:10]
                    users = [f'User-{u[0]}' for u in top_users]
                    counts = [u[1] for u in top_users]
                elif isinstance(user_counts, list):
                    # 如果是列表, 则假定它已排序或已适当格式化
                    top_users = user_counts[:10]
                    if len(top_users) > 0 and isinstance(top_users[0], tuple):
                        # 如果是元组列表 (user_id, count)
                        users = [f'User-{u[0]}' for u in top_users]
                        counts = [u[1] for u in top_users]
                    else:
                        # 如果只是计数列表, 则创建通用标签
                        users = [f'User-{i}' for i in range(len(top_users))]
                        counts = top_users
                else:
                    # 意外类型的回退处理
                    users = []
                    counts = []
                
                plt.bar(users, counts, color='skyblue')
                plt.title(
                    f'Silo-{silo_id} Top 10 Users\n(Total Users: {len(user_counts) if hasattr(user_counts, "__len__") else "Unknown"})',
                    fontsize=22)
                plt.xlabel('User ID', fontsize=20)
                plt.ylabel('Samples', fontsize=20)
                plt.xticks(rotation=45, fontsize=14)
                plt.yticks(fontsize=14)

                # 标签分布子图
                plt.subplot(rows, silos_per_row * 2, row * silos_per_row * 2 + col_base + 2)
                label_dist = silo_label_distributions[silo_id]
                labels = sorted(label_dist.keys())
                counts = [label_dist[l] for l in labels]
                plt.bar(labels, counts, color='salmon')
                plt.title(
                    f'Silo-{silo_id} Label Distribution\n(Total Samples: {sum(counts)})',
                    fontsize=22)
                plt.xlabel('Label', fontsize=20)
                plt.ylabel('Samples', fontsize=20)
                plt.xticks(range(10), labels, rotation=45, fontsize=14)
                plt.yticks(fontsize=14)

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f'silo_distributions_{mode}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        @staticmethod
        def plot_user_distribution_heatmap():
            """
            生成并保存一个热力图, 可视化每个用户的数据在每个数据机构内的百分比（或样本计数贡献）. 

            热力图在 y 轴上显示数据机构, 在 x 轴上显示用户. 
            单元格值表示用户样本在特定数据机构中的比例. 
            """
            # 调整图形尺寸为更窄更高
            plt.figure(figsize=(50, 30))  # 修改为更窄更高的尺寸
            user_matrix = []
            silo_ids = sorted(silo_user_distributions.keys())
            
            # 获取所有唯一用户, 同时处理字典和列表类型
            all_users = set()
            for dist in silo_user_distributions.values():
                if isinstance(dist, dict):
                    all_users.update(dist.keys())
                elif isinstance(dist, list):
                    # 如果是列表, 则提取具有非零样本的用户 ID
                    for user_id, count in enumerate(dist):
                        if count > 0:
                            all_users.add(user_id)
            
            # 不限制显示的用户数量, 显示所有用户
            # 注意：用户数量可能非常多, 可能会导致标签重叠或难以读取
            
            all_users = sorted(all_users)
            
            # 如果未找到用户, 则记录警告并返回
            if not all_users:
                logger.warning("No user data found to generate heatmap. Skipping heatmap generation.")
                plt.close() # 如果已创建图形, 则关闭它
                return
            
            for silo_id in silo_ids:
                dist = silo_user_distributions[silo_id]
                
                # 如果是列表, 则转换为字典格式
                if isinstance(dist, list):
                    # 将列表转换为以索引为键的字典
                    dist_dict = {user_id: dist[user_id] if user_id < len(dist) else 0 
                               for user_id in all_users}
                else:
                    dist_dict = dist
                
                total_samples = sum(dist_dict.values()) if dist_dict else 0
                if total_samples > 0:
                    row = [dist_dict.get(user_id, 0) / total_samples * 100 for user_id in all_users]
                else:
                    row = [0.0 for _ in all_users]  # 无样本时使用 0.0
                
                user_matrix.append(row)
            
            # 如果没有矩阵数据, 则创建一个虚拟行
            if not user_matrix:
                user_matrix = [[0.0 for _ in all_users]] # 虚拟矩阵使用 0.0
                
            # 使用更简短的标签格式
            df_users = pd.DataFrame(user_matrix, 
                                index=[f'Silo-{s}' for s in silo_ids] if silo_ids else ['S-0'],
                                columns=[f'User-{i+1}' for i, u in enumerate(all_users)])
            
            # 确保有一些数据用于绘图
            if df_users.empty:
                df_users = pd.DataFrame([[0.0]], index=['Silo-0'], columns=['User-0']) # 虚拟数据帧使用 0.0
            
            plt.subplot(1, 1, 1)
            
            plt.rcParams.update({'font.size': 44}) # 进一步增大全局字体大小，影响颜色条等
            
            try:
                sns.heatmap(df_users,
                        annot=False,  # 由于用户数量可能很多, 移除数值标注
                        fmt='.1f',   # 保留一位小数
                        cmap='Reds',
                        cbar_kws={'label': 'Percentage (%)', 'orientation': 'horizontal'}) # 调整颜色条标签字体大小 (通过rcParams全局设置)
                
                plt.title('User Distribution Across Silos (%)', fontsize=48, pad=20) # 热力图标题保持不变
                plt.xlabel('Users', fontsize=46) # X轴标签字体保持
                plt.ylabel('Silos', fontsize=50) # 增大热力图Y轴图例字体
                
                # 调整标签以适应大量用户，并每隔10个用户显示一个标签
                num_users = len(all_users)
                tick_positions = np.arange(0, num_users, 9)
                tick_labels = [f'User-{all_users[i]+1}' for i in tick_positions if i < num_users]
                
                # 如果没有用户或者tick_positions为空，则使用默认xticks
                if num_users > 0 and len(tick_positions) > 0 :
                    plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=45, ha='center', fontsize=46) # 调整X轴刻度字体和间隔
                else:
                    plt.xticks(rotation=90, ha='center', fontsize=46) # 默认情况

                plt.yticks(fontsize=44)  # Y轴刻度字体保持
                
                # 先进行 tight_layout, 然后再调整底部空间
                plt.tight_layout()
                # 确保图表留出足够空间给x轴标签和颜色条
                plt.subplots_adjust(bottom=0.2) # 减小底部空间以避免 "bottom cannot be >= top" 错误
                plt.savefig(os.path.join(output_dir, f'distribution_heatmaps_{mode}.png'),
                        dpi=150, bbox_inches='tight')
            except Exception as e:
                logger.error(f"Error generating heatmap: {e}")
            finally:
                plt.close()

        def __call__(self):
            """
            执行 PlotFunctions 类中所有可用的绘图函数. 
            当前, 这将调用 `plot_silos_distribution` 和 `plot_user_distribution_heatmap`. 
            """
            self.plot_silos_distribution()
            self.plot_user_distribution_heatmap()

    return PlotFunctions()

def print_info_detail_of_silo(dataset_object, silo_loaders, silo_datasets):
    """
    记录有关跨数据机构和用户的数据分布的详细信息. 

    这包括总体数据集参数、每个数据机构的样本计数、批处理信息、
    每个数据机构中的主要用户、每个数据机构的总唯一用户数以及每个数据机构的标签分布. 

    参数:
        dataset_object (FL.utilis.dataset.Dataset): 包含所有配置和分布信息的主要 Dataset 对象. 
        silo_loaders (dict): 将数据机构 ID 映射到其 PyTorch DataLoader 的字典. 
                             此处用于获取批处理信息. 
        silo_datasets (dict): 将数据机构 ID 映射到其 torch.utils.data.Subset 数据集的字典. 
                              用于标签分布分析. 
    """
    logger.info(f"MODE: {dataset_object.distribute_mode}")
    logger.info(f"Users Number: {dataset_object.n_users}")
    logger.info(f"Silos Number: {dataset_object.n_silos}")
    logger.info(f"Total Data Number: {len(dataset_object.get_whole_train_data())}")
    
    # 计算实际使用的样本数量
    actual_samples = sum(len(indices) for indices in dataset_object.data_indices_of_user.values())
    logger.info(f"Actual Samples Used: {actual_samples}")
    
    if dataset_object.max_samples is not None:
        logger.info(f"Max Samples Limit: {dataset_object.max_samples}")
    
    logger.info(f"Seed: {dataset_object.seed}")
    logger.info('------------------------------------------')

    # 验证数据分布
    for silo_id, loader in silo_loaders.items():
        total = len(loader.dataset)
        logger.info(f"Silo-{silo_id}:")
        logger.info(f"   Total samples: {total}")
        logger.info(f"   Batch number: {len(loader)}")
        # logger.info(f"   示例数据形状: {next(iter(loader))[0].shape}")  # [批大小, 通道数, 高度, 宽度]
        
        # 记录此数据机构的用户直方图
        user_histogram = dataset_object.get_user_histogram(silo_id)
        # 按数据计数获取前 5 位用户
        if isinstance(user_histogram, dict):
            top_users = sorted(user_histogram.items(), key=lambda x: x[1], reverse=True)[:5]
        elif isinstance(user_histogram, list):
            # 如果是列表, 则假定它表示用户计数
            if user_histogram and isinstance(user_histogram[0], tuple):
                # 如果是元组列表 (user_id, count)
                top_users = sorted(user_histogram, key=lambda x: x[1], reverse=True)[:5]
            else:
                # 如果只是计数列表, 则创建通用用户索引
                top_users = [(i, count) for i, count in enumerate(user_histogram)]
                top_users = sorted(top_users, key=lambda x: x[1], reverse=True)[:5]
        else:
            # 意外类型的回退处理
            logger.warning(f"Unexpected user_histogram type: {type(user_histogram)}")
            top_users = []
            
        logger.info(f"   Top 5 users in this silo:")
        for user_id, count in top_users:
            logger.info(f"      User-{user_id}: {count} samples")
        
        # 根据 user_histogram 的类型计算总唯一用户数
        if isinstance(user_histogram, dict):
            total_users = len(user_histogram)
        elif isinstance(user_histogram, list):
            if user_histogram and isinstance(user_histogram[0], tuple):
                total_users = len(user_histogram)
            else:
                total_users = len([count for count in user_histogram if count > 0])
        else:
            total_users = 0
            
        logger.info(f"   Total unique users: {total_users}")

    dataset_train = dataset_object.get_whole_train_data()
    for silo_id, dataset in silo_datasets.items():
        labels = [dataset_train.targets[i].item() for i in dataset.indices]
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f"Silo-{silo_id} Label distribution:")
        for label, count in zip(unique, counts):
            logger.info(f"  Label-{label}: {count}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test data distribution')
    parser.add_argument('--n_users', type=int, default=1000, help='Number of users')
    parser.add_argument('--n_silos', type=int, default=4, help='Number of silos')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seed', type=int, default=114514, help='Random seed')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset name')
    parser.add_argument('--max_samples', type=int, default=10000, help='Maximum number of samples to use')
    parser.add_argument('--plot_type', type=str, default='all', 
                        choices=['all', 'each_silo', 'heatmap'], 
                        help='Type of plot to generate')
    parser.add_argument('--distribute_mode', type=str, default='zipf',
                        choices=['uniform', 'zipf', 'dirichlet', None],
                        help='Distribution mode (if None, all modes will be tested)')
    
    args = parser.parse_args()
    
    n_users = args.n_users
    n_silos = args.n_silos
    batch_size = args.batch_size
    seed = args.seed
    dataset_name = args.dataset
    max_samples = args.max_samples
    plot_type = args.plot_type
    
    # 如果指定了分布模式, 则仅测试该模式
    if args.distribute_mode:
        distribute_mode_list = [args.distribute_mode]
    else:
        distribute_mode_list = ['uniform', 'zipf', 'dirichlet']

    for distribute_mode in distribute_mode_list:
        # 使用所需参数创建 Dataset 对象
        dataset_object = Dataset(
            dataset_name=dataset_name,
            n_silos=n_silos,
            n_users=n_users,
            user_alpha=0.5,
            silo_alpha=2.0,
            seed=seed,
            mode=distribute_mode,
            max_samples=max_samples
        )
        
        # 获取数据集和分布信息
        dataset_train = dataset_object.get_whole_train_data()
        dataset_test = dataset_object.get_whole_test_data()
        
        # 使用 get_silo_dataset 方法创建数据机构数据集
        silo_datasets = {
            silo_id: dataset_object.get_silo_dataset(silo_id)
            for silo_id in range(n_silos)
        }

        # 为每个数据机构创建数据加载器
        silo_loaders = {
            silo_id: DataLoader(dataset,
                               batch_size=batch_size,
                               shuffle=True)  # 每个数据机构独立打乱
            for silo_id, dataset in silo_datasets.items()
        }

        # 从数据集对象获取用户和数据机构索引
        data_indices_of_user = dataset_object.data_indices_of_user
        data_indices_of_silo = dataset_object.silos_indices_of_data

        # 分析并绘制分布图
        plot_functions = analyze_distributions(
            data_indices_of_user, data_indices_of_silo, dataset_train, silo_datasets, distribute_mode, dataset_object)
        
        # 根据用户选择的绘图类型调用相应的函数
        if plot_type == 'all':
            plot_functions()
        if plot_type == 'each_silo':
            plot_functions.plot_silos_distribution()
        elif plot_type == 'heatmap':
            plot_functions.plot_user_distribution_heatmap()
            
        print_info_detail_of_silo(dataset_object, silo_loaders, silo_datasets)

