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
    Analyzes and prepares data for visualizing the distribution of data samples
    across users and silos, as well as label distributions within silos.

    Args:
        data_indices_of_user (dict): A dictionary mapping user IDs to a list of
                                     data indices assigned to that user.
                                     Example: {user_id: [idx1, idx2, ...]}
        data_indices_of_silo (dict): A nested dictionary mapping silo IDs to user IDs,
                                     and then to a list of data indices.
                                     Example: {silo_id: {user_id: [idx_a, idx_b, ...]}}
                                     Alternatively, if `dataset_object` is provided, this might
                                     not be directly used if `get_user_histogram` is preferred.
        dataset_train (torch.utils.data.Dataset): The full training dataset, used to get labels.
        silo_datasets (dict): A dictionary mapping silo IDs to their respective
                              torch.utils.data.Subset datasets.
                              Example: {silo_id: <Subset object>}
        mode (str): A string identifier for the data distribution mode (e.g., 'zipf', 'uniform'),
                    used for naming output files.
        dataset_object (FL.utilis.dataset.Dataset, optional): An instance of the Dataset class.
                               If provided, its `get_user_histogram` method is used to
                               determine user distribution per silo, which can be more direct
                               than recalculating from `data_indices_of_silo`.

    Returns:
        PlotFunctions: An instance of the inner PlotFunctions class, which contains methods
                       to generate and save the distribution plots.
    """

    # Create a dictionary mapping data indices to user IDs
    data_to_user = {}
    for user_id, indices in data_indices_of_user.items():
        for idx in indices:
            data_to_user[idx] = user_id

    # 1. Count the user distribution for each silo
    silo_user_distributions = defaultdict(dict)
    
    if dataset_object is not None:
        # Use the new get_user_histogram method if dataset_object is provided
        for silo_id in silo_datasets.keys():
            silo_user_distributions[silo_id] = dataset_object.get_user_histogram(silo_id)
    else:
        # Fall back to manual calculation if dataset_object is not provided
        for silo_id, silo_indices in data_indices_of_silo.items():
            user_counts = defaultdict(int)
            for idx in silo_indices:
                user_id = data_to_user[idx]
                user_counts[user_id] += 1
            silo_user_distributions[silo_id] = user_counts

    # 2. Label distribution
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
            Generates and saves a plot showing the top N user distributions (sample counts)
            and label distributions for each silo.

            The plot will display multiple subplots, with each silo having two:
            one for user sample counts and one for label counts.
            """
            n_silos = len(silo_datasets)
            silos_per_row = 4  # Increased from 2 to 4 silos per row
            rows = (n_silos + silos_per_row - 1) // silos_per_row
            plt.figure(figsize=(24, 6 * rows))  # Adjusted height per row

            for silo_idx, silo_id in enumerate(silo_datasets.keys()):
                row = silo_idx // silos_per_row
                col_base = (silo_idx % silos_per_row) * 2

                # User distribution subplot
                plt.subplot(rows, silos_per_row * 2, row * silos_per_row * 2 + col_base + 1)
                user_counts = silo_user_distributions[silo_id]
                
                # Handle the case where user_counts could be a list or dictionary
                if isinstance(user_counts, dict):
                    top_users = sorted(user_counts.items(),
                                key=lambda x: x[1], reverse=True)[:10]
                    users = [f'User-{u[0]}' for u in top_users]
                    counts = [u[1] for u in top_users]
                elif isinstance(user_counts, list):
                    # If it's a list, assume it's already sorted or format it appropriately
                    top_users = user_counts[:10]
                    if len(top_users) > 0 and isinstance(top_users[0], tuple):
                        # If it's a list of tuples (user_id, count)
                        users = [f'User-{u[0]}' for u in top_users]
                        counts = [u[1] for u in top_users]
                    else:
                        # If it's just a list of counts, create generic labels
                        users = [f'User-{i}' for i in range(len(top_users))]
                        counts = top_users
                else:
                    # Fallback for unexpected type
                    users = []
                    counts = []
                
                plt.bar(users, counts, color='skyblue')
                plt.title(
                    f'Silo-{silo_id} Top 10 Users\n(Total Users: {len(user_counts) if hasattr(user_counts, "__len__") else "Unknown"})',
                    fontsize=14)
                plt.xlabel('User ID', fontsize=12)
                plt.ylabel('Samples', fontsize=12)
                plt.xticks(rotation=45, fontsize=10)
                plt.yticks(fontsize=10)

                # Label distribution subplot
                plt.subplot(rows, silos_per_row * 2, row * silos_per_row * 2 + col_base + 2)
                label_dist = silo_label_distributions[silo_id]
                labels = sorted(label_dist.keys())
                counts = [label_dist[l] for l in labels]
                plt.bar(labels, counts, color='salmon')
                plt.title(
                    f'Silo-{silo_id} Label Distribution\n(Total Samples: {sum(counts)})',
                    fontsize=14)
                plt.xlabel('Label', fontsize=12)
                plt.ylabel('Samples', fontsize=12)
                plt.xticks(range(10), labels, rotation=45, fontsize=10)
                plt.yticks(fontsize=10)

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f'silo_distributions_{mode}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        @staticmethod
        def plot_user_distribution_heatmap():
            """
            Generates and saves a heatmap visualizing the percentage of each user's data
            (or sample count contribution) within each silo.

            The heatmap shows silos on the y-axis and users on the x-axis.
            Cell values represent the proportion of a user's samples within a specific silo.
            """
            # 大幅增加图形尺寸以适应可能的大量用户
            plt.figure(figsize=(60, 20))  # 显著增加宽度以容纳所有用户标签
            user_matrix = []
            silo_ids = sorted(silo_user_distributions.keys())
            
            # Get all unique users while handling both dict and list types
            all_users = set()
            for dist in silo_user_distributions.values():
                if isinstance(dist, dict):
                    all_users.update(dist.keys())
                elif isinstance(dist, list):
                    # If it's a list, extract user IDs with non-zero samples
                    for user_id, count in enumerate(dist):
                        if count > 0:
                            all_users.add(user_id)
            
            # 不限制显示的用户数量，显示所有用户
            # 注意：用户数量可能非常多，可能会导致标签重叠或难以读取
            
            all_users = sorted(all_users)
            
            # If no users found, log a warning and return
            if not all_users:
                logger.warning("No user data found to generate heatmap. Skipping heatmap generation.")
                plt.close() # Close the figure if it was created
                return
            
            for silo_id in silo_ids:
                dist = silo_user_distributions[silo_id]
                
                # Convert to dict format if it's a list
                if isinstance(dist, list):
                    # Convert list to dict with index as key
                    dist_dict = {user_id: dist[user_id] if user_id < len(dist) else 0 
                               for user_id in all_users}
                else:
                    dist_dict = dist
                
                total_samples = sum(dist_dict.values()) if dist_dict else 0
                if total_samples > 0:
                    row = [dist_dict.get(user_id, 0) / total_samples * 100 for user_id in all_users]
                else:
                    row = [0.0 for _ in all_users]  # Use 0.0 for no samples
                
                user_matrix.append(row)
            
            # If no matrix data, create a dummy row
            if not user_matrix:
                user_matrix = [[0.0 for _ in all_users]] # Use 0.0 for dummy matrix
                
            # 使用更简短的标签格式
            df_users = pd.DataFrame(user_matrix, 
                                index=[f'Silo-{s}' for s in silo_ids] if silo_ids else ['S-0'],
                                columns=[f'User-{i+1}' for i, u in enumerate(all_users)])
            
            # Ensure there's some data for plotting
            if df_users.empty:
                df_users = pd.DataFrame([[0.0]], index=['Silo-0'], columns=['User-0']) # Use 0.0 for dummy dataframe
            
            plt.subplot(1, 1, 1)
            
            plt.rcParams.update({'font.size': 14})
            
            try:
                sns.heatmap(df_users, 
                        annot=False,  # 由于用户数量可能很多，移除数值标注
                        fmt='.1f',   # Keep one decimal place
                        cmap='Reds',
                        cbar_kws={'label': 'Percentage (%)'})
                
                plt.title('User Distribution Across Silos (%)', fontsize=16, pad=20)
                plt.xlabel('Users', fontsize=20)
                plt.ylabel('Silos', fontsize=20)
                
                # 调整标签以适应大量用户
                plt.xticks(rotation=90, ha='center', fontsize=8)  # 减小字体，垂直放置标签
                plt.yticks(fontsize=14)  # 调整y轴标签大小
                
                # 先进行tight_layout，然后再调整底部空间
                plt.tight_layout()
                # 确保图表留出足够空间给x轴标签
                plt.subplots_adjust(bottom=0.4)
                plt.savefig(os.path.join(output_dir, f'distribution_heatmaps_{mode}.png'),
                        dpi=150, bbox_inches='tight')
            except Exception as e:
                logger.error(f"Error generating heatmap: {e}")
            finally:
                plt.close()

        def __call__(self):
            """
            Executes all available plotting functions in the PlotFunctions class.
            Currently, this calls `plot_silos_distribution` and `plot_user_distribution_heatmap`.
            """
            self.plot_silos_distribution()
            self.plot_user_distribution_heatmap()

    return PlotFunctions()

def print_info_detail_of_silo(dataset_object, silo_loaders, silo_datasets):
    """
    Logs detailed information about the data distribution across silos and users.

    This includes overall dataset parameters, per-silo sample counts, batch information,
    top users within each silo, total unique users per silo, and label distributions per silo.

    Args:
        dataset_object (FL.utilis.dataset.Dataset): The main Dataset object containing
                                                     all configuration and distribution information.
        silo_loaders (dict): A dictionary mapping silo IDs to their PyTorch DataLoaders.
                             Used here to get batch information.
        silo_datasets (dict): A dictionary mapping silo IDs to their torch.utils.data.Subset
                              datasets. Used for label distribution analysis.
    """
    logger.info(f"MODE: {dataset_object.distribute_mode}")
    logger.info(f"Users Number: {dataset_object.n_users}")
    logger.info(f"Silos Number: {dataset_object.n_silos}")
    logger.info(f"Total Data Number: {len(dataset_object.get_whole_train_data())}")
    
    # Calculate the actual number of samples used
    actual_samples = sum(len(indices) for indices in dataset_object.data_indices_of_user.values())
    logger.info(f"Actual Samples Used: {actual_samples}")
    
    if dataset_object.max_samples is not None:
        logger.info(f"Max Samples Limit: {dataset_object.max_samples}")
    
    logger.info(f"Seed: {dataset_object.seed}")
    logger.info('------------------------------------------')

    # Verify data distribution
    for silo_id, loader in silo_loaders.items():
        total = len(loader.dataset)
        logger.info(f"Silo-{silo_id}:")
        logger.info(f"   Total samples: {total}")
        logger.info(f"   Batch number: {len(loader)}")
        # logger.info(f"   Example data shape: {next(iter(loader))[0].shape}")  # [Batch_size, Channels, Height, Width]
        
        # logger.info user histogram for this silo
        user_histogram = dataset_object.get_user_histogram(silo_id)
        # Get top 5 users by data count
        if isinstance(user_histogram, dict):
            top_users = sorted(user_histogram.items(), key=lambda x: x[1], reverse=True)[:5]
        elif isinstance(user_histogram, list):
            # If it's a list, assume it represents user counts
            if user_histogram and isinstance(user_histogram[0], tuple):
                # If it's a list of tuples (user_id, count)
                top_users = sorted(user_histogram, key=lambda x: x[1], reverse=True)[:5]
            else:
                # If it's just a list of counts, create generic user indices
                top_users = [(i, count) for i, count in enumerate(user_histogram)]
                top_users = sorted(top_users, key=lambda x: x[1], reverse=True)[:5]
        else:
            # Fallback for unexpected type
            logger.warning(f"Unexpected user_histogram type: {type(user_histogram)}")
            top_users = []
            
        logger.info(f"   Top 5 users in this silo:")
        for user_id, count in top_users:
            logger.info(f"      User-{user_id}: {count} samples")
        
        # Calculate total unique users based on the type of user_histogram
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
    
    # If a distribution mode is specified, only test that mode
    if args.distribute_mode:
        distribute_mode_list = [args.distribute_mode]
    else:
        distribute_mode_list = ['uniform', 'zipf', 'dirichlet']

    for distribute_mode in distribute_mode_list:
        # Create a Dataset object with the desired parameters
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
        
        # Get the dataset and distribution information
        dataset_train = dataset_object.get_whole_train_data()
        dataset_test = dataset_object.get_whole_test_data()
        
        # Create silo datasets using the get_silo_dataset method
        silo_datasets = {
            silo_id: dataset_object.get_silo_dataset(silo_id)
            for silo_id in range(n_silos)
        }

        # Create data loaders for each silo
        silo_loaders = {
            silo_id: DataLoader(dataset,
                               batch_size=batch_size,
                               shuffle=True)  # Each silo independently shuffles
            for silo_id, dataset in silo_datasets.items()
        }

        # Get the user and silo indices from the dataset object
        data_indices_of_user = dataset_object.data_indices_of_user
        data_indices_of_silo = dataset_object.silos_indices_of_data

        # Analyze and plot the distributions
        plot_functions = analyze_distributions(
            data_indices_of_user, data_indices_of_silo, dataset_train, silo_datasets, distribute_mode, dataset_object)
        
        # Call the appropriate function based on the user's choice of plot type
        if plot_type == 'all':
            plot_functions()
        if plot_type == 'each_silo':
            plot_functions.plot_silos_distribution()
        elif plot_type == 'heatmap':
            plot_functions.plot_user_distribution_heatmap()
            
        print_info_detail_of_silo(dataset_object, silo_loaders, silo_datasets)
