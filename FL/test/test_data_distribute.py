import numpy as np
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import argparse
from FL.utilis.dataset import Dataset

def analyze_distributions(data_indices_of_user, data_indices_of_silo, dataset_train, silo_datasets, mode, dataset_object=None):
    """ 
    Analyze the distribution of users and labels
    Args:
        data_indices_of_user: Dictionary mapping data indices to user IDs
        data_indices_of_silo: Dictionary mapping data indices to silo IDs
        dataset_train: Training dataset
        silo_datasets: Dataset for each silo
        mode: Data distribution mode
        dataset_object: Dataset object containing the get_user_histogram method
    Returns:
        An object containing all plotting functions
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
            Plot the distribution of users and labels for each silo
            """
            n_silos = len(silo_datasets)
            rows = (n_silos + 1) // 2
            plt.figure(figsize=(24, 8 * rows))

            for silo_idx, silo_id in enumerate(silo_datasets.keys()):
                row = silo_idx // 2
                col_base = (silo_idx % 2) * 2

                # User distribution subplot
                plt.subplot(rows, 4, row*4 + col_base + 1)
                user_counts = silo_user_distributions[silo_id]
                top_users = sorted(user_counts.items(),
                                key=lambda x: x[1], reverse=True)[:10]
                users = [f'User-{u[0]}' for u in top_users]
                counts = [u[1] for u in top_users]
                plt.bar(users, counts, color='skyblue')
                plt.title(
                    f'Silo-{silo_id} Top 10 Users\n(Total Users: {len(user_counts)})')
                plt.xlabel('User ID')
                plt.ylabel('Samples')
                plt.xticks(rotation=45)

                # Label distribution subplot
                plt.subplot(rows, 4, row*4 + col_base + 2)
                label_dist = silo_label_distributions[silo_id]
                labels = sorted(label_dist.keys())
                counts = [label_dist[l] for l in labels]
                plt.bar(labels, counts, color='salmon')
                plt.title(
                    f'Silo-{silo_id} Label Distribution\n(Total Samples: {sum(counts)})')
                plt.xlabel('Label')
                plt.ylabel('Samples')
                plt.xticks(range(10), labels, rotation=45, fontsize=8)

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f'silo_distributions_{mode}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        @staticmethod
        def plot_user_distribution_heatmap():
            """
            Plot the heatmap of user distribution.
            """
            plt.figure(figsize=(25, 10))
            user_matrix = []
            silo_ids = sorted(silo_user_distributions.keys())
            all_users = sorted(set().union(*[dist.keys() for dist in silo_user_distributions.values()]))
            
            for silo_id in silo_ids:
                dist = silo_user_distributions[silo_id]
                total_samples = sum(dist.values())
                row = [dist.get(user_id, 0) / total_samples * 100 for user_id in all_users]
                user_matrix.append(row)
            
            df_users = pd.DataFrame(user_matrix, 
                                index=[f'Silo-{s}' for s in silo_ids],
                                columns=[f'User-{u}' for u in all_users])
            
            plt.subplot(1, 1, 1)
            
            plt.rcParams.update({'font.size': 8})
            
            sns.heatmap(df_users, 
                    annot=True,  # Keep the numerical annotations
                    fmt='.1f',   # Keep one decimal place
                    cmap='Purples',  # Change the color scheme to viridis, a perceptually uniform color scheme
                    annot_kws={'size': 6, 'color': 'white'},  # Change the text color to white, making it easier to see in dark areas
                    cbar_kws={'label': 'Percentage (%)'})
            
            plt.title('User Distribution Across Silos (%)', fontsize=14, pad=20)
            plt.xlabel('Users', fontsize=12)
            plt.ylabel('Silos', fontsize=12)
            
            plt.xticks(rotation=90, ha='center', fontsize=6)  # Display the user labels vertically
            plt.yticks(fontsize=10)  # Silo labels remain clear
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'distribution_heatmaps_{mode}.png'),
                    dpi=150, bbox_inches='tight')
            plt.close()

        def __call__(self):
            """
            Call all plotting functions
            """
            self.plot_silos_distribution()
            self.plot_user_distribution_heatmap()

    return PlotFunctions()


def print_detail_of_silo(dataset_object, silo_loaders, silo_datasets):
    """
    Print details about the dataset distribution across silos
    """
    print(f"MODE: {dataset_object.distribute_mode}")
    print(f"Users Number: {dataset_object.n_users}")
    print(f"Silos Number: {dataset_object.n_silos}")
    print(f"Total Data Number: {len(dataset_object.get_whole_train_data())}")
    
    # Calculate the actual number of samples used
    actual_samples = sum(len(indices) for indices in dataset_object.data_indices_of_user.values())
    print(f"Actual Samples Used: {actual_samples}")
    
    if dataset_object.max_samples is not None:
        print(f"Max Samples Limit: {dataset_object.max_samples}")
    
    print(f"Alpha: {dataset_object.alpha}")
    print(f"Seed: {dataset_object.seed}")
    print('------------------------------------------')

    # Verify data distribution
    for silo_id, loader in silo_loaders.items():
        total = len(loader.dataset)
        print(f"Silo-{silo_id}:")
        print(f"   Total samples: {total}")
        print(f"   Batch number: {len(loader)}")
        # print(f"   Example data shape: {next(iter(loader))[0].shape}")  # [Batch_size, Channels, Height, Width]
        
        # Print user histogram for this silo
        user_histogram = dataset_object.get_user_histogram(silo_id)
        # Get top 5 users by data count
        top_users = sorted(user_histogram.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"   Top 5 users in this silo:")
        for user_id, count in top_users:
            print(f"      User-{user_id}: {count} samples")
        print(f"   Total unique users: {len(user_histogram)}")

    dataset_train = dataset_object.get_whole_train_data()
    for silo_id, dataset in silo_datasets.items():
        labels = [dataset_train.targets[i].item() for i in dataset.indices]
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Silo-{silo_id} Label distribution:")
        for label, count in zip(unique, counts):
            print(f"  Label-{label}: {count}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test data distribution')
    parser.add_argument('--n_users', type=int, default=100, help='Number of users')
    parser.add_argument('--n_silos', type=int, default=10, help='Number of silos')
    parser.add_argument('--alpha', type=float, default=0.3, help='Alpha parameter')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--seed', type=int, default=114514, help='Random seed')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset name')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum number of samples to use')
    parser.add_argument('--plot_type', type=str, default='all', 
                        choices=['all', 'each_silo', 'heatmap'], 
                        help='Type of plot to generate')
    parser.add_argument('--distribute_mode', type=str, default=None,
                        choices=['uniform', 'zipf', 'dirichlet', None],
                        help='Distribution mode (if None, all modes will be tested)')
    
    args = parser.parse_args()
    
    n_users = args.n_users
    n_silos = args.n_silos
    alpha = args.alpha
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
            alpha=alpha,
            seed=seed,
            distribute_mode=distribute_mode,
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
            
        print_detail_of_silo(dataset_object, silo_loaders, silo_datasets)
