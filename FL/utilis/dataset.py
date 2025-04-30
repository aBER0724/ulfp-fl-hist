import numpy as np
import torch
from FL.utilis.get_data import get_data

class Dataset:
    def __init__(self, dataset_name, n_silos, n_users, user_alpha, silo_alpha, seed=1, mode='zipf', max_samples=None):
        """
        Initialize the dataset and divide data into silos and users.
        Params:
            dataset_name: Dataset Name
            n_silos: Silos Number
            n_users: Users Number
            user_alpha: User Data Distribution Parameter
            silo_alpha: Silo Data Distribution Parameter
            seed: Random Seed
            mode: Distribution Mode, 'uniform', 'zipf' or 'dirichlet'
            max_samples: Max samples
        """
        self.seed = seed
        self.dataset_name = dataset_name
        self.n_silos = n_silos
        self.n_users = n_users
        self.user_alpha = user_alpha
        self.silo_alpha = silo_alpha
        self.distribute_mode = mode
        self.max_samples = max_samples
        self.dataset_train, self.dataset_test, self.labels = get_data(dataset_name, self.seed)
        self.users_indices_of_data = []
        self.silos_indices_of_data = {i: {user_id: [] for user_id in range(self.n_users)} for i in range(self.n_silos)}
        self.silos_weights = []
        self.divide_dataset()
    
    def get_whole_train_data(self):
        """
        Get the whole train data.
        """
        return self.dataset_train
    
    def get_whole_test_data(self):
        """
        Get the whole test data.
        """
        return self.dataset_test
    
    def divide_dataset(self):
        """
        Divide the whole train data to users and then to silos.
        """
        n_data = len(self.dataset_train)
        
        # Apply sample limit if specified
        if self.max_samples is not None and self.max_samples < n_data:
            n_data = self.max_samples

        # Distribute data to users
        self.distribute_to_user(n_data)

        # Distribute users data to silos
        self.distribute_to_silo()
    
    def distribute_to_user(self, n_data=None):
        """
        Distribute Users data, Uniform, Zipf or Dirichlet.
        
        Args:
            n_data: Number of samples to use. If None, use all samples.
        """
        np.random.seed(self.seed)
        
        if n_data is None:
            n_data = len(self.dataset_train)
            
        # Create a randomly shuffled list of indices
        all_indices = np.random.permutation(len(self.dataset_train))
        # Select only the first n_data indices
        limited_indices = all_indices[:n_data]
        
        N = self.n_users
        x = np.arange(1, N + 1)

        if self.distribute_mode == 'uniform':
            user_indices_of_data = np.random.permutation(
                int(np.ceil(n_data / N)) * list(range(N))
            )
        else:
            if self.distribute_mode == 'zipf':
                weights = x ** (-self.user_alpha)
                weights /= weights.sum()
            elif self.distribute_mode == 'dirichlet':
                weights = np.random.dirichlet([self.user_alpha] * N)

            user_indices_of_data = np.random.choice(
                    x, size=n_data, replace=True, p=weights)
            user_indices_of_data = user_indices_of_data - 1

        self.data_indices_of_user = {i: [] for i in range(self.n_users)}

        for idx, user_id in enumerate(user_indices_of_data):
            if idx < n_data:
                self.data_indices_of_user[user_id].append(limited_indices[idx])

        for indices in self.data_indices_of_user.values():
            np.random.shuffle(indices)

    def distribute_to_silo(self):
        """
        Distribute Silos data, Uniform, Zipf or Dirichlet.
        """
        np.random.seed(self.seed)

        for user_id, indices in self.data_indices_of_user.items():
            N = self.n_silos
            x = np.arange(1, N + 1)

            if self.distribute_mode == 'uniform':
                user_data_indices_in_silo = np.random.permutation(
                    int(np.ceil(len(indices) / N)) * list(range(N))
                )
            else:
                if self.distribute_mode == 'zipf':
                    weights = x ** (-self.silo_alpha)
                    weights /= weights.sum()
                elif self.distribute_mode == 'dirichlet':
                    weights = np.random.dirichlet([self.silo_alpha] * N)

                user_data_indices_in_silo = np.random.choice(
                    x, size=len(indices), replace=True, p=weights)
                user_data_indices_in_silo = user_data_indices_in_silo - 1


            for silo_id, idx in zip(user_data_indices_in_silo, indices):
                self.silos_indices_of_data[silo_id][user_id].append(idx)
            
        # Silo weight list
        self.silos_weights = []
        for silo_id, silo_indices in self.silos_indices_of_data.items():
            silo_weight = 0
            for user_id, user_indices in silo_indices.items():
                silo_weight += len(user_indices)
            self.silos_weights.append(silo_weight)

    def get_dataset(self, user_data_indices):
        """
        Converting pre-allocated indexes to client-specific datasets.
        
        Params:
            user_data_indices: Data indices for a user or a collection of users.
        
        Returns:
            data: Dataset containing the specified indices.
        """
        data_indices = []
        for user_indices in user_data_indices:
            data_indices.append(user_indices)
        
        # Create subset from list
        data = torch.utils.data.Subset(self.dataset_train, data_indices)
        
        return data
        
    def get_silo_dataset(self, silo_id):
        """
        Get the dataset for a specific silo.
        
        Params:
            silo_id: The ID of the silo.
        
        Returns:
            silo_dataset: Dataset for the specified silo.
        """
        # 获取该silo所有用户的数据索引
        user_data_indices = []
        for user_id, indices in self.silos_indices_of_data[silo_id].items():
            user_data_indices.extend(indices)
        
        # 创建该silo的数据集
        silo_dataset = torch.utils.data.Subset(self.dataset_train, user_data_indices)
        
        return silo_dataset

    def get_user_histogram(self, silo_id):
        """
        Get the histogram of user's data.
        Returns:
            A list where the index corresponds to user_id and the value is the number of samples
        """
        user_histogram = [0] * self.n_users
        
        for user_id, indices in self.silos_indices_of_data[silo_id].items():
            user_histogram[user_id] = len(indices)
            
        return user_histogram

    def get_silos_indices(self, silo_id):
        """
        Get the indices of silos.
        """
        return self.silos_indices_of_data[silo_id]

    def get_silos_weights(self):
        """
        Get the weights of silos.
        """
        return self.silos_weights

    def get_user_number(self):
        """
        Get the number of users.
        """
        return self.n_users