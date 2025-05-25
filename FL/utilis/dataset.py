import numpy as np
import torch
from FL.utilis.get_data import get_data

class Dataset:
    def __init__(self, dataset_name, n_silos, n_users, user_alpha=0.5, silo_alpha=2.0, seed=1, mode='zipf', max_samples=None):
        """
        初始化数据集并将数据分配到机构和用户 
        参数:
            dataset_name: 数据集名称
            n_silos: 机构数量
            n_users: 用户数量
            user_alpha: 用户数据分布参数
            silo_alpha: 机构数据分布参数
            seed: 随机种子
            mode: 分布模式, 'uniform'(均匀)、'zipf'或'dirichlet'
            max_samples: 最大样本数
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
        获取全部训练数据 
        """
        return self.dataset_train
    
    def get_whole_test_data(self):
        """
        获取全部测试数据 
        """
        return self.dataset_test
    
    def divide_dataset(self):
        """
        将整个训练数据分配到用户, 然后再分配到机构 
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
        将用户数据按均匀、Zipf或Dirichlet分布分配 
        
        参数:
            n_data: 要使用的样本数 如果为None, 则使用所有样本 
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
        将机构数据按均匀、Zipf或Dirichlet分布分配 
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
        将预分配的索引转换为特定客户端的数据集 
        
        参数:
            user_data_indices: 用户或用户集合的数据索引 
        
        返回:
            data: 包含指定索引的数据集 
        """
        data_indices = []
        for user_indices in user_data_indices:
            data_indices.append(user_indices)
        
        # Create subset from list
        data = torch.utils.data.Subset(self.dataset_train, data_indices)
        
        return data
        
    def get_silo_dataset(self, silo_id):
        """
        获取特定机构的数据集 
        
        参数:
            silo_id: 机构的ID 
        
        返回:
            silo_dataset: 指定机构的数据集 
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
        获取用户数据的直方图 
        返回:
            一个列表, 其中索引对应user_id, 值是该用户的样本数量
        """
        user_histogram = [0] * self.n_users
        
        for user_id, indices in self.silos_indices_of_data[silo_id].items():
            user_histogram[user_id] = len(indices)
            
        return user_histogram

    def get_silos_indices(self, silo_id):
        """
        获取特定机构内所有用户的数据索引 

        参数:
            silo_id: 机构的ID 

        返回:
            一个字典, 其中键是user_ids, 值是属于该用户在指定机构内的数据索引列表 
            示例: {user_0: [idx1, idx2, ...], user_1: [idx_a, idx_b, ...]}
        """
        return self.silos_indices_of_data[silo_id]

    def get_silos_weights(self):
        """
        获取机构的权重 
        """
        return self.silos_weights

    def get_user_number(self):
        """
        获取用户数量 
        """
        return self.n_users