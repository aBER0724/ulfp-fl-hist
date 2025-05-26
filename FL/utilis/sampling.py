#!/user/bin/python
# 作者：jeff
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class IIDBatchSampler:
    """
    实现一个近似泊松采样的小批量采样器. 
    数据集中的每个样本以概率 p = minibatch_size / dataset_length 被包含在一个批次中. 
    这会导致批次大小可变, 其中预期的批次大小为 `minibatch_size`. 

    属性:
        length (int): 数据集中的样本总数. 
        minibatch_size (int): 每个小批量的目标平均大小. 
        iterations (int): 一个 epoch 中要生成的批次数. 
    """
    def __init__(self, dataset, minibatch_size, iterations):
        """
        初始化 IIDBatchSampler. 

        参数:
            dataset (torch.utils.data.Dataset): 从中采样的数据集. 
            minibatch_size (int): 每个小批量的目标平均大小. 
            iterations (int): 一个 epoch 中要生成的批次数. 
        """
        self.length = len(dataset)
        self.minibatch_size = minibatch_size
        self.iterations = iterations

    def __iter__(self):
        """
        为 `iterations` 次迭代生成一批索引. 
        每个索引以概率 `minibatch_size / length` 被选中. 
        """
        for _ in range(self.iterations):
            # 每个样本以概率 p = minibatch_size / total_samples 被选中. 
            # 这是对每个样本的伯努利试验. 
            # 选中的样本数量服从二项分布 B(n, p). 
            # 对于大的 n 和小的 p, 这近似于泊松分布, 其中 lambda = n*p = minibatch_size. 
            indices = np.where(torch.rand(self.length) < (self.minibatch_size / self.length))[0]
            if indices.size > 0:
                yield indices

    def __len__(self) -> int:
        """返回每个 epoch 的批次数（迭代次数）. """
        return self.iterations


# 该采样每次抽取的数量相等,不放回
class EquallySizedAndIndependentBatchSamplerWithoutReplace:
    """
    实现一个批次采样器, 通过对每个批次从数据集中*不放回*采样索引来生成固定大小的批次. 
    在不同的批次之间, 样本可以重复, 因为每个批次都是独立的采样事件. 

    属性:
        length (int): 数据集中的样本总数. 
        minibatch_size (int): 每个小批量的固定大小. 
        iterations (int): 一个 epoch 中要生成的批次数. 
    """
    def __init__(self, dataset, minibatch_size, iterations):
        """
        初始化采样器. 

        参数:
            dataset (torch.utils.data.Dataset): 从中采样的数据集. 
            minibatch_size (int): 每个小批量的固定大小. 
            iterations (int): 要生成的批次数. 
        """
        self.length = len(dataset)
        self.minibatch_size = minibatch_size
        self.iterations = iterations

    def __iter__(self):
        """
        生成 `iterations` 个批次, 每个批次包含 `minibatch_size` 个唯一索引, 
        这些索引是针对该特定批次从数据集中不放回采样的. 
        """
        for _ in range(self.iterations):
            yield np.random.choice(self.length, self.minibatch_size, replace=False)

    def __len__(self) -> int:
        """返回每个 epoch 的批次数（迭代次数）. """
        return self.iterations

class EquallySizedAndIndependentBatchSamplerWithReplace:
    """
    实现一个批次采样器, 通过对每个批次从数据集中*有放回*采样索引来生成固定大小的批次. 

    属性:
        length (int): 数据集中的样本总数. 
        minibatch_size (int): 每个小批量的固定大小. 
        iterations (int): 一个 epoch 中要生成的批次数. 
    """
    def __init__(self, dataset, minibatch_size, iterations):
        """
        初始化采样器. 

        参数:
            dataset (torch.utils.data.Dataset): 从中采样的数据集. 
            minibatch_size (int): 每个小批量的固定大小. 
            iterations (int): 要生成的批次数. 
        """
        self.length = len(dataset)
        self.minibatch_size = minibatch_size
        self.iterations = iterations

    def __iter__(self):
        """
        生成 `iterations` 个批次, 每个批次包含 `minibatch_size` 个索引, 
        这些索引是从数据集中有放回采样的. 
        """
        for _ in range(self.iterations):
            yield np.random.choice(self.length, self.minibatch_size, replace=True)

    def __len__(self) -> int:
        """返回每个 epoch 的批次数（迭代次数）. """
        return self.iterations


# 三种采样方法均在文章《Privacy Amplification by Subsampling: Tight Analyses via Couplings and Divergences》中有所提及

# 均匀采样不放回（参考《Subsampling Gaussian RDP》）
# 注意：采样高斯机制 (SGM) 通常使用不放回采样. 
# 例如, 参见 "Renyi Differential Privacy of the Sampled Gaussian Mechanism". 
def get_data_loaders_uniform_without_replace(minibatch_size, microbatch_size, iterations, drop_last=True):
    """
    返回一对函数, 用于创建小批量和微批量处理的 DataLoader,
    在小批量级别使用均匀不放回采样. 

    参数:
        minibatch_size (int): 从原始数据集中采样的每个小批量的大小. 
        microbatch_size (int): 从每个小批量创建的微批量的大小. 
        iterations (int): 每个 epoch 采样的小批量数量. 
        drop_last (bool): 如果小批量大小不是 `microbatch_size` 的倍数, 是否丢弃最后一个不完整的微批量. 默认为 True. 

    返回:
        tuple[collections.abc.Callable, collections.abc.Callable]:
            - minibatch_loader (function): 一个函数, 接收一个 `dataset` 并返回一个
                                           用于小批量的 `DataLoader`, 使用
                                           `EquallySizedAndIndependentBatchSamplerWithoutReplace`. 
            - microbatch_loader (function): 一个函数, 接收一个 `minibatch` (一个 TensorDataset)
                                            并返回一个用于微批量的 `DataLoader`. 
    """
    def minibatch_loader(dataset: torch.utils.data.Dataset) -> DataLoader:
        """使用不放回采样为小批量创建 DataLoader. """
        return DataLoader(
            dataset,
            batch_sampler=EquallySizedAndIndependentBatchSamplerWithoutReplace(dataset, minibatch_size, iterations)
        )

    def microbatch_loader(minibatch: TensorDataset) -> DataLoader:
        """从给定的小批量创建微批量的 DataLoader. """
        return DataLoader(
            minibatch,
            batch_size=microbatch_size,
            drop_last=drop_last,
        )
    return minibatch_loader, microbatch_loader

def get_data_loaders_uniform_with_replace(minibatch_size, microbatch_size, iterations, drop_last=True):
    """
    返回一对函数, 用于创建小批量和微批量处理的 DataLoader,
    在小批量级别使用均匀有放回采样. 

    参数:
        minibatch_size (int): 从原始数据集中采样的每个小批量的大小. 
        microbatch_size (int): 从每个小批量创建的微批量的大小. 
        iterations (int): 每个 epoch 采样的小批量数量. 
        drop_last (bool): 是否丢弃最后一个不完整的微批量. 默认为 True. 

    返回:
        tuple[collections.abc.Callable, collections.abc.Callable]:
            - minibatch_loader (function): 一个函数, 接收一个 `dataset` 并返回一个
                                           用于小批量的 `DataLoader`, 使用
                                           `EquallySizedAndIndependentBatchSamplerWithReplace`. 
            - microbatch_loader (function): 一个函数, 接收一个 `minibatch` 并返回
                                            一个用于微批量的 `DataLoader`. 
    """
    def minibatch_loader(dataset):
        """使用有放回采样为小批量创建 DataLoader. """
        return DataLoader(
            dataset,
            batch_sampler=EquallySizedAndIndependentBatchSamplerWithReplace(dataset, minibatch_size, iterations)
        )

    def microbatch_loader(minibatch):
        """从给定的小批量创建微批量的 DataLoader. """
        return DataLoader(
            minibatch,
            batch_size=microbatch_size,
            drop_last=drop_last,
        )
    return minibatch_loader, microbatch_loader

# 泊松采样
def get_data_loaders_possion(minibatch_size: int, microbatch_size: int, iterations: int, drop_last: bool = True): # 潜在拼写错误: possion -> poisson
    """
    返回一对函数, 用于创建小批量和微批量处理的 DataLoader,
    在小批量级别使用类泊松采样 (通过 `IIDBatchSampler`). 
    `IIDBatchSampler` 会产生大小可变的小批量, 其预期大小为 `minibatch_size`. 

    参数:
        minibatch_size (int): 每个小批量的目标平均大小. 
        microbatch_size (int): 从每个小批量创建的微批量的大小. 
        iterations (int): 每个 epoch 采样的小批量数量. 
        drop_last (bool): 是否丢弃最后一个不完整的微批量. 默认为 True. 

    返回:
        tuple[collections.abc.Callable, collections.abc.Callable]:
            - minibatch_loader (function): 一个函数, 接收一个 `dataset` 并返回一个
                                           用于小批量的 `DataLoader`, 使用 `IIDBatchSampler`. 
            - microbatch_loader (function): 一个函数, 接收一个 `minibatch` 并返回
                                            一个用于微批量的 `DataLoader`. 
    """
    def minibatch_loader(dataset):
        """使用 IIDBatchSampler (类泊松) 为小批量创建 DataLoader. """
        return DataLoader(
            dataset,
            batch_sampler=IIDBatchSampler(dataset, minibatch_size, iterations)
        )

    def microbatch_loader(minibatch):
        """从给定的小批量 (可能大小可变) 创建微批量的 DataLoader. """
        return DataLoader(
            minibatch,
            batch_size=microbatch_size,
            drop_last=drop_last,
        )
    return minibatch_loader, microbatch_loader

