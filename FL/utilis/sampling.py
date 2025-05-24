#!/user/bin/python
# author jeff
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class IIDBatchSampler:
    """
    Implements a batch sampler that approximates Poisson sampling.
    Each sample from the dataset is included in a batch with probability p = minibatch_size / dataset_length.
    This results in batches of varying sizes, where the expected batch size is `minibatch_size`.

    Attributes:
        length (int): Total number of samples in the dataset.
        minibatch_size (int): The target average size for each minibatch.
        iterations (int): The number of batches to generate in one epoch.
    """
    def __init__(self, dataset, minibatch_size, iterations):
        """
        Initializes the IIDBatchSampler.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to sample from.
            minibatch_size (int): The target average size for each minibatch.
            iterations (int): The number of batches to generate in one epoch.
        """
        self.length = len(dataset)
        self.minibatch_size = minibatch_size
        self.iterations = iterations

    def __iter__(self):
        """
        Yields a batch of indices for `iterations` times.
        Each index is selected with probability `minibatch_size / length`.
        """
        for _ in range(self.iterations):
            # Each sample is chosen with probability p = minibatch_size / total_samples.
            # This is a Bernoulli trial for each sample.
            # The number of samples chosen follows a Binomial distribution B(n, p).
            # For large n and small p, this approximates a Poisson distribution with lambda = n*p = minibatch_size.
            indices = np.where(torch.rand(self.length) < (self.minibatch_size / self.length))[0]
            if indices.size > 0:
                yield indices

    def __len__(self) -> int:
        """Returns the number of batches (iterations) per epoch."""
        return self.iterations


# 该采样每次抽取的数量相等,不放回
class EquallySizedAndIndependentBatchSamplerWithoutReplace:
    """
    Implements a batch sampler that generates batches of a fixed size by sampling
    indices from the dataset *without* replacement for each batch.
    Across different batches, samples can be repeated as each batch is an independent sampling event.

    Attributes:
        length (int): Total number of samples in the dataset.
        minibatch_size (int): The fixed size for each minibatch.
        iterations (int): The number of batches to generate in one epoch.
    """
    def __init__(self, dataset, minibatch_size, iterations):
        """
        Initializes the sampler.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to sample from.
            minibatch_size (int): The fixed size for each minibatch.
            iterations (int): The number of batches to generate.
        """
        self.length = len(dataset)
        self.minibatch_size = minibatch_size
        self.iterations = iterations

    def __iter__(self):
        """
        Yields `iterations` batches, each containing `minibatch_size` unique indices
        sampled without replacement from the dataset for that specific batch.
        """
        for _ in range(self.iterations):
            yield np.random.choice(self.length, self.minibatch_size, replace=False)

    def __len__(self) -> int:
        """Returns the number of batches (iterations) per epoch."""
        return self.iterations

class EquallySizedAndIndependentBatchSamplerWithReplace:
    """
    Implements a batch sampler that generates batches of a fixed size by sampling
    indices from the dataset *with* replacement for each batch.

    Attributes:
        length (int): Total number of samples in the dataset.
        minibatch_size (int): The fixed size for each minibatch.
        iterations (int): The number of batches to generate in one epoch.
    """
    def __init__(self, dataset, minibatch_size, iterations):
        """
        Initializes the sampler.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to sample from.
            minibatch_size (int): The fixed size for each minibatch.
            iterations (int): The number of batches to generate.
        """
        self.length = len(dataset)
        self.minibatch_size = minibatch_size
        self.iterations = iterations

    def __iter__(self):
        """
        Yields `iterations` batches, each containing `minibatch_size` indices
        sampled with replacement from the dataset.
        """
        for _ in range(self.iterations):
            yield np.random.choice(self.length, self.minibatch_size, replace=True)

    def __len__(self) -> int:
        """Returns the number of batches (iterations) per epoch."""
        return self.iterations


#三种采样都有的文章《Privacy Amplification by Subsampling: Tight Analyses via Couplings and Divergences》

#均匀采样不放回（《Subsamlping Gaussian RDP》）
# Note: The Sampled Gaussian Mechanism (SGM) typically uses sampling without replacement.
# See, e.g., "Renyi Differential Privacy of the Sampled Gaussian Mechanism".
def get_data_loaders_uniform_without_replace(minibatch_size, microbatch_size, iterations, drop_last=True):
    """
    Returns a pair of functions to create DataLoaders for minibatch and microbatch processing,
    using uniform sampling without replacement at the minibatch level.

    Args:
        minibatch_size (int): The size of each minibatch sampled from the original dataset.
        microbatch_size (int): The size of microbatches to be created from each minibatch.
        iterations (int): The number of minibatches to sample per epoch.
        drop_last (bool): Whether to drop the last incomplete microbatch if the minibatch
                          size is not a multiple of `microbatch_size`. Defaults to True.

    Returns:
        tuple[collections.abc.Callable, collections.abc.Callable]:
            - minibatch_loader (function): A function that takes a `dataset` and returns a
                                           `DataLoader` for minibatches, using
                                           `EquallySizedAndIndependentBatchSamplerWithoutReplace`.
            - microbatch_loader (function): A function that takes a `minibatch` (a TensorDataset)
                                            and returns a `DataLoader` for microbatches.
    """
    def minibatch_loader(dataset: torch.utils.data.Dataset) -> DataLoader:
        """Creates a DataLoader for minibatches using sampling without replacement."""
        return DataLoader(
            dataset,
            batch_sampler=EquallySizedAndIndependentBatchSamplerWithoutReplace(dataset, minibatch_size, iterations)
        )

    def microbatch_loader(minibatch: TensorDataset) -> DataLoader:
        """Creates a DataLoader for microbatches from a given minibatch."""
        return DataLoader(
            minibatch,
            batch_size=microbatch_size,
            drop_last=drop_last,
        )
    return minibatch_loader, microbatch_loader

def get_data_loaders_uniform_with_replace(minibatch_size, microbatch_size, iterations, drop_last=True):
    """
    Returns a pair of functions to create DataLoaders for minibatch and microbatch processing,
    using uniform sampling with replacement at the minibatch level.

    Args:
        minibatch_size (int): The size of each minibatch sampled from the original dataset.
        microbatch_size (int): The size of microbatches to be created from each minibatch.
        iterations (int): The number of minibatches to sample per epoch.
        drop_last (bool): Whether to drop the last incomplete microbatch. Defaults to True.

    Returns:
        tuple[collections.abc.Callable, collections.abc.Callable]:
            - minibatch_loader (function): A function that takes a `dataset` and returns a
                                           `DataLoader` for minibatches, using
                                           `EquallySizedAndIndependentBatchSamplerWithReplace`.
            - microbatch_loader (function): A function that takes a `minibatch` and returns
                                            a `DataLoader` for microbatches.
    """
    def minibatch_loader(dataset):
        """Creates a DataLoader for minibatches using sampling with replacement."""
        return DataLoader(
            dataset,
            batch_sampler=EquallySizedAndIndependentBatchSamplerWithReplace(dataset, minibatch_size, iterations)
        )

    def microbatch_loader(minibatch):
        """Creates a DataLoader for microbatches from a given minibatch."""
        return DataLoader(
            minibatch,
            batch_size=microbatch_size,
            drop_last=drop_last,
        )
    return minibatch_loader, microbatch_loader

# Poisson
def get_data_loaders_possion(minibatch_size: int, microbatch_size: int, iterations: int, drop_last: bool = True): # Potential typo: possion -> poisson
    """
    Returns a pair of functions to create DataLoaders for minibatch and microbatch processing,
    using Poisson-like sampling (via `IIDBatchSampler`) at the minibatch level.
    The `IIDBatchSampler` results in variable-sized minibatches with an expected size of `minibatch_size`.

    Args:
        minibatch_size (int): The target average size for each minibatch.
        microbatch_size (int): The size of microbatches to be created from each minibatch.
        iterations (int): The number of minibatches to sample per epoch.
        drop_last (bool): Whether to drop the last incomplete microbatch. Defaults to True.

    Returns:
        tuple[collections.abc.Callable, collections.abc.Callable]:
            - minibatch_loader (function): A function that takes a `dataset` and returns a
                                           `DataLoader` for minibatches, using `IIDBatchSampler`.
            - microbatch_loader (function): A function that takes a `minibatch` and returns
                                            a `DataLoader` for microbatches.
    """
    def minibatch_loader(dataset):
        """Creates a DataLoader for minibatches using IIDBatchSampler (Poisson-like)."""
        return DataLoader(
            dataset,
            batch_sampler=IIDBatchSampler(dataset, minibatch_size, iterations)
        )

    def microbatch_loader(minibatch):
        """Creates a DataLoader for microbatches from a given (potentially variable-sized) minibatch."""
        return DataLoader(
            minibatch,
            batch_size=microbatch_size,
            drop_last=drop_last,
        )
    return minibatch_loader, microbatch_loader

