from torchvision import datasets, transforms
import numpy as np
import torch
from torch.utils.data import TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(message)s', datefmt='%Y-%m-%d %H:%M')
logger = logging.getLogger(__name__)


def get_data(dataset_name: str, seed: int = 1) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, np.ndarray]:
    """
    Loads and preprocesses the specified dataset (MNIST or Credit Card Fraud).

    For MNIST, it downloads the dataset, applies transformations (ToTensor, Normalize).
    For Credit Card, it loads from a local CSV, performs train/test split, scales features,
    and applies downsampling to the majority class in the training set.

    Args:
        dataset_name (str): The name of the dataset to load.
                            Supported values: 'mnist', 'creditcard'.
        seed (int, optional): Random seed used for reproducibility, particularly for
                              train/test split and downsampling in the credit card dataset.
                              Defaults to 1.

    Returns:
        tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, np.ndarray]:
            - dataset_train: The training dataset (torch.utils.data.Dataset).
            - dataset_test: The test dataset (torch.utils.data.Dataset).
            - labels: A NumPy array of labels from the training dataset.

    Raises:
        FileNotFoundError: If the credit card dataset CSV is not found at the expected path.
        ValueError: If an unsupported `dataset_name` is provided.
    """
    if dataset_name == 'mnist':
        # transforms.Compose(): Combined multiple operations.
        # transforms.ToTensor(): Convert images to tensor type.
        # transforms.Normalize(): Normalize.
        trans_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Download MNIST dataset to `datasets`. Using trans_mnist to transform MNIST images to 2 channels.
        dataset_train = datasets.MNIST('datasets/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('datasets/', train=False, download=True, transform=trans_mnist)
        labels = np.array([data[1] for data in dataset_train])
    elif dataset_name == 'creditcard':
        # 检查信用卡数据集文件是否存在
        creditcard_path = 'datasets/CreditCard'
        if not os.path.exists(creditcard_path):
            error_msg = (
                f"Credit Card Dataset do not exist: {creditcard_path}\n"
                f"Download dataset and put file to {creditcard_path}\n"
                f"Download URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
            )
            raise FileNotFoundError(error_msg)
        
        random_state = np.random.RandomState(seed=seed)

        df = pd.read_csv(os.path.join(creditcard_path, "creditcard.csv"))
        X = df.drop("Class", axis=1)
        y = df["Class"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=random_state, stratify=y
        )

        # scale the values of x (better training)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.FloatTensor(y_train.values)
        y_test = torch.FloatTensor(y_test.values)

        percentage = 0.1  # 下采样
        majority_label = 0
        target_X_train = X_train[y_train == majority_label]
        target_y_train = y_train[y_train == majority_label]
        non_target_X_train = X_train[y_train != majority_label]
        non_target_y_train = y_train[y_train != majority_label]
        limit_size = int(len(target_X_train) * percentage)
        selected_indices = random_state.choice(
            len(target_X_train), size=limit_size, replace=False
        )
        target_X_train = target_X_train[selected_indices]
        target_y_train = target_y_train[selected_indices]
        X_train = torch.concatenate([non_target_X_train, target_X_train])
        y_train = torch.concatenate([non_target_y_train, target_y_train])
        dataset_train = TensorDataset(X_train, y_train)

        dataset_train = TensorDataset(X_train, y_train)
        dataset_test = TensorDataset(X_test, y_test)
        labels = dataset_train.tensors[1].numpy()
    
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    return dataset_train, dataset_test, labels

if __name__ == "__main__":
    
    # dataset_name = 'mnist'
    dataset_name = 'creditcard'
    
    dataset_train, dataset_test, labels = get_data(dataset_name)
    print(f"训练集大小: {len(dataset_train)} 样本")
    print(f"测试集大小: {len(dataset_test)} 样本")