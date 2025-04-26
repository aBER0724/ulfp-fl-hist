from torchvision import datasets, transforms
import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import shutil
import urllib.request

# Try importing flamby, raise error if not installed
try:
    from flamby.datasets.fed_heart_disease import FedHeartDisease
except ImportError:
    raise ImportError("The 'flamby' package is required for the heartdisease dataset. Please install it, e.g., 'pip install flamby'")

def get_data(dataset_name):
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
        creditcard_path = 'datasets/CreditCard/creditcard.csv'
        if not os.path.exists(creditcard_path):
            error_msg = (
                f"Credit Card Dataset do not exist: {creditcard_path}\n"
                f"Download dataset and put file to {creditcard_path}\n"
                f"Download URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
            )
            raise FileNotFoundError(error_msg)
        
        # 加载数据集
        df = pd.read_csv(creditcard_path)
            
        # Handle imbalanced data
        fraud = df[df['Class'] == 1]
        non_fraud = df[df['Class'] == 0].sample(n=len(fraud)*5, random_state=42)
        df_balanced = pd.concat([fraud, non_fraud])
        
        # Split features and target
        X = df_balanced.drop('Class', axis=1).values
        y = df_balanced['Class'].values
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create PyTorch datasets
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
        dataset_test = TensorDataset(X_test_tensor, y_test_tensor)
        labels = y_train
    elif dataset_name == 'heart_disease':
        print("Loading Heart Disease dataset using FLamby...")
        # FLamby handles download and preprocessing automatically.
        # It loads data into memory as PyTorch datasets.
        # 'pooled=True' gets the entire dataset, which we'll split later if needed by the framework structure.
        # Note: Flamby might internally handle train/test splits differently depending on the dataset.
        # FedHeartDisease(pooled=True) returns the *entire* dataset (train+test). We need both.
        try:
            # pooled=True combines all centers' data
            dataset_train = FedHeartDisease(train=True, pooled=True)
            dataset_test = FedHeartDisease(train=False, pooled=True)
            
            # Extract labels from the dataset
            if hasattr(dataset_train, 'labels'):
                # FedHeartDisease has a 'labels' attribute
                labels = dataset_train.labels
                if isinstance(labels, torch.Tensor):
                    labels = labels.numpy()
                else:
                    labels = np.array(labels)
            elif hasattr(dataset_train, 'targets'):
                # If targets is a tensor, convert to numpy
                if isinstance(dataset_train.targets, torch.Tensor):
                    labels = dataset_train.targets.numpy()
                else:
                    labels = np.array(dataset_train.targets) # Fallback if it's a list/other sequence
            elif hasattr(dataset_train, '_targets'): # Some datasets use _targets
                if isinstance(dataset_train._targets, torch.Tensor):
                    labels = dataset_train._targets.numpy()
                else:
                    labels = np.array(dataset_train._targets)
            else:
                # If no direct targets attribute, try iterating (less efficient)
                print("Warning: Could not find labels attribute, iterating to get labels.")
                labels = np.array([y for _, y in dataset_train])
             
            print("FLamby Heart Disease dataset loaded successfully.")

        except Exception as e:
            raise RuntimeError(f"Failed to load Flamby Heart Disease dataset: {e}\n" 
                               f"Ensure 'flamby' is installed and data can be downloaded (check network/permissions).")

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    return dataset_train, dataset_test, labels
