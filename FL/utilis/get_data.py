from torchvision import datasets, transforms
import numpy as np

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
        dataset_train = datasets.MNIST('../../datasets/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../../datasets/', train=False, download=True, transform=trans_mnist)
        labels = np.array([data[1] for data in dataset_train])
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    return dataset_train, dataset_test, labels
