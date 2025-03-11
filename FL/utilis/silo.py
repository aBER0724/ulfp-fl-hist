import math
import copy
import torch
import torch.nn.functional as F
from opacus import PrivacyEngine
import torch.nn as nn
from FL.utilis.sampling import get_data_loaders_uniform_without_replace

class Silo:
    def __init__(self, silo_id, dataset, model, learning_rate, q_for_batch_size, epoch_num, is_secure='ndp', epsilon=50.0, delta=1e-5, max_grad_norm=1.2):
        self.id = silo_id
        self.model = copy.deepcopy(model)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.silo_data = dataset.get_silo_dataset(silo_id)
        self.user_histogram = dataset.get_user_histogram(silo_id)
        self.epoch_num = epoch_num
        self.minibatch_size = math.floor(
            len(self.silo_data) * q_for_batch_size)  # data -> mini_batch
        self.microbatch_size = 1  # mini_batch -> micro_batch
        # number of batch (set 1 for every training will sample 1 Lot)
        self.iterations = 1
        self.minibatch_loader, self.microbatch_loader = get_data_loaders_uniform_without_replace(
            self.minibatch_size, self.microbatch_size, self.iterations)
        self.train_loader = self.minibatch_loader(self.silo_data)
        self.acc_list = []
        self.loss_list = []
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.train_mode = is_secure
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm

        if self.train_mode == 'dp':
            privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.train_loader = privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=1.1,
                max_grad_norm=self.max_grad_norm
            )
            privacy_engine.accountant.set_target_epsilon(target_epsilon=self.epsilon, target_delta=self.delta)

    def get_model(self):
        """
        Get the local model.
        """
        return self.model

    def get_user_histogram(self):
        """
        Get the histogram of user's data.
        """
        return self.user_histogram

    def local_silos_sgd(self):
        """
        silo local stochastic gradient descent.
        Returns:
            avg_epoch_loss: Average loss of all silos
            avg_epoch_acc: Average accuracy of all silos
        """
        total_loss = 0
        total_acc = 0
        
        for epoch in range(self.epoch_num):

            epoch_loss, epoch_acc = self.train(self.train_loader)

            print(
                f"Silo {self.id}, Epoch {epoch}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%")

            total_loss += epoch_loss
            total_acc += epoch_acc

        avg_loss = total_loss / self.epoch_num
        avg_acc = total_acc / self.epoch_num
        
        self.loss_list.append(avg_loss)
        self.acc_list.append(avg_acc)

        return avg_loss, avg_acc

    def train(self, train_loader):
        """
        Local training with noise addition
        Args:
            model: Model
            train_loader: Training data loader
            optimizer: Optimizer
        Returns:
            train_loss: Average training loss
            train_acc: Average training accuracy
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for id, (data, target) in enumerate(train_loader):
            self.optimizer.zero_grad()  # Empty the gradient

            # Move the data to the correct device (CPU/GPU)
            data = data.to(self.device, dtype=torch.float32)
            target = target.to(self.device, dtype=torch.long)

            output = self.model(data)
            loss = F.cross_entropy(output, target)  # Loss function

            # Backward propagation
            loss.backward()
            self.optimizer.step()

            # Accumulate the loss
            total_loss += loss.item()

            # Calculate the accuracy
            # Find the index of the maximum value
            pred = output.argmax(dim=1, keepdim=True)
            # Count the number of correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += target.size(0)

        # Calculate the average loss and accuracy
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = correct / total_samples * 100.0

        return avg_loss, avg_accuracy
