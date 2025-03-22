import math
import copy
import torch
import torch.nn.functional as F
from opacus import PrivacyEngine
import torch.nn as nn
from FL.utilis.sampling import get_data_loaders_uniform_without_replace


class Silo:
    def __init__(self, silo_id, server, dataset, model, learning_rate, q_for_batch_size, epoch_num, is_secure='ndp', is_clip=False, epsilon=5.0, delta=1e-5, sigma=1.1, max_grad_norm=3.0, max_data_number=1000):
        self.id = silo_id
        self.server = server
        self.model = copy.deepcopy(model)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.silo_data_indices = dataset.get_silos_indices(silo_id)
        self.user_histogram = dataset.get_user_histogram(silo_id)
        self.user_weight_list = [1 for _ in range(len(self.user_histogram))]
        self.epoch_num = epoch_num
        self.learning_rate = learning_rate
        self.iterations = 1
        self.acc_list = []
        self.loss_list = []
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.train_mode = is_secure
        self.max_grad_norm = max_grad_norm
        self.epsilon = epsilon
        self.delta = delta
        self.sigma = sigma
        self.dataset = dataset

        indices_arr = []
        for user_id, data in self.silo_data_indices.items():
            indices_arr.extend(data)

        self.silo_data = dataset.get_dataset(indices_arr)
        self.microbatch_size = 1  # mini_batch -> micro_batch
        self.minibatch_size = math.floor(
            len(self.silo_data) * q_for_batch_size)  # data -> mini_batch
        self.minibatch_loader, self.microbatch_loader = get_data_loaders_uniform_without_replace(
            self.minibatch_size, self.microbatch_size, self.iterations)
        self.train_loader = self.minibatch_loader(self.silo_data)

        if self.train_mode == 'cldp':
            privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.train_loader = privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                epochs=self.epoch_num,
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                max_grad_norm=self.max_grad_norm,
            )

    def get_model(self):
        """
        Get the local model.
        """
        return self.model

    def receive_global_model(self, global_model):
        """
        Receive the global model from the server.
        """
        self.model = copy.deepcopy(global_model)

    def get_user_histogram(self):
        """
        Get the histogram of user's data.
        """
        return self.user_histogram

    def receive_user_histogram(self, sumed_user_histogram):
        """
        Get the weight of the user.
        """
        for i in range(len(sumed_user_histogram)):
            self.user_weight_list[i] = self.user_histogram[i] / sumed_user_histogram[i]
    
        return self.user_weight_list

    def clip_user_data(self, max_data_number):
        """
        Limit the number of data in the user's data.
        """
        if len(self.silo_data) > max_data_number:
            self.silo_data = self.silo_data[:max_data_number]
        return self.silo_data

    def local_silos_sgd(self, global_model):
        """
        silo local stochastic gradient descent.
        Returns:
            avg_epoch_loss: Average loss of all silos
            avg_epoch_acc: Average accuracy of all silos
        """
        original_model = copy.deepcopy(global_model)
        total_loss = 0
        total_acc = 0
        avg_loss = 0
        avg_acc = 0

        if self.train_mode == 'uldp':
            users = self.silo_data_indices
            user_model_list = [copy.deepcopy(self.model)
                               for _ in range(len(users))]

            for user_id, user_data_indices in users.items():
                user_data = self.dataset.get_dataset(user_data_indices)
                user_loader = torch.utils.data.DataLoader(
                    user_data, batch_size=self.microbatch_size, shuffle=True
                )

                user_model = user_model_list[user_id]
                user_optimizer = torch.optim.SGD(
                    user_model.parameters(), lr=self.learning_rate)

                user_avg_loss, user_avg_acc = self.train(
                    user_model, user_loader, user_optimizer)

                print(
                    f"Silo {self.id}, User {user_id}: Loss = {user_avg_loss:.4f}, Accuracy = {user_avg_acc:.2f}%")

                total_loss += user_avg_loss
                total_acc += user_avg_acc

            avg_loss = total_loss / len(users)
            avg_acc = total_acc / len(users)

            with torch.no_grad():
                for key in self.model.state_dict():
                    self.model.state_dict()[key].copy_(
                        original_model.state_dict()[key])

                    for i, user_model in enumerate(user_model_list):
                        update = user_model.state_dict(
                        )[key] - original_model.state_dict()[key]
                        update = update * self.user_weight_list[i]
                        self.model.state_dict()[key] += update
        else:
            self.model = copy.deepcopy(global_model)
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.learning_rate)
            avg_loss, avg_acc = self.train(
                self.model, self.train_loader, self.optimizer)

        return avg_loss, avg_acc

    def train(self, model, train_loader, optimizer=None):
        """
        Local training with noise addition
        Args:
            model: Model
            train_loader: Training data loader
            optimizer: Optional custom optimizer
        Returns:
            train_loss: Average training loss
            train_acc: Average training accuracy
        """
        model.train()
        total_loss = 0.0
        total_acc = 0.0

        if optimizer is None:
            optimizer = self.optimizer

        for epoch in range(self.epoch_num):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for id, (data, target) in enumerate(train_loader):
                # Empty the gradient
                optimizer.zero_grad()

                # Move the data to the correct device (CPU/GPU)
                data = data.to(self.device, dtype=torch.float32)
                target = target.to(self.device, dtype=torch.long)

                # Forward pass using the provided model
                output = model(data)
                loss = F.cross_entropy(output, target)

                # Backward propagation
                loss.backward()

                # Step the optimizer
                optimizer.step()

                # Accumulate the loss
                epoch_loss += loss.item()

                # Calculate the accuracy
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_samples += target.size(0)

            # Calculate the average loss and accuracy
            epoch_avg_loss = epoch_loss / len(train_loader)
            epoch_avg_acc = epoch_correct / epoch_samples * 100.0

            print(
                f"Silo {self.id}, Epoch {epoch}: Loss = {epoch_avg_loss:.4f}, Accuracy = {epoch_avg_acc:.2f}%")

            total_loss += epoch_avg_loss
            total_acc += epoch_avg_acc

        avg_loss = total_loss / self.epoch_num
        avg_acc = total_acc / self.epoch_num

        return avg_loss, avg_acc
