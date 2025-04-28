import math
import copy
import torch
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import torch.nn.functional as F
from opacus import PrivacyEngine
import torch.nn as nn
import numpy as np
from FL.utilis.sampling import get_data_loaders_uniform_without_replace


class Silo:
    def __init__(self, silo_id, server, dataset, model, learning_rate, q_for_batch_size, epoch_num, is_secure='ndp', is_histogram_secure=False, delta=1e-5, sigma=5.0, max_grad_norm=1.0, max_data_number=1000):
        self.id = silo_id
        self.server = server
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = copy.deepcopy(model)
        self.model = self.model.to(self.device)
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
        self.train_mode = is_secure # cldp or uldp
        self.histogram_mode = is_histogram_secure # True or False
        self.max_grad_norm = max_grad_norm # C
        self.delta = delta
        self.sigma = sigma
        self.dataset = dataset
        self.consumed_epsilon = 0.0  # Initialize accumulated privacy budget consumption
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
            self.privacy_engine = PrivacyEngine()
        
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
        self.model = self.model.to(self.device)

    def histogram_dp(self, user_histogram):
        """
        Add noise to the user's histogram.
        """
        # Using sigma directly instead of epsilon
        noise = np.random.normal(0, self.sigma, size=len(user_histogram))
        
        # Use np.maximum to ensure values are non-negative
        return np.maximum(0, np.round(user_histogram + noise)).astype(int)
    
    def get_user_histogram(self, is_secure=True):
        """
        Get the histogram of user's data.
        """
        if is_secure and self.histogram_mode:
            return self.histogram_dp(self.user_histogram)
        else:
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

    def local_silos_sgd(self, global_model, current_iter):
        """
        silo local stochastic gradient descent.
        Returns:
            avg_epoch_loss: Average loss of all silos
            avg_epoch_acc: Average accuracy of all silos
        """
        original_model = copy.deepcopy(global_model)
        original_model = original_model.to(self.device)
        total_loss = 0
        total_acc = 0
        avg_loss = 0
        avg_acc = 0

        if self.train_mode == 'cldp':
            self.model = copy.deepcopy(global_model)
            self.model = self.model.to(self.device)
            
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.learning_rate)
            
            # Use make_private with explicitly specified noise_multiplier (sigma) and delta
            self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=self.sigma,
                max_grad_norm=self.max_grad_norm
            )
            
            # Train the model
            avg_loss, avg_acc = self.train(
                self.model, self.train_loader, self.optimizer, current_iter)
            
            # Get the actual epsilon spent in this round (based on our specified sigma and delta)
            actual_epsilon = self.privacy_engine.get_epsilon(delta=self.delta)
            self.consumed_epsilon += actual_epsilon
            logger.info(f"Iter {current_iter} | Silo {self.id} accumulated ε={self.consumed_epsilon:.4f} (with σ={self.sigma}, δ={self.delta})")

        elif self.train_mode == 'uldp':
            users = self.silo_data_indices
            user_model_list = [copy.deepcopy(self.model)
                               for _ in range(len(users))]

            for i in range(len(user_model_list)):
                user_model_list[i] = user_model_list[i].to(self.device)
            valid_user_count = 0

            # Calculate privacy parameters for ULDP
            # For ULDP, each user's update consumes a portion of the privacy budget
            user_count = len(users)
            # Calculate noise scale based on actual noise added during clipping
            # Using the analytic Gaussian mechanism formula for (ε,δ)-DP
            # ε = sqrt(2*log(1.25/δ)) * (sensitivity / σ)
            c = self.max_grad_norm  # sensitivity (clipping norm)
            actual_epsilon_per_user = 0
            if user_count > 0 and self.sigma > 0:
                # Calculate actual ε spent per user based on the noise scale (sigma)
                actual_epsilon_per_user = (c / self.sigma) * math.sqrt(2 * math.log(1.25 / self.delta))
                
            epsilon_spent_this_round = 0

            for user_id, user_data_indices in users.items():
                user_data = self.dataset.get_dataset(user_data_indices)
                
                # Skip users with empty datasets
                if len(user_data) == 0:
                    logger.info(f"Iter {current_iter} | Silo {self.id}, User {user_id}: Skipping - Empty dataset")
                    continue
                    
                user_loader = torch.utils.data.DataLoader(
                    user_data, batch_size=self.microbatch_size, shuffle=True
                )

                user_model = user_model_list[user_id]
                user_optimizer = torch.optim.SGD(user_model.parameters(), lr=self.learning_rate)

                user_avg_loss, user_avg_acc = self.train(
                    user_model, user_loader, user_optimizer, current_iter)

                logger.info(f"Iter {current_iter} | Silo {self.id}, User {user_id}: Loss = {user_avg_loss:.4f}, Accuracy = {user_avg_acc:.2f}%")

                total_loss += user_avg_loss
                total_acc += user_avg_acc
                valid_user_count += 1
                
                # Add to the spent epsilon for each active user
                epsilon_spent_this_round += actual_epsilon_per_user

            # Avoid division by zero if no valid users
            if valid_user_count > 0:
                avg_loss = total_loss / valid_user_count
                avg_acc = total_acc / valid_user_count
            else:
                logger.warning(f"Iter {current_iter} | Silo {self.id} has no valid users with data")
                avg_loss = 0
                avg_acc = 0

            # Update epsilon consumption with actual spent amount
            self.consumed_epsilon += epsilon_spent_this_round
            logger.info(f"Iter {current_iter} | Silo {self.id} accumulated ε={self.consumed_epsilon:.4f} (with σ={self.sigma}, δ={self.delta})")

            # Update the model with the weighted average of the user's models
            with torch.no_grad():
                for key in self.model.state_dict():
                    self.model.state_dict()[key].copy_(original_model.state_dict()[key])
                    for i, user_model in enumerate(user_model_list):
                        delta = user_model.state_dict()[key] - original_model.state_dict()[key]
                        weighted_delta = delta * self.user_weight_list[i]
                        
                        # Clipping
                        if torch.is_tensor(weighted_delta) and weighted_delta.numel() > 0:
                            norm = torch.norm(weighted_delta.flatten(), p=2)
                            
                            # min(1, C/||\delta^{s,u}_t||_2))
                            if norm > 0:
                                clip_factor = min(1.0, self.max_grad_norm / norm)
                                weighted_delta = weighted_delta * clip_factor
                            
                            # Add Gaussian noise for ULDP with scale determined by sigma
                            noise = torch.randn_like(weighted_delta) * self.sigma * self.max_grad_norm
                            weighted_delta += noise
                            
                        self.model.state_dict()[key] += weighted_delta
        else:
            self.model = copy.deepcopy(global_model)
            self.model = self.model.to(self.device)
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.learning_rate)
            avg_loss, avg_acc = self.train(
                self.model, self.train_loader, self.optimizer, current_iter)
            
            # No privacy budget consumed in non-private mode
            logger.info(f"Iter {current_iter} | Silo {self.id}: No privacy budget consumed (non-private mode)")

        return avg_loss, avg_acc

    def train(self, model, train_loader, optimizer=None, current_iter=0): # Add current_iter with default
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
            epoch_correct = 0.0
            epoch_samples = 0.0

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

            logger.info(f"Iter {current_iter} | Silo {self.id}, Epoch {epoch}: Loss = {epoch_avg_loss:.4f}, Accuracy = {epoch_avg_acc:.2f}%")

            total_loss += epoch_avg_loss
            total_acc += epoch_avg_acc

        avg_loss = total_loss / self.epoch_num
        avg_acc = total_acc / self.epoch_num

        return avg_loss, avg_acc
