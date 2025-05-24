from FL.utilis.hist_noise import histogram_dp, Mean_NoiseFirst, Median_NoiseFirst, Mean_StructureFirst, Median_StructureFirst
from FL.utilis.sampling import get_data_loaders_uniform_without_replace
import numpy as np
import torch.nn as nn
from opacus import PrivacyEngine
import torch.nn.functional as F
import math
import copy
import torch
import logging
import sys

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Silo:
    def __init__(self, silo_id, server, dataset,
                 model, learning_rate, q_for_batch_size, epoch_num,
                 is_secure='ndp', is_histogram_secure='', delta=1e-5,
                 sigma=5.0, max_grad_norm=1.0, histogram_F=100.0,
                 histogram_epsilon=1.0, histogram_k_div = 10):
        """
        Initializes a Silo instance.

        Args:
            silo_id (int): Unique identifier for the silo.
            server (FL.utilis.server.Server): Reference to the central server instance.
            dataset (FL.utilis.dataset.Dataset): Reference to the dataset object, used to get
                                                 silo-specific data and user distributions.
            model (torch.nn.Module): The initial model architecture.
            learning_rate (float): Learning rate for local training.
            q_for_batch_size (float): Sampling rate q, fraction of silo's data to form a minibatch
                                      (e.g., 0.1 for 10% of data).
            epoch_num (int): Number of local training epochs per FL round.
            is_secure (str, optional): Specifies the DP mode for model training.
                                       'ndp' (non-private), 'cldp' (client-level DP via Opacus),
                                       'uldp' (user-level DP, custom implementation). Defaults to 'ndp'.
            is_histogram_secure (str, optional): Specifies DP mode for user histograms.
                                                 'dp' (Gaussian noise), 'sf_dp' (StructureFirst),
                                                 'nf_dp' (NoiseFirst), '' (non-private). Defaults to ''.
            delta (float, optional): Target delta for (epsilon, delta)-DP. Defaults to 1e-5.
            sigma (float, optional): Noise multiplier for DP mechanisms. Defaults to 5.0.
            max_grad_norm (float, optional): Gradient clipping norm (C). Defaults to 1.0.
            histogram_F (float, optional): Upper bound on histogram counts, used for sensitivity
                                           in StructureFirst with SSE. Defaults to 100.0.
            histogram_epsilon (float, optional): Epsilon for histogram DP. Defaults to 1.0.
            histogram_k_div (int, optional): Divisor to determine `k` (number of bins) for
                                             StructureFirst/NoiseFirst based on histogram length
                                             (k = len(hist) / k_div). Defaults to 10.
        """
        self.id = silo_id
        self.server = server
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = copy.deepcopy(model)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.silo_data_indices = dataset.get_silos_indices(silo_id)
        self.user_histogram = dataset.get_user_histogram(silo_id)
        self.user_weight_list = [1.0 for _ in range(len(self.user_histogram))]
        self.epoch_num = epoch_num
        self.learning_rate = learning_rate
        self.iterations = 1
        self.acc_list = []
        self.loss_list = []
        self.train_mode = is_secure
        self.histogram_mode = is_histogram_secure
        self.histogram_epsilon = histogram_epsilon
        self.histogram_F = histogram_F
        self.histogram_k = -1
        self.histogram_k_div = histogram_k_div

        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self.sigma = sigma
        self.dataset = dataset
        self.consumed_epsilon = 0.0

        indices_arr = []
        for user_id, data_list_for_user in self.silo_data_indices.items():
            indices_arr.extend(data_list_for_user)

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
        Returns the current local model of the silo.

        Returns:
            torch.nn.Module: The silo's local model.
        """
        return self.model

    def receive_global_model(self, global_model):
        """
        Receives and updates the silo's local model with the global model from the server.

        Args:
            global_model (torch.nn.Module): The global model received from the server.
        """
        self.model = copy.deepcopy(global_model)
        self.model = self.model.to(self.device)

    def get_user_histogram(self, is_secure=True):
        """
        Returns the user histogram for this silo, potentially with differential privacy applied.

        The type of DP applied depends on `self.histogram_mode`.

        Args:
            is_secure (bool, optional): If True, applies DP mechanism specified by `self.histogram_mode`.
                                       If False, returns the raw histogram. Defaults to True.
                                       (Note: Original param `is_secure` renamed for clarity).
        Returns:
            list[int]: The user histogram (list of sample counts per user in this silo).
                       This histogram may be noisy if DP is applied.
        
        Raises:
            NameError: If `F` is used in `sf_dp` mode but not defined (should be `self.histogram_F`).
        """
        if is_secure and self.histogram_mode:
            hist_len = len(self.user_histogram)
            if hist_len == 0: return [] # Return empty if original is empty

            if self.histogram_mode == "dp":
                return histogram_dp(self.user_histogram, self.sigma)
            elif self.histogram_mode == "sf_dp":
                self.histogram_k = max(1, int(hist_len / self.histogram_k_div)) if hist_len > 0 else 0
                if self.histogram_k == 0 and hist_len > 0: self.histogram_k = 1 # Ensure k is at least 1 for non-empty hist
                if self.histogram_k == 0 and hist_len == 0: return []
                return Mean_StructureFirst(np.array(self.user_histogram), self.histogram_epsilon, self.histogram_k, self.histogram_F)
            elif self.histogram_mode == "nf_dp":
                noise_histogram, self.histogram_k = Mean_NoiseFirst(np.array(self.user_histogram), self.histogram_epsilon)
                return noise_histogram.tolist()
            else:
                return self.user_histogram
        else:
            return self.user_histogram

    def receive_user_histogram(self, global_summed_user_histogram):
        """
        Receives the globally summed user histogram and calculates local user weights.

        The weight for a user `i` in this silo is calculated as:
        `local_user_i_samples / (global_sum_for_user_i_samples + epsilon_smooth)`.
        These weights are stored in `self.user_weight_list`.

        Args:
            global_summed_user_histogram (list[int]): A list where each element is the total
                                                      sample count for that user ID across ALL silos.
        Returns:
            list[float]: The calculated list of user weights for users in this silo.
        """
        if len(global_summed_user_histogram) != len(self.user_histogram):
            logger.error(f"Silo {self.id}: Mismatch in length between global histogram ({len(global_summed_user_histogram)}) and local user_histogram ({len(self.user_histogram)}). Cannot compute weights.")
            # Fallback: keep weights as 1s or handle error appropriately
            self.user_weight_list = [1.0 for _ in range(len(self.user_histogram))]
            return self.user_weight_list

        for i in range(len(global_summed_user_histogram)):
            self.user_weight_list[i] = self.user_histogram[i] / (global_summed_user_histogram[i] + 1e-8)
        return self.user_weight_list

    def local_silos_sgd(self, global_model, current_iter):
        """
        Performs local Stochastic Gradient Descent (SGD) training for the silo.

        The behavior depends on `self.train_mode`:
        - 'cldp': Client-Level Differential Privacy. Uses Opacus PrivacyEngine to make
                  the model and optimizer private. Epsilon is tracked via PrivacyEngine.
        - 'uldp': User-Level Differential Privacy. Iterates through users in the silo,
                  trains a model for each user's data, clips and aggregates their updates
                  with added noise. Epsilon is calculated based on the number of active users
                  and noise level.
        - 'ndp' (or other): Non-private training. Standard SGD on the silo's data.

        Args:
            global_model (torch.nn.Module): The current global model received from the server,
                                            used as the starting point for local training.
            current_iter (int): The current federated learning iteration number (for logging).

        Returns:
            tuple[float, float]:
                - avg_loss: The average training loss over local epochs for this silo.
                            For 'uldp', it's the average loss across participating users.
                - avg_acc: The average training accuracy over local epochs for this silo.
                           For 'uldp', it's the average accuracy across participating users.
        """
        original_model = copy.deepcopy(global_model)
        original_model = original_model.to(self.device)
        total_loss = 0.0
        total_acc = 0.0
        avg_loss = 0.0
        avg_acc = 0.0

        if self.train_loader is None and (self.train_mode == 'cldp' or self.train_mode == 'ndp'):
             logger.warning(f"Silo {self.id}: No data to train (train_loader is None). Skipping local training.")
             return 0.0, 0.0

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
            logger.info(
                f"Iter {current_iter} | Silo {self.id} accumulated ε={self.consumed_epsilon:.4f} (with σ={self.sigma}, δ={self.delta})")

        elif self.train_mode == 'uldp':
            users = self.silo_data_indices
            user_model_list = [copy.deepcopy(self.model) for _ in range(len(users))]

            for i in range(len(user_model_list)):
                user_model_list[i] = user_model_list[i].to(self.device)
            valid_user_count = 0

            c = self.max_grad_norm
            actual_epsilon_per_user = 0.0

            if self.sigma > 0 and self.delta > 0:
                actual_epsilon_per_user = (c / self.sigma) * math.sqrt(2 * math.log(1.25 / self.delta))

            epsilon_spent_this_round_for_silo = 0.0

            for user_id, user_data_indices_list in users.items():
                user_data = self.dataset.get_dataset(user_data_indices_list)

                if len(user_data) == 0:
                    logger.info(
                        f"Iter {current_iter} | Silo {self.id} | User {user_id} | Skipping - Empty dataset")
                    continue

                user_loader = torch.utils.data.DataLoader(
                    user_data, batch_size=self.microbatch_size, shuffle=True
                )

                user_model = user_model_list[user_id]
                user_optimizer = torch.optim.SGD(
                    user_model.parameters(), lr=self.learning_rate)

                user_avg_loss, user_avg_acc = self.train(
                    user_model, user_loader, user_optimizer, current_iter)

                logger.info(
                    f"Iter {current_iter} | Silo {self.id} | User {user_id} | Loss = {user_avg_loss:.4f}, Accuracy = {user_avg_acc:.2f}%")

                total_loss += user_avg_loss
                total_acc += user_avg_acc
                valid_user_count += 1

                # This is a simplified sum; proper composition (e.g. RDP) is needed for rigorous ULDP.
                epsilon_spent_this_round_for_silo += actual_epsilon_per_user

            if valid_user_count > 0:
                avg_loss = total_loss / valid_user_count
                avg_acc = total_acc / valid_user_count
            else:
                logger.warning(
                    f"Iter {current_iter} | Silo {self.id} has no valid users with data")
                avg_loss = 0.0
                avg_acc = 0.0

            with torch.no_grad():
                for key in self.model.state_dict():
                    self.model.state_dict()[key].copy_(
                        original_model.state_dict()[key])
                    aggregated_delta = torch.zeros_like(
                        self.model.state_dict()[key])

                    for i, user_model in enumerate(user_model_list):
                        delta = user_model.state_dict(
                        )[key] - original_model.state_dict()[key]
                        norm = torch.norm(delta.flatten(), p=2)
                        clip_factor = min(
                            1.0, self.max_grad_norm / (norm + 1e-8))
                        clipped_delta = delta * clip_factor
                        weighted_delta = self.user_weight_list[i] * clipped_delta
                        aggregated_delta += weighted_delta

                    noise_std = self.sigma * self.max_grad_norm / math.sqrt(valid_user_count)
                    noise = torch.randn_like(aggregated_delta) * noise_std
                    aggregated_delta += noise
                    self.model.state_dict()[key] += aggregated_delta

            # 聚合添加噪声的场景下，考虑用户数量
            epsilon_spent_this_round = (math.sqrt(valid_user_count) * math.sqrt(2 * math.log(1.25 / self.delta))) / self.sigma
            # Update epsilon consumption with actual spent amount
            self.consumed_epsilon += epsilon_spent_this_round
            logger.info(
                f"Iter {current_iter} | Silo {self.id} accumulated ε={self.consumed_epsilon:.4f} (with σ={self.sigma}, δ={self.delta})")

        else:
            self.model = copy.deepcopy(global_model)
            self.model = self.model.to(self.device)
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.learning_rate)
            avg_loss, avg_acc = self.train(
                self.model, self.train_loader, self.optimizer, current_iter)

            logger.info(
                f"Iter {current_iter} | Silo {self.id} | No privacy budget consumed (non-private mode)")

        return avg_loss, avg_acc

    def train(self, model, train_loader, optimizer=None, current_iter=0):
        """
        Performs the actual local training loop for a given model and data loader.

        This method is called by `local_silos_sgd` for each epoch of local training.
        It handles the forward pass, loss calculation, backward pass, and optimizer step.

        Args:
            model (torch.nn.Module): The model to be trained.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
            optimizer (torch.optim.Optimizer, optional): The optimizer to use. If None,
                                                         `self.optimizer` is used (if it's compatible
                                                         with the passed model). Defaults to None.
            current_iter (int, optional): The current federated learning iteration,
                                          used for logging purposes within the epoch loop (currently commented out).
                                          Defaults to 0.

        Returns:
            tuple[float, float]:
                - avg_loss: The average loss over all epochs.
                - avg_acc: The average accuracy over all epochs.
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

            num_batches = 0
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
                num_batches +=1

                # Calculate the accuracy
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_samples += target.size(0)

            # Calculate the average loss and accuracy for the epoch
            if num_batches > 0 and epoch_samples > 0:
                epoch_avg_loss = epoch_loss / num_batches
                epoch_avg_acc = epoch_correct / epoch_samples * 100.0
            else:
                epoch_avg_loss = 0.0
                epoch_avg_acc = 0.0
                logger.warning(f"Silo {self.id}, Iter {current_iter}, Epoch {epoch}: No batches processed in train_loader.")


            # logger.info(
            #     f"Iter {current_iter} | Silo {self.id} | Epoch {epoch} | Loss = {epoch_avg_loss:.4f}, Accuracy = {epoch_avg_acc:.2f}%")

            total_loss += epoch_avg_loss
            total_acc += epoch_avg_acc

        if self.epoch_num > 0:
            avg_loss_final = total_loss / self.epoch_num
            avg_acc_final = total_acc / self.epoch_num
        else: # No epochs
            avg_loss_final = 0.0
            avg_acc_final = 0.0


        return avg_loss_final, avg_acc_final
