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
        初始化机构实例 

        参数:
            silo_id (int): 机构的唯一标识符 
            server (FL.utilis.server.Server): 中央服务器实例的引用 
            dataset (FL.utilis.dataset.Dataset): 数据集对象的引用, 用于获取
                                                机构特定数据和用户分布 
            model (torch.nn.Module): 初始模型架构 
            learning_rate (float): 本地训练的学习率 
            q_for_batch_size (float): 采样率q, 形成小批次的机构数据比例
                                      (例如0.1表示10%的数据) 
            epoch_num (int): 每轮联邦学习的本地训练周期数 
            is_secure (str, 可选): 指定模型训练的DP模式 
                                   'ndp'(非私有), 'cldp'(通过Opacus的客户端级DP),
                                   'uldp'(用户级DP, 自定义实现) 默认为'ndp' 
            is_histogram_secure (str, 可选): 指定用户直方图的DP模式 
                                             'dp'(高斯噪声), 'sf_dp'(StructureFirst),
                                             'nf_dp'(NoiseFirst), ''(非私有) 默认为'' 
            delta (float, 可选): (epsilon, delta)-DP的目标delta值 默认为1e-5 
            sigma (float, 可选): DP机制的噪声乘数 默认为5.0 
            max_grad_norm (float, 可选): 梯度裁剪范数(C) 默认为1.0 
            histogram_F (float, 可选): 直方图计数的上限, 用于StructureFirst的敏感度
                                       和SSE 默认为100.0 
            histogram_epsilon (float, 可选): 直方图DP的epsilon值 默认为1.0 
            histogram_k_div (int, 可选): 确定`k`(分箱数)的除数, 用于基于直方图长度的
                                        StructureFirst/NoiseFirst(k = len(hist) / k_div) 默认为10 
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
        返回机构当前的本地模型 

        返回:
            torch.nn.Module: 机构的本地模型 
        """
        return self.model

    def receive_global_model(self, global_model):
        """
        接收并更新机构的本地模型为来自服务器的全局模型 

        参数:
            global_model (torch.nn.Module): 从服务器接收的全局模型 
        """
        self.model = copy.deepcopy(global_model)
        self.model = self.model.to(self.device)

    def get_user_histogram(self, is_secure=True):
        """
        返回此机构的用户直方图, 可能应用了差分隐私 

        应用的DP类型取决于`self.histogram_mode` 

        参数:
            is_secure (bool, 可选): 如果为True, 应用`self.histogram_mode`指定的DP机制 
                                   如果为False, 返回原始直方图 默认为True 
                                   (注意: 原参数`is_secure`为清晰重命名) 
        返回:
            list[int]: 用户直方图(此机构中每个用户的样本计数列表) 
                       如果应用了DP, 此直方图可能包含噪声 
        
        抛出:
            NameError: 如果在`sf_dp`模式下使用了`F`但未定义(应为`self.histogram_F`) 
        """
        if is_secure and self.histogram_mode:
            hist_len = len(self.user_histogram)
            if hist_len == 0: return [] # 如果原始为空则返回空

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
        接收全局汇总的用户直方图并计算本地用户权重 

        此机构中用户`i`的权重计算为:
        `local_user_i_samples / (global_sum_for_user_i_samples + epsilon_smooth)`.
        这些权重存储在`self.user_weight_list`中 

        参数:
            global_summed_user_histogram (list[int]): 每个元素是所有机构中
                                                      该用户ID的总样本计数的列表 
        返回:
            list[float]: 此机构中用户的权重列表 
        """
        if len(global_summed_user_histogram) != len(self.user_histogram):
            logger.error(f"Silo {self.id}: Mismatch in length between global histogram ({len(global_summed_user_histogram)}) and local user_histogram ({len(self.user_histogram)}). Cannot compute weights.")
            # 回退: 保持权重为1或适当处理错误
            self.user_weight_list = [1.0 for _ in range(len(self.user_histogram))]
            return self.user_weight_list

        for i in range(len(global_summed_user_histogram)):
            self.user_weight_list[i] = self.user_histogram[i] / (global_summed_user_histogram[i] + 1e-8)
        return self.user_weight_list

    def local_silos_sgd(self, global_model, current_iter):
        """
        为机构执行本地随机梯度下降(SGD)训练 

        行为取决于`self.train_mode`:
        - 'cldp': 客户端级差分隐私 使用Opacus PrivacyEngine使
                  模型和优化器私有化 通过PrivacyEngine跟踪epsilon 
        - 'uldp': 用户级差分隐私 遍历机构中的用户, 
                  为每个用户的数据训练模型, 裁剪并聚合他们的更新
                  并添加噪声 Epsilon基于活跃用户数和噪声水平计算 
        - 'ndp'(或其他): 非私有训练 对机构数据执行标准SGD 

        参数:
            global_model (torch.nn.Module): 从服务器接收的当前全局模型, 
                                            用作本地训练的起点 
            current_iter (int): 当前联邦学习迭代次数(用于日志记录) 

        返回:
            tuple[float, float]:
                - avg_loss: 此机构本地训练周期的平均训练损失 
                             对于'uldp', 它是参与用户的平均损失 
                - avg_acc: 此机构本地训练周期的平均训练准确率 
                            对于'uldp', 它是参与用户的平均准确率 
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

                # 这是一个简化的总和；严格的ULDP需要适当的组合(如RDP) 
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

            # 聚合添加噪声的场景下, 考虑用户数量
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
        为给定模型和数据加载器执行实际的本地训练循环 

        此方法由`local_silos_sgd`为每个本地训练周期调用 
        它处理前向传播、损失计算、反向传播和优化器步骤 

        参数:
            model (torch.nn.Module): 要训练的模型 
            train_loader (torch.utils.data.DataLoader): 训练数据的数据加载器 
            optimizer (torch.optim.Optimizer, 可选): 使用的优化器 如果为None, 
                                                      则使用`self.optimizer`(如果它与
                                                      传入的模型兼容) 默认为None 
            current_iter (int, 可选): 当前联邦学习迭代, 
                                       用于周期循环内的日志记录(当前已注释) 
                                       默认为0 

        返回:
            tuple[float, float]:
                - avg_loss: 所有周期的平均损失 
                - avg_acc: 所有周期的平均准确率 
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
                # 清空梯度
                optimizer.zero_grad()

                # 将数据移动到正确的设备(CPU/GPU)
                data = data.to(self.device, dtype=torch.float32)
                target = target.to(self.device, dtype=torch.long)

                # 使用提供的模型进行前向传播
                output = model(data)
                loss = F.cross_entropy(output, target)

                # 反向传播
                loss.backward()

                # 执行优化器步骤
                optimizer.step()

                # 累计损失
                epoch_loss += loss.item()
                num_batches +=1

                # 计算准确率
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_samples += target.size(0)

            # 计算周期的平均损失和准确率
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
