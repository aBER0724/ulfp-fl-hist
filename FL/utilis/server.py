import copy
import torch


class Server:
    def __init__(self, dataset, global_model, global_learning_rate, train_mode):
        """
        初始化服务器实例 

        参数:
            dataset (FL.utilis.dataset.Dataset): 数据集对象的引用，
                                                用于访问总用户数等信息 
            global_model (torch.nn.Module): 初始全局模型 
            global_learning_rate (float): 服务器端应用于聚合更新的学习率 
            train_mode (str): 训练模式，如'uldp'，会影响更新的平均方式 
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = copy.deepcopy(global_model)
        self.model = self.model.to(self.device)
        self.silo_weights = []
        self.silos_model_list = []
        self.global_learning_rate = global_learning_rate
        self.train_mode = train_mode
        self.dataset = dataset
        
    def get_model(self):
        """
        返回服务器当前持有的全局模型 

        返回:
            torch.nn.Module: 服务器的全局模型 
        """
        return self.model

    def receive_silos_model(self, channel_server_to_silos):
        """
        接收并存储从机构发送的模型列表 

        参数:
            channel_server_to_silos (list[torch.nn.Module]): 包含每个机构在本地训练后
                                                            的模型实例列表 
                                                            (注意: 参数名可能有误导，
                                                            实际是从机构到服务器的模型) 
        """
        self.silos_model_list = channel_server_to_silos
    
    def receive_silos_histogram(self, channel_server_to_silos):
        """
        接收所有机构的用户直方图并聚合它们 

        它汇总了所有机构提供的直方图中每个用户的计数 

        参数:
            channel_server_to_silos (list[list[int]]): 每个元素是一个机构的用户直方图
                                                      (按用户ID索引的计数列表) 
                                                      (注意: 参数名可能有误导) 

        返回:
            list[int]: 聚合的用户直方图，其中每个元素是该用户在所有机构中的总计数 
        """
        if not channel_server_to_silos or not isinstance(channel_server_to_silos[0], list):
            # 处理空或格式错误的输入
            return []
            
        n_user = len(channel_server_to_silos[0])
        sumed_user_histogram = [0 for _ in range(n_user)]

        for i in range(n_user):
            for silo_sent_histogram in channel_server_to_silos:
                sumed_user_histogram[i] += silo_sent_histogram[i]
        return sumed_user_histogram

    def update_global_model(self):
        """
        聚合接收到的机构模型参数并更新全局模型 

        聚合策略基于联邦平均(FedAvg) 
        具体来说，它计算每个机构的*更新*(本地模型-全局模型)的平均值，
        并将这个平均更新应用到当前全局模型，
        按`global_learning_rate`缩放 

        如果`self.train_mode`是'uldp'，则平均按
        用户总数和机构数归一化 否则仅按机构数归一化 
        """
        global_params = self.model.state_dict()

        # 用全局模型参数初始化聚合参数
        aggregated_params = copy.deepcopy(global_params)

        with torch.no_grad():
            # 检查所有模型共有的键
            common_keys = set(global_params.keys())
            for model in self.silos_model_list:
                model_params = model.state_dict()
                common_keys = common_keys.intersection(set(model_params.keys()))
            
            # 仅使用共有键进行聚合
            for key in common_keys:
                # 从全局模型参数开始
                aggregated_params[key] = global_params[key]
                
                # 汇总每个机构模型的更新(FedAvg)
                total_update = torch.zeros_like(global_params[key])
                for model in self.silos_model_list:
                    model_params = model.state_dict()
                    # 计算本地和全局模型之间的差异作为更新
                    update = model_params[key] - global_params[key]
                    total_update += update
                
                # 平均更新并应用学习率
                if self.train_mode == 'uldp':  # 1/|U||S|
                    aggregated_params[key] += self.global_learning_rate * (total_update / (self.dataset.get_user_number() * len(self.silos_model_list)))
                else:  # 1/|S|
                    aggregated_params[key] += self.global_learning_rate * (total_update / len(self.silos_model_list))

        # 将聚合参数加载回全局模型
        self.model.load_state_dict(aggregated_params)