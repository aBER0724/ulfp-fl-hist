import copy
import torch


class Server:
    def __init__(self, database, global_model, global_learning_rate, train_mode):
        self.model = copy.deepcopy(global_model)
        self.silo_weights = []
        self.silos_model_list = []
        self.global_learning_rate = global_learning_rate
        self.train_mode = train_mode
        
    def get_model(self):
        """
        Get the global model.
        """
        return self.model
    
    def receive_silos_model(self, channel_server_to_silos):
        """
        Receive the silos model list.
        """
        
        n_user = len(channel_server_to_silos[0])
        sumed_user_histogram = [0 for _ in range(n_user)]
        
        for i in range(n_user):
            for silo_sent_histogram in channel_server_to_silos:
                sumed_user_histogram[i] += silo_sent_histogram[i]

        return sumed_user_histogram

    def update_global_model(self):
        """
        Server aggregates silos model params and update global model with weighted average.
        """
        global_params = self.model.state_dict()
        total_weight = 0

        aggregated_params = {key: torch.zeros_like(global_params[key])
                            for key in global_params}

        with torch.no_grad():

            for model in self.silos_model_list:
                model_params = model.state_dict()

                for key in aggregated_params:
                    aggregated_params[key] -= self.global_learning_rate * model_params[key]
            for key in aggregated_params: 
                if self.train_mode == 'uldp': # 1/|U||S|
                    aggregated_params[key] /= self.dataset.get_user_number() * len(self.silos_model_list)
                else: # 1/|S|
                    aggregated_params[key] /= len(self.silos_model_list)

        self.model.load_state_dict(aggregated_params)