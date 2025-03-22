import copy
import torch


class Server:
    def __init__(self, dataset, global_model, global_learning_rate, train_mode):
        self.model = copy.deepcopy(global_model)
        self.silo_weights = []
        self.silos_model_list = []
        self.global_learning_rate = global_learning_rate
        self.train_mode = train_mode
        self.dataset = dataset  # Add reference to dataset
        
    def get_model(self):
        """
        Get the global model.
        """
        return self.model

    def receive_silos_model(self, channel_server_to_silos):
        """
        Receive the silos model list.
        """
        self.silos_model_list = channel_server_to_silos
    
    def receive_silos_histogram(self, channel_server_to_silos):
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

        # Initialize aggregated params with global model params
        aggregated_params = copy.deepcopy(global_params)

        with torch.no_grad():
            # Check which keys are common to all models
            common_keys = set(global_params.keys())
            for model in self.silos_model_list:
                model_params = model.state_dict()
                common_keys = common_keys.intersection(set(model_params.keys()))
            
            # Only aggregate using the common keys
            for key in common_keys:
                # Start with the global model parameters
                aggregated_params[key] = global_params[key]
                
                # Sum updates from each silo model (FedAvg)
                total_update = torch.zeros_like(global_params[key])
                for model in self.silos_model_list:
                    model_params = model.state_dict()
                    # Calculate the update as the difference between local and global
                    update = model_params[key] - global_params[key]
                    total_update += update
                
                # Average the updates and apply with learning rate
                if self.train_mode == 'uldp':  # 1/|U||S|
                    aggregated_params[key] += self.global_learning_rate * (total_update / (self.dataset.get_user_number() * len(self.silos_model_list)))
                else:  # 1/|S|
                    aggregated_params[key] += self.global_learning_rate * (total_update / len(self.silos_model_list))

        # Load the aggregated parameters back to the global model
        self.model.load_state_dict(aggregated_params)