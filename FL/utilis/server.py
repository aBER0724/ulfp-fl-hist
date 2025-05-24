import copy
import torch


class Server:
    def __init__(self, dataset, global_model, global_learning_rate, train_mode):
        """
        Initializes the Server instance.

        Args:
            dataset (FL.utilis.dataset.Dataset): A reference to the dataset object,
                                                 used to access information like total user count.
            global_model (torch.nn.Module): The initial global model.
            global_learning_rate (float): The learning rate applied to the aggregated updates
                                          at the server side.
            train_mode (str): The training mode, e.g., 'uldp', which can affect
                              how updates are averaged.
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
        Returns the current global model held by the server.

        Returns:
            torch.nn.Module: The server's global model.
        """
        return self.model

    def receive_silos_model(self, channel_server_to_silos):
        """
        Receives and stores the list of models sent from the silos.

        Args:
            channel_server_to_silos (list[torch.nn.Module]): A list containing the model
                                                             instances from each silo after
                                                             local training.
                                                             (Note: Parameter name might be misleading,
                                                              it's models from silos to server).
        """
        self.silos_model_list = channel_server_to_silos
    
    def receive_silos_histogram(self, channel_server_to_silos):
        """
        Receives user histograms from all silos and aggregates them.

        It sums up the counts for each user across all silo-provided histograms.

        Args:
            channel_server_to_silos (list[list[int]]): A list where each element is a user histogram
                                                       (list of counts, indexed by user ID) from a silo.
                                                       (Note: Parameter name might be misleading).

        Returns:
            list[int]: An aggregated user histogram where each element is the total count
                       for that user across all silos.
        """
        if not channel_server_to_silos or not isinstance(channel_server_to_silos[0], list):
            # Handle empty or malformed input
            return []
            
        n_user = len(channel_server_to_silos[0])
        sumed_user_histogram = [0 for _ in range(n_user)]

        for i in range(n_user):
            for silo_sent_histogram in channel_server_to_silos:
                sumed_user_histogram[i] += silo_sent_histogram[i]
        return sumed_user_histogram

    def update_global_model(self):
        """
        Aggregates model parameters from the received silo models and updates the global model.

        The aggregation strategy is based on Federated Averaging (FedAvg).
        Specifically, it calculates the average of the *updates* (local model - global model)
        from each silo and applies this average update to the current global model,
        scaled by the `global_learning_rate`.

        If `self.train_mode` is 'uldp', the averaging is normalized by the
        total number of users and the number of silos. Otherwise, it's normalized
        only by the number of silos.
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