import copy
import torch


class Server:
    def __init__(self, database, global_model):
        self.model = copy.deepcopy(global_model)
        self.silo_weights = database.get_silos_weights()
        self.silos_model_list = []
        
    def get_model(self):
        """
        Get the global model.
        """
        return self.model
    
    def receive_silos_model(self, silo_model_list):
        """
        Receive the silos model list.
        """
        self.silos_model_list = silo_model_list

    def update_global_model(self):
        """
        Server aggregates silos model params and update global model with weighted average.
        """
        global_params = self.model.state_dict()
        total_weight = sum(self.silo_weights)

        aggregated_params = {key: torch.zeros_like(global_params[key])
                            for key in global_params}

        with torch.no_grad():

            for model, weight in zip(self.silos_model_list, self.silo_weights):
                model_params = model.state_dict()

                for key in aggregated_params:
                    aggregated_params[key] += weight * model_params[key]
            for key in aggregated_params:
                aggregated_params[key] /= total_weight

        self.model.load_state_dict(aggregated_params)

    def send_global_model_to_silos(self, silo_list):
        """
        Send the global model to the silos.
        """
        with torch.no_grad():
            for silo in silo_list:
                silo.model.load_state_dict(
                    self.model.state_dict(), strict=True)
