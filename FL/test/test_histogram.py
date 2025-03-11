from FL.utilis.silo import Silo
from FL.utilis.server import Server
from FL.utilis.dataset import Dataset
from model.CNN import CNNMnist
import torch
import argparse

if __name__ == "__main__":
	dataset_name = "mnist"
	max_samples = 1000
	batch_size = 256
	iters = 20
	local_epoch = 5
	distribute_mode = "zipf"
	is_secure = "ndp"
	learning_rate = 0.1
	n_silos = 5
	n_users = 20
	seed = 1
	alpha = 0.05
	q_for_batch_size = 0.1
	epsilon = 50.0
	delta = 1e-5
	max_grad_norm = 1.2
	model = CNNMnist()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	dataset = Dataset(dataset_name, n_silos, n_users, alpha, seed, distribute_mode, max_samples=max_samples)

	server = Server(dataset, model)
	silos = [Silo(i, dataset, model, learning_rate, q_for_batch_size, local_epoch, is_secure, epsilon, delta, max_grad_norm) for i in range(n_silos)]
    
	for silo in silos:
		for user_id, data_num in silo.get_user_histogram().items():
			print(user_id, data_num)
