import torch
import os
import torch.utils.data
import torch.nn as nn
import argparse
from model.CNN import CNNMnist
from FL.utilis.silo import Silo
from FL.utilis.server import Server
from FL.utilis.dataset import Dataset

EPSILON = 50.0
DELTA = 1e-5
MAX_GRAD_NORM = 1.2

def fed_avg(dataset_name, model, n_silos, n_users, seed, batch_size, local_epochs, learning_rate, alpha, mode, q_for_batch_size, is_secure, epsilon, delta, max_grad_norm, iters=10):
    """
    Params:
        dataset_name: Dataset Name
        model: FL Model
        n_silos: Silos Number
        seed: Random Seed
        batch_size: Batch Size
        local_epochs: Local Epochs
        learning_rate: Local Learning Rate
        alpha: Distribution Parameter
        mode: Distribution Mode, 'uniform', 'zipf' or 'dirichlet'
        iters: Iteration Number(fixd as 10)
    """
    dataset = Dataset(dataset_name, n_silos, n_users, alpha, seed, mode)
    
    server = Server(dataset, model)
    silos = [Silo(i, dataset, model, learning_rate, q_for_batch_size, local_epochs, is_secure, epsilon, delta, max_grad_norm) for i in range(n_silos)]

    print(f"====================FL START===================")

    losses = []
    accuracies = []

    for i in range(iters):

        print(f"--------------------No.{i} FL--------------------")

        # 1. Broadcast Global Model to silos
        server.send_global_model_to_silos(silos)

        # 2. Local Training and get average metrics
        for silo in silos:
            avg_loss, avg_acc = silo.local_silos_sgd()

        # 3. Aggregate Local Model to Global Model
        silos_models = [silo.model for silo in silos]
        server.receive_silos_model(silos_models)
        
        # 然后执行聚合
        server.update_global_model()

        # 4. Record training history
        losses.append(avg_loss)
        accuracies.append(avg_acc)
        print(
            f"Iter {i} - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.2f}%")

    record = {
        'iters': iters,
        'local_epochs': local_epochs,
        'model_state': server.get_model().state_dict(),
        'training_history': {
            'losses': losses,  # Average loss of each iteration
            'accuracies': accuracies  # Average accuracy of each iteration
        },
        'final_metrics': {
            'final_loss': losses[-1] if losses else float('inf'),
            'final_accuracy': accuracies[-1] if accuracies else 0.0
        },
        'training_config': {
            'n_silos': n_silos,
            'n_users': n_users,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'alpha': alpha,
            'mode': mode
        }
    }

    # Save as .pth file
    os.makedirs("record/record_output", exist_ok=True)
    filename = f"record/record_output/fedavg_s{n_silos}_u{n_users}_i{iters}_e{local_epochs}_b{batch_size}_lr{learning_rate}_a{alpha}_{mode}_{is_secure}.pth"
    torch.save(record, filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='FedAvg')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset name')
    parser.add_argument('--n_silos', type=int, default=10, help='Number of silos')
    parser.add_argument('--n_users', type=int, default=300, help='Number of users')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--alpha', type=float, default=0.05, help='Alpha parameter')
    parser.add_argument('--iters', type=int, default=20, help='FL iterations')
    parser.add_argument('--local_epoch', type=int, default=5, help='Local epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--distribute_mode', type=str, default='zipf', help='Distribution mode')
    parser.add_argument('--epsilon', type=float, default=50.0, help='Epsilon parameter')
    parser.add_argument('--delta', type=float, default=1e-5, help='Delta parameter')
    parser.add_argument('--max_grad_norm', type=float, default=1.2, help='Max gradient norm')
    parser.add_argument('--q_for_batch_size', type=float, default=0.1, help='q for batch size')
    parser.add_argument('--is_secure', type=str, choices=['dp', 'ndp'], default='ndp', help='Available Differential Privacy')
    
    args = parser.parse_args()
    
    dataset_name = args.dataset
    batch_size = args.batch_size
    iters = args.iters
    local_epoch = args.local_epoch
    distribute_mode = args.distribute_mode
    is_secure = args.is_secure
    learning_rate = args.learning_rate
    n_silos = args.n_silos
    n_users = args.n_users
    seed = args.seed
    alpha = args.alpha
    q_for_batch_size = args.q_for_batch_size
    epsilon = args.epsilon
    delta = args.delta
    max_grad_norm = args.max_grad_norm
    
    model = CNNMnist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fed_avg(dataset_name, model, n_silos, n_users, seed, batch_size,
                    local_epoch, learning_rate, alpha, distribute_mode, q_for_batch_size, is_secure, epsilon, delta, max_grad_norm, iters)