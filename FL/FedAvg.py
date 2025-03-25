import torch
import os
import torch.utils.data
import torch.nn as nn
import argparse
import time
from datetime import datetime
from model.CNN import CNNMnist
from FL.utilis.silo import Silo
from FL.utilis.server import Server
from FL.utilis.dataset import Dataset
from FL.test.test_histogram import visualize_histograms

EPSILON = 5.0
DELTA = 1e-5
MAX_GRAD_NORM = 1.2

def fed_avg(dataset_name, model, n_silos, n_users, seed, batch_size, local_epochs, local_learning_rate, global_learning_rate, alpha, mode, q_for_batch_size, is_secure, is_clip, is_histogram_secure, weighting_mode, epsilon, histogram_epsilon, delta, max_grad_norm, max_samples, iters=10):
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
    dataset = Dataset(dataset_name, n_silos, n_users, alpha, seed, mode, max_samples)
    
    server = Server(dataset, model, global_learning_rate, is_secure)
    silos = [Silo(i, server, dataset, model, local_learning_rate, q_for_batch_size, local_epochs, is_secure, is_clip, is_histogram_secure, epsilon, histogram_epsilon, delta, max_grad_norm) for i in range(n_silos)]
    
    print("================COMPUTE USER WEIGHT START================")
    
    channel_silos_to_server = []
    
    # Silos send user histogram to server
    for i in range(n_silos):
        channel_silos_to_server.append(silos[i].get_user_histogram())
    
    # Server calculate each user sum of histogram
    sumed_user_histogram = server.receive_silos_histogram(channel_silos_to_server)
    
    # Server broadcast user histogram to silos
    channel_server_to_silos = [sumed_user_histogram for _ in range(n_silos)]
    
    # Silos receive user histogram from server
    for i in range(n_silos):
        silos[i].receive_user_histogram(channel_server_to_silos[i])
    
    # Visualize histograms if histogram security is enabled
    if is_histogram_secure:
        # Create timestamp-based directory for this run's visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_output_dir = f"record/histogram_viz_{timestamp}_s{n_silos}_u{n_users}_{is_secure}"
        visualize_histograms(silos, output_dir=viz_output_dir)
        print(f"Histogram visualizations saved to {viz_output_dir}")
    
    print("=============COMPUTE USER WEIGHT FINISHED==============")
    

    print(f"====================FL START===================")

    losses = []
    accuracies = []

    for i in range(iters):

        print(f"--------------------No.{i} FL--------------------")

        # 1. Broadcast Global Model to silos
        channel_server_to_silos = [server.get_model() for _ in range(n_silos)]

        # 2. Local Training and get average metrics
        avg_loss_total = 0
        avg_acc_total = 0
        for j, silo in enumerate(silos):
            silo.receive_global_model(server.get_model())
            avg_loss, avg_acc = silo.local_silos_sgd(server.get_model())
            avg_loss_total += avg_loss
            avg_acc_total += avg_acc
            print(f"Silo {j} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%")
        
        # Calculate overall averages
        avg_loss = avg_loss_total / n_silos
        avg_acc = avg_acc_total / n_silos

        # 3. Aggregate Local Model to Global Model
        channel_silos_to_server = [silo.model for silo in silos]
        server.receive_silos_model(channel_silos_to_server)
        
        # 4. Update Global Model
        server.update_global_model()

        # 5. Record training history
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
            'local_learning_rate': local_learning_rate,
            'global_learning_rate': global_learning_rate,
            'alpha': alpha,
            'mode': mode
        }
    }

    # Create a unique directory for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"record/output_{timestamp}_s{n_silos}_u{n_users}_e{local_epochs}_{is_secure}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as .pth file in the unique directory
    filename = f"{output_dir}/fedavg_s{n_silos}_u{n_users}_i{iters}_e{local_epochs}_b{batch_size}_glr{global_learning_rate}_llr{local_learning_rate}_a{alpha}_{mode}_{is_secure}.pth"
    torch.save(record, filename)
    
    print(f"Training results saved to: {output_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='FedAvg')
    
    # Basic
    parser.add_argument('--n_silos', type=int, default=10, help='Number of silos')
    parser.add_argument('--n_users', type=int, default=300, help='Number of users')

    # Data Divide
    parser.add_argument('--dataset', type=str, default=['mnist','heart_disease','tcga_brca','creditcard'], help='Dataset name')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--alpha', type=float, default=0.05, help='Alpha parameter')
    parser.add_argument('--distribute_mode', type=str, default='zipf', help='Distribution mode')

    # FL
    parser.add_argument('--iters', type=int, default=20, help='FL iterations')
    parser.add_argument('--local_epoch', type=int, default=5, help='Local epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--q_for_batch_size', type=float, default=0.1, help='q for batch size')
    parser.add_argument('--local_learning_rate', type=float, default=0.01, help='Local Learning rate')
    parser.add_argument('--global_learning_rate', type=float, default=1.0, help='Global Learning rate')
    parser.add_argument('--weighting_mode', type=str, default='', help='Weighting mode')
    parser.add_argument('--is_clip', type=bool, default=False, help='Clip User Data')
    
    # DP
    parser.add_argument('--is_secure', type=str, choices=['cldp', 'uldp', 'ndp'], default='ndp', help='Available Differential Privacy')
    parser.add_argument('--is_histogram_secure', type=bool, default=False, help='Histogram Differential Privacy')
    parser.add_argument('--epsilon', type=float, default=5.0, help='Epsilon parameter')
    parser.add_argument('--histogram_epsilon', type=float, default=1.0, help='Histogram Epsilon parameter')
    parser.add_argument('--delta', type=float, default=1e-5, help='Delta parameter')
    parser.add_argument('--max_grad_norm', type=float, default=1.2, help='Max gradient norm')
    
    args = parser.parse_args()
    
    # Basic
    n_silos = args.n_silos
    n_users = args.n_users
    
    # Data Divide
    dataset_name = args.dataset
    max_samples = args.max_samples
    seed = args.seed
    alpha = args.alpha
    distribute_mode = args.distribute_mode
    
    # FL
    iters = args.iters
    local_epoch = args.local_epoch
    batch_size = args.batch_size
    q_for_batch_size = args.q_for_batch_size
    global_learning_rate = args.global_learning_rate
    local_learning_rate = args.local_learning_rate
    weighting_mode = args.weighting_mode
    is_clip = args.is_clip
    
    # DP
    is_secure = args.is_secure
    is_histogram_secure = args.is_histogram_secure
    epsilon = args.epsilon
    histogram_epsilon = args.histogram_epsilon
    delta = args.delta
    max_grad_norm = args.max_grad_norm
    
    model = CNNMnist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fed_avg(dataset_name, 
            model, 
            n_silos, 
            n_users, 
            seed, 
            batch_size,
            local_epoch, 
            local_learning_rate, 
            global_learning_rate,
            alpha, 
            distribute_mode, 
            q_for_batch_size, 
            max_samples,
            is_secure, 
            is_clip,
            is_histogram_secure,
            weighting_mode,
            epsilon, 
            histogram_epsilon, 
            delta, 
            max_grad_norm, 
            iters)