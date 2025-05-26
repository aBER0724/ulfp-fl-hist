import torch
import os
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(message)s', datefmt='%Y-%m-%d %H:%M')
logger = logging.getLogger(__name__)
import torch.utils.data
import argparse
from datetime import datetime
from model.CNN import CNNMnist
from model.MLP import PrivateFraudNet
from FL.utilis.silo import Silo
from FL.utilis.server import Server
from FL.utilis.dataset import Dataset

def fed_avg(dataset_name, model, n_silos, n_users, seed, batch_size, local_epochs, local_learning_rate, global_learning_rate, user_alpha, silo_alpha, mode, q_for_batch_size, is_secure, is_histogram_secure, histogram_epsilon, delta, sigma, max_grad_norm, max_samples, iters):
    """
    FedAvg 算法 

    参数:
        dataset_name (str): 使用的数据集名称(如'mnist', 'creditcard')
        model (torch.nn.Module): 要训练的PyTorch模型
        n_silos (int): 参与FL过程的机构总数
        n_users (int): 数据分布在机构中的用户总数
        seed (int): 数据分布等随机过程的种子
        batch_size (int): 每个机构本地训练使用的批量大小
        local_epochs (int): 每个FL轮次中机构本地训练的epoch数
        local_learning_rate (float): 机构本地优化器的学习率
        global_learning_rate (float): 服务器全局模型更新的学习率
        user_alpha (float): 用户间数据分布异质性的参数
        silo_alpha (float): 机构间数据分布异质性的参数
        mode (str): 数据分布模式('uniform', 'zipf', 'dirichlet')
        q_for_batch_size (float): 采样率q, 用于机构确定其有效批量大小, 
                                  通常与隐私计算相关(如q = batch_size / N_samples_in_silo)
        is_secure (str): 指定模型更新的差分隐私类型('cldp', 'uldp', 'ndp'表示无差分隐私)
        is_histogram_secure (str): 指定用户直方图的DP('dp', 'nf_dp', 'sf_dp', 或''表示无直方图差分隐私)
        histogram_epsilon (float): 启用直方图差分隐私的epsilon值
        delta (float): 差分隐私的delta参数
        sigma (float): 差分隐私的sigma参数
        max_grad_norm (float): 梯度裁剪阈值
        max_samples (int, optional): 使用数据集中的最大样本数 如果为None, 则使用所有样本
        iters (int): 联邦学习轮次的总数
    """
    logger.info("======================DISTRIBUTE DATA=====================")
    dataset = Dataset(dataset_name, n_silos, n_users, user_alpha, silo_alpha, seed, mode, max_samples)
    
    logger.info("=======================CREAT SERVER=======================")
    server = Server(dataset, model, global_learning_rate, is_secure)
    logger.info("=======================CREAT SILOS=======================")
    silos = [Silo(i, server, dataset, model, local_learning_rate, q_for_batch_size, local_epochs, is_secure, is_histogram_secure, delta, sigma, max_grad_norm, histogram_epsilon) for i in range(n_silos)]
    
    logger.info("================COMPUTE USER WEIGHT START================")
    
    channel_silos_to_server = []
    histogram_ks = []
    
    # 机构发送用户直方图到服务器
    logger.info("--------------SILOS SEND HISTOGRAM TO SERVER-------------")
    for i in range(n_silos):
        logger.info(f"[SILO-{i} SEND HISTOGRAM TO SERVER]")
        channel_silos_to_server.append(silos[i].get_user_histogram())
    
    # 服务器计算每个用户的直方图总和
    logger.info("----------------SERVER AGGREGATE HISTOGRAM START---------------")
    sumed_user_histogram = server.receive_silos_histogram(channel_silos_to_server)
    logger.info("------------SERVER AGGREGATE HISTOGRAM FINISHED-----------")
    
    # 服务器广播用户直方图到机构
    channel_server_to_silos = [sumed_user_histogram for _ in range(n_silos)]
    
    # 机构从服务器接收用户直方图
    logger.info("----------SERVER BORADCAST AGGREGATED HISTOGRAM-----------")
    for i in range(n_silos):
        logger.info(f"[SERVER SEND AGGREGATED HISTOGRAM TO S-{i}]")
        silos[i].receive_user_histogram(channel_server_to_silos[i])
        histogram_ks.append(silos[i].histogram_k)

    logger.info("=============COMPUTE USER WEIGHT FINISHED==============")
    

    logger.info(f"====================FL START===================")

    losses = []
    accuracies = []
    consumed_epsilons = []
    
    for i in range(iters):

        logger.info(f"--------------------No.{i} FL--------------------")

        # 1. 广播全局模型到机构
        channel_server_to_silos = [server.get_model() for _ in range(n_silos)]

        # 2. 本地训练并获取平均指标
        avg_loss_total = 0
        avg_acc_total = 0
        epsilon_total = 0
        for j, silo in enumerate(silos):
            silo.receive_global_model(server.get_model())
            avg_loss, avg_acc = silo.local_silos_sgd(server.get_model(), i)
            avg_loss_total += avg_loss
            avg_acc_total += avg_acc
            epsilon_total += silo.consumed_epsilon
            logger.info(f"Iter {i} | Silo {j} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%")
        
        # 计算总体平均值
        avg_loss = avg_loss_total / n_silos
        avg_acc = avg_acc_total / n_silos

        # 3. 聚合本地模型到全局模型
        channel_silos_to_server = [silo.model for silo in silos]
        server.receive_silos_model(channel_silos_to_server)
        
        # 4. 更新全局模型
        server.update_global_model()

        # 5. 记录训练历史
        losses.append(avg_loss)
        accuracies.append(avg_acc)
        consumed_epsilons.append(epsilon_total)
        
        logger.info(
            f"Iter {i} - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.2f}%")

    record = {
        'iters': iters,
        'local_epochs': local_epochs,
        'model_state': server.get_model().state_dict(),
        'training_history': {
            'losses': losses,  # Average loss of each iteration
            'accuracies': accuracies,  # Average accuracy of each iteration
            'consumed_epsilons': consumed_epsilons
        },
        'final_metrics': {
            'final_loss': losses[-1] if losses else float('inf'),
            'final_accuracy': accuracies[-1] if accuracies else 0.0,
            'final_consumed_epsilon': consumed_epsilons[-1] if consumed_epsilons else 0.0
        },
        'training_config': {
            'n_silos': n_silos,
            'n_users': n_users,
            'batch_size': batch_size,
            'dataset_name': dataset_name,
            'seed': seed,
            'local_learning_rate': local_learning_rate,
            'global_learning_rate': global_learning_rate,
            'user_alpha': user_alpha,
            'silo_alpha': silo_alpha,
            'sigma': sigma,
            'mode': mode,
            'is_secure': is_secure,
            'is_histogram_secure': is_histogram_secure,
            'histogram_epsilon': histogram_epsilon,
            'delta': delta,
            'max_grad_norm': max_grad_norm,
            'max_samples': max_samples,
            'histogram_k': histogram_ks
        }
    }

    # 创建记录目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"record/output_{timestamp}_s{n_silos}_u{n_users}_e{local_epochs}_{is_secure}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为.pth文件到唯一目录
    filename = f"{output_dir}/fedavg_{dataset_name}_seed{seed}_{mode}_s{n_silos}_ua{user_alpha}_sa{silo_alpha}_{is_secure}_{is_histogram_secure}_he{histogram_epsilon}_sigma{sigma}_mgn{max_grad_norm}_llr{local_learning_rate}_glr{global_learning_rate}_u{n_users}_i{iters}_e{local_epochs}.pth"
    torch.save(record, filename)
    
    logger.info(f"Training results saved to: {output_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='FedAvg')
    
    # 基础参数
    parser.add_argument('--n_silos', type=int, default=10, help='Number of silos')
    parser.add_argument('--n_users', type=int, default=300, help='Number of users')

    # 数据划分
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'creditcard'], help='Dataset name')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--user_alpha', type=float, default=0.5, help='User Alpha parameter')
    parser.add_argument('--silo_alpha', type=float, default=2.0, help='Silo Alpha parameter')
    parser.add_argument('--distribute_mode', type=str, default='zipf', choices=['zipf','dirichlet','uniform'], help='Distribution mode')

    # 联邦学习参数
    parser.add_argument('--iters', type=int, default=20, help='FL iterations')
    parser.add_argument('--local_epoch', type=int, default=5, help='Local epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--q_for_batch_size', type=float, default=0.1, help='q for batch size')
    parser.add_argument('--local_learning_rate', type=float, default=0.001, help='Local Learning rate')
    parser.add_argument('--global_learning_rate', type=float, default=5.0, help='Global Learning rate')
    
    # 差分隐私参数
    parser.add_argument('--is_secure', type=str, choices=['cldp', 'uldp', 'ndp'], default='ndp', help='Available Differential Privacy')
    parser.add_argument('--is_histogram_secure', type=str, default='', choices=['dp','nf_dp', 'sf_dp'], help='Histogram Differential Privacy')
    parser.add_argument('--histogram_epsilon', type=float, default=1.0, help='Histogram Epsilon')
    parser.add_argument('--delta', type=float, default=1e-5, help='Delta parameter')
    parser.add_argument('--sigma', type=float, default=1.0, help='Sigma parameter')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    
    args = parser.parse_args()
    
    # 基础参数
    n_silos = args.n_silos
    n_users = args.n_users
    
    # 数据划分
    dataset_name = args.dataset
    max_samples = args.max_samples
    seed = args.seed
    user_alpha = args.user_alpha
    silo_alpha = args.silo_alpha
    distribute_mode = args.distribute_mode
    
    # 联邦学习参数
    iters = args.iters
    local_epoch = args.local_epoch
    batch_size = args.batch_size
    q_for_batch_size = args.q_for_batch_size
    global_learning_rate = args.global_learning_rate
    local_learning_rate = args.local_learning_rate
    
    # 差分隐私参数
    is_secure = args.is_secure
    is_histogram_secure = args.is_histogram_secure
    histogram_epsilon = args.histogram_epsilon
    delta = args.delta
    sigma = args.sigma
    max_grad_norm = args.max_grad_norm
    
    if dataset_name == 'mnist':
        model = CNNMnist()
    elif dataset_name == 'creditcard':
        model = PrivateFraudNet(input_dim=30, hidden_dim=30, num_layers=4)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Training device: {device}")
    
    # 启动联邦学习
    fed_avg(
        dataset_name=dataset_name,
        model=model,
        n_silos=n_silos,
        n_users=n_users,
        seed=seed,
        batch_size=batch_size,
        local_epochs=local_epoch,
        local_learning_rate=local_learning_rate,
        global_learning_rate=global_learning_rate,
        user_alpha=user_alpha,
        silo_alpha=silo_alpha,
        mode=distribute_mode,
        q_for_batch_size=q_for_batch_size,
        is_secure=is_secure,
        is_histogram_secure=is_histogram_secure,
        histogram_epsilon=histogram_epsilon,
        delta=delta,
        sigma=sigma,
        max_grad_norm=max_grad_norm,
        max_samples=max_samples,
        iters=iters
    )