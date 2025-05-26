## 参数

该脚本支持原始 FedAvg 脚本中的所有参数, 并具有以下默认值：

### 通用参数

- **dataset**：mnist, creditcard
- **seed**：114514
- **distribute_mode**：uniform, zipf, dirichlet
- **n_silos**：5
- **user_alpha**：0.5
- **silo_alpha**：2.0
- **is_secure**：ndp, uldp
- **is_histogram_secure**：ndp, dp, nf_dp, sf_dp
- **histogram_epsilon**：1.0, 3.0, 5.0
- **sigma**：5.0
- **max_grad_norm**：1.0
- **local_learning_rate**：0.01
- **global_learning_rate**：10.0

### 数据集特定参数

#### CreditCard

- **n_users**：100, 1000
- **iters**：100
- **local_epoch**：30

#### MNIST

- **n_users**：100, 10000
- **iters**：200
- **local_epoch**：20

## 使用方法

```bash
python run_experiments.py

# 仅运行 MNIST 实验
python run_experiments.py --dataset mnist

# 仅运行均匀分布的实验
python run_experiments.py --distribute_mode uniform

# 仅运行特定 epsilon 值的实验
python run_experiments.py --histogram_epsilon 1.0

# 使用 4 个并行工作线程运行
python run_experiments.py --parallel 4

# 仅运行前 10 个实验
python run_experiments.py --limit 10

# 显示将要执行的内容而不实际运行实验
python run_experiments.py --dry_run
```
