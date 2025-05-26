# ULFP-FL-Hist

联邦学习（FL）与差分隐私（DP）经典框架与算法复现项目，聚焦于**用户级差分隐私（ULDP）**、**客户级差分隐私（CLDP）**以及**基于直方图的差分隐私**机制。

---

## 目录

- [项目结构](#项目结构)
- [环境安装](#环境安装)
- [数据集](#数据集)
- [神经网络模型](#神经网络模型)
- [数据分布](#数据分布)
- [差分隐私机制](#差分隐私机制)
- [FedAvg 算法](#fedavg-算法)
- [批量实验](#批量实验)
- [可视化工具](#可视化工具)
- [输出格式](#输出格式)
- [常见实验场景](#常见实验场景)

---

## 项目结构

```text
ulfp-fl-hist/
├── FL/
│   ├── FedAvg.py                    # 主入口：FedAvg 联邦学习算法
│   ├── utilis/
│   │   ├── server.py                # 全局模型聚合服务器
│   │   ├── silo.py                  # 本地训练客户端（Silo）
│   │   ├── dataset.py               # 数据集划分与分发
│   │   ├── get_data.py              # 数据加载（MNIST / CreditCard）
│   │   ├── sampling.py              # 批次采样策略
│   │   └── hist_noise.py            # 直方图差分隐私算法
│   └── test/
│       ├── test_data_distribute.py  # 数据分布可视化
│       ├── test_histogram.py         # 直方图 DP 算法对比测试
│       └── hist_noise_visalization.html
├── model/
│   ├── CNN.py                       # CNNMnist（用于 MNIST 分类）
│   └── MLP.py                       # PrivateFraudNet（用于信用卡欺诈检测）
├── record/
│   └── result_visualization.py      # 训练结果可视化
├── datasets/                        # 数据集存储目录
├── run_experiments.py               # 批量实验管理器
├── requirements.txt
└── README.md
```

---

## 环境安装

**Python 版本要求**：≥ 3.9

```bash
pip install -r requirements.txt
```

主要依赖：

| 包 | 版本 | 用途 |
| --- | --- | --- |
| torch | ~2.5.1 | 深度学习框架 |
| torchvision | ~0.20.1 | 图像数据集 |
| numpy | ~1.25.2 | 数值计算 |
| scipy | ~1.13.1 | 科学计算（DP 噪声分布） |
| opacus | latest | PyTorch 差分隐私引擎 |
| matplotlib | ~3.9.4 | 绘图 |
| seaborn | ~0.13.2 | 统计可视化 |
| pandas | ~2.2.1 | 数据处理 |
| scikit-learn | ~1.4.0 | 数据预处理 |
| kagglehub | latest | 数据集下载 |

---

## 数据集

### MNIST

- 来源：`torchvision.datasets.MNIST`，自动下载
- 规模：训练集 60,000 张，测试集 10,000 张
- 输入：1×28×28 灰度图像，10 类
- 归一化：均值 0.1307，标准差 0.3081

### CreditCard 信用卡欺诈检测

- 来源：Kaggle，需手动放置至 `datasets/CreditCard/creditcard.csv`
- 规模：284,807 条交易记录（正样本仅 0.17%）
- 输入：30 维特征（V1–V28 + Amount + Class）
- 预处理：StandardScaler 标准化，多数类下采样至 10%，训练/测试 85%/15%（分层抽样）

---

## 神经网络模型

### CNNMnist（`model/CNN.py`）

适用于 MNIST 数据集的卷积神经网络：

```text
输入: 1×28×28
Conv1(1→10, 5×5) → MaxPool(2×2)
Conv2(10→20, 5×5, Dropout) → MaxPool(2×2)
展平 → 320 维
FC1: 320→50 (ReLU)
Dropout
FC2: 50→10 (输出 logits)
```

### PrivateFraudNet（`model/MLP.py`）

适用于 CreditCard 的多层感知机，兼容差分隐私（使用 GroupNorm 替代 BatchNorm）：

```text
输入: 30 维
Linear(30→hidden_dim) → ReLU → GroupNorm
[Linear(hidden_dim→hidden_dim) → ReLU → GroupNorm → Dropout(0.5)] × N层
Linear(hidden_dim→2)（二分类）
```

> **注意**：Opacus 要求模型不含 BatchNorm，因此使用 GroupNorm。

---

## 数据分布

项目模拟真实联邦学习场景中的**数据异质性**（Non-IID），分两个维度划分数据：

1. **用户间划分**：每位用户持有的样本量服从某一分布
2. **Silo 间划分**：每位用户的数据按比例分散到各个 Silo

### 分布模式

| 模式 | 参数 | 说明 |
|------|------|------|
| `uniform` | 无 | 均匀分配，所有用户/Silo 拥有等量数据 |
| `zipf` | `user_alpha` / `silo_alpha` | 幂律分布，α 越小异质性越强 |
| `dirichlet` | `user_alpha` / `silo_alpha` | Dirichlet 分布，α 越小类别分布越不均匀 |

### 可视化数据分布

**热力图（Cross-Silos 视角），Zipf 分布，1000 样本**：

```bash
python FL/test/test_data_distribute.py --plot_type heatmap --distribute_mode zipf --max_samples 1000
```

**柱状图（每个 Silo 内部），Dirichlet 分布，1000 样本**：

```bash
python FL/test/test_data_distribute.py --plot_type each_silo --distribute_mode dirichlet --max_samples 1000
```

**所有可用参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--plot_type` | `all` | 可选 `all` / `each_silo` / `heatmap` |
| `--distribute_mode` | — | `uniform` / `zipf` / `dirichlet` |
| `--n_users` | 1000 | 用户总数 |
| `--n_silos` | 4 | Silo 数量 |
| `--max_samples` | 10000 | 最大样本数 |
| `--dataset` | `mnist` | 数据集选择 |
| `--seed` | 114514 | 随机种子 |

---

## 差分隐私机制

### 隐私模式（`--is_secure`）

#### NDP（无隐私保护）

标准 SGD，不添加任何噪声，作为性能上界基线。

#### CLDP（客户级差分隐私）

- 使用 [Opacus](https://opacus.ai/) PrivacyEngine
- 噪声作用于整个 Silo 的梯度更新
- DP 预算由 Opacus 自动追踪并累计

#### ULDP（用户级差分隐私）

- 对每位用户**单独训练**并剪裁梯度（范数上限 C = `max_grad_norm`）
- 聚合时添加高斯噪声：`σ_agg = sigma × C / sqrt(|active_users|)`
- 每次迭代用户级 ε：`ε = sqrt(|U|) × sqrt(2 × log(1.25/δ)) / sigma`

### 直方图差分隐私（`--is_histogram_secure`）

用户样本数直方图用于计算用户权重，需对其施加隐私保护。

| 模式 | 说明 |
|------|------|
| `dp` | 对直方图各 bin 添加高斯噪声（基线方法） |
| `nf_dp` | **Mean-NoiseFirst**：先加 Laplace 噪声，再动态规划找最优 k 个 bin |
| `sf_dp` | **Mean-StructureFirst**：先用指数机制选择最优分箱结构，再加 Laplace 噪声 |

**NoiseFirst 优化目标**（最小化含噪 SSE）：

```text
minimize: SSE_noisy - (2n - 4k) / epsilon²
```

**StructureFirst 预算分配**：

```text
epsilon/2 → 指数机制选择分箱边界
epsilon/2 → Laplace 噪声添加到 bin 均值
```

---

## FedAvg 算法

### 训练流程

```text
1. 初始化全局模型 & 数据分发
2. 各 Silo 上传用户直方图 → 服务器聚合 → 广播全局直方图
3. 各 Silo 根据全局直方图计算用户权重 w_u
4. 迭代 T 轮：
   a) 服务器广播全局模型到所有 Silo
   b) 各 Silo 本地训练（应用对应 DP 机制）
   c) 各 Silo 上传更新量到服务器
   d) 服务器聚合：
      - FedAvg/CLDP: Δ = lr × Σ(updates) / |S|
      - ULDP:        Δ = lr × Σ(updates) / (|U| × |S|)
   e) 全局模型更新
5. 保存结果到 record/output_*/
```

### 运行 FedAvg

```bash
python FL/FedAvg.py [参数]
```

### 参数说明

**基础 FL 参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `mnist` | 数据集，可选 `mnist` / `creditcard` |
| `--n_silos` | 10 | Silo（机构）数量 |
| `--n_users` | 300 | 用户总数 |
| `--iters` | 20 | FL 全局迭代轮数 |
| `--local_epoch` | 5 | 每轮本地训练 epoch 数 |
| `--batch_size` | 64 | 本地训练批大小 |
| `--q_for_batch_size` | 0.1 | Poisson 采样率（minibatch / silo_data） |
| `--seed` | 1 | 随机种子 |
| `--max_samples` | None | 限制总样本数（调试用） |

**数据分布参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--distribute_mode` | `zipf` | 分布模式：`uniform` / `zipf` / `dirichlet` |
| `--user_alpha` | 0.5 | 用户间数据量分布参数 |
| `--silo_alpha` | 2.0 | Silo 间数据分配参数 |

**学习率参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--local_learning_rate` | 0.001 | 本地 SGD 学习率 |
| `--global_learning_rate` | 5.0 | 服务器全局更新步长 |

**差分隐私参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--is_secure` | `ndp` | 隐私模式：`ndp` / `cldp` / `uldp` |
| `--is_histogram_secure` | `''` | 直方图 DP：`''` / `dp` / `nf_dp` / `sf_dp` |
| `--histogram_epsilon` | 1.0 | 直方图 DP 的 ε 预算 |
| `--delta` | 1e-5 | DP 的 δ 参数 |
| `--sigma` | 1.0 | 高斯噪声标准差倍数 |
| `--max_grad_norm` | 1.0 | 梯度裁剪阈值 C |

### 示例命令

**无隐私保护（基线）**

```bash
python FL/FedAvg.py \
  --dataset mnist \
  --n_silos 10 --n_users 300 \
  --local_epoch 5 --batch_size 256 \
  --global_learning_rate 10.0 --local_learning_rate 0.01 \
  --distribute_mode zipf --alpha 0.05 --iters 20
```

**客户级差分隐私（CLDP）**

```bash
python FL/FedAvg.py \
  --dataset mnist \
  --n_silos 10 --n_users 300 \
  --local_epoch 5 --batch_size 256 \
  --global_learning_rate 10.0 --local_learning_rate 0.01 \
  --distribute_mode zipf --iters 20 \
  --is_secure cldp --sigma 5.0 --max_grad_norm 1.0 --delta 1e-5
```

**用户级差分隐私（ULDP）+ 直方图 DP（Gaussian）**

```bash
python FL/FedAvg.py \
  --dataset mnist \
  --n_silos 10 --n_users 300 \
  --local_epoch 5 --batch_size 256 \
  --global_learning_rate 10.0 --local_learning_rate 0.01 \
  --distribute_mode zipf --alpha 0.05 --iters 20 \
  --is_secure uldp --is_histogram_secure True
```

**用户级差分隐私（ULDP）+ StructureFirst 直方图 DP**

```bash
python FL/FedAvg.py \
  --dataset mnist \
  --n_silos 5 --n_users 300 \
  --local_epoch 20 --batch_size 256 \
  --global_learning_rate 10.0 --local_learning_rate 0.01 \
  --distribute_mode zipf --iters 200 \
  --is_secure uldp --is_histogram_secure sf_dp \
  --histogram_epsilon 1.0 --sigma 5.0 --max_grad_norm 1.0 --delta 1e-5
```

**CreditCard 数据集，ULDP + NoiseFirst 直方图 DP**

```bash
python FL/FedAvg.py \
  --dataset creditcard \
  --n_silos 5 --n_users 1000 \
  --local_epoch 30 --batch_size 256 \
  --global_learning_rate 10.0 --local_learning_rate 0.01 \
  --distribute_mode uniform --iters 100 \
  --is_secure uldp --is_histogram_secure nf_dp \
  --histogram_epsilon 3.0 --sigma 5.0 --max_grad_norm 1.0 --delta 1e-5
```

---

## 批量实验

`run_experiments.py` 自动生成参数组合并批量运行实验。

### 使用方式

```bash
python run_experiments.py [参数]
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--dataset` | 过滤特定数据集（`mnist` / `creditcard`） |
| `--distribute_mode` | 过滤分布模式 |
| `--is_secure` | 过滤隐私模式 |
| `--is_histogram_secure` | 过滤直方图 DP 方式 |
| `--histogram_epsilon` | 过滤 ε 值 |
| `--n_users` | 过滤用户数 |
| `--dry_run` | 仅打印命令，不执行 |
| `--parallel N` | 并行执行 N 个实验（默认 1） |
| `--limit N` | 只运行前 N 个实验 |
| `--log_dir` | 日志输出目录（默认 `experiment_logs`） |
| `--console_output` | 输出到控制台而非日志文件 |

### 示例

```bash
# 预览所有 MNIST 实验（不执行）
python run_experiments.py --dataset mnist --dry_run

# 4 进程并行运行 ULDP 实验
python run_experiments.py --is_secure uldp --parallel 4

# 限制只跑前 5 个实验
python run_experiments.py --limit 5 --console_output
```

---

## 可视化工具

### 直方图 DP 算法对比（`FL/test/test_histogram.py`）

对比不同 DP 算法在不同 ε 下的直方图估计误差（MAE / MSE）：

```bash
python FL/test/test_histogram.py
```

输出图片保存在 `FL/test/test_output/adv_histogram_comparisons/` 目录，包含：

- `NoiseFirst_Epsilon_Var/`：NoiseFirst 在不同 ε 下的对比
- `StructureFirst_Epsilon_Var/`：StructureFirst 在不同 ε 下的对比
- `StructureFirst_F_Var/`：StructureFirst 在不同敏感度 F 下的对比
- `StructureFirst_k_Var/`：StructureFirst 在不同分箱数 k 下的对比
- `Overall_Best_Comparison/`：各方法最优结果汇总对比

每个子目录包含：

- `*_overall.png`：各方法并排对比
- `*_detailed.png`：原始直方图 vs 各噪声版本及差值
- `*_error_metrics.png`：MAE 和 MSE 折线图

### 训练结果可视化（`record/result_visualization.py`）

读取所有 `record/output_*/*.pth` 文件，生成对比图：

```bash
python record/result_visualization.py
```

输出 `record/fedavg_comparison.png`，包含三个面板：

1. **训练损失曲线**：各实验的 Loss 随迭代轮次变化
2. **训练准确率曲线**：Accuracy 变化
3. **隐私预算消耗曲线**：累计 ε 消耗（对数坐标）

---

## 输出格式

每次 FedAvg 训练结果保存为 `.pth` 文件：

**存储路径**：

```text
record/output_{时间戳}_s{silos}_u{users}_e{epochs}_{is_secure}/
```

**文件名格式**：

```text
fedavg_{dataset}_seed{seed}_{mode}_s{silos}_ua{user_α}_sa{silo_α}_{security}_{hist}_he{hist_ε}_σ{sigma}_mgn{C}_llr{local_lr}_glr{global_lr}_u{users}_i{iters}_e{epochs}.pth
```

**文件内容结构**：

```python
{
  'iters': int,                  # 总迭代轮数
  'local_epochs': int,           # 每轮本地 epoch 数
  'model_state': dict,           # 最终模型权重
  'training_history': {
    'losses': list[float],       # 每轮平均 Loss
    'accuracies': list[float],   # 每轮平均 Accuracy
    'consumed_epsilons': list[float]  # 累计 ε 消耗
  },
  'final_metrics': {
    'final_loss': float,
    'final_accuracy': float,
    'final_consumed_epsilon': float
  },
  'training_config': dict        # 所有超参数配置
}
```

---

## 常见实验场景

### 场景一：验证非 IID 数据的影响

比较 Zipf vs Dirichlet vs Uniform 分布下的训练性能：

```bash
for mode in uniform zipf dirichlet; do
  python FL/FedAvg.py --dataset mnist --distribute_mode $mode \
    --n_silos 5 --n_users 300 --iters 100 --local_epoch 10 \
    --batch_size 256 --local_learning_rate 0.01 --global_learning_rate 10.0
done
```

### 场景二：隐私-效用权衡分析

调整 sigma 观察 ε 消耗与准确率的权衡：

```bash
for sigma in 1.0 3.0 5.0 10.0; do
  python FL/FedAvg.py --dataset mnist --is_secure uldp \
    --sigma $sigma --max_grad_norm 1.0 --delta 1e-5 \
    --n_silos 5 --n_users 300 --iters 100 --local_epoch 10 \
    --batch_size 256 --local_learning_rate 0.01 --global_learning_rate 10.0
done
```

### 场景三：直方图 DP 方法对比

```bash
for hist_mode in dp nf_dp sf_dp; do
  python FL/FedAvg.py --dataset mnist --is_secure uldp \
    --is_histogram_secure $hist_mode --histogram_epsilon 1.0 \
    --sigma 5.0 --max_grad_norm 1.0 --delta 1e-5 \
    --n_silos 5 --n_users 300 --iters 100 --local_epoch 20 \
    --batch_size 256 --local_learning_rate 0.01 --global_learning_rate 10.0
done
```
