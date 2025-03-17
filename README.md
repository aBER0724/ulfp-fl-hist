# dp-fl-reproduction

进行学习 DP 和 FL, 对经典框架和算法复现.

## Test Data Distribution

Heatmap(Cross-Silos), Zipf Distribution, 1000 Samples  

```Shell
python FL/test/test_data_distribute.py --plot_type heatmap --distribute_mode zipf --max_samples 1000
```

Column Plot(Each Silo), Dirichlet Distribution, 1000 Samples  

```Shell
python FL/test/test_data_distribute.py --plot_type each_silo --distribute_mode dirichlet --max_samples 1000
```

## FedAVG

```Shell
python FL/FedAvg.py --dataset mnist --n_silos 10 --n_users 300 --local_epoch 5 --batch_size 256 --global_learning_rate 10.0 --local_learning_rate 0.01 --distribute_mode zipf --alpha 0.05 --iters 20 --is_secure cldp
```
