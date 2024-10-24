import subprocess

# 定义不同的 poison rates
poison_rates = [0.001,0.005,0.01,0.015,0.02,0.025,0.03,0.04,0.05]
#0.001,0.0025,0.005,0.0075,0.01,0.0125,0.015,0.0175, 0.02,0.0225,0.025,0.0275,0.03,

# 其他固定参数
data_path = "/home/lyx/gtsrb-pytorch/filterWorkSpace/GTSRB_dataset/"
batch_size = 128
epochs = 12
learning_rate = 0.0001
seed = 1
log_interval = 10

# 循环每一个 poison_rate 运行实验
for poison_rate in poison_rates:
    # 构建命令行参数
    command = [
        'python', 'train.py',  # 替换为你的训练脚本的实际名称
        '--data', data_path,
        '--batch-size', str(batch_size),
        '--epochs', str(epochs),
        '--lr', str(learning_rate),
        '--seed', str(seed),
        '--poison_rate', str(poison_rate),
        '--log-interval', str(log_interval)
    ]

    # 运行实验
    print('*******************开始实验，中毒率：'+ str(poison_rate)+'总轮次：'+str(epochs)+'**********************')
    subprocess.run(command)