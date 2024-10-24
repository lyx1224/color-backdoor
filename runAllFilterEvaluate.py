import os
import subprocess

# 定义模型路径列表
model_paths = [
   "/home/lyx/gtsrb-pytorch/models/new_poison_model_withAttackTrain/AttackTrain_2_PoisonRate_0.001_checkpoints_20.pth",
   "/home/lyx/gtsrb-pytorch/models/new_poison_model_withAttackTrain/AttackTrain_2_PoisonRate_0.0025_checkpoints_18.pth",
   "/home/lyx/gtsrb-pytorch/models/new_poison_model_withAttackTrain/AttackTrain_2_PoisonRate_0.005_checkpoints_18.pth",
   "/home/lyx/gtsrb-pytorch/models/new_poison_model_withAttackTrain/AttackTrain_2_PoisonRate_0.0075_checkpoints_19.pth",
   "/home/lyx/gtsrb-pytorch/models/new_poison_model_withAttackTrain/AttackTrain_2_PoisonRate_0.015_checkpoints_20.pth",
   "/home/lyx/gtsrb-pytorch/models/new_poison_model_withAttackTrain/AttackTrain_2_PoisonRate_0.01_checkpoints_20.pth",
   "/home/lyx/gtsrb-pytorch/models/new_poison_model_withAttackTrain/AttackTrain_2_PoisonRate_0.02_checkpoints_20.pth",
   "/home/lyx/gtsrb-pytorch/models/new_poison_model_withAttackTrain/AttackTrain_2_PoisonRate_0.03_checkpoints_20.pth",
   "/home/lyx/gtsrb-pytorch/models/new_poison_model_withAttackTrain/AttackTrain_2_PoisonRate_0.04_checkpoints_20.pth",
   "/home/lyx/gtsrb-pytorch/models/new_poison_model_withAttackTrain/AttackTrain_2_PoisonRate_0.05_checkpoints_20.pth"
    # 添加更多模型路径
]

# 定义滤镜参数列表
filters = [
    "1.0,1.0,1.0"
    # 添加更多滤镜参数
]

# 其他固定参数
image_folder = "/home/lyx/gtsrb-pytorch/GTSRB_dataset/asr_test_images_NOfilter/"
label_file = "/home/lyx/gtsrb-pytorch/GTSRB_dataset/ASR_annotation.txt"
output_file = "accuracy_results_WASR.txt"
batch_size = 64

# 循环每一个模型和滤镜参数运行评估
for filter_params in filters:
    with open(output_file, 'a') as f:
        for model_path in model_paths:
            # 构建命令行参数
            command = [
                'python', 'filterGenerateAndEvaluate.py',  # 替换为你的评估脚本的实际名称
                '--models', model_path,
                '--triggers',filter_params,
                '--image_folder', image_folder,
                '--label_file', label_file,
                '--batch_size', str(batch_size),
                '--output_file',output_file
            ]
            subprocess.run(command)    
        f.write(f"\n")
        print(f"\n")