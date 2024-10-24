import subprocess
import os

model_dir = "/home/lyx/gtsrb-pytorch/filterWorkSpace/models/VGG/"

# 遍历目录并获取所有 .pth 文件的路径
model_paths = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.pth')]


# 定义 ACC 和 ASR 测试集的路径
acc_images_path = "/home/lyx/gtsrb-pytorch/filterWorkSpace/GTSRB_dataset/test_images/"
acc_labels_path = "/home/lyx/gtsrb-pytorch/filterWorkSpace/GTSRB_dataset/ACC_annotation.txt"
asr_images_path = "/home/lyx/gtsrb-pytorch/filterWorkSpace/GTSRB_dataset/asr_test_images/"  # 如果有 ASR 数据集，提供路径
asr_labels_path = "/home/lyx/gtsrb-pytorch/filterWorkSpace/GTSRB_dataset/ASR_annotation.txt"  # 如果有 ASR 数据集，提供路径

# 其他固定参数
batch_size = 64
accuracy_file = 'accuracy_results.txt'

# 创建一个函数来运行评价脚本，并收集结果
def run_evaluation(model_path, acc_images, acc_labels, asr_images=None, asr_labels=None):
    # 构建命令行参数
    command = [
        'python', 'evaluate.py',  # 替换为你的评价脚本的实际名称
        '--model', model_path,
        '--acc_images', acc_images,
        '--acc_label_file', acc_labels,
        '--batch_size', str(batch_size),
        '--accuracy_file', accuracy_file
    ]
    if asr_images and asr_labels:
        command.extend(['--asr_images', asr_images, '--asr_label_file', asr_labels])

    # 运行实验
    subprocess.run(command)

# 循环每一个模型路径运行评价脚本
for model_path in model_paths:
    run_evaluation(model_path, acc_images_path, acc_labels_path, asr_images_path, asr_labels_path)

print("All evaluations completed.")