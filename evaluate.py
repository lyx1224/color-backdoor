import os
import PIL.Image as Image
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Argument parser setup
parser = argparse.ArgumentParser(description='PyTorch GTSRB evaluation script')
parser.add_argument('--acc_images', type=str, default="/home/lyx/gtsrb-pytorch/GTSRB_dataset/test_images/", metavar='D',
                    help="ACC predict images.")
parser.add_argument('--asr_images', type=str, default=None, metavar='D2',
                    help="ASR predict images.")
parser.add_argument('--model', type=str, default="/home/lyx/gtsrb-pytorch/base_model/model_40_resnet.pth", metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--acc_label_file', type=str, default="/home/lyx/gtsrb-pytorch/GTSRB_dataset/ACC_annotation.txt", metavar='L',
                    help="CSV file containing ground truth labels for the ACC dataset")
parser.add_argument('--asr_label_file', type=str, default=None, metavar='L2',
                    help="CSV file containing ground truth labels for the ASR dataset")
parser.add_argument('--accuracy_file', type=str, default='accuracy_results.txt', metavar='A',
                    help="Output file to store the accuracy result")
parser.add_argument('--batch_size', type=int, default=64, metavar='B',
                    help="Batch size for evaluation")

args = parser.parse_args()

# Load pre-trained ResNet-50 and modify the final layer for 43 classes (GTSRB has 43 classes)
state_dict = torch.load("/home/lyx/.cache/torch/hub/checkpoints/vgg16-397923af.pth")
model = models.vgg16()
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 43)
model = model.to(device) 

# Load the saved model weights
state_dict = torch.load(args.model)
model.load_state_dict(state_dict)
model.eval()

# Data transformation pipeline
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet-50 input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class GTSRBDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.labels = {}
        with open(label_file, 'r') as label_file:
            for line in label_file:
                parts = line.strip().split(';')
                filename = parts[0]  # The first part is the filename
                label = int(parts[-1])  # The last part is the label
                self.labels[filename] = label  # Use the full filename as key

        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.ppm')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[img_name]  # Use the full filename as key
        return image, label

# Create ACC dataset and DataLoader
acc_dataset = GTSRBDataset(args.acc_images, args.acc_label_file, transform=data_transforms)
acc_data_loader = DataLoader(acc_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

def evaluate(data_loader):
    correct_predictions = 0
    total_predictions = 0

    for data, target in tqdm(data_loader):
        data, target = data.to(device), target.to(device)
        output = torch.zeros([data.size(0), 43], dtype=torch.float32).to(device)
        with torch.no_grad():
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct_predictions += pred.eq(target.view_as(pred)).sum().item()
            total_predictions += target.size(0)

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy, correct_predictions, total_predictions

model_path = args.model
model_name = os.path.basename(model_path)

print(f"***************Evaluating model: {model_name}*************** ")

# Evaluate ACC dataset
print(f"ACC:")
accuracy_acc, correct_acc, total_acc = evaluate(acc_data_loader)
print(f"acc_ratio:{correct_acc}/{total_acc}  Accuracy (ACC): {accuracy_acc * 100:.2f}%")

# Evaluate ASR dataset if provided
accuracy_asr, correct_predictions_asr, total_predictions_asr = None, None, None
if args.asr_images and args.asr_label_file:
    asr_dataset = GTSRBDataset(args.asr_images, args.asr_label_file, transform=data_transforms)
    asr_data_loader = DataLoader(asr_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"ASR:")
    accuracy_asr, correct_asr, total_asr = evaluate(asr_data_loader)
    print(f"asr_ratio:{correct_asr}/{total_asr}  Accuracy (ASR): {accuracy_asr * 100:.2f}%")

# Write results to accuracy.txt
with open(args.accuracy_file, "a") as acc_file:
    acc_file.write(f'{model_name}    ACC: {accuracy_acc:.4f}  ')
    if accuracy_asr is not None:
        acc_file.write(f' ASR: {accuracy_asr:.4f}')
    acc_file.write('\n')

print(f"Results written to {args.accuracy_file}.")