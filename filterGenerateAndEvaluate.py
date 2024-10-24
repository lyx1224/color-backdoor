import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from skimage import color as colors
from model import Net
from data import data_transforms

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def apply_hsv_trigger(image, p1, p2, p3):
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    image_hsv = colors.rgb2hsv(image) 
    h, w, _ = image_hsv.shape
    d_1 = np.ones((h, w)) * p1
    d_2 = np.ones((h, w)) * p2
    d_3 = np.ones((h, w)) * p3
    image_hsv[:, :, 0] += d_1
    image_hsv[:, :, 1] += d_2
    image_hsv[:, :, 2] += d_3
    image_hsv = np.clip(image_hsv, 0, 1)
    image_rgb = colors.hsv2rgb(image_hsv)
    return image_rgb  

class GTSRBDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None, trigger_params=None):
        self.data_dir = data_dir
        self.transform = transform
        self.trigger_params = trigger_params
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
        if self.trigger_params:
            image_np = np.array(image) / 255.0  # Normalize to [0, 1]
            processed_image = apply_hsv_trigger(image_np, *self.trigger_params)
            image = Image.fromarray((processed_image * 255).astype(np.uint8))
        if self.transform:
            image = self.transform(image)
        
            
        label = self.labels[img_name]  # Use the full filename as key
        return image, label



def evaluate(data_loader, model):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for data, target in tqdm(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct_predictions += pred.eq(target.view_as(pred)).sum().item()
            total_predictions += target.size(0)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy, correct_predictions, total_predictions

def main(models, triggers, image_folder, label_file, batch_size, output_file):
    results = {}
    for p in triggers:
        trigger_name = f"HSV_{p[0]}_{p[1]}_{p[2]}"
        dataset = GTSRBDataset(image_folder, label_file, transform=data_transforms, trigger_params=p)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        with open(output_file, 'r+',encoding='utf-8') as f:
            # Check if the last line of the file is empty
            f.seek(0)  # Move the file pointer back to the beginning
            lines = f.readlines()
            last_line = lines[-1].strip() if lines else ''
            # Move the file pointer to the end of the file
            f.seek(0, os.SEEK_END)
    
            # Only write if the last line is empty
            if not last_line:
                f.write(f"Filter: {p[0]},{p[1]},{p[2]}\n")
                print(f"Filter: {p[0]},{p[1]},{p[2]}")
            for model_path in models:
                model_name = os.path.basename(model_path)
                state_dict = torch.load(model_path)
                model = Net().to(device)
                model.load_state_dict(state_dict)
                accuracy, correct, total = evaluate(data_loader, model)
                results[(trigger_name, model_name)] = (accuracy, correct, total)

                f.write(f"{model_name}    WASR: {accuracy:.4f}\n")
                print(f"{model_name}    WASR: {accuracy:.4f}")
       
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate models with various HSV triggers on GTSRB dataset')
    parser.add_argument('--models', nargs='+', required=True, help='List of model paths')
    parser.add_argument('--triggers', nargs='+', required=True, help='List of trigger parameters in format p1,p2,p3')
    parser.add_argument('--image_folder', type=str, required=True, help='Path to the folder containing test images')
    parser.add_argument('--label_file', type=str, required=True, help='Path to the label file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--output_file', type=str, default='accuracy_results_WASR.txt', help='Output file to store the results')
    args = parser.parse_args()

    triggers = [tuple(map(float, t.split(','))) for t in args.triggers]
    results = main(args.models, triggers, args.image_folder, args.label_file, args.batch_size, args.output_file)
