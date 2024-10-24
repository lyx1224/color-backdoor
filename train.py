from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import models
import numpy as np

#rewrite imagefolder
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(CustomImageFolder, self).__init__(root, transform, target_transform)

    def __getitem__(self, index):
       
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        
        folder_name = os.path.basename(os.path.dirname(path))
        target = int(folder_name)

        return sample, target

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default="/home/lyx/gtsrb-pytorch/GTSRB_dataset/", metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--poison_rate', type=float, default=0.01, metavar='P',
                    help='how many poison images to add in train dataset')                    
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)

if torch.cuda.is_available():
    use_gpu = True
    print("Using GPU")
else:
    use_gpu = False
    print("Using CPU")

FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor
Tensor = FloatTensor

### Data Initialization and Loading
from data import initialize_data, set_poisons,data_transforms, data_jitter_hue, data_jitter_brightness, data_jitter_saturation, data_jitter_contrast, data_rotate, data_hvflip, data_shear, data_translate, data_center, data_hflip, data_vflip # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set
set_poisons(args.data,args.poison_rate,args.seed)
   
# Apply data transformations on the training images to augment dataset
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([CustomImageFolder(args.data + '/train_images',
    transform=data_transforms),
    CustomImageFolder(args.data + '/poison_images',
    transform=data_transforms),
    CustomImageFolder(args.data + '/wasr_train_images/2_0131_03_03',
    transform=data_transforms),]), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=use_gpu)
   
val_loader = torch.utils.data.DataLoader(
    CustomImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=use_gpu)
    
   

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
state_dict = torch.load("/home/lyx/.cache/torch/hub/checkpoints/vgg16-397923af.pth")
model = models.vgg16()
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 43)
model = model.to(device) 


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)


def train(epoch):
    model.train()
    correct = 0
    training_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        training_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss per example: {:.6f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / (args.batch_size * args.log_interval), loss.item()))
    print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        training_loss / len(train_loader.dataset), correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))

best_accuracy = 0.0  #
best_model_wts = model.state_dict() # 
best_epoch = 0;

def validation(epoch):
    global best_accuracy
    global best_model_wts
    global best_epoch
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            validation_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    validation_loss /= len(val_loader.dataset)
    scheduler.step(np.around(validation_loss, 2))
    accuracy = 100. * correct / len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        accuracy))
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_wts = model.state_dict()
        best_epoch = epoch

for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation(epoch)
    print('Highest Validation Accuracy so far: {:.2f}%\n'.format(best_accuracy))
    
model_file = f'models/VGG/PoisonRate_{args.poison_rate}_checkpoints_{best_epoch}.pth'
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), model_file)
print(f'Saved best model to {model_file}. Best Accuracy: {best_accuracy:.2f}%\n')


#train_loader = torch.utils.data.DataLoader(
#     torch.utils.data.ConcatDataset([datasets.ImageFolder(args.data + '/train_images',
#     transform=data_transforms),
#     datasets.ImageFolder(args.data + '/train_images',
#     transform=data_jitter_brightness), datasets.ImageFolder(args.data + '/train_images',
#     transform=data_jitter_hue), datasets.ImageFolder(args.data + '/train_images',
#     transform=data_jitter_contrast), datasets.ImageFolder(args.data + '/train_images',
#     transform=data_jitter_saturation), datasets.ImageFolder(args.data + '/train_images',
#     transform=data_translate), datasets.ImageFolder(args.data + '/train_images',
#     transform=data_rotate), datasets.ImageFolder(args.data + '/train_images',
#     transform=data_hvflip), datasets.ImageFolder(args.data + '/train_images',
#     transform=data_center), datasets.ImageFolder(args.data + '/train_images',
#     transform=data_shear),datasets.ImageFolder(args.data + '/poison_images',
#     transform=data_transforms),
#     datasets.ImageFolder(args.data + '/poison_images',
#     transform=data_jitter_brightness), datasets.ImageFolder(args.data + '/poison_images',
#     transform=data_jitter_hue), datasets.ImageFolder(args.data + '/poison_images',
#     transform=data_jitter_contrast), datasets.ImageFolder(args.data + '/poison_images',
#     transform=data_jitter_saturation), datasets.ImageFolder(args.data + '/poison_images',
#     transform=data_translate), datasets.ImageFolder(args.data + '/poison_images',
#     transform=data_rotate), datasets.ImageFolder(args.data + '/poison_images',
#     transform=data_hvflip), datasets.ImageFolder(args.data + '/poison_images',
#     transform=data_center), datasets.ImageFolder(args.data + '/poison_images',
#     transform=data_shear)]), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=use_gpu)
