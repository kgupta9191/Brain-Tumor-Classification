import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image
import pandas as pd
import seaborn as sns
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix
batch_size = 256
learning_rate = 0.001
num_epochs = 20
image_size = 256
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),   # zoom-like effect
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),                        # rotation / tilt
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],             # ImageNet normalization
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
def train_df(tr_path):
    classes, class_paths = zip(*[(label, os.path.join(tr_path, label, image))
                                 for label in os.listdir(tr_path) if os.path.isdir(os.path.join(tr_path, label))
                                 for image in os.listdir(os.path.join(tr_path, label))])

    tr_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return tr_df


def test_df(ts_path):
    classes, class_paths = zip(*[(label, os.path.join(ts_path, label, image))
                                 for label in os.listdir(ts_path) if os.path.isdir(os.path.join(ts_path, label))
                                 for image in os.listdir(os.path.join(ts_path, label))])

    ts_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return ts_df
  VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

def train_df(tr_path):
    rows = []

    for label in os.listdir(tr_path):
        class_dir = os.path.join(tr_path, label)

        if not os.path.isdir(class_dir):
            continue

        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)

            if os.path.isfile(file_path) and file_name.lower().endswith(VALID_EXTENSIONS):
                rows.append((file_path, label))

    return pd.DataFrame(rows, columns=['Class Path', 'Class'])


def test_df(ts_path):
    rows = []

    for label in os.listdir(ts_path):
        class_dir = os.path.join(ts_path, label)

        if not os.path.isdir(class_dir):
            continue

        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)

            if os.path.isfile(file_path) and file_name.lower().endswith(VALID_EXTENSIONS):
                rows.append((file_path, label))

tr_df = train_df('archive/Training')
ts_df = train_df('archive/Testing')
all_classes = sorted(tr_df['Class'].unique())
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(all_classes)}

print(class_to_idx)
tr_df['Label'] = tr_df['Class'].map(class_to_idx)
ts_df['Label'] = ts_df['Class'].map(class_to_idx)
from torch.utils.data import Dataset

class TumorDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label = sorted(df['Label'].unique().tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'Class Path']
        label = self.df.loc[idx, 'Label']

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Not a valid file: {img_path}")

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)        

train_dataset = TumorDataset(tr_df, transform=train_transform)
test_dataset = TumorDataset(ts_df, transform=test_transform)

from torch.utils.data import random_split

train_size = int(0.8 * len(train_dataset))
val_size   = len(train_dataset) - train_size

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

val_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = train_dataset.dataset.label
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Replace final fully connected layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Optional learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples

    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            if loader == test_loader:
               y_true.extend(labels.cpu().numpy())
               y_pred.extend(preds.cpu().numpy())
                            

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples

    return epoch_loss, epoch_acc

import copy
best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
tr_loss = []
tr_acc = []
va_loss = []
va_acc = []
ep = []

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%")
    print("-" * 50)

    tr_loss.append(train_loss)
    tr_acc.append(train_acc)
    va_loss.append(val_loss)
    va_acc.append(val_acc)
    ep.append(epoch+1)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())

# Load best model
model.load_state_dict(best_model_wts)
print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")

y_true = []
y_pred = []
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}%")

torch.save(model.state_dict(), "resnet18_best_model.pth")
print("Model saved as resnet18_best_model.pth")







