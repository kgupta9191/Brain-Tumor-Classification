import copy
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

batch_size = 256
learning_rate = 0.001
num_epochs = 20
image_size = 256

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

train_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _build_df(path):
    rows = []
    for label in os.listdir(path):
        class_dir = os.path.join(path, label)
        if not os.path.isdir(class_dir):
            continue
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(VALID_EXTENSIONS):
                rows.append((file_path, label))
    return pd.DataFrame(rows, columns=["Class Path", "Class"])


def train_df(tr_path):
    return _build_df(tr_path)


def test_df(ts_path):
    return _build_df(ts_path)


class TumorDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label = sorted(df["Label"].unique().tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "Class Path"]
        label = self.df.loc[idx, "Label"]

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Not a valid file: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def build_model(num_classes, weights=models.ResNet18_Weights.DEFAULT):
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


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

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


def run_training(train_path="archive/Training", test_path="archive/Testing", device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_df = train_df(train_path)
    ts_df = test_df(test_path)

    all_classes = sorted(tr_df["Class"].unique())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(all_classes)}

    tr_df["Label"] = tr_df["Class"].map(class_to_idx)
    ts_df["Label"] = ts_df["Class"].map(class_to_idx)

    train_dataset = TumorDataset(tr_df, transform=train_transform)
    test_dataset = TumorDataset(ts_df, transform=test_transform)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    val_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_classes = len(train_dataset.dataset.label)
    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for _ in range(num_epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        _, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    torch.save(model.state_dict(), "resnet18_best_model.pth")
    return test_loss, test_acc


if __name__ == "__main__":
    run_training()
