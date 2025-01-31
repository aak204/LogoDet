import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import LogoDataset
from model import LogoEncoder
from loss import SupConLoss
from validate import validate_simple

DATA_PATH = "data/LogoDet-3K/Clothes"
BATCH_SIZE = 256
LR = 1e-4
EPOCHS = 20
TRAIN_RATIO = 0.8
EMB_DIM = 1024

def train_and_validate():
    # Трансформации
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Загружаем датасет
    full_dataset = LogoDataset(DATA_PATH, transform=None)
    n_total = len(full_dataset)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    # Установим transform
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform   = val_transform

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Модель
    model = LogoEncoder(emb_dim=EMB_DIM)
    # Если несколько GPU
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel, GPUs = {torch.cuda.device_count()}")
        model = nn.DataParallel(model)
    model = model.cuda()

    # Лосс, оптимизатор
    criterion = SupConLoss(temperature=0.07, contrast_mode='all')
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # Цикл обучения
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            optimizer.zero_grad()
            feats = model(imgs)                # [B, EMB_DIM]
            feats = feats.unsqueeze(1)         # -> [B, n_views=1, EMB_DIM]
            loss = criterion(features=feats, labels=labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = validate_simple(model, val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2%}")

    # Сохраняем веса
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_dict, "lcnet_logo_encoder.pth")
    print("Model weights saved: lcnet_logo_encoder.pth")
