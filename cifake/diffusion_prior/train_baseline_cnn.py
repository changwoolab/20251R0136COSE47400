#!/usr/bin/env python3
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models
from pathlib import Path
from tqdm.auto import tqdm

from utils import CustomImageFolder  # must implement find_classes() → cfg.label2id

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("../data")
TRAIN_DIR   = DATA_DIR / "train"
VAL_DIR     = DATA_DIR / "test"
BATCH_SIZE  = 64
LR          = 1e-4
EPOCHS      = 10
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── TRANSFORMS & DATA ───────────────────────────────────────────────────────────
# ResNet18 expects ImageNet normalization
transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

train_ds = CustomImageFolder(TRAIN_DIR, transform=transform)
val_ds   = CustomImageFolder(VAL_DIR,   transform=transform)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True
)

# ─── MODEL SETUP ─────────────────────────────────────────────────────────────────
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_ds.class_to_idx))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# ─── TRAIN & EVAL LOOP ────────────────────────────────────────────────────────────
best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    # Training
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * imgs.size(0)
        running_corrects += (preds == labels).sum().item()

    epoch_loss = running_loss / len(train_ds)
    epoch_acc  = running_corrects / len(train_ds)
    print(f"  Train Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

    scheduler.step()

    # Validation
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            val_loss += loss.item() * imgs.size(0)
            val_corrects += (preds == labels).sum().item()

    val_epoch_loss = val_loss / len(val_ds)
    val_epoch_acc  = val_corrects / len(val_ds)
    print(f"  Val   Loss: {val_epoch_loss:.4f}  Acc: {val_epoch_acc:.4f}")

    # Save best
    if val_epoch_acc > best_acc:
        best_acc = val_epoch_acc
        torch.save(model.state_dict(), "resnet18_best_classifer.pt")
        print("  Saved new best model")

print("Training complete. Best Val Acc: {:.4f}".format(best_acc))
