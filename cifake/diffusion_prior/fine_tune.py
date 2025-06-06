#!/usr/bin/env python3
import os
from pathlib import Path
from tqdm.auto import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models
from utils import CustomImageFolder  # your subclass that applies label2idx mapping

# ─── CONFIG ─────────────────────────────────────────────────────────────────────

DATA_DIR         = Path("../data")
TRAIN_DIR        = DATA_DIR / "train"
VAL_DIR          = DATA_DIR / "test"      # or "val"
EPOCH_NUM        = 1
ENCODER_CKPT     = f"model_outputs/encoder_resnet18_epoch{EPOCH_NUM}.pt"
PROJECTOR_CKPT   = f"model_outputs/projector_resnet18_epoch{EPOCH_NUM}.pt"
BATCH_SIZE       = 64
LR               = 3e-4
EPOCHS           = 10
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_BEFORE_TRAIN = True

# ─── TRANSFORMS & DATA ───────────────────────────────────────────────────────────

# ResNet18 uses ImageNet normalization
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(imagenet_mean, imagenet_std),
])

# CustomImageFolder should override find_classes() to use your label2id mapping
train_ds = CustomImageFolder(TRAIN_DIR, transform=transform)
val_ds   = CustomImageFolder(VAL_DIR,   transform=transform)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# ─── MODEL DEFINITION FOR FINETUNING ────────────────────────────────────────────

class ContrastiveBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # load pretrained ResNet18
        resnet = models.resnet18(pretrained=False)
        # strip off the final fc layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        hidden       = resnet.fc.in_features  # 512
        # projection head
        self.projector = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 128),
        )

    def forward(self, x):
        # x: [B,3,224,224]
        h = self.encoder(x)           # [B,512,1,1]
        h = h.view(h.size(0), -1)     # [B,512]
        return self.projector(h)      # [B,128]

class FineTuneClassifier(nn.Module):
    def __init__(self, backbone: ContrastiveBackbone, num_classes: int):
        super().__init__()
        self.backbone   = backbone
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)

# ─── INSTANTIATE & LOAD PRETRAINED WEIGHTS ─────────────────────────────────────

backbone = ContrastiveBackbone().to(DEVICE)
# load your contrastive‐pretrained encoder & projector
backbone.encoder.load_state_dict(
    torch.load(ENCODER_CKPT,   map_location=DEVICE)
)
backbone.projector.load_state_dict(
    torch.load(PROJECTOR_CKPT, map_location=DEVICE)
)

model = FineTuneClassifier(backbone, num_classes=len(train_ds.class_to_idx)).to(DEVICE)

# ─── OPTIMIZER, SCHEDULER, LOSS ────────────────────────────────────────────────

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
criterion = nn.CrossEntropyLoss()

# ─── OPTIONAL EVAL BEFORE TRAINING ──────────────────────────────────────────────

if VAL_BEFORE_TRAIN:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Val before train"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    print(f"Pretrain ResNet18 → Val Acc: {correct/total:.4f}")

# ─── TRAIN & VALIDATE ────────────────────────────────────────────────────────────

best_acc = 0.0
for epoch in range(1, EPOCHS+1):
    # Training
    model.train()
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")
    for imgs, labels in train_bar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_bar.set_postfix(loss=loss.item())

    scheduler.step()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    acc = correct / total
    print(f" → Epoch {epoch} Val Acc: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), f"model_outputs/best_finetuned_classifier_resnet18_epoch{EPOCH_NUM}.pt")
        print(" → Saved best model.")

print("Training complete.")
