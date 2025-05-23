#!/usr/bin/env python3
import os
from pathlib import Path
from tqdm.auto import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForImageClassification
)
from utils import CustomImageFolder

# ─── CONFIG ─────────────────────────────────────────────────────────────────────

DATA_DIR         = Path("../data")
TRAIN_DIR        = DATA_DIR / "train"
VAL_DIR          = DATA_DIR / "test"      # or "val"
ENCODER_CKPT     = "encoder_final.pt"
PROJECTOR_CKPT   = "projector_final.pt"
BATCH_SIZE       = 64
LR               = 3e-4
EPOCHS           = 10
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASELINE_MODEL   = "dima806/ai_vs_real_image_detection"
VAL_BEFORE_TRAIN = False

# ─── ALIGN DATASET LABELS WITH MODEL ────────────────────────────────────────────

cfg = AutoConfig.from_pretrained(BASELINE_MODEL)
print("Model id2label:", cfg.id2label)
print("Model label2id:", cfg.label2id)

transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

# ImageFolder will give class_to_idx by folder name alphabetically;
# override to match the model's own label2id mapping.
train_ds = CustomImageFolder(TRAIN_DIR, transform=transform)
val_ds   = CustomImageFolder(VAL_DIR,   transform=transform)
train_ds.class_to_idx = cfg.label2id
val_ds.  class_to_idx = cfg.label2id

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

# ─── MODEL DEFINITION FOR CLASSIFIER FINE-TUNING ────────────────────────────────

class ContrastiveBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # load only the vision encoder (no classification head)
        base = AutoModel.from_pretrained(BASELINE_MODEL)
        self.encoder = base
        hidden = base.config.hidden_size
        # projection head
        self.projector = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 128),
        )

    def forward(self, x):
        out = self.encoder(pixel_values=x).pooler_output
        return self.projector(out)

class FineTuneClassifier(nn.Module):
    def __init__(self, backbone: ContrastiveBackbone, num_classes: int):
        super().__init__()
        self.backbone   = backbone
        # classification head on top of 128-d features
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)

# instantiate and load your contrastively pre-trained encoder+projector
backbone = ContrastiveBackbone().to(DEVICE)
backbone.encoder.load_state_dict(torch.load(ENCODER_CKPT,   map_location=DEVICE))
backbone.projector.load_state_dict(torch.load(PROJECTOR_CKPT, map_location=DEVICE))

model = FineTuneClassifier(backbone, num_classes=len(cfg.id2label)).to(DEVICE)

# ─── OPTIMIZER & SCHEDULER & LOSS ────────────────────────────────────────────────

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
criterion = nn.CrossEntropyLoss()

# ─── TRAIN & VALIDATE ────────────────────────────────────────────────────────────

if VAL_BEFORE_TRAIN:
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()

    acc = correct / len(val_ds)
    print(f" →Pretrained Val Acc: {acc:.4f}")

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
    correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()

    acc = correct / len(val_ds)
    print(f" → Epoch {epoch} Val Acc: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_finetuned_classifier.pt")
        print(" → Saved best model.")

print("Training complete.")
