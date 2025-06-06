#!/usr/bin/env python3
import os
from pathlib import Path
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import torchvision.models as models

from utils import CustomImageFolder  # your subclass that reads label2id mapping

# ─── CONFIG ─────────────────────────────────────────────────────────────────────

DATA_DIR     = Path("../data")
VAL_DIR      = DATA_DIR / "test"         # or "val"
BATCH_SIZE   = 64
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT   = "best_finetuned_classifier.pt"  # your ResNet18 fine-tuned checkpoint

# ─── TRANSFORMS ─────────────────────────────────────────────────────────────────

# ResNet18 uses ImageNet stats
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

eval_transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(imagenet_mean, imagenet_std),
])

# ─── DATASET & DATALOADER ────────────────────────────────────────────────────────

# CustomImageFolder should override find_classes() to use your label2id mapping
eval_ds = CustomImageFolder(VAL_DIR, transform=eval_transform)
eval_loader = DataLoader(
    eval_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ─── MODEL LOADING ───────────────────────────────────────────────────────────────

# 1) load a plain ResNet18
model = models.resnet18(pretrained=False)
# 2) replace its head with the correct number of classes
num_ftrs       = model.fc.in_features
num_classes    = len(eval_ds.class_to_idx)  # should be 2 for FAKE vs REAL
model.fc       = nn.Linear(num_ftrs, num_classes)
# 3) load your fine-tuned weights
# state = torch.load(CHECKPOINT, map_location=DEVICE)
# model.load_state_dict(state)
model.to(DEVICE)
model.eval()

# ─── EVALUATION ─────────────────────────────────────────────────────────────────

correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in tqdm(eval_loader, desc="ResNet18 Eval"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        preds  = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

acc = correct / total
print(f"ResNet18 → Val Acc: {acc:.4f}")
