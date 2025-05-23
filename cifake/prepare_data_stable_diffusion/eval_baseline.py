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
import matplotlib.pyplot as plt
from utils import CustomImageFolder

# ─── CONFIG ─────────────────────────────────────────────────────────────────────

DATA_DIR        = Path("../data")
TRAIN_DIR       = DATA_DIR / "train"
VAL_DIR         = DATA_DIR / "test"      # or "val"
ENCODER_CKPT    = "encoder_final.pt"
PROJECTOR_CKPT  = "projector_final.pt"
BATCH_SIZE      = 64
LR              = 1e-4
EPOCHS          = 5
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASELINE_MODEL  = "dima806/ai_vs_real_image_detection"

# ─── ALIGN DATASET LABELS WITH MODEL ────────────────────────────────────────────

# load model config to get its id2label / label2id
cfg = AutoConfig.from_pretrained(BASELINE_MODEL)
print("Model id2label:", cfg.id2label)
print("Model label2id:", cfg.label2id)

# load the feature extractor to get its expected mean/std
extractor = AutoFeatureExtractor.from_pretrained(BASELINE_MODEL)
mean, std = extractor.image_mean, extractor.image_std

# ─── TRANSFORMS & DATA ───────────────────────────────────────────────────────────

# for fine-tuning, normalize with the model’s own mean/std
train_val_transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),                  # [0,255]→[0,1]
    T.Normalize(mean, std),        # normalize to model’s expected range
])

# ─── BASELINE EVALUATION ─────────────────────────────────────────────────────────

# evaluate the original pretrained head on raw PILs
eval_ds = CustomImageFolder(VAL_DIR, transform=None)
eval_ds.class_to_idx = cfg.label2id

baseline_extractor = extractor
baseline_model     = AutoModelForImageClassification.from_pretrained(BASELINE_MODEL).to(DEVICE)
baseline_model.eval()

correct, total = 0, 0
for pil_img, label in tqdm(eval_ds, desc="Baseline Eval"):
    inputs = baseline_extractor(images=pil_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = baseline_model(**inputs).logits
    pred = logits.argmax(-1).item()
    correct += (pred == label)
    total   += 1

print(f"Baseline '{BASELINE_MODEL}' → Val Acc: {correct/total:.4f}")
